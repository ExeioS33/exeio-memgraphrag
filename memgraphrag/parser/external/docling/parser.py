"""Docling Serve adapter for MemGraphRAG.

Adapted from LightRAG ``lightrag/parser/external/docling/`` — simplified
submit+poll client that writes a sidecar and returns ``parse_format=lightrag``.
Only selectable when ``DOCLING_ENDPOINT`` is set (enforced by the registry).
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import tempfile
import time
from pathlib import Path
from typing import Any
from urllib.parse import quote

from memgraphrag.parser.base import BaseParser, ParseContext, ParseResult
from memgraphrag.parser.external._zip import safe_extract_zip
from memgraphrag.parser.registry import PARSER_ENGINE_DOCLING
from memgraphrag.sidecar.writer import FULL_DOCS_FORMAT_LIGHTRAG, write_sidecar
from memgraphrag.utils.http_ssl import ssl_verify
from memgraphrag.utils.step_log import done_step, fail_step, main_step, sub_step

logger = logging.getLogger(__name__)

CONVERT_PATH = "/v1/convert/file/async"
POLL_PATH = "/v1/status/poll/{task_id}"
RESULT_PATH = "/v1/result/{task_id}"

SUCCESS_STATES = {"success"}
FAILURE_STATES = {"failure", "partial_success", "skipped"}
IN_PROGRESS_STATES = {"pending", "started"}

DEFAULT_POLL_WAIT_SECONDS = 5
DEFAULT_MAX_POLLS = 240


def _endpoint() -> str:
    return os.getenv("DOCLING_ENDPOINT", "").strip().rstrip("/")


class DoclingParser(BaseParser):
    """POST file to Docling Serve, poll status, write sidecar."""

    engine_name = PARSER_ENGINE_DOCLING

    async def parse(self, ctx: ParseContext) -> ParseResult:
        endpoint = _endpoint()
        if not endpoint:
            raise RuntimeError("DOCLING_ENDPOINT is required for DoclingParser")

        main_step(
            logger,
            "parse.docling",
            doc_id=ctx.doc_id,
            file=Path(ctx.file_path).name,
        )
        try:
            import httpx
        except ImportError as exc:
            fail_step(logger, "parse.docling", doc_id=ctx.doc_id, error="httpx_missing")
            raise RuntimeError(
                "httpx is required for Docling parsing but is not installed"
            ) from exc

        source = ctx.source_path()
        if not source.is_file():
            fail_step(
                logger,
                "parse.docling",
                doc_id=ctx.doc_id,
                error="source_not_found",
            )
            raise FileNotFoundError(f"docling source file not found: {source}")

        parsed_dir = ctx.resolve_parsed_dir()
        poll_wait = int(os.getenv("DOCLING_POLL_INTERVAL_SECONDS", DEFAULT_POLL_WAIT_SECONDS))
        max_polls = int(os.getenv("DOCLING_MAX_POLLS", DEFAULT_MAX_POLLS))
        if poll_wait <= 0:
            poll_wait = DEFAULT_POLL_WAIT_SECONDS
        if max_polls <= 0:
            max_polls = DEFAULT_MAX_POLLS

        timeout = httpx.Timeout(120.0, connect=30.0)
        async with httpx.AsyncClient(timeout=timeout, verify=ssl_verify()) as client:
            sub_step(
                logger,
                "parse.docling.submit",
                doc_id=ctx.doc_id,
                bytes=source.stat().st_size,
            )
            task_id = await self._submit(client, endpoint, source)
            sub_step(
                logger,
                "parse.docling.poll",
                task_id=task_id,
                poll_wait=poll_wait,
                max_polls=max_polls,
            )
            await self._poll_until_done(client, endpoint, task_id, poll_wait, max_polls)
            sub_step(logger, "parse.docling.fetch", task_id=task_id)
            content, blocks = await self._fetch_result(
                client, endpoint, task_id, source.name
            )

        if not content.strip() and not blocks:
            fail_step(
                logger,
                "parse.docling",
                doc_id=ctx.doc_id,
                error="empty_content",
            )
            raise ValueError(
                f"Docling produced empty content for {ctx.file_path} (doc_id={ctx.doc_id})"
            )

        if not blocks:
            blocks = [{"content": content, "heading": "", "level": 0}]

        sub_step(
            logger,
            "parse.docling.sidecar",
            doc_id=ctx.doc_id,
            blocks=len(blocks),
            content_chars=len(content),
        )
        sidecar = write_sidecar(
            parsed_dir,
            blocks,
            doc_id=ctx.doc_id,
            document_name=source.name,
            engine=self.engine_name,
        )

        done_step(
            logger,
            "parse.docling",
            doc_id=ctx.doc_id,
            blocks=len(blocks),
            content_chars=len(sidecar["content"] or content),
            has_blocks_path=bool(sidecar["blocks_path"]),
        )
        return ParseResult(
            doc_id=ctx.doc_id,
            file_path=ctx.file_path,
            parse_format=FULL_DOCS_FORMAT_LIGHTRAG,
            content=sidecar["content"] or content,
            blocks_path=sidecar["blocks_path"],
            parse_engine=self.engine_name,
        )

    async def _submit(
        self, client: Any, endpoint: str, source: Path
    ) -> str:
        url = f"{endpoint}{CONVERT_PATH}"
        data = {
            "pipeline": "standard",
            "target_type": "zip",
            "to_formats": ["json", "md"],
            "image_export_mode": "referenced",
        }
        with source.open("rb") as fh:
            files = {"files": (source.name, fh, "application/octet-stream")}
            resp = await client.post(url, data=data, files=files)
        resp.raise_for_status()
        payload = resp.json() if resp.text else {}
        task_id = str(payload.get("task_id") or payload.get("id") or "").strip()
        if not task_id:
            raise RuntimeError(f"Docling upload response missing task_id: {payload!r}")
        return task_id

    async def _poll_until_done(
        self,
        client: Any,
        endpoint: str,
        task_id: str,
        poll_wait: int,
        max_polls: int,
    ) -> None:
        encoded = quote(task_id, safe="")
        url = f"{endpoint}{POLL_PATH.format(task_id=encoded)}"
        for _ in range(max_polls):
            started = time.monotonic()
            resp = await client.get(url, params={"wait": poll_wait})
            resp.raise_for_status()
            payload = resp.json() if resp.text else {}
            status = str(
                payload.get("task_status") or payload.get("status") or ""
            ).lower()
            if status in SUCCESS_STATES:
                return
            if status in FAILURE_STATES:
                err = (
                    payload.get("error_message")
                    or payload.get("error")
                    or payload.get("message")
                    or "<no error>"
                )
                raise RuntimeError(f"Docling task {task_id} ended in {status}: {err}")
            remaining = poll_wait - (time.monotonic() - started)
            if remaining > 0:
                await asyncio.sleep(remaining)
        raise TimeoutError(f"Docling task {task_id} polling timeout")

    async def _fetch_result(
        self, client: Any, endpoint: str, task_id: str, filename: str
    ) -> tuple[str, list[dict[str, Any]]]:
        encoded = quote(task_id, safe="")
        url = f"{endpoint}{RESULT_PATH.format(task_id=encoded)}"
        resp = await client.get(url)
        resp.raise_for_status()
        ctype = (resp.headers.get("content-type") or "").lower()
        body = resp.content or b""
        is_zip = "zip" in ctype or body[:2] == b"PK"

        # Docling Serve returns a ZIP bundle when target_type=zip (LightRAG path).
        if is_zip:
            markdown, blocks = self._extract_from_zip(body, filename)
            return markdown, blocks

        # Prefer JSON envelope when available; otherwise treat body as markdown.
        if "json" in ctype or body.lstrip().startswith((b"{", b"[")):
            try:
                payload = resp.json()
            except Exception:
                payload = json.loads(body)
            return self._extract_from_json(payload, filename)

        # Refuse binary non-zip bodies that would become garbage "text" chunks.
        if b"\x00" in body[:4096]:
            raise RuntimeError(
                f"Docling result {task_id} looks binary (content-type={ctype!r}); "
                "expected zip or JSON/markdown"
            )

        text = body.decode("utf-8", errors="replace")
        return text, [{"content": text, "heading": "", "level": 0}] if text.strip() else []

    def _extract_from_zip(
        self, payload: bytes, filename: str
    ) -> tuple[str, list[dict[str, Any]]]:
        """Unpack Docling zip and prefer markdown / DoclingDocument JSON."""
        stem = Path(filename).stem
        with tempfile.TemporaryDirectory(prefix="memgraphrag-docling-") as tmp:
            raw_dir = Path(tmp)
            names = safe_extract_zip(payload, raw_dir)
            md_path = self._pick_extracted_file(raw_dir, names, stem, (".md", ".markdown"))
            json_path = self._pick_extracted_file(raw_dir, names, stem, (".json",))

            markdown = ""
            if md_path and md_path.is_file():
                markdown = md_path.read_text(encoding="utf-8", errors="replace")

            document: dict[str, Any] | None = None
            if json_path and json_path.is_file():
                try:
                    loaded = json.loads(json_path.read_text(encoding="utf-8"))
                except json.JSONDecodeError:
                    loaded = None
                if isinstance(loaded, dict):
                    # ConvertDocumentResponse envelope or bare DoclingDocument
                    nested = loaded.get("document")
                    if isinstance(nested, dict):
                        if not markdown and isinstance(nested.get("md_content"), str):
                            markdown = nested["md_content"]
                        jc = nested.get("json_content")
                        if isinstance(jc, dict):
                            document = jc
                        elif any(k in nested for k in ("texts", "body", "tables")):
                            document = nested
                    elif any(k in loaded for k in ("texts", "body", "tables", "schema_name")):
                        document = loaded
                    if not markdown and isinstance(loaded.get("md_content"), str):
                        markdown = loaded["md_content"]

            if document is not None:
                envelope: dict[str, Any] = {
                    "document": {
                        "md_content": markdown,
                        "json_content": document,
                    }
                }
                md2, blocks = self._extract_from_json(envelope, filename)
                return (md2 or markdown), blocks

            if markdown.strip():
                return markdown, [{"content": markdown, "heading": "", "level": 0}]

            raise RuntimeError(
                f"Docling zip for {filename} had no usable .md/.json "
                f"(entries={names[:20]!r})"
            )

    @staticmethod
    def _pick_extracted_file(
        raw_dir: Path,
        names: list[str],
        stem: str,
        suffixes: tuple[str, ...],
    ) -> Path | None:
        candidates: list[Path] = []
        for name in names:
            path = raw_dir / name
            if not path.is_file():
                continue
            if path.suffix.lower() not in suffixes:
                continue
            candidates.append(path)
        if not candidates:
            return None
        # Prefer filename matching the upload stem.
        for path in candidates:
            if path.stem == stem or stem in path.stem:
                return path
        return candidates[0]

    def _extract_from_json(
        self, payload: Any, filename: str
    ) -> tuple[str, list[dict[str, Any]]]:
        markdown = ""
        document: dict[str, Any] | None = None

        if isinstance(payload, dict):
            nested = payload.get("document")
            if isinstance(nested, dict):
                md = nested.get("md_content")
                if isinstance(md, str):
                    markdown = md
                json_content = nested.get("json_content")
                if isinstance(json_content, dict):
                    document = json_content
                elif any(k in nested for k in ("texts", "body", "tables")):
                    document = nested
            if not markdown and isinstance(payload.get("md_content"), str):
                markdown = payload["md_content"]
            if document is None and any(
                k in payload for k in ("texts", "body", "tables", "schema_name")
            ):
                document = payload

        blocks: list[dict[str, Any]] = []
        if document:
            texts = document.get("texts")
            if isinstance(texts, list):
                for item in texts:
                    if not isinstance(item, dict):
                        continue
                    text = str(item.get("text") or item.get("orig") or "").strip()
                    if not text:
                        continue
                    label = str(item.get("label") or "")
                    level = 0
                    heading = ""
                    if "title" in label.lower() or "heading" in label.lower():
                        heading = text
                        level = 1
                    blocks.append(
                        {"content": text, "heading": heading, "level": level}
                    )

        if not markdown and blocks:
            markdown = "\n\n".join(b["content"] for b in blocks)
        if not blocks and markdown.strip():
            blocks = [{"content": markdown, "heading": "", "level": 0}]

        logger.debug(
            "[docling] extracted %d blocks from %s", len(blocks), filename
        )
        sub_step(
            logger,
            "parse.docling.extract_blocks",
            filename=filename,
            blocks=len(blocks),
            markdown_chars=len(markdown),
        )
        return markdown, blocks
