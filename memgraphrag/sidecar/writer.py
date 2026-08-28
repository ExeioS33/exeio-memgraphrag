"""Simplified sidecar writer for MemGraphRAG.

Adapted from LightRAG ``lightrag/sidecar/writer.py`` — lean API that writes
``blocks.jsonl`` (and optional tables/assets) without the full IRDoc stack.
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

FULL_DOCS_FORMAT_LIGHTRAG = "lightrag"


def write_sidecar(
    parsed_dir: str | Path,
    blocks: list[dict[str, Any]],
    tables: dict[str, Any] | None = None,
    assets: dict[str, Any] | None = None,
    *,
    doc_id: str = "",
    document_name: str = "document",
    engine: str = "legacy",
    clean_parsed_dir: bool = True,
) -> dict[str, Any]:
    """Write a ``*.parsed/`` sidecar directory with ``blocks.jsonl``.

    Args:
        parsed_dir: Output directory (created/replaced when ``clean_parsed_dir``).
        blocks: Content block dicts. Each should include at least ``content``;
            optional ``heading``, ``level``, ``parent_headings``, ``blockid``.
        tables: Optional tables mapping written to ``<base>.tables.json``.
        assets: Optional asset metadata (not copied; recorded in meta only).
        doc_id: Document id for blockid hashing / meta.
        document_name: Source document basename used for file naming.
        engine: Parse engine name recorded in meta.
        clean_parsed_dir: When True, clear ``parsed_dir`` before writing.

    Returns:
        Dict with ``content``, ``blocks_path``, ``parse_format``, etc.
    """
    parsed_dir = Path(parsed_dir)
    if clean_parsed_dir and parsed_dir.exists():
        shutil.rmtree(parsed_dir)
    parsed_dir.mkdir(parents=True, exist_ok=True)

    base_name = Path(document_name).stem or document_name or "document"
    blocks_path = parsed_dir / f"{base_name}.blocks.jsonl"
    tables_path = parsed_dir / f"{base_name}.tables.json"

    blocks_lines: list[str] = []
    merged_parts: list[str] = []

    for index, block in enumerate(blocks):
        content = str(block.get("content") or "").strip()
        if not content:
            continue
        heading = str(block.get("heading") or "")
        blockid = (
            block.get("blockid")
            or hashlib.md5(f"{doc_id}:{index}:{heading}:{content}".encode("utf-8")).hexdigest()
        )
        row = {
            "type": "content",
            "blockid": blockid,
            "format": block.get("format", "plain_text"),
            "content": content,
            "heading": heading,
            "parent_headings": list(block.get("parent_headings") or []),
            "level": int(block.get("level") or 0),
            "session_type": block.get("session_type") or "body",
            "table_slice": block.get("table_slice") or "none",
            "positions": list(block.get("positions") or []),
        }
        blocks_lines.append(json.dumps(row, ensure_ascii=False))
        merged_parts.append(content)

    merged_text = "\n\n".join(p for p in merged_parts if p.strip())
    document_hash = hashlib.sha256(merged_text.encode("utf-8")).hexdigest()
    meta = {
        "type": "meta",
        "format": FULL_DOCS_FORMAT_LIGHTRAG,
        "version": "1.0",
        "document_name": document_name,
        "document_hash": f"sha256:{document_hash}",
        "table_file": bool(tables),
        "asset_dir": bool(assets),
        "blocks": len(blocks_lines),
        "doc_id": doc_id,
        "parse_engine": engine,
        "parse_time": datetime.now(timezone.utc).isoformat(),
    }

    blocks_path.write_text(
        "\n".join([json.dumps(meta, ensure_ascii=False)] + blocks_lines) + "\n",
        encoding="utf-8",
    )

    if tables:
        tables_path.write_text(
            json.dumps({"version": "1.0", "tables": tables}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )

    logger.info(
        "[sidecar] wrote %d blocks for doc_id=%s engine=%s",
        len(blocks_lines),
        doc_id,
        engine,
    )

    return {
        "doc_id": doc_id,
        "file_path": document_name,
        "parse_format": FULL_DOCS_FORMAT_LIGHTRAG,
        "content": merged_text,
        "blocks_path": str(blocks_path),
    }
