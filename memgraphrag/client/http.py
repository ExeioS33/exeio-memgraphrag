"""Synchronous HTTP client for the MemGraphRAG API server."""

from __future__ import annotations

import mimetypes
import os
import tempfile
from pathlib import Path
from typing import Any, Iterator, Optional
from urllib.parse import urlparse

import httpx

from memgraphrag.client.params import SUPPORTED_EXTENSIONS, clean_params
from memgraphrag.utils.http_ssl import ssl_verify

DEFAULT_BASE_URL = "http://localhost:9621"
DEFAULT_TIMEOUT = httpx.Timeout(connect=10.0, read=300.0, write=120.0, pool=10.0)


class ClientSSLError(RuntimeError):
    """Raised when outbound HTTPS fails TLS verification (URL ingest, etc.)."""


def _has_supported_suffix(name: str) -> bool:
    """True when ``name`` ends with a known ingest extension (e.g. ``.pdf``)."""
    return Path(name).suffix.lower() in SUPPORTED_EXTENSIONS


def _sniff_extension(data: bytes) -> str | None:
    """Best-effort magic-byte extension when URL/basename has no real suffix."""
    if data[:5] == b"%PDF-" or data[:4] == b"%PDF":
        return ".pdf"
    if data[:2] == b"PK":  # zip-based office formats; default to .docx
        return ".docx"
    if data.lstrip()[:1] in (b"<", b"{") or data[:3] in (b"\xef\xbb\xbf",):
        # HTML / JSON / text-ish — leave to content-type when possible
        return None
    return None


def _filename_from_headers(
    content_disp: str, content_type: str, fallback: str
) -> str:
    """Derive a filename from Content-Disposition / Content-Type when URL has none."""
    name = fallback
    cd = content_disp or ""
    # Prefer RFC 5987 filename*=utf-8''...
    if "filename*=" in cd.lower():
        try:
            part = cd.split("filename*=")[-1].split(";")[0].strip().strip("\"'")
            if "''" in part:
                part = part.split("''", 1)[1]
            from urllib.parse import unquote

            candidate = Path(unquote(part)).name
            if candidate:
                name = candidate
        except Exception:
            pass
    elif "filename=" in cd:
        candidate = Path(cd.split("filename=")[-1].split(";")[0].strip().strip("\"'")).name
        if candidate:
            name = candidate
    if not _has_supported_suffix(name):
        ctype = (content_type or "").split(";")[0].strip().lower()
        if ctype == "application/pdf":
            guessed = ".pdf"
        else:
            guessed = mimetypes.guess_extension(ctype) if ctype else None
        if guessed:
            # arXiv paths like ``2605.18490v1`` look like they have a suffix
            # (``.18490v1``) but are not real extensions — append the real one.
            name = f"{name}{guessed}"
    return name


def normalize_download_filename(
    url_name: str,
    *,
    content_disp: str = "",
    content_type: str = "",
    data: bytes | None = None,
    explicit: str | None = None,
) -> str:
    """Resolve a safe upload filename with a supported suffix.

    Handles arXiv-style basenames (``2605.18490v1``) where ``Path.suffix`` is a
    false-positive version fragment rather than a real file type.
    """
    if explicit:
        name = Path(explicit).name
    else:
        name = Path(url_name).name or "download.bin"
        name = _filename_from_headers(content_disp, content_type, name)
    if not _has_supported_suffix(name) and data is not None:
        sniffed = _sniff_extension(data)
        if sniffed:
            name = f"{name}{sniffed}"
    if not _has_supported_suffix(name):
        # Last resort so the server always sees a typed name.
        name = f"{name}.bin"
    return name


class MemGraphRAGClient:
    """Thin httpx wrapper around MemGraphRAG REST endpoints.

    Auth: set ``api_key`` (or env ``MEMGRAPHRAG_API_KEY``) → ``X-API-Key``.
    Base URL: ``base_url`` or env ``MEMGRAPHRAG_SERVER_URL`` (default localhost:9621).
    """

    def __init__(
        self,
        base_url: Optional[str] = None,
        api_key: Optional[str] = None,
        timeout: Optional[httpx.Timeout] = None,
        transport: Optional[httpx.BaseTransport] = None,
        verify: Any = None,
    ) -> None:
        self.base_url = (
            base_url
            or os.environ.get("MEMGRAPHRAG_SERVER_URL")
            or DEFAULT_BASE_URL
        ).rstrip("/")
        self.api_key = api_key if api_key is not None else os.environ.get(
            "MEMGRAPHRAG_API_KEY"
        )
        self.verify = ssl_verify() if verify is None else verify
        headers: dict[str, str] = {"Accept": "application/json"}
        if self.api_key:
            headers["X-API-Key"] = self.api_key
        self._client = httpx.Client(
            base_url=self.base_url,
            headers=headers,
            timeout=timeout or DEFAULT_TIMEOUT,
            transport=transport,
            follow_redirects=True,
            verify=self.verify,
        )

    def close(self) -> None:
        self._client.close()

    def __enter__(self) -> "MemGraphRAGClient":
        return self

    def __exit__(self, *exc: Any) -> None:
        self.close()

    # ------------------------------------------------------------------ health
    def health(self) -> dict[str, Any]:
        return self._get("/health")

    # ------------------------------------------------------------------ query
    def query(self, question: str, **params: Any) -> dict[str, Any]:
        body = {"query": question, **self._query_body(params)}
        return self._post("/query", json=body)

    def query_data(self, question: str, **params: Any) -> dict[str, Any]:
        body = {"query": question, **self._query_body(params)}
        return self._post("/query/data", json=body)

    def query_stream(self, question: str, **params: Any) -> Iterator[str]:
        """Yield SSE ``data:`` payloads (raw JSON strings or ``[DONE]``)."""
        body = {"query": question, "stream": True, **self._query_body(params)}
        with self._client.stream("POST", "/query/stream", json=body) as resp:
            self._raise(resp)
            for line in resp.iter_lines():
                if not line:
                    continue
                if line.startswith("data:"):
                    yield line[5:].strip()

    # -------------------------------------------------------------- documents
    def list_documents(self) -> dict[str, Any]:
        return self._get("/documents/")

    def get_document(self, doc_id: str) -> dict[str, Any]:
        return self._get(f"/documents/{doc_id}")

    def delete_document(
        self, doc_id: str, *, delete_file: bool = False
    ) -> dict[str, Any]:
        resp = self._client.delete(
            f"/documents/{doc_id}",
            params={"delete_file": str(delete_file).lower()},
        )
        self._raise(resp)
        return resp.json()

    def delete_documents(
        self, doc_ids: list[str], *, delete_file: bool = False
    ) -> dict[str, Any]:
        return self._post(
            "/documents/delete",
            json={"doc_ids": list(doc_ids), "delete_file": delete_file},
        )

    def requeue_document(self, doc_id: str) -> dict[str, Any]:
        return self._post(f"/documents/{doc_id}/requeue")

    def upload_file(self, path: str | Path, filename: Optional[str] = None) -> dict[str, Any]:
        path = Path(path)
        if not path.is_file():
            raise FileNotFoundError(f"Not a file: {path}")
        name = filename or path.name
        mime, _ = mimetypes.guess_type(name)
        mime = mime or "application/octet-stream"
        with path.open("rb") as fh:
            files = {"file": (name, fh, mime)}
            resp = self._client.post("/documents/upload", files=files)
        self._raise(resp)
        return resp.json()

    def upload_bytes(
        self, data: bytes, filename: str, content_type: Optional[str] = None
    ) -> dict[str, Any]:
        mime = content_type or mimetypes.guess_type(filename)[0] or "application/octet-stream"
        files = {"file": (filename, data, mime)}
        resp = self._client.post("/documents/upload", files=files)
        self._raise(resp)
        return resp.json()

    def upload_directory(
        self,
        directory: str | Path,
        recursive: bool = True,
        extensions: Optional[set[str]] = None,
    ) -> list[dict[str, Any]]:
        directory = Path(directory)
        if not directory.is_dir():
            raise NotADirectoryError(f"Not a directory: {directory}")
        exts = {e.lower() if e.startswith(".") else f".{e.lower()}" for e in (extensions or SUPPORTED_EXTENSIONS)}
        pattern = "**/*" if recursive else "*"
        results: list[dict[str, Any]] = []
        for path in sorted(directory.glob(pattern)):
            if not path.is_file():
                continue
            if path.name.startswith("."):
                continue
            if path.suffix.lower() not in exts:
                continue
            results.append(self.upload_file(path))
        return results

    def upload_url(self, url: str, filename: Optional[str] = None) -> dict[str, Any]:
        """Download ``url`` then multipart-upload it to the server.

        Honors the same TLS settings as the rest of MemGraphRAG (``ssl_verify()`` /
        ``SSL_CERT_FILE`` / ``SSL_VERIFY``). Corporate TLS inspection often needs a
        Fortinet/Zscaler CA PEM plus OpenSSL-3 AKI relax (handled in ``http_ssl``).
        """
        url = (url or "").strip()
        if not url:
            raise ValueError("URL is empty")
        parsed = urlparse(url)
        if parsed.scheme not in {"http", "https"}:
            raise ValueError(f"Unsupported URL scheme {parsed.scheme!r}; use http(s)")
        if not parsed.netloc:
            raise ValueError(f"URL is missing a host: {url!r}")

        verify = self.verify
        try:
            with httpx.Client(
                timeout=DEFAULT_TIMEOUT,
                follow_redirects=True,
                verify=verify,
            ) as dl:
                resp = dl.get(url)
                resp.raise_for_status()
                data = resp.content
                content_disp = resp.headers.get("content-disposition", "")
                content_type = resp.headers.get("content-type", "")
        except Exception as exc:  # noqa: BLE001 — wrap SSL then re-raise
            msg = str(exc)
            is_ssl = (
                "CERTIFICATE_VERIFY_FAILED" in msg
                or "SSL" in type(exc).__name__
                or "ssl" in type(exc).__module__
            )
            if is_ssl:
                raise ClientSSLError(
                    "TLS verification failed while downloading the URL. "
                    "Behind corporate TLS inspection, set "
                    "MEMGRAPHRAG_SSL_CERT_FILE (or SSL_CERT_FILE) to your "
                    "Fortinet/Zscaler CA PEM, or place it at certs/corporate-ca.crt. "
                    "Lab-only escape hatch: SSL_VERIFY=false. "
                    f"Original error: {msg}"
                ) from exc
            raise

        if not data:
            raise ValueError(f"Downloaded empty body from {url}")

        raw_name = Path(parsed.path).name or "download.bin"
        name = normalize_download_filename(
            raw_name,
            content_disp=content_disp,
            content_type=content_type,
            data=data,
            explicit=filename,
        )
        suffix = Path(name).suffix or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
            tmp.write(data)
            tmp_path = Path(tmp.name)
        try:
            return self.upload_file(tmp_path, filename=name)
        finally:
            tmp_path.unlink(missing_ok=True)

    def insert_text(self, text: str, doc_id: Optional[str] = None) -> dict[str, Any]:
        body: dict[str, Any] = {"text": text}
        if doc_id:
            body["doc_id"] = doc_id
        return self._post("/documents/text", json=body)

    def scan_input_dir(self) -> dict[str, Any]:
        return self._post("/documents/scan")

    def clear_documents(
        self, *, confirm: bool = True, delete_files: bool = False
    ) -> dict[str, Any]:
        resp = self._client.delete(
            "/documents/",
            params={
                "confirm": str(confirm).lower(),
                "delete_files": str(delete_files).lower(),
            },
        )
        self._raise(resp)
        return resp.json()

    # ------------------------------------------------------------------ graph
    def list_labels(self) -> dict[str, Any]:
        return self._get("/graph/label/list")

    def explore_graph(
        self, label: Optional[str] = None, limit: int = 200
    ) -> dict[str, Any]:
        params: dict[str, Any] = {"limit": limit}
        if label:
            params["label"] = label
        return self._get("/graphs", params=params)

    # --------------------------------------------------------------- helpers
    @staticmethod
    def _query_body(params: dict[str, Any]) -> dict[str, Any]:
        """Clean tunable knobs; preserve API flags like ``only_need_context``."""
        params = dict(params)
        only_need_context = params.pop("only_need_context", None)
        body = clean_params(params)
        if only_need_context is not None:
            body["only_need_context"] = bool(only_need_context)
        return body

    def _get(self, path: str, params: Optional[dict[str, Any]] = None) -> dict[str, Any]:
        resp = self._client.get(path, params=params)
        self._raise(resp)
        return resp.json()

    def _post(
        self, path: str, json: Optional[dict[str, Any]] = None
    ) -> dict[str, Any]:
        resp = self._client.post(path, json=json)
        self._raise(resp)
        return resp.json()

    @staticmethod
    def _raise(resp: httpx.Response) -> None:
        if resp.is_success:
            return
        detail: Any
        try:
            detail = resp.json()
        except Exception:
            detail = resp.text
        raise httpx.HTTPStatusError(
            f"{resp.status_code} {resp.reason_phrase}: {detail}",
            request=resp.request,
            response=resp,
        )
