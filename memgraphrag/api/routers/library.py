"""Filesystem-backed document library routes for MemGraphRAG.

A read-only window on a directory of source documents (``LIBRARY_ROOT``): list the
tree, stream one file to the browser's own PDF viewer, extract per-page text, and look
up the passages the memory graph already holds for that file. Nothing here writes to
disk and nothing here enqueues an ingestion — ``/documents`` owns that side.

Every caller-supplied path goes through :func:`_safe_path`, which resolves it against
the root and refuses anything that lands outside. That helper is the only place in this
module where a filesystem path is built from request input: a route that joins
``root / path`` on its own is one ``../../..`` away from serving ``/etc/shadow``, and a
symlink planted in the library is the same hole with fewer dots.

Responses never carry an absolute path for an entry. The browser gets root-relative
paths and hands them straight back; the server's directory layout is not the UI's
business.
"""

from __future__ import annotations

import asyncio
import logging
import mimetypes
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger("memgraphrag.api.library")

try:
    from fastapi import APIRouter, Depends, HTTPException, Query, Request
    from fastapi.responses import FileResponse
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[misc, assignment]
    HTTPException = None  # type: ignore[misc, assignment]
    Query = None  # type: ignore[misc, assignment]
    Request = None  # type: ignore[misc, assignment]
    FileResponse = None  # type: ignore[misc, assignment]

# Fallback when ``app.state.args`` predates LIBRARY_ROOT; mirrors the config default.
DEFAULT_LIBRARY_ROOT = "~/Desktop/project/lightrag/cf_lightrag/data/rfe-igor"

# A recursive listing of an unbounded directory is a denial of service against our own
# worker as much as against the browser: cap both breadth and depth, and report what was
# left out instead of silently truncating.
MAX_TREE_FILES = 5000
MAX_TREE_DEPTH = 12

# Per-request page window for /library/preview. Twenty pages of extracted text is
# already a large JSON body; a whole 900-page PDF in one response is not a preview.
MAX_PREVIEW_PAGES = 20

# Text files are read whole for page 1, so the read itself needs a ceiling.
MAX_TEXT_PREVIEW_BYTES = 2 * 1024 * 1024

# Passage bodies are corpus text. The UI shows an excerpt next to the page, not the
# chunk in full, so the wire carries an excerpt.
PASSAGE_CONTENT_LIMIT = 1200

# Suffixes previewed by decoding the bytes. Anything else (docx, pptx, images…) is
# refused rather than served as mojibake — the parser stack handles those at ingest.
_TEXT_SUFFIXES = frozenset(
    {
        "txt",
        "md",
        "markdown",
        "csv",
        "tsv",
        "json",
        "jsonl",
        "yaml",
        "yml",
        "log",
        "xml",
        "html",
        "htm",
        "tex",
        "rst",
        "ini",
        "cfg",
        "toml",
        "py",
        "sql",
    }
)

# Content types the browser must get right for an inline render; mimetypes guesses the
# rest, and an unknown suffix falls back to a download-safe octet-stream.
_EXPLICIT_MEDIA_TYPES = {
    "pdf": "application/pdf",
    "txt": "text/plain; charset=utf-8",
    "md": "text/markdown; charset=utf-8",
    "markdown": "text/markdown; charset=utf-8",
    "csv": "text/csv; charset=utf-8",
    "json": "application/json",
    "log": "text/plain; charset=utf-8",
}


class LibraryPreviewError(RuntimeError):
    """Raised when a file exists but its text cannot be extracted."""


def _library_root(request: Any) -> Path:
    """Resolved LIBRARY_ROOT for this app.

    Resolved once here so every containment check compares against the same real path;
    a root that is itself a symlink would otherwise make ``is_relative_to`` reject
    files that are legitimately inside it.
    """
    state = getattr(getattr(request, "app", None), "state", None)
    raw = getattr(getattr(state, "args", None), "library_root", None) or DEFAULT_LIBRARY_ROOT
    return Path(str(raw)).expanduser().resolve()


def _safe_path(root: Path, rel: str) -> Path:
    """Resolve ``rel`` under ``root`` and refuse anything that escapes it.

    ``root`` must already be resolved. The checks are layered because each one alone
    has a hole: rejecting ``..`` misses an absolute path (``root / "/etc/passwd"`` is
    ``/etc/passwd`` — pathlib drops the left side), rejecting absolute paths misses a
    symlink pointing out of the library, and the ``resolve()`` + ``is_relative_to``
    pair — which catches both — is worth keeping explicit rather than implied.
    """
    raw = (rel or "").strip()
    if not raw:
        raise HTTPException(status_code=400, detail="Query parameter 'path' is required")
    if "\x00" in raw:
        raise HTTPException(status_code=400, detail="Invalid path")
    candidate = Path(raw)
    if candidate.is_absolute() or raw.startswith(("/", "\\")):
        raise HTTPException(
            status_code=400,
            detail="Path must be relative to the library root",
        )
    if ".." in candidate.parts:
        raise HTTPException(status_code=400, detail="Path must not traverse outside the library")
    try:
        resolved = (root / candidate).resolve()
    except OSError as exc:  # e.g. a symlink loop
        raise HTTPException(status_code=400, detail="Invalid path") from exc
    if resolved != root and not resolved.is_relative_to(root):
        # Covers ``..`` smuggled through a symlink as well as a symlink whose target
        # simply lives elsewhere on the filesystem.
        raise HTTPException(status_code=400, detail="Path is outside the library root")
    return resolved


def _relative(root: Path, path: Path) -> str:
    """Root-relative POSIX path, never absolute (see the module docstring)."""
    try:
        return path.relative_to(root).as_posix()
    except ValueError:  # pragma: no cover - _safe_path already guarantees containment
        return path.name


def _not_found(root: Path, target: Path) -> Any:
    return HTTPException(status_code=404, detail=f"File not found: {_relative(root, target)}")


def _iso(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()


def _entry(path: Path, root: Path, *, is_dir: bool, stat_result: os.stat_result) -> dict[str, Any]:
    return {
        "name": path.name,
        "path": _relative(root, path),
        "is_dir": is_dir,
        "size": 0 if is_dir else int(stat_result.st_size),
        "modified": _iso(stat_result.st_mtime),
        "ext": "" if is_dir else path.suffix.lower().lstrip("."),
    }


def _scan(directory: Path, root: Path, depth: int, counter: dict[str, int]) -> list[dict[str, Any]]:
    """One directory level, recursing into subdirectories, dotfiles skipped."""
    out: list[dict[str, Any]] = []
    try:
        with os.scandir(directory) as iterator:
            children = list(iterator)
    except OSError as exc:
        logger.warning("library: cannot list %s: %s", _relative(root, directory) or ".", exc)
        return out

    def _is_dir(item: os.DirEntry[str]) -> bool:
        try:
            return item.is_dir()
        except OSError:
            return False

    # Directories first, then case-insensitive by name — the order the UI renders.
    children.sort(key=lambda item: (not _is_dir(item), item.name.lower()))

    for child in children:
        if child.name.startswith("."):
            continue
        path = Path(child.path)
        try:
            resolved = path.resolve()
        except OSError:
            continue
        if resolved != root and not resolved.is_relative_to(root):
            # A symlink pointing out of the library is not part of the library.
            continue
        try:
            is_dir = child.is_dir()
            if not is_dir and not child.is_file():
                continue  # sockets, fifos, broken symlinks
            stat_result = child.stat()
        except OSError:
            continue

        record = _entry(path, root, is_dir=is_dir, stat_result=stat_result)
        if is_dir:
            record["children"] = (
                _scan(path, root, depth + 1, counter) if depth + 1 < MAX_TREE_DEPTH else []
            )
        else:
            counter["files"] += 1
            if counter["emitted"] >= MAX_TREE_FILES:
                continue
            counter["emitted"] += 1
        out.append(record)
    return out


def _build_tree(root: Path) -> tuple[list[dict[str, Any]], int, bool]:
    counter = {"files": 0, "emitted": 0}
    entries = _scan(root, root, 0, counter)
    return entries, counter["files"], counter["files"] > counter["emitted"]


def _media_type(path: Path) -> str:
    ext = path.suffix.lower().lstrip(".")
    explicit = _EXPLICIT_MEDIA_TYPES.get(ext)
    if explicit:
        return explicit
    guessed, _encoding = mimetypes.guess_type(path.name)
    return guessed or "application/octet-stream"


def _read_pdf_pages(path: Path, start: int, count: int) -> tuple[int, list[dict[str, Any]]]:
    """Extract text for a page window.

    ``extractors.extract_text`` concatenates every page into one string, which is the
    right shape for the ingest pipeline and the wrong one here: a paged preview needs
    the boundaries that concatenation throws away. Encryption is handled the same way
    it is in ``memgraphrag/parser/legacy/extractors.py`` (``PDF_DECRYPT_PASSWORD``).
    """
    try:
        from pypdf import PdfReader  # type: ignore
    except ImportError as exc:  # pragma: no cover - pypdf ships with the api extra
        raise LibraryPreviewError("pypdf is not installed; cannot preview PDF files") from exc

    reader = PdfReader(str(path))
    if reader.is_encrypted:
        password = os.getenv("PDF_DECRYPT_PASSWORD") or None
        if reader.decrypt(password or "") == 0:
            if password:
                raise LibraryPreviewError("Incorrect PDF password")
            raise LibraryPreviewError("PDF is encrypted but no password provided")

    page_count = len(reader.pages)
    pages: list[dict[str, Any]] = []
    for index in range(start - 1, min(start - 1 + count, page_count)):
        try:
            text = reader.pages[index].extract_text() or ""
        except Exception as exc:  # noqa: BLE001 - one unreadable page must not kill the window
            logger.warning("library: page %d of %s is unreadable: %s", index + 1, path.name, exc)
            text = ""
        pages.append({"page": index + 1, "text": text})
    return page_count, pages


def _read_text_page(path: Path) -> tuple[int, list[dict[str, Any]], bool]:
    """Whole file as page 1, capped so one big log file cannot blow up the response."""
    try:
        with path.open("rb") as handle:
            raw = handle.read(MAX_TEXT_PREVIEW_BYTES + 1)
    except OSError as exc:
        raise LibraryPreviewError(f"Cannot read file: {exc}") from exc
    truncated = len(raw) > MAX_TEXT_PREVIEW_BYTES
    text = raw[:MAX_TEXT_PREVIEW_BYTES].decode("utf-8", errors="replace")
    return 1, [{"page": 1, "text": text}], truncated


def _header_bearer(request: Any) -> Optional[str]:
    header = request.headers.get("Authorization") or ""
    if not header.lower().startswith("bearer "):
        return None
    return header[7:].strip() or None


def _token_grants_access(request: Any, token: str, api_key_configured: bool) -> bool:
    """Validate a token exactly as the combined dependency does, minus the header.

    Mirrors ``get_combined_auth_dependency``'s role logic on purpose: a guest token
    must not stand in for a missing API key just because it arrived in the query
    string instead of a header.
    """
    from memgraphrag.api.dependencies import resolve_auth_context

    handler, _patterns, configured = resolve_auth_context(request)
    try:
        info = handler.validate_token(token)
    except Exception:  # noqa: BLE001 - an unreadable token is simply not an identity
        return False
    role = str(info.get("role") or "")
    if configured:
        return role != "guest"
    return role == "guest" and not api_key_configured


def _passage_row(record: Any) -> dict[str, str]:
    chunk_id = str(record.get("chunk_id") or "")
    content = str(record.get("content") or "")
    return {"chunk_id": chunk_id, "content": content[:PASSAGE_CONTENT_LIMIT]}


def _neo4j_graph(rag: Any) -> Any:
    """The graph store when it can answer Cypher, else ``None``.

    Duck-typed rather than ``isinstance``: importing ``neo4j_impl`` here would drag the
    optional ``neo4j`` driver into every deployment that runs on the igraph default.
    """
    graph = getattr(rag, "graph", None)
    if graph is None:
        return None
    if hasattr(graph, "_session") and hasattr(graph, "_workspace_label"):
        return graph
    return None


def create_library_router(api_key: Optional[str] = None) -> Any:
    """Build the ``/library`` router."""
    if APIRouter is None:
        raise RuntimeError("fastapi is required; install memgraphrag[api]")

    from memgraphrag.api.dependencies import get_combined_auth_dependency

    router = APIRouter(prefix="/library", tags=["library"])
    combined_auth = get_combined_auth_dependency(api_key)
    api_key_configured = bool(api_key)

    async def file_auth(
        request: Request,
        token: Optional[str] = Query(
            default=None,
            description="Bearer token, for embeds (<iframe>) that cannot send headers",
        ),
    ) -> None:
        """Header auth, plus a query-string token for this one read-only route.

        A browser PDF viewer is loaded through ``<iframe src=...>``; that request
        carries no ``Authorization`` header and no amount of client code can add one.
        The affordance is deliberately scoped to ``GET /library/file``: it is
        read-only, it never touches the billed LLM, and a token in a URL is a token in
        the referrer log and the browser history, so no other route gets it.
        """
        if token and _token_grants_access(request, token, api_key_configured):
            return
        # No usable query token: fall through to the standard check. The Security
        # parameters are resolved by hand because the dependency is invoked directly
        # rather than by FastAPI.
        await combined_auth(
            request,
            token=_header_bearer(request),
            api_key_header_value=request.headers.get("X-API-Key"),
        )

    @router.get("/tree", dependencies=[Depends(combined_auth)])
    async def library_tree(request: Request):
        root = _library_root(request)
        if not root.is_dir():
            raise HTTPException(
                status_code=404,
                detail="Library root does not exist; set LIBRARY_ROOT to a readable directory",
            )
        entries, total_files, truncated = await asyncio.to_thread(_build_tree, root)
        return {
            "root": str(root),
            "entries": entries,
            "total_files": total_files,
            "truncated": truncated,
        }

    @router.get("/file", dependencies=[Depends(file_auth)])
    async def library_file(
        request: Request,
        path: str = Query(..., description="Path relative to the library root"),
    ):
        root = _library_root(request)
        target = _safe_path(root, path)
        if not target.is_file():
            raise _not_found(root, target)
        # ``inline`` is what makes the browser render the PDF instead of downloading it;
        # Starlette percent-encodes a non-ASCII filename for us.
        return FileResponse(
            path=target,
            media_type=_media_type(target),
            filename=target.name.replace("\r", "").replace("\n", ""),
            content_disposition_type="inline",
        )

    @router.get("/preview", dependencies=[Depends(combined_auth)])
    async def library_preview(
        request: Request,
        path: str = Query(..., description="Path relative to the library root"),
        page: int = Query(default=1, ge=1, description="First page to extract (1-based)"),
        pages: int = Query(default=1, ge=1, le=MAX_PREVIEW_PAGES, description="Pages to extract"),
    ):
        root = _library_root(request)
        target = _safe_path(root, path)
        if not target.is_file():
            raise _not_found(root, target)

        suffix = target.suffix.lower().lstrip(".")
        truncated = False
        try:
            if suffix == "pdf":
                page_count, extracted = await asyncio.to_thread(
                    _read_pdf_pages, target, page, pages
                )
            elif suffix in _TEXT_SUFFIXES:
                page_count, extracted, truncated = await asyncio.to_thread(_read_text_page, target)
            else:
                raise HTTPException(
                    status_code=415,
                    detail=(
                        f"No text preview for .{suffix or '(none)'} files; "
                        "PDF and plain-text formats only"
                    ),
                )
        except LibraryPreviewError as exc:
            # The file is there and the path is legal; extraction is what failed.
            raise HTTPException(status_code=422, detail=str(exc)) from exc

        return {
            "path": _relative(root, target),
            "page_count": page_count,
            "pages": extracted,
            "truncated": truncated,
        }

    @router.get("/passages", dependencies=[Depends(combined_auth)])
    async def library_passages(
        request: Request,
        path: str = Query(..., description="Path relative to the library root"),
        limit: int = Query(default=30, ge=1, le=200),
    ):
        root = _library_root(request)
        target = _safe_path(root, path)
        relative = _relative(root, target)
        empty = {"path": relative, "passages": [], "total": 0, "linked": False}

        graph = _neo4j_graph(getattr(request.app.state, "rag", None))
        if graph is None:
            # igraph/GraphML deployments carry no queryable file_path index; the UI
            # renders the document without a provenance panel rather than an error.
            return empty

        workspace = graph._workspace_label()
        # Matched on the stored path first, then on the basename: provenance written by
        # a different run (or a moved library) keeps the file name but not the prefix.
        query = (
            f"MATCH (n:`{workspace}`:Passage) "
            "WHERE n.file_path = $fp OR n.file_path ENDS WITH $name "
            "RETURN n.entity_id AS chunk_id, n.content AS content "
            "LIMIT $limit"
        )
        rows: list[dict[str, str]] = []
        try:
            async with graph._session(default_access_mode="READ") as session:
                result = await session.run(
                    query, fp=str(target), name=target.name, limit=int(limit)
                )
                async for record in result:
                    rows.append(_passage_row(record))
                await result.consume()
        except Exception as exc:  # noqa: BLE001 - a graph outage is not a library error
            logger.warning("library: passage lookup failed for %s: %s", relative, exc)
            return empty

        if rows:
            return {"path": relative, "passages": rows, "total": len(rows), "linked": True}

        # No hit. That means either this document was never ingested, or provenance was
        # never backfilled at all — the two are worth telling apart, because the second
        # is a missing script run rather than a missing document.
        linked = False
        try:
            probe = (
                f"MATCH (n:`{workspace}`:Passage) WHERE n.file_path IS NOT NULL RETURN n LIMIT 1"
            )
            async with graph._session(default_access_mode="READ") as session:
                result = await session.run(probe)
                record = await result.single()
                await result.consume()
                linked = record is not None
        except Exception as exc:  # noqa: BLE001
            logger.warning("library: provenance probe failed: %s", exc)
        return {"path": relative, "passages": [], "total": 0, "linked": linked}

    return router
