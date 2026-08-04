"""Shared zip-bundle extraction for external parser engines.

Provenance: adapted from LightRAG ``lightrag/parser/external/_zip.py``.
"""

from __future__ import annotations

import io
import os
import zipfile
from pathlib import Path


def safe_extract_zip(
    payload: bytes,
    dest_dir: Path,
    *,
    max_entries: int | None = 256,
    max_total_bytes: int | None = 200_000_000,
) -> list[str]:
    """Extract a zip archive into ``dest_dir``, refusing unsafe paths."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    buf = io.BytesIO(payload)
    with zipfile.ZipFile(buf) as zf:
        infos = zf.infolist()
        if max_entries is not None and len(infos) > max_entries:
            raise RuntimeError(
                f"Refusing zip with {len(infos)} entries (max {max_entries})"
            )
        if max_total_bytes is not None:
            total = sum(info.file_size for info in infos)
            if total > max_total_bytes:
                raise RuntimeError(
                    f"Refusing zip: uncompressed size {total} bytes "
                    f"exceeds limit {max_total_bytes}"
                )
        names = zf.namelist()
        for name in names:
            norm = os.path.normpath(name)
            if (
                norm.startswith("..")
                or os.path.isabs(norm)
                or norm.startswith(("/", os.sep))
            ):
                raise RuntimeError(f"Refusing zip entry with unsafe path: {name!r}")
        zf.extractall(dest_dir)
    return names


__all__ = ["safe_extract_zip"]
