"""Content hashing helpers for MemGraphRAG.

Adapted from LightRAG ``lightrag/utils.py`` (``compute_mdhash_id`` /
``compute_args_hash``).
"""

from __future__ import annotations

import hashlib


def compute_mdhash_id(content: str, prefix: str = "") -> str:
    """Return ``prefix`` plus the MD5 hex digest of ``content``.

    Args:
        content: String to hash.
        prefix: Optional ID prefix (e.g. ``"doc-"``, ``"chunk-"``).

    Returns:
        Prefixed MD5 hash string.
    """
    digest = hashlib.md5(content.encode("utf-8")).hexdigest()
    return f"{prefix}{digest}"
