"""MemGraphRAG exception types.

Adapted from LightRAG ``lightrag/exceptions.py`` (StorageCapabilityError,
StorageRecordNotFoundError) with MemGraphRAG-native AuthError, NotReadyError,
CorruptKVFileError and PipelineError.
"""

from __future__ import annotations


class StorageCapabilityError(Exception):
    """Raised when a storage backend lacks a capability the caller requires."""


class StorageRecordNotFoundError(Exception):
    """Raised when a targeted storage update references a missing record."""


class CorruptKVFileError(RuntimeError):
    """Raised when a key-value file exists on disk but cannot be parsed.

    Kept distinct from a missing file: an unreadable index used to be swallowed
    into an empty dict, and the first upsert then overwrote the surviving records.
    """


class NotReadyError(Exception):
    """Raised when an operation is attempted before the engine is ready."""


class AuthError(Exception):
    """Raised when authentication or authorization fails."""


class PipelineError(Exception):
    """Raised when the ingestion / processing pipeline fails."""
