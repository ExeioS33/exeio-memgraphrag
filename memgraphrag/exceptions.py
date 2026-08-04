"""MemGraphRAG exception types.

Adapted from LightRAG ``lightrag/exceptions.py`` (StorageCapabilityError,
StorageRecordNotFoundError) with MemGraphRAG-native AuthError, NotReadyError,
and PipelineError.
"""

from __future__ import annotations


class StorageCapabilityError(Exception):
    """Raised when a storage backend lacks a capability the caller requires."""


class StorageRecordNotFoundError(Exception):
    """Raised when a targeted storage update references a missing record."""


class NotReadyError(Exception):
    """Raised when an operation is attempted before the engine is ready."""


class AuthError(Exception):
    """Raised when authentication or authorization fails."""


class PipelineError(Exception):
    """Raised when the ingestion / processing pipeline fails."""
