"""Storage backend class factory.

Adapted from LightRAG ``lightrag/kg/factory.py``. Resolves a storage backend
name (e.g. ``"JsonKVStorage"``) to its concrete implementation class via
``importlib``.
"""

from __future__ import annotations

import importlib
from typing import Any, Callable

from memgraphrag.storage import STORAGES


def get_storage_class(storage_name: str) -> Callable[..., Any]:
    """Return the storage backend class for ``storage_name``.

    Args:
        storage_name: Registered class name (must exist in ``STORAGES``).

    Returns:
        The storage class object.

    Raises:
        KeyError: If ``storage_name`` is not in ``STORAGES``.
        AttributeError: If the module does not export ``storage_name``.
    """
    import_path = STORAGES[storage_name]
    module = importlib.import_module(import_path, package="memgraphrag")
    return getattr(module, storage_name)
