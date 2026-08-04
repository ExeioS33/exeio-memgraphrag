"""Environment variable helpers for MemGraphRAG.

Adapted from LightRAG ``lightrag/utils.py`` (``get_env_value``).
"""

from __future__ import annotations

import json
import math
import os
from typing import Any, TypeVar, overload

T = TypeVar("T")


@overload
def get_env_value(key: str, default: T, cast_type: type[T] = ...) -> T: ...


@overload
def get_env_value(key: str, default: Any, cast_type: type = str) -> Any: ...


def get_env_value(key: str, default: Any, cast_type: type = str) -> Any:
    """Read an environment variable with type conversion.

    Args:
        key: Environment variable name.
        default: Value returned when the variable is unset or conversion fails.
        cast_type: Target type (``str``, ``int``, ``float``, ``bool``, ``list``).

    Returns:
        Converted value from the environment, or ``default``.
    """
    value = os.getenv(key)
    if value is None:
        return default

    if cast_type is bool:
        return value.strip().lower() in ("true", "1", "yes", "t", "on")

    if cast_type is list:
        try:
            parsed = json.loads(value)
            if isinstance(parsed, list):
                return parsed
        except (json.JSONDecodeError, TypeError, ValueError):
            pass
        return default

    try:
        converted = cast_type(value)
    except (ValueError, TypeError):
        return default

    if cast_type is float and isinstance(converted, float) and not math.isfinite(converted):
        return default

    return converted
