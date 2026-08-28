"""Tolerant JSON object extraction from LLM text responses.

Provenance: lifted from ``memgraphrag.openie.openai_openie`` for reuse by
ontology / conflict stages.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

logger = logging.getLogger(__name__)

try:
    import json_repair  # type: ignore

    def _loads(text: str) -> Any:
        return json_repair.loads(text)

except ImportError:  # pragma: no cover - optional dependency

    def _loads(text: str) -> Any:
        return json.loads(text)


def extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response, repairing when possible."""
    text = (text or "").strip()
    if not text:
        return {}

    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    try:
        data = _loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = _loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception as exc:
            # Small instruction-tuned models routinely break JSON on French text:
            # an unescaped quote inside a string, a trailing comma, a truncated
            # last element. Measured on the RFE corpus, 2 % of extraction calls
            # came back like that and were treated as empty. json_repair (the same
            # fallback LightRAG relies on) recovers the well-formed prefix.
            repaired = _repair(match.group(0))
            if repaired is not None:
                logger.debug("Repaired malformed LLM JSON (%s)", exc)
                return repaired
            logger.warning("Failed to parse LLM JSON: %s", exc)
    return {}


def _repair(text: str) -> dict[str, Any] | None:
    try:
        from json_repair import repair_json

        data = repair_json(text, return_objects=True)
    except Exception:
        return None
    return data if isinstance(data, dict) and data else None
