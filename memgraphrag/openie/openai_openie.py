"""OpenAI-compatible OpenIE (NER + triple extraction).

Provenance: adapted from MemGraphRAG ``code/src/information_extraction/openie_openai.py``
for async OpenAI-compatible LLM callables used by the industrialized package.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
from typing import Any, Awaitable, Callable, Sequence

from memgraphrag.prompts.templates import render_ner, render_triple_extraction
from memgraphrag.utils.debug_log import agent_dbg

logger = logging.getLogger(__name__)

LLMFunc = Callable[..., Awaitable[str]]

try:
    import json_repair  # type: ignore

    def _loads(text: str) -> Any:
        return json_repair.loads(text)

except ImportError:  # pragma: no cover - optional dependency

    def _loads(text: str) -> Any:
        return json.loads(text)


def _extract_json_object(text: str) -> dict[str, Any]:
    """Parse a JSON object from an LLM response, repairing when possible."""
    text = (text or "").strip()
    if not text:
        return {}

    # Prefer fenced JSON if present
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fence:
        text = fence.group(1)

    try:
        data = _loads(text)
        if isinstance(data, dict):
            return data
    except Exception:
        pass

    # Fallback: first {...} block
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            data = _loads(match.group(0))
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("Failed to parse OpenIE JSON: %s", exc)
    return {}


def _normalize_entities(raw: Any) -> list[str]:
    if isinstance(raw, dict):
        raw = raw.get("named_entities", [])
    if not isinstance(raw, list):
        return []
    entities: list[str] = []
    for item in raw:
        if isinstance(item, str) and item.strip():
            entities.append(item.strip())
        elif isinstance(item, dict):
            name = item.get("name") or item.get("entity") or item.get("text")
            if isinstance(name, str) and name.strip():
                entities.append(name.strip())
    # preserve order, drop dupes
    return list(dict.fromkeys(entities))


def _normalize_triples(raw: Any) -> list[list[str]]:
    if isinstance(raw, dict):
        raw = raw.get("triples", [])
    if not isinstance(raw, list):
        return []
    triples: list[list[str]] = []
    for item in raw:
        if isinstance(item, (list, tuple)) and len(item) == 3:
            h, r, t = (str(item[0]).strip(), str(item[1]).strip(), str(item[2]).strip())
            if h and r and t:
                triples.append([h, r, t])
        elif isinstance(item, dict):
            for key in ("triple", "processed_triple", "raw_triple"):
                val = item.get(key)
                if isinstance(val, (list, tuple)) and len(val) == 3:
                    h, r, t = (str(val[0]).strip(), str(val[1]).strip(), str(val[2]).strip())
                    if h and r and t:
                        triples.append([h, r, t])
                    break
    return triples


class OpenIE:
    """Async Open Information Extraction via an LLM complete function."""

    def __init__(
        self,
        llm_model_func: LLMFunc,
        max_concurrency: int = 4,
    ) -> None:
        self.llm_model_func = llm_model_func
        self.max_concurrency = max(1, int(max_concurrency))

    async def ner(self, passage: str) -> list[str]:
        system, user = render_ner(passage)
        try:
            response = await self.llm_model_func(user, system_prompt=system)
            data = _extract_json_object(str(response))
            return _normalize_entities(data)
        except Exception as exc:
            # #region agent log
            agent_dbg(
                "C",
                "openai_openie.py:ner",
                "NER failed",
                {
                    "passage_len": len(passage or ""),
                    "error": str(exc)[:300],
                },
            )
            # #endregion
            logger.warning("NER failed: %s", exc)
            return []

    async def triple_extraction(
        self, passage: str, named_entities: Sequence[str]
    ) -> list[list[str]]:
        system, user = render_triple_extraction(passage, list(named_entities))
        try:
            response = await self.llm_model_func(user, system_prompt=system)
            data = _extract_json_object(str(response))
            return _normalize_triples(data)
        except Exception as exc:
            logger.warning("Triple extraction failed: %s", exc)
            return []

    async def openie_one(self, idx: str, passage: str) -> dict[str, Any]:
        entities = await self.ner(passage)
        triples = await self.triple_extraction(passage, entities)
        return {
            "idx": idx,
            "passage": passage,
            "extracted_entities": entities,
            "extracted_triples": triples,
        }

    async def batch_openie(
        self,
        docs: Sequence[str] | Sequence[dict[str, str]],
    ) -> list[dict[str, Any]]:
        """Run NER then triples for each document.

        ``docs`` may be plain strings or dicts with ``idx`` / ``passage``
        (or ``content``) keys.
        """
        prepared: list[tuple[str, str]] = []
        for i, doc in enumerate(docs):
            if isinstance(doc, str):
                prepared.append((str(i), doc))
            elif isinstance(doc, dict):
                idx = str(doc.get("idx", doc.get("id", i)))
                passage = str(doc.get("passage", doc.get("content", "")))
                prepared.append((idx, passage))
            else:
                prepared.append((str(i), str(doc)))

        sem = asyncio.Semaphore(self.max_concurrency)

        async def _one(idx: str, passage: str) -> dict[str, Any]:
            async with sem:
                return await self.openie_one(idx, passage)

        results = await asyncio.gather(*[_one(i, p) for i, p in prepared])
        return list(results)
