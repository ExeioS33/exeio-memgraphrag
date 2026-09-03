"""Context budget for the agent loop.

The real ceiling on a multi-hop turn is not the iteration count, it is the context
window. One ``retrieve`` result carries ``top_k`` passages of ``CHUNK_SIZE`` tokens
— around 12 000 tokens at the shipped defaults — so a third hop overflows most
served windows. Without an eviction policy the failure is not "the agent loops",
it is a provider error mid-answer, after the 200 has been committed.

Eviction is deliberately blunt and visible: the oldest tool results are replaced by
a one-line placeholder, newest first kept, and the caller reports what happened in
the trace. Summarising evicted passages would cost another LLM round trip to hide
information loss that the user is better off seeing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

from memgraphrag.utils.env import get_env_value

#: Conservative default: fits the smallest window we are likely to be pointed at.
DEFAULT_CONTEXT_BUDGET = 24_000

_EVICTED = "[Earlier tool result dropped to stay inside the context window.]"


@dataclass
class BudgetReport:
    """What eviction did, for the trace and the logs."""

    tokens_before: int
    tokens_after: int
    evicted: int

    @property
    def applied(self) -> bool:
        return self.evicted > 0


def context_budget() -> int:
    """Token ceiling for the agent's message list."""
    return get_env_value("AGENT_CONTEXT_BUDGET", DEFAULT_CONTEXT_BUDGET, int)


def _encoder() -> Any:
    """A tokenizer, or ``None`` when tiktoken is unavailable.

    Counting is advisory: a wrong count evicts slightly early or slightly late,
    while a hard failure here would take down a working answer. The character
    fallback (≈ 4 chars/token) is deliberately pessimistic.
    """
    try:
        from memgraphrag.utils.tokenizer import TiktokenTokenizer

        return TiktokenTokenizer(os.getenv("TOKENIZER_MODEL") or "gpt-4o-mini")
    except Exception:
        return None


def count_tokens(messages: list[dict[str, Any]]) -> int:
    """Approximate the token cost of an OpenAI message list."""
    encoder = _encoder()
    total = 0
    for message in messages:
        content = message.get("content") or ""
        text = content if isinstance(content, str) else str(content)
        for call in message.get("tool_calls") or []:
            function = (call or {}).get("function") or {}
            text += str(function.get("name") or "") + str(function.get("arguments") or "")
        if encoder is not None:
            try:
                total += len(encoder.encode(text)) + 4
                continue
            except Exception:
                pass
        total += len(text) // 4 + 4
    return total


def enforce(messages: list[dict[str, Any]], budget: int | None = None) -> BudgetReport:
    """Evict oldest tool results in place until ``messages`` fits ``budget``.

    Only ``role="tool"`` messages are touched. The system prompt, the question and
    the assistant turns carrying ``tool_calls`` all have to stay: dropping an
    assistant turn would orphan the tool message answering it, which most providers
    reject outright.
    """
    ceiling = budget if budget is not None else context_budget()
    before = count_tokens(messages)
    if before <= ceiling:
        return BudgetReport(tokens_before=before, tokens_after=before, evicted=0)

    evicted = 0
    for message in messages:
        if count_tokens(messages) <= ceiling:
            break
        if message.get("role") != "tool":
            continue
        if message.get("content") == _EVICTED:
            continue
        message["content"] = _EVICTED
        evicted += 1
    return BudgetReport(tokens_before=before, tokens_after=count_tokens(messages), evicted=evicted)
