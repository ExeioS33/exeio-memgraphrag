"""Which models can actually call tools.

This exists because the failure is silent. A model without tool support receives
``tools``, ignores it, and answers in prose — the same shape as a model that simply
decided no retrieval was needed. Nothing errors, nothing logs, and the symptom
reaches the user as "the agent ignores my documents".

Two layers, and the second is the one that actually decides:

1. :func:`precheck_model` refuses names that cannot possibly work (embedding,
   rerank, transcription, image models) before a call is billed. It is a filter,
   not an oracle: an unknown name passes.
2. The loop forces the first turn with ``tool_choice`` naming a function. A model
   that supports tools must answer with a tool call; one that answers prose to a
   forced call has proved it cannot. That is a runtime fact rather than a guess
   from a name, which is why the built-in list below stays deliberately small.
"""

from __future__ import annotations

import os

#: Substrings that mark a model as unusable for chat tool calling. These are not
#: "probably weak" names — they are models that do not do chat completion at all,
#: or do it without any function-calling surface.
_INCAPABLE_MARKERS = (
    "embed",
    "rerank",
    "whisper",
    "stable-diffusion",
    "flux",
    "dall-e",
    "tts",
    "guard",  # safety classifiers answer with a verdict, not with tool calls
    "moderation",
)


class AgentUnsupportedModelError(RuntimeError):
    """Raised when the selected model cannot drive the agent loop."""


def _override_list(name: str) -> tuple[str, ...]:
    raw = (os.getenv(name) or "").strip()
    return tuple(part.strip().lower() for part in raw.split(",") if part.strip())


def precheck_model(model: str | None) -> None:
    """Refuse a model that cannot call tools, before spending a request on it.

    ``AGENT_TOOL_MODELS`` pins an explicit allow-list when an operator knows their
    gateway better than a substring test does; ``AGENT_TOOL_DENY`` extends the
    built-in markers. Both are matched as substrings, case-folded.
    """
    name = (model or "").strip().lower()
    if not name:
        return

    allow = _override_list("AGENT_TOOL_MODELS")
    if allow:
        if not any(marker in name for marker in allow):
            raise AgentUnsupportedModelError(
                f"Model {model!r} is not in AGENT_TOOL_MODELS, so the agent mode refuses "
                "to run it. Pick a tool-capable model, or add this one to the list."
            )
        return

    markers = _INCAPABLE_MARKERS + _override_list("AGENT_TOOL_DENY")
    for marker in markers:
        if marker in name:
            raise AgentUnsupportedModelError(
                f"Model {model!r} does not support tool calling (matched {marker!r}), so "
                "the agent mode cannot use it. Choose a chat model that supports "
                "functions, or query with mode=ppr."
            )


def unsupported_after_forced_call(model: str | None) -> AgentUnsupportedModelError:
    """The error raised when a forced tool call came back as prose."""
    return AgentUnsupportedModelError(
        f"Model {model or 'the selected model'} answered a forced tool call with plain "
        "text, which means it does not implement tool calling. The agent mode needs a "
        "tool-capable model; mode=ppr works with any model."
    )
