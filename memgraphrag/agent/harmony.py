"""Keep only the final channel of a harmony-format answer.

gpt-oss models answer in the harmony format — a reasoning channel, then the actual
answer — whenever the conversation carries tool messages. That is not a property of
passing ``tools`` on the request: it survives the closing, tools-free call, because
the message list still holds the assistant turn with ``tool_calls`` and the
``role="tool"`` results answering it. Which is why ``mode=ppr``, whose conversation
has neither, is unaffected and needs none of this.

Two renderings reach us, and both were observed on the same gateway:

*Tagged* — the markers are forwarded verbatim::

    <|channel|>analysis<|message|>We need to answer…<|end|>
    <|start|>assistant<|channel|>final<|message|>Voici les thèmes…

*Untagged* — the gateway strips ``<|…|>`` but leaves the channel names welded to
their text, separated by ``assistantfinal``::

    analysisWe need to answer… assistantfinal**Principaux thèmes**…

Unfiltered, either one puts the model's private reasoning — and, worse, a
transcript in which it invents its own ``<<<PASSAGE n>>>`` blocks — in front of the
user. One measured question: 775 clean characters through ``ppr``, 9 661 characters
of scaffolding through ``agent``, with the real answer buried in the last 1 100.

The filter is conservative in both directions. A stream showing neither signature
passes through untouched. A stream that opens a channel but never closes it falls
back to the text with its scaffolding stripped rather than to nothing at all.
"""

from __future__ import annotations

import re

TAGGED_FINAL = "<|channel|>final<|message|>"
UNTAGGED_FINAL = "assistantfinal"

#: How much of the stream to hold before deciding which rendering this is. The
#: untagged signature is at the very start, so this is short enough to be
#: imperceptible and long enough to be unambiguous.
PROBE_CHARS = 24

_UNTAGGED_OPEN = re.compile(r"^(analysis|commentary)(?=\S)")
_TAG = re.compile(r"<\|[^|>]*\|>")
_CHANNEL_NAME = re.compile(r"^(analysis|final|commentary)\b\s*")

_DECIDING = "deciding"
_PASSTHROUGH = "passthrough"
_WAITING = "waiting"
_EMITTING = "emitting"
_CLOSED = "closed"


class HarmonyFilter:
    """Feed streamed text in, get user-facing text out."""

    def __init__(self) -> None:
        self._buffer = ""
        self._state = _DECIDING
        self._separator = ""

    @property
    def filtered(self) -> bool:
        """Whether the stream was recognised as harmony-formatted."""
        return self._state in (_WAITING, _EMITTING, _CLOSED) and bool(self._separator)

    def feed(self, text: str) -> str:
        if self._state == _CLOSED:
            return ""
        self._buffer += text

        if self._state == _DECIDING:
            self._decide()
        if self._state == _DECIDING:
            return ""

        if self._state == _WAITING:
            index = self._buffer.find(self._separator)
            if index == -1:
                # Hold everything: the answer has not started, and the reasoning
                # must never be shown and then taken back.
                return ""
            self._buffer = self._buffer[index + len(self._separator) :]
            self._state = _EMITTING

        return self._release()

    def flush(self) -> str:
        """Close the stream and return whatever is still owed to the user."""
        if self._state == _CLOSED:
            return ""
        remainder = self._buffer
        self._buffer = ""
        state, self._state = self._state, _CLOSED
        if state == _WAITING:
            # A channel opened and never closed. A de-scaffolded answer beats a
            # blank one, even if it carries some reasoning with it.
            return _strip_scaffolding(remainder)
        return remainder

    # -- internals ---------------------------------------------------------- #

    def _decide(self) -> None:
        buffer = self._buffer
        if TAGGED_FINAL in buffer or "<|" in buffer:
            self._separator = TAGGED_FINAL
            self._state = _WAITING
            return
        if _UNTAGGED_OPEN.match(buffer):
            self._separator = UNTAGGED_FINAL
            self._state = _WAITING
            return
        if len(buffer) >= PROBE_CHARS:
            self._state = _PASSTHROUGH

    def _release(self) -> str:
        """Emit what is safe, holding back a possible partial marker."""
        if self._state == _EMITTING and self._separator == TAGGED_FINAL:
            cut = self._buffer.find("<|")
            if cut != -1:
                out = self._buffer[:cut]
                self._buffer = ""
                self._state = _CLOSED
                return out
        hold = 1 if self._buffer.endswith("<") else 0
        out = self._buffer[: len(self._buffer) - hold]
        self._buffer = self._buffer[len(self._buffer) - hold :]
        return out


def _strip_scaffolding(text: str) -> str:
    """Remove ``<|…|>`` tags and leading channel names."""
    parts = [_CHANNEL_NAME.sub("", chunk) for chunk in _TAG.split(text)]
    return "".join(parts).strip()
