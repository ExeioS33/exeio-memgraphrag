"""Strip OpenAI-harmony channel markers from a streamed agent answer.

gpt-oss models answer a *tool-enabled* request in the harmony format, and an
OpenAI-compatible gateway passes its channel tags through verbatim in `content`:

    <|channel|>analysis<|message|>We need to answer… <|end|>
    <|start|>assistant<|channel|>final<|message|>Voici les thèmes…

Forwarded raw, that puts the model's private reasoning — tags included — in front
of the user. Measured on this repo's default model, one question answered in 707
clean characters through ``mode=ppr`` came back as 4 599 characters of scaffolding
through ``mode=agent``. The same model, the same corpus: what changes is that the
agent path attaches ``tools``, and the model switches output format when it sees
them. So this filter belongs to the tool-calling path and nowhere else.

The filter is conservative in both directions. A stream with no ``<|`` in it is
passed through untouched, so a model that does not use harmony is unaffected. And a
stream that opens channels but never emits a ``final`` one falls back to the tags
stripped out rather than to an empty answer — showing the reasoning is bad, showing
nothing is worse.
"""

from __future__ import annotations

import re

FINAL_MARKER = "<|channel|>final<|message|>"
_TAG = re.compile(r"<\|[^|>]*\|>")
_CHANNEL_NAME = re.compile(r"^(analysis|final|commentary)\b\s*")


class HarmonyFilter:
    """Feed streamed text in, get user-facing text out."""

    def __init__(self) -> None:
        self._buffer = ""
        self._saw_markers = False
        self._emitting = False
        self._closed = False

    @property
    def saw_markers(self) -> bool:
        """Whether the stream used the harmony format at all."""
        return self._saw_markers

    def feed(self, text: str) -> str:
        """Return the part of ``text`` that should reach the user."""
        if self._closed:
            return ""
        self._buffer += text

        if self._emitting:
            return self._emit_until_next_tag()

        index = self._buffer.find(FINAL_MARKER)
        if index != -1:
            self._saw_markers = True
            self._emitting = True
            self._buffer = self._buffer[index + len(FINAL_MARKER) :]
            return self._emit_until_next_tag()

        if "<|" in self._buffer:
            # Inside a non-final channel: hold everything back. The buffer keeps
            # growing because the final channel may still be coming, and `flush`
            # needs the whole thing to fall back on if it never does.
            self._saw_markers = True
            return ""

        # No markers yet. A trailing "<" could be the first byte of one, so it is
        # held until the next delta decides.
        return self._release_all_but_partial()

    def flush(self) -> str:
        """Close the stream and return whatever is still owed to the user."""
        if self._closed:
            return ""
        self._closed = True
        remainder = self._buffer
        self._buffer = ""
        if self._emitting:
            cut = remainder.find("<|")
            return remainder if cut == -1 else remainder[:cut]
        if self._saw_markers:
            # Channels were opened but no final one arrived. Better a de-tagged
            # answer than a blank one.
            return _strip_tags(remainder)
        return remainder

    # -- internals ---------------------------------------------------------- #

    def _emit_until_next_tag(self) -> str:
        cut = self._buffer.find("<|")
        if cut != -1:
            out = self._buffer[:cut]
            self._buffer = ""
            self._emitting = False
            self._closed = True  # the final channel has ended; nothing follows
            return out
        return self._release_all_but_partial()

    def _release_all_but_partial(self) -> str:
        hold = 1 if self._buffer.endswith("<") else 0
        out = self._buffer[: len(self._buffer) - hold]
        self._buffer = self._buffer[len(self._buffer) - hold :]
        return out


def _strip_tags(text: str) -> str:
    """Remove ``<|…|>`` tags and the channel name that follows one."""
    parts = []
    for chunk in _TAG.split(text):
        parts.append(_CHANNEL_NAME.sub("", chunk))
    return "".join(parts).strip()
