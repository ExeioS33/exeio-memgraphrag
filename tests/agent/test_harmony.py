"""Harmony channel stripping on the agent's streamed answer.

Provoked by a measured failure, not by a hypothesis: the same question answered in
707 clean characters through `mode=ppr` came back as 4 599 characters of
`<|channel|>analysis<|message|>…` through `mode=agent`, because attaching `tools`
makes a gpt-oss model switch output format and the gateway forwards its tags.
"""

from __future__ import annotations

import pytest

from memgraphrag.agent.harmony import HarmonyFilter

pytestmark = pytest.mark.offline


def run(filter_: HarmonyFilter, chunks: list[str]) -> str:
    out = [filter_.feed(chunk) for chunk in chunks]
    out.append(filter_.flush())
    return "".join(out)


def test_a_stream_without_markers_passes_through_untouched() -> None:
    """Most models never use harmony; they must be entirely unaffected."""
    text = "Les principaux thèmes sont [1] la facturation et [2] l'adressage."
    assert run(HarmonyFilter(), list(text)) == text


def test_only_the_final_channel_reaches_the_user() -> None:
    stream = (
        "<|channel|>analysis<|message|>We need to answer the question. "
        "The corpus is about invoicing.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        "Les thèmes sont la facturation [1] et l'adressage [2]."
    )
    assert (
        run(HarmonyFilter(), [stream]) == "Les thèmes sont la facturation [1] et l'adressage [2]."
    )


def test_markers_split_across_deltas_are_still_recognised() -> None:
    """Tokens arrive a few characters at a time; a marker is rarely whole."""
    stream = (
        "<|channel|>analysis<|message|>thinking<|end|>"
        "<|start|>assistant<|channel|>final<|message|>Answer here."
    )
    chunks = [stream[i : i + 3] for i in range(0, len(stream), 3)]
    assert run(HarmonyFilter(), chunks) == "Answer here."


def test_a_trailing_marker_after_the_final_channel_is_cut() -> None:
    stream = "<|channel|>final<|message|>Voici la réponse.<|return|>"
    assert run(HarmonyFilter(), [stream]) == "Voici la réponse."


def test_channels_without_a_final_one_fall_back_to_stripped_text() -> None:
    """An empty answer would be worse than a de-tagged one."""
    stream = "<|channel|>analysis<|message|>Only reasoning, no final channel.<|end|>"
    out = run(HarmonyFilter(), [stream])
    assert "<|" not in out
    assert "Only reasoning" in out


def test_a_lone_angle_bracket_is_not_swallowed() -> None:
    """`<` is the first byte of a marker and also ordinary text."""
    text = "a < b and c <= d"
    assert run(HarmonyFilter(), list(text)) == text


def test_nothing_is_emitted_before_the_final_channel_opens() -> None:
    """The reasoning must never appear and then be retracted."""
    f = HarmonyFilter()
    assert f.feed("<|channel|>analysis<|message|>secret reasoning") == ""
    assert f.feed(" more reasoning<|end|>") == ""
    assert "Réponse" in f.feed("<|start|>assistant<|channel|>final<|message|>Réponse")
