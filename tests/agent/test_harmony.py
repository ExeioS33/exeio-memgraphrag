"""Keeping only the final channel of a harmony-format answer.

Provoked by measurement, not by hypothesis. The same question answered in 775 clean
characters through `mode=ppr` came back as 9 661 characters through `mode=agent`,
the real answer buried in the last 1 100 behind the model's reasoning and a
transcript in which it invented its own `<<<PASSAGE n>>>` blocks. Two renderings of
the same format were observed on one gateway, so both are pinned here.
"""

from __future__ import annotations

import pytest

from memgraphrag.agent.harmony import HarmonyFilter

pytestmark = pytest.mark.offline


def run(chunks: list[str]) -> str:
    f = HarmonyFilter()
    out = [f.feed(chunk) for chunk in chunks]
    out.append(f.flush())
    return "".join(out)


def test_a_normal_answer_passes_through_untouched() -> None:
    """Most models never use harmony; they must be entirely unaffected."""
    text = "Les principaux thèmes sont [1] la facturation et [2] l'adressage."
    assert run(list(text)) == text
    assert run([text]) == text


def test_a_short_answer_is_not_swallowed_by_the_probe() -> None:
    """The filter holds a few characters before deciding; flush must return them."""
    assert run(["Oui."]) == "Oui."
    assert run(list("Non.")) == "Non."


def test_tagged_rendering_keeps_only_the_final_channel() -> None:
    stream = (
        "<|channel|>analysis<|message|>We need to answer. The corpus is about "
        "invoicing.<|end|>"
        "<|start|>assistant<|channel|>final<|message|>"
        "Les thèmes sont la facturation [1] et l'adressage [2]."
    )
    assert run([stream]) == "Les thèmes sont la facturation [1] et l'adressage [2]."


def test_untagged_rendering_keeps_only_what_follows_assistantfinal() -> None:
    """The gateway strips the tags but welds the channel names to their text."""
    stream = (
        "analysisWe need to answer the question. The corpus likely includes "
        "documents about invoicing.assistantfinal**Principaux thèmes**\n\n- Facturation [3]"
    )
    assert run([stream]) == "**Principaux thèmes**\n\n- Facturation [3]"


def test_a_fabricated_tool_transcript_never_reaches_the_user() -> None:
    """The dangerous half. Asked to stream a tool turn, the model wrote its own
    tool call *and its own passages*; showing that would look like grounded text."""
    stream = (
        "analysisLet me search.assistantcommentary to=functions.retrieve "
        'json{"query":"cas"}assistantcommentary to=functions.retrieve '
        'json"<<<PASSAGE 18>>> source=invented.pdf\\nfabricated content"'
        "assistantfinalVoici la vraie réponse [1]."
    )
    out = run([stream])
    assert out == "Voici la vraie réponse [1]."
    assert "PASSAGE 18" not in out
    assert "functions.retrieve" not in out


def test_markers_split_across_deltas_are_still_recognised() -> None:
    """Tokens arrive a few characters at a time; a separator is rarely whole."""
    stream = "analysisthinking about it a whileassistantfinalAnswer here."
    chunks = [stream[i : i + 3] for i in range(0, len(stream), 3)]
    assert run(chunks) == "Answer here."


def test_a_trailing_tag_after_the_final_channel_is_cut() -> None:
    assert run(["<|channel|>final<|message|>Voici la réponse.<|return|>"]) == ("Voici la réponse.")


def test_a_channel_that_never_closes_falls_back_to_stripped_text() -> None:
    """A blank answer would be worse than a de-scaffolded one."""
    out = run(["<|channel|>analysis<|message|>Only reasoning, no final channel.<|end|>"])
    assert "<|" not in out
    assert "Only reasoning" in out


def test_a_word_that_merely_starts_with_analysis_is_not_treated_as_a_channel() -> None:
    """`analysis` opens a channel only when welded to the next word, which is the
    concatenation artefact. Ordinary prose must survive."""
    text = "analysis of the corpus shows three themes."
    assert run([text]) == text


def test_nothing_is_emitted_before_the_final_channel_opens() -> None:
    f = HarmonyFilter()
    assert f.feed("analysisprivate reasoning") == ""
    assert f.feed(" and more reasoning") == ""
    assert f.feed("assistantfinalRéponse") == "Réponse"


def test_a_lone_angle_bracket_is_not_swallowed() -> None:
    text = "a < b and c <= d, which is fine"
    assert run(list(text)) == text
