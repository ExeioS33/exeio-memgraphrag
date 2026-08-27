"""Guards tying the harness to its documentation and its two entry points.

A frozen definition is only frozen while the document that records it says the
same thing as the code. These tests fail when a constant is edited without the
decision being re-documented, and when either script loses a flag the protocol
in docs/Evaluation.md tells people to type.

The scripts are read with ``ast``, never imported: importing them loads the
developer's ``.env`` into the test process, which is exactly the hermeticity
failure AGENTS.md warns about.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from memgraphrag.evaluation.judge import LLM_ACC_PROMPT_VERSION, summarize_verdicts
from memgraphrag.evaluation.qa_metrics import (
    STR_ACC_VERBOSITY_RATIO,
    STR_ACC_WINDOW_TOKENS,
    exact_match,
    f1_score,
    str_acc,
)
from memgraphrag.evaluation.retrieval_metrics import (
    context_relevance,
    evidence_recall,
    recall_at_k,
)

pytestmark = pytest.mark.offline

REPO = Path(__file__).resolve().parents[2]
DOC = REPO / "docs" / "Evaluation.md"
EVALUATE = REPO / "scripts" / "evaluate.py"
BENCH = REPO / "scripts" / "bench.py"


def _emitted_metric_names() -> set[str]:
    """Every pooled key the harness can report, taken from the metrics themselves."""
    gold = [["Paris"]]
    predicted = ["Paris"]
    names: set[str] = set()
    for result in (
        str_acc(gold, predicted),
        exact_match(gold, predicted),
        f1_score(gold, predicted),
        recall_at_k([["Doc"]], [["Doc"]]),
        evidence_recall([["Doc"]], [["Doc"]]),
        context_relevance([["Doc"]], [["Doc"]]),
        summarize_verdicts([]),
    ):
        names |= set(result.pooled)
    return names


def test_documentation_exists_and_supersedes_the_reproduce_note() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert "Reproduce.md" in text
    assert "arXiv:2606.00610" in text


def test_every_reported_metric_is_documented() -> None:
    """A metric nobody documented is a number nobody can defend."""
    text = DOC.read_text(encoding="utf-8")
    undocumented = sorted(name for name in _emitted_metric_names() if name not in text)
    assert not undocumented, f"metrics reported but absent from {DOC.name}: {undocumented}"


def test_frozen_constants_match_the_documented_values() -> None:
    text = DOC.read_text(encoding="utf-8")
    assert str(STR_ACC_WINDOW_TOKENS) in text
    assert str(int(STR_ACC_VERBOSITY_RATIO)) in text
    assert LLM_ACC_PROMPT_VERSION in text


def test_the_corrected_replication_target_is_on_the_record() -> None:
    """Table 1's G-Novel cell is a misprint; a campaign must not chase it."""
    text = DOC.read_text(encoding="utf-8")
    for figure in ("59.63", "59.25", "54.41", "57.41", "56.48", "38.30"):
        assert figure in text, f"{figure} missing from {DOC.name}"


@pytest.mark.parametrize(
    ("script", "flags"),
    [
        (
            EVALUATE,
            [
                "--dataset-root",
                "--list-datasets",
                "--runs",
                "--sample",
                "--judge",
                "--write-golden",
                "--compare",
                "--tolerance",
                "--output",
            ],
        ),
        (BENCH, ["--dataset-root", "--queries", "--concurrency", "--warmup", "--with-answer"]),
    ],
    ids=["evaluate", "bench"],
)
def test_scripts_expose_the_documented_flags(script: Path, flags: list[str]) -> None:
    source = script.read_text(encoding="utf-8")
    ast.parse(source, filename=str(script))
    missing = [flag for flag in flags if f'"{flag}"' not in source]
    assert not missing, f"{script.name} lost documented flags: {missing}"


@pytest.mark.parametrize("script", [EVALUATE, BENCH], ids=["evaluate", "bench"])
def test_scripts_degrade_with_a_message_when_the_checkout_is_absent(script: Path) -> None:
    """Without the research datasets both scripts must explain themselves, not traceback."""
    source = script.read_text(encoding="utf-8")
    assert "DatasetUnavailableError" in source
    assert "return 2" in source
