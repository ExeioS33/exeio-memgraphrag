"""Answer-side metrics: Exact Match, token F1 and the frozen Str-Acc.

Provenance: ``exact_match`` and ``f1_score`` are adapted from
``MemGraphRAG/code/src/evaluation/qa_eval.py`` (``QAExactMatch`` / ``QAF1Score``,
themselves the MRQA official evaluation), reduced to plain functions and to this
repository's normalizer.

``str_acc`` has no upstream implementation. The paper (arXiv:2606.00610) only
says the metric checks "whether the gold answer is included in the generated
answer after normalizing them to lowercase words", which does not say whether
inclusion is a substring test or a token test, nor which normalization applies,
nor what stops a long Markdown answer from containing the gold span by accident.
This module freezes those choices as an EXEIO decision; ``docs/Evaluation.md``
records the rationale and the alternatives that were rejected.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass, field
from typing import Callable, Sequence

from .normalization import normalize_tokens

#: Only the first N normalized tokens of a prediction are scanned for the gold
#: span. Inclusion is monotone in answer length: with no bound, a verbose enough
#: answer eventually contains almost any short gold string, so an unbounded
#: Str-Acc rewards verbosity rather than correctness. 200 tokens is roughly the
#: length of a direct answer plus one paragraph of justification.
STR_ACC_WINDOW_TOKENS = 200

#: A hit is flagged "verbose" when the prediction is this many times longer than
#: the gold answer. Verbose hits are counted, not discarded: the count is the
#: evidence that a Str-Acc figure does or does not rest on padding.
STR_ACC_VERBOSITY_RATIO = 10.0


@dataclass(frozen=True)
class MetricResult:
    """Pooled scores plus the per-example scores they were averaged from.

    Per-example scores are kept because a mean alone cannot tell a uniform
    mediocre run from a bimodal one, and the golden-set comparison needs the
    individual answers to explain a drop.
    """

    pooled: dict[str, float] = field(default_factory=dict)
    per_example: list[dict[str, float]] = field(default_factory=list)


def _check_lengths(gold_answers: Sequence[Sequence[str]], predictions: Sequence[str]) -> None:
    if len(gold_answers) != len(predictions):
        raise ValueError(
            f"gold answers ({len(gold_answers)}) and predictions ({len(predictions)}) "
            "must be aligned one-to-one"
        )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def exact_match(
    gold_answers: Sequence[Sequence[str]],
    predictions: Sequence[str],
    aggregate: Callable[[Sequence[float]], float] = max,
) -> MetricResult:
    """Exact Match: the whole normalized prediction equals a normalized gold answer."""
    _check_lengths(gold_answers, predictions)
    per_example: list[dict[str, float]] = []
    for golds, predicted in zip(gold_answers, predictions):
        predicted_norm = " ".join(normalize_tokens(predicted))
        scores = [
            1.0 if " ".join(normalize_tokens(gold)) == predicted_norm else 0.0 for gold in golds
        ]
        per_example.append({"ExactMatch": float(aggregate(scores)) if scores else 0.0})
    return MetricResult(
        pooled={"ExactMatch": _mean([row["ExactMatch"] for row in per_example])},
        per_example=per_example,
    )


def _token_f1(gold: str, predicted: str) -> float:
    gold_tokens = normalize_tokens(gold)
    predicted_tokens = normalize_tokens(predicted)
    if not gold_tokens or not predicted_tokens:
        # Two empty strings are an exact match for EM but have no F1 support;
        # scoring them 1.0 would let an engine that answers nothing look perfect.
        return 0.0
    common = Counter(predicted_tokens) & Counter(gold_tokens)
    num_same = sum(common.values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(predicted_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def f1_score(
    gold_answers: Sequence[Sequence[str]],
    predictions: Sequence[str],
    aggregate: Callable[[Sequence[float]], float] = max,
) -> MetricResult:
    """Bag-of-tokens F1 between prediction and gold answer (MRQA definition)."""
    _check_lengths(gold_answers, predictions)
    per_example = [
        {"F1": float(aggregate([_token_f1(gold, predicted) for gold in golds])) if golds else 0.0}
        for golds, predicted in zip(gold_answers, predictions)
    ]
    return MetricResult(
        pooled={"F1": _mean([row["F1"] for row in per_example])},
        per_example=per_example,
    )


def contains_token_span(gold: str, predicted: str, window_tokens: int) -> bool:
    """True when the gold token sequence occurs contiguously in the prediction.

    Contiguous token containment, not raw substring containment: a substring test
    fires on "art" inside "start" and on "US" inside "USSR", which inflates
    Str-Acc on exactly the short gold answers (dates, yes/no, single names) that
    dominate HotpotQA and 2WikiMultihopQA.
    """
    gold_tokens = normalize_tokens(gold)
    if not gold_tokens:
        # An empty gold answer is unscorable; treating it as "contained" would
        # hand a free point to every prediction, including the empty one.
        return False
    predicted_tokens = normalize_tokens(predicted)[:window_tokens]
    span = len(gold_tokens)
    return any(
        predicted_tokens[i : i + span] == gold_tokens
        for i in range(len(predicted_tokens) - span + 1)
    )


def str_acc(
    gold_answers: Sequence[Sequence[str]],
    predictions: Sequence[str],
    window_tokens: int = STR_ACC_WINDOW_TOKENS,
    verbosity_ratio: float = STR_ACC_VERBOSITY_RATIO,
) -> MetricResult:
    """Str-Acc, frozen as bounded contiguous token containment of a gold answer.

    Pooled output carries two audit figures next to the score itself:

    * ``StrAccVerboseHitRate`` — share of *hits* whose prediction is more than
      ``verbosity_ratio`` times longer than the gold answer.
    * ``StrAccTruncatedRate`` — share of predictions long enough to be cut by
      ``window_tokens``, i.e. the share of the run where the bound was load-bearing.

    A campaign that reports Str-Acc without these two numbers cannot tell an
    accuracy gain from a verbosity gain.
    """
    _check_lengths(gold_answers, predictions)
    per_example: list[dict[str, float]] = []
    for golds, predicted in zip(gold_answers, predictions):
        predicted_tokens = normalize_tokens(predicted)
        hit = any(contains_token_span(gold, predicted, window_tokens) for gold in golds)
        shortest_gold = min((len(normalize_tokens(g)) for g in golds if g), default=0)
        verbose = bool(
            hit
            and shortest_gold > 0
            and len(predicted_tokens) > verbosity_ratio * shortest_gold
        )
        per_example.append(
            {
                "StrAcc": 1.0 if hit else 0.0,
                "StrAccVerboseHit": 1.0 if verbose else 0.0,
                "StrAccTruncated": 1.0 if len(predicted_tokens) > window_tokens else 0.0,
            }
        )
    hits = [row["StrAcc"] for row in per_example]
    verbose_hits = [row["StrAccVerboseHit"] for row in per_example]
    total_hits = sum(hits)
    return MetricResult(
        pooled={
            "StrAcc": _mean(hits),
            "StrAccVerboseHitRate": (sum(verbose_hits) / total_hits) if total_hits else 0.0,
            "StrAccTruncatedRate": _mean([row["StrAccTruncated"] for row in per_example]),
        },
        per_example=per_example,
    )
