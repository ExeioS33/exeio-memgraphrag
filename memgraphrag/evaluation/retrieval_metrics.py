"""Retrieval-side metrics: Recall@k, Evidence Recall and Context Relevance.

Provenance: ``recall_at_k`` is adapted from
``MemGraphRAG/code/src/evaluation/retrieval_eval.py`` (``RetrievalRecall``),
with set membership moved onto :func:`normalize_doc_key` so that a title
differing only by punctuation or case still counts as the same document.

``evidence_recall`` and ``context_relevance`` have no upstream implementation at
all: the paper (arXiv:2606.00610) names them and defers to GraphRAG-Bench
without giving a formula, and the published research code implements neither.
The definitions below are EXEIO decisions, documented in ``docs/Evaluation.md``.
"""

from __future__ import annotations

from typing import Sequence

from .normalization import normalize_doc_key
from .qa_metrics import MetricResult

#: Recall cut-offs reported by default, matching the research script's k list.
DEFAULT_K_LIST = (1, 5, 10, 20)


def _keys(docs: Sequence[str]) -> list[str]:
    """Normalized identity keys, de-duplicated while preserving rank order."""
    seen: set[str] = set()
    out: list[str] = []
    for doc in docs:
        key = normalize_doc_key(doc)
        if key and key not in seen:
            seen.add(key)
            out.append(key)
    return out


def _check_lengths(gold_docs: Sequence[Sequence[str]], retrieved: Sequence[Sequence[str]]) -> None:
    if len(gold_docs) != len(retrieved):
        raise ValueError(
            f"gold docs ({len(gold_docs)}) and retrieved docs ({len(retrieved)}) "
            "must be aligned one-to-one"
        )


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def recall_at_k(
    gold_docs: Sequence[Sequence[str]],
    retrieved_docs: Sequence[Sequence[str]],
    k_list: Sequence[int] = DEFAULT_K_LIST,
) -> MetricResult:
    """Recall@k: share of gold documents present in the top-k retrieved list.

    A question with no gold documents scores 0.0 rather than being dropped, so
    that a dataset slice whose labels are missing cannot quietly raise the mean.
    """
    _check_lengths(gold_docs, retrieved_docs)
    ks = sorted({int(k) for k in k_list if int(k) > 0})
    per_example: list[dict[str, float]] = []
    for golds, retrieved in zip(gold_docs, retrieved_docs):
        gold_keys = set(_keys(golds))
        retrieved_keys = _keys(retrieved)
        row = {}
        for k in ks:
            if not gold_keys:
                row[f"Recall@{k}"] = 0.0
                continue
            hits = gold_keys & set(retrieved_keys[:k])
            row[f"Recall@{k}"] = len(hits) / len(gold_keys)
        per_example.append(row)
    pooled = {f"Recall@{k}": _mean([row[f"Recall@{k}"] for row in per_example]) for k in ks}
    return MetricResult(pooled=pooled, per_example=per_example)


def evidence_recall(
    gold_docs: Sequence[Sequence[str]],
    retrieved_docs: Sequence[Sequence[str]],
) -> MetricResult:
    """Evidence Recall: share of gold supporting documents inside the whole context.

    EXEIO definition. Unlike :func:`recall_at_k` it takes no ``k``: it scores the
    context the generator actually received, however many passages that was.
    Reporting both matters — Recall@5 answers "is the ranker good", Evidence
    Recall answers "did the answer have a chance of being grounded", and a
    configuration change that raises ``TOP_K`` moves the second without moving
    the first.
    """
    _check_lengths(gold_docs, retrieved_docs)
    per_example: list[dict[str, float]] = []
    for golds, retrieved in zip(gold_docs, retrieved_docs):
        gold_keys = set(_keys(golds))
        retrieved_keys = set(_keys(retrieved))
        score = len(gold_keys & retrieved_keys) / len(gold_keys) if gold_keys else 0.0
        per_example.append({"EvidenceRecall": score})
    return MetricResult(
        pooled={"EvidenceRecall": _mean([row["EvidenceRecall"] for row in per_example])},
        per_example=per_example,
    )


def context_relevance(
    gold_docs: Sequence[Sequence[str]],
    retrieved_docs: Sequence[Sequence[str]],
) -> MetricResult:
    """Context Relevance: share of the retrieved context that is gold evidence.

    EXEIO definition — the precision counterpart of Evidence Recall, i.e. the
    signal density of the context window. It is the metric that punishes the
    cheap way to win Evidence Recall (retrieve everything); the two are only
    meaningful reported together, which is why the runner always emits both.

    A question that retrieved nothing scores 0.0: an empty context is not
    perfectly relevant, it is a retrieval failure.
    """
    _check_lengths(gold_docs, retrieved_docs)
    per_example: list[dict[str, float]] = []
    for golds, retrieved in zip(gold_docs, retrieved_docs):
        gold_keys = set(_keys(golds))
        retrieved_keys = _keys(retrieved)
        score = (
            len([key for key in retrieved_keys if key in gold_keys]) / len(retrieved_keys)
            if retrieved_keys
            else 0.0
        )
        per_example.append({"ContextRelevance": score})
    return MetricResult(
        pooled={"ContextRelevance": _mean([row["ContextRelevance"] for row in per_example])},
        per_example=per_example,
    )
