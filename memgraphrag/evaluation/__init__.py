"""Evaluation harness for MemGraphRAG.

Implements the four metrics the paper (arXiv:2606.00610) reports — Str-Acc,
LLM-Acc, Context Relevance, Evidence Recall — plus the EM / F1 / Recall@k floor
ported from the research code, dataset loaders for the four benchmark sets, a
variance-aware campaign runner and a golden-set non-regression check.

The paper does not define its metrics operationally and publishes no judge
prompt, so every definition here is a frozen EXEIO decision. Read
``docs/Evaluation.md`` before comparing any number produced by this package to a
number printed in the paper.

Entry points: ``scripts/evaluate.py`` (quality) and ``scripts/bench.py``
(performance).
"""

from __future__ import annotations

from .benchmark import CallMeter, LatencyStats, LoadResult, percentile, run_load
from .datasets import (
    DATASETS,
    CorpusDocument,
    DatasetFormatError,
    DatasetSpec,
    DatasetUnavailableError,
    EvaluationExample,
    available_datasets,
    dataset_root,
    load_corpus,
    load_questions,
    sample_examples,
)
from .golden import (
    ComparisonReport,
    GoldenSet,
    GoldenSetError,
    MetricComparison,
    build_golden_set,
    compare_to_golden,
    load_golden_set,
    write_golden_set,
)
from .judge import (
    LLM_ACC_PROMPT_VERSION,
    LLM_ACC_SYSTEM,
    LLM_ACC_USER_TEMPLATE,
    JudgeVerdict,
    LLMAccJudge,
    summarize_verdicts,
)
from .normalization import normalize_answer, normalize_doc_key, normalize_tokens
from .qa_metrics import MetricResult, exact_match, f1_score, str_acc
from .retrieval_metrics import context_relevance, evidence_recall, recall_at_k
from .runner import (
    CampaignReport,
    MetricStat,
    Prediction,
    RunResult,
    aggregate_runs,
    run_campaign,
    run_once,
)

__all__ = [
    "CallMeter",
    "CampaignReport",
    "ComparisonReport",
    "CorpusDocument",
    "DATASETS",
    "DatasetFormatError",
    "DatasetSpec",
    "DatasetUnavailableError",
    "EvaluationExample",
    "GoldenSet",
    "GoldenSetError",
    "JudgeVerdict",
    "LLM_ACC_PROMPT_VERSION",
    "LLM_ACC_SYSTEM",
    "LLM_ACC_USER_TEMPLATE",
    "LLMAccJudge",
    "LatencyStats",
    "LoadResult",
    "MetricComparison",
    "MetricResult",
    "MetricStat",
    "Prediction",
    "RunResult",
    "aggregate_runs",
    "available_datasets",
    "build_golden_set",
    "compare_to_golden",
    "context_relevance",
    "dataset_root",
    "evidence_recall",
    "exact_match",
    "f1_score",
    "load_corpus",
    "load_golden_set",
    "load_questions",
    "normalize_answer",
    "normalize_doc_key",
    "normalize_tokens",
    "percentile",
    "recall_at_k",
    "run_campaign",
    "run_load",
    "run_once",
    "sample_examples",
    "str_acc",
    "summarize_verdicts",
    "write_golden_set",
]
