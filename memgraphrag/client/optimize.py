"""Hybrid parameter optimization: retrieval metrics grid + LLM-as-judge."""

from __future__ import annotations

import itertools
import json
import re
from dataclasses import asdict, dataclass, field
from typing import Any, Callable, Optional, Sequence

from memgraphrag.client.http import MemGraphRAGClient
from memgraphrag.client.params import clean_params, default_sweep_grid

ProgressCb = Optional[Callable[[str, int, int], None]]


@dataclass
class SweepResult:
    """One grid combination after phase-1 (and optionally phase-2)."""

    params: dict[str, Any]
    retrieval_score: float
    mean_doc_score: float
    max_doc_score: float
    n_docs: float
    judge_score: Optional[float] = None
    answer: Optional[str] = None
    judge_rationale: Optional[str] = None
    errors: list[str] = field(default_factory=list)

    @property
    def final_score(self) -> float:
        """Prefer judge score when present; otherwise retrieval score."""
        if self.judge_score is not None:
            return 0.4 * self.retrieval_score + 0.6 * (self.judge_score / 10.0)
        return self.retrieval_score


@dataclass
class OptimizeReport:
    question: str
    results: list[SweepResult]
    recommended: dict[str, Any]
    phase1_count: int
    phase2_count: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "question": self.question,
            "recommended": self.recommended,
            "phase1_count": self.phase1_count,
            "phase2_count": self.phase2_count,
            "results": [asdict(r) for r in self.results],
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.to_dict(), indent=indent, ensure_ascii=False)


def _extract_doc_scores(payload: dict[str, Any]) -> list[float]:
    """Pull doc_scores from a /query/data response."""
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    if not isinstance(data, dict):
        return []
    scores = data.get("doc_scores") or []
    out: list[float] = []
    for s in scores:
        try:
            out.append(float(s))
        except (TypeError, ValueError):
            continue
    return out


def _extract_docs(payload: dict[str, Any]) -> list[str]:
    data = payload.get("data") if isinstance(payload.get("data"), dict) else payload
    if not isinstance(data, dict):
        return []
    docs = data.get("docs") or []
    return [str(d) for d in docs]


def retrieval_metrics(payload: dict[str, Any]) -> dict[str, float]:
    """Score a /query/data response for ranking.

    Combines mean / max doc scores with a soft reward for returning evidence.
    """
    scores = _extract_doc_scores(payload)
    docs = _extract_docs(payload)
    n = float(len(docs) or len(scores))
    mean_s = sum(scores) / len(scores) if scores else 0.0
    max_s = max(scores) if scores else 0.0
    # Soft saturation on doc count so empty retrieval is heavily penalized.
    coverage = min(1.0, n / 5.0)
    combined = 0.5 * mean_s + 0.3 * max_s + 0.2 * coverage
    return {
        "retrieval_score": combined,
        "mean_doc_score": mean_s,
        "max_doc_score": max_s,
        "n_docs": n,
    }


def _parse_judge_score(text: str) -> tuple[Optional[float], str]:
    """Extract a 0–10 score from the judge reply."""
    # Prefer SCORE: N.N pattern
    m = re.search(r"SCORE\s*[:=]\s*(\d+(?:\.\d+)?)", text, flags=re.IGNORECASE)
    if m:
        val = float(m.group(1))
        return max(0.0, min(10.0, val)), text.strip()
    # Fallback: first number in 0..10 range
    for m in re.finditer(r"\b(\d+(?:\.\d+)?)\b", text):
        val = float(m.group(1))
        if 0.0 <= val <= 10.0:
            return val, text.strip()
    return None, text.strip()


JUDGE_PROMPT = """You are a strict evaluator grading a RAG answer.

Question:
{question}

Answer to grade:
{answer}

Score the answer from 0 to 10 for relevance, groundedness, and completeness.
Reply in exactly this format:
SCORE: <number>
RATIONALE: <one short paragraph>
"""


def expand_grid(grid: dict[str, Sequence[Any]]) -> list[dict[str, Any]]:
    """Cartesian product of a param → values mapping."""
    if not grid:
        return [{}]
    keys = list(grid.keys())
    values = [list(grid[k]) for k in keys]
    combos: list[dict[str, Any]] = []
    for prod in itertools.product(*values):
        combos.append(dict(zip(keys, prod)))
    return combos


def run_optimize(
    client: MemGraphRAGClient,
    question: str,
    grid: Optional[dict[str, Sequence[Any]]] = None,
    *,
    questions: Optional[Sequence[str]] = None,
    top_n: int = 3,
    judge: bool = True,
    progress: ProgressCb = None,
) -> OptimizeReport:
    """Hybrid sweep: /query/data metrics over the grid, then LLM-judge top-N.

    Parameters
    ----------
    question:
        Primary evaluation question (also used for phase-2 answers).
    questions:
        Optional extra questions averaged into the phase-1 retrieval score.
    top_n:
        How many phase-1 winners advance to full ``/query`` + judge.
    judge:
        When False, skip phase 2 entirely (retrieval-only ranking).
    """
    eval_questions = list(questions or [])
    if question not in eval_questions:
        eval_questions.insert(0, question)

    sweep_grid = grid if grid is not None else default_sweep_grid()
    # Drop empty axes
    sweep_grid = {k: list(v) for k, v in sweep_grid.items() if v}
    combos = expand_grid(sweep_grid)
    total = len(combos)
    results: list[SweepResult] = []

    for i, combo in enumerate(combos, start=1):
        if progress:
            progress("phase1", i, total)
        params = clean_params(combo)
        metrics_acc = {
            "retrieval_score": 0.0,
            "mean_doc_score": 0.0,
            "max_doc_score": 0.0,
            "n_docs": 0.0,
        }
        errors: list[str] = []
        for q in eval_questions:
            try:
                payload = client.query_data(q, **params)
                m = retrieval_metrics(payload)
                for k in metrics_acc:
                    metrics_acc[k] += m[k]
            except Exception as exc:  # noqa: BLE001 — collect and continue sweep
                errors.append(f"{q!r}: {exc}")
        n_q = max(1, len(eval_questions))
        for k in metrics_acc:
            metrics_acc[k] /= n_q
        results.append(
            SweepResult(
                params=params,
                retrieval_score=metrics_acc["retrieval_score"],
                mean_doc_score=metrics_acc["mean_doc_score"],
                max_doc_score=metrics_acc["max_doc_score"],
                n_docs=metrics_acc["n_docs"],
                errors=errors,
            )
        )

    results.sort(key=lambda r: r.retrieval_score, reverse=True)
    phase2_count = 0

    if judge and results:
        winners = results[: max(1, top_n)]
        for j, row in enumerate(winners, start=1):
            if progress:
                progress("phase2", j, len(winners))
            try:
                ans_payload = client.query(question, **row.params)
                answer = ans_payload.get("answer") or ans_payload.get("response") or ""
                row.answer = str(answer)
                judge_query = JUDGE_PROMPT.format(question=question, answer=row.answer or "(empty)")
                judge_payload = client.query(judge_query, mode="bypass", only_need_context=False)
                judge_text = str(judge_payload.get("answer") or judge_payload.get("response") or "")
                score, rationale = _parse_judge_score(judge_text)
                row.judge_score = score
                row.judge_rationale = rationale
                phase2_count += 1
            except Exception as exc:  # noqa: BLE001
                row.errors.append(f"judge: {exc}")
        # Re-rank with final_score so judge winners float up
        results.sort(key=lambda r: r.final_score, reverse=True)

    recommended = results[0].params if results else {}
    return OptimizeReport(
        question=question,
        results=results,
        recommended=recommended,
        phase1_count=total,
        phase2_count=phase2_count,
    )
