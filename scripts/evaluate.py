#!/usr/bin/env python
"""Quality evaluation harness: run a benchmark dataset through the engine.

Scores Str-Acc, LLM-Acc, Context Relevance and Evidence Recall (the paper's four
columns, defined operationally in ``docs/Evaluation.md``) plus the EM / F1 /
Recall@k floor, over N repeated runs so that mean **and** standard deviation are
reported. A single number is not a result: the paper publishes none of its run
counts or spreads, and this harness exists so that ours are always on the record.

This costs real LLM calls. Run it with the project's ``.env`` in place:

    uv run python scripts/evaluate.py --list-datasets
    uv run python scripts/evaluate.py --dataset hotpotqa --limit 50 --runs 3 \
        --output data/eval/hotpotqa.json
    uv run python scripts/evaluate.py --dataset hotpotqa --limit 50 --runs 5 \
        --write-golden data/eval/hotpotqa.golden.json
    uv run python scripts/evaluate.py --dataset hotpotqa --limit 50 --runs 3 \
        --compare data/eval/hotpotqa.golden.json --tolerance 2

Exit codes: 0 = done (and no regression when ``--compare`` is used), 1 = a metric
regressed beyond tolerance, 2 = the run could not start (dataset or config).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
import time
from pathlib import Path
from typing import Any

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memgraphrag.api.config import load_env_file  # noqa: E402

load_env_file(str(REPO / ".env"))

from memgraphrag.base import QueryParam  # noqa: E402
from memgraphrag.core import MemGraphRAG  # noqa: E402
from memgraphrag.evaluation import (  # noqa: E402
    CallMeter,
    DatasetFormatError,
    DatasetUnavailableError,
    EvaluationExample,
    GoldenSetError,
    LLMAccJudge,
    Prediction,
    available_datasets,
    build_golden_set,
    compare_to_golden,
    dataset_root,
    load_corpus,
    load_golden_set,
    load_questions,
    run_campaign,
    sample_examples,
    write_golden_set,
)
from memgraphrag.llm.openai_compatible import openai_complete, openai_embed  # noqa: E402
from memgraphrag.utils.misc import QuerySolution  # noqa: E402


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--dataset", default="hotpotqa", help="hotpotqa | 2wikimultihopqa | musique | medical"
    )
    parser.add_argument("--dataset-root", default=None, help="research checkout dataset/ directory")
    parser.add_argument(
        "--list-datasets", action="store_true", help="list datasets found under the root and exit"
    )
    parser.add_argument("--limit", type=int, default=50, help="first N questions (0 = all)")
    parser.add_argument(
        "--sample", type=int, default=0, help="random subset of N questions instead of the first N"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="seed for --sample (recorded in the report)"
    )
    parser.add_argument(
        "--question-types", default="", help="comma-separated type filter (grouped datasets)"
    )
    parser.add_argument(
        "--runs", type=int, default=1, help="number of identical runs, for variance"
    )
    parser.add_argument("--concurrency", type=int, default=1, help="queries in flight per run")
    parser.add_argument(
        "--mode", default="ppr", choices=["ppr", "naive", "context"], help="retrieval mode"
    )
    parser.add_argument("--top-k", type=int, default=0, help="override TOP_K (0 = project default)")
    parser.add_argument(
        "--working-dir", default="", help="storage dir to build or reuse (default: temporary)"
    )
    parser.add_argument(
        "--ingest",
        dest="ingest",
        action="store_true",
        default=True,
        help="index the corpus first (default)",
    )
    parser.add_argument(
        "--no-ingest",
        dest="ingest",
        action="store_false",
        help="reuse an already indexed --working-dir",
    )
    parser.add_argument(
        "--corpus-limit",
        type=int,
        default=0,
        help="index only the first N corpus documents (0 = all)",
    )
    parser.add_argument(
        "--judge", action="store_true", help="score LLM-Acc with the versioned judge prompt"
    )
    parser.add_argument(
        "--judge-model",
        default="",
        help="judge model (default: LLM_MODEL); use a different model from the one under test",
    )
    parser.add_argument("--output", default="", help="write the full JSON report here")
    parser.add_argument("--write-golden", default="", help="write a golden set from this campaign")
    parser.add_argument("--compare", default="", help="compare this campaign to a golden set")
    parser.add_argument(
        "--tolerance", type=float, default=2.0, help="allowed drop, in standard deviations"
    )
    return parser


def _passage_titles(solution: QuerySolution) -> list[str]:
    """Document identity per retrieved passage, in rank order.

    Passages indexed by this script are ``"<title>\\n<text>"`` (see
    ``CorpusDocument.to_chunk``), because ``ainsert`` takes bare chunks and never
    records a ``file_path`` for them — so ``QuerySolution.sources`` is empty on
    this path and the title has to be read back off the passage itself. The
    source label is still preferred when the engine did supply one.
    """
    titles: list[str] = []
    sources = list(solution.sources or [])
    for index, doc in enumerate(solution.docs or []):
        label = str(sources[index]).strip() if index < len(sources) else ""
        if label and label.lower() != "unknown":
            titles.append(label)
            continue
        titles.append(str(doc or "").split("\n", 1)[0].strip())
    return titles


async def _build_engine(args: argparse.Namespace, meter: CallMeter) -> tuple[MemGraphRAG, Path]:
    working_dir = (
        Path(args.working_dir)
        if args.working_dir
        else REPO / "data" / "eval_storage" / args.dataset
    )
    working_dir.mkdir(parents=True, exist_ok=True)

    async def llm_model_func(prompt: str, **kwargs: Any) -> str:
        return str(await openai_complete(prompt, model=os.getenv("LLM_MODEL"), **kwargs))

    async def embedding_func(texts: Any, **kwargs: Any) -> Any:
        return await openai_embed(
            texts,
            model=os.getenv("EMBEDDING_MODEL"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM") or 1024),
            **kwargs,
        )

    rag = MemGraphRAG(
        working_dir=str(working_dir),
        llm_model_func=meter.wrap(llm_model_func),
        embedding_func=embedding_func,
        embedding_dim=int(os.getenv("EMBEDDING_DIM") or 1024),
        max_async_llm=int(os.getenv("MAX_ASYNC_LLM") or 4),
    )
    await rag.initialize_storages()
    return rag, working_dir


def _print_metrics(report: Any) -> None:
    print(f"\n{'metric':<26} {'mean':>9} {'stdev':>9} {'min':>9} {'max':>9}  runs")
    print("-" * 74)
    for name in sorted(report.metrics):
        stat = report.metrics[name]
        print(
            f"{name:<26} {stat.mean:>9.4f} {stat.stdev:>9.4f} "
            f"{stat.minimum:>9.4f} {stat.maximum:>9.4f}  {stat.runs}"
        )
    if report.runs and len(report.runs) == 1:
        print("\nNOTE: one run measures no variance. Use --runs 3 or more before")
        print("      calling any difference a regression or an improvement.")


async def main() -> int:
    args = build_parser().parse_args()

    if args.list_datasets:
        root = dataset_root(args.dataset_root)
        found = available_datasets(args.dataset_root)
        print(f"dataset root: {root}")
        print(
            "available   : "
            + (", ".join(found) if found else "(none — is the research checkout present?)")
        )
        return 0 if found else 2

    try:
        examples: list[EvaluationExample] = load_questions(
            args.dataset,
            root=args.dataset_root,
            limit=None if args.sample else (args.limit or None),
            question_types=[t for t in args.question_types.split(",") if t] or None,
        )
    except (DatasetUnavailableError, DatasetFormatError) as exc:
        print(f"cannot evaluate: {exc}", file=sys.stderr)
        return 2
    if args.sample:
        examples = sample_examples(examples, args.sample, seed=args.seed)

    meter = CallMeter(model=os.getenv("LLM_MODEL") or "gpt-4o-mini")
    rag, working_dir = await _build_engine(args, meter)
    print(f"dataset    : {args.dataset} ({len(examples)} questions)")
    print(f"working dir: {working_dir}")
    print(f"LLM        : {os.getenv('LLM_MODEL')} @ {os.getenv('LLM_BINDING_HOST')}")

    if args.ingest:
        corpus = load_corpus(args.dataset, root=args.dataset_root, limit=args.corpus_limit or None)
        print(f"indexing   : {len(corpus)} corpus documents (this is the expensive part)")
        started = time.perf_counter()
        await rag.ainsert([doc.to_chunk() for doc in corpus])
        print(f"indexed in : {time.perf_counter() - started:.1f}s")
    await rag.prepare_retrieval()

    param = QueryParam(mode=args.mode)
    if args.top_k:
        param.top_k = args.top_k

    async def answer_fn(example: EvaluationExample) -> Prediction:
        solution = await rag.aquery(example.question, param=param)
        if isinstance(solution, str):
            return Prediction(example_id=example.id, question=example.question, answer=solution)
        return Prediction(
            example_id=example.id,
            question=example.question,
            answer=str(solution.answer or ""),
            retrieved_docs=_passage_titles(solution),
        )

    judge = None
    if args.judge:
        judge = LLMAccJudge(
            complete=openai_complete,
            model=args.judge_model or os.getenv("LLM_MODEL"),
            max_concurrency=max(1, args.concurrency),
        )

    report = await run_campaign(
        examples,
        answer_fn,
        runs=args.runs,
        judge=judge,
        concurrency=args.concurrency,
        dataset=args.dataset,
        metadata={
            "mode": args.mode,
            "top_k": param.top_k,
            "llm_model": os.getenv("LLM_MODEL") or "",
            "embedding_model": os.getenv("EMBEDDING_MODEL") or "",
            "sample_seed": args.seed if args.sample else None,
            "sample_size": args.sample or None,
            "working_dir": str(working_dir),
            **meter.snapshot(),
            **meter.per_query(len(examples) * max(1, args.runs)),
        },
    )
    await rag.finalize_storages()

    _print_metrics(report)

    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(
            json.dumps(report.to_dict(), indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
        )
        print(f"\nreport: {target}")

    if args.write_golden:
        golden = build_golden_set(
            name=Path(args.write_golden).stem, examples=examples, report=report
        )
        print(f"golden: {write_golden_set(args.write_golden, golden)}")

    exit_code = 0
    if args.compare:
        try:
            reference = load_golden_set(args.compare)
        except GoldenSetError as exc:
            print(f"cannot compare: {exc}", file=sys.stderr)
            return 2
        comparison = compare_to_golden(reference, report, tolerance_sigma=args.tolerance)
        print(f"\ncompared to {reference.name} (created {reference.created_at})")
        for row in comparison.comparisons:
            flag = "REGRESSION" if row.regressed else ("improved" if row.improved else "ok")
            print(
                f"  {row.metric:<26} {row.reference_mean:>8.4f} -> {row.current_mean:>8.4f} "
                f"({row.delta:+.4f}, {row.z_score:+.2f}σ, σ={row.sigma:.4f}) {flag}"
            )
        if comparison.missing:
            print(f"  metrics missing from one side: {', '.join(comparison.missing)}")
        exit_code = 0 if comparison.passed else 1

    return exit_code


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
