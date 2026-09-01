#!/usr/bin/env python
"""Performance harness: retrieval latency, throughput and LLM cost per query.

``docs/Reproduce.md`` promises RPS, p50/p95 and cost per query but ships nothing
that produces them, and the paper's headline "0.061 s per retrieval" has never
been measurable in this repository. This script measures all of it against a
real engine and a real corpus, and prints the paper's figure next to ours so the
comparison is explicit rather than implied.

Costs real LLM calls at index time (and at query time unless ``--retrieval-only``
is kept). Run it with the project's ``.env`` in place:

    uv run python scripts/bench.py --dataset hotpotqa --corpus-limit 200 --queries 50
    uv run python scripts/bench.py --dataset hotpotqa --no-ingest \
        --working-dir data/eval_storage/hotpotqa --queries 100 --concurrency 4
    uv run python scripts/bench.py --dataset hotpotqa --no-ingest --with-answer

What is measured, and what is not:

* **Retrieval latency** — wall time of ``MemGraphRAG.aretrieve`` per question,
  reported as p50 / p95 / p99 by nearest rank (never interpolated).
* **Throughput** — completed queries per wall second at a fixed concurrency.
* **LLM calls and tokens** — counted by wrapping the completion function, so
  every engine path is included. Tokens are *estimated locally* with tiktoken,
  not read from provider billing: the binding does not surface ``usage``.
* **TTFB is deliberately absent.** ``POST /query/stream`` awaits the whole answer
  before its first frame, so a TTFB number here would measure the transport, not
  the engine. Do not add one until streaming is real.

Exit codes: 0 = measured, 2 = could not start (dataset or config).
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
    LatencyStats,
    load_corpus,
    load_questions,
    run_load,
)
from memgraphrag.llm.openai_compatible import openai_complete, openai_embed  # noqa: E402

#: Retrieval latency claimed by arXiv:2606.00610. Printed for contrast only —
#: their hardware, corpus size and embedding endpoint are all unstated, so a
#: difference is not by itself evidence of anything.
PAPER_RETRIEVAL_SECONDS = 0.061


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Measure MemGraphRAG retrieval latency, throughput and LLM cost.",
    )
    parser.add_argument("--dataset", default="hotpotqa", help="dataset supplying the questions")
    parser.add_argument("--dataset-root", default=None, help="research checkout dataset/ directory")
    parser.add_argument("--queries", type=int, default=50, help="number of questions to send")
    parser.add_argument("--repeat", type=int, default=1, help="passes over the question set")
    parser.add_argument("--concurrency", type=int, default=1, help="queries in flight")
    parser.add_argument("--warmup", type=int, default=1, help="untimed queries before measuring")
    parser.add_argument("--mode", default="ppr", choices=["ppr", "naive"], help="retrieval mode")
    parser.add_argument("--top-k", type=int, default=0, help="override TOP_K (0 = project default)")
    parser.add_argument(
        "--with-answer",
        action="store_true",
        help="time the full QA path (retrieval + generation) instead of retrieval alone",
    )
    parser.add_argument("--working-dir", default="", help="storage dir to build or reuse")
    parser.add_argument(
        "--ingest", dest="ingest", action="store_true", default=True, help="index the corpus first"
    )
    parser.add_argument(
        "--no-ingest",
        dest="ingest",
        action="store_false",
        help="reuse an already indexed --working-dir",
    )
    parser.add_argument(
        "--corpus-limit", type=int, default=0, help="index only the first N documents (0 = all)"
    )
    parser.add_argument("--output", default="", help="write the JSON report here")
    return parser


async def _build_engine(args: argparse.Namespace, meter: CallMeter) -> tuple[MemGraphRAG, Path]:
    working_dir = (
        Path(args.working_dir)
        if args.working_dir
        else REPO / "data" / "eval_storage" / args.dataset
    )
    working_dir.mkdir(parents=True, exist_ok=True)

    async def llm_model_func(prompt: str, **kwargs: Any) -> str:
        # `model` and `provider` now arrive from the engine on any query path;
        # binding the default unconditionally raised "TypeError: got multiple
        # values for keyword argument 'model'". Scripts run on the server binding,
        # so a per-request provider is accepted and ignored.
        model = kwargs.pop("model", None) or os.getenv("LLM_MODEL")
        kwargs.pop("provider", None)
        return str(await openai_complete(prompt, model=model, **kwargs))

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


def _print_report(payload: dict[str, Any]) -> None:
    latency = payload["latency"]
    print(f"\n{'-' * 62}")
    print(f"queries        : {latency['count']} (concurrency {payload['concurrency']})")
    print(f"errors         : {payload['errors']}")
    print(
        f"p50 / p95 / p99: {latency['p50_s']:.4f} / "
        f"{latency['p95_s']:.4f} / {latency['p99_s']:.4f} s"
    )
    print(f"mean / max     : {latency['mean_s']:.4f} / {latency['max_s']:.4f} s")
    print(f"throughput     : {payload['throughput_rps']:.2f} queries/s")
    calls, per_call = payload["llm_calls"], payload["llm_calls_per_query"]
    tokens, per_query = payload["total_tokens"], payload["tokens_per_query"]
    print(f"LLM calls      : {calls} total, {per_call} per query")
    print(f"tokens (est.)  : {tokens} total, {per_query} per query")
    print(f"paper reference: {PAPER_RETRIEVAL_SECONDS:.3f} s per retrieval (arXiv:2606.00610)")
    if not payload["with_answer"]:
        ratio = latency["p50_s"] / PAPER_RETRIEVAL_SECONDS if PAPER_RETRIEVAL_SECONDS else 0.0
        print(f"                 our p50 is {ratio:.1f}x that figure on this deployment")
    print(f"{'-' * 62}")


async def main() -> int:
    args = build_parser().parse_args()

    try:
        examples = load_questions(args.dataset, root=args.dataset_root, limit=args.queries or None)
    except (DatasetUnavailableError, DatasetFormatError) as exc:
        print(f"cannot benchmark: {exc}", file=sys.stderr)
        return 2

    questions = [example.question for example in examples] * max(1, args.repeat)
    meter = CallMeter(model=os.getenv("LLM_MODEL") or "gpt-4o-mini")
    rag, working_dir = await _build_engine(args, meter)
    print(f"dataset    : {args.dataset} ({len(questions)} queries)")
    print(f"working dir: {working_dir}")
    print(f"LLM        : {os.getenv('LLM_MODEL')} @ {os.getenv('LLM_BINDING_HOST')}")

    index_seconds = 0.0
    if args.ingest:
        corpus = load_corpus(args.dataset, root=args.dataset_root, limit=args.corpus_limit or None)
        print(f"indexing   : {len(corpus)} corpus documents")
        started = time.perf_counter()
        await rag.ainsert([doc.to_chunk() for doc in corpus])
        index_seconds = time.perf_counter() - started
        print(f"indexed in : {index_seconds:.1f}s")

    # Prepare retrieval before the timed window: the first query otherwise pays
    # for loading storage and building the PPR graph, which is start-up cost, not
    # retrieval cost, and it lands entirely in p95.
    await rag.prepare_retrieval()

    param = QueryParam(mode=args.mode)
    if args.top_k:
        param.top_k = args.top_k

    # Index-time LLM traffic must not be charged to the queries.
    query_meter_start = meter.snapshot()

    async def call(question: str) -> Any:
        if args.with_answer:
            return await rag.aquery(question, param=param)
        return await rag.aretrieve(question, param=param)

    result = await run_load(call, questions, concurrency=args.concurrency, warmup=args.warmup)
    await rag.finalize_storages()

    latency = LatencyStats.from_samples(result.latencies)
    query_calls = meter.calls - int(query_meter_start["llm_calls"])
    query_tokens = (meter.prompt_tokens + meter.completion_tokens) - int(
        query_meter_start["total_tokens"]
    )
    timed = max(1, len(result.latencies))
    payload = {
        "dataset": args.dataset,
        "mode": args.mode,
        "top_k": param.top_k,
        "with_answer": bool(args.with_answer),
        "concurrency": args.concurrency,
        "warmup": args.warmup,
        "working_dir": str(working_dir),
        "llm_model": os.getenv("LLM_MODEL") or "",
        "embedding_model": os.getenv("EMBEDDING_MODEL") or "",
        "index_seconds": round(index_seconds, 3),
        "index_llm_calls": int(query_meter_start["llm_calls"]),
        "latency": latency.to_dict(),
        "wall_seconds": round(result.wall_seconds, 4),
        "throughput_rps": round(result.throughput, 4),
        "errors": result.errors,
        "llm_calls": query_calls,
        "llm_calls_per_query": round(query_calls / timed, 4),
        "total_tokens": query_tokens,
        "tokens_per_query": round(query_tokens / timed, 2),
        "paper_retrieval_seconds": PAPER_RETRIEVAL_SECONDS,
    }
    _print_report(payload)

    if args.output:
        target = Path(args.output)
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
        print(f"report: {target}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
