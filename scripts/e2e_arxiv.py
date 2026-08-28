#!/usr/bin/env python
"""End-to-end check against a real arXiv PDF, using the real LLM bindings.

Runs the whole pipeline — parse, chunk, OpenIE, memory, ontology, conflicts, graph
install, PPR retrieval, answer generation — on one paper, then asks questions whose
answers are known from the source, and reports what the engine returned.

This is a manual verification tool, not a unit test: it costs real LLM calls. Run it
with the project's .env in place:

    uv run python scripts/e2e_arxiv.py [path/to/paper.pdf]
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import shutil
import sys
import tempfile
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memgraphrag.api.config import load_env_file  # noqa: E402

load_env_file(str(REPO / ".env"))

from memgraphrag.base import DocStatus, QueryParam  # noqa: E402
from memgraphrag.core import MemGraphRAG  # noqa: E402
from memgraphrag.llm.openai_compatible import openai_complete, openai_embed  # noqa: E402
from memgraphrag.pipeline import enqueue_document, process_pending  # noqa: E402
from memgraphrag.utils.hashing import compute_mdhash_id  # noqa: E402

DEFAULT_PDF = REPO / "data/inputs/2605.18490v1.pdf"

# Ground truth read straight from the PDF, so an answer can be judged rather than
# admired. `expect` lists strings that a correct answer should contain (case-folded).
QUESTIONS: list[dict] = [
    {
        "kind": "single-fact",
        "q": "How many questions and how many papers were used in the comparison?",
        "expect": ["13", "24"],
    },
    {
        "kind": "single-fact",
        "q": "Who wrote this paper and what organization are they from?",
        "expect": ["cochran"],
    },
    {
        "kind": "multi-hop",
        "q": "Which judge model gave near-perfect scores on holistic criteria, and which judge used more of the scale?",
        "expect": ["gemini", "gpt-5.4"],
    },
    {
        "kind": "synthesis",
        "q": "Did the wiki recover its upfront build cost through cheaper queries? Explain why or why not.",
        "expect": ["not", "token"],
    },
    {
        "kind": "single-fact",
        "q": "What chunk size and overlap did the RAG system use?",
        "expect": ["512"],
    },
    {
        "kind": "negative-control",
        "q": "What is the melting point of tungsten according to this paper?",
        "expect": [],  # the paper says nothing about this; a good answer admits it
    },
]


# Models emit typographic dashes (U+2010/2011/2012/2013) and CJK brackets, so a raw
# substring test scores a correct answer as a miss.
_DASHES = dict.fromkeys(map(ord, "\u2010\u2011\u2012\u2013\u2014\u2212"), "-")


def normalize(text: str) -> str:
    return " ".join(text.translate(_DASHES).lower().split())


_CITATION = re.compile(r"[\[\u3010]\s*\d+\s*[\]\u3011]")


def banner(text: str) -> None:
    print(f"\n{'=' * 78}\n{text}\n{'=' * 78}")


async def main() -> int:
    pdf = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_PDF
    if not pdf.exists():
        print(f"PDF not found: {pdf}")
        return 2

    reuse = os.getenv("MGR_E2E_WORKDIR")
    workdir = Path(reuse) if reuse else Path(tempfile.mkdtemp(prefix="mgr-e2e-"))
    inputs = workdir / "inputs"
    if not reuse:
        inputs.mkdir(parents=True)
        shutil.copy(pdf, inputs / pdf.name)

    banner(f"MemGraphRAG end-to-end — {pdf.name}")
    print(f"LLM        : {os.getenv('LLM_MODEL')} @ {os.getenv('LLM_BINDING_HOST')}")
    print(f"Embedding  : {os.getenv('EMBEDDING_MODEL')} (dim {os.getenv('EMBEDDING_DIM')})")
    print(f"Working dir: {workdir}")

    async def llm_model_func(prompt: str, **kwargs):
        return str(await openai_complete(prompt, model=os.getenv("LLM_MODEL"), **kwargs))

    async def embedding_func(texts, **kwargs):
        return await openai_embed(
            texts,
            model=os.getenv("EMBEDDING_MODEL"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM") or 1024),
            **kwargs,
        )

    rag = MemGraphRAG(
        working_dir=str(workdir / "rag_storage"),
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        embedding_dim=int(os.getenv("EMBEDDING_DIM") or 1024),
        max_async_llm=int(os.getenv("MAX_ASYNC_LLM") or 4),
    )
    await rag.initialize_storages()

    banner("1. Ingestion")
    t0 = time.perf_counter()
    if reuse:
        print("reusing existing corpus (MGR_E2E_WORKDIR set); skipping ingest")
    if reuse:
        summary = {"processed": 0, "failed": 0, "doc_ids": ["<reused>"]}
    else:
        doc_id = compute_mdhash_id(pdf.read_bytes().hex()[:4096], prefix="doc-")
        await enqueue_document(
            doc_id=doc_id,
            file_path=str(inputs / pdf.name),
            doc_status_storage=rag.doc_status,
        )
        summary = await process_pending(rag, rag.doc_status, input_dir=inputs)
    ingest_s = time.perf_counter() - t0
    print(f"pipeline summary : {summary}")
    print(f"ingest wall time : {ingest_s:.1f}s")

    statuses = await rag.doc_status.get_all()
    record = next(iter(statuses.values()), {})
    status = str(record.get("status") or "")
    print(f"document status  : {status}")
    if status != DocStatus.PROCESSED.value:
        print(f"FAILED: document did not reach PROCESSED ({record.get('error')})")
        return 1

    banner("2. Memory and graph shape")
    await rag.prepare_retrieval()
    mem = rag.memory
    print(f"passages : {len(mem.passage_layer)}")
    print(f"facts    : {len(mem.fact_layer)}")
    print(f"schemas  : {len(mem.schema_layer)}")

    edges = await rag.graph.get_all_edges()
    nodes = await rag.graph.get_all_nodes()
    by_type: dict[str, int] = {}
    for e in edges:
        by_type[str(e.get("type"))] = by_type.get(str(e.get("type")), 0) + 1
    print(f"graph    : {len(nodes)} nodes, {len(edges)} edges")
    for k in sorted(by_type):
        print(f"  {k:16} {by_type[k]}")

    checks: list[tuple[str, bool, str]] = []
    checks.append(
        (
            "G_fac present (ENTITY_RELATION edges)",
            by_type.get("ENTITY_RELATION", 0) > 0,
            f"{by_type.get('ENTITY_RELATION', 0)} edges",
        )
    )
    checks.append(
        (
            "ontology layer wired (ENTITY_TO_TYPE edges)",
            by_type.get("ENTITY_TO_TYPE", 0) > 0,
            f"{by_type.get('ENTITY_TO_TYPE', 0)} edges",
        )
    )
    checks.append(("passages indexed", len(mem.passage_layer) > 0, str(len(mem.passage_layer))))
    checks.append(("facts extracted", len(mem.fact_layer) > 0, str(len(mem.fact_layer))))

    banner("3. Questions")
    # Project default TOP_K; do not hand-tune it for the demo.
    param = QueryParam(mode="ppr")
    results = []
    for item in QUESTIONS:
        t = time.perf_counter()
        sol = await rag.aquery(item["q"], param=param)
        dt = time.perf_counter() - t
        if isinstance(sol, str):
            answer, refs, docs = sol.strip(), [], []
        else:
            answer = (sol.answer or "").strip()
            refs = sol.references or []
            docs = sol.docs or []
        norm = normalize(answer)
        hits = [e for e in item["expect"] if normalize(e) in norm]
        ok = len(hits) == len(item["expect"])
        results.append({**item, "answer": answer, "ok": ok, "hits": hits, "s": dt})

        print(f"\n[{item['kind']}] {item['q']}")
        print(f"  latency   : {dt:.2f}s   passages: {len(docs)}   refs: {len(refs)}")
        print(f"  answer    : {answer[:600]}")
        if item["expect"]:
            print(f"  expected  : {item['expect']}  ->  found {hits}  {'OK' if ok else 'MISS'}")
        cited = bool(_CITATION.search(answer))
        print(f"  citations : {'present' if cited else 'absent'}")

    banner("4. Verdict")
    graded = [r for r in results if r["expect"]]
    passed = sum(1 for r in graded if r["ok"])
    for name, ok, detail in checks:
        print(f"  [{'OK ' if ok else 'FAIL'}] {name}: {detail}")
    print(
        f"  [{'OK ' if passed == len(graded) else 'PART'}] answers grounded: {passed}/{len(graded)}"
    )

    (workdir / "e2e_report.json").write_text(
        json.dumps(
            {
                "pdf": pdf.name,
                "ingest_seconds": round(ingest_s, 2),
                "passages": len(mem.passage_layer),
                "facts": len(mem.fact_layer),
                "schemas": len(mem.schema_layer),
                "edges_by_type": by_type,
                "results": [{k: v for k, v in r.items() if k != "expect"} for r in results],
            },
            indent=2,
            ensure_ascii=False,
        )
    )
    print(f"\nreport: {workdir / 'e2e_report.json'}")

    await rag.finalize_storages()
    structural_ok = all(ok for _, ok, _ in checks)
    return 0 if (structural_ok and passed == len(graded)) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
