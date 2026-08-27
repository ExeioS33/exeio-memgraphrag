#!/usr/bin/env python
"""Ask the RFE question set against an ingested MemGraphRAG corpus.

Reads `data/rfe/questions.json`, whose ground truth was read straight from the
imported sidecars. Each question records what it proves: `tableau` questions have
answers that exist only inside a table, `image-ocr` ones only inside a scanned page.
They are the evidence that reusing LightRAG's Docling+VLM artefacts was worth it —
a text-only ingestion cannot answer them at all.

Usage:
    uv run python scripts/ask_rfe.py --workspace rfe_mgr
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import re
import sys
import time
import unicodedata
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

from memgraphrag.api.config import load_env_file, resolve_storage_backends  # noqa: E402

QUESTIONS = REPO / "data/rfe/questions.json"
_CITATION = re.compile(r"[\[【]\s*\d+\s*[\]】]")


def fold(text: str) -> str:
    """Accent- and case-insensitive comparison key.

    A correct answer written "Plateforme Agréée" must match an expectation written
    "plateforme agr", and models emit typographic dashes where the source has ASCII.
    """
    decomposed = unicodedata.normalize("NFKD", text.lower())
    stripped = "".join(c for c in decomposed if not unicodedata.combining(c))
    for dash in "‐‑‒–—−":
        stripped = stripped.replace(dash, "-")
    return " ".join(stripped.split())


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", default="rfe_mgr")
    parser.add_argument("--storage", type=Path, default=REPO / "data/rfe/storage")
    parser.add_argument("--top-k", type=int, default=0, help="0 = TOP_K from env")
    parser.add_argument("--only", default="", help="run a single question by id")
    parser.add_argument("--output", type=Path, default=REPO / "data/rfe/answers.json")
    args = parser.parse_args()

    load_env_file(str(REPO / ".env"))
    os.environ["WORKSPACE"] = args.workspace

    from memgraphrag.base import QueryParam
    from memgraphrag.core import MemGraphRAG
    from memgraphrag.llm.openai_compatible import openai_complete, openai_embed

    spec = json.loads(QUESTIONS.read_text(encoding="utf-8"))
    questions = spec["questions"]
    if args.only:
        questions = [q for q in questions if q["id"] == args.only]

    async def llm_model_func(prompt: str, **kwargs):
        return str(await openai_complete(prompt, model=os.getenv("LLM_MODEL"), **kwargs))

    async def embedding_func(texts, **kwargs):
        return await openai_embed(
            texts,
            model=os.getenv("EMBEDDING_MODEL"),
            embedding_dim=int(os.getenv("EMBEDDING_DIM") or 1024),
            **kwargs,
        )

    # MemGraphRAG's constructor defaults are literals (core.py:233-236); only
    # api/config.py reads MEMGRAPHRAG_*_STORAGE. A script that omits them silently
    # ignores the configured backends and always writes files.
    backends = resolve_storage_backends()
    rag = MemGraphRAG(
        working_dir=str(args.storage),
        workspace=args.workspace,
        llm_model_func=llm_model_func,
        embedding_func=embedding_func,
        embedding_dim=int(os.getenv("EMBEDDING_DIM") or 1024),
        max_async_llm=int(os.getenv("MAX_ASYNC_LLM") or 4),
        **backends,
    )
    await rag.initialize_storages()
    await rag.prepare_retrieval()

    print(
        f"corpus     : {len(rag.memory.passage_layer)} passages, "
        f"{len(rag.memory.fact_layer)} faits, {len(rag.memory.schema_layer)} schémas"
    )
    print(f"workspace  : {args.workspace}")
    print(f"LLM        : {os.getenv('LLM_MODEL')}\n")

    param = QueryParam(mode="ppr")
    if args.top_k:
        param.top_k = args.top_k

    results = []
    for item in questions:
        started = time.perf_counter()
        solution = await rag.aquery(item["q"], param=param)
        elapsed = time.perf_counter() - started
        if isinstance(solution, str):
            answer, refs, docs = solution.strip(), [], []
        else:
            answer = (solution.answer or "").strip()
            refs = solution.references or []
            docs = solution.docs or []

        folded = fold(answer)
        hits = [e for e in item["expect"] if fold(e) in folded]
        ok = len(hits) == len(item["expect"])
        results.append({**item, "answer": answer, "ok": ok, "hits": hits, "seconds": elapsed})

        print(f"[{item['kind']}] {item['q']}")
        print(
            f"  {elapsed:.1f}s · {len(docs)} passages · {len(refs)} sources · "
            f"citations {'oui' if _CITATION.search(answer) else 'non'}"
        )
        print(f"  {answer[:400]}")
        if item["expect"]:
            print(f"  attendu {item['expect']} -> trouvé {hits}  {'OK' if ok else 'MANQUE'}")
        if item.get("why"):
            print(f"  ({item['why']})")
        print()

    graded = [r for r in results if r["expect"]]
    passed = sum(1 for r in graded if r["ok"])
    by_kind: dict[str, list[bool]] = {}
    for r in graded:
        by_kind.setdefault(r["kind"], []).append(r["ok"])
    print(f"score : {passed}/{len(graded)}")
    for kind, oks in sorted(by_kind.items()):
        print(f"  {kind:16} {sum(oks)}/{len(oks)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"\nréponses : {args.output}")

    await rag.finalize_storages()
    return 0 if passed == len(graded) else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
