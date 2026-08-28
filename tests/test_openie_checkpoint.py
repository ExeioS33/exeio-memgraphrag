"""The OpenIE cache must survive a run that dies before the corpus is done.

Extraction is the billed part of ingestion: two LLM calls per chunk. The cache
used to be written once, after the last chunk, so a kill at 95 % re-billed 100 %
on relaunch. These tests pin that each sub-batch is durable as soon as it
completes, that a relaunch extracts only what is missing, and that a failed
sub-batch does not discard the successful chunks around it.
"""

from __future__ import annotations

import pytest

from memgraphrag.exceptions import PipelineError
from test_document_admin import _make_rag

pytestmark = pytest.mark.offline


def _fake_openie(calls: list[list[str]], *, fail_idx: set[str] = frozenset(), fail_times: int = 99):
    """``fail_idx`` chunks fail on their first ``fail_times`` attempts, then succeed."""
    attempts: dict[str, int] = {}

    async def fake_batch(docs):
        idxs = [str(d["idx"]) for d in docs]
        calls.append(idxs)
        out = []
        for d in docs:
            idx = str(d["idx"])
            attempts[idx] = attempts.get(idx, 0) + 1
            if idx in fail_idx and attempts[idx] <= fail_times:
                out.append({"idx": idx, "failed": True, "error": "boom"})
            else:
                out.append(
                    {
                        "idx": idx,
                        "passage": d["content"],
                        "extracted_entities": ["X"],
                        "extracted_triples": [["X", "rel", "Y"]],
                    }
                )
        return out

    return fake_batch


def _chunks(n: int) -> list[dict]:
    return [{"content": f"Passage number {i} about X."} for i in range(n)]


@pytest.mark.asyncio
async def test_each_sub_batch_is_persisted_before_the_next_starts(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENIE_CHECKPOINT_SIZE", "3")
    rag = _make_rag(tmp_path, monkeypatch=monkeypatch)
    await rag.initialize_storages()
    calls: list[list[str]] = []
    seen_cached: list[int] = []
    real = _fake_openie(calls)

    async def spying_batch(docs):
        # What is already durable when this sub-batch starts.
        seen_cached.append(len(await rag.openie_kv.get_all()))
        return await real(docs)

    rag.openie.batch_openie = spying_batch  # type: ignore[method-assign]
    prepared = rag._normalize_chunks(_chunks(7))
    await rag.ainsert(prepared, run_conflicts=False)

    assert [len(c) for c in calls] == [3, 3, 1]
    assert seen_cached == [0, 3, 6], "each sub-batch must be written before the next runs"
    assert len(await rag.openie_kv.get_all()) == 7
    await rag.finalize_storages()


@pytest.mark.asyncio
async def test_relaunch_after_a_kill_extracts_only_the_missing_chunks(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENIE_CHECKPOINT_SIZE", "2")
    rag = _make_rag(tmp_path, monkeypatch=monkeypatch)
    await rag.initialize_storages()
    prepared = rag._normalize_chunks(_chunks(5))

    # Simulate the kill: the third sub-batch never returns.
    calls: list[list[str]] = []
    real = _fake_openie(calls)

    async def dying_batch(docs):
        if len(calls) == 2:
            raise RuntimeError("process killed")
        return await real(docs)

    rag.openie.batch_openie = dying_batch  # type: ignore[method-assign]
    with pytest.raises(RuntimeError):
        await rag.ainsert(prepared, run_conflicts=False)
    assert len(await rag.openie_kv.get_all()) == 4, "the two completed sub-batches survive"
    await rag.finalize_storages()

    # Relaunch on the same working dir: only the 5th chunk is billed.
    rag2 = _make_rag(tmp_path, monkeypatch=monkeypatch)
    await rag2.initialize_storages()
    calls2: list[list[str]] = []
    rag2.openie.batch_openie = _fake_openie(calls2)  # type: ignore[method-assign]
    await rag2.ainsert(prepared, run_conflicts=False)
    assert calls2 == [[prepared[4]["idx"]]]
    assert len(await rag2.openie_kv.get_all()) == 5
    await rag2.finalize_storages()


@pytest.mark.asyncio
async def test_failed_chunk_does_not_discard_its_neighbours(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENIE_CHECKPOINT_SIZE", "2")
    rag = _make_rag(tmp_path, monkeypatch=monkeypatch)
    await rag.initialize_storages()
    prepared = rag._normalize_chunks(_chunks(4))
    calls: list[list[str]] = []
    rag.openie.batch_openie = _fake_openie(calls, fail_idx={prepared[1]["idx"]})  # type: ignore[method-assign]

    with pytest.raises(PipelineError, match="1/4 chunks after a retry"):
        await rag.ainsert(prepared, run_conflicts=False)

    cached = await rag.openie_kv.get_all()
    assert prepared[1]["idx"] not in cached, "a failure is never cached"
    assert {prepared[0]["idx"], prepared[2]["idx"], prepared[3]["idx"]} <= set(cached)
    assert [len(c) for c in calls] == [2, 2, 1], "corpus attempted to the end, then one retry"
    await rag.finalize_storages()


@pytest.mark.asyncio
async def test_transient_failure_is_retried_and_does_not_abort_the_run(tmp_path, monkeypatch):
    monkeypatch.setenv("OPENIE_CHECKPOINT_SIZE", "2")
    rag = _make_rag(tmp_path, monkeypatch=monkeypatch)
    await rag.initialize_storages()
    prepared = rag._normalize_chunks(_chunks(4))
    calls: list[list[str]] = []
    rag.openie.batch_openie = _fake_openie(  # type: ignore[method-assign]
        calls, fail_idx={prepared[1]["idx"]}, fail_times=1
    )

    await rag.ainsert(prepared, run_conflicts=False)

    assert calls[-1] == [prepared[1]["idx"]], "only the failed chunk is retried"
    assert len(await rag.openie_kv.get_all()) == 4
    assert rag.memory is not None and len(rag.memory.passage_layer) == 4
    await rag.finalize_storages()
