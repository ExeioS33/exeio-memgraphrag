"""Offline unit tests for the hybrid client optimizer."""

from __future__ import annotations

import json

import httpx
import pytest

from memgraphrag.client.http import MemGraphRAGClient
from memgraphrag.client.optimize import (
    expand_grid,
    retrieval_metrics,
    run_optimize,
    _parse_judge_score,
)


@pytest.mark.offline
def test_expand_grid_cartesian() -> None:
    combos = expand_grid({"mode": ["ppr", "naive"], "top_k": [3, 5]})
    assert len(combos) == 4
    assert {"mode": "ppr", "top_k": 3} in combos
    assert expand_grid({}) == [{}]


@pytest.mark.offline
def test_retrieval_metrics_empty_and_scored() -> None:
    empty = retrieval_metrics({"data": {"docs": [], "doc_scores": []}})
    assert empty["retrieval_score"] == 0.0
    assert empty["n_docs"] == 0.0

    scored = retrieval_metrics(
        {"data": {"docs": ["a", "b", "c", "d", "e"], "doc_scores": [1.0, 0.5, 0.5, 0.5, 0.5]}}
    )
    assert scored["n_docs"] == 5.0
    assert scored["mean_doc_score"] == pytest.approx(0.6)
    assert scored["max_doc_score"] == 1.0
    assert scored["retrieval_score"] > 0.5


@pytest.mark.offline
def test_parse_judge_score() -> None:
    score, _ = _parse_judge_score("SCORE: 8.5\nRATIONALE: solid")
    assert score == 8.5
    score2, _ = _parse_judge_score("I give this a 7 overall.")
    assert score2 == 7.0
    score3, _ = _parse_judge_score("no numbers here")
    assert score3 is None


@pytest.mark.offline
def test_run_optimize_hybrid_ranks_with_judge() -> None:
    """Phase-1 scores /query/data; phase-2 judges top-N with mode=bypass."""

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8")) if request.content else {}
        path = request.url.path

        if path == "/query/data":
            # Prefer top_k=5 in retrieval metrics
            top_k = int(body.get("top_k") or 0)
            score = 0.9 if top_k == 5 else 0.2
            payload = {
                "data": {
                    "docs": ["evidence"] * max(1, top_k),
                    "doc_scores": [score] * max(1, top_k),
                }
            }
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=json.dumps(payload).encode(),
                request=request,
            )

        if path == "/query":
            mode = body.get("mode")
            if mode == "bypass":
                # Judge always returns a fixed high score
                payload = {"answer": "SCORE: 9\nRATIONALE: good"}
            else:
                payload = {
                    "answer": f"answer for top_k={body.get('top_k')}",
                    "docs": ["d"],
                    "doc_scores": [0.5],
                }
            return httpx.Response(
                200,
                headers={"content-type": "application/json"},
                content=json.dumps(payload).encode(),
                request=request,
            )

        return httpx.Response(404, json={"error": "unexpected"}, request=request)

    transport = httpx.MockTransport(handler)
    progress_calls: list[tuple[str, int, int]] = []

    with MemGraphRAGClient(base_url="http://test", transport=transport) as client:
        report = run_optimize(
            client,
            "What is memory?",
            grid={"top_k": [3, 5], "mode": ["ppr"]},
            top_n=1,
            judge=True,
            progress=lambda phase, i, total: progress_calls.append((phase, i, total)),
        )

    assert report.phase1_count == 2
    assert report.phase2_count == 1
    assert report.recommended["top_k"] == 5
    assert report.results[0].judge_score == 9.0
    assert report.results[0].answer is not None
    assert any(p[0] == "phase1" for p in progress_calls)
    assert any(p[0] == "phase2" for p in progress_calls)
    # Ensure judge request used bypass (captured via recommended winner path)
    assert "only_need_context" not in report.recommended  # not a sweep param


@pytest.mark.offline
def test_run_optimize_no_judge() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content.decode("utf-8"))
        assert request.url.path == "/query/data"
        top_k = int(body.get("top_k") or 0)
        payload = {
            "data": {"docs": ["x"] * top_k, "doc_scores": [0.8] * top_k},
        }
        return httpx.Response(
            200,
            headers={"content-type": "application/json"},
            content=json.dumps(payload).encode(),
            request=request,
        )

    transport = httpx.MockTransport(handler)
    with MemGraphRAGClient(base_url="http://test", transport=transport) as client:
        report = run_optimize(
            client,
            "q",
            grid={"top_k": [2, 4]},
            judge=False,
        )

    assert report.phase2_count == 0
    assert all(r.judge_score is None for r in report.results)
    assert report.recommended["top_k"] == 4
    as_json = json.loads(report.to_json())
    assert as_json["phase1_count"] == 2
