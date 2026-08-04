"""Query routes for MemGraphRAG.

Adapted from LightRAG ``lightrag/api/routers/query_routes.py`` (slimmed).
Modes: ``ppr`` | ``naive`` | ``context`` | ``bypass``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Literal, Optional

from memgraphrag.base import QueryParam
from memgraphrag.utils.misc import QuerySolution

logger = logging.getLogger("memgraphrag.api.query")

try:
    from fastapi import APIRouter, Depends, Request
    from fastapi.responses import StreamingResponse
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[misc, assignment]
    Request = None  # type: ignore[misc, assignment]
    StreamingResponse = None  # type: ignore[misc, assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    Field = lambda *a, **k: None  # type: ignore[misc, assignment]


class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, description="User query")
    mode: Literal["ppr", "naive", "context", "bypass"] = Field(default="ppr")
    top_k: Optional[int] = Field(default=None, ge=1)
    linking_top_k: Optional[int] = Field(default=None, ge=1)
    passage_node_weight: Optional[float] = None
    damping: Optional[float] = None
    fact_similarity_threshold: Optional[float] = None
    skip_fact_rerank: Optional[bool] = None
    only_need_context: Optional[bool] = False
    conversation_history: Optional[list[dict[str, str]]] = None
    user_prompt: Optional[str] = None
    stream: Optional[bool] = False


def _build_param(body: QueryRequest, rag: Any) -> QueryParam:
    kwargs: dict[str, Any] = {"mode": body.mode}
    if body.top_k is not None:
        kwargs["top_k"] = body.top_k
    elif getattr(rag, "top_k", None) is not None:
        kwargs["top_k"] = rag.top_k
    if body.linking_top_k is not None:
        kwargs["linking_top_k"] = body.linking_top_k
    if body.passage_node_weight is not None:
        kwargs["passage_node_weight"] = body.passage_node_weight
    if body.damping is not None:
        kwargs["damping"] = body.damping
    if body.fact_similarity_threshold is not None:
        kwargs["fact_similarity_threshold"] = body.fact_similarity_threshold
    if body.skip_fact_rerank is not None:
        kwargs["skip_fact_rerank"] = body.skip_fact_rerank
    if body.only_need_context is not None:
        kwargs["only_need_context"] = body.only_need_context
    if body.conversation_history is not None:
        kwargs["conversation_history"] = body.conversation_history
    if body.user_prompt is not None:
        kwargs["user_prompt"] = body.user_prompt
    if body.stream:
        kwargs["stream"] = True
    return QueryParam(**kwargs)


def _solution_payload(sol: QuerySolution | str) -> dict[str, Any]:
    if isinstance(sol, str):
        return {"response": sol, "answer": sol}
    data = sol.to_dict()
    data["response"] = sol.answer
    data["answer"] = sol.answer
    return data


def create_query_router(api_key: Optional[str] = None) -> Any:
    if APIRouter is None:
        raise RuntimeError("fastapi is required; install memgraphrag[api]")

    from memgraphrag.api.dependencies import get_combined_auth_dependency

    router = APIRouter(tags=["query"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post("/query", dependencies=[Depends(combined_auth)])
    async def query(request: Request, body: QueryRequest):
        rag = request.app.state.rag
        param = _build_param(body, rag)
        if body.only_need_context or body.mode == "context":
            param.only_need_context = True
        sol = await rag.arag_qa(body.query, param=param)
        return _solution_payload(sol)

    @router.post("/query/data", dependencies=[Depends(combined_auth)])
    async def query_data(request: Request, body: QueryRequest):
        """Retrieval-only structured response."""
        rag = request.app.state.rag
        param = _build_param(body, rag)
        param.only_need_context = True
        if param.mode == "bypass":
            param.mode = "ppr"
        sols = await rag.aretrieve(body.query, param=param)
        sol = sols[0] if sols else QuerySolution(question=body.query, docs=[])
        return {
            "status": "success",
            "data": _solution_payload(sol),
            "metadata": {"mode": param.mode},
        }

    @router.post("/query/stream", dependencies=[Depends(combined_auth)])
    async def query_stream(request: Request, body: QueryRequest):
        rag = request.app.state.rag
        param = _build_param(body, rag)
        param.stream = True

        async def event_gen() -> AsyncIterator[str]:
            try:
                sol = await rag.arag_qa(body.query, param=param)
                payload = _solution_payload(sol)
                answer = payload.get("answer") or payload.get("response") or ""
                # Yield full answer as one SSE chunk (no token streaming yet)
                yield f"data: {json.dumps({'response': answer}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                logger.exception("query/stream failed: %s", exc)
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    return router
