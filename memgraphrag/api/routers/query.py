"""Query routes for MemGraphRAG.

Adapted from LightRAG ``lightrag/api/routers/query_routes.py`` (slimmed).
Modes: ``ppr`` | ``naive`` | ``context`` | ``bypass``.
"""

from __future__ import annotations

import json
import logging
from typing import Any, AsyncIterator, Literal, Optional

from memgraphrag.base import QueryParam
from memgraphrag.observability.langfuse_trace import flush_langfuse
from memgraphrag.utils.misc import QuerySolution
from memgraphrag.utils.step_log import done_step, fail_step, main_step, truncate

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
    schema_top_k: Optional[int] = Field(default=None, ge=0)
    schema_node_weight: Optional[float] = None
    only_need_context: Optional[bool] = False
    conversation_history: Optional[list[dict[str, str]]] = None
    user_prompt: Optional[str] = None
    stream: Optional[bool] = False


class QueryResponse(BaseModel):
    """LightRAG-compatible ``POST /query`` response shape."""

    response: Optional[str] = Field(default=None, description="Generated answer text")
    references: list[dict[str, Any]] = Field(
        default_factory=list,
        description="Unique source documents supporting the answer",
    )


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
    if body.schema_top_k is not None:
        kwargs["schema_top_k"] = body.schema_top_k
    if body.schema_node_weight is not None:
        kwargs["schema_node_weight"] = body.schema_node_weight
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
    """LightRAG-style envelope: ``response`` + ``references``."""
    if isinstance(sol, str):
        return {"response": sol, "references": []}
    if hasattr(sol, "ensure_references"):
        sol.ensure_references()
    return {
        "response": sol.answer,
        "references": list(getattr(sol, "references", None) or []),
    }


def _query_data_payload(sol: QuerySolution | str) -> dict[str, Any]:
    """Richer retrieval payload for ``/query/data`` (includes docs)."""
    base = _solution_payload(sol)
    if isinstance(sol, str):
        return base
    data = sol.to_dict()
    base["docs"] = data.get("docs") or []
    base["doc_scores"] = data.get("doc_scores")
    base["question"] = data.get("question")
    return base


def create_query_router(api_key: Optional[str] = None) -> Any:
    if APIRouter is None:
        raise RuntimeError("fastapi is required; install memgraphrag[api]")

    from memgraphrag.api.dependencies import get_combined_auth_dependency

    router = APIRouter(tags=["query"])
    combined_auth = get_combined_auth_dependency(api_key)

    @router.post(
        "/query",
        dependencies=[Depends(combined_auth)],
        response_model=QueryResponse,
    )
    async def query(request: Request, body: QueryRequest):
        rag = request.app.state.rag
        param = _build_param(body, rag)
        if body.only_need_context or body.mode == "context":
            param.only_need_context = True
        main_step(
            logger,
            "api.query",
            mode=param.mode,
            query=truncate(body.query),
            stream=False,
            only_need_context=bool(param.only_need_context),
        )
        try:
            sol = await rag.arag_qa(body.query, param=param)
            payload = _solution_payload(sol)
            answer = payload.get("response") or ""
            docs = getattr(sol, "docs", None) or []
            done_step(
                logger,
                "api.query",
                mode=param.mode,
                docs=len(docs),
                answer_chars=len(str(answer)),
                references=len(payload.get("references") or []),
            )
            return payload
        except Exception as exc:
            fail_step(
                logger,
                "api.query",
                mode=param.mode,
                exc=exc,
                exc_info=True,
            )
            raise
        finally:
            flush_langfuse()

    @router.post("/query/data", dependencies=[Depends(combined_auth)])
    async def query_data(request: Request, body: QueryRequest):
        """Retrieval-only structured response."""
        rag = request.app.state.rag
        param = _build_param(body, rag)
        param.only_need_context = True
        if param.mode == "bypass":
            param.mode = "ppr"
        main_step(
            logger,
            "api.query.data",
            mode=param.mode,
            query=truncate(body.query),
        )
        try:
            sols = await rag.aretrieve(body.query, param=param)
            sol = sols[0] if sols else QuerySolution(question=body.query, docs=[])
            sol.ensure_references()
            done_step(
                logger,
                "api.query.data",
                mode=param.mode,
                docs=len(sol.docs),
            )
            return {
                "status": "success",
                "data": _query_data_payload(sol),
                "metadata": {"mode": param.mode},
            }
        except Exception as exc:
            fail_step(
                logger,
                "api.query.data",
                mode=param.mode,
                exc=exc,
                exc_info=True,
            )
            raise

    @router.post("/query/stream", dependencies=[Depends(combined_auth)])
    async def query_stream(request: Request, body: QueryRequest):
        rag = request.app.state.rag
        param = _build_param(body, rag)
        param.stream = True
        main_step(
            logger,
            "api.query.stream",
            mode=param.mode,
            query=truncate(body.query),
        )

        async def event_gen() -> AsyncIterator[str]:
            try:
                sol = await rag.arag_qa(body.query, param=param)
                payload = _solution_payload(sol)
                answer = payload.get("response") or ""
                refs = payload.get("references") or []
                docs = getattr(sol, "docs", None) or []
                done_step(
                    logger,
                    "api.query.stream",
                    mode=param.mode,
                    docs=len(docs),
                    answer_chars=len(str(answer)),
                    references=len(refs),
                )
                # LightRAG-compatible order: references first, then response.
                yield f"data: {json.dumps({'references': refs}, ensure_ascii=False)}\n\n"
                yield f"data: {json.dumps({'response': answer}, ensure_ascii=False)}\n\n"
                yield "data: [DONE]\n\n"
            except Exception as exc:
                fail_step(
                    logger,
                    "api.query.stream",
                    mode=param.mode,
                    exc=exc,
                    exc_info=True,
                )
                yield f"data: {json.dumps({'error': str(exc)})}\n\n"

        return StreamingResponse(event_gen(), media_type="text/event-stream")

    return router
