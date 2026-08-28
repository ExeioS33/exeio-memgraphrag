"""Ollama-compatible API emulation for MemGraphRAG.

Adapted from LightRAG ``lightrag/api/routers/ollama_api.py`` (slimmed).
Mode prefixes in messages: ``/naive``, ``/context``, ``/bypass`` (default PPR+QA).
"""

from __future__ import annotations

import logging
import re
import time
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Optional

from memgraphrag.base import QueryParam
from memgraphrag.constants import DEFAULT_OLLAMA_MODEL_NAME, DEFAULT_OLLAMA_MODEL_TAG

logger = logging.getLogger("memgraphrag.api.ollama")

try:
    from fastapi import APIRouter, Depends, Request
    from pydantic import BaseModel, Field
except ImportError:  # pragma: no cover
    APIRouter = None  # type: ignore[misc, assignment]
    Depends = None  # type: ignore[misc, assignment]
    Request = None  # type: ignore[misc, assignment]
    BaseModel = object  # type: ignore[misc, assignment]
    # Stub so the module still imports without the [api] extra; never called.
    Field = lambda *a, **k: None  # type: ignore[misc, assignment]  # noqa: E731


class SearchMode(str, Enum):
    ppr = "ppr"
    naive = "naive"
    context = "context"
    bypass = "bypass"


class OllamaMessage(BaseModel):
    role: str
    content: str
    images: Optional[list[str]] = None


class OllamaChatRequest(BaseModel):
    model: str = Field(default=DEFAULT_OLLAMA_MODEL_NAME)
    messages: list[OllamaMessage]
    stream: bool = False
    options: Optional[dict[str, Any]] = None
    system: Optional[str] = None


class OllamaGenerateRequest(BaseModel):
    model: str = Field(default=DEFAULT_OLLAMA_MODEL_NAME)
    prompt: str
    system: Optional[str] = None
    stream: bool = False
    options: Optional[dict[str, Any]] = None


def parse_query_mode(query: str) -> tuple[str, SearchMode, bool, Optional[str]]:
    """Parse ``/naive``, ``/context``, ``/bypass`` (and optional ``[user_prompt]``)."""
    user_prompt: Optional[str] = None
    bracket_match = re.match(r"^/([a-z]*)\[(.*?)\]([\s\S]*)", query)
    if bracket_match:
        mode_prefix = bracket_match.group(1)
        user_prompt = bracket_match.group(2)
        remaining = bracket_match.group(3).lstrip()
        mode_map_bracket = {
            "naive": SearchMode.naive,
            "context": SearchMode.context,
            "bypass": SearchMode.bypass,
            "ppr": SearchMode.ppr,
            "": SearchMode.ppr,
        }
        mode = mode_map_bracket.get(mode_prefix, SearchMode.ppr)
        only_ctx = mode == SearchMode.context
        if mode == SearchMode.context:
            mode = SearchMode.ppr
        return remaining, mode, only_ctx, user_prompt

    mode_map = {
        "/naive ": (SearchMode.naive, False),
        "/bypass ": (SearchMode.bypass, False),
        "/ppr ": (SearchMode.ppr, False),
        "/context": (SearchMode.ppr, True),
        "/naivecontext": (SearchMode.naive, True),
    }
    for prefix, (mode, only_need_context) in mode_map.items():
        if query.startswith(prefix):
            return query[len(prefix) :].lstrip(), mode, only_need_context, user_prompt
    return query, SearchMode.ppr, False, user_prompt


def create_ollama_router(
    api_key: Optional[str] = None,
    model_name: str = DEFAULT_OLLAMA_MODEL_NAME,
    model_tag: str = DEFAULT_OLLAMA_MODEL_TAG,
) -> Any:
    if APIRouter is None:
        raise RuntimeError("fastapi is required; install memgraphrag[api]")

    from memgraphrag.api.dependencies import get_combined_auth_dependency

    router = APIRouter(prefix="/api", tags=["ollama"])
    combined_auth = get_combined_auth_dependency(api_key)
    full_model = f"{model_name}:{model_tag}"

    @router.get("/tags", dependencies=[Depends(combined_auth)])
    async def tags():
        now = datetime.now(timezone.utc).isoformat()
        return {
            "models": [
                {
                    "name": full_model,
                    "model": full_model,
                    "size": 0,
                    "digest": "memgraphrag",
                    "modified_at": now,
                    "details": {
                        "parent_model": "",
                        "format": "memgraphrag",
                        "family": "memgraphrag",
                        "families": ["memgraphrag"],
                        "parameter_size": "n/a",
                        "quantization_level": "n/a",
                    },
                }
            ]
        }

    @router.get("/version", dependencies=[Depends(combined_auth)])
    async def version():
        from memgraphrag import __version__

        return {"version": __version__}

    @router.get("/ps", dependencies=[Depends(combined_auth)])
    async def ps():
        now = datetime.now(timezone.utc).isoformat()
        return {
            "models": [
                {
                    "name": full_model,
                    "model": full_model,
                    "size": 0,
                    "digest": "memgraphrag",
                    "details": {
                        "parent_model": "",
                        "format": "memgraphrag",
                        "family": "memgraphrag",
                        "families": ["memgraphrag"],
                        "parameter_size": "n/a",
                        "quantization_level": "n/a",
                    },
                    "expires_at": now,
                    "size_vram": 0,
                }
            ]
        }

    async def _run_rag(
        rag: Any,
        query: str,
        history: list[dict[str, str]] | None = None,
        system: str | None = None,
    ) -> str:
        cleaned, mode, only_need_context, user_prompt = parse_query_mode(query)
        param = QueryParam(
            mode=mode.value,  # type: ignore[arg-type]
            only_need_context=only_need_context,
            conversation_history=history or [],
            user_prompt=user_prompt or system,
            top_k=getattr(rag, "top_k", 10),
        )
        sol = await rag.arag_qa(cleaned, param=param)
        if isinstance(sol, str):
            return sol
        if only_need_context or param.mode == "context":
            return "\n\n".join(sol.docs or [])
        return sol.answer or ""

    @router.post("/chat", dependencies=[Depends(combined_auth)])
    async def chat(request: Request, body: OllamaChatRequest):
        rag = request.app.state.rag
        if not body.messages:
            return {
                "model": body.model,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "message": {"role": "assistant", "content": ""},
                "done": True,
            }
        query = body.messages[-1].content
        history = [{"role": m.role, "content": m.content} for m in body.messages[:-1]]
        answer = await _run_rag(rag, query, history=history, system=body.system)
        return {
            "model": body.model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "message": {"role": "assistant", "content": answer},
            "done": True,
        }

    @router.post("/generate", dependencies=[Depends(combined_auth)])
    async def generate(request: Request, body: OllamaGenerateRequest):
        rag = request.app.state.rag
        t0 = time.time_ns()
        answer = await _run_rag(rag, body.prompt, system=body.system)
        return {
            "model": body.model,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "response": answer,
            "done": True,
            "context": None,
            "total_duration": time.time_ns() - t0,
            "load_duration": 0,
            "prompt_eval_count": None,
            "prompt_eval_duration": None,
            "eval_count": None,
            "eval_duration": None,
        }

    return router
