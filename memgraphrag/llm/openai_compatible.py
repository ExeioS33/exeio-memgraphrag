"""OpenAI-compatible LLM and embedding bindings for MemGraphRAG.

Adapted from LightRAG ``lightrag/llm/openai.py`` (async AsyncOpenAI complete /
embed pattern) and MemGraphRAG ``code/src/llm/openai_gpt.py`` (env-driven
OpenAI-compatible client). Env vars: ``LLM_BINDING_*``, ``LLM_MODEL``,
``EMBEDDING_BINDING_*``, ``EMBEDDING_MODEL``, ``EMBEDDING_DIM``.
"""

from __future__ import annotations

import logging
import os
from typing import Any, AsyncIterator, Sequence

import numpy as np
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    InternalServerError,
    RateLimitError,
)
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from memgraphrag.constants import EMBEDDING_DIM
from memgraphrag.utils.debug_log import agent_dbg
from memgraphrag.utils.env import get_env_value
from memgraphrag.utils.http_ssl import ssl_verify

logger = logging.getLogger(__name__)


def _http_client():
    """Shared httpx client so corporate CAs (SSL_CERT_FILE) are honored."""
    import httpx

    verify = ssl_verify()
    # #region agent log
    agent_dbg(
        "E",
        "openai_compatible.py:_http_client",
        "ssl verify for OpenAI client",
        {"verify": verify if isinstance(verify, bool) else str(verify)},
        run_id="post-fix",
    )
    # #endregion
    return httpx.AsyncClient(verify=verify, timeout=httpx.Timeout(150.0, connect=30.0))


def _llm_client() -> AsyncOpenAI:
    api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY") or "no-key"
    base_url = os.getenv("LLM_BINDING_HOST") or None
    return AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=_http_client())


def _embed_client() -> AsyncOpenAI:
    api_key = (
        os.getenv("EMBEDDING_BINDING_API_KEY")
        or os.getenv("LLM_BINDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "no-key"
    )
    base_url = os.getenv("EMBEDDING_BINDING_HOST") or os.getenv("LLM_BINDING_HOST") or None
    return AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=_http_client())


def _llm_model(explicit: str | None = None) -> str:
    return explicit or os.getenv("LLM_MODEL") or "gpt-4o-mini"


def _embed_model(explicit: str | None = None) -> str:
    return explicit or os.getenv("EMBEDDING_MODEL") or "text-embedding-3-small"


def _embed_dim(explicit: int | None = None) -> int:
    if explicit is not None:
        return explicit
    return get_env_value("EMBEDDING_DIM", EMBEDDING_DIM, int)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(InternalServerError)
    ),
    reraise=True,
)
async def openai_complete(
    prompt: str,
    system_prompt: str | None = None,
    history_messages: list[dict[str, str]] | None = None,
    model: str | None = None,
    temperature: float = 0.0,
    max_tokens: int | None = None,
    stream: bool = False,
    **kwargs: Any,
) -> str | AsyncIterator[str]:
    """Chat-complete via an OpenAI-compatible endpoint.

    Reads ``LLM_BINDING_HOST``, ``LLM_BINDING_API_KEY``, ``LLM_MODEL`` from the
    environment when not overridden by kwargs / ``model``.
    """
    history_messages = history_messages or []
    messages: list[dict[str, str]] = []
    if system_prompt:
        messages.append({"role": "system", "content": system_prompt})
    messages.extend(history_messages)
    messages.append({"role": "user", "content": prompt})

    params: dict[str, Any] = {
        "model": _llm_model(model),
        "messages": messages,
        "temperature": temperature,
        "stream": stream,
    }
    if max_tokens is not None:
        params["max_tokens"] = max_tokens
    # Allow callers to pass through extra OpenAI kwargs (seed, response_format, …)
    for key in ("seed", "response_format", "n", "top_p", "stop"):
        if key in kwargs and kwargs[key] is not None:
            params[key] = kwargs[key]

    client = _llm_client()
    logger.debug("openai_complete model=%s stream=%s", params["model"], stream)

    if stream:
        response = await client.chat.completions.create(**params)

        async def _gen() -> AsyncIterator[str]:
            async for chunk in response:
                delta = chunk.choices[0].delta.content if chunk.choices else None
                if delta:
                    yield delta

        return _gen()

    response = await client.chat.completions.create(**params)
    content = response.choices[0].message.content or ""
    return content


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=30),
    retry=(
        retry_if_exception_type(RateLimitError)
        | retry_if_exception_type(APIConnectionError)
        | retry_if_exception_type(APITimeoutError)
        | retry_if_exception_type(InternalServerError)
    ),
    reraise=True,
)
async def openai_embed(
    texts: Sequence[str],
    model: str | None = None,
    embedding_dim: int | None = None,
    instruction: str | None = None,
    query_prefix: str | None = None,
    context: str = "document",
    **kwargs: Any,
) -> np.ndarray:
    """Embed texts via an OpenAI-compatible embeddings endpoint.

    Optional asymmetric query prefix: when ``context=="query"`` and
    ``instruction`` / ``query_prefix`` is set, each text is prefixed before
    embedding (MemGraphRAG linking instructions).
    """
    prefix = query_prefix or instruction
    if context == "query" and prefix:
        texts = [f"{prefix} {t}" if not t.startswith(prefix) else t for t in texts]
    elif instruction and context != "document":
        texts = [f"{instruction} {t}" for t in texts]

    texts_list = [t if t is not None else "" for t in texts]
    dim = _embed_dim(embedding_dim)
    model_name = _embed_model(model)

    client = _embed_client()
    logger.debug("openai_embed model=%s n=%d dim=%s", model_name, len(texts_list), dim)

    create_kwargs: dict[str, Any] = {
        "model": model_name,
        "input": texts_list,
    }
    # Many OpenAI-compatible servers accept dimensions; ignore if unsupported upstream.
    if dim:
        create_kwargs["dimensions"] = dim
    create_kwargs.update({k: v for k, v in kwargs.items() if v is not None})

    try:
        response = await client.embeddings.create(**create_kwargs)
    except Exception as exc:
        # Retry without dimensions for providers that reject the param.
        if "dimensions" in create_kwargs and "dimension" in str(exc).lower():
            create_kwargs.pop("dimensions", None)
            logger.warning("Retrying embeddings without dimensions: %s", exc)
            response = await client.embeddings.create(**create_kwargs)
        else:
            raise

    vectors = [item.embedding for item in response.data]
    arr = np.asarray(vectors, dtype=np.float32)
    return arr
