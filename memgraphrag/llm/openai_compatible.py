"""OpenAI-compatible LLM and embedding bindings for MemGraphRAG.

Adapted from LightRAG ``lightrag/llm/openai.py`` (async AsyncOpenAI complete /
embed pattern) and MemGraphRAG ``code/src/llm/openai_gpt.py`` (env-driven
OpenAI-compatible client). Env vars: ``LLM_BINDING_*``, ``LLM_MODEL``,
``EMBEDDING_BINDING_*``, ``EMBEDDING_MODEL``, ``EMBEDDING_DIM``.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, AsyncIterator, Sequence

import numpy as np
from openai import (
    APIConnectionError,
    APITimeoutError,
    AsyncOpenAI,
    BadRequestError,
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
from memgraphrag.utils.env import get_env_value
from memgraphrag.utils.http_ssl import ssl_verify

logger = logging.getLogger(__name__)

_httpx_client = None
_llm_openai: AsyncOpenAI | None = None
_embed_openai: AsyncOpenAI | None = None


def _http_client():
    """Shared httpx client so corporate CAs (SSL_CERT_FILE) are honored."""
    global _httpx_client
    if _httpx_client is not None:
        return _httpx_client
    import httpx

    verify = ssl_verify()
    _httpx_client = httpx.AsyncClient(
        verify=verify, timeout=httpx.Timeout(150.0, connect=30.0)
    )
    return _httpx_client


def _llm_client() -> AsyncOpenAI:
    global _llm_openai
    if _llm_openai is not None:
        return _llm_openai
    api_key = os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY") or "no-key"
    base_url = os.getenv("LLM_BINDING_HOST") or None
    _llm_openai = AsyncOpenAI(
        api_key=api_key, base_url=base_url, http_client=_http_client()
    )
    return _llm_openai


def _embed_client() -> AsyncOpenAI:
    global _embed_openai
    if _embed_openai is not None:
        return _embed_openai
    api_key = (
        os.getenv("EMBEDDING_BINDING_API_KEY")
        or os.getenv("LLM_BINDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "no-key"
    )
    base_url = os.getenv("EMBEDDING_BINDING_HOST") or os.getenv("LLM_BINDING_HOST") or None
    _embed_openai = AsyncOpenAI(
        api_key=api_key, base_url=base_url, http_client=_http_client()
    )
    return _embed_openai


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
def _embedding_max_tokens() -> int | None:
    """Optional hard cap for embed inputs (e.g. e5 family = 512)."""
    raw = (os.getenv("EMBEDDING_MAX_TOKENS") or "").strip()
    if not raw:
        return None
    try:
        value = int(raw)
    except ValueError:
        return None
    return value if value > 0 else None


def _embedding_token_safety(model_name: str) -> float:
    """Tiktoken→provider tokenizer safety factor for embed truncation.

    gpt-4o-mini tiktoken systematically undercounts WordPiece/e5 tokenizers.
    Runtime probe: 480 tiktoken tokens → Together reported 748 for e5-instruct.
    """
    raw = (os.getenv("EMBEDDING_TOKEN_SAFETY") or "").strip()
    if raw:
        try:
            value = float(raw)
            if 0.1 <= value <= 1.0:
                return value
        except ValueError:
            pass
    lowered = (model_name or "").lower()
    if any(tag in lowered for tag in ("e5", "bge", "gte-", "nomic-embed")):
        return 0.60
    return 0.95


def _embedding_budget_tokens(model_name: str) -> int | None:
    """Effective tiktoken budget after applying tokenizer safety."""
    configured = _embedding_max_tokens()
    if configured is None:
        return None
    safety = _embedding_token_safety(model_name)
    budget = max(32, int(configured * safety))
    if budget < configured:
        logger.info(
            "Embed token budget %d (configured=%d safety=%.2f model=%s)",
            budget,
            configured,
            safety,
            model_name,
        )
    return budget


def _truncate_for_embedding(texts: list[str], max_tokens: int) -> list[str]:
    """Truncate texts to ``max_tokens`` using tiktoken when available.

    Providers like Together's e5 endpoints reject inputs above 512 tokens.
    Tiktoken is an approximation of the model tokenizer; callers should pass a
    safety-reduced budget (see ``_embedding_budget_tokens``).
    """
    try:
        from memgraphrag.utils.tokenizer import TiktokenTokenizer

        tok = TiktokenTokenizer("gpt-4o-mini")
    except Exception:
        # ~2.5 chars/token is conservative for scientific English vs WordPiece.
        char_limit = max(int(max_tokens * 2.5), 64)
        out = []
        for t in texts:
            if len(t) > char_limit:
                logger.warning(
                    "Truncating embed input from %d to %d chars (no tokenizer)",
                    len(t),
                    char_limit,
                )
                out.append(t[:char_limit])
            else:
                out.append(t)
        return out

    out = []
    truncated = 0
    for t in texts:
        ids = tok.encode(t)
        if len(ids) > max_tokens:
            truncated += 1
            out.append(tok.decode(ids[:max_tokens]))
        else:
            out.append(t)
    if truncated:
        logger.warning(
            "Truncated %d/%d embed inputs to %d tokens",
            truncated,
            len(texts),
            max_tokens,
        )
    return out


_CONTEXT_LEN_RE = re.compile(
    r"maximum context length is (?P<max>\d+).*?requested (?P<req>\d+)",
    re.IGNORECASE | re.DOTALL,
)


def _parse_context_length_error(exc: BaseException) -> tuple[int, int] | None:
    match = _CONTEXT_LEN_RE.search(str(exc))
    if not match:
        return None
    return int(match.group("max")), int(match.group("req"))


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
    budget = _embedding_budget_tokens(model_name)
    if budget is not None:
        texts_list = _truncate_for_embedding(texts_list, budget)

    client = _embed_client()
    logger.debug("openai_embed model=%s n=%d dim=%s", model_name, len(texts_list), dim)

    create_kwargs: dict[str, Any] = {
        "model": model_name,
        "input": texts_list,
    }
    # Many OpenAI-compatible servers accept dimensions; ignore if unsupported.
    send_dims = get_env_value("EMBEDDING_SEND_DIMENSIONS", True, bool)
    if dim and send_dims:
        create_kwargs["dimensions"] = dim
    create_kwargs.update({k: v for k, v in kwargs.items() if v is not None})

    try:
        response = await client.embeddings.create(**create_kwargs)
    except Exception as exc:
        # Retry without dimensions for providers that reject the param.
        if "dimensions" in create_kwargs and "dimension" in str(exc).lower():
            create_kwargs.pop("dimensions", None)
            logger.warning("Retrying embeddings without dimensions: %s", exc)
            try:
                response = await client.embeddings.create(**create_kwargs)
            except Exception as exc2:
                exc = exc2
            else:
                vectors = [item.embedding for item in response.data]
                return np.asarray(vectors, dtype=np.float32)

        parsed = _parse_context_length_error(exc)
        if parsed and budget is not None:
            provider_max, provider_req = parsed
            # Scale tiktoken budget by observed provider overshoot, with margin.
            scaled = max(
                32,
                int(budget * (provider_max / max(provider_req, 1)) * 0.90),
            )
            if scaled < budget:
                logger.warning(
                    "Embed context overflow (provider %d/%d); retry with budget %d→%d",
                    provider_req,
                    provider_max,
                    budget,
                    scaled,
                )
                texts_list = _truncate_for_embedding(
                    [t if t is not None else "" for t in texts],
                    scaled,
                )
                create_kwargs["input"] = texts_list
                response = await client.embeddings.create(**create_kwargs)
            else:
                raise
        elif isinstance(exc, BadRequestError) and budget is not None:
            # Fallback when error text is non-standard: hard-cut budget in half.
            scaled = max(32, budget // 2)
            logger.warning(
                "Embed BadRequestError; retry with budget %d→%d: %s",
                budget,
                scaled,
                exc,
            )
            texts_list = _truncate_for_embedding(
                [t if t is not None else "" for t in texts],
                scaled,
            )
            create_kwargs["input"] = texts_list
            response = await client.embeddings.create(**create_kwargs)
        else:
            raise

    vectors = [item.embedding for item in response.data]
    arr = np.asarray(vectors, dtype=np.float32)
    return arr
