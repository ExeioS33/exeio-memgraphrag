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
from memgraphrag.utils.step_log import embed_call, llm_call, truncate

logger = logging.getLogger(__name__)

_httpx_client = None
# Keyed on (base_url, api_key) rather than being one global per role. The previous
# two singletons froze the base URL and credential at the first call in the process,
# which made a per-request provider impossible: a caller could already ask for a
# model from another provider and it would be sent to whichever host happened to be
# configured, producing a provider-side 404 instead of a routing decision.
_openai_clients: dict[tuple[str | None, str], AsyncOpenAI] = {}


def _http_client():
    """Shared httpx client so corporate CAs (SSL_CERT_FILE) are honored.

    Deliberately ONE instance across every provider: it carries the connection pool
    and the SSL context, both of which are transport concerns. Building one per
    provider would redo the TLS handshake on every switch and leak sockets under
    MAX_ASYNC_LLM concurrency.
    """
    global _httpx_client
    if _httpx_client is not None:
        return _httpx_client
    import httpx

    verify = ssl_verify()
    _httpx_client = httpx.AsyncClient(verify=verify, timeout=httpx.Timeout(150.0, connect=30.0))
    return _httpx_client


def _client_for(base_url: str | None, api_key: str) -> AsyncOpenAI:
    """Return a cached client for one (endpoint, credential) pair."""
    key = (base_url or None, api_key)
    cached = _openai_clients.get(key)
    if cached is not None:
        return cached
    client = AsyncOpenAI(api_key=api_key, base_url=base_url, http_client=_http_client())
    _openai_clients[key] = client
    return client


def _llm_client(base_url: str | None = None, api_key: str | None = None) -> AsyncOpenAI:
    """Completion client. Explicit arguments win; otherwise the server binding."""
    resolved_key = (
        api_key or os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY") or "no-key"
    )
    resolved_url = base_url if base_url is not None else (os.getenv("LLM_BINDING_HOST") or None)
    return _client_for(resolved_url, resolved_key)


def _embed_client() -> AsyncOpenAI:
    """Embedding client — resolved ONLY from EMBEDDING_*, never from a request.

    The corpus is indexed with one embedding model at one dimension; routing
    embeddings to a different provider at query time returns vectors from a
    different space and silently degrades every answer (or trips the Postgres
    dimension guard). This function therefore takes no arguments by design.

    Note the LLM_BINDING_HOST fallback below: it is the one place where completion
    configuration can still reach embeddings, kept for deployments that set only the
    LLM host. It reads the env, never a per-request override.
    """
    api_key = (
        os.getenv("EMBEDDING_BINDING_API_KEY")
        or os.getenv("LLM_BINDING_API_KEY")
        or os.getenv("OPENAI_API_KEY")
        or "no-key"
    )
    base_url = os.getenv("EMBEDDING_BINDING_HOST") or os.getenv("LLM_BINDING_HOST") or None
    return _client_for(base_url, api_key)


def reset_client_cache() -> None:
    """Drop cached clients and catalogues. For tests that swap env between cases."""
    _openai_clients.clear()
    _model_catalogue.clear()


#: Model ids advertised by an endpoint, cached per (base_url, api_key).
_model_catalogue: dict[tuple[str | None, str], tuple[str, ...]] = {}

#: Catalogue entry kinds that can serve a chat completion.
_CHAT_MODEL_TYPES = {"chat", "language", "code", "completion"}


def _is_chat_model(entry: dict[str, Any]) -> bool:
    """Keep only what can answer a completion.

    Together AI advertises 278 models on one endpoint — embeddings, rerankers and
    image generators included. Offering those in a model picker would let someone
    pick an image model to answer a question about invoicing. Providers that do not
    label their entries (OpenAI, Ollama, vLLM) are left untouched.
    """
    kind = entry.get("type")
    if not isinstance(kind, str):
        return True
    return kind.strip().lower() in _CHAT_MODEL_TYPES


async def list_models(base_url: str | None = None, api_key: str | None = None) -> tuple[str, ...]:
    """Model ids the endpoint advertises via ``GET /v1/models``.

    Every provider in the registry implements this — it is part of the OpenAI
    protocol they all speak. Used as the allow-list when an operator has not pinned
    one explicitly, so a newly configured provider works without a second env var
    while a model name is still checked against something real.

    Returns an empty tuple on any failure: an unreachable catalogue must not take
    the query path down with it, and the caller falls back to its own allow-list.
    """
    resolved_key = (
        api_key or os.getenv("LLM_BINDING_API_KEY") or os.getenv("OPENAI_API_KEY") or "no-key"
    )
    root = (base_url or os.getenv("LLM_BINDING_HOST") or "https://api.openai.com/v1").rstrip("/")
    cache_key = (root, resolved_key)
    cached = _model_catalogue.get(cache_key)
    if cached is not None:
        return cached

    names: tuple[str, ...] = ()
    try:
        response = await _http_client().get(
            f"{root}/models",
            headers={"Authorization": f"Bearer {resolved_key}"},
            timeout=20.0,
        )
        response.raise_for_status()
        payload = response.json()
        # Deliberately NOT client.models.list(): the OpenAI SDK insists on the
        # {"object": "list", "data": [...]} envelope, while Together AI answers with
        # a bare JSON array and the SDK raises while parsing a perfectly good 200.
        entries = payload.get("data", payload) if isinstance(payload, dict) else payload
        if isinstance(entries, list):
            names = tuple(
                sorted(
                    {
                        str(item["id"])
                        for item in entries
                        if isinstance(item, dict) and item.get("id") and _is_chat_model(item)
                    }
                )
            )
    except Exception as exc:  # noqa: BLE001 - catalogue is advisory, never fatal
        logger.warning("Model catalogue unavailable for %s: %s", root, exc)

    _model_catalogue[cache_key] = names
    return names


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
    base_url: str | None = None,
    api_key: str | None = None,
    **kwargs: Any,
) -> str | AsyncIterator[str]:
    """Chat-complete via an OpenAI-compatible endpoint.

    Reads ``LLM_BINDING_HOST``, ``LLM_BINDING_API_KEY``, ``LLM_MODEL`` from the
    environment when not overridden by kwargs / ``model``.

    Logging kwargs (stripped before the HTTP call, never sent to the provider):

    - ``agent``: multi-agent role id (e.g. ``openie.ner``, ``schema.extract``,
      ``conflict.detect``, ``qa.reading``)
    - ``llm_action`` / ``action``: short verb (default ``complete``)
    """
    history_messages = history_messages or []
    agent = kwargs.pop("agent", None)
    llm_action = kwargs.pop("llm_action", None) or kwargs.pop("action", None)
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

    # Declared parameters, not **kwargs: the pass-through whitelist below silently
    # drops anything it does not recognise, so a routing argument smuggled through
    # kwargs would be swallowed and the call would go to the wrong provider with no
    # error at all.
    client = _llm_client(base_url=base_url, api_key=api_key)
    if agent or llm_action:
        llm_call(
            logger,
            agent=str(agent or "llm"),
            action=str(llm_action or "complete"),
            model=params["model"],
            stream=stream,
            prompt_chars=len(prompt or ""),
            system_chars=len(system_prompt or ""),
            history_turns=len(history_messages),
            preview=truncate(prompt, 80),
        )
    else:
        logger.debug(
            "openai_complete model=%s stream=%s prompt_chars=%d",
            params["model"],
            stream,
            len(prompt or ""),
        )

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
    if agent or llm_action:
        llm_call(
            logger,
            agent=str(agent or "llm"),
            action="complete_done",
            model=params["model"],
            response_chars=len(content),
        )
    return content


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


def _split_embedding_batches(texts: list[str], model_name: str) -> list[list[str]]:
    """Split texts into requests bounded by item count AND token budget.

    Providers cap both: OpenAI allows 2048 inputs and ~300k tokens per embeddings
    request. The defaults here stay well under any known ceiling; raise
    EMBEDDING_BATCH_SIZE / EMBEDDING_BATCH_MAX_TOKENS on a provider you control.
    """
    max_items = max(1, get_env_value("EMBEDDING_BATCH_SIZE", 64, int))
    max_tokens = max(1, get_env_value("EMBEDDING_BATCH_MAX_TOKENS", 100_000, int))

    try:
        from memgraphrag.utils.tokenizer import TiktokenTokenizer

        encode = TiktokenTokenizer().encode
    except Exception:  # tokenizer unavailable: fall back to a character heuristic

        def encode(text: str):
            return range(max(1, len(text) // 4))

    batches: list[list[str]] = []
    current: list[str] = []
    current_tokens = 0
    for text in texts:
        try:
            size = len(encode(text))
        except Exception:
            size = max(1, len(text) // 4)
        if current and (len(current) >= max_items or current_tokens + size > max_tokens):
            batches.append(current)
            current, current_tokens = [], 0
        current.append(text)
        current_tokens += size
    if current:
        batches.append(current)
    return batches or [[]]


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
    if not texts_list:
        return np.zeros((0, _embed_dim(embedding_dim)), dtype=np.float32)

    model_name = _embed_model(model)
    budget = _embedding_budget_tokens(model_name)

    # Split into requests BEFORE calling the provider. The per-item truncation below
    # bounds each text, but nothing bounded the request: every caller sent its whole
    # list in one shot, so indexing a real corpus (1 700 chunks x 1 200 tokens ~ 2M
    # tokens) blew past the provider's per-request ceiling and the document failed.
    batches = _split_embedding_batches(texts_list, model_name)
    if len(batches) > 1:
        logger.info(
            "Embedding %d texts in %d requests (model=%s)",
            len(texts_list),
            len(batches),
            model_name,
        )
    chunks = []
    for batch in batches:
        chunks.append(
            await _embed_request_bisecting(
                batch,
                model=model,
                embedding_dim=embedding_dim,
                budget=budget,
                **kwargs,
            )
        )
    return np.vstack(chunks) if len(chunks) > 1 else chunks[0]


async def _embed_request_bisecting(
    texts_list: list[str],
    *,
    model: str | None,
    embedding_dim: int | None,
    budget: int | None,
    **kwargs: Any,
) -> np.ndarray:
    """One embedding request, halving the batch when the provider refuses it.

    Per-item truncation cannot help when it is the *number* of items that overflows,
    and the previous fallbacks were all gated on ``EMBEDDING_MAX_TOKENS`` being set —
    unset by default, so the error was simply re-raised.
    """
    try:
        return await _embed_request(
            texts_list, model=model, embedding_dim=embedding_dim, budget=budget, **kwargs
        )
    except BadRequestError:
        if len(texts_list) <= 1:
            raise
        mid = len(texts_list) // 2
        logger.warning(
            "Embedding request of %d texts refused; splitting into %d + %d",
            len(texts_list),
            mid,
            len(texts_list) - mid,
        )
        left = await _embed_request_bisecting(
            texts_list[:mid], model=model, embedding_dim=embedding_dim, budget=budget, **kwargs
        )
        right = await _embed_request_bisecting(
            texts_list[mid:], model=model, embedding_dim=embedding_dim, budget=budget, **kwargs
        )
        return np.vstack([left, right])


# Retry the individual request, not the batching wrapper: a transient 429 on one
# batch must not re-embed every batch that already succeeded. (This block also used to
# sit above `_embedding_max_tokens`, a synchronous pure function, so embeddings had no
# retry at all while completions did.)
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
async def _embed_request(
    texts_list: list[str],
    *,
    model: str | None = None,
    embedding_dim: int | None = None,
    budget: int | None = None,
    context: str = "document",
    **kwargs: Any,
) -> np.ndarray:
    """Issue a single embeddings request for an already-batched list."""
    texts = list(texts_list)
    dim = _embed_dim(embedding_dim)
    model_name = _embed_model(model)
    if budget is not None:
        texts_list = _truncate_for_embedding(texts_list, budget)

    client = _embed_client()
    embed_call(
        logger,
        context=str(context or "document"),
        model=model_name,
        n=len(texts_list),
        dim=dim,
    )

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
