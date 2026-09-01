"""Declarative registry of LLM providers.

Every provider here speaks the OpenAI wire protocol — Together AI, Ollama's `/v1`
shim, vLLM and OpenAI itself are all reachable with the same client. So a
"connector" is not a new protocol implementation: it is a *base URL plus a
credential*, resolved server-side.

Credentials are deliberately resolved from the environment and never accepted from
a request. A browser sends a provider **id**; the server decides what key that id
maps to. Letting a caller supply `api_key` would turn this endpoint into an open
relay for whatever host they name.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field

#: Provider used when a request names none — the binding the server was started with.
DEFAULT_PROVIDER_ID = "default"


@dataclass(frozen=True)
class Provider:
    """One reachable OpenAI-compatible endpoint."""

    id: str
    label: str
    #: Default base URL. ``None`` means "the OpenAI SDK default" (api.openai.com).
    base_url: str | None
    #: Env vars tried in order for the credential.
    api_key_env: tuple[str, ...] = ()
    #: Env var overriding ``base_url``, for self-hosted deployments.
    base_url_env: str | None = None
    #: Used when no env var resolves. Only meaningful for endpoints that ignore
    #: the credential entirely, such as a local Ollama.
    fallback_api_key: str | None = None
    doc_url: str = ""
    note: str = ""

    def resolved_base_url(self) -> str | None:
        if self.base_url_env:
            override = os.getenv(self.base_url_env)
            if override and override.strip():
                return override.strip()
        return self.base_url

    def resolved_api_key(self) -> str | None:
        for name in self.api_key_env:
            value = os.getenv(name)
            if value and value.strip():
                return value.strip()
        return self.fallback_api_key

    def is_available(self) -> bool:
        """True when this provider has everything it needs to be called."""
        return bool(self.resolved_api_key())


@dataclass(frozen=True)
class ResolvedProvider:
    """A provider flattened into the two values the client actually needs."""

    id: str
    label: str
    base_url: str | None
    api_key: str
    models_env: str = ""
    #: Models the operator allow-listed for this provider.
    models: tuple[str, ...] = field(default_factory=tuple)


_REGISTRY: dict[str, Provider] = {
    DEFAULT_PROVIDER_ID: Provider(
        id=DEFAULT_PROVIDER_ID,
        label="Binding du serveur",
        base_url=None,
        base_url_env="LLM_BINDING_HOST",
        api_key_env=("LLM_BINDING_API_KEY", "OPENAI_API_KEY"),
        note="Le couple LLM_BINDING_HOST / LLM_BINDING_API_KEY du démarrage.",
    ),
    "together": Provider(
        id="together",
        label="Together AI",
        base_url="https://api.together.ai/v1",
        base_url_env="TOGETHER_BASE_URL",
        api_key_env=("TOGETHER_API_KEY", "LLM_BINDING_API_KEY"),
        doc_url="https://docs.together.ai/docs/inference/openai-compatibility",
    ),
    "openai": Provider(
        id="openai",
        label="OpenAI",
        base_url=None,
        base_url_env="OPENAI_BASE_URL",
        api_key_env=("OPENAI_API_KEY",),
        doc_url="https://platform.openai.com/docs/api-reference",
    ),
    "ollama": Provider(
        id="ollama",
        label="Ollama (local)",
        base_url="http://localhost:11434/v1",
        base_url_env="OLLAMA_BASE_URL",
        api_key_env=("OLLAMA_API_KEY",),
        # Ollama's OpenAI shim requires the header to exist but ignores its value,
        # so this provider is usable with no credential configured at all.
        fallback_api_key="ollama",
        doc_url="https://docs.ollama.com/openai",
    ),
    "vllm": Provider(
        id="vllm",
        label="vLLM (auto-hébergé)",
        base_url=None,
        base_url_env="VLLM_BASE_URL",
        api_key_env=("VLLM_API_KEY", "LLM_BINDING_API_KEY"),
        doc_url="https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html",
    ),
}


def known_providers() -> tuple[Provider, ...]:
    return tuple(_REGISTRY.values())


def get_provider(provider_id: str | None) -> Provider | None:
    if not provider_id:
        return _REGISTRY[DEFAULT_PROVIDER_ID]
    return _REGISTRY.get(str(provider_id).strip().lower())


def _models_env_name(provider_id: str) -> str:
    """Allow-list env var for a provider: LLM_MODELS for the server binding,
    ``<PROVIDER>_MODELS`` otherwise."""
    if provider_id == DEFAULT_PROVIDER_ID:
        return "LLM_MODELS"
    return f"{provider_id.upper()}_MODELS"


def models_for_provider(provider_id: str) -> tuple[str, ...]:
    raw = os.getenv(_models_env_name(provider_id)) or ""
    return tuple(m.strip() for m in raw.split(",") if m.strip())


def resolve(provider_id: str | None) -> ResolvedProvider:
    """Flatten a provider id into base_url + credential.

    Raises ``ValueError`` for an unknown id or one with no usable credential —
    both are operator misconfigurations that must surface as a 400, not as an
    opaque 401 from someone else's API.
    """
    provider = get_provider(provider_id)
    if provider is None:
        available = ", ".join(sorted(_REGISTRY))
        raise ValueError(f"Fournisseur inconnu : {provider_id!r}. Disponibles : {available}.")
    api_key = provider.resolved_api_key()
    if not api_key:
        wanted = " ou ".join(provider.api_key_env) or "(aucune)"
        raise ValueError(
            f"Le fournisseur {provider.id!r} n'a pas de clé configurée. Renseignez {wanted}."
        )
    return ResolvedProvider(
        id=provider.id,
        label=provider.label,
        base_url=provider.resolved_base_url(),
        api_key=api_key,
        models_env=_models_env_name(provider.id),
        models=models_for_provider(provider.id),
    )


def describe_available() -> list[dict[str, object]]:
    """Providers the UI may offer, with their allow-listed models.

    A provider with no credential is reported as unavailable rather than hidden:
    an operator who set TOGETHER_MODELS but forgot the key should see why the
    entry is greyed out instead of wondering where it went.
    """
    out: list[dict[str, object]] = []
    for provider in known_providers():
        out.append(
            {
                "id": provider.id,
                "label": provider.label,
                "available": provider.is_available(),
                "base_url": provider.resolved_base_url(),
                "models": list(models_for_provider(provider.id)),
                "models_env": _models_env_name(provider.id),
                "doc_url": provider.doc_url,
                "note": provider.note,
            }
        )
    return out
