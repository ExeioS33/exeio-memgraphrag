"""Provider registry and the client cache it depends on."""

from __future__ import annotations

import pytest

from memgraphrag.llm import providers
from memgraphrag.llm.openai_compatible import _llm_client, reset_client_cache

pytestmark = pytest.mark.offline


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for name in (
        "TOGETHER_API_KEY",
        "TOGETHER_MODELS",
        "TOGETHER_BASE_URL",
        "OLLAMA_API_KEY",
        "OLLAMA_BASE_URL",
        "OPENAI_API_KEY",
        "LLM_BINDING_API_KEY",
        "LLM_BINDING_HOST",
        "LLM_MODELS",
        "VLLM_API_KEY",
    ):
        monkeypatch.delenv(name, raising=False)
    reset_client_cache()
    yield
    reset_client_cache()


def test_together_defaults_to_the_documented_base_url() -> None:
    provider = providers.get_provider("together")
    assert provider is not None
    assert provider.base_url == "https://api.together.ai/v1"


def test_ollama_is_usable_without_a_credential() -> None:
    """Ollama's OpenAI shim requires the header to exist but ignores its value, so
    this provider must not be reported as unavailable just because no key is set."""
    provider = providers.get_provider("ollama")
    assert provider is not None and provider.is_available()
    assert providers.resolve("ollama").api_key == "ollama"


def test_provider_without_a_credential_raises_by_name(monkeypatch) -> None:
    with pytest.raises(ValueError) as exc:
        providers.resolve("together")
    assert "TOGETHER_API_KEY" in str(exc.value)


def test_unknown_provider_lists_the_known_ones() -> None:
    with pytest.raises(ValueError) as exc:
        providers.resolve("anthropic")
    message = str(exc.value)
    assert "anthropic" in message and "together" in message


def test_base_url_env_overrides_the_default(monkeypatch) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", "sk-test")
    monkeypatch.setenv("TOGETHER_BASE_URL", "https://proxy.interne/v1")
    assert providers.resolve("together").base_url == "https://proxy.interne/v1"


def test_allow_list_env_name_is_per_provider(monkeypatch) -> None:
    monkeypatch.setenv("TOGETHER_API_KEY", "sk-test")
    monkeypatch.setenv("TOGETHER_MODELS", "a, b ,c")
    resolved = providers.resolve("together")
    assert resolved.models == ("a", "b", "c")
    assert resolved.models_env == "TOGETHER_MODELS"

    monkeypatch.setenv("LLM_BINDING_API_KEY", "sk-server")
    assert providers.resolve(None).models_env == "LLM_MODELS"


def test_describe_available_reports_unavailable_rather_than_hiding(monkeypatch) -> None:
    """An operator who set TOGETHER_MODELS but forgot the key must see why the entry
    is greyed out, not wonder where it went."""
    monkeypatch.setenv("TOGETHER_MODELS", "some-model")
    entry = next(p for p in providers.describe_available() if p["id"] == "together")
    assert entry["available"] is False
    assert entry["models"] == ["some-model"]
    assert entry["models_env"] == "TOGETHER_MODELS"


def test_clients_are_cached_per_endpoint_and_credential() -> None:
    """The previous singleton froze base_url at the first call, which made
    per-request routing impossible."""
    a = _llm_client(base_url="https://api.together.ai/v1", api_key="k1")
    again = _llm_client(base_url="https://api.together.ai/v1", api_key="k1")
    other_host = _llm_client(base_url="http://localhost:11434/v1", api_key="k1")
    other_key = _llm_client(base_url="https://api.together.ai/v1", api_key="k2")

    assert a is again, "same endpoint and credential must reuse one client"
    assert a is not other_host
    assert a is not other_key
    assert str(a.base_url).rstrip("/") == "https://api.together.ai/v1"
    assert str(other_host.base_url).rstrip("/") == "http://localhost:11434/v1"


def test_clients_share_one_http_transport() -> None:
    """The httpx client carries the pool and the corporate CA context; building one
    per provider would redo the TLS handshake on every switch."""
    a = _llm_client(base_url="https://api.together.ai/v1", api_key="k1")
    b = _llm_client(base_url="http://localhost:11434/v1", api_key="k2")
    assert a._client is b._client
