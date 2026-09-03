"""The agent loop's failure modes.

Every case here is one that fails *silently* in the wild: a model that ignores
`tools` and answers in prose, a loop that repeats one call forever, arguments that
are almost-JSON, a third hop that overflows the context window. None of them raise
on their own, which is why they are provoked rather than hoped for.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

from memgraphrag.agent.budget import enforce
from memgraphrag.agent.capabilities import AgentUnsupportedModelError, precheck_model
from memgraphrag.agent.loop import run_agent
from memgraphrag.agent.tools import ToolBox, _fence_from, _parse_arguments
from memgraphrag.base import QueryParam
from memgraphrag.utils.misc import QuerySolution

pytestmark = pytest.mark.offline


# --------------------------------------------------------------------------- #
# Doubles
# --------------------------------------------------------------------------- #


def _tool_message(name: str, arguments: str, call_id: str = "call_1") -> SimpleNamespace:
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(
                id=call_id,
                function=SimpleNamespace(name=name, arguments=arguments),
            )
        ],
    )


class FakeRag:
    """An engine whose retrieval always answers, so the loop is what is under test."""

    def __init__(self, docs: list[str] | None = None) -> None:
        self.docs = docs or ["passage about the corpus"]
        self.queries: list[str] = []

    async def aretrieve(self, query: str, param: Any = None):
        self.queries.append(query)
        return [
            QuerySolution(
                question=query,
                docs=list(self.docs),
                sources=[f"doc{i}.pdf" for i in range(len(self.docs))],
                passage_ids=[f"chunk-{i}" for i in range(len(self.docs))],
            )
        ]


def scripted_llm(turns: list[Any]):
    """An `llm_model_func` that plays `turns` in order.

    A list entry is either a message object (buffered turn) or a list of stream
    chunks (streamed turn).
    """
    state = {"i": 0}

    async def llm(prompt: str, **kwargs: Any):
        turn = turns[min(state["i"], len(turns) - 1)]
        state["i"] += 1
        if isinstance(turn, list):

            async def _gen():
                for chunk in turn:
                    yield chunk

            return _gen()
        return turn

    llm.calls = state  # type: ignore[attr-defined]
    return llm


async def drain(agen) -> list[dict]:
    return [frame async for frame in agen]


# --------------------------------------------------------------------------- #
# Capability
# --------------------------------------------------------------------------- #


def test_precheck_refuses_a_model_that_cannot_chat() -> None:
    with pytest.raises(AgentUnsupportedModelError) as exc:
        precheck_model("intfloat/multilingual-e5-large-instruct-embed")
    assert "tool calling" in str(exc.value)


def test_precheck_allows_an_unknown_name() -> None:
    """The name list is a filter, not an oracle — the runtime check decides."""
    precheck_model("some-vendor/brand-new-model")


def test_allow_list_overrides_the_built_in_rules(monkeypatch) -> None:
    monkeypatch.setenv("AGENT_TOOL_MODELS", "gpt-oss")
    precheck_model("openai/gpt-oss-20b")
    with pytest.raises(AgentUnsupportedModelError):
        precheck_model("meta-llama/Llama-3.3-70B")


@pytest.mark.asyncio
async def test_a_model_that_ignores_tools_is_refused_not_degraded() -> None:
    """The decisive check: a *forced* tool call answered in prose proves the model
    cannot call tools. Letting it through would produce an ungrounded answer that
    looks exactly like a working one."""
    llm = scripted_llm([SimpleNamespace(content="Sure! Here is my answer.", tool_calls=None)])
    toolbox = ToolBox(FakeRag(), QueryParam())
    with pytest.raises(AgentUnsupportedModelError):
        await drain(run_agent(question="q", llm=llm, toolbox=toolbox, model="prose-only/model"))


# --------------------------------------------------------------------------- #
# Termination
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_repeated_identical_call_does_not_run_twice() -> None:
    """The commonest way a mid-sized model loops. The second call is answered from
    the loop instead of being billed again."""
    args = '{"query": "what is in the corpus"}'
    rag = FakeRag()
    llm = scripted_llm(
        [
            _tool_message("retrieve", args),
            _tool_message("retrieve", args, call_id="c2"),
            ["Final answer [1]."],
        ]
    )
    frames = await drain(
        run_agent(question="q", llm=llm, toolbox=ToolBox(rag, QueryParam()), model="m", max_steps=4)
    )
    assert rag.queries == ["what is in the corpus"], "the duplicate must not reach retrieval"
    assert frames[-1]["done"] is True
    assert "Final answer" in frames[-1]["answer"]


@pytest.mark.asyncio
async def test_hitting_the_step_ceiling_answers_instead_of_erroring() -> None:
    """A partial answer that says so beats a failed request."""
    rag = FakeRag()

    async def closing(prompt: str, **kwargs: Any):
        if kwargs.get("tools"):
            return _tool_message("retrieve", '{"query": "c"}', call_id="c3")

        async def _gen():
            yield "Partial answer from what I have."

        return _gen()

    frames = await drain(
        run_agent(
            question="q", llm=closing, toolbox=ToolBox(rag, QueryParam()), model="m", max_steps=2
        )
    )
    assert frames[-1]["stop"]["reason"] == "max_steps"
    assert "Partial answer" in frames[-1]["answer"]


# --------------------------------------------------------------------------- #
# Streaming mechanics
# --------------------------------------------------------------------------- #


@pytest.mark.asyncio
async def test_a_second_hop_reaches_retrieval() -> None:
    """Multi-hop: the loop keeps deciding until a turn asks for no more tools."""
    rag = FakeRag()
    llm = scripted_llm(
        [
            _tool_message("retrieve", '{"query": "first"}'),
            _tool_message("retrieve", '{"query": "second hop"}', call_id="c2"),
            SimpleNamespace(content="ready", tool_calls=None),
            ["Done."],
        ]
    )
    await drain(
        run_agent(question="q", llm=llm, toolbox=ToolBox(rag, QueryParam()), model="m", max_steps=4)
    )
    assert rag.queries == ["first", "second hop"]


@pytest.mark.asyncio
async def test_a_decision_turn_never_reaches_the_user() -> None:
    """Only the closing tools-free call is streamed.

    A decision turn's prose is harmony scaffolding on the models this ships with —
    measured, not assumed: streaming one made `openai/gpt-oss-20b` write a fake
    tool transcript with invented `<<<PASSAGE n>>>` blocks and answer from them.
    """
    rag = FakeRag()
    llm = scripted_llm(
        [
            _tool_message("retrieve", '{"query": "first"}'),
            SimpleNamespace(content="analysis: scaffolding nobody should read", tool_calls=None),
            ["Answer."],
        ]
    )
    frames = await drain(
        run_agent(question="q", llm=llm, toolbox=ToolBox(rag, QueryParam()), model="m", max_steps=4)
    )
    tokens = "".join(f["token"] for f in frames if "token" in f)
    assert tokens == "Answer."
    assert "scaffolding" not in tokens


@pytest.mark.asyncio
async def test_the_closing_call_withholds_the_tools() -> None:
    """Withholding them is what keeps a harmony model in plain prose."""
    seen: list[bool] = []

    async def llm(prompt: str, **kwargs: Any):
        seen.append(bool(kwargs.get("tools")))
        if kwargs.get("tools"):
            if len(seen) == 1:
                return _tool_message("retrieve", '{"query": "x"}')
            return SimpleNamespace(content="ready", tool_calls=None)

        async def _gen():
            yield "Answer."

        return _gen()

    await drain(
        run_agent(question="q", llm=llm, toolbox=ToolBox(FakeRag(), QueryParam()), model="m")
    )
    assert seen[-1] is False, "the answering call must carry no tools"


@pytest.mark.asyncio
async def test_tool_call_frames_report_progress() -> None:
    rag = FakeRag()
    llm = scripted_llm([_tool_message("retrieve", '{"query": "x"}'), ["ok"]])
    frames = await drain(
        run_agent(question="q", llm=llm, toolbox=ToolBox(rag, QueryParam()), model="m")
    )
    calls = [f["tool_call"] for f in frames if "tool_call" in f]
    assert calls and calls[0]["name"] == "retrieve"


# --------------------------------------------------------------------------- #
# Arguments, budget, citations
# --------------------------------------------------------------------------- #


def test_almost_json_arguments_are_repaired() -> None:
    """Weaker models emit trailing commas and unquoted keys. The repo already has
    json-repair for this; not using it would turn a recoverable turn into a crash."""
    assert _parse_arguments('{"query": "a",}') == {"query": "a"}
    assert _parse_arguments('{"query": "a"}') == {"query": "a"}
    assert _parse_arguments("") == {}


@pytest.mark.asyncio
async def test_unparsable_arguments_go_back_to_the_model_as_a_tool_result() -> None:
    """The tool-result channel is the only one a model can read a correction from."""
    result = await ToolBox(FakeRag(), QueryParam()).run("retrieve", "not json at all {{{")
    assert not result.ok
    assert "JSON" in result.text or "required" in result.text


@pytest.mark.asyncio
async def test_an_unknown_tool_name_is_reported_not_raised() -> None:
    result = await ToolBox(FakeRag(), QueryParam()).run("cypher", '{"query": "MATCH (n) RETURN n"}')
    assert not result.ok
    assert "no tool named" in result.text


def test_eviction_replaces_the_oldest_tool_results_only() -> None:
    """The system prompt, the question and the assistant turns carrying tool_calls
    all have to survive: dropping an assistant turn orphans the tool message
    answering it, which most providers reject outright."""
    messages = [
        {"role": "system", "content": "s"},
        {"role": "user", "content": "q"},
        {"role": "assistant", "content": "", "tool_calls": [{"id": "1", "function": {}}]},
        {"role": "tool", "tool_call_id": "1", "content": "x" * 8000},
        {"role": "tool", "tool_call_id": "2", "content": "y" * 8000},
    ]
    report = enforce(messages, budget=1000)
    assert report.applied
    assert messages[0]["role"] == "system" and messages[0]["content"] == "s"
    assert "tool_calls" in messages[2]
    assert messages[3]["content"] != "x" * 8000


def test_multi_hop_fences_continue_the_numbering() -> None:
    """Two hops each starting at [1] put two different [1] markers in one answer."""
    first = _fence_from(1, ["a", "b"], ["x.pdf", "y.pdf"])
    second = _fence_from(3, ["c"], ["z.pdf"])
    assert "PASSAGE 1" in first and "PASSAGE 2" in first
    assert "PASSAGE 3" in second
    assert "PASSAGE 1" not in second


@pytest.mark.asyncio
async def test_references_accumulate_across_hops() -> None:
    rag = FakeRag(docs=["p1", "p2"])
    toolbox = ToolBox(rag, QueryParam())
    await toolbox.run("retrieve", '{"query": "a"}')
    await toolbox.run("retrieve", '{"query": "b"}')
    ids = [ref["reference_id"] for ref in toolbox.references]
    assert ids == ["1", "2", "3", "4"], "the second hop must not restart at 1"


# --------------------------------------------------------------------------- #
# Cost of a deciding turn
# --------------------------------------------------------------------------- #


def test_the_decide_cap_has_a_floor(monkeypatch) -> None:
    """A cap of zero would make every deciding turn emit nothing at all."""
    from memgraphrag.agent.budget import DEFAULT_DECIDE_MAX_TOKENS, decide_max_tokens

    assert decide_max_tokens() == DEFAULT_DECIDE_MAX_TOKENS
    monkeypatch.setenv("AGENT_DECIDE_MAX_TOKENS", "0")
    assert decide_max_tokens() >= 32
    monkeypatch.setenv("AGENT_DECIDE_MAX_TOKENS", "1024")
    assert decide_max_tokens() == 1024


@pytest.mark.asyncio
async def test_deciding_turns_are_capped_and_answering_turns_are_not() -> None:
    """The measured bottleneck, pinned.

    A deciding turn's prose is discarded — only the tool call, or its absence, is
    read — yet uncapped it cost 37 s and 4 606 characters of reasoning on the
    reference corpus, against 2.7 s capped. The answer must stay uncapped: that
    text is the deliverable.
    """
    seen: list[dict[str, Any]] = []

    async def llm(prompt: str, **kwargs: Any):
        seen.append(kwargs)
        if kwargs.get("tools"):
            if len(seen) == 1:
                return _tool_message("retrieve", '{"query": "x"}')
            return SimpleNamespace(content="ready", tool_calls=None)

        async def _gen():
            yield "Answer."

        return _gen()

    await drain(
        run_agent(question="q", llm=llm, toolbox=ToolBox(FakeRag(), QueryParam()), model="m")
    )
    deciding = [call for call in seen if call.get("tools")]
    answering = [call for call in seen if not call.get("tools")]
    assert deciding and all(call.get("max_tokens") for call in deciding)
    assert answering and all(call.get("max_tokens") is None for call in answering)


@pytest.mark.asyncio
async def test_a_choice_object_is_unwrapped_like_a_message() -> None:
    """`return_choice=True` is what makes `finish_reason` readable, so the loop
    receives a choice where it used to receive a message."""
    choice = SimpleNamespace(
        finish_reason="tool_calls", message=_tool_message("retrieve", '{"query": "x"}')
    )
    rag = FakeRag()
    llm = scripted_llm([choice, SimpleNamespace(content="ready", tool_calls=None), ["ok"]])
    await drain(run_agent(question="q", llm=llm, toolbox=ToolBox(rag, QueryParam()), model="m"))
    assert rag.queries == ["x"]


@pytest.mark.asyncio
async def test_a_truncated_deciding_turn_is_reported(caplog, monkeypatch) -> None:
    """Silently losing multi-hop to the cap is the one outcome worth naming."""
    truncated = SimpleNamespace(
        finish_reason="length", message=SimpleNamespace(content="reasoning…", tool_calls=None)
    )
    llm = scripted_llm([_tool_message("retrieve", '{"query": "x"}'), truncated, ["Answer."]])
    # The warning fires once per process, so the flag has to be cleared here or this
    # test passes or fails depending on which tests ran before it.
    monkeypatch.setattr("memgraphrag.agent.loop._warned_about_cap", False)
    with caplog.at_level("WARNING"):
        await drain(
            run_agent(question="q", llm=llm, toolbox=ToolBox(FakeRag(), QueryParam()), model="m")
        )
    assert any("AGENT_DECIDE_MAX_TOKENS" in record.message for record in caplog.records)
