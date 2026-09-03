"""The shape of the Langfuse trace, not merely the absence of an exception.

Two reasons this asserts nesting rather than "it did not crash".

The first is the regression it exists to prevent. `astream_qa` shipped with no
observation at all: it called `flush_langfuse()` in a `finally` having never opened
a trace, so the retrieval spans created below it arrived orphaned and the generation
— its model, its output, its tokens — was not traced. Nothing raised, nothing
logged, and since the web UI streams exclusively, tracing was dark on the only path
users take. A test that only checked for exceptions would have passed throughout.

The second is that the agent loop is an async generator whose spans straddle
`yield`. Nesting there rests on `contextvars` propagating through a suspended
generator. If that does not hold, the tree flattens silently — the same failure
mode, one layer down.
"""

from __future__ import annotations

from contextlib import contextmanager
from types import SimpleNamespace
from typing import Any

import pytest

from memgraphrag.agent.loop import run_agent
from memgraphrag.agent.tools import ToolBox
from memgraphrag.base import QueryParam
from tests.agent.test_agent_loop import FakeRag, drain, scripted_llm

pytestmark = pytest.mark.offline


class RecordingClient:
    """A Langfuse stand-in that records the tree it is asked to build."""

    def __init__(self) -> None:
        self.events: list[tuple[str, str, tuple[str, ...]]] = []
        self._stack: list[str] = []

    @contextmanager
    def start_as_current_observation(self, **kwargs: Any):
        name = str(kwargs.get("name"))
        self.events.append((name, str(kwargs.get("as_type")), tuple(self._stack)))
        self._stack.append(name)
        try:
            yield SimpleNamespace(update=lambda **_: None)
        finally:
            self._stack.pop()

    def names(self) -> list[str]:
        return [name for name, _, _ in self.events]

    def parents_of(self, name: str) -> tuple[str, ...]:
        for recorded, _, parents in self.events:
            if recorded == name:
                return parents
        raise AssertionError(f"{name} was never opened; opened: {self.names()}")

    def type_of(self, name: str) -> str:
        for recorded, as_type, _ in self.events:
            if recorded == name:
                return as_type
        raise AssertionError(f"{name} was never opened; opened: {self.names()}")


@pytest.fixture()
def recorder(monkeypatch) -> RecordingClient:
    client = RecordingClient()
    monkeypatch.setattr(
        "memgraphrag.observability.langfuse_trace.get_langfuse_client", lambda: client
    )
    return client


def _text_chunk(text: str) -> SimpleNamespace:
    return SimpleNamespace(
        choices=[SimpleNamespace(delta=SimpleNamespace(content=text, tool_calls=None))]
    )


def _tool_message(name: str, arguments: str) -> SimpleNamespace:
    return SimpleNamespace(
        content=None,
        tool_calls=[
            SimpleNamespace(id="c1", function=SimpleNamespace(name=name, arguments=arguments))
        ],
    )


@pytest.mark.asyncio
async def test_the_agent_tree_is_nested_not_flat(recorder: RecordingClient) -> None:
    llm = scripted_llm([_tool_message("retrieve", '{"query": "x"}'), [_text_chunk("Answer.")]])
    await drain(
        run_agent(
            question="q",
            llm=llm,
            toolbox=ToolBox(FakeRag(), QueryParam()),
            model="m",
            max_steps=3,
        )
    )

    names = recorder.names()
    assert "memgraphrag.agent" in names
    assert "memgraphrag.agent.think" in names
    assert "memgraphrag.agent.act" in names

    # The root is a span, the thinking turn is a generation — the distinction is
    # what makes token cost visible in Langfuse rather than merely recorded.
    assert recorder.type_of("memgraphrag.agent") == "span"
    assert recorder.type_of("memgraphrag.agent.think") == "generation"

    # Nesting is the assertion that matters: a flat tree tells you nothing about
    # which retrieval belonged to which decision.
    assert recorder.parents_of("memgraphrag.agent") == ()
    assert recorder.parents_of("memgraphrag.agent.think") == ("memgraphrag.agent",)
    assert recorder.parents_of("memgraphrag.agent.act") == ("memgraphrag.agent",)


@pytest.mark.asyncio
async def test_spans_survive_the_yields_of_an_async_generator(
    recorder: RecordingClient,
) -> None:
    """The loop yields between opening a span and closing it.

    If `contextvars` did not carry through a suspended generator, `act` would open
    at depth 0 instead of under the root and nobody would ever see an error.
    """
    llm = scripted_llm([_tool_message("retrieve", '{"query": "x"}'), [_text_chunk("Answer.")]])
    frames = []
    async for frame in run_agent(
        question="q", llm=llm, toolbox=ToolBox(FakeRag(), QueryParam()), model="m"
    ):
        frames.append(frame)
        # Consuming one frame at a time is exactly what the SSE route does.
        assert isinstance(frame, dict)

    depths = {name: len(parents) for name, _, parents in recorder.events}
    assert depths["memgraphrag.agent"] == 0
    assert depths["memgraphrag.agent.act"] == 1
