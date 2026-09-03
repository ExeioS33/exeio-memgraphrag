"""The tool-calling loop.

Written by hand rather than on a framework. The research behind this iteration
looked hard at LangChain and LangGraph and found that adopting either means
adopting a second, opaque copy of conversation state — ``PostgresSaver`` owns four
tables with hard-coded names and serialises message lists into BYTEA — while we
already own persistence, streaming and provider routing. What would be left of the
framework is a tool loop, and that is this file.

Two decisions shape everything below.

**Decision turns commit before they stream.** Until a turn has emitted its first
delta we do not know whether it will produce an answer or a tool call, and text
shown then retracted is worse than a spinner. Content is forwarded only once the
turn has committed to being prose; a turn that emitted a tool-call delta forwards
nothing at all.

**The first turn is forced.** ``tool_choice`` names ``retrieve`` on the opening
call, for two reasons: a question about the corpus should always be grounded, and a
model that answers a *forced* tool call with prose has just proved it cannot call
tools. That is a runtime fact, where a name-based capability list is a guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from memgraphrag.agent.budget import context_budget, enforce
from memgraphrag.agent.capabilities import precheck_model, unsupported_after_forced_call
from memgraphrag.agent.prompts import render_agent_system
from memgraphrag.agent.tools import ToolBox, tool_specs
from memgraphrag.observability.langfuse_trace import observation, update_observation
from memgraphrag.utils.step_log import stage, truncate

logger = logging.getLogger(__name__)

#: Hard ceiling on tool calls executed in one assistant turn. A model may return
#: several; retrieval takes the engine's corpus lock, so they serialise anyway.
MAX_CALLS_PER_TURN = 3


@dataclass
class AgentStop:
    """Why the loop ended — carried into the trace and the final frame."""

    reason: str
    steps: int
    tool_calls: int

    def to_dict(self) -> dict[str, Any]:
        return {"reason": self.reason, "steps": self.steps, "tool_calls": self.tool_calls}


def _tool_call_key(call: dict[str, Any]) -> str:
    function = call.get("function") or {}
    return f"{function.get('name')}::{(function.get('arguments') or '').strip()}"


def _as_turn(message: Any) -> dict[str, Any]:
    """Normalise a provider message object into a plain assistant turn."""
    if isinstance(message, dict):
        content = message.get("content")
        raw_calls = message.get("tool_calls") or []
    else:
        content = getattr(message, "content", None)
        raw_calls = getattr(message, "tool_calls", None) or []
    calls: list[dict[str, Any]] = []
    for call in raw_calls:
        if isinstance(call, dict):
            function = call.get("function") or {}
            name = function.get("name") or ""
            arguments = function.get("arguments") or ""
            call_id = call.get("id")
        else:
            function = getattr(call, "function", None)
            name = getattr(function, "name", "") or ""
            arguments = getattr(function, "arguments", "") or ""
            call_id = getattr(call, "id", None)
        calls.append(
            {
                "id": call_id or f"call_{len(calls)}",
                "type": "function",
                "function": {"name": name, "arguments": arguments},
            }
        )
    turn: dict[str, Any] = {"role": "assistant", "content": content or ""}
    if calls:
        turn["tool_calls"] = calls
    return turn


def _fold_tool_call_deltas(chunk: Any, accumulator: dict[int, dict[str, Any]]) -> bool:
    """Fold one streaming chunk's tool-call deltas into ``accumulator``.

    Tool calls arrive in pieces: an index and an id first, then the function name,
    then the arguments a few characters at a time. Nothing is executable until the
    stream ends, which is the mechanical reason a decision turn cannot be streamed
    straight through to the browser.
    """
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return False
    delta = getattr(choices[0], "delta", None)
    deltas = getattr(delta, "tool_calls", None) if delta is not None else None
    if not deltas:
        return False
    for piece in deltas:
        index = getattr(piece, "index", 0) or 0
        slot = accumulator.setdefault(
            index,
            {"id": None, "type": "function", "function": {"name": "", "arguments": ""}},
        )
        if getattr(piece, "id", None):
            slot["id"] = piece.id
        function = getattr(piece, "function", None)
        if function is not None:
            if getattr(function, "name", None):
                slot["function"]["name"] += function.name
            if getattr(function, "arguments", None):
                slot["function"]["arguments"] += function.arguments
    return True


def _chunk_text(chunk: Any) -> str:
    choices = getattr(chunk, "choices", None) or []
    if not choices:
        return ""
    delta = getattr(choices[0], "delta", None)
    return (getattr(delta, "content", None) if delta is not None else None) or ""


async def run_agent(
    *,
    question: str,
    llm: Callable[..., Any],
    toolbox: ToolBox,
    model: str | None = None,
    provider: str | None = None,
    history: list[dict[str, str]] | None = None,
    max_steps: int = 4,
    language: str | None = None,
) -> AsyncIterator[dict[str, Any]]:
    """Run the loop, yielding UI frames.

    Frames: ``{"tool_call": {...}}`` while the loop works, one ``{"references": [...]}``
    once retrieval has produced any, ``{"token": "..."}`` for the answer, and a final
    ``{"done": True, "answer": ..., "stop": {...}}``.

    ``llm`` is the engine's ``llm_model_func``, so provider routing, structured
    logging and the corporate-CA HTTP client all still apply — calling the OpenAI
    SDK directly here would silently drop the per-request provider selection the web
    UI depends on.
    """
    precheck_model(model)

    messages: list[dict[str, Any]] = [{"role": "system", "content": render_agent_system(language)}]
    messages.extend(history or [])
    messages.append({"role": "user", "content": question})

    specs = tool_specs()
    seen_calls: set[str] = set()
    answer_parts: list[str] = []
    references_sent = False
    stop = AgentStop(reason="max_steps", steps=0, tool_calls=0)

    with observation(
        "memgraphrag.agent",
        as_type="span",
        input={"question": truncate(question, 400), "history_turns": len(history or [])},
        metadata={
            "mode": "agent",
            "provider": provider,
            "model": model,
            "max_steps": max_steps,
        },
    ) as root_span:
        answered = False
        for step in range(max_steps):
            stop.steps = step + 1
            report = enforce(messages, context_budget())
            if report.applied:
                stage(
                    logger,
                    "agent.budget.evict",
                    evicted=report.evicted,
                    tokens_before=report.tokens_before,
                    tokens_after=report.tokens_after,
                )

            force_tool = step == 0
            turn: dict[str, Any] = {}
            streamed = False
            async for event in _decide(
                llm=llm,
                messages=messages,
                specs=specs,
                model=model,
                provider=provider,
                step=step,
                force_tool=force_tool,
                tokens_in=report.tokens_after,
            ):
                if "token" in event:
                    answer_parts.append(event["token"])
                    yield event
                else:
                    turn = event["turn"]
                    streamed = event["streamed"]

            calls = turn.get("tool_calls") or []

            if force_tool and not calls:
                # A forced call answered in prose: the model cannot call tools. Say
                # so rather than degrading into an ungrounded answer.
                raise unsupported_after_forced_call(model)

            if not calls:
                if not streamed:
                    text = str(turn.get("content") or "")
                    if text:
                        answer_parts.append(text)
                        yield {"token": text}
                stop.reason = "answered"
                answered = True
                break

            messages.append(turn)
            for call in calls[:MAX_CALLS_PER_TURN]:
                function = call["function"]
                name = function["name"]
                key = _tool_call_key(call)
                if key in seen_calls:
                    # The most common way a mid-sized model loops: the same call,
                    # verbatim, forever. Answer it from here rather than paying for
                    # the round trip again.
                    messages.append(
                        {
                            "role": "tool",
                            "tool_call_id": call["id"],
                            "content": (
                                "You already ran this exact call. Use the passages you "
                                "have, or search with different terms."
                            ),
                        }
                    )
                    continue
                seen_calls.add(key)

                yield {
                    "tool_call": {
                        "name": name,
                        "arguments": function["arguments"],
                        "step": step,
                    }
                }
                with observation(
                    "memgraphrag.agent.act",
                    as_type="span",
                    input={"tool": name, "arguments": function["arguments"]},
                    metadata={"step": step},
                ) as act_span:
                    result = await toolbox.run(name, function["arguments"])
                    update_observation(
                        act_span,
                        output={"ok": result.ok, "text": truncate(result.text, 2000)},
                    )
                stop.tool_calls += 1
                messages.append(
                    {"role": "tool", "tool_call_id": call["id"], "content": result.text}
                )

            if toolbox.references and not references_sent:
                references_sent = True
                yield {"references": list(toolbox.references)}

        if not answered:
            # Iteration ceiling: answer with what we have rather than erroring. A
            # partial answer that says it is partial beats a failed request.
            enforce(messages, context_budget())
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "Answer now with the passages you already retrieved. Say plainly "
                        "if they are not enough to answer fully."
                    ),
                }
            )
            async for token in _final_answer(
                llm=llm, messages=messages, model=model, provider=provider
            ):
                answer_parts.append(token)
                yield {"token": token}
            stop.reason = "max_steps"

        if toolbox.references and not references_sent:
            references_sent = True
            yield {"references": list(toolbox.references)}
        elif not references_sent:
            yield {"references": []}

        answer = "".join(answer_parts)
        update_observation(
            root_span,
            output={"answer": truncate(answer, 2000), **stop.to_dict()},
        )

    yield {"done": True, "answer": answer, "stop": stop.to_dict()}


async def _decide(
    *,
    llm: Callable[..., Any],
    messages: list[dict[str, Any]],
    specs: list[dict[str, Any]],
    model: str | None,
    provider: str | None,
    step: int,
    force_tool: bool,
    tokens_in: int | None,
) -> AsyncIterator[dict[str, Any]]:
    """One thinking turn.

    Yields ``{"token": …}`` for text the turn has committed to, then exactly one
    ``{"turn": …, "streamed": …}``. The forced opening turn is buffered — it is a
    tool call by construction, so there is nothing to stream.
    """
    with observation(
        "memgraphrag.agent.think",
        as_type="generation",
        input={"messages": len(messages)},
        model=model,
        metadata={"step": step, "forced": force_tool, "tokens_in": tokens_in},
    ) as span:
        if force_tool:
            message = await llm(
                "",
                messages=messages,
                tools=specs,
                tool_choice={"type": "function", "function": {"name": "retrieve"}},
                model=model,
                provider=provider,
                agent="qa.agent",
                llm_action="decide",
            )
            turn = _as_turn(message)
            update_observation(span, output=_think_output(turn, streamed=False))
            yield {"turn": turn, "streamed": False}
            return

        produced = await llm(
            "",
            messages=messages,
            tools=specs,
            tool_choice="auto",
            model=model,
            provider=provider,
            stream=True,
            raw_stream=True,
            agent="qa.agent",
            llm_action="decide",
        )

        if not hasattr(produced, "__aiter__"):
            # An llm_model_func that ignores `stream` (a test double, or a binding
            # without streaming) still works; it just arrives in one piece.
            turn = _as_turn(produced)
            update_observation(span, output=_think_output(turn, streamed=False))
            yield {"turn": turn, "streamed": False}
            return

        accumulator: dict[int, dict[str, Any]] = {}
        parts: list[str] = []
        committed_to_text = False
        async for chunk in produced:
            if _fold_tool_call_deltas(chunk, accumulator) and not committed_to_text:
                continue
            text = _chunk_text(chunk)
            if not text:
                continue
            if not accumulator:
                committed_to_text = True
            parts.append(text)
            if committed_to_text:
                yield {"token": text}

        if committed_to_text and accumulator:
            # A provider that emits prose and *then* a tool call. Keeping the prose
            # and dropping the call is the only outcome that never shows the user
            # text and then takes it back.
            logger.warning("agent: dropped %d late tool call(s) after text", len(accumulator))
            accumulator = {}

        turn = {"role": "assistant", "content": "".join(parts)}
        if accumulator:
            turn["tool_calls"] = [
                {
                    "id": slot["id"] or f"call_{index}",
                    "type": "function",
                    "function": {
                        "name": slot["function"]["name"],
                        "arguments": slot["function"]["arguments"],
                    },
                }
                for index, slot in sorted(accumulator.items())
            ]
        update_observation(span, output=_think_output(turn, streamed=committed_to_text))
        yield {"turn": turn, "streamed": committed_to_text}


def _think_output(turn: dict[str, Any], *, streamed: bool) -> dict[str, Any]:
    return {
        "tool_calls": [c["function"]["name"] for c in turn.get("tool_calls") or []],
        "content": truncate(turn.get("content") or "", 500),
        "streamed": streamed,
    }


async def _final_answer(
    *,
    llm: Callable[..., Any],
    messages: list[dict[str, Any]],
    model: str | None,
    provider: str | None,
) -> AsyncIterator[str]:
    """Stream a closing answer with tools withheld, so it cannot call one."""
    with observation(
        "memgraphrag.agent.answer",
        as_type="generation",
        input={"messages": len(messages)},
        model=model,
    ) as span:
        produced = await llm(
            "",
            messages=messages,
            model=model,
            provider=provider,
            stream=True,
            agent="qa.agent",
            llm_action="answer",
        )
        parts: list[str] = []
        if hasattr(produced, "__aiter__"):
            async for token in produced:
                text = str(token)
                if text:
                    parts.append(text)
                    yield text
        else:
            text = str(produced)
            if text:
                parts.append(text)
                yield text
        update_observation(span, output={"answer": truncate("".join(parts), 2000)})
