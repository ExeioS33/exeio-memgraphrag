"""The tool-calling loop.

Written by hand rather than on a framework. The research behind this iteration
looked hard at LangChain and LangGraph and found that adopting either means
adopting a second, opaque copy of conversation state — ``PostgresSaver`` owns four
tables with hard-coded names and serialises message lists into BYTEA — while we
already own persistence, streaming and provider routing. What would be left of the
framework is a tool loop, and that is this file.

Two decisions shape everything below.

**Decision turns are never streamed.** Two reasons, and the second was measured
rather than reasoned. First, until a turn ends we do not know whether it produced an
answer or a tool call, and text shown then retracted is worse than a spinner.
Second, and decisively: asked to stream a tool-enabled turn, this repo's default
model (`openai/gpt-oss-20b` through Together) does not emit structured `tool_calls`
deltas at all — it writes the harmony transcript as plain text, invents its own
`<<<PASSAGE n>>>` blocks, and answers from them. The *same* model on the *same*
request answers with a real tool call when the turn is buffered.

**The answer is not written by the loop.** Once the model stops asking for tools,
the passages it gathered are handed to ``render_rag_qa`` — the prompt every other
mode uses — and *that* is what streams. The tool transcript never reaches the
answering call. This is not tidiness: a conversation carrying ``role="tool"``
messages keeps a harmony model in channel mode, so its answer arrives wrapped in
reasoning and, on a bad turn, in passages it made up. Answering from a plain
question-plus-context prompt is the path `mode=ppr` has always taken, and it
produces the same clean, cited prose. What the loop contributes is *which*
passages are in that context — which is the whole point of the mode.

**The first turn is forced.** ``tool_choice`` names ``retrieve`` on the opening
call, for two reasons: a question about the corpus should always be grounded, and a
model that answers a *forced* tool call with prose has just proved it cannot call
tools. That is a runtime fact, where a name-based capability list is a guess.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, AsyncIterator, Callable

from memgraphrag.agent.budget import context_budget, decide_max_tokens, enforce
from memgraphrag.agent.capabilities import precheck_model, unsupported_after_forced_call
from memgraphrag.agent.harmony import HarmonyFilter
from memgraphrag.agent.prompts import render_agent_system
from memgraphrag.agent.tools import ToolBox, tool_specs
from memgraphrag.observability.langfuse_trace import observation, update_observation
from memgraphrag.prompts.templates import render_rag_qa
from memgraphrag.utils.step_log import stage, truncate

logger = logging.getLogger(__name__)

#: Hard ceiling on tool calls executed in one assistant turn. A model may return
#: several; retrieval takes the engine's corpus lock, so they serialise anyway.
MAX_CALLS_PER_TURN = 3

#: The cap warning is worth saying once and then never again. On a model that
#: reasons at length before answering — which is the common case, and the reason the
#: cap exists — it would otherwise fire on every single turn, and a warning that
#: always fires is one nobody reads.
_warned_about_cap = False


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
    """Normalise a provider message (or choice) into a plain assistant turn."""
    inner = getattr(message, "message", None)
    if inner is not None and not isinstance(message, dict):
        message = inner
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
            turn = await _decide(
                llm=llm,
                messages=messages,
                specs=specs,
                model=model,
                provider=provider,
                step=step,
                force_tool=force_tool,
                tokens_in=report.tokens_after,
            )
            calls = turn.get("tool_calls") or []

            if force_tool and not calls:
                # A forced call answered in prose: the model cannot call tools. Say
                # so rather than degrading into an ungrounded answer.
                raise unsupported_after_forced_call(model)

            if not calls:
                # Ready to answer. The turn's own content is discarded: on a
                # harmony-format model it is scaffolding, not prose.
                stop.reason = "answered"
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

        # The answer is a fresh call built from the gathered passages, with no
        # tool messages in it. `fence_passages` numbers them from 1 and the
        # references were numbered the same way as they accumulated, so the [n] the
        # model writes matches the list the UI renders.
        system, user = render_rag_qa(question, toolbox.docs, toolbox.sources)
        if stop.reason == "max_steps":
            # Ceiling reached: answer with what we have rather than erroring. A
            # partial answer that says it is partial beats a failed request.
            user += (
                "\n\nAnswer with these passages only. Say plainly if they are not "
                "enough to answer fully."
            )
        async for token in _final_answer(
            llm=llm,
            system=system,
            user=user,
            history=list(history or []),
            model=model,
            provider=provider,
        ):
            answer_parts.append(token)
            yield {"token": token}

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
) -> dict[str, Any]:
    """One thinking turn, buffered, returning the assistant turn it produced.

    Buffered because a streamed tool-enabled turn is where this model stops
    emitting structured tool calls — see the module docstring. The turn's content is
    never shown to the user: whatever prose a decision turn contains is harmony
    scaffolding, and the answer comes from the tools-free call that follows.
    """
    with observation(
        "memgraphrag.agent.think",
        as_type="generation",
        input={"messages": len(messages)},
        model=model,
        metadata={"step": step, "forced": force_tool, "tokens_in": tokens_in},
    ) as span:
        tool_choice: Any = "auto"
        if force_tool:
            # Forcing the opening call is also the capability check: a model that
            # answers a forced call with prose cannot call tools at all.
            tool_choice = {"type": "function", "function": {"name": "retrieve"}}
        cap = decide_max_tokens()
        choice = await llm(
            "",
            messages=messages,
            tools=specs,
            tool_choice=tool_choice,
            # The turn's prose is discarded either way — only the tool call, or its
            # absence, is read. Uncapped, this model spends half a minute writing
            # reasoning nobody sees.
            max_tokens=cap,
            return_choice=True,
            model=model,
            provider=provider,
            agent="qa.agent",
            llm_action="decide",
        )
        turn = _as_turn(choice)
        finish_reason = getattr(choice, "finish_reason", None)
        global _warned_about_cap
        if finish_reason == "length" and not turn.get("tool_calls") and not _warned_about_cap:
            # The cap cut the turn off before it could ask for a tool. The loop reads
            # that as "ready to answer" and still produces a grounded answer from the
            # passages it already has, so this is a quiet loss of a possible second
            # hop rather than a failure — which is exactly why it is said out loud,
            # once. Every turn's finish_reason is on its span regardless.
            _warned_about_cap = True
            logger.warning(
                "agent: a deciding turn hit AGENT_DECIDE_MAX_TOKENS=%d before emitting a "
                "tool call, so this turn made no second search. Raise it if this model "
                "needs room to reason first. Said once per process; per-turn detail is "
                "on the memgraphrag.agent.think spans.",
                cap,
            )
        update_observation(
            span,
            output=_think_output(turn),
            metadata={"max_tokens": cap, "finish_reason": finish_reason},
        )
        return turn


def _think_output(turn: dict[str, Any]) -> dict[str, Any]:
    return {
        "tool_calls": [c["function"]["name"] for c in turn.get("tool_calls") or []],
        "content": truncate(turn.get("content") or "", 500),
    }


async def _final_answer(
    *,
    llm: Callable[..., Any],
    system: str,
    user: str,
    history: list[dict[str, str]],
    model: str | None,
    provider: str | None,
) -> AsyncIterator[str]:
    """Stream the answer from a plain question-plus-context prompt.

    No ``tools``, and — more importantly — no tool messages: this is an ordinary
    RAG-QA call, identical in shape to the one ``mode=ppr`` makes.
    """
    with observation(
        "memgraphrag.agent.answer",
        as_type="generation",
        input={"question_chars": len(user)},
        model=model,
    ) as span:
        produced = await llm(
            user,
            system_prompt=system,
            history_messages=history or None,
            model=model,
            provider=provider,
            stream=True,
            agent="qa.agent",
            llm_action="answer",
        )
        parts: list[str] = []
        # A prompt with no tool messages keeps this model in plain prose — `ppr`
        # proves it — but the filter costs nothing on a clean stream and saves the
        # answer if a model ever opens a channel here anyway.
        harmony = HarmonyFilter()
        if hasattr(produced, "__aiter__"):
            async for token in produced:
                visible = harmony.feed(str(token))
                if visible:
                    parts.append(visible)
                    yield visible
        else:
            visible = harmony.feed(str(produced))
            if visible:
                parts.append(visible)
                yield visible
        trailing = harmony.flush()
        if trailing:
            parts.append(trailing)
            yield trailing
        update_observation(span, output={"answer": truncate("".join(parts), 2000)})
