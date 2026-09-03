"""Tool definitions shared by the in-process agent loop and the MCP server.

One implementation, two exposure surfaces. The bodies are thin — ``retrieve`` is
``(await rag.aretrieve(q, param))[0]`` plus fencing — because the retrieval work
already exists; what lives here is the schema, the argument validation and the
fencing that makes a tool result safe to hand back to a model.

Note which tool is *absent* from :data:`AGENT_TOOL_NAMES`: ``cypher``. A retrieval
tool takes a search string, so a poisoned passage that steers it stays bounded by
what the corpus can answer. A Cypher tool takes graph code, and a passage that
steers *that* is not bounded at all. Exposing it over MCP to a human-driven client
is a different threat model from exposing it to a loop that reads the corpus.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any

from memgraphrag.base import QueryParam
from memgraphrag.prompts.templates import fence_passages
from memgraphrag.utils.json_llm import extract_json_object
from memgraphrag.utils.misc import QuerySolution

# Bound on what one tool call may pull back. `top_k` is a model-supplied argument,
# and an unbounded one turns a single call into a context-window failure.
MAX_TOOL_TOP_K = 30

RETRIEVE_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "retrieve",
        "description": (
            "Search the knowledge graph built from the ingested documents and return "
            "the passages that best answer a question. Pass a standalone question: "
            "the search does not see the conversation, so resolve pronouns and "
            "references yourself before calling (write 'the second obligation of the "
            "association', not 'and the second one?')."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "A self-contained question or search phrase.",
                },
                "top_k": {
                    "type": "integer",
                    "description": (
                        f"How many passages to return (1-{MAX_TOOL_TOP_K}). "
                        "Leave unset for the server default."
                    ),
                },
            },
            "required": ["query"],
        },
    },
}

SEARCH_DOCUMENTS_TOOL: dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "search_documents",
        "description": (
            "List the ingested documents whose name matches a term. Use it to find "
            "out what the corpus contains, not to read a document's content."
        ),
        "parameters": {
            "type": "object",
            "properties": {
                "term": {"type": "string", "description": "Substring to match on file names."}
            },
            "required": ["term"],
        },
    },
}

#: Tools the in-process loop may call. Deliberately narrower than the MCP surface.
AGENT_TOOL_NAMES = ("retrieve", "search_documents")

_ALL_TOOLS = {
    "retrieve": RETRIEVE_TOOL,
    "search_documents": SEARCH_DOCUMENTS_TOOL,
}


def tool_specs(names: tuple[str, ...] = AGENT_TOOL_NAMES) -> list[dict[str, Any]]:
    """Return the OpenAI ``tools`` payload for ``names``."""
    return [_ALL_TOOLS[name] for name in names if name in _ALL_TOOLS]


@dataclass
class ToolResult:
    """What a tool hands back to the model, and what the caller needs besides.

    ``text`` is fenced and safe to place in a ``role="tool"`` message. ``solution``
    carries the retrieval result so the caller can merge its references into the
    turn's citation list — the model never sees that object.
    """

    text: str
    solution: QuerySolution | None = None
    error: str | None = None
    arguments: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None


class ToolBox:
    """Executes tool calls against a live engine.

    ``fence_start`` is the running citation offset for the turn. ``fence_passages``
    restarts at 1 on every call and ``ensure_references`` used to renumber per
    solution, so two retrievals in one turn produced two ``[1]`` markers and the
    numbers in the prose stopped matching the list under it. Numbering continues
    across calls here, which is the only place that can know the running total.
    """

    def __init__(self, rag: Any, base_param: QueryParam | None = None) -> None:
        self._rag = rag
        self._base = base_param or QueryParam()
        self.fence_start = 1
        self.references: list[dict[str, Any]] = []
        # The passages themselves, in citation order. The loop decides *what* to
        # retrieve; the answer is then written from this union by the same RAG-QA
        # prompt every other mode uses, which is what keeps the two comparable.
        self.docs: list[str] = []
        self.sources: list[str] = []

    async def run(self, name: str, raw_arguments: str | dict[str, Any] | None) -> ToolResult:
        """Validate arguments, dispatch, and never raise into the loop."""
        arguments = _parse_arguments(raw_arguments)
        if arguments is None:
            return ToolResult(
                text=(
                    "Tool call failed: the arguments were not valid JSON. Re-issue the "
                    "call with a well-formed JSON object."
                ),
                error="bad-arguments",
            )
        if name == "retrieve":
            return await self._retrieve(arguments)
        if name == "search_documents":
            return await self._search_documents(arguments)
        return ToolResult(
            text=f"Tool call failed: no tool named {name!r}.",
            error="unknown-tool",
            arguments=arguments,
        )

    async def _retrieve(self, arguments: dict[str, Any]) -> ToolResult:
        query = str(arguments.get("query") or "").strip()
        if not query:
            return ToolResult(
                text="Tool call failed: 'query' is required and must be a non-empty string.",
                error="bad-arguments",
                arguments=arguments,
            )
        param = _derive_param(self._base, arguments)
        try:
            solutions = await self._rag.aretrieve(query, param=param)
        except Exception as exc:  # surfaced to the model, not to the user
            return ToolResult(
                text=f"Tool call failed: retrieval error ({type(exc).__name__}: {exc}).",
                error="tool-error",
                arguments=arguments,
            )
        solution = solutions[0] if solutions else None
        if solution is None or not solution.docs:
            return ToolResult(
                text="No passage in the corpus matched that query.",
                solution=solution,
                arguments=arguments,
            )

        # Continue the turn's numbering rather than restarting at 1. `aretrieve`
        # has already numbered this solution from 1, so the rebuild is required:
        # `ensure_references` would hand back that original list and the second hop
        # would cite `[1]` again.
        refs = solution.build_references(start=self.fence_start)
        body = _fence_from(self.fence_start, solution.docs, solution.sources)
        self.fence_start += len(solution.docs)
        self.references.extend(refs)
        self.docs.extend(solution.docs)
        self.sources.extend(list(solution.sources or []))
        return ToolResult(text=body, solution=solution, arguments=arguments)

    async def _search_documents(self, arguments: dict[str, Any]) -> ToolResult:
        term = str(arguments.get("term") or "").strip().lower()
        try:
            records = await self._rag._doc_status_all()
        except Exception as exc:
            return ToolResult(
                text=f"Tool call failed: document listing error ({type(exc).__name__}: {exc}).",
                error="tool-error",
                arguments=arguments,
            )
        names: list[str] = []
        for record in (records or {}).values():
            path = str((record or {}).get("file_path") or "")
            if not path:
                continue
            if term and term not in path.lower():
                continue
            names.append(path)
        if not names:
            return ToolResult(
                text="No ingested document matched that term.",
                arguments=arguments,
            )
        listing = "\n".join(f"- {name}" for name in sorted(set(names))[:50])
        return ToolResult(text=f"Ingested documents:\n{listing}", arguments=arguments)


def _fence_from(start: int, docs: list[str], sources: list[str] | None) -> str:
    """Fence ``docs`` with markers numbered from ``start``.

    ``fence_passages`` always starts at 1; padding the lists is cheaper and less
    fragile than duplicating its escaping rules with an offset parameter.
    """
    if start <= 1:
        return fence_passages(docs, sources)
    pad = start - 1
    padded_docs = [""] * pad + list(docs)
    padded_sources = [""] * pad + list(sources or [])
    fenced = fence_passages(padded_docs, padded_sources)
    blocks = fenced.split("\n\n")
    return "\n\n".join(blocks[pad:])


def _derive_param(base: QueryParam, arguments: dict[str, Any]) -> QueryParam:
    """Copy the request's parameters, overriding only what the tool may set.

    Everything else — provider, model, damping, thresholds — stays as the caller
    configured it. A tool argument may narrow the result size; it may not re-point
    the request at a different provider.
    """
    from dataclasses import replace

    top_k = base.top_k
    raw = arguments.get("top_k")
    if raw is not None:
        try:
            top_k = max(1, min(MAX_TOOL_TOP_K, int(raw)))
        except (TypeError, ValueError):
            top_k = base.top_k
    return replace(base, mode="ppr", top_k=top_k, stream=False, conversation_history=[])


def _parse_arguments(raw: str | dict[str, Any] | None) -> dict[str, Any] | None:
    """Best-effort JSON arguments.

    Weaker models emit trailing commas and unescaped quotes in ``arguments``; the
    repo already carries ``json-repair`` for exactly that, through
    ``extract_json_object``. A repair that still fails returns ``None`` and the
    error goes back to the model as a tool result, which is the only channel it can
    read a correction from.
    """
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return raw
    text = str(raw).strip()
    if not text:
        return {}
    try:
        parsed = json.loads(text)
    except (ValueError, TypeError):
        parsed = extract_json_object(text)
    return parsed if isinstance(parsed, dict) else None
