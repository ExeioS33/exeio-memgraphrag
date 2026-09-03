"""Agentic query loop: the model decides when and what to retrieve.

Kept out of ``api/`` on purpose — this is a capability of the engine, not a route.
``tools.py`` is imported by ``memgraphrag/mcp/`` as well, so a tool is defined once
and exposed twice: in-process to this loop, and over MCP to third-party clients.
"""

from memgraphrag.agent.capabilities import AgentUnsupportedModelError, precheck_model
from memgraphrag.agent.loop import AgentStop, run_agent
from memgraphrag.agent.tools import ToolBox, ToolResult, tool_specs

__all__ = [
    "AgentStop",
    "AgentUnsupportedModelError",
    "ToolBox",
    "ToolResult",
    "precheck_model",
    "run_agent",
    "tool_specs",
]
