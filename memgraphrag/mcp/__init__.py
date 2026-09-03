"""MCP exposure of MemGraphRAG's read surface.

A second package rather than a router: MCP is another way in, not another route.
The tools it serves are the ones defined in :mod:`memgraphrag.agent.tools`, so a
capability is written once and exposed twice.
"""

from memgraphrag.mcp.auth import ApiTokenVerifier
from memgraphrag.mcp.server import MOUNT_PATH, allowed_hosts, build_mcp_server, mcp_enabled

__all__ = [
    "ApiTokenVerifier",
    "MOUNT_PATH",
    "allowed_hosts",
    "build_mcp_server",
    "mcp_enabled",
]
