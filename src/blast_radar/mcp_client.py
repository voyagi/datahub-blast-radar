"""Thin async wrapper around the official DataHub MCP server.

Blast Radar deliberately talks to DataHub through `mcp-server-datahub` rather
than through the REST/GraphQL API. That is the point of the project: every
piece of context the agent reasons about, and every write it makes back, goes
over the same MCP surface an agent platform would use.

Verified against mcp 1.28.1: the stable client entrypoints are
`mcp.client.stdio.stdio_client` plus `mcp.ClientSession`.
"""

from __future__ import annotations

import json
import sys
from collections.abc import AsyncIterator, Iterator
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from typing import Any, Protocol, TextIO

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from .config import Settings


class ToolCaller(Protocol):
    """The only thing this wrapper needs from an MCP session.

    Narrower than ClientSession on purpose: the wrapper calls exactly one
    method, so depending on the whole session type would be a lie about the
    coupling and would force tests to build a real transport to exercise
    result parsing.
    """

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> Any: ...


class McpToolError(RuntimeError):
    """A tool call reached the server but the server reported failure."""


class McpToolMissingError(McpToolError):
    """The server does not expose a tool this run needs.

    Raised as its own type because the fix is operational (enable mutations,
    upgrade the server) rather than a bug in the caller.
    """


@dataclass(frozen=True)
class ToolSpec:
    """What the server says a tool is, as advertised by `tools/list`."""

    name: str
    description: str
    input_schema: dict[str, Any]

    def required_arguments(self) -> list[str]:
        required = self.input_schema.get("required", [])
        return [str(item) for item in required]

    def argument_names(self) -> list[str]:
        return sorted(self.input_schema.get("properties", {}))


class DataHubMCP:
    """An open MCP session against DataHub."""

    def __init__(self, session: ToolCaller, tools: dict[str, ToolSpec]) -> None:
        self._session = session
        self._tools = tools

    @property
    def tools(self) -> dict[str, ToolSpec]:
        return self._tools

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def require_tool(self, name: str) -> ToolSpec:
        try:
            return self._tools[name]
        except KeyError:
            raise McpToolMissingError(
                f"The DataHub MCP server does not expose '{name}'. "
                f"Available tools: {', '.join(sorted(self._tools)) or '(none)'}. "
                "Write tools need TOOLS_IS_MUTATION_ENABLED=true and "
                "mcp-server-datahub v0.5.0+."
            ) from None

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        """Call a tool and return its parsed payload.

        MCP tool results arrive as content blocks. Structured content is
        preferred when the server provides it; otherwise text blocks are
        JSON-decoded when possible and returned raw when not.
        """
        self.require_tool(name)
        result = await self._session.call_tool(name, arguments)

        if result.isError:
            raise McpToolError(f"{name} failed: {_render_content(result.content)}")

        structured = getattr(result, "structuredContent", None)
        if structured:
            return structured
        return _parse_content(result.content)


@asynccontextmanager
async def open_datahub(settings: Settings) -> AsyncIterator[DataHubMCP]:
    """Start the MCP server as a subprocess and yield a ready session."""
    params = StdioServerParameters(
        command=settings.mcp_command,
        args=list(settings.mcp_args),
        env=settings.mcp_env(),
    )
    # The server logs every GraphQL query it sends to stderr at DEBUG. Useful
    # when something is wrong, unreadable when it is not, so it goes to a log
    # file unless BLAST_RADAR_DEBUG asks for it inline.
    with _server_errlog(settings) as errlog:
        async with stdio_client(params, errlog=errlog) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                listed = await session.list_tools()
                tools = {
                    tool.name: ToolSpec(
                        name=tool.name,
                        description=tool.description or "",
                        input_schema=tool.inputSchema or {},
                    )
                    for tool in listed.tools
                }
                yield DataHubMCP(session, tools)


@contextmanager
def _server_errlog(settings: Settings) -> Iterator[TextIO]:
    if settings.debug:
        yield sys.stderr
        return

    path = settings.server_log_path
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        yield handle


def _render_content(content: Any) -> str:
    parts: list[str] = []
    for block in content or []:
        text = getattr(block, "text", None)
        parts.append(text if text is not None else repr(block))
    return " ".join(parts) or "no error detail returned"


def _parse_content(content: Any) -> Any:
    """Return structured data from text blocks when the server encodes JSON."""
    payloads: list[Any] = []
    for block in content or []:
        text = getattr(block, "text", None)
        if text is None:
            continue
        try:
            payloads.append(json.loads(text))
        except (TypeError, ValueError):
            payloads.append(text)

    if not payloads:
        return None
    if len(payloads) == 1:
        return payloads[0]
    return payloads
