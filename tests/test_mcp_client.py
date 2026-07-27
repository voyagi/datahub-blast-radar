"""Tests for the MCP session wrapper.

A fake session stands in for the server so the result-parsing and error paths
are covered without a DataHub instance. Parsing is where an integration like
this usually breaks quietly: a tool returns text instead of structured content,
or reports failure in a field nobody checks.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pytest

from blast_radar.config import Settings
from blast_radar.mcp_client import (
    DataHubMCP,
    McpToolError,
    McpToolMissingError,
    ToolSpec,
    _parse_content,
    _render_content,
    _server_errlog,
)


@dataclass
class FakeTextBlock:
    text: str


# Field names mirror the MCP wire format rather than Python casing, because
# the wrapper reads them off the real result object by those exact names.
@dataclass
class FakeResult:
    content: list[Any] = field(default_factory=list)
    isError: bool = False
    structuredContent: Any = None


class FakeSession:
    def __init__(self, result: FakeResult) -> None:
        self.result = result
        self.calls: list[tuple[str, dict[str, Any]]] = []

    async def call_tool(self, name: str, arguments: dict[str, Any]) -> FakeResult:
        self.calls.append((name, arguments))
        return self.result


SEARCH_SPEC = ToolSpec(
    name="search",
    description="Search DataHub",
    input_schema={
        "properties": {"query": {"type": "string"}, "num_results": {"type": "integer"}},
        "required": ["query"],
    },
)


def build(result: FakeResult, tools: dict[str, ToolSpec] | None = None):
    session = FakeSession(result)
    return DataHubMCP(session, tools or {"search": SEARCH_SPEC}), session


@pytest.mark.anyio
async def test_structured_content_is_preferred_over_text():
    client, _ = build(
        FakeResult(content=[FakeTextBlock('{"ignored": true}')], structuredContent={"hits": 3})
    )

    assert await client.call("search", {"query": "orders"}) == {"hits": 3}


@pytest.mark.anyio
async def test_json_text_blocks_are_decoded():
    client, _ = build(FakeResult(content=[FakeTextBlock('{"hits": 2}')]))

    assert await client.call("search", {"query": "orders"}) == {"hits": 2}


@pytest.mark.anyio
async def test_non_json_text_survives_as_a_string():
    client, _ = build(FakeResult(content=[FakeTextBlock("no results found")]))

    assert await client.call("search", {"query": "orders"}) == "no results found"


@pytest.mark.anyio
async def test_multiple_blocks_come_back_as_a_list():
    client, _ = build(FakeResult(content=[FakeTextBlock('{"a": 1}'), FakeTextBlock('{"b": 2}')]))

    assert await client.call("search", {"query": "orders"}) == [{"a": 1}, {"b": 2}]


@pytest.mark.anyio
async def test_server_side_failure_raises_with_the_server_text():
    client, _ = build(FakeResult(content=[FakeTextBlock("entity not found")], isError=True))

    with pytest.raises(McpToolError, match="entity not found"):
        await client.call("search", {"query": "nope"})


@pytest.mark.anyio
async def test_calling_an_absent_tool_names_what_is_available():
    client, session = build(FakeResult())

    with pytest.raises(McpToolMissingError) as excinfo:
        await client.call("add_tags", {})

    message = str(excinfo.value)
    assert "add_tags" in message
    assert "search" in message
    assert "TOOLS_IS_MUTATION_ENABLED" in message
    assert session.calls == []  # never reached the wire


def test_tool_spec_exposes_required_and_optional_arguments():
    assert SEARCH_SPEC.required_arguments() == ["query"]
    assert SEARCH_SPEC.argument_names() == ["num_results", "query"]


def test_tool_spec_handles_a_schema_with_no_properties():
    spec = ToolSpec(name="get_me", description="", input_schema={})

    assert spec.required_arguments() == []
    assert spec.argument_names() == []


def test_render_content_falls_back_when_a_block_has_no_text():
    assert "no error detail" in _render_content([])
    # A block with no `text` still has to render as something identifiable,
    # since this string is the only detail an McpToolError carries.
    rendered = _render_content([object()])
    assert rendered
    assert "object" in rendered


def test_parse_content_returns_none_for_empty_content():
    assert _parse_content([]) is None


def test_content_carrying_no_text_at_all_parses_to_none():
    """An image or resource block is not a payload this tool can read, and
    guessing one from it would be worse than saying there was none."""
    assert _parse_content([object(), object()]) is None


def test_the_session_reports_what_the_server_advertised():
    datahub, _ = build(FakeResult())

    assert datahub.tools == {"search": SEARCH_SPEC}
    assert datahub.has_tool("search") is True
    assert datahub.has_tool("save_document") is False


def test_the_server_log_goes_to_a_file_that_is_not_the_terminal(tmp_path):
    """The DataHub MCP server logs every GraphQL query it sends at DEBUG.

    That is unreadable inline and it is the one place a request header could
    end up on disk, so it lands in a directory the repo ignores rather than in
    the developer's scrollback. The directory is created here because the
    server writes to it before anything else has reason to.
    """
    log = tmp_path / "nested" / "mcp-server.log"

    with _server_errlog(settings_with(server_log_path=log, debug=False)) as handle:
        handle.write("query { me }")

    assert log.read_text(encoding="utf-8") == "query { me }"


def test_debug_mode_sends_the_server_log_to_stderr_instead(tmp_path):
    log = tmp_path / "mcp-server.log"

    with _server_errlog(settings_with(server_log_path=log, debug=True)) as handle:
        assert handle is sys.stderr

    assert not log.exists()


def settings_with(*, server_log_path: Path, debug: bool) -> Settings:
    return Settings(
        gms_url="http://localhost:8080",
        gms_token=None,
        mcp_command="uvx",
        mcp_args=("mcp-server-datahub==0.6.0",),
        mutations_enabled=False,
        max_hops=5,
        max_nodes=400,
        results_per_hop=100,
        debug=debug,
        server_log_path=server_log_path,
    )


@pytest.fixture
def anyio_backend():
    return "asyncio"
