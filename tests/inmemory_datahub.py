"""An in-memory stand-in for the half of DataHub this tool actually touches.

Not a mock. It answers the six MCP tools the code calls, in the payload shapes
a real v1.6.0 instance returns, and it keeps the writes so a test can assert
what landed in the catalog. That makes an end-to-end test possible without a
running DataHub, which is the property the whole layering exists to protect:
the walk takes a fetch callable, scoring takes plain data, and write-back
returns intents.

Shapes here are the contract with `datahub_adapter`. If a real response ever
stops matching one of these, the adapter tests are what should catch it - this
file is the fixture, not the specification.
"""

from __future__ import annotations

from typing import Any

from blast_radar.mcp_client import McpToolError, ToolSpec

DEFAULT_SCHEMA = {
    "properties": {"urn": {"type": "string"}, "count": {"type": "integer"}},
    "required": ["urn"],
}

READ_TOOLS = ("get_entities", "get_lineage", "get_dataset_queries")
WRITE_TOOLS = (
    "add_tags",
    "remove_tags",
    "add_structured_properties",
    "save_document",
    "search_documents",
)


def asset(
    urn: str,
    *,
    name: str | None = None,
    owners: tuple[str, ...] = (),
    tags: tuple[str, ...] = (),
    terms: tuple[str, ...] = (),
    description: str | None = None,
) -> dict[str, Any]:
    """One entity, in the shape `get_entities` returns it."""
    payload: dict[str, Any] = {
        "urn": urn,
        "name": name or urn.rsplit(":", 1)[-1].strip("()"),
        "ownership": {
            "owners": [
                {"owner": {"urn": f"urn:li:corpuser:{owner}", "properties": {"displayName": owner}}}
                for owner in owners
            ]
        },
        "tags": {"tags": [{"tag": {"urn": f"urn:li:tag:{tag}"}} for tag in tags]},
        "glossaryTerms": {"terms": [{"term": {"urn": f"urn:li:glossaryTerm:{t}"}} for t in terms]},
    }
    if description:
        payload["properties"] = {"description": description}
    return payload


class InMemoryDataHub:
    """A DataHub MCP session backed by dictionaries."""

    def __init__(
        self,
        *,
        entities: dict[str, dict[str, Any]] | None = None,
        lineage: dict[str, list[str]] | None = None,
        queries: dict[str, int] | None = None,
        tool_names: tuple[str, ...] = READ_TOOLS + WRITE_TOOLS,
        failing_tools: dict[str, str] | None = None,
        lineage_total: dict[str, int] | None = None,
        schema: dict[str, Any] | None = None,
    ) -> None:
        self._entities = entities or {}
        self._lineage = lineage or {}
        self._queries = queries or {}
        self._failing = failing_tools or {}
        # A total larger than the returned list is how DataHub says it capped
        # its own reply, which the reader has to record rather than ignore.
        self._lineage_total = lineage_total or {}
        # A real schema by default, not an empty one: `tools --schema` renders
        # the argument names, so a fixture with no properties cannot tell
        # whether they were rendered safely. `is None` rather than `or`,
        # because an empty schema is a case a caller asks for on purpose.
        self._tools = {
            name: ToolSpec(
                name=name,
                description=f"{name} tool",
                input_schema=DEFAULT_SCHEMA if schema is None else schema,
            )
            for name in tool_names
        }
        self.calls: list[tuple[str, dict[str, Any]]] = []
        self.tagged: list[dict[str, Any]] = []
        self.untagged: list[dict[str, Any]] = []
        self.scored: list[dict[str, Any]] = []
        self.documents: dict[str, dict[str, Any]] = {}

    @property
    def tools(self) -> dict[str, ToolSpec]:
        return self._tools

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    async def call(self, name: str, arguments: dict[str, Any]) -> Any:
        self.calls.append((name, arguments))
        if name not in self._tools:
            raise McpToolError(f"{name} is not available")
        if name in self._failing:
            raise McpToolError(f"{name} failed: {self._failing[name]}")
        return getattr(self, f"_{name}")(arguments)

    def _get_entities(self, arguments: dict[str, Any]) -> dict[str, Any]:
        found = [self._entities[urn] for urn in arguments["urns"] if urn in self._entities]
        return {"result": found}

    def _get_lineage(self, arguments: dict[str, Any]) -> dict[str, Any]:
        urn = arguments["urn"]
        children = self._lineage.get(urn, [])
        limit = arguments.get("max_results", 100)
        returned = children[:limit]
        block: dict[str, Any] = {
            "total": self._lineage_total.get(urn, len(children)),
            "searchResults": [
                {"entity": self._entities.get(child, {"urn": child})} for child in returned
            ],
        }
        return {"downstreams": block}

    def _get_dataset_queries(self, arguments: dict[str, Any]) -> dict[str, Any]:
        return {"total": self._queries.get(arguments["urn"], 0)}

    def _add_tags(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.tagged.append(arguments)
        return {"ok": True}

    def _remove_tags(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.untagged.append(arguments)
        return {"ok": True}

    def _add_structured_properties(self, arguments: dict[str, Any]) -> dict[str, Any]:
        self.scored.append(arguments)
        return {"ok": True}

    def _save_document(self, arguments: dict[str, Any]) -> dict[str, Any]:
        # save_document upserts on a urn it is given and mints one otherwise,
        # which is the behavior that makes updating in place possible at all.
        urn = arguments.get("urn") or f"urn:li:document:{len(self.documents) + 1}"
        self.documents[urn] = {**arguments, "urn": urn}
        return {"urn": urn}

    def _search_documents(self, arguments: dict[str, Any]) -> dict[str, Any]:
        query = arguments.get("query", "")
        return {
            "searchResults": [
                {
                    "entity": {
                        "urn": urn,
                        "info": {"title": document["title"], "lastModified": {"time": index}},
                    }
                }
                for index, (urn, document) in enumerate(self.documents.items())
                # Keyword search is fuzzy on a real instance, so this returns
                # near matches too. The exact-title check lives in the code
                # under test, not in the fixture.
                if query.split(":")[-1].strip()[:6] in document["title"]
            ]
        }
