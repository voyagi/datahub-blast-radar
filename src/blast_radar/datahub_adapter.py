"""Mapping between DataHub's MCP payloads and Blast Radar's models.

This is the only module that knows what DataHub's responses look like. It was
written against real responses from a v1.6.0 instance rather than from the
docs, because the tool schemas describe arguments, not result shapes.

Every accessor is defensive in one specific way: a field that is absent means
"not set on this asset", never an error. Catalogs are half-populated by nature,
and a scan that crashes on the first asset without an owner is useless.
"""

from __future__ import annotations

import re
import unicodedata
from typing import Any

from .mcp_client import DataHubMCP
from .models import CappedLineage, Entity, Owner

# urn:li:dataset:(...), urn:li:dashboard:(...), urn:li:chart:(...) and so on.
URN_TYPE = re.compile(r"^urn:li:([a-zA-Z]+):")

# Unicode classes that carry no glyph: control (Cc), format (Cf, which is where
# the bidirectional overrides live), and the line and paragraph separators
# (Zl, Zp). None of them belong in a catalog label, and each of them changes
# what a rendered brief means rather than how it looks.
INVISIBLE_CATEGORIES = frozenset({"Cc", "Cf", "Zl", "Zp"})

# DataHub's URN segment for an entity is camelCase; the risk model keys off the
# upper-case GraphQL style, so normalise once here.
URN_TYPE_TO_ENTITY_TYPE = {
    "dataset": "DATASET",
    "dashboard": "DASHBOARD",
    "chart": "CHART",
    "datajob": "DATAJOB",
    "dataflow": "DATAFLOW",
    "mlmodel": "MLMODEL",
    "mlfeaturetable": "MLFEATURETABLE",
    "notebook": "NOTEBOOK",
    "dataproduct": "DATAPRODUCT",
    "container": "CONTAINER",
}


class LineageParseError(RuntimeError):
    """The lineage response was not in a shape this adapter understands."""


class DataHubReader:
    """Read side of the integration."""

    def __init__(self, datahub: DataHubMCP, results_per_hop: int = 100) -> None:
        self._datahub = datahub
        self._results_per_hop = results_per_hop
        # Assets whose lineage came back capped. The scan reports these: a hub
        # table with 300 consumers contributing 50 of them, with no note, is a
        # brief that reads complete and is not.
        self.capped: dict[str, CappedLineage] = {}

    async def get_entity(self, urn: str) -> Entity:
        entities = await self.get_entities([urn])
        if urn not in entities:
            raise LookupError(f"DataHub has no entity with urn {urn}")
        return entities[urn]

    async def get_entities(self, urns: list[str]) -> dict[str, Entity]:
        """Batch-fetch full metadata. One call, however many URNs."""
        if not urns:
            return {}
        payload = await self._datahub.call("get_entities", {"urns": urns})
        results = payload.get("result", []) if isinstance(payload, dict) else []
        entities = [entity_from_payload(item) for item in results]
        return {entity.urn: entity for entity in entities}

    async def fetch_downstream(self, urn: str) -> list[Entity]:
        """One hop of downstream lineage.

        `max_hops=1` on purpose: the walker owns depth, so that hop numbers and
        fan-out counts come from one consistent traversal rather than from
        DataHub's flattened multi-hop view.
        """
        payload = await self._datahub.call(
            "get_lineage",
            {
                "urn": urn,
                "upstream": False,
                "max_hops": 1,
                "max_results": self._results_per_hop,
            },
        )
        block = payload.get("downstreams") if isinstance(payload, dict) else None
        if not isinstance(block, dict):
            # An empty list here would be indistinguishable from a genuinely
            # empty cone, and the brief renders that as "nothing downstream
            # depends on this asset. Safe to change." A shape we cannot read
            # has to fail loudly rather than clear a migration.
            raise LineageParseError(
                f"Could not read a downstream lineage block for {urn}. "
                f"Expected a 'downstreams' object, got {type(payload).__name__}: "
                f"{str(payload)[:200]}"
            )

        total = block.get("total")
        results = block.get("searchResults")
        if results is None:
            # The data-bearing key needs the same suspicion as the outer one.
            # Only an explicit zero total may be read as an empty cone.
            if total == 0:
                return []
            raise LineageParseError(
                f"Lineage for {urn} carried no 'searchResults' and did not "
                f"report an empty cone (total={total!r}). Keys present: "
                f"{sorted(block)}"
            )
        if not isinstance(results, list):
            raise LineageParseError(
                f"Lineage 'searchResults' for {urn} was {type(results).__name__}, not a list."
            )

        entities = [
            entity_from_payload(result.get("entity", {}))
            for result in results
            if isinstance(result, dict) and result.get("entity")
        ]

        # A hit with no URN is not addressable. It cannot be enriched, it
        # cannot be tagged, and the walk dedupes on exactly that string - so
        # two of them collapse into one node and the cone quietly reports one
        # consumer where there were two. Dropped, and counted as a shortfall
        # so the brief says the reply was short rather than reading complete.
        addressable = [entity for entity in entities if entity.urn]
        # Counted against the raw reply rather than against `entities`: a
        # result that carried no entity object at all never became one, and it
        # is just as much a consumer DataHub reported and this scan does not
        # have. Counting the difference here catches both shapes at once.
        dropped = len(results) - len(addressable)

        flagged = block.get("hasMore") is True
        short = isinstance(total, int) and total > len(addressable)
        if dropped or short or flagged:
            self.capped[urn] = CappedLineage(
                returned=len(addressable),
                total=_shortfall_total(total, flagged, len(results)),
            )

        return addressable

    async def query_count(self, urn: str) -> int | None:
        """How many recorded queries reference this asset, or None if unknown.

        Best-effort: query history is an optional part of a catalog, and its
        absence should cost the scan one factor, not the whole run. None and
        zero are different answers though - a failed lookup that returned 0
        would be published as "no queries reference it".
        """
        try:
            payload = await self._datahub.call("get_dataset_queries", {"urn": urn, "count": 1})
        except Exception:
            return None
        if not isinstance(payload, dict):
            return None
        total = payload.get("total")
        return int(total) if isinstance(total, (int, float)) else None


def _shortfall_total(total: Any, flagged: bool, hits: int) -> int | None:
    """The number a capped reply's shortfall can honestly be measured against.

    Three cases, and only two of them are a number.

    The reply's own total wins when it has one: it counts everything DataHub
    holds, dropped hits included. Otherwise, a reply that flagged more results
    without saying how many leaves the shortfall unknown, and it stays unknown
    however many hits were dropped here - counting only the drops would report
    an exact number on a reply that is short by an unknown amount. Only when
    nothing was flagged does the raw hit count serve as the total, and there
    it is exact: DataHub sent that many, and the difference is what could not
    be used.
    """
    if isinstance(total, int):
        return total
    return None if flagged else hits


def entity_from_payload(payload: dict[str, Any]) -> Entity:
    """Build an Entity from either a get_entities result or a lineage hit."""
    # `or ""` rather than a default: DataHub sends an explicit null for a URN
    # it has no value for, and `str(None)` is the string "None", which is
    # truthy, addressable-looking, and the same for every such hit.
    urn = str(payload.get("urn") or "")
    properties = payload.get("properties") or {}
    editable = payload.get("editableProperties") or {}

    return Entity(
        urn=urn,
        entity_type=_entity_type(payload, urn),
        name=_one_line(payload.get("name") or properties.get("name") or urn) or urn,
        platform=_platform(payload),
        # editableProperties wins: a description a human wrote in the UI is a
        # better signal of "someone looks after this" than an ingested one.
        description=editable.get("description") or properties.get("description"),
        owners=_owners(payload),
        tags=_tags(payload),
        glossary_terms=_glossary_terms(payload),
        domain=_domain(payload),
    )


def _one_line(value: Any) -> str:
    """Flatten a catalog label to a single line of printable text.

    Everything this adapter returns is written by whoever has ingestion rights
    to the catalog, and it ends up in a brief that gets published back into
    DataHub for other people to read. A newline in a display name splits a
    table row in that brief, so the half after it reads as a separate asset
    with a score nobody computed. A carriage return or an escape sequence does
    the same to a terminal.

    Format characters go too, not just control ones. A right-to-left override
    reverses everything printed after it, which is enough to make one asset's
    name read as another's in a ranked table.

    Cleaned here rather than in each renderer because it is the same answer
    everywhere: a display name is one line, and no legitimate one carries a
    character with no glyph. What each renderer still owns is escaping its own
    syntax, which is a different question with a different answer per format.
    """
    text = str(value)
    printable = "".join(
        " " if unicodedata.category(character) in INVISIBLE_CATEGORIES else character
        for character in text
    )
    return " ".join(printable.split())


def _entity_type(payload: dict[str, Any], urn: str) -> str:
    """Prefer the payload's own type, fall back to parsing the URN.

    get_entities results carry no `type` field while lineage hits do, so the
    URN is the only source that is always available.
    """
    declared = payload.get("type")
    if isinstance(declared, str):
        # Flattened like every other catalog string, since the type is
        # rendered in the ranked table and in the evidence line beside it and
        # a newline in it splits a row the way a newline in a name does. The
        # type check stays: a payload whose `type` is a number is a payload
        # this adapter does not understand, and the URN below is a better
        # answer than the number stringified.
        flattened = _one_line(declared).upper()
        if flattened:
            return flattened

    match = URN_TYPE.match(urn)
    if not match:
        return "UNKNOWN"
    segment = match.group(1).lower()
    return URN_TYPE_TO_ENTITY_TYPE.get(segment, segment.upper())


def _platform(payload: dict[str, Any]) -> str | None:
    platform = payload.get("platform") or {}
    name = platform.get("name") if isinstance(platform, dict) else None
    # `or None`, because `_one_line` can flatten an all-invisible name to "".
    # The contract here is a real platform or None, never the empty string.
    return (_one_line(name) or None) if name else None


def _owners(payload: dict[str, Any]) -> list[Owner]:
    ownership = payload.get("ownership") or {}
    owners: list[Owner] = []
    for record in ownership.get("owners", []) if isinstance(ownership, dict) else []:
        owner = record.get("owner") or {}
        urn = str(owner.get("urn", ""))
        if not urn:
            continue
        properties = owner.get("properties") or owner.get("info") or {}
        display = properties.get("displayName") or owner.get("name") or urn
        ownership_type = (record.get("ownershipType") or {}).get("info", {}).get("name")
        # The owner name reaches the brief twice: in the who-to-tell list and
        # inside the ownership factor's evidence string.
        owners.append(Owner(urn=urn, name=_one_line(display) or urn, ownership_type=ownership_type))
    return owners


def _tags(payload: dict[str, Any]) -> list[str]:
    tags = payload.get("tags") or {}
    names: list[str] = []
    for record in tags.get("tags", []) if isinstance(tags, dict) else []:
        tag = record.get("tag") or {}
        properties = tag.get("properties") or {}
        # Flatten before the guard, not after: a name that is nothing but
        # invisible characters flattens to "", and an empty string in this
        # list renders as a blank governance marker.
        label = _one_line(properties.get("name") or _urn_tail(str(tag.get("urn", ""))))
        if label:
            names.append(label)
    return names


def _glossary_terms(payload: dict[str, Any]) -> list[str]:
    terms = payload.get("glossaryTerms") or {}
    names: list[str] = []
    for record in terms.get("terms", []) if isinstance(terms, dict) else []:
        term = record.get("term") or {}
        properties = term.get("properties") or {}
        label = _one_line(properties.get("name") or _urn_tail(str(term.get("urn", ""))))
        if label:
            names.append(label)
    return names


def _domain(payload: dict[str, Any]) -> str | None:
    wrapper = payload.get("domain") or {}
    domain = wrapper.get("domain") if isinstance(wrapper, dict) else None
    if not isinstance(domain, dict):
        return None
    properties = domain.get("properties") or {}
    name = properties.get("name") or _urn_tail(str(domain.get("urn", "")))
    return (_one_line(name) or None) if name else None


def _urn_tail(urn: str) -> str:
    """Last readable segment of a URN, for when no display name is set."""
    return urn.rsplit(":", 1)[-1] if urn else ""
