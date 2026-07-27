"""What a scan publishes back into DataHub.

Split deliberately in two:

* `plan_writeback` decides *what* should be written. Pure, testable, and the
  place where the publishing policy lives.
* the executor (see `execute`) turns those intents into MCP tool calls. It is
  the only part that depends on a specific server's tool argument names.

Keeping the policy free of tool schemas means a server upgrade that renames an
argument cannot silently change which assets get flagged.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .brief import render_markdown, render_summary
from .mcp_client import DataHubMCP
from .models import BlastReport, Entity, Severity

# Structured properties are the right home for a score: they are typed, they
# show on the asset page, and they can be filtered in search. Tags carry the
# at-a-glance signal; the document carries the reasoning.
RISK_PROPERTY = "urn:li:structuredProperty:blastRadar.riskScore"
# Tag URNs are `urn:li:tag:<name>`, and DataHub parses the remainder as one
# name. A colon inside it (blast-radar:high) is rejected outright, so the
# severity is joined with a hyphen.
SEVERITY_TAG_PREFIX = "urn:li:tag:blast-radar"
SEVERITY_TAG_SEPARATOR = "-"


@dataclass(frozen=True)
class TagIntent:
    urn: str
    tag: str
    reason: str


@dataclass(frozen=True)
class PropertyIntent:
    urn: str
    property_urn: str
    value: float
    reason: str


@dataclass(frozen=True)
class DocumentIntent:
    title: str
    content: str
    related_urns: list[str]
    reason: str


WriteIntent = TagIntent | PropertyIntent | DocumentIntent


@dataclass(frozen=True)
class WritebackPolicy:
    """Publishing thresholds.

    Defaults are deliberately conservative. Writing a score onto all 400 nodes
    of a wide cone turns the graph into noise, and the assets nobody looks at
    are exactly the ones nobody needed flagged.
    """

    min_severity: Severity = Severity.HIGH
    max_tagged_assets: int = 25
    publish_document: bool = True
    publish_scores: bool = True


def plan_writeback(report: BlastReport, policy: WritebackPolicy | None = None) -> list[WriteIntent]:
    """Decide what this scan publishes back to DataHub."""
    policy = policy or WritebackPolicy()
    intents: list[WriteIntent] = []

    flagged = [item for item in report.ranked if item.severity.rank >= policy.min_severity.rank][
        : policy.max_tagged_assets
    ]

    for item in flagged:
        intents.append(
            TagIntent(
                urn=item.node.urn,
                tag=f"{SEVERITY_TAG_PREFIX}{SEVERITY_TAG_SEPARATOR}{item.severity.value}",
                reason=f"scored {item.score} against a change to {report.root.name}",
            )
        )
        if policy.publish_scores:
            intents.append(
                PropertyIntent(
                    urn=item.node.urn,
                    property_urn=RISK_PROPERTY,
                    # Rounded to a whole number on purpose: DataHub stores
                    # numeric properties as 32-bit floats, so 56.3 renders in
                    # the UI as 56.2999992370605. The tenth of a point is not
                    # worth that, and the severity band is what people act on.
                    value=item.published_score,
                    reason="blast radius risk score",
                )
            )

    # Only publish a brief when something was actually flagged. A document
    # saying "no consumer scored high or above", linked to nothing, is noise in
    # a knowledge base that people search.
    if policy.publish_document and flagged:
        intents.append(
            DocumentIntent(
                # No date in the title: the document is updated in place on the
                # next scan, and the title is what finds it again, so a date
                # would both go stale and break the lookup.
                title=brief_title(report.root),
                content=render_markdown(report),
                # The document is linked to the changed asset and to everything
                # it endangers, so it surfaces from either end of the blast.
                related_urns=[report.root.urn, *[item.node.urn for item in flagged]],
                reason=render_summary(report),
            )
        )

    return intents


def brief_title(root: Entity) -> str:
    """Title for the brief about `root`, unique per asset.

    Display names are not unique: the same `order_details` exists in dbt and
    in Snowflake, in PROD and in DEV. Since the title is what finds the
    previous brief for updating, a display-name title means scanning the DEV
    copy overwrites the PROD report. The URN's own coordinates disambiguate
    it while staying readable in DataHub's UI.
    """
    coordinates = _urn_coordinates(root.urn)
    if coordinates:
        return f"Blast radius: {coordinates}"
    return f"Blast radius: {root.name} [{root.urn}]"


def _urn_coordinates(urn: str) -> str | None:
    """`dataset:(dataPlatform:dbt,shop.orders,PROD)` -> `dbt shop.orders PROD`."""
    match = re.search(r"\(urn:li:dataPlatform:([^,]+),([^,]+),([^)]+)\)", urn)
    if not match:
        return None
    platform, name, fabric = match.groups()
    return f"{name} ({platform}, {fabric})"


async def find_document_by_title(datahub: DataHubMCP, title: str) -> str | None:
    """URN of the newest existing document with exactly this title, if any.

    Keyword search is fuzzy, so the title is compared exactly afterwards - a
    near match is a different brief, and updating it would overwrite a report
    about a different asset.
    """
    if not datahub.has_tool("search_documents"):
        return None
    try:
        # Generous: the search is keyword-based and a catalog can hold many
        # briefs with similar titles. Too small a page and the exact match
        # falls off the end, which silently creates a duplicate instead of
        # updating.
        payload = await datahub.call("search_documents", {"query": title, "num_results": 100})
    except Exception:
        # Broad on purpose. Not being able to look up the old brief is never a
        # reason to lose the new one, whether the failure is a tool error, a
        # dropped connection, or a payload we cannot read.
        return None

    results = payload.get("searchResults", []) if isinstance(payload, dict) else []
    matches = [
        entity
        for entity in (result.get("entity", {}) for result in results)
        if (entity.get("info") or {}).get("title") == title and entity.get("urn")
    ]
    if not matches:
        return None

    matches.sort(
        key=lambda entity: ((entity.get("info") or {}).get("lastModified") or {}).get("time", 0),
        reverse=True,
    )
    return str(matches[0]["urn"])


@dataclass
class WriteOutcome:
    """What actually happened when a plan was executed."""

    published: list[str]
    skipped: list[str]

    @property
    def ok(self) -> bool:
        return not self.skipped


async def execute(
    datahub: DataHubMCP, intents: list[WriteIntent], document_type: str = "Analysis"
) -> WriteOutcome:
    """Turn intents into MCP tool calls.

    Every operation is isolated. A publish that dies halfway leaves a catalog
    carrying half a verdict with nothing saying so, so each failure is caught,
    recorded, and the rest of the plan still runs. The exception handlers are
    deliberately broad: a dropped connection mid-publish is exactly the case
    that must not abort the remaining writes.
    """
    published: list[str] = []
    skipped: list[str] = []

    tag_groups: dict[str, list[str]] = {}
    for intent in intents:
        if isinstance(intent, TagIntent):
            tag_groups.setdefault(intent.tag, []).append(intent.urn)

    await _clear_stale_severity_tags(datahub, tag_groups, published, skipped)
    await _apply_tags(datahub, tag_groups, published, skipped)
    await _apply_properties(datahub, intents, published, skipped)
    await _apply_documents(datahub, intents, document_type, published, skipped)

    return WriteOutcome(published=published, skipped=skipped)


async def _apply_tags(
    datahub: DataHubMCP,
    tag_groups: dict[str, list[str]],
    published: list[str],
    skipped: list[str],
) -> None:
    for tag, urns in tag_groups.items():
        try:
            await datahub.call("add_tags", {"tag_urns": [tag], "entity_urns": urns})
            published.append(f"tagged {len(urns)} asset(s) with {tag}")
        except Exception as exc:
            skipped.append(f"tagging with {tag}: {exc}")


async def _apply_properties(
    datahub: DataHubMCP,
    intents: list[WriteIntent],
    published: list[str],
    skipped: list[str],
) -> None:
    for intent in intents:
        if not isinstance(intent, PropertyIntent):
            continue
        try:
            await datahub.call(
                "add_structured_properties",
                {
                    # The tool takes lists of values keyed by property URN, and
                    # the value type has to match the property's definition.
                    "property_values": {intent.property_urn: [intent.value]},
                    "entity_urns": [intent.urn],
                },
            )
            published.append(f"scored {intent.urn} at {intent.value}")
        except Exception as exc:
            skipped.append(f"scoring {intent.urn}: {exc}")


async def _apply_documents(
    datahub: DataHubMCP,
    intents: list[WriteIntent],
    document_type: str,
    published: list[str],
    skipped: list[str],
) -> None:
    for intent in intents:
        if not isinstance(intent, DocumentIntent):
            continue
        try:
            arguments: dict[str, object] = {
                "document_type": document_type,
                "title": intent.title,
                "content": intent.content,
                "related_assets": intent.related_urns,
            }
            # save_document upserts on a urn that already exists and mints a
            # fresh one otherwise, so the only way to update in place is to go
            # find the previous brief first. Without this, every CI run leaves
            # another near-identical document in the knowledge base.
            existing = await find_document_by_title(datahub, intent.title)
            if existing:
                arguments["urn"] = existing
            await datahub.call("save_document", arguments)
            published.append(f"published brief '{intent.title}'")
        except Exception as exc:
            skipped.append(f"publishing brief: {exc}")


async def _clear_stale_severity_tags(
    datahub: DataHubMCP,
    tag_groups: dict[str, list[str]],
    published: list[str],
    skipped: list[str],
) -> None:
    """Strip severity tags this scan is not about to re-apply.

    Severity is one value, but tags accumulate. Without this, an asset that
    scored critical last week and high today ends up carrying both, while the
    single-cardinality score property says only one of them - and the more
    alarming of the two labels is the stale one.
    """
    if not tag_groups or not datahub.has_tool("remove_tags"):
        return

    for stale_tag in _all_severity_tags():
        targets = [urn for tag, urns in tag_groups.items() if tag != stale_tag for urn in urns]
        if not targets:
            continue
        try:
            await datahub.call("remove_tags", {"tag_urns": [stale_tag], "entity_urns": targets})
        except Exception as exc:
            # Removing a tag an asset never had is normal and not worth
            # reporting; a real failure is, but it must not stop the publish.
            skipped.append(f"clearing {stale_tag}: {exc}")


def _all_severity_tags() -> list[str]:
    return [
        f"{SEVERITY_TAG_PREFIX}{SEVERITY_TAG_SEPARATOR}{severity.value}" for severity in Severity
    ]


def describe_plan(intents: list[WriteIntent]) -> str:
    """Human summary of a plan, used by --dry-run and by the demo narration."""
    if not intents:
        return "Nothing to publish: nothing met the publishing threshold."

    tags = sum(1 for intent in intents if isinstance(intent, TagIntent))
    properties = sum(1 for intent in intents if isinstance(intent, PropertyIntent))
    documents = sum(1 for intent in intents if isinstance(intent, DocumentIntent))

    parts = []
    if tags:
        parts.append(f"{tags} severity tag{'s' if tags != 1 else ''}")
    if properties:
        parts.append(f"{properties} risk score{'s' if properties != 1 else ''}")
    if documents:
        parts.append(f"{documents} impact brief{'s' if documents != 1 else ''}")
    return "Would publish " + ", ".join(parts) + " back to DataHub."
