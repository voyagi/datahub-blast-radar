"""The scan pipeline: lineage in, ranked impact out."""

from __future__ import annotations

from datetime import UTC, datetime

from .config import RunContext
from .datahub_adapter import DataHubReader
from .graph import walk_downstream
from .models import BlastReport, ConsumerNode, Entity
from .scoring import score_all

# get_entities takes a list of URNs, so ownership, tags, and terms for the
# whole cone cost a handful of batched calls. Every node gets enriched: an
# asset that was never looked up cannot be reported as having no owner.
ENRICH_BATCH = 50

# Query history is the opposite shape - one call per asset - so it is capped.
# Nodes past the cap report their usage as unknown rather than as zero.
USAGE_LIMIT = 40


async def run_scan(
    root_urn: str,
    change_summary: str,
    reader: DataHubReader,
    context: RunContext,
    *,
    with_usage: bool = True,
) -> BlastReport:
    """Walk, enrich, score."""
    settings = context.settings
    root = await reader.get_entity(root_urn)

    walk = await walk_downstream(
        root_urn,
        reader.fetch_downstream,
        max_hops=settings.max_hops,
        max_nodes=settings.max_nodes,
    )

    await _enrich(walk.nodes, reader)
    if with_usage:
        await _add_usage(walk.nodes, reader)

    return BlastReport(
        root=root,
        change_summary=change_summary,
        generated_at=datetime.now(UTC),
        assessments=score_all(walk.nodes, context.weights, settings.max_hops),
        hops_traversed=walk.hops_traversed,
        truncated=walk.truncated,
        capped_lineage=dict(reader.capped),
    )


async def _enrich(nodes: list[ConsumerNode], reader: DataHubReader) -> None:
    """Fill in ownership, tags, terms, and domain for every node."""
    for start in range(0, len(nodes), ENRICH_BATCH):
        batch = nodes[start : start + ENRICH_BATCH]
        detailed = await reader.get_entities([node.urn for node in batch])
        for node in batch:
            full = detailed.get(node.urn)
            if full is None:
                # Lineage returned it but the entity fetch did not. Keep the
                # lineage version rather than dropping a real consumer, and
                # leave `enriched` false so nothing claims it has no owner.
                continue
            node.entity = merge_entities(node.entity, full)
            node.enriched = True


async def _add_usage(nodes: list[ConsumerNode], reader: DataHubReader) -> None:
    """Look up query history for the assets most likely to top the ranking."""
    ordered = sorted(nodes, key=lambda node: (node.hop, -node.downstream_count))
    for node in ordered[:USAGE_LIMIT]:
        count = await reader.query_count(node.urn)
        if count is None:
            # The lookup failed. Leaving usage_known False keeps the brief
            # saying "not checked" instead of "no queries reference it".
            continue
        node.query_count = count
        node.usage_known = True


def merge_entities(from_lineage: Entity, from_fetch: Entity) -> Entity:
    """Combine both views of the same asset, field by field.

    Neither response is a superset of the other: the lineage hit is the only
    place some assets' tags appear, while the entity fetch is the only place
    full ownership does. Replacing one with the other silently drops whichever
    signal the winning side happened not to carry.
    """
    return Entity(
        urn=from_fetch.urn or from_lineage.urn,
        entity_type=from_fetch.entity_type or from_lineage.entity_type,
        name=from_fetch.name or from_lineage.name,
        platform=from_fetch.platform or from_lineage.platform,
        description=from_fetch.description or from_lineage.description,
        owners=from_fetch.owners or from_lineage.owners,
        tags=_merge_labels(from_fetch.tags, from_lineage.tags),
        glossary_terms=_merge_labels(from_fetch.glossary_terms, from_lineage.glossary_terms),
        domain=from_fetch.domain or from_lineage.domain,
    )


def _merge_labels(preferred: list[str], fallback: list[str]) -> list[str]:
    """Union of two label lists, order-stable and de-duplicated."""
    merged = list(preferred)
    for label in fallback:
        if label not in merged:
            merged.append(label)
    return merged
