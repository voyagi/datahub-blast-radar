"""Tests for the scan pipeline.

The pipeline is where the three "we did not look" flags are set, so these
tests are mostly about the difference between a value that is absent and one
that was never fetched. A batch that silently drops an asset, or a usage cap
that reports the assets past it as having no queries, both produce a brief
that reads complete and is not.

The reader is faked rather than mocked: `run_scan` takes the reader as an
argument precisely so the pipeline can be exercised with no catalog and no
network, and a fake that records its calls is what lets the batching and the
cap be asserted at all.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from blast_radar.config import RunContext, Settings
from blast_radar.models import Entity, Owner
from blast_radar.scan import ENRICH_BATCH, USAGE_LIMIT, merge_entities, run_scan


@pytest.fixture
def anyio_backend():
    return "asyncio"


def settings(**overrides) -> Settings:
    """Settings for a scan, built directly rather than from the environment.

    `Settings.from_env` reads the developer's own shell, which would make
    these tests pass or fail depending on whose machine they run on.
    """
    base = {
        "gms_url": "http://localhost:8080",
        "gms_token": None,
        "mcp_command": "uvx",
        "mcp_args": ("mcp-server-datahub==0.6.0",),
        "mutations_enabled": False,
        "max_hops": 5,
        "max_nodes": 400,
        "results_per_hop": 100,
        "debug": False,
        "server_log_path": Path(".blast-radar/mcp-server.log"),
    }
    return Settings(**{**base, **overrides})


def dataset(name: str, **fields) -> Entity:
    return Entity(urn=f"urn:li:dataset:{name}", entity_type="DATASET", name=name, **fields)


class FakeReader:
    """A DataHubReader that answers from dictionaries instead of a catalog."""

    def __init__(
        self,
        *,
        lineage: dict[str, list[Entity]] | None = None,
        detail: dict[str, Entity] | None = None,
        usage: dict[str, int | None] | None = None,
    ) -> None:
        self._lineage = lineage or {}
        self._detail = detail or {}
        self._usage = usage or {}
        self.capped: dict[str, object] = {}
        self.entity_batches: list[list[str]] = []
        self.usage_lookups: list[str] = []

    async def get_entity(self, urn: str) -> Entity:
        return self._detail.get(urn) or dataset(urn.rsplit(":", 1)[-1])

    async def get_entities(self, urns: list[str]) -> dict[str, Entity]:
        self.entity_batches.append(list(urns))
        return {urn: self._detail[urn] for urn in urns if urn in self._detail}

    async def fetch_downstream(self, urn: str) -> list[Entity]:
        return self._lineage.get(urn, [])

    async def query_count(self, urn: str) -> int | None:
        self.usage_lookups.append(urn)
        return self._usage.get(urn)


async def scan_with(reader: FakeReader, *, with_usage: bool = True, **setting_overrides):
    return await run_scan(
        "urn:li:dataset:shop.orders",
        "dropping column promo_code",
        reader,  # type: ignore[arg-type]
        RunContext(settings=settings(**setting_overrides)),
        with_usage=with_usage,
    )


@pytest.mark.anyio
async def test_a_scan_walks_enriches_and_scores_in_one_pass():
    reader = FakeReader(
        lineage={"urn:li:dataset:shop.orders": [dataset("exec.revenue")]},
        detail={
            "urn:li:dataset:exec.revenue": dataset(
                "exec.revenue", owners=[Owner("urn:li:corpuser:ana", "Ana")]
            )
        },
        usage={"urn:li:dataset:exec.revenue": 12},
    )

    report = await scan_with(reader)

    assert [item.node.entity.name for item in report.assessments] == ["exec.revenue"]
    node = report.assessments[0].node
    assert node.enriched is True
    assert node.usage_known is True
    assert node.query_count == 12


@pytest.mark.anyio
async def test_an_asset_the_entity_fetch_missed_is_kept_but_not_marked_enriched():
    """Lineage saw it, the detail call did not. Dropping it loses a real
    consumer; marking it enriched publishes "no owner" about an asset nobody
    ever looked up."""
    reader = FakeReader(lineage={"urn:li:dataset:shop.orders": [dataset("ghost")]}, detail={})

    report = await scan_with(reader)

    node = report.assessments[0].node
    assert node.entity.name == "ghost"
    assert node.enriched is False
    assert report.unowned == []
    assert len(report.unchecked_ownership) == 1


@pytest.mark.anyio
async def test_every_node_is_enriched_however_wide_the_cone():
    """Enrichment is batched, not capped. A cap here would put assets in the
    brief whose ownership was never checked, for no reason a reader could see."""
    children = [dataset(f"child{index}") for index in range(ENRICH_BATCH * 2 + 3)]
    reader = FakeReader(
        lineage={"urn:li:dataset:shop.orders": children},
        detail={child.urn: child for child in children},
    )

    report = await scan_with(reader, with_usage=False)

    assert len(report.assessments) == len(children)
    assert all(item.node.enriched for item in report.assessments)
    assert [len(batch) for batch in reader.entity_batches] == [ENRICH_BATCH, ENRICH_BATCH, 3]


@pytest.mark.anyio
async def test_usage_is_capped_and_the_assets_past_the_cap_say_so():
    """One call per asset is the expensive shape, so it is capped. The assets
    that miss out report their usage as unknown, never as zero queries."""
    children = [dataset(f"child{index}") for index in range(USAGE_LIMIT + 5)]
    reader = FakeReader(
        lineage={"urn:li:dataset:shop.orders": children},
        detail={child.urn: child for child in children},
        usage=dict.fromkeys((child.urn for child in children), 7),
    )

    report = await scan_with(reader)

    assert len(reader.usage_lookups) == USAGE_LIMIT
    checked = [item for item in report.assessments if item.node.usage_known]
    unchecked = [item for item in report.assessments if not item.node.usage_known]
    assert len(checked) == USAGE_LIMIT
    assert len(unchecked) == 5
    assert all(item.node.query_count == 0 for item in unchecked)
    assert all(
        "not checked" in factor.evidence
        for item in unchecked
        for factor in item.factors
        if factor.name == "usage"
    )


@pytest.mark.anyio
async def test_the_usage_budget_is_spent_nearest_first():
    """Spending it on the far edge of the cone would leave the top of the
    ranking unmeasured, since proximity is the second-heaviest factor."""
    near = dataset("near")
    far = [dataset(f"far{index}") for index in range(USAGE_LIMIT + 1)]
    reader = FakeReader(
        lineage={
            "urn:li:dataset:shop.orders": [near],
            near.urn: far,
        },
        detail={entity.urn: entity for entity in [near, *far]},
        usage=dict.fromkeys((entity.urn for entity in [near, *far]), 3),
    )

    report = await scan_with(reader)

    assert reader.usage_lookups[0] == near.urn
    by_urn = {item.node.urn: item.node for item in report.assessments}
    assert by_urn[near.urn].usage_known is True
    # The cone is two past the budget, and both report unknown, not zero.
    assert len(by_urn) == USAGE_LIMIT + 2
    assert sum(1 for node in by_urn.values() if not node.usage_known) == 2


@pytest.mark.anyio
async def test_assets_the_same_distance_out_are_ordered_by_fan_out():
    """The second half of the ordering, which the nearest-first case above
    cannot exercise because everything in it sits at a different hop.

    It decides which assets fall off the end of the budget: at equal distance,
    the one that breaks more things behind it is the one worth measuring.
    """
    ring = [dataset(f"ring{index}") for index in range(USAGE_LIMIT + 2)]
    # Fan-out ascending with the index, so the LAST listed is the widest and
    # the first two are the ones the cap should drop.
    lineage = {"urn:li:dataset:shop.orders": ring}
    for index, entity in enumerate(ring):
        lineage[entity.urn] = [dataset(f"child{index}_{child}") for child in range(index)]
    reader = FakeReader(
        lineage=lineage,
        detail={entity.urn: entity for entity in ring},
        usage=dict.fromkeys((entity.urn for entity in ring), 3),
    )

    # The ring alone is two past the budget, and it all sits one hop out, so
    # distance cannot decide anything here and the fan-out counts do.
    report = await scan_with(reader)

    ranked_urns = [item.node.urn for item in report.assessments if item.node.usage_known]
    assert len(ranked_urns) == USAGE_LIMIT
    # The two narrowest were dropped, not the two listed last.
    assert ring[0].urn not in ranked_urns
    assert ring[1].urn not in ranked_urns
    assert ring[-1].urn in ranked_urns


@pytest.mark.anyio
async def test_a_failed_usage_lookup_leaves_the_asset_unchecked_not_idle():
    reader = FakeReader(
        lineage={"urn:li:dataset:shop.orders": [dataset("exec.revenue")]},
        detail={"urn:li:dataset:exec.revenue": dataset("exec.revenue")},
        usage={"urn:li:dataset:exec.revenue": None},
    )

    report = await scan_with(reader)

    node = report.assessments[0].node
    assert node.usage_known is False
    assert node.query_count == 0


@pytest.mark.anyio
async def test_skipping_usage_skips_the_calls_entirely():
    reader = FakeReader(
        lineage={"urn:li:dataset:shop.orders": [dataset("exec.revenue")]},
        detail={"urn:li:dataset:exec.revenue": dataset("exec.revenue")},
        usage={"urn:li:dataset:exec.revenue": 12},
    )

    report = await scan_with(reader, with_usage=False)

    assert reader.usage_lookups == []
    assert report.assessments[0].node.usage_known is False


@pytest.mark.anyio
async def test_the_report_carries_what_the_reader_had_to_leave_out():
    """`capped_lineage` is the only route by which a per-hop cap reaches the
    brief. Copied, not referenced, so a later scan cannot rewrite this one."""
    reader = FakeReader(lineage={"urn:li:dataset:shop.orders": [dataset("exec.revenue")]})
    reader.capped["urn:li:dataset:hub"] = "recorded-by-the-reader"

    report = await scan_with(reader, with_usage=False)

    assert report.capped_lineage == {"urn:li:dataset:hub": "recorded-by-the-reader"}
    reader.capped["urn:li:dataset:later"] = "a-later-scan"
    assert "urn:li:dataset:later" not in report.capped_lineage


@pytest.mark.anyio
async def test_a_truncated_walk_reaches_the_report():
    children = [dataset(f"child{index}") for index in range(5)]
    reader = FakeReader(lineage={"urn:li:dataset:shop.orders": children})

    report = await scan_with(reader, with_usage=False, max_nodes=2)

    assert report.truncated is True
    assert len(report.assessments) == 2


def test_merging_two_views_of_an_asset_loses_neither():
    """Neither response is a superset. The fetch is the only place full
    ownership appears; the lineage hit is the only place some tags do."""
    from_lineage = Entity(
        urn="urn:li:dataset:shop.orders",
        entity_type="DATASET",
        name="shop.orders",
        platform="snowflake",
        description="from lineage",
        tags=["tier1", "shared"],
        glossary_terms=["revenue"],
        domain="Sales",
    )
    from_fetch = Entity(
        urn="urn:li:dataset:shop.orders",
        entity_type="DATASET",
        name="shop.orders",
        owners=[Owner("urn:li:corpuser:ana", "Ana")],
        tags=["pii", "shared"],
        glossary_terms=["gdpr"],
    )

    merged = merge_entities(from_lineage, from_fetch)

    assert [owner.name for owner in merged.owners] == ["Ana"]
    assert merged.platform == "snowflake"
    assert merged.description == "from lineage"
    assert merged.domain == "Sales"
    # Union, preferred side first, and `shared` appears once rather than twice.
    assert merged.tags == ["pii", "shared", "tier1"]
    assert merged.glossary_terms == ["gdpr", "revenue"]


def test_each_field_prefers_the_side_that_actually_carries_it():
    """Six fields, six preferences, and only the two obvious ones were pinned.

    Both sides carry all six of the fields this checks, and they disagree on
    every one. A fixture that blanks one side cannot tell the two directions
    apart: `a or b` and `b or a` return the same value when one of them is
    empty, so it passes whichever way the preference points, which is how four
    of these six went unpinned behind a test named for pinning them. The other
    three fields the merge handles - description, tags and terms - are pinned
    by the two tests either side of this one.
    """
    from_lineage = Entity(
        urn="urn:li:dataset:lineage",
        entity_type="DATASET",
        name="stale name",
        platform="hive",
        domain="Legacy",
        owners=[Owner("urn:li:corpuser:steve", "Stale Steve")],
    )
    from_fetch = Entity(
        urn="urn:li:dataset:fetch",
        entity_type="DASHBOARD",
        name="current name",
        platform="snowflake",
        domain="Sales",
        owners=[Owner("urn:li:corpuser:ana", "Ana")],
    )

    merged = merge_entities(from_lineage, from_fetch)

    assert merged.urn == "urn:li:dataset:fetch"
    assert merged.entity_type == "DASHBOARD"
    assert merged.name == "current name"
    assert merged.platform == "snowflake"
    assert merged.domain == "Sales"
    assert [owner.name for owner in merged.owners] == ["Ana"]


def test_the_fetched_view_wins_a_field_both_sides_carry():
    from_lineage = Entity(
        urn="urn:li:dataset:a", entity_type="DATASET", name="stale", description="stale"
    )
    from_fetch = Entity(
        urn="urn:li:dataset:a", entity_type="DATASET", name="current", description="current"
    )

    merged = merge_entities(from_lineage, from_fetch)

    assert (merged.name, merged.description) == ("current", "current")
