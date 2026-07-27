"""Tests for the publishing policy.

The policy decides what a scan writes back into the metadata graph, so the
cases that matter are the restraint ones: not flagging everything, not
publishing when there is nothing to say.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from blast_radar.config import ScoringWeights
from blast_radar.models import BlastReport, ConsumerNode, Entity, Owner, Severity
from blast_radar.scoring import score_node
from blast_radar.writeback import (
    DocumentIntent,
    PropertyIntent,
    TagIntent,
    WritebackPolicy,
    brief_title,
    describe_plan,
    execute,
    find_document_by_title,
    plan_writeback,
)
from inmemory_datahub import InMemoryDataHub

WEIGHTS = ScoringWeights()
MAX_HOPS = 4


def loud_node(name: str) -> ConsumerNode:
    """A node that reliably scores critical."""
    return ConsumerNode(
        entity=Entity(
            urn=f"urn:li:dashboard:{name}",
            entity_type="DASHBOARD",
            name=name,
            tags=["Tier1"],
        ),
        hop=1,
        transform_hop=1,
        downstream_count=20,
        query_count=90,
        enriched=True,
        usage_known=True,
    )


def quiet_node(name: str) -> ConsumerNode:
    """A node that reliably scores low."""
    return ConsumerNode(
        entity=Entity(
            urn=f"urn:li:dataset:{name}",
            entity_type="NOTEBOOK",
            name=name,
            owners=[Owner("urn:li:corpuser:ana", "Ana")],
        ),
        hop=4,
        transform_hop=4,
        downstream_count=0,
        query_count=0,
        enriched=True,
        usage_known=True,
    )


def report_for(nodes: list[ConsumerNode]) -> BlastReport:
    return BlastReport(
        root=Entity(urn="urn:li:dataset:shop.orders", entity_type="DATASET", name="shop.orders"),
        change_summary="dropped column promo_code",
        generated_at=datetime(2026, 7, 23, 10, 0, tzinfo=UTC),
        assessments=[score_node(node, WEIGHTS, MAX_HOPS) for node in nodes],
        hops_traversed=2,
    )


def test_high_severity_assets_get_a_tag_and_a_score():
    intents = plan_writeback(report_for([loud_node("exec.revenue")]))

    tags = [intent for intent in intents if isinstance(intent, TagIntent)]
    properties = [intent for intent in intents if isinstance(intent, PropertyIntent)]
    assert len(tags) == 1
    assert tags[0].tag.endswith("critical")
    assert properties[0].property_urn.endswith("blastRadar.riskScore")
    assert properties[0].value > 0


def test_low_severity_assets_are_left_alone():
    intents = plan_writeback(report_for([quiet_node("staging.tmp")]))

    assert not [intent for intent in intents if isinstance(intent, TagIntent)]


def test_the_document_links_the_root_and_the_flagged_assets():
    intents = plan_writeback(report_for([loud_node("exec.revenue"), quiet_node("staging.tmp")]))

    document = next(intent for intent in intents if isinstance(intent, DocumentIntent))
    assert "urn:li:dataset:shop.orders" in document.related_urns
    assert "urn:li:dashboard:exec.revenue" in document.related_urns
    assert "urn:li:dataset:staging.tmp" not in document.related_urns
    assert "shop.orders" in document.title
    assert "dropped column promo_code" in document.content


def test_two_environments_of_the_same_table_get_different_briefs():
    """Display names are not unique; overwriting the PROD brief from DEV is real."""
    prod = Entity(
        urn="urn:li:dataset:(urn:li:dataPlatform:dbt,shop.orders,PROD)",
        entity_type="DATASET",
        name="orders",
    )
    dev = Entity(
        urn="urn:li:dataset:(urn:li:dataPlatform:dbt,shop.orders,DEV)",
        entity_type="DATASET",
        name="orders",
    )
    snowflake = Entity(
        urn="urn:li:dataset:(urn:li:dataPlatform:snowflake,shop.orders,PROD)",
        entity_type="DATASET",
        name="orders",
    )

    titles = {brief_title(prod), brief_title(dev), brief_title(snowflake)}

    assert len(titles) == 3
    assert brief_title(prod) == "Blast radius: shop.orders (dbt, PROD)"


def test_a_urn_without_platform_coordinates_still_gets_a_unique_title():
    entity = Entity(
        urn="urn:li:dashboard:(looker,dashboards.53)", entity_type="DASHBOARD", name="Sales"
    )

    title = brief_title(entity)

    assert "Sales" in title
    assert "urn:li:dashboard:(looker,dashboards.53)" in title


def test_the_brief_title_carries_no_date():
    """The document is updated in place, so a dated title would go stale."""
    document = next(
        intent
        for intent in plan_writeback(report_for([loud_node("exec.revenue")]))
        if isinstance(intent, DocumentIntent)
    )

    assert "2026" not in document.title


def test_nothing_downstream_publishes_nothing():
    assert plan_writeback(report_for([])) == []


def test_tagging_is_capped_so_a_wide_cone_does_not_flood_the_graph():
    nodes = [loud_node(f"dash{index}") for index in range(40)]

    intents = plan_writeback(report_for(nodes), WritebackPolicy(max_tagged_assets=5))

    assert len([intent for intent in intents if isinstance(intent, TagIntent)]) == 5


def test_scores_can_be_suppressed_while_still_tagging():
    policy = WritebackPolicy(publish_scores=False)

    intents = plan_writeback(report_for([loud_node("exec.revenue")]), policy)

    assert [intent for intent in intents if isinstance(intent, TagIntent)]
    assert not [intent for intent in intents if isinstance(intent, PropertyIntent)]


def test_lowering_the_threshold_pulls_in_moderate_assets():
    node = ConsumerNode(
        entity=Entity(
            urn="urn:li:dataset:mid",
            entity_type="DATASET",
            name="mid",
            owners=[Owner("urn:li:corpuser:ana", "Ana")],
        ),
        hop=2,
        downstream_count=3,
        query_count=5,
    )
    report = report_for([node])
    assert report.ranked[0].severity is Severity.MODERATE

    strict = plan_writeback(report)
    relaxed = plan_writeback(report, WritebackPolicy(min_severity=Severity.MODERATE))

    assert not [intent for intent in strict if isinstance(intent, TagIntent)]
    assert [intent for intent in relaxed if isinstance(intent, TagIntent)]


class FakeDocumentSearch:
    """Stands in for an MCP session that only answers search_documents."""

    def __init__(self, results, *, has_tool: bool = True, raises: Exception | None = None):
        self._results = results
        self._has_tool = has_tool
        self._raises = raises
        self.queries: list[str] = []

    def has_tool(self, name: str) -> bool:
        return self._has_tool

    async def call(self, name: str, arguments):
        self.queries.append(arguments.get("query", ""))
        if self._raises:
            raise self._raises
        return {"searchResults": self._results}


def document(urn: str, title: str, modified: int):
    return {"entity": {"urn": urn, "info": {"title": title, "lastModified": {"time": modified}}}}


@pytest.mark.anyio
async def test_the_newest_document_with_an_exact_title_match_wins():
    datahub = FakeDocumentSearch(
        [
            document("urn:li:document:old", "Blast radius: shop.orders", 100),
            document("urn:li:document:new", "Blast radius: shop.orders", 900),
        ]
    )

    found = await find_document_by_title(datahub, "Blast radius: shop.orders")

    assert found == "urn:li:document:new"


@pytest.mark.anyio
async def test_a_near_title_match_is_not_treated_as_the_same_brief():
    """Keyword search is fuzzy; overwriting a brief about another asset is not."""
    datahub = FakeDocumentSearch(
        [document("urn:li:document:other", "Blast radius: shop.customers", 900)]
    )

    assert await find_document_by_title(datahub, "Blast radius: shop.orders") is None


@pytest.mark.anyio
async def test_a_failed_lookup_creates_a_new_brief_rather_than_losing_it():
    from blast_radar.mcp_client import McpToolError

    datahub = FakeDocumentSearch([], raises=McpToolError("search is down"))

    assert await find_document_by_title(datahub, "Blast radius: shop.orders") is None


@pytest.mark.anyio
async def test_an_instance_without_document_search_skips_the_lookup():
    datahub = FakeDocumentSearch([], has_tool=False)

    assert await find_document_by_title(datahub, "anything") is None
    assert datahub.queries == []


@pytest.fixture
def anyio_backend():
    return "asyncio"


def test_describe_plan_is_honest_about_an_empty_plan():
    assert "Nothing to publish" in describe_plan([])
    assert "impact brief" in describe_plan(plan_writeback(report_for([loud_node("d")])))


def test_the_dry_run_sentence_names_only_what_the_policy_would_write():
    """This line is the whole output of a run without --publish, so it has to
    match the plan rather than describe a default one."""
    report = report_for([loud_node("one"), loud_node("two")])
    scores_off = WritebackPolicy(publish_scores=False, publish_document=False)

    assert describe_plan(plan_writeback(report, scores_off)) == (
        "Would publish 2 severity tags back to DataHub."
    )
    assert describe_plan(plan_writeback(report)) == (
        "Would publish 2 severity tags, 2 risk scores, 1 impact brief back to DataHub."
    )


# --- Executing a plan -------------------------------------------------------
#
# The executor's whole job is that one failed write does not take the others
# with it. A publish that dies halfway leaves the catalog carrying half a
# verdict with nothing saying so, so every case below is a failure case.


def plan_for_one_loud_asset():
    return plan_writeback(report_for([loud_node("exec.revenue"), quiet_node("staging.tmp")]))


@pytest.mark.anyio
async def test_a_failed_tag_write_is_reported_and_the_score_still_lands():
    datahub = InMemoryDataHub(failing_tools={"add_tags": "mutations are disabled"})

    outcome = await execute(datahub, plan_for_one_loud_asset())

    assert outcome.ok is False
    assert any("tagging with" in line for line in outcome.skipped)
    assert datahub.scored
    assert datahub.documents


@pytest.mark.anyio
async def test_a_failed_score_write_does_not_cost_the_brief():
    datahub = InMemoryDataHub(failing_tools={"add_structured_properties": "property undefined"})

    outcome = await execute(datahub, plan_for_one_loud_asset())

    assert outcome.ok is False
    assert any("scoring urn:li:dashboard:exec.revenue" in line for line in outcome.skipped)
    assert datahub.tagged
    assert datahub.documents


@pytest.mark.anyio
async def test_a_failed_brief_is_named_rather_than_swallowed():
    datahub = InMemoryDataHub(failing_tools={"save_document": "document service down"})

    outcome = await execute(datahub, plan_for_one_loud_asset())

    assert outcome.ok is False
    assert any("publishing brief" in line for line in outcome.skipped)
    assert datahub.tagged
    assert datahub.scored


@pytest.mark.anyio
async def test_a_failed_stale_tag_removal_is_recorded_but_does_not_stop_the_publish():
    datahub = InMemoryDataHub(failing_tools={"remove_tags": "tag service down"})

    outcome = await execute(datahub, plan_for_one_loud_asset())

    assert any("clearing urn:li:tag:blast-radar" in line for line in outcome.skipped)
    assert datahub.tagged
    assert datahub.scored


@pytest.mark.anyio
async def test_a_clean_publish_reports_every_write_it_made():
    datahub = InMemoryDataHub()

    outcome = await execute(datahub, plan_for_one_loud_asset())

    assert outcome.ok is True
    assert outcome.skipped == []
    assert any("tagged 1 asset(s)" in line for line in outcome.published)
    assert any("scored urn:li:dashboard:exec.revenue" in line for line in outcome.published)
    assert any("published brief" in line for line in outcome.published)


@pytest.mark.anyio
async def test_an_empty_plan_touches_nothing():
    """Nothing met the threshold, so nothing is written. Not even a document
    saying so: an unlinked "no consumer scored high" note is noise in a
    knowledge base people search."""
    datahub = InMemoryDataHub()

    outcome = await execute(datahub, [])

    assert outcome.ok is True
    assert datahub.calls == []


@pytest.mark.anyio
async def test_assets_sharing_a_severity_are_tagged_in_one_call():
    """Per-asset calls would turn a wide cone into 25 round trips."""
    datahub = InMemoryDataHub()
    plan = plan_writeback(report_for([loud_node("one"), loud_node("two")]))

    await execute(datahub, plan)

    assert datahub.tagged == [
        {
            "tag_urns": ["urn:li:tag:blast-radar-critical"],
            "entity_urns": ["urn:li:dashboard:one", "urn:li:dashboard:two"],
        }
    ]


@pytest.mark.anyio
async def test_a_second_scan_updates_the_brief_it_already_published():
    datahub = InMemoryDataHub()
    plan = plan_for_one_loud_asset()

    await execute(datahub, plan)
    await execute(datahub, plan)

    assert len(datahub.documents) == 1
    saved = [call for call in datahub.calls if call[0] == "save_document"]
    assert "urn" not in saved[0][1]
    assert saved[1][1]["urn"] in datahub.documents
