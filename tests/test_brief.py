"""Tests for report aggregation and brief rendering."""

from __future__ import annotations

from datetime import UTC, datetime

from blast_radar.brief import disclosure_notes, render_markdown, render_summary
from blast_radar.config import ScoringWeights
from blast_radar.models import (
    BlastReport,
    CappedLineage,
    ConsumerNode,
    Entity,
    Owner,
    Severity,
)
from blast_radar.scoring import score_node

WEIGHTS = ScoringWeights()
MAX_HOPS = 4


def build_report(nodes: list[ConsumerNode], truncated: bool = False) -> BlastReport:
    root = Entity(
        urn="urn:li:dataset:(urn:li:dataPlatform:snowflake,shop.orders,PROD)",
        entity_type="DATASET",
        name="shop.orders",
    )
    return BlastReport(
        root=root,
        change_summary="dropped column promo_code",
        generated_at=datetime(2026, 7, 23, 10, 0, tzinfo=UTC),
        assessments=[score_node(node, WEIGHTS, MAX_HOPS) for node in nodes],
        hops_traversed=3,
        truncated=truncated,
    )


def node(
    name: str,
    *,
    hop: int = 1,
    entity_type: str = "DATASET",
    downstream_count: int = 0,
    query_count: int = 0,
    owners: list[Owner] | None = None,
    tags: list[str] | None = None,
) -> ConsumerNode:
    entity = Entity(
        urn=f"urn:li:{entity_type.lower()}:{name}",
        entity_type=entity_type,
        name=name,
        owners=owners or [],
        tags=tags or [],
    )
    return ConsumerNode(
        entity=entity,
        hop=hop,
        transform_hop=hop,
        downstream_count=downstream_count,
        query_count=query_count,
        enriched=True,
        usage_known=True,
    )


def test_ranking_puts_the_worst_first():
    report = build_report(
        [
            node("staging.orders_tmp", hop=4, owners=[Owner("urn:li:corpuser:ana", "Ana")]),
            node("exec.revenue_dashboard", hop=1, entity_type="DASHBOARD", query_count=90),
        ]
    )

    assert report.ranked[0].node.entity.name == "exec.revenue_dashboard"


def test_unowned_lists_only_assets_without_owners():
    report = build_report(
        [
            node("owned.table", owners=[Owner("urn:li:corpuser:ana", "Ana")]),
            node("orphan.table"),
        ]
    )

    assert [item.node.entity.name for item in report.unowned] == ["orphan.table"]


def test_empty_downstream_reports_safe_to_change():
    markdown = render_markdown(build_report([]))

    assert "nothing downstream depends on this asset" in markdown


def test_brief_names_the_change_the_asset_and_the_owners():
    report = build_report(
        [
            node(
                "exec.revenue_dashboard",
                hop=1,
                entity_type="DASHBOARD",
                query_count=90,
                downstream_count=4,
                owners=[Owner("urn:li:corpuser:ana", "Ana Silva")],
                tags=["Tier1"],
            )
        ]
    )
    markdown = render_markdown(report)

    assert "dropped column promo_code" in markdown
    assert "shop.orders" in markdown
    assert "exec.revenue_dashboard" in markdown
    assert "Ana Silva" in markdown
    assert "## Who to tell" in markdown


def test_the_who_to_tell_list_says_when_it_stopped_listing():
    """An owner of nine flagged assets gets six of them and a count.

    The cut is the same rule as everywhere else in this brief: a list that
    stops without saying so reads as the whole answer. Untested until now,
    which meant the note could have gone missing without anything failing.
    """
    ana = Owner(urn="urn:li:corpuser:ana", name="Ana")
    nodes = [
        node(
            f"dash{index:02d}",
            entity_type="DASHBOARD",
            downstream_count=20,
            query_count=90,
            owners=[ana],
            tags=["tier1"],
        )
        for index in range(9)
    ]

    markdown = render_markdown(build_report(nodes))

    notify = next(line for line in markdown.splitlines() if line.startswith("- **Ana**"))
    assert notify.count("dash") == 6
    assert "(+3 more)" in notify


def test_the_middle_verdict_says_normal_review_applies():
    """Three verdicts, and this is the one a scan lands on most often: real
    consumers, none of them alarming."""
    markdown = render_markdown(build_report([node("staging", hop=4, entity_type="NOTEBOOK")]))

    assert "none scored above moderate" in markdown
    assert "Normal review applies" in markdown


def test_who_to_tell_skips_low_severity_owners():
    report = build_report(
        [
            node(
                "quiet.table",
                hop=4,
                owners=[Owner("urn:li:corpuser:quiet", "Quiet Owner")],
            )
        ]
    )
    markdown = render_markdown(report)

    assert report.ranked[0].severity.rank < Severity.HIGH.rank
    assert "Quiet Owner" not in markdown


def test_unenriched_assets_are_reported_as_unknown_not_as_orphans():
    unchecked = ConsumerNode(
        entity=Entity(urn="urn:li:dataset:skipped", entity_type="DATASET", name="skipped.table"),
        hop=1,
        transform_hop=1,
        enriched=False,
    )
    report = build_report([node("orphan.table"), unchecked])

    markdown = render_markdown(report)

    assert [item.node.entity.name for item in report.unowned] == ["orphan.table"]
    assert [item.node.entity.name for item in report.unchecked_ownership] == ["skipped.table"]
    assert "## Ownership not checked" in markdown
    assert "unknown rather than absent" in markdown


def test_the_capped_note_reports_what_is_missing_not_the_total():
    """The number a reader acts on is the shortfall, not the catalog's total."""
    report = build_report([node("a.table")])
    report.capped_lineage = {
        "urn:li:dataset:hub": CappedLineage(returned=100, total=150),
        "urn:li:dataset:other": CappedLineage(returned=10, total=None),
    }

    markdown = render_markdown(report)

    assert report.consumers_missing == 50
    assert "50 consumer(s) are missing" in markdown
    assert "unstated number from 1 asset(s)" in markdown
    assert "150" not in markdown


def test_truncation_is_disclosed_rather_than_silent():
    markdown = render_markdown(build_report([node("a.table")], truncated=True))

    assert "node budget" in markdown


def test_a_whole_cone_has_nothing_to_disclose():
    assert disclosure_notes(build_report([node("a.table")])) == []


def test_an_uncounted_shortfall_is_never_rendered_as_zero():
    """Every capped asset unknown means the shortfall is unknown, not none.

    Summing no known shortfalls gives zero, and "0 consumer(s) are missing"
    sits on the one sentence whose job is to say something was left out.
    """
    report = build_report([node("a.table")])
    report.capped_lineage = {
        "urn:li:dataset:hub": CappedLineage(returned=100, total=None),
        "urn:li:dataset:other": CappedLineage(returned=10, total=None),
    }

    markdown = render_markdown(report)

    assert report.consumers_missing is None
    assert "0 consumer(s)" not in markdown
    assert "unknown" in markdown


def test_a_cut_list_says_it_was_cut():
    report = build_report([node(f"orphan{index}.table") for index in range(4)])

    markdown = render_markdown(report, limit=2)

    assert len(report.unowned) == 4
    assert "4 impacted asset(s) have nobody to notify, the 2 highest-scoring" in markdown
    assert "The 2 highest-scoring of 4 consumers." in markdown


def test_every_disclosure_the_brief_makes_is_available_to_other_renderers():
    """The terminal renders from this list too.

    While these sentences lived inside `render_markdown`, the terminal - the
    surface the headline command prints - disclosed neither of them.
    """
    report = build_report([node("a.table")], truncated=True)
    report.capped_lineage = {"urn:li:dataset:hub": CappedLineage(returned=100, total=150)}

    notes = disclosure_notes(report)
    markdown = render_markdown(report)

    assert len(notes) == 2
    for note in notes:
        assert note in markdown


def test_summary_counts_match_the_report():
    report = build_report(
        [
            node("exec.dash", hop=1, entity_type="DASHBOARD", query_count=90, tags=["tier1"]),
            node("orphan.table", hop=3),
        ]
    )

    summary = render_summary(report)
    assert "2 downstream assets" in summary
    assert "without an owner" in summary
