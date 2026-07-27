"""Characterization tests: they pin what the model does TODAY, not what it should do.

Every other suite in this repo asserts a property (a dashboard outranks a staging
table, an unchecked asset is never reported as an orphan). Those pass just as
happily if every score in the system shifts by the same amount, because they only
compare scores to each other.

That matters here because the published numbers are load-bearing. `README.md`
prints a real run, `docs/submission.md` quotes it, and `docs/screenshots/` froze it
as images that cannot be re-shot without standing DataHub back up. A refactor that
moves the scale would leave the code correct, every behavioral test green, and all
three of those documents quietly wrong.

So these tests exist to FAIL on change. A diff here is not automatically a bug: it
means the published numbers, the screenshots, and the submission copy need to be
re-checked, and the golden values updated in the same commit. Treat a failure as a
question, not a verdict.
"""

from __future__ import annotations

from datetime import UTC, datetime

import pytest

from blast_radar.brief import render_markdown, render_summary
from blast_radar.config import ScoringWeights
from blast_radar.models import (
    BlastReport,
    CappedLineage,
    ConsumerNode,
    Entity,
    Owner,
    RiskAssessment,
    RiskFactor,
)
from blast_radar.scoring import score_all, score_node

WEIGHTS = ScoringWeights()
# The shipped default, not the 4 the other suites use. These golden values are
# meant to match what a real run produces.
MAX_HOPS = 5

ANA = Owner(urn="urn:li:corpuser:ana", name="Ana Ortiz")
PLATFORM = Owner(urn="urn:li:corpGroup:platform", name="Platform Guild")


def make_node(
    *,
    hop: int,
    transform_hop: int,
    entity_type: str,
    name: str = "shop.thing",
    downstream_count: int = 0,
    query_count: int = 0,
    owners: tuple[Owner, ...] = (),
    tags: tuple[str, ...] = (),
    expanded: bool = True,
    enriched: bool = True,
    usage_known: bool = True,
) -> ConsumerNode:
    entity = Entity(
        urn=f"urn:li:{entity_type.lower()}:(urn:li:dataPlatform:dbt,{name},PROD)",
        entity_type=entity_type,
        name=name,
        owners=list(owners),
        tags=list(tags),
    )
    return ConsumerNode(
        entity=entity,
        hop=hop,
        transform_hop=transform_hop,
        downstream_count=downstream_count,
        query_count=query_count,
        expanded=expanded,
        enriched=enriched,
        usage_known=usage_known,
    )


# Node shapes chosen to cover every factor's curve, both saturation points, all
# three unknown flags, and the ceiling and floor of the weight budget.
CASES: dict[str, ConsumerNode] = {
    "readme_order_details": make_node(
        hop=1,
        transform_hop=1,
        entity_type="DATASET",
        downstream_count=14,
        owners=(ANA,),
        tags=("Tier1",),
        usage_known=False,
    ),
    "readme_customer_analysis_chart": make_node(
        hop=3,
        transform_hop=2,
        entity_type="CHART",
        downstream_count=1,
        usage_known=False,
    ),
    "type_dashboard": make_node(hop=2, transform_hop=2, entity_type="DASHBOARD", owners=(ANA,)),
    "type_chart": make_node(hop=2, transform_hop=2, entity_type="CHART", owners=(ANA,)),
    "type_mlmodel": make_node(hop=2, transform_hop=2, entity_type="MLMODEL", owners=(ANA,)),
    "type_mlfeaturetable": make_node(
        hop=2, transform_hop=2, entity_type="MLFEATURETABLE", owners=(ANA,)
    ),
    "type_datajob": make_node(hop=2, transform_hop=2, entity_type="DATAJOB", owners=(ANA,)),
    "type_dataflow": make_node(hop=2, transform_hop=2, entity_type="DATAFLOW", owners=(ANA,)),
    "type_dataset": make_node(hop=2, transform_hop=2, entity_type="DATASET", owners=(ANA,)),
    "type_notebook": make_node(hop=2, transform_hop=2, entity_type="NOTEBOOK", owners=(ANA,)),
    "type_unrecognised": make_node(
        hop=2, transform_hop=2, entity_type="UNRECOGNISED", owners=(ANA,)
    ),
    "proximity_transform_hop_1": make_node(
        hop=1, transform_hop=1, entity_type="DATASET", owners=(ANA,)
    ),
    "proximity_transform_hop_2": make_node(
        hop=2, transform_hop=2, entity_type="DATASET", owners=(ANA,)
    ),
    "proximity_transform_hop_3": make_node(
        hop=3, transform_hop=3, entity_type="DATASET", owners=(ANA,)
    ),
    "proximity_transform_hop_4": make_node(
        hop=4, transform_hop=4, entity_type="DATASET", owners=(ANA,)
    ),
    "proximity_transform_hop_5": make_node(
        hop=5, transform_hop=5, entity_type="DATASET", owners=(ANA,)
    ),
    "proximity_transform_hop_6": make_node(
        hop=6, transform_hop=6, entity_type="DATASET", owners=(ANA,)
    ),
    "proximity_transform_hop_12": make_node(
        hop=12, transform_hop=12, entity_type="DATASET", owners=(ANA,)
    ),
    "fanout_0": make_node(
        hop=1, transform_hop=1, entity_type="DATASET", downstream_count=0, owners=(ANA,)
    ),
    "fanout_1": make_node(
        hop=1, transform_hop=1, entity_type="DATASET", downstream_count=1, owners=(ANA,)
    ),
    "fanout_5": make_node(
        hop=1, transform_hop=1, entity_type="DATASET", downstream_count=5, owners=(ANA,)
    ),
    "fanout_14": make_node(
        hop=1, transform_hop=1, entity_type="DATASET", downstream_count=14, owners=(ANA,)
    ),
    "fanout_25": make_node(
        hop=1, transform_hop=1, entity_type="DATASET", downstream_count=25, owners=(ANA,)
    ),
    "fanout_200": make_node(
        hop=1, transform_hop=1, entity_type="DATASET", downstream_count=200, owners=(ANA,)
    ),
    "fanout_10000": make_node(
        hop=1, transform_hop=1, entity_type="DATASET", downstream_count=10_000, owners=(ANA,)
    ),
    "usage_0": make_node(
        hop=1, transform_hop=1, entity_type="DATASET", query_count=0, owners=(ANA,)
    ),
    "usage_1": make_node(
        hop=1, transform_hop=1, entity_type="DATASET", query_count=1, owners=(ANA,)
    ),
    "usage_10": make_node(
        hop=1, transform_hop=1, entity_type="DATASET", query_count=10, owners=(ANA,)
    ),
    "usage_50": make_node(
        hop=1, transform_hop=1, entity_type="DATASET", query_count=50, owners=(ANA,)
    ),
    "usage_500": make_node(
        hop=1, transform_hop=1, entity_type="DATASET", query_count=500, owners=(ANA,)
    ),
    "unknown_not_expanded": make_node(
        hop=1,
        transform_hop=1,
        entity_type="DATASET",
        downstream_count=99,
        owners=(ANA,),
        expanded=False,
    ),
    "unknown_not_enriched": make_node(
        hop=1, transform_hop=1, entity_type="DATASET", enriched=False
    ),
    "unknown_usage": make_node(
        hop=1,
        transform_hop=1,
        entity_type="DATASET",
        query_count=99,
        owners=(ANA,),
        usage_known=False,
    ),
    "orphan": make_node(hop=1, transform_hop=1, entity_type="DATASET"),
    "owned": make_node(hop=1, transform_hop=1, entity_type="DATASET", owners=(ANA,)),
    "governance_marked": make_node(
        hop=1, transform_hop=1, entity_type="DATASET", owners=(ANA,), tags=("Tier1", "PII")
    ),
    "governance_plain": make_node(
        hop=1, transform_hop=1, entity_type="DATASET", owners=(ANA,), tags=("sandbox",)
    ),
    "worst_case": make_node(
        hop=1,
        transform_hop=1,
        entity_type="DASHBOARD",
        downstream_count=10_000,
        query_count=10_000,
        tags=("tier1", "pii"),
    ),
    "best_case": make_node(hop=12, transform_hop=12, entity_type="NOTEBOOK", owners=(ANA,)),
}

# (raw score, published score, severity) as of 2026-07-24.
GOLDEN_SCORES: dict[str, tuple[float, int, str]] = {
    "readme_order_details": (61.6, 62, "high"),
    "readme_customer_analysis_chart": (61.3, 61, "high"),
    "type_dashboard": (50.0, 50, "moderate"),
    "type_chart": (47.0, 47, "moderate"),
    "type_mlmodel": (50.0, 50, "moderate"),
    "type_mlfeaturetable": (44.0, 44, "moderate"),
    "type_datajob": (41.0, 41, "moderate"),
    "type_dataflow": (41.0, 41, "moderate"),
    "type_dataset": (35.0, 35, "moderate"),
    "type_notebook": (32.0, 32, "low"),
    "type_unrecognised": (32.0, 32, "low"),
    "proximity_transform_hop_1": (40.0, 40, "moderate"),
    "proximity_transform_hop_2": (35.0, 35, "moderate"),
    "proximity_transform_hop_3": (30.0, 30, "low"),
    "proximity_transform_hop_4": (25.0, 25, "low"),
    "proximity_transform_hop_5": (20.0, 20, "low"),
    "proximity_transform_hop_6": (15.0, 15, "low"),
    "proximity_transform_hop_12": (15.0, 15, "low"),
    "fanout_0": (40.0, 40, "moderate"),
    "fanout_1": (44.3, 44, "moderate"),
    "fanout_5": (51.0, 51, "moderate"),
    "fanout_14": (56.6, 57, "high"),
    "fanout_25": (60.0, 60, "high"),
    "fanout_200": (60.0, 60, "high"),
    "fanout_10000": (60.0, 60, "high"),
    "usage_0": (40.0, 40, "moderate"),
    "usage_1": (41.8, 42, "moderate"),
    "usage_10": (46.1, 46, "moderate"),
    "usage_50": (50.0, 50, "moderate"),
    "usage_500": (50.0, 50, "moderate"),
    "unknown_not_expanded": (40.0, 40, "moderate"),
    "unknown_not_enriched": (40.0, 40, "moderate"),
    "unknown_usage": (40.0, 40, "moderate"),
    "orphan": (50.0, 50, "moderate"),
    "owned": (40.0, 40, "moderate"),
    "governance_marked": (45.0, 45, "moderate"),
    "governance_plain": (40.0, 40, "moderate"),
    "worst_case": (100.0, 100, "critical"),
    "best_case": (12.0, 12, "low"),
}


def test_every_case_has_a_golden_value():
    """Guards the guard: a new case with no pinned value would silently not be checked."""
    assert set(CASES) == set(GOLDEN_SCORES)


def test_scores_match_the_pinned_values():
    """Compares the whole table at once, so a drift shows every affected case."""
    actual = {}
    for label, node in CASES.items():
        assessment = score_node(node, WEIGHTS, MAX_HOPS)
        actual[label] = (
            assessment.score,
            assessment.published_score,
            assessment.severity.value,
        )

    assert actual == GOLDEN_SCORES


@pytest.mark.parametrize(
    ("label", "published_score"),
    [
        # README.md prints "1 hop downstream; 14 assets depend on it in turn" at 61.6
        ("readme_order_details", 61.6),
        # and "Chart consumer; 3 hops downstream (2 transforms deep)" at 61.3.
        ("readme_customer_analysis_chart", 61.3),
    ],
)
def test_the_numbers_printed_in_the_readme_still_reproduce(label, published_score):
    """The README's sample output is a real run. Keep it a true one.

    If this fails, the README output block and docs/screenshots are now showing
    numbers the code no longer produces.
    """
    assert score_node(CASES[label], WEIGHTS, MAX_HOPS).score == published_score


def test_saturation_points_stay_where_they_are():
    """Both curves flatten deliberately. A change here rescales every real run."""
    assert GOLDEN_SCORES["fanout_25"] == GOLDEN_SCORES["fanout_10000"]
    assert GOLDEN_SCORES["usage_50"] == GOLDEN_SCORES["usage_500"]
    # Proximity floors at zero rather than going negative past the budget.
    assert GOLDEN_SCORES["proximity_transform_hop_6"] == GOLDEN_SCORES["proximity_transform_hop_12"]


def test_the_ceiling_is_the_whole_weight_budget():
    assert GOLDEN_SCORES["worst_case"][0] == WEIGHTS.total() == 100.0


def test_the_three_unknown_flags_score_identically_to_a_clean_absence():
    """Not the same claim as the behavioral suite makes.

    test_scoring.py proves an unknown is not scored as a positive finding. This
    pins the actual number, so a refactor cannot quietly start charging a
    fraction of the weight for "we did not look".
    """
    baseline = GOLDEN_SCORES["owned"][0]
    assert GOLDEN_SCORES["unknown_not_expanded"][0] == baseline
    assert GOLDEN_SCORES["unknown_usage"][0] == baseline
    # Not enriched means ownership and markers are both unknown, and an unknown
    # owner must not attract the orphan penalty that "orphan" carries.
    assert GOLDEN_SCORES["unknown_not_enriched"][0] == baseline
    assert GOLDEN_SCORES["orphan"][0] > baseline


# A report shaped to exercise every optional section and both disclosure notes.
SNAPSHOT_NODES = [
    make_node(
        name="Executive Summary",
        entity_type="DASHBOARD",
        hop=5,
        transform_hop=3,
        query_count=40,
        owners=(ANA,),
        tags=("Tier1",),
    ),
    make_node(
        name="Customer Analysis",
        entity_type="CHART",
        hop=3,
        transform_hop=2,
        downstream_count=1,
        query_count=12,
    ),
    make_node(
        name="customer_analytics_measures",
        entity_type="DATASET",
        hop=2,
        transform_hop=2,
        downstream_count=9,
        query_count=3,
        owners=(PLATFORM,),
    ),
    make_node(
        name="stg_orders", entity_type="DATASET", hop=1, transform_hop=1, downstream_count=14
    ),
    make_node(
        name="ml_churn_features",
        entity_type="MLFEATURETABLE",
        hop=4,
        transform_hop=3,
        enriched=False,
        usage_known=False,
    ),
]


def snapshot_report() -> BlastReport:
    return BlastReport(
        root=Entity(
            urn="urn:li:dataset:(urn:li:dataPlatform:dbt,shop.order_details,PROD)",
            entity_type="DATASET",
            name="order_details",
        ),
        change_summary="dropping column promo_code",
        generated_at=datetime(2026, 7, 24, 9, 30, tzinfo=UTC),
        assessments=score_all(SNAPSHOT_NODES, WEIGHTS, MAX_HOPS),
        hops_traversed=5,
        truncated=True,
        capped_lineage={
            "urn:li:dataset:(urn:li:dataPlatform:dbt,stg_orders,PROD)": CappedLineage(
                returned=100, total=137
            ),
            "urn:li:dataset:(urn:li:dataPlatform:dbt,other,PROD)": CappedLineage(
                returned=100, total=None
            ),
        },
    )


# Held as lines rather than one triple-quoted block so the markdown rule below the
# "Who to tell" section stays a quoted token instead of a bare line in the source.
EXPECTED_BRIEF = "\n".join(
    [
        "# Blast radius: order_details",
        "",
        "**Change:** dropping column promo_code",
        "**Asset:** urn:li:dataset:(urn:li:dataPlatform:dbt,shop.order_details,PROD)",
        "**Scanned:** 5 downstream assets across 5 hops at 2026-07-24 09:30 UTC",
        "**Note:** Traversal hit its node budget, so the cone reported here is a subset, "
        "and which consumers survived is arbitrary with respect to distance. Raise "
        "`BLAST_RADAR_MAX_NODES` for the full picture.",
        "**Note:** 2 asset(s) returned fewer consumers than DataHub holds. 37 consumer(s) "
        "are missing from this scan, plus an unstated number from 1 asset(s) whose reply "
        "carried no usable total. Raise `BLAST_RADAR_RESULTS_PER_HOP` and re-run.",
        "",
        "**Verdict:** 3 high-or-critical consumers. Notify the owners below before this ships.",
        "",
        "## Ranked impact",
        "",
        "| # | Asset | Type | Score | Severity | Why |",
        "|---|---|---|---:|---|---|",
        "| 1 | Customer Analysis | Chart | 67.8 | HIGH | Chart consumer; 3 hops downstream "
        "of the change (2 transforms deep) |",
        "| 2 | stg_orders | Dataset | 66.6 | HIGH | 1 hop downstream of the change; "
        "14 assets depend on it in turn |",
        "| 3 | Executive Summary | Dashboard | 59.4 | HIGH | Dashboard consumer; 5 hops "
        "downstream of the change (3 transforms deep) |",
        "| 4 | customer_analytics_measures | Dataset | 52.7 | MODERATE | 2 hops downstream "
        "of the change; Dataset consumer |",
        "| 5 | ml_churn_features | ML feature table | 39.0 | MODERATE | "
        "ML feature table consumer; "
        "4 hops downstream of the change (3 transforms deep) |",
        "",
        "## Impacted with no owner to notify",
        "",
        "2 impacted asset(s) have nobody to notify:",
        "",
        "- Customer Analysis (score 67.8)",
        "- stg_orders (score 66.6)",
        "",
        "## Ownership not checked",
        "",
        "1 impacted asset(s) could not be enriched, so their ownership is unknown rather "
        "than absent:",
        "",
        "- ml_churn_features (score 39.0)",
        "",
        "## Who to tell",
        "",
        "- **Ana Ortiz** - Executive Summary",
        "",
        "---",
        "Generated by [Blast Radar](https://github.com/voyagi/datahub-blast-radar). "
        "Scores are deterministic: same graph in, same ranking out.",
    ]
)


def test_the_rendered_brief_is_byte_for_byte_what_it_was():
    """The brief is written back into DataHub as a document, so its text is a contract.

    A published brief is found and updated by title on later runs. Wording drift
    here changes what a data engineer reads on an asset page, and the same string
    is what the terminal prints.
    """
    assert render_markdown(snapshot_report()) == EXPECTED_BRIEF


def test_the_summary_line_is_what_it_was():
    """This line becomes the tag description and the terminal headline."""
    assert render_summary(snapshot_report()) == (
        "5 downstream assets, 0 critical, 3 high, 2 without an owner"
    )


def test_ranking_order_and_its_tiebreak_hold():
    """Rank is by score, then by dependent count. Both halves are pinned."""
    ranked = snapshot_report().ranked

    assert [item.node.entity.name for item in ranked] == [
        "Customer Analysis",
        "stg_orders",
        "Executive Summary",
        "customer_analytics_measures",
        "ml_churn_features",
    ]

    # The tiebreak only decides anything when two scores are exactly equal,
    # and dependent count feeds the fan-out factor, so two real nodes that
    # differ in dependent count also differ in score and never reach it. The
    # two assessments below are given one identical factor, so their scores
    # match, and made to differ only in dependent count. `ranked` must then
    # put the wider-blast asset first, and the input order is the opposite so
    # that only the sort can produce it.
    def tied(name: str, downstream_count: int) -> RiskAssessment:
        node = make_node(
            name=name,
            entity_type="DATASET",
            hop=1,
            transform_hop=1,
            downstream_count=downstream_count,
            owners=(ANA,),
        )
        return RiskAssessment(node=node, factors=[RiskFactor("pinned", 60.0, 1.0, "equal")])

    tie_report = BlastReport(
        root=Entity(urn="urn:li:dataset:root", entity_type="DATASET", name="root"),
        change_summary="a change",
        generated_at=datetime(2026, 7, 27, tzinfo=UTC),
        assessments=[tied("fewer_dependents", 2), tied("more_dependents", 9)],
        hops_traversed=1,
    )

    assert [item.score for item in tie_report.ranked] == [60.0, 60.0]
    assert [item.node.entity.name for item in tie_report.ranked] == [
        "more_dependents",
        "fewer_dependents",
    ]
