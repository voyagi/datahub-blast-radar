"""Tests for the deterministic risk model.

These run with no DataHub instance and no network, which is the whole reason
the scoring layer takes plain data rather than MCP responses.
"""

from __future__ import annotations

import pytest

from blast_radar.config import ScoringWeights
from blast_radar.models import ConsumerNode, Entity, Owner, RiskAssessment, RiskFactor, Severity
from blast_radar.scoring import score_node

WEIGHTS = ScoringWeights()
MAX_HOPS = 4


def make_node(
    *,
    hop: int = 1,
    transform_hop: int | None = None,
    entity_type: str = "DATASET",
    downstream_count: int = 0,
    query_count: int = 0,
    owners: list[Owner] | None = None,
    tags: list[str] | None = None,
    terms: list[str] | None = None,
) -> ConsumerNode:
    """Build a node. transform_hop defaults to hop, i.e. a pure dataset chain."""
    entity = Entity(
        urn=f"urn:li:dataset:(urn:li:dataPlatform:snowflake,test.{entity_type},PROD)",
        entity_type=entity_type,
        name=f"test.{entity_type.lower()}",
        owners=owners or [],
        tags=tags or [],
        glossary_terms=terms or [],
    )
    return ConsumerNode(
        entity=entity,
        hop=hop,
        transform_hop=hop if transform_hop is None else transform_hop,
        downstream_count=downstream_count,
        query_count=query_count,
        # These helpers hand the scorer complete metadata, which is what an
        # enriched node looks like. The unknown cases are tested explicitly.
        enriched=True,
        usage_known=True,
    )


def factor(assessment, name: str):
    return next(item for item in assessment.factors if item.name == name)


def test_proximity_decays_with_distance():
    near = score_node(make_node(hop=1), WEIGHTS, MAX_HOPS)
    far = score_node(make_node(hop=4), WEIGHTS, MAX_HOPS)

    assert factor(near, "proximity").normalized == 1.0
    assert factor(far, "proximity").normalized < factor(near, "proximity").normalized
    assert near.score > far.score


def test_proximity_never_goes_negative_beyond_max_hops():
    beyond = score_node(make_node(hop=12), WEIGHTS, MAX_HOPS)

    assert factor(beyond, "proximity").normalized == 0.0


def test_a_dashboard_is_scored_on_transforms_not_on_graph_distance():
    """The case that motivated transform_hop.

    A dashboard four edges out, whose data passed through one transform before
    being charted, must not be discounted like a table four transforms deep.
    """
    dashboard = score_node(
        make_node(hop=4, transform_hop=1, entity_type="DASHBOARD"), WEIGHTS, MAX_HOPS
    )
    deep_table = score_node(
        make_node(hop=4, transform_hop=4, entity_type="DATASET"), WEIGHTS, MAX_HOPS
    )

    assert factor(dashboard, "proximity").normalized == 1.0
    assert dashboard.score > deep_table.score
    assert "4 hops downstream" in factor(dashboard, "proximity").evidence
    assert "1 transform deep" in factor(dashboard, "proximity").evidence


def test_proximity_evidence_stays_simple_when_the_two_agree():
    assessment = score_node(make_node(hop=3), WEIGHTS, MAX_HOPS)

    assert factor(assessment, "proximity").evidence == "3 hops downstream of the change"


def test_fan_out_is_logarithmic_not_linear():
    small = factor(score_node(make_node(downstream_count=5), WEIGHTS, MAX_HOPS), "fan_out")
    large = factor(score_node(make_node(downstream_count=200), WEIGHTS, MAX_HOPS), "fan_out")

    assert large.normalized > small.normalized
    # The point of log scaling: 40x the dependents is nowhere near 40x the score.
    assert large.normalized < small.normalized * 3


def test_dashboards_outrank_datasets_all_else_equal():
    dashboard = score_node(make_node(entity_type="DASHBOARD"), WEIGHTS, MAX_HOPS)
    dataset = score_node(make_node(entity_type="DATASET"), WEIGHTS, MAX_HOPS)

    assert dashboard.score > dataset.score


def test_unknown_entity_type_falls_back_without_crashing():
    assessment = score_node(make_node(entity_type="SOMETHING_NEW"), WEIGHTS, MAX_HOPS)

    assert factor(assessment, "consumer_type").normalized > 0


@pytest.mark.parametrize(
    ("entity_type", "label"),
    [
        ("DATASET", "Dataset"),
        ("MLFEATURETABLE", "ML feature table"),
        ("DATAJOB", "Data job"),
        ("MLMODEL", "ML model"),
        # A type DataHub adds later is still named, just less prettily.
        ("SOMETHING_NEW", "Something_New"),
        ("", "Unknown"),
    ],
)
def test_a_type_is_written_the_way_a_reader_would_write_it(entity_type, label):
    """`.title()` on DataHub's own spelling produces "Mlfeaturetable", which
    is what the brief and the terminal both used to print."""
    assessment = score_node(make_node(entity_type=entity_type), WEIGHTS, MAX_HOPS)

    assert assessment.node.entity.display_type == label
    assert factor(assessment, "consumer_type").evidence == f"{label} consumer"


def test_missing_owner_raises_risk_and_says_so():
    owned = score_node(
        make_node(owners=[Owner(urn="urn:li:corpuser:ana", name="Ana")]), WEIGHTS, MAX_HOPS
    )
    orphan = score_node(make_node(), WEIGHTS, MAX_HOPS)

    assert orphan.score > owned.score
    assert factor(orphan, "ownership_gap").evidence == "no owner to notify"
    assert "Ana" in factor(owned, "ownership_gap").evidence


def test_unchecked_ownership_is_not_scored_as_missing_ownership():
    """The finding that motivated this: unenriched assets were ranked as orphans."""
    unchecked = ConsumerNode(
        entity=Entity(urn="urn:li:dataset:x", entity_type="DATASET", name="x"),
        hop=1,
        transform_hop=1,
        enriched=False,
    )
    orphan = make_node()

    unchecked_assessment = score_node(unchecked, WEIGHTS, MAX_HOPS)
    orphan_assessment = score_node(orphan, WEIGHTS, MAX_HOPS)

    assert factor(unchecked_assessment, "ownership_gap").normalized == 0.0
    assert factor(unchecked_assessment, "ownership_gap").evidence == "ownership not checked"
    assert factor(orphan_assessment, "ownership_gap").normalized == 1.0
    # The phantom penalty is worth the full ownership weight.
    assert orphan_assessment.score > unchecked_assessment.score


def test_unchecked_usage_and_markers_say_so_instead_of_reporting_zero():
    node = ConsumerNode(
        entity=Entity(urn="urn:li:dataset:x", entity_type="DATASET", name="x"),
        hop=1,
        transform_hop=1,
        enriched=False,
        usage_known=False,
    )

    assessment = score_node(node, WEIGHTS, MAX_HOPS)

    assert factor(assessment, "usage").evidence == "query history not checked"
    assert factor(assessment, "governance_signal").evidence == "criticality markers not checked"


def test_governance_markers_match_case_insensitively_on_tags_and_terms():
    tagged = score_node(make_node(tags=["Tier1"]), WEIGHTS, MAX_HOPS)
    termed = score_node(make_node(terms=["Contains PII"]), WEIGHTS, MAX_HOPS)
    plain = score_node(make_node(tags=["sandbox"]), WEIGHTS, MAX_HOPS)

    assert factor(tagged, "governance_signal").normalized == 1.0
    assert factor(termed, "governance_signal").normalized == 1.0
    assert factor(plain, "governance_signal").normalized == 0.0


def assessment_scoring(raw: float) -> RiskAssessment:
    """A RiskAssessment whose raw score is exactly `raw`.

    One synthetic factor carries the whole score, so the band boundary is
    tested on its own rather than through whatever combination of the six real
    factors happens to land near an edge.
    """
    return RiskAssessment(
        node=make_node(),
        factors=[RiskFactor(name="synthetic", weight=raw, normalized=1.0, evidence="")],
    )


def test_severity_bands_on_the_published_score_not_the_raw_one():
    """The number on the asset page is rounded, so the band must round with it.

    74.6 publishes as 75 and has to read CRITICAL; 74.4 publishes as 74 and
    has to read HIGH. Banding the raw score instead would tag 74.6 "high" next
    to a 75 the band table calls critical - the exact mismatch the
    `published_score` banding in models.py exists to prevent.
    """
    just_over = assessment_scoring(74.6)
    just_under = assessment_scoring(74.4)

    assert just_over.score == 74.6
    assert just_over.published_score == 75
    assert just_over.severity is Severity.CRITICAL

    assert just_under.score == 74.4
    assert just_under.published_score == 74
    assert just_under.severity is Severity.HIGH


def test_the_high_boundary_rounds_the_same_way():
    """The 55 edge gates the notify list and the write-back, so pin it too."""
    just_over = assessment_scoring(54.6)
    just_under = assessment_scoring(54.4)

    assert just_over.published_score == 55
    assert just_over.severity is Severity.HIGH
    assert just_under.published_score == 54
    assert just_under.severity is Severity.MODERATE


def test_worst_case_lands_critical_and_best_case_lands_low():
    worst = score_node(
        make_node(
            hop=1,
            entity_type="DASHBOARD",
            downstream_count=30,
            query_count=80,
            tags=["Tier1"],
        ),
        WEIGHTS,
        MAX_HOPS,
    )
    best = score_node(
        make_node(
            hop=4,
            entity_type="NOTEBOOK",
            owners=[Owner(urn="urn:li:corpuser:ana", name="Ana")],
        ),
        WEIGHTS,
        MAX_HOPS,
    )

    assert worst.severity is Severity.CRITICAL
    assert best.severity is Severity.LOW


def test_score_never_exceeds_the_weight_budget():
    maxed = score_node(
        make_node(
            hop=1,
            entity_type="DASHBOARD",
            downstream_count=10_000,
            query_count=10_000,
            tags=["tier1", "pii"],
        ),
        WEIGHTS,
        MAX_HOPS,
    )

    assert maxed.score <= WEIGHTS.total()


def test_scoring_is_reproducible():
    node = make_node(hop=2, downstream_count=7, query_count=12, tags=["gold"])

    assert score_node(node, WEIGHTS, MAX_HOPS).score == score_node(node, WEIGHTS, MAX_HOPS).score


def test_top_factors_are_ordered_and_drop_zero_contributors():
    assessment = score_node(
        make_node(hop=1, owners=[Owner(urn="urn:li:corpuser:ana", name="Ana")]),
        WEIGHTS,
        MAX_HOPS,
    )

    contributions = [item.contribution for item in assessment.top_factors]
    assert contributions == sorted(contributions, reverse=True)
    assert all(item.contribution > 0 for item in assessment.top_factors)
    assert "ownership_gap" not in {item.name for item in assessment.top_factors}
