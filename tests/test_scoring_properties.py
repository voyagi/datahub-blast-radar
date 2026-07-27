"""Invariants of the risk model, checked over generated inputs.

The example tests in `test_scoring.py` pin behavior at points someone thought
of. These pin the rules that have to hold at every point, because the score is
the product: it sits in a release gate, it is published onto assets in a
catalog, and a ranking that can invert on an input nobody tried is worse than
no ranking at all.

The decision table at the bottom is the other half. Severity is banded on the
published (rounded) score, and the boundaries are exactly where a band gets
decided wrongly, so every value from 0 to 100 is checked rather than sampled.
"""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from blast_radar.config import ScoringWeights
from blast_radar.models import (
    BlastReport,
    ConsumerNode,
    Entity,
    Owner,
    RiskAssessment,
    RiskFactor,
    Severity,
)
from blast_radar.scoring import CONSUMER_TYPE_WEIGHTS, score_node

# Rounding to one decimal can lift a sum by at most half of the last place.
ROUNDING_SLACK = 0.05

ENTITY_TYPES = st.sampled_from([*CONSUMER_TYPE_WEIGHTS, "GLOSSARYNODE", "", "dataset"])

weights = st.builds(
    ScoringWeights,
    consumer_type=st.floats(0, 50),
    proximity=st.floats(0, 50),
    fan_out=st.floats(0, 50),
    usage=st.floats(0, 50),
    ownership_gap=st.floats(0, 50),
    governance_signal=st.floats(0, 50),
)


@st.composite
def nodes(draw, **overrides):
    """A consumer node with every field the score reads, including the flags."""
    fields = {
        "hop": draw(st.integers(1, 12)),
        "transform_hop": draw(st.integers(0, 12)),
        "downstream_count": draw(st.integers(0, 500)),
        "query_count": draw(st.integers(0, 5000)),
        "expanded": draw(st.booleans()),
        "enriched": draw(st.booleans()),
        "usage_known": draw(st.booleans()),
    }
    entity = Entity(
        urn=f"urn:li:dataset:generated{fields['hop']}",
        entity_type=draw(ENTITY_TYPES),
        name="generated",
        owners=draw(st.lists(st.just(Owner("urn:li:corpuser:ana", "Ana")), max_size=3)),
        tags=draw(st.lists(st.sampled_from(["tier1", "pii", "sandbox", "draft"]), max_size=4)),
    )
    return ConsumerNode(entity=entity, **{**fields, **overrides})


def replace(node: ConsumerNode, **changes) -> ConsumerNode:
    """A copy of `node` with some fields changed, for a one-variable comparison."""
    entity = changes.pop("entity", node.entity)
    fields = {
        "hop": node.hop,
        "transform_hop": node.transform_hop,
        "downstream_count": node.downstream_count,
        "query_count": node.query_count,
        "expanded": node.expanded,
        "enriched": node.enriched,
        "usage_known": node.usage_known,
        **changes,
    }
    return ConsumerNode(entity=entity, **fields)


@given(node=nodes(), scoring_weights=weights, max_hops=st.integers(1, 10))
def test_a_score_never_leaves_the_weight_budget(node, scoring_weights, max_hops):
    """The published number is read as "out of 100", so it has to be."""
    score = score_node(node, scoring_weights, max_hops).score

    assert 0.0 <= score <= scoring_weights.total() + ROUNDING_SLACK


@given(node=nodes(), scoring_weights=weights, max_hops=st.integers(1, 10))
def test_every_factor_is_normalized_and_carries_its_evidence(node, scoring_weights, max_hops):
    for factor in score_node(node, scoring_weights, max_hops).factors:
        assert 0.0 <= factor.normalized <= 1.0
        assert factor.evidence.strip()


@given(node=nodes(), scoring_weights=weights, max_hops=st.integers(1, 10))
def test_the_same_graph_scores_the_same_way_twice(node, scoring_weights, max_hops):
    """Determinism is the reason this is not an LLM. A gate cannot act on a
    number that moves between two runs of the same scan."""
    first = score_node(node, scoring_weights, max_hops)
    second = score_node(node, scoring_weights, max_hops)

    assert first.score == second.score
    assert [factor.evidence for factor in first.factors] == [
        factor.evidence for factor in second.factors
    ]


@given(
    node=nodes(expanded=True),
    extra=st.integers(1, 500),
    scoring_weights=weights,
    max_hops=st.integers(1, 10),
)
def test_more_dependents_never_lower_the_score(node, extra, scoring_weights, max_hops):
    wider = replace(node, downstream_count=node.downstream_count + extra)

    assert (
        score_node(wider, scoring_weights, max_hops).score
        >= score_node(node, scoring_weights, max_hops).score
    )


@given(
    node=nodes(usage_known=True),
    extra=st.integers(1, 5000),
    scoring_weights=weights,
    max_hops=st.integers(1, 10),
)
def test_more_queries_never_lower_the_score(node, extra, scoring_weights, max_hops):
    busier = replace(node, query_count=node.query_count + extra)

    assert (
        score_node(busier, scoring_weights, max_hops).score
        >= score_node(node, scoring_weights, max_hops).score
    )


@given(node=nodes(enriched=True), scoring_weights=weights, max_hops=st.integers(1, 10))
def test_finding_an_owner_never_raises_the_risk(node, scoring_weights, max_hops):
    """The ownership factor exists because nobody gets the heads-up. Someone to
    tell can only make the change safer to ship."""
    owned = replace(
        node,
        entity=Entity(
            urn=node.entity.urn,
            entity_type=node.entity.entity_type,
            name=node.entity.name,
            owners=[Owner("urn:li:corpuser:ana", "Ana")],
            tags=list(node.entity.tags),
        ),
    )
    unowned = replace(
        node,
        entity=Entity(
            urn=node.entity.urn,
            entity_type=node.entity.entity_type,
            name=node.entity.name,
            owners=[],
            tags=list(node.entity.tags),
        ),
    )

    assert (
        score_node(owned, scoring_weights, max_hops).score
        <= score_node(unowned, scoring_weights, max_hops).score
    )


@given(
    node=nodes(),
    extra=st.integers(1, 10),
    scoring_weights=weights,
    max_hops=st.integers(1, 10),
)
def test_distance_never_raises_the_risk(node, extra, scoring_weights, max_hops):
    """Each transform in between is another chance to absorb the change."""
    further = replace(node, transform_hop=node.transform_hop + extra)

    assert (
        score_node(further, scoring_weights, max_hops).score
        <= score_node(node, scoring_weights, max_hops).score
    )


@given(node=nodes(), scoring_weights=weights, max_hops=st.integers(1, 10))
def test_a_dashboard_is_never_the_safer_bet_than_a_dataset(node, scoring_weights, max_hops):
    """Where a human sees the breakage is the strongest single predictor, so
    the type weighting must not invert for any other combination of factors."""
    dashboard = replace(node, entity=_retyped(node.entity, "DASHBOARD"))
    dataset = replace(node, entity=_retyped(node.entity, "DATASET"))

    assert (
        score_node(dashboard, scoring_weights, max_hops).score
        >= score_node(dataset, scoring_weights, max_hops).score
    )


@given(node=nodes(), scoring_weights=weights, max_hops=st.integers(1, 10))
def test_an_unchecked_signal_is_scored_exactly_as_a_zero_one_is_not_worse(
    node, scoring_weights, max_hops
):
    """The whole "unknown is not zero" discipline in one property: not having
    looked contributes nothing, so it can never inflate a rank the way a
    guessed value would."""
    unchecked = replace(node, expanded=False, enriched=False, usage_known=False)
    assessment = score_node(unchecked, scoring_weights, max_hops)
    factors = {factor.name: factor for factor in assessment.factors}

    assert factors["fan_out"].contribution == 0.0
    assert factors["usage"].contribution == 0.0
    assert factors["ownership_gap"].contribution == 0.0
    assert factors["governance_signal"].contribution == 0.0
    assert "not checked" in factors["usage"].evidence
    assert "unknown" in factors["fan_out"].evidence


def _retyped(entity: Entity, entity_type: str) -> Entity:
    return Entity(
        urn=entity.urn,
        entity_type=entity_type,
        name=entity.name,
        owners=list(entity.owners),
        tags=list(entity.tags),
    )


@given(node_list=st.lists(nodes(), min_size=1, max_size=8))
def test_a_report_never_files_an_unchecked_asset_as_an_orphan(node_list):
    """The two lists are disjoint and every node lands in exactly one of them
    or neither. An asset in both would be reported as a governance gap and as
    unknown at once; an asset in the wrong one sends somebody hunting."""
    from datetime import UTC, datetime

    report = BlastReport(
        root=Entity(urn="urn:li:dataset:root", entity_type="DATASET", name="root"),
        change_summary="a change",
        generated_at=datetime(2026, 7, 24, tzinfo=UTC),
        assessments=[score_node(node, ScoringWeights(), 5) for node in node_list],
        hops_traversed=1,
    )

    unowned = {id(item.node) for item in report.unowned}
    unchecked = {id(item.node) for item in report.unchecked_ownership}

    # Set equality against the definition, not `all(...)` over whatever landed
    # in each list: every assertion of that shape holds when both lists come
    # back empty, so a report that filed nothing anywhere would pass.
    assert unowned == {
        id(node) for node in node_list if node.enriched and not node.entity.has_owner
    }
    assert unchecked == {id(node) for node in node_list if not node.enriched}
    assert unowned.isdisjoint(unchecked)


# --- Severity decision table ------------------------------------------------
#
# Every published score from 0 to 100, not a sample. The bands are what people
# act on and what the tag descriptions promise, and the boundary is the only
# interesting part of a band.


def assessment_scoring(value: float) -> RiskAssessment:
    """An assessment whose score is exactly `value`."""
    node = ConsumerNode(
        entity=Entity(urn="urn:li:dataset:x", entity_type="DATASET", name="x"), hop=1
    )
    return RiskAssessment(node=node, factors=[RiskFactor("only", value, 1.0, "generated")])


def expected_band(published: int) -> Severity:
    if published >= 75:
        return Severity.CRITICAL
    if published >= 55:
        return Severity.HIGH
    if published >= 35:
        return Severity.MODERATE
    return Severity.LOW


def test_every_published_score_falls_in_the_band_the_tag_descriptions_promise():
    for published in range(0, 101):
        assessment = assessment_scoring(float(published))

        assert assessment.published_score == published
        assert assessment.severity is expected_band(published), published


def test_the_bands_change_exactly_where_they_say_they_do():
    boundaries = {34: Severity.LOW, 35: Severity.MODERATE, 54: Severity.MODERATE}
    boundaries |= {55: Severity.HIGH, 74: Severity.HIGH, 75: Severity.CRITICAL}

    for published, severity in boundaries.items():
        assert assessment_scoring(float(published)).severity is severity


def test_a_score_that_rounds_up_into_a_band_is_tagged_with_the_band_it_publishes():
    """74.6 publishes as 75 and has to tag critical, or the asset page shows a
    number the tag description says belongs to the band above."""
    assessment = assessment_scoring(74.6)

    assert assessment.published_score == 75
    assert assessment.severity is Severity.CRITICAL


def test_severity_ranks_order_the_bands():
    ranks = [severity.rank for severity in (Severity.LOW, Severity.MODERATE, Severity.HIGH)]

    assert ranks == sorted(ranks)
    assert Severity.CRITICAL.rank > Severity.HIGH.rank
