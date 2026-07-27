"""The deterministic risk model.

Design decision worth stating plainly: the score is NOT produced by a language
model. Ranking "what breaks if this ships" has to be reproducible, auditable,
and identical on two consecutive runs, or a platform team cannot put it in a
release gate. An LLM writes the narrative on top of these numbers; it never
moves them.

Every factor returns a normalized 0..1 value plus the evidence behind it, so
the brief can always answer "why is this ranked first".
"""

from __future__ import annotations

import math

from .config import ScoringWeights
from .models import ConsumerNode, Entity, RiskAssessment, RiskFactor

# Consumers are not equal. A dashboard a director opens every morning is a
# louder failure than an intermediate staging table with one reader.
CONSUMER_TYPE_WEIGHTS: dict[str, float] = {
    "DASHBOARD": 1.0,
    "CHART": 0.9,
    "MLMODEL": 1.0,
    "MLFEATURETABLE": 0.8,
    "DATAJOB": 0.7,
    "DATAFLOW": 0.7,
    "DATASET": 0.5,
    "NOTEBOOK": 0.4,
}
DEFAULT_CONSUMER_WEIGHT = 0.4

# Tags and terms that mean "this one is load-bearing". Matched case-insensitively
# on a substring basis because every organisation spells them differently.
GOVERNANCE_MARKERS: tuple[str, ...] = (
    "tier1",
    "tier 1",
    "critical",
    "gold",
    "certified",
    "pii",
    "gdpr",
    "financial",
    "regulatory",
    "sox",
)


def score_node(node: ConsumerNode, weights: ScoringWeights, max_hops: int) -> RiskAssessment:
    """Score one downstream consumer."""
    factors = [
        _proximity(node, weights, max_hops),
        _fan_out(node, weights),
        _consumer_type(node, weights),
        _usage(node, weights),
        _ownership_gap(node, weights),
        _governance(node, weights),
    ]
    return RiskAssessment(node=node, factors=factors)


def score_all(
    nodes: list[ConsumerNode], weights: ScoringWeights, max_hops: int
) -> list[RiskAssessment]:
    return [score_node(node, weights, max_hops) for node in nodes]


def _proximity(node: ConsumerNode, weights: ScoringWeights, max_hops: int) -> RiskFactor:
    """Closer to the change means less chance of a transform absorbing it.

    Distance is counted in transforms, not in graph edges. A dashboard four
    edges away whose data passed through two transforms is exactly as exposed
    as the chart it contains: the last two edges rendered the data, they did
    not reshape it.
    """
    transform_hop = max(node.transform_hop, 1)
    # Linear decay across the traversal depth, floored at zero for anything
    # beyond it, so a deep node never scores negative.
    normalized = max(0.0, 1.0 - (transform_hop - 1) / max(max_hops, 1))

    hop = max(node.hop, 1)
    evidence = f"{hop} hop{'s' if hop != 1 else ''} downstream of the change"
    if transform_hop != hop:
        evidence += f" ({transform_hop} transform{'s' if transform_hop != 1 else ''} deep)"

    return RiskFactor(
        name="proximity",
        weight=weights.proximity,
        normalized=normalized,
        evidence=evidence,
    )


def _fan_out(node: ConsumerNode, weights: ScoringWeights) -> RiskFactor:
    """A consumer that feeds many others multiplies the breakage."""
    if not node.expanded:
        # The walk stopped here, so nothing is known about what sits beyond it.
        # Score it as zero but say why, rather than implying it is a leaf.
        return RiskFactor(
            name="fan_out",
            weight=weights.fan_out,
            normalized=0.0,
            evidence="dependents unknown - traversal depth reached here",
        )

    count = max(node.downstream_count, 0)
    # log1p keeps a table with 200 dependents from swamping one with 20; the
    # difference that matters is 0 vs 5, not 100 vs 200.
    normalized = min(1.0, math.log1p(count) / math.log1p(25))
    return RiskFactor(
        name="fan_out",
        weight=weights.fan_out,
        normalized=normalized,
        evidence=f"{count} asset{'s' if count != 1 else ''} depend on it in turn",
    )


def _consumer_type(node: ConsumerNode, weights: ScoringWeights) -> RiskFactor:
    entity_type = (node.entity.entity_type or "").upper()
    normalized = CONSUMER_TYPE_WEIGHTS.get(entity_type, DEFAULT_CONSUMER_WEIGHT)
    return RiskFactor(
        name="consumer_type",
        weight=weights.consumer_type,
        normalized=normalized,
        # The weight keys off the raw type; the evidence is for a person, and
        # is worded by the same map both renderers use.
        evidence=f"{node.entity.display_type} consumer",
    )


def _usage(node: ConsumerNode, weights: ScoringWeights) -> RiskFactor:
    """Query volume is the closest thing to "someone will notice"."""
    if not node.usage_known:
        return RiskFactor(
            name="usage",
            weight=weights.usage,
            normalized=0.0,
            evidence="query history not checked",
        )

    count = max(node.query_count, 0)
    normalized = min(1.0, math.log1p(count) / math.log1p(50))
    return RiskFactor(
        name="usage",
        weight=weights.usage,
        normalized=normalized,
        evidence=f"{count} recent quer{'ies' if count != 1 else 'y'} reference it",
    )


def _ownership_gap(node: ConsumerNode, weights: ScoringWeights) -> RiskFactor:
    """No owner means no one gets the heads-up, so the risk is higher.

    Not having looked is not the same as there being nobody. Scoring an
    unenriched asset as ownerless both inflates its rank by the full weight
    and puts it in the brief's "nobody to notify" list, which is a governance
    gap someone then goes hunting for.
    """
    if not node.ownership_known:
        return RiskFactor(
            name="ownership_gap",
            weight=weights.ownership_gap,
            normalized=0.0,
            evidence="ownership not checked",
        )

    has_owner = node.entity.has_owner
    return RiskFactor(
        name="ownership_gap",
        weight=weights.ownership_gap,
        normalized=0.0 if has_owner else 1.0,
        evidence=(
            f"owned by {', '.join(owner.name for owner in node.entity.owners)}"
            if has_owner
            else "no owner to notify"
        ),
    )


def _governance(node: ConsumerNode, weights: ScoringWeights) -> RiskFactor:
    if not node.enriched:
        # Tags and glossary terms arrive with enrichment, so their absence
        # before it means nothing at all.
        return RiskFactor(
            name="governance_signal",
            weight=weights.governance_signal,
            normalized=0.0,
            evidence="criticality markers not checked",
        )

    matches = _governance_markers(node.entity)
    return RiskFactor(
        name="governance_signal",
        weight=weights.governance_signal,
        normalized=1.0 if matches else 0.0,
        evidence=(f"marked {', '.join(sorted(matches))}" if matches else "no criticality markers"),
    )


def _governance_markers(entity: Entity) -> set[str]:
    """Return the criticality markers found on an entity's tags and terms."""
    labels = [label.lower() for label in (*entity.tags, *entity.glossary_terms)]
    found: set[str] = set()
    for label in labels:
        for marker in GOVERNANCE_MARKERS:
            if marker in label:
                found.add(label)
    return found
