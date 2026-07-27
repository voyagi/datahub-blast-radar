"""Domain models for a blast-radius scan.

These are deliberately free of any DataHub or MCP types. The scoring model is
pure data in, pure data out, which is what makes it testable without a running
instance and explainable in the brief.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum


class Severity(StrEnum):
    """Severity bands, lowest string form being what DataHub stores.

    StrEnum rather than `(str, Enum)`: the two differ only in what `str()` and
    an f-string produce, and every site that publishes a severity already reads
    `.value` explicitly, so the tag URNs and the brief labels are unaffected.
    """

    CRITICAL = "critical"
    HIGH = "high"
    MODERATE = "moderate"
    LOW = "low"

    @property
    def rank(self) -> int:
        return {"critical": 3, "high": 2, "moderate": 1, "low": 0}[self.value]


@dataclass(frozen=True)
class Owner:
    urn: str
    name: str
    ownership_type: str | None = None


# DataHub's entity types are one upper-case word, which `.title()` renders as
# "Mlfeaturetable" and "Datajob". The scoring model keys off the raw string;
# this is only how it is written down for a person.
ENTITY_TYPE_LABELS = {
    "DATASET": "Dataset",
    "DASHBOARD": "Dashboard",
    "CHART": "Chart",
    "DATAJOB": "Data job",
    "DATAFLOW": "Data flow",
    "MLMODEL": "ML model",
    "MLFEATURETABLE": "ML feature table",
    "NOTEBOOK": "Notebook",
    "DATAPRODUCT": "Data product",
    "CONTAINER": "Container",
    "UNKNOWN": "Unknown",
}


@dataclass
class Entity:
    """A DataHub entity, flattened to the fields the risk model cares about."""

    urn: str
    entity_type: str
    name: str
    platform: str | None = None
    description: str | None = None
    owners: list[Owner] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)
    glossary_terms: list[str] = field(default_factory=list)
    domain: str | None = None

    @property
    def has_owner(self) -> bool:
        return bool(self.owners)

    @property
    def display_type(self) -> str:
        """The entity type as a reader would write it.

        One definition, both renderers and the scoring evidence. A type
        DataHub adds after this map was written falls back to title case,
        which is merely ugly rather than wrong, and still names the type it
        was given.
        """
        declared = self.entity_type.upper()
        return ENTITY_TYPE_LABELS.get(declared) or declared.title() or "Unknown"


@dataclass
class ConsumerNode:
    """One node in the downstream cone of the changed asset."""

    entity: Entity
    hop: int
    # Hops that actually transform data, which is not the same as graph
    # distance. A chart is a view of a dataset and a dashboard is a frame
    # around charts; neither can absorb an upstream column change the way a
    # SQL transform can. Proximity scores off this, `hop` is for display.
    transform_hop: int = 1
    downstream_count: int = 0
    query_count: int = 0
    # Three separate "we did not look" flags, and none of them may be reported
    # as a zero. An asset whose ownership was never fetched is not an asset
    # with no owner, and saying so in a published brief sends people chasing a
    # governance gap that does not exist.
    #
    # False when the walk stopped before expanding this node, which means
    # downstream_count is unknown rather than zero.
    expanded: bool = True
    # False until full metadata was fetched: owners, tags, and terms are all
    # unknown rather than absent.
    enriched: bool = False
    # False when query history was not requested or could not be read.
    usage_known: bool = False

    @property
    def ownership_known(self) -> bool:
        return self.enriched

    @property
    def urn(self) -> str:
        return self.entity.urn


@dataclass(frozen=True)
class RiskFactor:
    """One scored dimension, kept with its evidence.

    The evidence string is what turns a number into something a data engineer
    will act on, so it travels with the score rather than being regenerated
    for display.
    """

    name: str
    weight: float
    normalized: float
    evidence: str

    @property
    def contribution(self) -> float:
        return self.weight * self.normalized


@dataclass
class RiskAssessment:
    node: ConsumerNode
    factors: list[RiskFactor]

    @property
    def score(self) -> float:
        return round(sum(factor.contribution for factor in self.factors), 1)

    @property
    def published_score(self) -> int:
        """The score as written to DataHub and shown on the asset page.

        Whole numbers, because DataHub stores numeric properties as 32-bit
        floats and 56.3 renders as 56.2999992370605.
        """
        return round(self.score)

    @property
    def severity(self) -> Severity:
        # Banded on the published number, not the raw one. Otherwise 74.6 is
        # tagged "high" and published as 75, next to a tag description saying
        # 75 and above is critical.
        score = self.published_score
        if score >= 75:
            return Severity.CRITICAL
        if score >= 55:
            return Severity.HIGH
        if score >= 35:
            return Severity.MODERATE
        return Severity.LOW

    @property
    def top_factors(self) -> list[RiskFactor]:
        """Factors that actually moved the score, largest first."""
        return sorted(
            (factor for factor in self.factors if factor.contribution > 0),
            key=lambda factor: factor.contribution,
            reverse=True,
        )

    def notify(self) -> list[Owner]:
        return self.node.entity.owners


@dataclass(frozen=True)
class CappedLineage:
    """A lineage reply that held fewer consumers than DataHub says exist.

    Pure data, so it lives with the rest of the domain models rather than in
    the adapter that fills it in - which is also what lets `BlastReport` name
    it as a type instead of reaching through a `getattr` shim.
    """

    returned: int
    # None when the server flagged more results without saying how many.
    total: int | None

    @property
    def missing(self) -> int | None:
        """How many consumers this reply left out, or None when unknowable.

        Two shapes are unknowable and neither may be reported as zero. The
        server can flag more without a total, and it can send a total that
        does not exceed what it returned - which cannot be a shortfall, so it
        cannot be counted either. This object exists only because something
        was left out, so zero is never the honest answer.
        """
        if self.total is None or self.total <= self.returned:
            return None
        return self.total - self.returned


@dataclass
class BlastReport:
    """The result of one scan, and the payload written back to DataHub."""

    root: Entity
    change_summary: str
    generated_at: datetime
    assessments: list[RiskAssessment]
    hops_traversed: int
    truncated: bool = False
    # Assets whose own downstream list came back capped by the per-hop limit,
    # mapped to what the reply said about the shortfall. Distinct from
    # `truncated`, which is the node budget: this one means a specific asset's
    # consumers are missing from the cone entirely.
    capped_lineage: dict[str, CappedLineage] = field(default_factory=dict)

    @property
    def consumers_missing(self) -> int | None:
        """How many consumers the capped replies left out, where it is known.

        None when not one capped asset came back with a countable shortfall.
        Summing an empty set gives zero, and zero on this line would tell a
        reader the cone is complete on the exact sentence that exists to say
        it is not.
        """
        known = [
            entry.missing for entry in self.capped_lineage.values() if entry.missing is not None
        ]
        if not known:
            return None
        return sum(known)

    @property
    def capped_without_a_count(self) -> int:
        """Capped assets where DataHub did not say how many were left out."""
        return sum(1 for entry in self.capped_lineage.values() if entry.missing is None)

    @property
    def ranked(self) -> list[RiskAssessment]:
        return sorted(
            self.assessments,
            key=lambda item: (item.score, item.node.downstream_count),
            reverse=True,
        )

    def by_severity(self, severity: Severity) -> list[RiskAssessment]:
        return [item for item in self.ranked if item.severity is severity]

    @property
    def unowned(self) -> list[RiskAssessment]:
        """Impacted assets that are known to have nobody to tell.

        This is the finding data teams act on fastest, so it gets its own
        accessor rather than being buried in the factor list. Assets whose
        ownership was never fetched are excluded: listing them here would
        report a governance gap that may not exist.
        """
        return [
            item
            for item in self.ranked
            if item.node.ownership_known and not item.node.entity.has_owner
        ]

    @property
    def unchecked_ownership(self) -> list[RiskAssessment]:
        """Impacted assets whose ownership was never looked up."""
        return [item for item in self.ranked if not item.node.ownership_known]

    @property
    def headline_count(self) -> int:
        return len(self.by_severity(Severity.CRITICAL)) + len(self.by_severity(Severity.HIGH))
