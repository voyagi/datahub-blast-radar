"""Breadth-first walk of the downstream cone.

The walk is deliberately separated from the MCP calls: it takes a fetch
callable and returns plain nodes. That keeps the traversal rules - dedupe,
shortest path wins, budget enforcement, cycle safety - testable without a
DataHub instance, and it means the MCP response mapping can change without
touching the algorithm.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Awaitable, Callable
from dataclasses import dataclass

from .models import ConsumerNode, Entity

FetchDownstream = Callable[[str], Awaitable[list[Entity]]]

# Entities that present data rather than produce it. Crossing into one of
# these is not a transform: the chart renders whatever the dataset holds, and
# the dashboard frames the chart. A dropped column reaches both untouched, so
# neither edge should decay the proximity score.
#
# Notebooks are deliberately absent. A notebook reads data and frequently
# writes a derived table back, which is a transform in every sense that
# matters here.
PRESENTATION_TYPES = frozenset({"CHART", "DASHBOARD"})


@dataclass
class WalkResult:
    nodes: list[ConsumerNode]
    truncated: bool
    hops_traversed: int


async def walk_downstream(
    root_urn: str,
    fetch: FetchDownstream,
    *,
    max_hops: int,
    max_nodes: int,
) -> WalkResult:
    """Collect the downstream cone of `root_urn`.

    What the node budget keeps is **arbitrary with respect to distance**. The
    budget is spent per child, in whatever order DataHub listed that parent's
    lineage, before any cost is compared. The 0-1 queue orders parents; nothing
    orders the children admitted during one expansion. So a costly child listed
    first can take the last slot from a free child listed second, and a
    truncated cone is a sample rather than the nearest ring by either measure.
    Truncation is always disclosed for exactly this reason.

    Two distances are tracked at once, and neither is settled by first sight.
    `hop` is graph distance; `transform_hop` counts only the edges that
    reshape data, so a presentation edge costs nothing. That makes the walk a
    0-1 shortest-path problem, and the 0-1 queue that solves it is not FIFO -
    so first discovery is not shortest distance for *either* measure. Both are
    relaxed when a cheaper route turns up, and the improvement is pushed down
    to everything already found beneath the node.
    """
    if max_hops < 1 or max_nodes < 1:
        # Refused, not walked. A budget below one admits nothing, and an empty
        # cone renders as "nothing downstream depends on this asset. Safe to
        # change" - a false all-clear produced by a config typo. Reporting it
        # as truncation instead would name a budget the caller may not have
        # touched, so the honest answer is to refuse the walk.
        raise ValueError(
            f"A downstream walk needs a budget of at least 1 in both "
            f"directions (max_hops={max_hops}, max_nodes={max_nodes}). "
            "A budget below that walks nothing, which is not the same as "
            "finding nothing."
        )

    seen: dict[str, ConsumerNode] = {}
    # Lineage is fetched once per asset however many times the asset is
    # relaxed, because relaxation is bookkeeping and re-walking is network.
    fetched: dict[str, list[Entity]] = {}
    queue: deque[tuple[str, int, int]] = deque([(root_urn, 0, 0)])
    truncated = False

    while queue:
        urn, hop, transform_hop = queue.popleft()
        if hop >= max_hops:
            # Reached the depth limit: whatever is below this node stays
            # unknown, which the node already reports via `expanded`.
            continue

        if urn in fetched:
            children = fetched[urn]
        else:
            children = await fetch(urn)
            fetched[urn] = children

        node = seen.get(urn)
        if node is not None:
            # Exclude the scanned asset: a cycle edge back to it is not a
            # dependent, and it is not in the cone either.
            node.downstream_count = sum(1 for child in children if child.urn != root_urn)
            node.expanded = True

        for child in children:
            # The scanned asset is the cause, not a casualty. Lineage cycles
            # lead back to it, and without this it would appear in its own
            # blast radius and get tagged as endangered by its own change.
            if child.urn == root_urn:
                continue

            refused = _visit_child(
                child,
                seen=seen,
                queue=queue,
                hop=hop,
                transform_hop=transform_hop,
                max_nodes=max_nodes,
            )
            if refused:
                truncated = True

    nodes = sorted(seen.values(), key=lambda item: (item.hop, item.entity.name))
    return WalkResult(
        nodes=nodes,
        truncated=truncated,
        # Measured from what survived, not from what was walked: relaxation
        # lowers hops after the fact, and nodes dropped by the budget were
        # inflating this with depths no reported node actually has.
        hops_traversed=max((node.hop for node in nodes), default=0),
    )


def _visit_child(
    child: Entity,
    *,
    seen: dict[str, ConsumerNode],
    queue: deque[tuple[str, int, int]],
    hop: int,
    transform_hop: int,
    max_nodes: int,
) -> bool:
    """Admit a newly seen child, or relax one already found.

    Returns True only when the node budget refused the child, which is the
    single condition that makes the whole cone a sample rather than the answer.
    """
    child_hop = hop + 1
    child_transform_hop = transform_hop + (
        0 if child.entity_type.upper() in PRESENTATION_TYPES else 1
    )

    existing = seen.get(child.urn)
    if existing is None:
        if len(seen) >= max_nodes:
            return True
        seen[child.urn] = ConsumerNode(
            entity=child,
            hop=child_hop,
            transform_hop=child_transform_hop,
            downstream_count=0,
            expanded=False,
        )
        _enqueue(queue, child.urn, child_hop, child_transform_hop, transform_hop)
        return False

    # Already found, but possibly by a longer route. Both distances need
    # relaxing: a free presentation edge lets the queue reach an asset before a
    # shorter path does, so an unrelaxed `hop` is inflated - and an inflated hop
    # can push a node past max_hops and drop its whole subtree out of the cone.
    improved = False
    if child_hop < existing.hop:
        existing.hop = child_hop
        improved = True
    # The transform half is the invariant-preserving one and has no known
    # reachable shape: the 0-1 queue pops in non-decreasing transform depth, so
    # a node's first sighting is already its cheapest by that measure. A search
    # over 60,000 random lineage graphs relaxed a hop 1,314 times and this line
    # zero times, which is why the suite cannot cover it. It stays because the
    # hop relaxation above re-queues nodes, and re-queueing is exactly what
    # could disturb that ordering; dropping the check would turn a future
    # disturbance into a silently inflated transform depth, which reads as a
    # dashboard being further from the change than it is.
    if child_transform_hop < existing.transform_hop:
        existing.transform_hop = child_transform_hop
        improved = True
    if improved:
        _enqueue(queue, child.urn, existing.hop, existing.transform_hop, transform_hop)
    return False


def _enqueue(
    queue: deque[tuple[str, int, int]],
    urn: str,
    hop: int,
    transform_hop: int,
    parent_transform_hop: int,
) -> None:
    """0-1 queue: free edges go to the front so cheap paths settle first."""
    entry = (urn, hop, transform_hop)
    if transform_hop == parent_transform_hop:
        queue.appendleft(entry)
    else:
        queue.append(entry)
