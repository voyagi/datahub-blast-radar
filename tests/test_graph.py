"""Tests for the downstream walk.

The fake lineage graph here is a dict, which is all the walker needs. Every
case below is one a real catalog produces: diamonds, cycles, depth limits, and
graphs bigger than the budget.
"""

from __future__ import annotations

import pytest

from blast_radar.graph import walk_downstream
from blast_radar.models import Entity


def entity(name: str, entity_type: str = "DATASET") -> Entity:
    return Entity(urn=f"urn:{name}", entity_type=entity_type, name=name)


def fetcher(graph: dict[str, list[str]], types: dict[str, str] | None = None):
    """Build a fetch callable over a urn -> child-names adjacency map.

    `types` overrides an entity's type by name, for the BI-cone cases where
    the distinction between a transform and a rendering is the whole point.
    """
    calls: list[str] = []
    types = types or {}

    async def fetch(urn: str) -> list[Entity]:
        calls.append(urn)
        return [entity(child, types.get(child, "DATASET")) for child in graph.get(urn, [])]

    return fetch, calls


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_empty_downstream_returns_nothing():
    fetch, _ = fetcher({})

    result = await walk_downstream("urn:root", fetch, max_hops=4, max_nodes=100)

    assert result.nodes == []
    assert result.truncated is False


@pytest.mark.anyio
async def test_hops_are_recorded_per_ring():
    fetch, _ = fetcher({"urn:root": ["a"], "urn:a": ["b"], "urn:b": ["c"]})

    result = await walk_downstream("urn:root", fetch, max_hops=4, max_nodes=100)

    hops = {node.entity.name: node.hop for node in result.nodes}
    assert hops == {"a": 1, "b": 2, "c": 3}
    assert result.hops_traversed == 3


@pytest.mark.anyio
async def test_diamond_dependency_keeps_the_shortest_path():
    # root -> a -> shared, and root -> shared directly. Shared is 1 hop away.
    fetch, _ = fetcher({"urn:root": ["a", "shared"], "urn:a": ["shared"]})

    result = await walk_downstream("urn:root", fetch, max_hops=4, max_nodes=100)

    shared = next(node for node in result.nodes if node.entity.name == "shared")
    assert shared.hop == 1
    assert len(result.nodes) == 2


@pytest.mark.anyio
async def test_the_scanned_asset_never_appears_in_its_own_blast_radius():
    """A cycle used to walk the root back into its own cone.

    Published, that means tagging the asset being changed as endangered by its
    own change.
    """
    fetch, calls = fetcher({"urn:root": ["a"], "urn:a": ["b"], "urn:b": ["root"]})

    result = await walk_downstream("urn:root", fetch, max_hops=6, max_nodes=100)

    assert calls.count("urn:root") == 1
    assert {node.entity.name for node in result.nodes} == {"a", "b"}
    assert "urn:root" not in {node.urn for node in result.nodes}


@pytest.mark.anyio
async def test_max_hops_stops_the_walk_and_marks_the_frontier_unexpanded():
    fetch, _ = fetcher({"urn:root": ["a"], "urn:a": ["b"], "urn:b": ["c"]})

    result = await walk_downstream("urn:root", fetch, max_hops=2, max_nodes=100)

    names = {node.entity.name for node in result.nodes}
    assert names == {"a", "b"}
    frontier = next(node for node in result.nodes if node.entity.name == "b")
    assert frontier.expanded is False
    assert frontier.downstream_count == 0


@pytest.mark.anyio
async def test_expanded_nodes_carry_their_dependent_count():
    fetch, _ = fetcher({"urn:root": ["a"], "urn:a": ["x", "y", "z"]})

    result = await walk_downstream("urn:root", fetch, max_hops=3, max_nodes=100)

    node_a = next(node for node in result.nodes if node.entity.name == "a")
    assert node_a.expanded is True
    assert node_a.downstream_count == 3


@pytest.mark.anyio
async def test_node_budget_truncates_and_says_so():
    fetch, _ = fetcher({"urn:root": [f"child{index}" for index in range(10)]})

    result = await walk_downstream("urn:root", fetch, max_hops=3, max_nodes=4)

    assert len(result.nodes) == 4
    assert result.truncated is True


@pytest.mark.anyio
async def test_the_node_budget_keeps_the_nearest_ring_in_a_dataset_only_cone():
    """With no free edges, transform depth and edge count agree."""
    graph = {
        "urn:root": ["near1", "near2"],
        "urn:near1": ["far1", "far2"],
        "urn:near2": ["far3"],
    }
    fetch, _ = fetcher(graph)

    result = await walk_downstream("urn:root", fetch, max_hops=3, max_nodes=2)

    assert {node.entity.name for node in result.nodes} == {"near1", "near2"}


@pytest.mark.anyio
async def test_truncation_can_drop_a_nearer_consumer_than_one_it_keeps():
    """Truncation is arbitrary with respect to distance, and that is documented.

    Two claims have already been made and disproved here: that the budget
    keeps the hop-nearest ring, and then that it keeps the transform-nearest
    one. The budget is spent per child in DataHub's listing order, before any
    cost is compared, so neither holds. This test pins the *absence* of the
    guarantee so a third version of the claim cannot quietly reappear.
    """
    graph = {
        "urn:root": ["c1", "b"],
        "urn:c1": ["c2"],
        "urn:c2": ["c3"],
        "urn:b": ["b2"],
    }
    fetch, _ = fetcher(graph, types={"c1": "CHART", "c2": "CHART", "c3": "CHART"})

    result = await walk_downstream("urn:root", fetch, max_hops=4, max_nodes=4)

    kept = {node.entity.name: node for node in result.nodes}
    assert result.truncated is True
    assert len(kept) == 4
    # b2 at hop 2 was dropped while c3 at hop 3 was kept: strictly nearer,
    # strictly dropped.
    assert "b2" not in kept
    assert kept["c3"].hop == 3


@pytest.mark.anyio
async def test_a_costly_child_listed_first_can_take_the_last_slot():
    """The mechanism behind the missing guarantee, at its smallest.

    One slot, two children of the same parent: the dataset is listed first and
    takes it, so the chart zero transforms away is refused.
    """
    fetch, _ = fetcher({"urn:root": ["table", "chart"]}, types={"chart": "CHART"})

    result = await walk_downstream("urn:root", fetch, max_hops=3, max_nodes=1)

    assert result.truncated is True
    assert [node.entity.name for node in result.nodes] == ["table"]


# Two free edges on one route and one costly edge on the other is the only
# shape that makes the walk find a node by the LONGER path first, which is the
# one shape where the hop relaxation does anything. `extract` is three edges
# away through the chart and the dashboard, two edges away through `staging`,
# and the 0-1 queue drains both free edges before it ever pops `staging`.
INFLATED_HOP_GRAPH = {
    "urn:root": ["chart", "staging"],
    "urn:chart": ["dashboard"],
    "urn:dashboard": ["extract"],
    "urn:staging": ["extract"],
}
RENDERING_TYPES = {"chart": "CHART", "dashboard": "DASHBOARD"}


@pytest.mark.anyio
async def test_a_free_edge_cannot_inflate_a_reported_hop():
    """The regression this guards.

    Without relaxing `hop`, `extract` is reported three hops out because the
    free chart edges reached it first, while it sits two hops from the change.
    """
    fetch, _ = fetcher(INFLATED_HOP_GRAPH, types=RENDERING_TYPES)

    result = await walk_downstream("urn:root", fetch, max_hops=5, max_nodes=100)

    extract = next(node for node in result.nodes if node.entity.name == "extract")
    assert extract.hop == 2
    assert extract.transform_hop == 1
    assert result.hops_traversed == 2


@pytest.mark.anyio
async def test_an_inflated_hop_cannot_drop_a_consumer_off_the_depth_limit():
    """An unrelaxed hop pushes a node past max_hops and loses its subtree.

    `extract` is first found at hop 3, which is the depth limit here, so
    without the relaxation it is never expanded and `leaf` - a real consumer
    two transforms from the change - is missing from the cone entirely.
    """
    graph = {**INFLATED_HOP_GRAPH, "urn:extract": ["leaf"]}
    fetch, _ = fetcher(graph, types=RENDERING_TYPES)

    result = await walk_downstream("urn:root", fetch, max_hops=3, max_nodes=100)

    by_name = {node.entity.name: node for node in result.nodes}
    assert by_name["extract"].hop == 2
    assert "leaf" in by_name


@pytest.mark.anyio
async def test_hops_traversed_reflects_the_reported_cone_not_the_walk():
    """Nodes dropped by the budget must not inflate the depth headline."""
    graph = {"urn:root": ["a", "b"], "urn:a": ["deep"], "urn:b": ["deeper"]}
    fetch, _ = fetcher(graph)

    result = await walk_downstream("urn:root", fetch, max_hops=4, max_nodes=2)

    assert result.truncated is True
    assert result.hops_traversed == max(node.hop for node in result.nodes)


@pytest.mark.anyio
async def test_a_cycle_edge_back_to_the_root_is_not_counted_as_a_dependent():
    fetch, _ = fetcher({"urn:root": ["a"], "urn:a": ["root", "b"]})

    result = await walk_downstream("urn:root", fetch, max_hops=4, max_nodes=100)

    node_a = next(node for node in result.nodes if node.entity.name == "a")
    assert node_a.downstream_count == 1


@pytest.mark.anyio
async def test_rendering_edges_do_not_count_as_transforms():
    # The shape of every BI cone: two transforms, then a chart, then the
    # dashboard that frames it.
    graph = {
        "urn:root": ["staging"],
        "urn:staging": ["mart"],
        "urn:mart": ["chart"],
        "urn:chart": ["dashboard"],
    }
    fetch, _ = fetcher(graph, types={"chart": "CHART", "dashboard": "DASHBOARD"})

    result = await walk_downstream("urn:root", fetch, max_hops=5, max_nodes=100)

    by_name = {node.entity.name: node for node in result.nodes}
    # Split rather than combined with `and`, so a failure names which of the two
    # counters drifted instead of just reporting that the pair did.
    assert by_name["mart"].hop == 2
    assert by_name["mart"].transform_hop == 2
    # The chart is a third edge out but not a third transform.
    assert by_name["chart"].hop == 3
    assert by_name["chart"].transform_hop == 2
    # And the dashboard is exactly as exposed as the chart it contains.
    assert by_name["dashboard"].hop == 4
    assert by_name["dashboard"].transform_hop == 2


@pytest.mark.anyio
async def test_a_notebook_counts_as_a_transform():
    """Notebooks read data and write derived tables, so they are not renderings."""
    graph = {"urn:root": ["notebook"], "urn:notebook": ["derived"]}
    fetch, _ = fetcher(graph, types={"notebook": "NOTEBOOK"})

    result = await walk_downstream("urn:root", fetch, max_hops=4, max_nodes=100)

    by_name = {node.entity.name: node for node in result.nodes}
    assert by_name["notebook"].transform_hop == 1
    assert by_name["derived"].transform_hop == 2


@pytest.mark.anyio
async def test_diamond_keeps_the_lowest_transform_depth_too():
    # shared is reached directly (1 transform) and via a chart (still 1).
    graph = {"urn:root": ["chart", "shared"], "urn:chart": ["shared"]}
    fetch, _ = fetcher(graph, types={"chart": "CHART"})

    result = await walk_downstream("urn:root", fetch, max_hops=4, max_nodes=100)

    shared = next(node for node in result.nodes if node.entity.name == "shared")
    assert shared.transform_hop == 1


@pytest.mark.anyio
async def test_the_cheaper_transform_route_settles_before_the_costlier_one():
    """The 0-1 queue is what makes the cheap route win, not a later correction.

    `mid` is reachable through a chart for one transform and through a table
    for two. Free edges go to the front of the queue, so the one-transform
    route is explored first and everything beneath `mid` inherits the cheaper
    depth on first sight. Named for what it proves: nothing here is relaxed
    afterwards, and a test asserting a correction would pass without one.
    """
    graph = {
        "urn:root": ["viaTable", "viaChart"],
        "urn:viaTable": ["mid"],
        "urn:viaChart": ["mid"],
        "urn:mid": ["leaf"],
        "urn:leaf": ["deepLeaf"],
    }
    fetch, _ = fetcher(graph, types={"viaChart": "CHART"})

    result = await walk_downstream("urn:root", fetch, max_hops=6, max_nodes=100)

    by_name = {node.entity.name: node for node in result.nodes}
    assert by_name["mid"].transform_hop == 1
    assert by_name["leaf"].transform_hop == 2
    assert by_name["deepLeaf"].transform_hop == 3


@pytest.mark.anyio
async def test_lineage_is_fetched_once_per_asset_however_often_it_is_relaxed():
    """Relaxation is bookkeeping; re-walking would be network.

    `extract` really is queued twice here - once as found, once after its hop
    is relaxed - so the second visit has to come out of the cache. The graph
    matters: on a shape where nothing is ever relaxed, no node is queued twice
    and this passes without the cache existing at all.
    """
    graph = {**INFLATED_HOP_GRAPH, "urn:extract": ["leaf"]}
    fetch, calls = fetcher(graph, types=RENDERING_TYPES)

    result = await walk_downstream("urn:root", fetch, max_hops=6, max_nodes=100)

    assert next(node for node in result.nodes if node.entity.name == "extract").hop == 2
    assert calls.count("urn:extract") == 1
    assert len(calls) == len(set(calls))


@pytest.mark.anyio
async def test_a_budget_below_one_is_refused_rather_than_walked():
    """An unwalked cone is not an empty one.

    Returning no nodes renders as "nothing downstream depends on this asset.
    Safe to change", so a mistyped budget would clear a migration it never
    looked at. Reporting it as truncation instead would name a budget the
    caller may not have touched.
    """
    fetch, calls = fetcher({"urn:root": ["a", "b"]})

    with pytest.raises(ValueError, match="at least 1"):
        await walk_downstream("urn:root", fetch, max_hops=0, max_nodes=10)
    with pytest.raises(ValueError, match="at least 1"):
        await walk_downstream("urn:root", fetch, max_hops=3, max_nodes=0)

    assert calls == []
