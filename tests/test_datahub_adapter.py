"""Tests for the DataHub payload mapping.

The cases here are the ones where being wrong is silent: a response shape the
adapter cannot read must never come back as "nothing downstream".
"""

from __future__ import annotations

import pytest

from blast_radar.datahub_adapter import DataHubReader, LineageParseError, entity_from_payload


class FakeMCP:
    def __init__(self, payload):
        self._payload = payload
        self.calls: list[tuple[str, dict]] = []

    def has_tool(self, name: str) -> bool:
        return True

    async def call(self, name: str, arguments: dict):
        self.calls.append((name, arguments))
        if isinstance(self._payload, Exception):
            raise self._payload
        return self._payload


@pytest.fixture
def anyio_backend():
    return "asyncio"


@pytest.mark.anyio
async def test_an_empty_cone_is_reported_as_empty():
    reader = DataHubReader(FakeMCP({"downstreams": {"total": 0, "searchResults": []}}))

    assert await reader.fetch_downstream("urn:li:dataset:x") == []


@pytest.mark.anyio
@pytest.mark.parametrize(
    "payload",
    [
        "no results found",
        {"downstream": {"searchResults": []}},
        {"downstreams": "unexpected"},
        None,
        [],
        # The data-bearing key deserves the same suspicion as the outer one:
        # each of these used to return [] and render "Safe to change".
        {"downstreams": {"results": [{"entity": {"urn": "urn:li:dataset:y"}}]}},
        {"downstreams": {"total": 3}},
        {"downstreams": {"searchResults": "nope"}},
        {"downstreams": {}},
    ],
    ids=[
        "plain-text",
        "renamed-outer-key",
        "outer-not-an-object",
        "null",
        "list",
        "renamed-results-key",
        "results-key-absent-with-consumers",
        "results-not-a-list",
        "empty-block-without-a-total",
    ],
)
async def test_an_unreadable_response_raises_instead_of_clearing_the_migration(payload):
    """The failure mode this guards: an unparsed payload rendering as
    "nothing downstream depends on this asset. Safe to change."
    """
    reader = DataHubReader(FakeMCP(payload))

    with pytest.raises(LineageParseError):
        await reader.fetch_downstream("urn:li:dataset:x")


@pytest.mark.anyio
async def test_an_explicit_zero_total_is_the_one_readable_empty_cone():
    reader = DataHubReader(FakeMCP({"downstreams": {"total": 0}}))

    assert await reader.fetch_downstream("urn:li:dataset:x") == []


@pytest.mark.anyio
async def test_a_capped_lineage_reply_is_recorded_rather_than_silently_dropped():
    """DataHub returning 2 of 300 consumers must not read as 2 consumers."""
    reader = DataHubReader(
        FakeMCP(
            {
                "downstreams": {
                    "total": 300,
                    "hasMore": True,
                    "searchResults": [
                        {"entity": {"urn": "urn:li:dataset:a", "name": "a"}},
                        {"entity": {"urn": "urn:li:dataset:b", "name": "b"}},
                    ],
                }
            }
        )
    )

    entities = await reader.fetch_downstream("urn:li:dataset:hub")

    assert len(entities) == 2
    capped = reader.capped["urn:li:dataset:hub"]
    assert capped.returned == 2
    assert capped.total == 300
    # The number that goes in the brief is what is MISSING, not the total.
    assert capped.missing == 298


@pytest.mark.anyio
async def test_a_flagged_cap_without_a_total_reports_an_unknown_shortfall():
    """hasMore with no usable total must not claim a count it does not have."""
    reader = DataHubReader(
        FakeMCP(
            {
                "downstreams": {
                    "hasMore": True,
                    "searchResults": [{"entity": {"urn": "urn:li:dataset:a", "name": "a"}}],
                }
            }
        )
    )

    await reader.fetch_downstream("urn:li:dataset:hub")

    assert reader.capped["urn:li:dataset:hub"].missing is None


@pytest.mark.anyio
async def test_a_total_that_contradicts_hasmore_is_unknown_not_zero():
    """`hasMore` with a total that does not exceed the reply cannot be counted.

    Both facts cannot hold at once, so the total is not usable. Subtracting to
    zero here would put "0 consumer(s) are missing" in the brief, on the note
    whose entire job is to say something was left out.
    """
    reader = DataHubReader(
        FakeMCP(
            {
                "downstreams": {
                    "total": 1,
                    "hasMore": True,
                    "searchResults": [{"entity": {"urn": "urn:li:dataset:a", "name": "a"}}],
                }
            }
        )
    )

    await reader.fetch_downstream("urn:li:dataset:hub")

    capped = reader.capped["urn:li:dataset:hub"]
    assert capped.returned == 1
    assert capped.missing is None


@pytest.mark.anyio
async def test_a_complete_reply_records_no_cap():
    reader = DataHubReader(
        FakeMCP(
            {
                "downstreams": {
                    "total": 1,
                    "hasMore": False,
                    "searchResults": [{"entity": {"urn": "urn:li:dataset:a", "name": "a"}}],
                }
            }
        )
    )

    await reader.fetch_downstream("urn:li:dataset:small")

    assert reader.capped == {}


@pytest.mark.anyio
async def test_two_hits_with_no_urn_are_not_one_consumer():
    """The walk dedupes on the URN, so two nameless hits would collapse into
    one node and the cone would report one consumer where there were two."""
    reader = DataHubReader(
        FakeMCP(
            {
                "downstreams": {
                    "total": 4,
                    "searchResults": [
                        {"entity": {"urn": "urn:li:dataset:real", "name": "real"}},
                        {"entity": {"name": "no urn"}},
                        {"entity": {"name": "also no urn"}},
                        # No entity object at all. Filtered before an Entity is
                        # ever built, so it has to be counted against the raw
                        # reply or it disappears without a note.
                        {},
                    ],
                }
            }
        )
    )

    entities = await reader.fetch_downstream("urn:li:dataset:x")

    assert [entity.urn for entity in entities] == ["urn:li:dataset:real"]
    # Dropped, and disclosed: the brief says the reply was short rather than
    # rendering a cone that reads complete. All three unusable hits count.
    assert reader.capped["urn:li:dataset:x"].missing == 3


@pytest.mark.anyio
async def test_a_null_urn_is_absent_rather_than_the_word_none():
    """DataHub sends an explicit null for a URN it has no value for, and
    `str(None)` is "None": truthy, addressable-looking, and identical for
    every such hit, so they would merge instead of being dropped."""
    reader = DataHubReader(
        FakeMCP(
            {
                "downstreams": {
                    "total": 2,
                    "searchResults": [
                        {"entity": {"urn": None, "name": "ghost"}},
                        {"entity": {"urn": None, "name": "other ghost"}},
                    ],
                }
            }
        )
    )

    assert await reader.fetch_downstream("urn:li:dataset:x") == []
    assert reader.capped["urn:li:dataset:x"].missing == 2


@pytest.mark.anyio
async def test_a_reply_flagged_for_more_keeps_an_unknown_shortfall():
    """`hasMore` without a total means more consumers exist and nobody said
    how many. Counting the hits that were dropped here would publish an exact
    number for a reply that is short by an unknown amount."""
    reader = DataHubReader(
        FakeMCP(
            {
                "downstreams": {
                    "hasMore": True,
                    "searchResults": [
                        {"entity": {"urn": "urn:li:dataset:real"}},
                        {"entity": {"name": "no urn"}},
                    ],
                }
            }
        )
    )

    await reader.fetch_downstream("urn:li:dataset:x")

    assert reader.capped["urn:li:dataset:x"].missing is None


def test_a_type_that_is_not_a_string_falls_back_to_the_urn():
    """A number in that field is a payload this adapter does not understand,
    and the URN is a better answer than the number stringified: a dashboard
    reported as `7` would score on the unknown default rather than as one."""
    entity = entity_from_payload({"urn": "urn:li:dashboard:(looker,53)", "type": 7})

    assert entity.entity_type == "DASHBOARD"


def test_a_type_that_flattens_to_nothing_falls_back_to_the_urn():
    entity = entity_from_payload({"urn": "urn:li:chart:(looker,9)", "type": "   "})

    assert entity.entity_type == "CHART"


def test_an_entity_type_is_flattened_like_every_other_catalog_string():
    """The type is rendered in the ranked table and in the evidence beside it,
    so a newline in it splits a row the way a newline in a name does."""
    entity = entity_from_payload(
        {"urn": "urn:li:dataset:x", "type": "DASHBOARD\n| 99 | CRITICAL |"}
    )

    assert "\n" not in entity.entity_type
    assert entity.entity_type == "DASHBOARD | 99 | CRITICAL |"


@pytest.mark.anyio
async def test_a_result_with_no_entity_object_is_counted_as_a_shortfall():
    """The shape most likely to slip past a drop counter, since it is filtered
    out before an Entity is ever built.

    Deliberately without a `total` or a `hasMore`: with either of those the
    reply is already known to be short, and the count would be recorded
    whether or not this hit was noticed.
    """
    reader = DataHubReader(
        FakeMCP(
            {
                "downstreams": {
                    "searchResults": [
                        {"entity": {"urn": "urn:li:dataset:real", "name": "real"}},
                        {},
                    ]
                }
            }
        )
    )

    entities = await reader.fetch_downstream("urn:li:dataset:x")

    assert [entity.urn for entity in entities] == ["urn:li:dataset:real"]
    assert reader.capped["urn:li:dataset:x"].missing == 1


@pytest.mark.anyio
async def test_a_dropped_hit_is_counted_even_when_the_reply_had_no_total():
    reader = DataHubReader(
        FakeMCP(
            {
                "downstreams": {
                    "searchResults": [
                        {"entity": {"urn": "urn:li:dataset:real", "name": "real"}},
                        {"entity": {"name": "no urn"}},
                    ]
                }
            }
        )
    )

    await reader.fetch_downstream("urn:li:dataset:x")

    assert reader.capped["urn:li:dataset:x"].missing == 1


@pytest.mark.anyio
async def test_a_failed_query_lookup_is_unknown_not_zero():
    reader = DataHubReader(FakeMCP(RuntimeError("connection reset")))

    assert await reader.query_count("urn:li:dataset:x") is None


@pytest.mark.anyio
async def test_downstream_asks_for_downstream():
    mcp = FakeMCP({"downstreams": {"searchResults": []}})
    reader = DataHubReader(mcp)

    await reader.fetch_downstream("urn:li:dataset:x")

    _, arguments = mcp.calls[0]
    assert arguments["upstream"] is False
    assert arguments["max_hops"] == 1


@pytest.mark.anyio
async def test_a_real_zero_query_count_is_reported_as_zero():
    reader = DataHubReader(FakeMCP({"total": 0}))

    assert await reader.query_count("urn:li:dataset:x") == 0


def test_entity_type_falls_back_to_the_urn_when_the_payload_omits_it():
    """get_entities results carry no `type`; lineage hits do."""
    from_fetch = entity_from_payload(
        {"urn": "urn:li:dashboard:(looker,dashboards.53)", "name": "Sales"}
    )
    from_lineage = entity_from_payload(
        {"urn": "urn:li:dashboard:(looker,dashboards.53)", "type": "DASHBOARD", "name": "Sales"}
    )

    assert from_fetch.entity_type == "DASHBOARD"
    assert from_lineage.entity_type == "DASHBOARD"


def test_owners_survive_either_shape_datahub_uses():
    entity = entity_from_payload(
        {
            "urn": "urn:li:dataset:x",
            "ownership": {
                "owners": [
                    {
                        "owner": {
                            "urn": "urn:li:corpuser:ana",
                            "properties": {"displayName": "Ana Silva"},
                        },
                        "ownershipType": {"info": {"name": "Technical Owner"}},
                    },
                    {"owner": {"urn": "urn:li:corpGroup:platform", "name": "Platform"}},
                ]
            },
        }
    )

    assert [owner.name for owner in entity.owners] == ["Ana Silva", "Platform"]
    assert entity.owners[0].ownership_type == "Technical Owner"


def test_an_owner_entry_without_a_urn_is_dropped_rather_than_named_empty():
    entity = entity_from_payload(
        {"urn": "urn:li:dataset:x", "ownership": {"owners": [{"owner": {}}]}}
    )

    assert entity.owners == []
    assert entity.has_owner is False


@pytest.mark.anyio
async def test_asking_for_no_entities_asks_datahub_nothing():
    """An empty batch is a normal shape when a hop turns out to be a leaf."""
    datahub = FakeMCP({"result": []})

    assert await DataHubReader(datahub).get_entities([]) == {}
    assert datahub.calls == []


@pytest.mark.anyio
async def test_a_query_reply_that_is_not_an_object_is_unknown_not_zero():
    reader = DataHubReader(FakeMCP("the server sent prose"))

    assert await reader.query_count("urn:li:dataset:x") is None


def test_a_urn_the_adapter_cannot_parse_is_typed_unknown_not_guessed():
    """An unparseable URN scores on the default weight, and says so in the
    brief, rather than being silently filed as a dataset."""
    entity = entity_from_payload({"urn": "something-that-is-not-a-urn", "name": "mystery"})

    assert entity.entity_type == "UNKNOWN"


def test_an_entity_type_datahub_adds_later_survives_the_mapping():
    """The map covers today's types; an unmapped one keeps its own name."""
    entity = entity_from_payload({"urn": "urn:li:mlPrimaryKey:(a,b)", "name": "key"})

    assert entity.entity_type == "MLPRIMARYKEY"


def test_labels_fall_back_to_the_urn_when_nobody_set_a_display_name():
    """Half-populated catalogs are the normal case, and the governance factor
    matches on these strings, so a tag with no display name still has to
    arrive as a name rather than as an empty entry."""
    entity = entity_from_payload(
        {
            "urn": "urn:li:dataset:x",
            "tags": {"tags": [{"tag": {"urn": "urn:li:tag:tier1"}}, {"tag": {}}]},
            "glossaryTerms": {
                "terms": [
                    {"term": {"urn": "urn:li:glossaryTerm:PII"}},
                    {"term": {"properties": {"name": "Revenue"}, "urn": "urn:li:glossaryTerm:r"}},
                ]
            },
        }
    )

    assert entity.tags == ["tier1"]
    assert entity.glossary_terms == ["PII", "Revenue"]


def test_a_label_with_neither_a_name_nor_a_urn_is_dropped():
    """An empty string in the list would match no governance marker and would
    render as a blank entry, so it is not a label at all."""
    entity = entity_from_payload(
        {
            "urn": "urn:li:dataset:x",
            "tags": {"tags": [{"tag": {}}]},
            "glossaryTerms": {"terms": [{"term": {}}]},
        }
    )

    assert entity.tags == []
    assert entity.glossary_terms == []


def test_the_domain_arrives_however_datahub_nests_it():
    named = entity_from_payload(
        {
            "urn": "urn:li:dataset:x",
            "domain": {"domain": {"urn": "urn:li:domain:1", "properties": {"name": "Finance"}}},
        }
    )
    unnamed = entity_from_payload(
        {"urn": "urn:li:dataset:x", "domain": {"domain": {"urn": "urn:li:domain:sales"}}}
    )
    absent = entity_from_payload({"urn": "urn:li:dataset:x", "domain": {}})

    assert named.domain == "Finance"
    assert unnamed.domain == "sales"
    assert absent.domain is None


def test_the_platform_is_read_where_datahub_puts_it():
    named = entity_from_payload(
        {"urn": "urn:li:dataset:x", "platform": {"name": "snowflake"}},
    )
    absent = entity_from_payload({"urn": "urn:li:dataset:x"})

    assert named.platform == "snowflake"
    assert absent.platform is None


def test_a_label_that_is_only_invisible_characters_never_survives_as_empty():
    """A name that flattens to "" must be dropped or reported as None, not kept
    as a blank tag or an empty-string platform where None is the contract.

    The whole "unknown is not zero" discipline reaching the sanitizer: an
    empty string is neither the label nor its absence."""
    zwsp = chr(0x200B)
    entity = entity_from_payload(
        {
            "urn": "urn:li:dataset:x",
            "platform": {"name": zwsp},
            "domain": {"domain": {"urn": "urn:li:domain:d", "properties": {"name": zwsp}}},
            "tags": {"tags": [{"tag": {"properties": {"name": zwsp}, "urn": "urn:li:tag:t"}}]},
            "glossaryTerms": {
                "terms": [{"term": {"properties": {"name": zwsp}, "urn": "urn:li:glossaryTerm:g"}}]
            },
        }
    )

    assert entity.platform is None
    assert entity.domain is None
    assert entity.tags == []
    assert entity.glossary_terms == []


def test_a_human_written_description_beats_an_ingested_one():
    """Someone wrote it in the UI, which is a better signal that the asset is
    looked after than whatever the ingestion job carried over."""
    entity = entity_from_payload(
        {
            "urn": "urn:li:dataset:x",
            "properties": {"description": "from ingestion"},
            "editableProperties": {"description": "written by a human"},
        }
    )

    assert entity.description == "written by a human"
