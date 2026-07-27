"""Tests for the command line surface.

What the terminal prints is not a lesser copy of the brief: `--out` is
optional, so for the headline command in the README this table is the entire
output. These tests exist because it once said nothing about a cone that hit
its budget.

The end-to-end tests at the bottom run the real pipeline - walk, enrich,
score, render, publish - against an in-memory catalog, with only the MCP
transport replaced. They are the check on the claim the README makes, that
everything but the transport is testable with no DataHub instance.

The typer import is guarded because the suite must also run under a bare
`pytest` (see conftest.py), and an interpreter outside the project venv can
carry a typer/click pair that raises on import. Skipping there beats erroring
the file; `uv run pytest` runs these for real.

The guard is narrowed to interpreters outside the venv on purpose. Inside it,
typer is a declared dependency, so an import failure is a real break and has
to be loud - a blanket `except` would turn that into a silent skip, which is
the "zero results reads as clean" failure this project is written against.
`pytest.importorskip` is not usable here: the global typer raises TypeError,
not ImportError.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime

import pytest

try:
    from typer.testing import CliRunner
except Exception as exc:  # pragma: no cover - depends on the interpreter
    if sys.prefix != sys.base_prefix:
        raise
    pytest.skip(
        f"typer is not importable outside the project venv: {exc}",
        allow_module_level=True,
    )

from blast_radar import cli
from blast_radar.config import ScoringWeights
from blast_radar.models import BlastReport, CappedLineage, ConsumerNode, Entity
from blast_radar.scoring import score_node
from inmemory_datahub import READ_TOOLS, WRITE_TOOLS, InMemoryDataHub, asset

URN = "urn:li:dataset:(urn:li:dataPlatform:snowflake,shop.orders,PROD)"

# Every name `Settings.from_env` reads, not just the obvious ones. A developer
# who has `BLAST_RADAR_MCP_COMMAND` or `BLAST_RADAR_DEBUG` exported would
# otherwise change what a scan does under test, or point it somewhere real.
# `BLAST_RADAR_SERVER_LOG_DATE_ONLY` is deliberately absent: config.py reads no
# such variable.
ENV_VARS = (
    "DATAHUB_GMS_URL",
    "DATAHUB_GMS_TOKEN",
    "BLAST_RADAR_MCP_COMMAND",
    "BLAST_RADAR_MCP_ARGS",
    "BLAST_RADAR_SERVER_LOG",
    "BLAST_RADAR_DEBUG",
    "BLAST_RADAR_ALLOW_INSECURE_TOKEN",
    "BLAST_RADAR_MAX_HOPS",
    "BLAST_RADAR_MAX_NODES",
    "BLAST_RADAR_RESULTS_PER_HOP",
    "TOOLS_IS_MUTATION_ENABLED",
)


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    """No test here may depend on what the developer has exported.

    Every command starts by reading the environment, so a shell pointing at a
    real instance would otherwise change what these tests assert - or, worse,
    let one of them reach it.
    """
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    # Rich wraps to 80 columns when stdout is not a terminal, and a wrapped
    # line splits the sentences these tests look for. Widening the console
    # pins the content without coupling any assertion to a line break.
    monkeypatch.setenv("COLUMNS", "200")


def build_report(
    *,
    truncated: bool = False,
    capped: dict[str, CappedLineage] | None = None,
    consumers: int = 1,
) -> BlastReport:
    nodes = [
        ConsumerNode(
            entity=Entity(
                urn=f"urn:li:dashboard:exec.revenue{index}",
                entity_type="DASHBOARD",
                name=f"exec.revenue{index}",
            ),
            hop=1,
            transform_hop=1,
            downstream_count=2,
            enriched=True,
            usage_known=True,
        )
        for index in range(consumers)
    ]
    return BlastReport(
        root=Entity(urn=URN, entity_type="DATASET", name="shop.orders"),
        change_summary="dropping column promo_code",
        generated_at=datetime(2026, 7, 23, 10, 0, tzinfo=UTC),
        assessments=[score_node(node, ScoringWeights(), 4) for node in nodes],
        hops_traversed=1,
        truncated=truncated,
        capped_lineage=capped or {},
    )


def invoke(monkeypatch: pytest.MonkeyPatch, report: BlastReport, *args: str):
    """Run `scan` with the DataHub round trip replaced by a finished report."""

    async def fake_scan(urn, change, context, *, with_usage, publish):
        return report, None

    monkeypatch.setattr(cli, "_scan", fake_scan)
    return CliRunner().invoke(cli.app, ["scan", URN, *args])


def test_the_terminal_discloses_a_truncated_cone(monkeypatch):
    result = invoke(monkeypatch, build_report(truncated=True))

    assert result.exit_code == 0
    assert "budget" in result.output
    assert "BLAST_RADAR_MAX_NODES" in result.output
    # The sentences are shared with the Markdown brief, which spells env vars
    # in backticks. A terminal prints that punctuation literally.
    assert "`" not in result.output


def test_the_terminal_discloses_a_capped_lineage_reply(monkeypatch):
    capped = {"urn:li:dataset:hub": CappedLineage(returned=100, total=150)}
    result = invoke(monkeypatch, build_report(capped=capped))

    assert result.exit_code == 0
    assert "consumer(s)" in result.output
    assert "BLAST_RADAR_RESULTS_PER_HOP" in result.output


def test_the_terminal_never_prints_a_shortfall_it_does_not_know(monkeypatch):
    capped = {"urn:li:dataset:hub": CappedLineage(returned=100, total=None)}
    result = invoke(monkeypatch, build_report(capped=capped))

    assert result.exit_code == 0
    assert "0 consumer(s)" not in result.output
    assert "unknown" in result.output


def test_the_terminal_says_when_the_table_stops_short(monkeypatch):
    result = invoke(monkeypatch, build_report(consumers=3), "--limit", "1")

    assert result.exit_code == 0
    assert "highest-scoring of 3" in result.output
    # The note and the cut are two different things, and only the note was
    # checked: a limit that disclosed itself while printing every row would
    # have passed. One line per row here, since each name is short enough that
    # Rich does not wrap it at the width this suite pins.
    printed = [line for line in result.output.splitlines() if " exec.revenue" in line]
    assert len(printed) == 1


def test_a_limit_below_one_is_refused_rather_than_slicing_backwards(monkeypatch):
    """`ranked[:-1]` would drop a row under a sentence disclosing the cut."""
    result = invoke(monkeypatch, build_report(consumers=3), "--limit", "0")

    assert result.exit_code == 2


def test_a_bad_budget_is_one_readable_line_not_a_traceback(monkeypatch):
    monkeypatch.setenv("BLAST_RADAR_MAX_NODES", "0")

    async def never(*args, **kwargs):
        raise AssertionError("the scan must not start with an unusable budget")

    monkeypatch.setattr(cli, "_scan", never)
    result = CliRunner().invoke(cli.app, ["scan", URN])

    assert result.exit_code == 1
    assert "Configuration error" in result.output
    assert "BLAST_RADAR_MAX_NODES" in result.output


def test_a_complete_scan_prints_no_note(monkeypatch):
    result = invoke(monkeypatch, build_report())

    assert result.exit_code == 0
    assert "Note:" not in result.output
    assert "highest-scoring" not in result.output


DASHBOARD = "urn:li:dashboard:exec.revenue"
STAGING = "urn:li:dataset:staging.orders_enriched"
ORPHAN = "urn:li:dataset:analytics.orphan_extract"


def catalog(**overrides) -> InMemoryDataHub:
    """A small catalog with one loud consumer, one quiet one, and one orphan."""
    settings = {
        "entities": {
            URN: asset(URN, name="shop.orders", owners=("ana",)),
            DASHBOARD: asset(DASHBOARD, name="exec.revenue", owners=("ana",), tags=("tier1",)),
            STAGING: asset(STAGING, name="staging.orders_enriched", owners=("bob",)),
            ORPHAN: asset(ORPHAN, name="analytics.orphan_extract"),
        },
        "lineage": {URN: [DASHBOARD, STAGING, ORPHAN]},
        "queries": {DASHBOARD: 40, STAGING: 5, ORPHAN: 0},
    }
    return InMemoryDataHub(**{**settings, **overrides})


def connect(monkeypatch, datahub: InMemoryDataHub) -> InMemoryDataHub:
    """Replace the transport, and only the transport.

    Everything downstream of `open_datahub` - the adapter, the walk, scoring,
    the renderer, the write-back executor - runs for real against `datahub`.
    """

    @asynccontextmanager
    async def fake_open(settings):
        yield datahub

    monkeypatch.setattr(cli, "open_datahub", fake_open)
    return datahub


def test_a_scan_ranks_the_cone_and_names_the_asset_with_nobody_to_tell(monkeypatch):
    connect(monkeypatch, catalog())

    result = CliRunner().invoke(cli.app, ["scan", URN])

    assert result.exit_code == 0
    assert "exec.revenue" in result.output
    assert "HIGH" in result.output
    assert "1 impacted asset(s) have no owner to notify" in result.output
    # Nothing was published, so the terminal has to say what it would have done.
    assert "Use --publish" in result.output


def test_an_asset_the_entity_fetch_missed_is_reported_as_unknown_not_as_an_orphan(monkeypatch):
    """Lineage returned it, the metadata call did not. It is still a real
    consumer, and its ownership is unknown rather than absent - which is a
    different sentence, in a different colour, in a different section."""
    missing = "urn:li:dataset:analytics.ghost"
    datahub = catalog(lineage={URN: [DASHBOARD, missing]})
    connect(monkeypatch, datahub)

    result = CliRunner().invoke(cli.app, ["scan", URN])

    assert result.exit_code == 0
    assert "1 asset(s) could not be enriched" in result.output
    assert "unknown, not absent" in result.output
    assert "no owner to notify" not in result.output


def test_the_written_brief_honours_the_same_limit_as_the_table(monkeypatch, tmp_path):
    """One flag, two surfaces. A brief that ignored it would hold every row
    while the terminal said it had shown the highest-scoring few, and the
    brief is the copy that gets published and read later."""
    connect(monkeypatch, catalog())
    out = tmp_path / "brief.md"

    result = CliRunner().invoke(cli.app, ["scan", URN, "--limit", "1", "--out", str(out)])

    assert result.exit_code == 0
    brief = out.read_text(encoding="utf-8")
    ranked_rows = [line for line in brief.splitlines() if line.startswith("| 1 ")]
    assert ranked_rows == [
        "| 1 | exec.revenue | Dashboard | 69.4 | HIGH | "
        "Dashboard consumer; 1 hop downstream of the change |"
    ]
    assert "staging.orders_enriched" not in brief
    assert "The 1 highest-scoring of 3 consumers." in brief


def test_a_scan_writes_the_brief_where_it_was_asked_to(monkeypatch, tmp_path):
    connect(monkeypatch, catalog())
    out = tmp_path / "brief.md"

    result = CliRunner().invoke(cli.app, ["scan", URN, "--out", str(out)])

    assert result.exit_code == 0
    brief = out.read_text(encoding="utf-8")
    assert brief.startswith("# Blast radius: shop.orders")
    assert "exec.revenue" in brief
    assert "## Impacted with no owner to notify" in brief


def test_publishing_tags_scores_and_documents_the_assets_that_scored_high(monkeypatch):
    datahub = connect(monkeypatch, catalog())

    result = CliRunner().invoke(cli.app, ["scan", URN, "--publish"])

    assert result.exit_code == 0
    assert datahub.tagged == [
        {"tag_urns": ["urn:li:tag:blast-radar-high"], "entity_urns": [DASHBOARD]}
    ]
    # The score published and the tag applied have to agree, or the asset page
    # shows a number in one band next to a label from another.
    scored = datahub.scored[0]
    published = scored["property_values"]["urn:li:structuredProperty:blastRadar.riskScore"][0]
    assert scored["entity_urns"] == [DASHBOARD]
    assert isinstance(published, int)
    assert 55 <= published < 75

    document = next(iter(datahub.documents.values()))
    assert document["title"] == "Blast radius: shop.orders (snowflake, PROD)"
    assert document["related_assets"] == [URN, DASHBOARD]
    assert "Blast radius: shop.orders" in document["content"]


def test_the_quiet_consumers_are_ranked_but_not_written_back(monkeypatch):
    """Writing a score onto every node of a cone turns the graph into noise."""
    datahub = connect(monkeypatch, catalog())

    result = CliRunner().invoke(cli.app, ["scan", URN, "--publish"])

    assert result.exit_code == 0
    assert "staging.orders_enriched" in result.output
    assert [call for call in datahub.calls if call[0] == "add_structured_properties"] == [
        ("add_structured_properties", datahub.scored[0])
    ]


def test_scanning_twice_updates_the_brief_instead_of_leaving_two(monkeypatch):
    """Without the lookup, every CI run leaves another near-identical document."""
    datahub = connect(monkeypatch, catalog())

    first = CliRunner().invoke(cli.app, ["scan", URN, "--publish"])
    second = CliRunner().invoke(cli.app, ["scan", URN, "--publish"])

    assert (first.exit_code, second.exit_code) == (0, 0)
    assert len(datahub.documents) == 1


def test_a_stale_severity_tag_is_cleared_before_the_new_one_lands(monkeypatch):
    """Severity is one value; tags accumulate. The stale one is usually the
    more alarming, which is the reason this runs at all."""
    datahub = connect(monkeypatch, catalog())

    CliRunner().invoke(cli.app, ["scan", URN, "--publish"])

    cleared = {call[1]["tag_urns"][0] for call in datahub.calls if call[0] == "remove_tags"}
    assert "urn:li:tag:blast-radar-critical" in cleared
    assert "urn:li:tag:blast-radar-high" not in cleared
    # Before, not after. Applying the new tag first and then clearing the
    # others leaves the same end state here and removes the fresh tag on any
    # asset whose severity did not change, so the order is the behavior.
    names = [call[0] for call in datahub.calls]
    assert names.index("remove_tags") < names.index("add_tags")


def test_an_instance_without_remove_tags_still_publishes(monkeypatch):
    without_remove = ("add_tags", "add_structured_properties", "save_document")
    datahub = connect(monkeypatch, catalog(tool_names=READ_TOOLS + without_remove))

    result = CliRunner().invoke(cli.app, ["scan", URN, "--publish"])

    assert result.exit_code == 0
    assert datahub.tagged
    assert [call for call in datahub.calls if call[0] == "remove_tags"] == []


def test_an_unwritable_brief_path_is_a_sentence_and_a_non_zero_exit(monkeypatch, tmp_path):
    """By the time this fails the scan has run, so it is not a traceback."""
    connect(monkeypatch, catalog())

    result = CliRunner().invoke(cli.app, ["scan", URN, "--out", str(tmp_path)])

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Could not write the brief" in result.output
    # The ranking still printed: the scan itself succeeded.
    assert "exec.revenue" in result.output


def test_a_failed_publish_and_a_failed_brief_are_both_reported(monkeypatch, tmp_path):
    """Whichever happened second used to hide the first."""
    connect(monkeypatch, catalog(failing_tools={"add_tags": "mutations are disabled"}))

    result = CliRunner().invoke(cli.app, ["scan", URN, "--publish", "--out", str(tmp_path)])

    assert result.exit_code == 1
    assert "write(s) failed" in result.output
    assert "Could not write the brief" in result.output


def test_a_failed_write_exits_non_zero_and_says_the_verdict_is_partial(monkeypatch):
    """A release gate that exits 0 after every write failed is not a gate."""
    datahub = connect(monkeypatch, catalog(failing_tools={"add_tags": "mutations are disabled"}))

    result = CliRunner().invoke(cli.app, ["scan", URN, "--publish"])

    assert result.exit_code == 1
    assert "write(s) failed" in result.output
    assert "partial verdict" in result.output
    # The rest of the plan still ran: one failure must not abort the publish.
    assert datahub.scored


def test_skipping_usage_skips_the_query_history_calls(monkeypatch, tmp_path):
    """One call per shortlisted asset is the expensive shape, so it is optional.

    The brief is where the consequence shows: the assets still rank, and their
    usage reads as unchecked rather than as no queries at all.
    """
    datahub = connect(monkeypatch, catalog())
    out = tmp_path / "brief.md"

    result = CliRunner().invoke(cli.app, ["scan", URN, "--no-usage", "--out", str(out)])

    assert result.exit_code == 0
    assert [call for call in datahub.calls if call[0] == "get_dataset_queries"] == []
    assert "exec.revenue" in out.read_text(encoding="utf-8")


def test_a_capped_lineage_reply_reaches_the_terminal_from_the_adapter(monkeypatch):
    """End to end rather than from a hand-built report: the cap is recorded in
    the adapter and has to survive the whole pipeline to be disclosed."""
    connect(monkeypatch, catalog(lineage_total={URN: 40}))

    result = CliRunner().invoke(cli.app, ["scan", URN])

    assert result.exit_code == 0
    assert "37 consumer(s) are missing" in result.output


def test_an_unknown_urn_is_one_line_not_a_traceback(monkeypatch):
    connect(monkeypatch, catalog(entities={}, lineage={}))

    result = CliRunner().invoke(cli.app, ["scan", "urn:li:dataset:nope"])

    assert result.exit_code == 1
    assert "no entity with urn" in result.output


def test_a_broken_connection_mid_scan_does_not_print_a_clean_bill_of_health(monkeypatch):
    connect(monkeypatch, catalog(failing_tools={"get_entities": "connection reset"}))

    result = CliRunner().invoke(cli.app, ["scan", URN])

    assert result.exit_code == 1
    assert "Scan failed" in result.output
    assert "doctor" in result.output
    assert "Downstream impact of" not in result.output


def test_tools_lists_what_the_server_advertises(monkeypatch):
    connect(monkeypatch, catalog())

    result = CliRunner().invoke(cli.app, ["tools"])

    assert result.exit_code == 0
    assert "get_lineage" in result.output
    assert "Mutations disabled" in result.output


def test_tools_can_print_the_raw_schemas_for_a_bug_report(monkeypatch):
    connect(monkeypatch, catalog())

    result = CliRunner().invoke(cli.app, ["tools", "--json"])

    assert result.exit_code == 0
    assert '"get_lineage"' in result.output


def test_tools_schema_view_names_the_arguments(monkeypatch):
    connect(monkeypatch, catalog())

    result = CliRunner().invoke(cli.app, ["tools", "--schema"])

    assert result.exit_code == 0
    assert "required" in result.output
    assert "count, urn" in result.output


def test_tools_schema_view_says_plainly_when_a_tool_takes_nothing(monkeypatch):
    connect(monkeypatch, catalog(schema={}))

    result = CliRunner().invoke(cli.app, ["tools", "--schema"])

    assert result.exit_code == 0
    assert "-" in result.output


def test_an_argument_name_from_the_server_is_not_markup(monkeypatch):
    """The schema is the server's, not ours. A property named `[/bold]` is a
    closing tag with nothing open, which Rich refuses to render at all."""
    hostile = {"properties": {"[/bold]": {"type": "string"}}, "required": ["[/bold]"]}
    connect(monkeypatch, catalog(schema=hostile))

    result = CliRunner().invoke(cli.app, ["tools", "--schema"])

    assert result.exit_code == 0
    assert result.exception is None
    assert "[/bold]" in result.output


def test_doctor_passes_a_healthy_instance(monkeypatch):
    connect(monkeypatch, catalog())

    result = CliRunner().invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "Connected" in result.output
    assert "All read tools a scan needs are present" in result.output
    assert "Write-back tools are available" in result.output
    assert "MISSING (DATAHUB_GMS_TOKEN)" in result.output


def test_doctor_says_scans_still_run_when_only_the_write_tools_are_off(monkeypatch):
    connect(monkeypatch, catalog(tool_names=READ_TOOLS))

    result = CliRunner().invoke(cli.app, ["doctor"])

    assert result.exit_code == 0
    assert "Write-back unavailable" in result.output
    assert "TOOLS_IS_MUTATION_ENABLED" in result.output


def test_doctor_fails_when_a_tool_a_scan_needs_is_missing(monkeypatch):
    connect(monkeypatch, catalog(tool_names=WRITE_TOOLS))

    result = CliRunner().invoke(cli.app, ["doctor"])

    assert result.exit_code == 1
    assert "Missing read tools" in result.output
    assert "get_lineage" in result.output


def test_doctor_turns_an_unreachable_server_into_a_fix_list(monkeypatch):
    @asynccontextmanager
    async def refuse(settings):
        raise ConnectionRefusedError("no server on 8080")
        yield  # pragma: no cover - unreachable, present so this stays a generator

    monkeypatch.setattr(cli, "open_datahub", refuse)

    result = CliRunner().invoke(cli.app, ["doctor"])

    assert result.exit_code == 1
    assert "Could not reach the MCP server" in result.output
    assert "datahub docker quickstart" in result.output


def test_setup_reports_each_thing_it_defined(monkeypatch):
    from blast_radar.bootstrap import BootstrapResult

    monkeypatch.setattr(
        cli, "ensure_risk_property", lambda url, token: BootstrapResult(True, "created property")
    )
    monkeypatch.setattr(
        cli,
        "ensure_severity_tags",
        lambda url, token: [BootstrapResult(False, "blast-radar-high: already defined")],
    )

    result = CliRunner().invoke(cli.app, ["setup"])

    assert result.exit_code == 0
    assert "created property" in result.output
    assert "already defined" in result.output


def test_the_console_script_entry_point_runs_the_app(monkeypatch):
    """`blast-radar` on the path calls this and nothing else."""
    calls: list[bool] = []
    monkeypatch.setattr(cli, "app", lambda: calls.append(True))

    cli.main()

    assert calls == [True]
