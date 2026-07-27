"""Catalog metadata is untrusted input, and these are the ways it bit.

Every string this tool renders - asset names, owner display names, tag labels
- is written by whoever has ingestion rights to the catalog. In a shared
corporate DataHub that is a wide set of people, and the brief this tool
publishes back into the graph is what someone reads before deciding whether a
change is safe to ship. So a name is data, never markup, in either renderer.

Two escapes, because each renderer is fooled by something different, and one
boundary clean for what fools all of them.
"""

from __future__ import annotations

import sys
from contextlib import asynccontextmanager
from datetime import UTC, datetime
from pathlib import Path

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
from blast_radar.brief import render_markdown
from blast_radar.config import ScoringWeights
from blast_radar.datahub_adapter import entity_from_payload
from blast_radar.models import BlastReport, ConsumerNode, Entity, Owner
from blast_radar.scoring import score_node
from blast_radar.writeback import WriteOutcome
from inmemory_datahub import InMemoryDataHub


def connect_datahub(monkeypatch, datahub):
    """Replace the transport, and only the transport."""

    @asynccontextmanager
    async def fake_open(settings):
        yield datahub

    monkeypatch.setattr(cli, "open_datahub", fake_open)
    return datahub


# A closing tag with nothing open. Rich raises MarkupError on this, which used
# to kill the render after the scan had run and, with --publish, already
# written its verdict into the catalog.
CRASHING_NAME = "orders[/bold]"
# Four extra cells, a score, and a severity nobody computed.
FORGING_NAME = "orders | 99 | LOW | nothing depends on this"

# Built with chr() rather than written out, because a file containing these is
# a file nobody can review: the override reverses the source line it sits on,
# and the other two are invisible.
RIGHT_TO_LEFT_OVERRIDE = chr(0x202E)
ZERO_WIDTH_SPACE = chr(0x200B)
LINE_SEPARATOR = chr(0x2028)

# Weights that put governance in the top two for any node, so a tag reliably
# reaches the evidence column. Under the shipped defaults it can get there as
# well, on the outer ring of a deep walk where proximity has decayed below
# the governance weight, but only for some nodes: setting the weights makes
# the case the test is about independent of the node it uses. Weights are
# exposed as data so an organisation can retune them, which is also what this
# is.
GOVERNANCE_FIRST = ScoringWeights(
    consumer_type=5.0,
    proximity=5.0,
    fan_out=5.0,
    usage=5.0,
    ownership_gap=50.0,
    governance_signal=30.0,
)


def consumer(name: str, *, owner: str, tag: str, urn: str, enriched: bool = True) -> ConsumerNode:
    return ConsumerNode(
        entity=Entity(
            urn=urn,
            entity_type="DASHBOARD",
            name=name,
            owners=[Owner("urn:li:corpuser:ana", owner)] if owner else [],
            tags=[tag],
        ),
        hop=1,
        transform_hop=1,
        downstream_count=20,
        query_count=90,
        enriched=enriched,
        usage_known=True,
    )


def report_named(
    name: str,
    *,
    owner: str = "Ana",
    tag: str = "tier1",
    weights: ScoringWeights | None = None,
    every_section: bool = False,
) -> BlastReport:
    """A report carrying `name`, optionally in every section of the brief.

    The brief has two list sections a ranked-only fixture never renders: the
    assets nobody owns, and the assets nobody checked. A test whose every node
    is owned and enriched cannot see an escape in either, which is how one
    survived an enumeration written to end exactly that.
    """
    nodes = [consumer(name, owner=owner, tag=tag, urn="urn:li:dashboard:hostile")]
    if every_section:
        nodes += [
            consumer(name, owner="", tag=tag, urn="urn:li:dashboard:ownerless"),
            consumer(name, owner="", tag=tag, urn="urn:li:dashboard:unchecked", enriched=False),
        ]
    return BlastReport(
        root=Entity(urn="urn:li:dataset:root", entity_type="DATASET", name="shop.orders"),
        change_summary="dropping column promo_code",
        generated_at=datetime(2026, 7, 24, tzinfo=UTC),
        assessments=[score_node(node, weights or ScoringWeights(), 5) for node in nodes],
        hops_traversed=1,
    )


def ranked_rows(brief: str) -> list[str]:
    return [line for line in brief.splitlines() if line.startswith("| 1 ")]


def unescaped(text: str, character: str) -> int:
    """How many of `character` a Markdown renderer would treat as syntax.

    Counted by walking the string rather than by subtracting a `count` of the
    escaped form. `\\\\|` is an escaped backslash followed by a live pipe, and
    a substring count reads it as an escaped pipe - calling a forged cell
    safe. Walking gets the parity right, and every assertion below depends on
    that being right.
    """
    live = 0
    index = 0
    while index < len(text):
        if text[index] == "\\":
            index += 2
            continue
        live += text[index] == character
        index += 1
    return live


def unescaped_pipes(row: str) -> int:
    return unescaped(row, "|")


def test_a_name_cannot_forge_a_cell_in_the_published_brief():
    """The brief is saved into DataHub as a document other people read."""
    brief = render_markdown(report_named(FORGING_NAME))

    rows = ranked_rows(brief)
    assert len(rows) == 1
    # Six columns means seven pipes. Anything more is a cell the asset's own
    # name opened.
    assert unescaped_pipes(rows[0]) == 7
    assert "\\| 99 \\| LOW" in rows[0]


def test_a_backslash_before_a_pipe_does_not_smuggle_one_through():
    """`orders \\| 99` escapes to an escaped backslash and a live pipe unless
    the backslash is escaped first, which is the whole reason `_cell` starts
    there. The naive check for this passes, which is how it hides."""
    brief = render_markdown(report_named("orders \\| 99 | LOW | safe"))

    rows = ranked_rows(brief)
    assert len(rows) == 1
    assert unescaped_pipes(rows[0]) == 7


def test_a_name_cannot_forge_a_whole_row():
    """A newline is the other half of the same trick, and it is stripped where
    the metadata enters rather than in each renderer."""
    entity = entity_from_payload(
        {"urn": "urn:li:dataset:x", "name": "orders\n| 2 | ghost | Dataset | 0 | LOW | safe |"}
    )
    brief = render_markdown(report_named(entity.name))

    assert "\n" not in entity.name
    assert len(ranked_rows(brief)) == 1
    assert "| 2 | ghost" not in brief


def test_a_tag_cannot_forge_a_cell_through_the_evidence_column():
    """A tag reaches the ranked table inside the governance factor's evidence.

    Under weights that put governance in the top two, which this test sets so
    the case does not depend on which node it happens to use. The evidence
    column shows the two largest factors, and on a node where proximity and
    fan-out both dominate it, the tag is not one of them.

    The owner name is not checked here, for a reason worth writing down: an
    owned asset scores zero on the ownership gap, `top_factors` drops zero
    contributors, and an unowned one has no name to print. So an owner name
    can never reach this column at all - it reaches the brief through the
    who-to-tell list, which the test below covers.
    """
    brief = render_markdown(report_named("orders", tag="tier1 | 0 | LOW", weights=GOVERNANCE_FIRST))

    rows = ranked_rows(brief)
    assert len(rows) == 1
    # Lower-cased by the marker match, escaped by the renderer.
    assert "marked tier1 \\| 0 \\| low" in rows[0]
    assert unescaped_pipes(rows[0]) == 7


def test_the_who_to_tell_list_survives_a_hostile_owner_name():
    brief = render_markdown(report_named("orders", owner="Ana | fired"))

    notify = [line for line in brief.splitlines() if line.startswith("- **")]
    assert notify == ["- **Ana \\| fired** - orders"]


def test_invisible_characters_never_reach_a_rendered_name():
    """A right-to-left override reverses everything printed after it, which is
    enough to make one asset's name read as another's.

    They become a space rather than nothing. Deleting a zero-width space would
    join the halves either side of it, so `orders<zwsp>_v2` would render as the
    name of a real and different asset.
    """
    override = entity_from_payload(
        {"urn": "urn:li:dataset:x", "name": f"orders{RIGHT_TO_LEFT_OVERRIDE}_dloc\tstaging "}
    )
    zero_width = entity_from_payload(
        {"urn": "urn:li:dataset:y", "name": f"orders{ZERO_WIDTH_SPACE}_v2"}
    )

    assert override.name == "orders _dloc staging"
    assert zero_width.name == "orders _v2"


def test_a_name_that_is_nothing_but_invisible_characters_falls_back_to_the_urn():
    """An empty name is a cell a reader cannot connect back to any asset."""
    entity = entity_from_payload(
        {"urn": "urn:li:dataset:x", "name": f"{ZERO_WIDTH_SPACE}{LINE_SEPARATOR} "}
    )

    assert entity.name == "urn:li:dataset:x"


def test_a_name_cannot_close_a_code_span_or_open_a_link():
    """A brief published onto an asset page carries the catalog's credibility,
    so a link an asset named itself into is phishing, not formatting."""
    hostile = "orders`  **[reset your credentials](https://not-datahub.example)**"
    row = ranked_rows(render_markdown(report_named(hostile)))[0]

    # No bracket left to open a link, and no backtick left to end a span.
    assert unescaped(row, "[") == 0
    assert unescaped(row, "]") == 0
    assert unescaped(row, "`") == 0
    assert "\\[reset your credentials\\]" in row


def test_a_name_cannot_open_a_raw_html_tag():
    """At least two renderers see this brief, and one of them may allow HTML."""
    row = ranked_rows(render_markdown(report_named("orders<img src=x onerror=alert(1)>")))[0]

    assert unescaped(row, "<") == 0
    assert "orders\\<img" in row


def test_the_change_summary_reaches_the_document_as_data_too():
    """It is typed by the caller, not read from the catalog, but it lands in
    the same published document and a CI job passes through whatever the
    branch was called."""
    report = report_named("orders")
    report.change_summary = "dropping [promo_code](https://not-datahub.example)"

    change_line = next(
        line for line in render_markdown(report).splitlines() if line.startswith("**Change:**")
    )

    assert unescaped(change_line, "[") == 0
    assert unescaped(change_line, "]") == 0
    assert "\\[promo_code\\]" in change_line


# --- Every string a report puts into the two renderers -------------------
#
# The spot checks above each pin one call site, and the list of call sites
# has been wrong three times: a missed one is invisible to a test that names
# the surface it forgot. These two drive every string a `BlastReport`'s own
# producers reach a renderer with, from that producer, through `scan`.
#
# That is a scope, not a guarantee. The strings the other three commands
# print come from elsewhere - the server's advertised tools, GMS's error
# text, the environment - and they are covered further down, separately,
# because no fixture built from a report can reach them.

MARKER = "zx7"
# Every character `_cell` escapes, behind a marker unique enough to find in
# the output and to make a substring check unambiguous.
STRUCTURAL = "|`[]<"
ESCAPED = "".join(f"\\{character}" for character in STRUCTURAL)

PRODUCERS = ("name", "root_name", "root_urn", "change", "entity_type", "owner", "tag")


def build_report(field: str, payload: str) -> BlastReport:
    """A report whose `field`, and only `field`, carries `payload`.

    The tag needs a governance marker in it or the factor never fires and the
    string never reaches a renderer, and it needs weights that put governance
    in the top two or it never reaches the evidence column. The owner needs
    the shipped weights, since an owner only appears in the who-to-tell list
    and that list is the high-and-above band.
    """
    tagged = f"{payload}tier1" if field == "tag" else "tier1"
    report = report_named(
        payload if field == "name" else "orders",
        owner=payload if field == "owner" else "Ana",
        tag=tagged,
        weights=GOVERNANCE_FIRST if field == "tag" else None,
        # Renders the two list sections as well as the ranked table, so an
        # escape that only exists in one of them cannot hide in the others.
        every_section=True,
    )
    entity = report.assessments[0].node.entity
    if field == "root_name":
        report.root.name = payload
    if field == "root_urn":
        report.root.urn = payload
    if field == "change":
        report.change_summary = payload
    if field == "entity_type":
        entity.entity_type = payload
    return report


@pytest.mark.parametrize("field", PRODUCERS)
def test_no_producer_can_put_markdown_syntax_in_the_published_brief(field):
    """Every string the brief renders, driven from its own producer.

    The spot checks above each pin one call site, and the list of call sites
    has been wrong twice: a missed one is invisible to a test that names the
    surface it forgot. Lower-cased on both sides because two of these paths
    change case on the way through.
    """
    payload = f"{MARKER}{field}{STRUCTURAL}"

    brief = render_markdown(build_report(field, payload)).lower()

    marked = f"{MARKER}{field}"
    assert marked in brief, f"{field} never reached the brief, so this would prove nothing"
    assert marked + STRUCTURAL not in brief
    assert marked + ESCAPED in brief


# Which of those strings the terminal actually prints. The brief is the
# fuller surface: it has a who-to-tell list, it names the scanned asset's URN,
# and it carries the change summary the report was built with. The table's
# title looks like that last one but is not - it renders the `--change`
# argument the command was given, which the test below drives separately.
TERMINAL_VISIBLE = ("name", "root_name", "entity_type", "tag")
BRIEF_ONLY = ("root_urn", "owner", "change")


@pytest.mark.parametrize("field", PRODUCERS)
def test_no_producer_can_break_the_terminal_render(monkeypatch, field):
    """The same enumeration against Rich, where the failure is louder: one
    unmatched closing tag refuses the whole render, and by then the scan has
    run and, with --publish, already written."""
    payload = f"{MARKER}{field}[/bold]"

    result = run_scan_printing(monkeypatch, build_report(field, payload))
    printed = result.output.lower()

    assert result.exit_code == 0
    assert result.exception is None
    if field in TERMINAL_VISIBLE:
        assert f"{MARKER}{field}" in printed
    if field in BRIEF_ONLY:
        assert f"{MARKER}{field}" not in printed


def run_scan_publishing(monkeypatch, outcome):
    async def fake_scan(urn, change, context, *, with_usage, publish):
        return report_named("orders"), outcome

    monkeypatch.setattr(cli, "_scan", fake_scan)
    return CliRunner().invoke(cli.app, ["scan", "urn:li:dataset:root", "--publish"])


def test_the_publish_lines_are_not_markup_either(monkeypatch):
    """Both lines quote back what was written, which means URNs and whatever
    the server said when a write failed. Reachable only with --publish, which
    is why an enumeration built on the no-publish path never saw them - and
    it is the path where a crash costs the most, since the catalog has
    already been changed by the time these print."""
    outcome = WriteOutcome(
        published=[f"published brief 'Blast radius: shop.orders [{MARKER}urn]'"],
        skipped=[f"tagging with urn:li:tag:x: {MARKER}error[/bold]"],
    )

    result = run_scan_publishing(monkeypatch, outcome)

    # Non-zero because a write was skipped, but rendered rather than raised.
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert f"{MARKER}urn" in result.output
    assert f"{MARKER}error[/bold]" in result.output


@pytest.mark.parametrize("failure", [LookupError, RuntimeError])
def test_a_failure_message_is_not_markup(monkeypatch, failure):
    """The text of these comes from DataHub: an unreadable lineage payload is
    reported with 200 characters of the payload itself in the message."""
    message = f"{MARKER}failure[/bold] in the reply"

    async def fake_scan(urn, change, context, *, with_usage, publish):
        raise failure(message)

    monkeypatch.setattr(cli, "_scan", fake_scan)
    result = CliRunner().invoke(cli.app, ["scan", "urn:li:dataset:root"])

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert f"{MARKER}failure[/bold]" in result.output


def test_the_written_path_is_not_markup(monkeypatch, tmp_path):
    """Square brackets are legal in a filename, and the success line prints
    the path back."""
    out = tmp_path / f"{MARKER}[bold].md"

    async def fake_scan(urn, change, context, *, with_usage, publish):
        return report_named("orders"), None

    monkeypatch.setattr(cli, "_scan", fake_scan)
    result = CliRunner().invoke(cli.app, ["scan", "urn:li:dataset:root", "--out", str(out)])

    assert result.exit_code == 0
    assert result.exception is None
    assert f"{MARKER}[bold].md" in result.output


def run_failing_write(monkeypatch, tmp_path, *, filename: str, message: str):
    """Run a scan whose `--out` write fails with exactly `message`.

    The real failure this stands in for repeats the path inside the operating
    system's own message, which is what makes it useless as a test: with the
    path in both halves of the line, removing either escape leaves the other
    one printing it and the assertion still holds. Choosing the message here
    is what separates the two sites.
    """

    async def fake_scan(urn, change, context, *, with_usage, publish):
        return report_named("orders"), None

    def refuse(self, *args, **kwargs):
        raise OSError(message)

    monkeypatch.setattr(cli, "_scan", fake_scan)
    monkeypatch.setattr(Path, "write_text", refuse)
    out = tmp_path / filename
    result = CliRunner().invoke(cli.app, ["scan", "urn:li:dataset:root", "--out", str(out)])
    # An exit code of 1 cannot tell a refusal from a traceback, and the whole
    # point of this path is that it is a sentence rather than a stack trace.
    assert result.exception is None or isinstance(result.exception, SystemExit)
    return result


def test_a_failed_write_prints_the_path_as_data(monkeypatch, tmp_path):
    result = run_failing_write(monkeypatch, tmp_path, filename=f"{MARKER}[bold].md", message="no")

    assert result.exit_code == 1
    assert "Could not write the brief" in result.output
    assert f"{MARKER}[bold].md" in result.output


def test_a_failed_write_prints_the_reason_as_data(monkeypatch, tmp_path):
    """The reason comes from the operating system, in whatever language and
    with whatever characters that uses."""
    result = run_failing_write(
        monkeypatch, tmp_path, filename="brief.md", message=f"{MARKER}denied[/bold]"
    )

    assert result.exit_code == 1
    assert f"{MARKER}denied[/bold]" in result.output


# --- The other three commands -------------------------------------------
#
# `scan` is not the only surface. `tools` prints what the server advertises,
# `setup` prints what GMS said about a mutation, and `doctor` prints the
# configuration and the exception behind a failed connection - which is the
# command `scan` tells people to run when it breaks, so a refused render
# there kills the diagnosis of the failure being diagnosed. None of these
# producers is ours.


def hostile_catalog(**overrides):
    """A server advertising a tool whose name is a closing markup tag."""
    return InMemoryDataHub(
        tool_names=(f"get_lineage{MARKER}[/bold]", "get_entities"),
        **overrides,
    )


@pytest.mark.parametrize("arguments", [["tools"], ["tools", "--schema"], ["tools", "--json"]])
def test_a_tool_name_from_the_server_is_not_markup(monkeypatch, arguments):
    connect_datahub(monkeypatch, hostile_catalog())

    result = CliRunner().invoke(cli.app, arguments)

    assert result.exit_code == 0
    assert result.exception is None


def test_doctor_survives_a_server_advertising_markup(monkeypatch):
    connect_datahub(monkeypatch, hostile_catalog())

    result = CliRunner().invoke(cli.app, ["doctor"])

    # Exit 1 because a read tool it needs is missing under that spelling, but
    # the diagnosis renders rather than raising.
    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Missing read tools" in result.output


def test_doctor_survives_an_unreachable_server_that_says_so_in_markup(monkeypatch):
    @asynccontextmanager
    async def refuse(settings):
        raise ConnectionRefusedError(f"{MARKER}refused[/bold]")
        yield  # pragma: no cover - unreachable, present so this stays a generator

    monkeypatch.setattr(cli, "open_datahub", refuse)

    result = CliRunner().invoke(cli.app, ["doctor"])

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert f"{MARKER}refused[/bold]" in result.output


def test_setup_survives_a_graphql_error_carrying_markup(monkeypatch):
    """GMS's own error text, passed through verbatim by `ensure_risk_property`."""
    from blast_radar.bootstrap import BootstrapResult

    refused = BootstrapResult(False, f"{MARKER}denied[/bold]")
    monkeypatch.setattr(cli, "ensure_risk_property", lambda url, token: refused)
    monkeypatch.setattr(cli, "ensure_severity_tags", lambda url, token: [refused])

    result = CliRunner().invoke(cli.app, ["setup"])

    assert result.exit_code == 0
    assert result.exception is None
    assert f"{MARKER}denied[/bold]" in result.output


def test_a_configuration_error_quotes_the_bad_value_without_rendering_it(monkeypatch):
    """The message names the value that was wrong, which is the whole point of
    it, and that value came from an environment nobody validated."""
    monkeypatch.setenv("BLAST_RADAR_MAX_NODES", f"{MARKER}[/bold]")

    result = CliRunner().invoke(cli.app, ["scan", "urn:li:dataset:root"])

    assert result.exit_code == 1
    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert "Configuration error" in result.output
    assert f"{MARKER}[/bold]" in result.output


def test_the_gms_url_is_printed_as_data(monkeypatch):
    """In the path, not the host: a bracket in a hostname is refused by URL
    parsing itself, which is the one part of this string that cannot carry
    markup. A GMS behind a gateway prefix is a documented setup, and that
    prefix is whatever the environment says it is."""
    monkeypatch.setenv("DATAHUB_GMS_URL", f"http://localhost:8080/{MARKER}[/bold]")
    connect_datahub(monkeypatch, InMemoryDataHub())

    # Both commands print it, and each has its own call site.
    doctored = CliRunner().invoke(cli.app, ["doctor"])
    listed = CliRunner().invoke(cli.app, ["tools"])

    assert doctored.exception is None or isinstance(doctored.exception, SystemExit)
    assert listed.exception is None
    assert f"{MARKER}[/bold]" in doctored.output
    assert f"{MARKER}[/bold]" in listed.output


def test_the_mcp_command_line_is_printed_as_data(monkeypatch):
    """`doctor` prints the command it would spawn, and both halves of it come
    from the environment so a judge can point this at their own build."""
    monkeypatch.setenv("BLAST_RADAR_MCP_COMMAND", f"{MARKER}uvx[/bold]")
    monkeypatch.setenv("BLAST_RADAR_MCP_ARGS", f"mcp-server-datahub {MARKER}arg[/bold]")
    connect_datahub(monkeypatch, InMemoryDataHub())

    result = CliRunner().invoke(cli.app, ["doctor"])

    assert result.exception is None or isinstance(result.exception, SystemExit)
    assert f"{MARKER}uvx[/bold]" in result.output
    assert f"{MARKER}arg[/bold]" in result.output


def test_a_legitimate_name_is_left_alone():
    """The escapes must not corrupt the normal case, which is all of them."""
    entity = entity_from_payload(
        {"urn": "urn:li:dataset:x", "name": "shop.orders_v2 (PROD)", "type": "DATASET"}
    )
    brief = render_markdown(report_named(entity.name))

    assert entity.name == "shop.orders_v2 (PROD)"
    assert "| shop.orders_v2 (PROD) |" in brief
    assert "\\" not in brief


ENV_VARS = ("DATAHUB_GMS_URL", "DATAHUB_GMS_TOKEN", "BLAST_RADAR_MAX_NODES")


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("COLUMNS", "200")


def run_scan_printing(monkeypatch, report: BlastReport):
    async def fake_scan(urn, change, context, *, with_usage, publish):
        return report, None

    monkeypatch.setattr(cli, "_scan", fake_scan)
    return CliRunner().invoke(cli.app, ["scan", "urn:li:dataset:root"])


def test_a_name_that_closes_a_markup_tag_does_not_crash_the_terminal(monkeypatch):
    """The scan is already finished by the time this renders, so a crash here
    loses a completed run - and with --publish, one that already wrote."""
    result = run_scan_printing(monkeypatch, report_named(CRASHING_NAME))

    assert result.exit_code == 0
    assert result.exception is None
    assert CRASHING_NAME in result.output


def test_a_name_cannot_colour_its_own_row(monkeypatch):
    """Severity is the one thing the colours mean, and it is ours to decide."""
    result = run_scan_printing(monkeypatch, report_named("[red]orders[/red]"))

    assert result.exit_code == 0
    assert "[red]orders[/red]" in result.output


def test_a_tag_cannot_crash_the_terminal_through_the_evidence_column(monkeypatch):
    """The evidence column is a third string reaching Rich, and it carries
    whatever the catalog spells its criticality markers with.

    Separate from the name test on purpose: an escape at one call site says
    nothing about the other two, and both were reachable with the same input.
    """
    hostile = report_named("orders", tag=f"tier1{CRASHING_NAME}", weights=GOVERNANCE_FIRST)

    result = run_scan_printing(monkeypatch, hostile)

    assert result.exit_code == 0
    assert result.exception is None
    assert "tier1orders[/bold]" in result.output


def test_the_scanned_asset_s_own_name_cannot_crash_the_terminal(monkeypatch):
    """The summary line prints the root's name, which comes back from the
    catalog like every other one. The consumer names in the table are escaped
    at a different call site and prove nothing about this one."""
    report = report_named("orders")
    report.root.name = CRASHING_NAME

    result = run_scan_printing(monkeypatch, report)

    assert result.exit_code == 0
    assert result.exception is None
    assert CRASHING_NAME in result.output


def test_the_change_summary_is_the_callers_own_text_but_still_not_markup(monkeypatch):
    async def fake_scan(urn, change, context, *, with_usage, publish):
        return report_named("orders"), None

    monkeypatch.setattr(cli, "_scan", fake_scan)
    result = CliRunner().invoke(
        cli.app, ["scan", "urn:li:dataset:root", "--change", "dropping [/bold]promo_code"]
    )

    assert result.exit_code == 0
    assert result.exception is None
