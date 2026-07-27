"""Blast Radar command line entrypoint."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import typer
from rich.console import Console
from rich.markup import escape
from rich.table import Table

from .bootstrap import ensure_risk_property, ensure_severity_tags
from .brief import disclosure_notes, render_markdown, render_summary
from .config import RunContext, Settings
from .datahub_adapter import DataHubReader
from .mcp_client import ToolSpec, open_datahub
from .models import BlastReport, Severity
from .scan import run_scan
from .writeback import WriteOutcome, describe_plan, execute, plan_writeback

app = typer.Typer(
    add_completion=False,
    help="Blast Radar - score the downstream blast radius of a data change, "
    "using DataHub's MCP server for both the reading and the writing back.",
)
console = Console()

# A scan cannot run without these. Only the tools actually called belong here:
# a doctor that demands a tool the code never uses fails a working instance.
REQUIRED_READ_TOOLS = {"get_entities", "get_lineage"}
# Without these the scan still runs; it just cannot publish its verdict.
# search_documents and remove_tags are in the list because publishing uses
# them to update the previous brief and to clear a stale severity tag.
WRITE_BACK_TOOLS = {
    "add_tags",
    "remove_tags",
    "add_structured_properties",
    "save_document",
    "search_documents",
}


def _settings() -> Settings:
    """Resolve settings, turning a bad environment into one readable line.

    `Settings.from_env` refuses a budget below one rather than scanning
    nothing and calling it nothing found. That is a configuration mistake, not
    a stack trace, so it gets the same treatment as a failed connection.
    """
    try:
        return Settings.from_env()
    except ValueError as exc:
        console.print(f"[red]Configuration error:[/red] {escape(str(exc))}")
        raise typer.Exit(code=1) from exc


@app.command()
def tools(
    json_out: bool = typer.Option(False, "--json", help="Print raw tool schemas as JSON."),
    show_schema: bool = typer.Option(
        False, "--schema", help="Show each tool's argument names and required fields."
    ),
) -> None:
    """List the tools the connected DataHub MCP server exposes.

    Run this first. It proves the connection works, and it is the fastest way
    to see whether the write tools are switched on for this instance.
    """
    settings = _settings()
    specs = asyncio.run(_list_tools(settings))

    if json_out:
        console.print_json(
            json.dumps(
                {name: spec.input_schema for name, spec in sorted(specs.items())},
                indent=2,
            )
        )
        return

    table = Table(title=f"DataHub MCP tools @ {escape(settings.gms_url)}")
    table.add_column("tool", style="bold")
    table.add_column("purpose")
    if show_schema:
        table.add_column("required")
        table.add_column("arguments")

    for name, spec in sorted(specs.items()):
        summary = spec.description.strip().splitlines()[0] if spec.description else ""
        row = [escape(name), escape(summary[:90])]
        if show_schema:
            # Argument names come out of the server's advertised schema, which
            # is no more ours than an asset name is.
            row.append(escape(", ".join(spec.required_arguments())) or "-")
            row.append(escape(", ".join(spec.argument_names())) or "-")
        table.add_row(*row)

    console.print(table)
    console.print(
        f"\n[bold]{len(specs)}[/bold] tools. "
        f"Mutations {'enabled' if settings.mutations_enabled else 'disabled'} "
        "(TOOLS_IS_MUTATION_ENABLED)."
    )


@app.command()
def scan(
    urn: str = typer.Argument(..., help="URN of the asset about to change."),
    change: str = typer.Option(
        "an unspecified change", "--change", "-c", help="What is changing, in words."
    ),
    # min=1: a cap below one cannot mean "show nothing", and a negative slice
    # would quietly drop rows under a sentence claiming to disclose the cut.
    limit: int = typer.Option(15, "--limit", min=1, help="Rows to show in the ranked table."),
    no_usage: bool = typer.Option(
        False, "--no-usage", help="Skip query-history lookups (one call per shortlisted asset)."
    ),
    out: str = typer.Option("", "--out", help="Write the Markdown brief to this path."),
    publish: bool = typer.Option(
        False, "--publish", help="Write the verdict back into DataHub (tags, scores, brief)."
    ),
) -> None:
    """Score the downstream blast radius of a change to URN."""
    context = RunContext(settings=_settings())
    try:
        report, outcome = asyncio.run(
            _scan(urn, change, context, with_usage=not no_usage, publish=publish)
        )
    except LookupError as exc:
        console.print(f"[red]{escape(str(exc))}[/red]")
        raise typer.Exit(code=1) from exc
    except Exception as exc:
        # A half-finished scan must not print a table that reads like a clean
        # bill of health. Say what broke and exit non-zero so CI notices.
        console.print(f"[red]Scan failed:[/red] {escape(str(exc))}")
        console.print("Run [bold]blast-radar doctor[/bold] to check the connection.")
        raise typer.Exit(code=1) from exc

    # escape(): every string that came out of the catalog is escaped before it
    # reaches Rich. An asset named `orders[/bold]` is a closing tag with no
    # opening one, which raises MarkupError and kills the render after the scan
    # has already run and, with --publish, already written. Names carrying
    # `[red]` are the quieter half: they colour a row to whatever severity
    # their author preferred. Our own markup is written here, not read out of a
    # catalog anyone can ingest into.
    console.print(f"\n[bold]{escape(report.root.name)}[/bold] - {render_summary(report)}\n")

    _print_ranked_table(report, change, limit)
    _print_disclosures(report, limit)

    published = _report_outcome(report, outcome)
    written = _write_brief(report, out, limit)

    # Both are reported before either decides the exit code. A publish that
    # partly failed and a brief that could not be saved are separate facts,
    # and whichever happened second used to hide the first.
    if not (published and written):
        raise typer.Exit(code=1)


def _report_outcome(report: BlastReport, outcome: WriteOutcome | None) -> bool:
    """Print what was published, and say whether the catalog is now complete."""
    if outcome is None:
        console.print(f"\n[dim]{describe_plan(plan_writeback(report))} Use --publish.[/dim]")
        return True

    for line in outcome.published:
        console.print(f"[green]published[/green] {escape(line)}")
    for line in outcome.skipped:
        console.print(f"[yellow]skipped[/yellow] {escape(line)}")

    if outcome.ok:
        return True
    # A release gate that exits 0 after every write failed is not a gate.
    console.print(
        f"\n[red]{len(outcome.skipped)} write(s) failed.[/red] DataHub holds a partial verdict."
    )
    return False


def _write_brief(report: BlastReport, out: str, limit: int) -> bool:
    """Save the Markdown brief, or say why it could not be saved.

    A directory, a read-only volume, or a path whose parent does not exist are
    all ordinary mistakes, and by the time they surface the scan has finished
    and, with --publish, already changed the catalog. That earns a sentence
    and a non-zero exit, not a traceback over a completed run.
    """
    if not out:
        return True
    try:
        Path(out).write_text(render_markdown(report, limit=limit), encoding="utf-8")
    except OSError as exc:
        console.print(
            f"\n[red]Could not write the brief to[/red] {escape(out)}: {escape(str(exc))}"
        )
        return False
    console.print(f"\nBrief written to [bold]{escape(out)}[/bold]")
    return True


def _print_ranked_table(report: BlastReport, change: str, limit: int) -> None:
    table = Table(title=f"Downstream impact of: {escape(change)}")
    table.add_column("#", justify="right")
    table.add_column("asset", style="bold")
    table.add_column("type")
    table.add_column("hop", justify="right")
    table.add_column("score", justify="right")
    table.add_column("severity")
    table.add_column("why")

    for index, item in enumerate(report.ranked[:limit], start=1):
        table.add_row(
            str(index),
            escape(item.node.entity.name),
            escape(item.node.entity.display_type),
            str(item.node.hop),
            f"{item.score}",
            _severity_markup(item.severity),
            escape("; ".join(factor.evidence for factor in item.top_factors[:2])),
        )

    console.print(table)


def _print_disclosures(report: BlastReport, limit: int) -> None:
    """Every reason this table shows less than the whole truth.

    Grouped into one place because they share a single rule: a cut that does not
    announce itself reads as the complete answer.
    """
    if len(report.ranked) > limit:
        console.print(
            f"[dim]Showing the {limit} highest-scoring of "
            f"{len(report.ranked)} consumers; raise --limit for more.[/dim]"
        )

    for note in disclosure_notes(report):
        # The brief is optional; this table is not. A cone that hit a budget
        # has to say so on the surface the headline command actually prints.
        # Backticks are Markdown - the terminal would show the punctuation.
        console.print(f"\n[yellow]Note:[/yellow] {note.replace('`', '')}")

    if report.unowned:
        console.print(
            f"\n[yellow]{len(report.unowned)} impacted asset(s) have no owner to notify.[/yellow]"
        )
    if report.unchecked_ownership:
        console.print(
            f"[dim]{len(report.unchecked_ownership)} asset(s) could not be enriched; "
            "their ownership is unknown, not absent.[/dim]"
        )


async def _scan(
    urn: str, change: str, context: RunContext, *, with_usage: bool, publish: bool
) -> tuple[BlastReport, WriteOutcome | None]:
    async with open_datahub(context.settings) as datahub:
        reader = DataHubReader(datahub, results_per_hop=context.settings.results_per_hop)
        report = await run_scan(urn, change, reader, context, with_usage=with_usage)
        if not publish:
            return report, None
        return report, await execute(datahub, plan_writeback(report))


def _severity_markup(severity: Severity) -> str:
    colour = {
        Severity.CRITICAL: "red",
        Severity.HIGH: "yellow",
        Severity.MODERATE: "cyan",
        Severity.LOW: "dim",
    }[severity]
    return f"[{colour}]{severity.value.upper()}[/{colour}]"


@app.command()
def setup() -> None:
    """Define the structured property that scans write scores into.

    Run once per DataHub instance, before the first `scan --publish`.
    """
    settings = _settings()
    outcomes = [
        ensure_risk_property(settings.gms_url, settings.gms_token),
        *ensure_severity_tags(settings.gms_url, settings.gms_token),
    ]
    for result in outcomes:
        colour = "green" if result.created else "yellow"
        console.print(f"[{colour}]{escape(result.detail)}[/{colour}]")


@app.command()
def doctor() -> None:
    """Check that this machine can actually run a scan.

    Judges and new contributors hit the same four problems - wrong GMS URL,
    missing token, an MCP server too old for the tools, and mutations left
    off - so each one gets a named check with the fix in the message.
    """
    settings = _settings()
    console.print(f"[bold]DataHub:[/bold] {escape(settings.gms_url)}")
    console.print(
        f"[bold]Token:[/bold] {'set' if settings.gms_token else 'MISSING (DATAHUB_GMS_TOKEN)'}"
    )
    console.print(
        f"[bold]MCP server:[/bold] {escape(settings.mcp_command)} "
        f"{escape(' '.join(settings.mcp_args))}"
    )

    try:
        specs = asyncio.run(_list_tools(settings))
    # Broad by design: doctor exists to turn any failure into a readable
    # diagnosis, so there is no exception type it should let through.
    except Exception as exc:
        console.print(f"\n[red]Could not reach the MCP server:[/red] {escape(str(exc))}")
        console.print(
            "Check that DataHub is up (datahub docker quickstart), that "
            "DATAHUB_GMS_URL points at GMS rather than the UI port, and that "
            "uvx can reach the network."
        )
        raise typer.Exit(code=1) from exc

    # These two are set differences of the module constants above, so they can
    # only ever hold our own literals. The escapes below are uniformity, not
    # protection, and no test holds them in place because none can. Replacing
    # every escape in this file with an identity call, one call site at a
    # time, leaves these two and nothing else: everything that can carry a
    # string from outside is held by a test.
    missing_reads = sorted(REQUIRED_READ_TOOLS - specs.keys())
    missing_writes = sorted(WRITE_BACK_TOOLS - specs.keys())

    console.print(f"\n[green]Connected.[/green] {len(specs)} tools advertised.")
    if missing_reads:
        console.print(
            f"[red]Missing read tools:[/red] {escape(', '.join(missing_reads))} - "
            "a scan cannot run without these. Upgrade mcp-server-datahub."
        )
    else:
        console.print("[green]All read tools a scan needs are present.[/green]")

    if missing_writes:
        console.print(
            "[yellow]Write-back unavailable:[/yellow] "
            f"{escape(', '.join(missing_writes))} missing. "
            "Set TOOLS_IS_MUTATION_ENABLED=true and use mcp-server-datahub v0.5.0+. "
            "Scans still run; they just cannot publish results back to DataHub."
        )
    else:
        console.print("[green]Write-back tools are available.[/green]")

    if missing_reads:
        raise typer.Exit(code=1)


async def _list_tools(settings: Settings) -> dict[str, ToolSpec]:
    async with open_datahub(settings) as datahub:
        return datahub.tools


def main() -> None:
    app()
