"""Runtime configuration for Blast Radar.

Everything is resolved from the environment so the tool behaves the same way
locally, in CI, and inside a judge's sandbox. No config file is required.
"""

from __future__ import annotations

import ipaddress
import os
from dataclasses import dataclass, field
from pathlib import Path
from urllib.parse import urlsplit

DEFAULT_GMS_URL = "http://localhost:8080"

# GMS speaks HTTP. Nothing else is a catalog, and `urlopen` would happily read a
# `file://` URL and hand back its contents as though a server had answered.
ALLOWED_GMS_SCHEMES = ("http", "https")

# Opt-out for sending a token over plaintext to a remote host. Named rather than
# implied, so a deployment that genuinely terminates TLS elsewhere has to say so.
PLAINTEXT_OPT_OUT_VAR = "BLAST_RADAR_ALLOW_INSECURE_TOKEN"

# The MCP server ships as a uvx-runnable package, which is how DataHub's own
# docs tell people to run it. Running it that way (instead of vendoring it)
# means we exercise the same server any MCP client would connect to.
#
# Pinned, not @latest. Running this points a subprocess at your catalog with
# mutations enabled and a token in its environment, and @latest means the
# version that happens to be newest at that minute. The adapter also depends on
# behavior verified in this version - notably that an empty lineage list is
# stripped from the reply while `total: 0` survives, which is what lets an
# absent `searchResults` be read as an empty cone. Override with
# BLAST_RADAR_MCP_ARGS after checking that still holds.
MCP_SERVER_VERSION = "0.6.0"
DEFAULT_MCP_COMMAND = "uvx"
DEFAULT_MCP_ARGS = (f"mcp-server-datahub=={MCP_SERVER_VERSION}",)


@dataclass(frozen=True)
class Settings:
    """Resolved settings for one Blast Radar run."""

    gms_url: str
    gms_token: str | None
    mcp_command: str
    mcp_args: tuple[str, ...]
    mutations_enabled: bool
    # Traversal bounds. Downstream graphs get wide fast, and an unbounded walk
    # turns a 30-second scan into a five-minute one for no extra signal.
    max_hops: int
    max_nodes: int
    # How many consumers to ask for per asset. DataHub caps its own reply, and
    # anything over this is reported as missing rather than silently dropped.
    results_per_hop: int
    debug: bool
    server_log_path: Path

    @classmethod
    def from_env(cls) -> Settings:
        gms_url = os.environ.get("DATAHUB_GMS_URL", DEFAULT_GMS_URL).rstrip("/")
        gms_token = os.environ.get("DATAHUB_GMS_TOKEN") or None
        _check_gms_url(gms_url, gms_token)
        return cls(
            gms_url=gms_url,
            gms_token=gms_token,
            mcp_command=os.environ.get("BLAST_RADAR_MCP_COMMAND", DEFAULT_MCP_COMMAND),
            mcp_args=tuple(
                os.environ.get("BLAST_RADAR_MCP_ARGS", " ".join(DEFAULT_MCP_ARGS)).split()
            ),
            # Write tools stay off in the MCP server unless this is set, so we
            # mirror the server's own switch rather than inventing a second one.
            mutations_enabled=_env_flag("TOOLS_IS_MUTATION_ENABLED"),
            # Five, not four: a normal BI cone is dataset -> dataset -> dataset
            # -> chart -> dashboard, so a four-hop budget stops one edge short
            # of the dashboards that matter most.
            max_hops=_budget("BLAST_RADAR_MAX_HOPS", "5"),
            max_nodes=_budget("BLAST_RADAR_MAX_NODES", "400"),
            results_per_hop=_budget("BLAST_RADAR_RESULTS_PER_HOP", "100"),
            debug=_env_flag("BLAST_RADAR_DEBUG"),
            server_log_path=Path(
                os.environ.get("BLAST_RADAR_SERVER_LOG", ".blast-radar/mcp-server.log")
            ),
        )

    def mcp_env(self) -> dict[str, str]:
        """Environment handed to the MCP server subprocess.

        The child inherits the parent environment so uvx, PATH, and proxy
        settings keep working; we only overlay what the server needs.
        """
        env = dict(os.environ)
        env["DATAHUB_GMS_URL"] = self.gms_url
        if self.gms_token:
            env["DATAHUB_GMS_TOKEN"] = self.gms_token
        env["TOOLS_IS_MUTATION_ENABLED"] = "true" if self.mutations_enabled else "false"
        return env


@dataclass(frozen=True)
class ScoringWeights:
    """Weights for the deterministic risk model. They sum to 100.

    Exposed as data rather than constants so the weights can be tuned per
    organisation without touching the scoring logic, and so the brief can show
    which weights produced a score.

    Why consumer_type outweighs proximity: a lineage graph already tells you
    what is near the change. What it does not tell you is where a human will
    see the breakage. A Tableau dashboard three hops out is a real failure
    someone reports on Monday; an intermediate staging table one hop out
    usually is not. Proximity still matters, because each transform in between
    is a chance for the change to be absorbed, but it is the weaker predictor
    of "this hurts".
    """

    consumer_type: float = 30.0
    proximity: float = 25.0
    fan_out: float = 20.0
    usage: float = 10.0
    ownership_gap: float = 10.0
    governance_signal: float = 5.0

    def total(self) -> float:
        return (
            self.proximity
            + self.fan_out
            + self.consumer_type
            + self.usage
            + self.ownership_gap
            + self.governance_signal
        )


@dataclass(frozen=True)
class RunContext:
    """Settings plus scoring weights for a single scan."""

    settings: Settings
    weights: ScoringWeights = field(default_factory=ScoringWeights)


def _check_gms_url(gms_url: str, token: str | None) -> None:
    """Refuse a GMS URL that is not a catalog, or that would leak the token.

    Checked here rather than at each call site because the URL leaves this
    process by two different routes: `bootstrap` posts to it directly, and
    `mcp_env` hands it to the MCP server subprocess, which does its own
    requests. One boundary check covers both, and turns what would otherwise be
    an opaque subprocess failure into a sentence naming the variable.

    Two separate refusals. A scheme other than http or https is never a catalog,
    and `file://` in particular would return the contents of a local path as
    though a server had answered. Sending a personal access token as a bearer
    header over plaintext to a host that is not loopback puts a credential with
    write access to the metadata graph on the wire, which is worth failing over
    rather than warning about. Localhost over http is the documented setup and
    stays fine.
    """
    parsed = urlsplit(gms_url)
    if parsed.scheme not in ALLOWED_GMS_SCHEMES:
        raise ValueError(
            f"DATAHUB_GMS_URL must start with http:// or https://, got {gms_url!r}. "
            f"A {parsed.scheme or 'missing'} scheme does not address a DataHub instance."
        )
    if not parsed.hostname:
        raise ValueError(f"DATAHUB_GMS_URL has no host, got {gms_url!r}.")

    if not token or parsed.scheme == "https" or _is_loopback(parsed.hostname):
        return
    if _env_flag(PLAINTEXT_OPT_OUT_VAR):
        return
    raise ValueError(
        f"Refusing to send DATAHUB_GMS_TOKEN over plaintext http to {parsed.hostname!r}. "
        "The token grants write access to the metadata graph, and a bearer header "
        "on http is readable in transit. Use https, or set "
        f"{PLAINTEXT_OPT_OUT_VAR}=1 if TLS genuinely terminates in front of GMS."
    )


def _is_loopback(hostname: str) -> bool:
    """Whether a host is this machine, judged on the literal only.

    Deliberately no DNS lookup. Resolving would make a config check depend on
    the network, and a name that resolves to loopback today can resolve
    elsewhere tomorrow, so it is not the thing worth trusting a credential to.
    """
    if hostname.lower() in {"localhost", "localhost.localdomain"}:
        return True
    try:
        return ipaddress.ip_address(hostname.strip("[]")).is_loopback
    except ValueError:
        return False


def _budget(name: str, default: str) -> int:
    """Read a traversal budget, refusing anything below one.

    This is the only place that can name the variable a typo landed in. A
    budget of zero scans nothing, and a scan that found nothing because it
    never looked reads exactly like a change that is safe to ship.
    """
    raw = os.environ.get(name, default)
    try:
        value = int(raw)
    except ValueError:
        raise ValueError(f"{name} must be a whole number, got {raw!r}.") from None
    if value < 1:
        raise ValueError(
            f"{name} must be at least 1, got {value}. A budget below that "
            "scans nothing, which is not the same as finding nothing."
        )
    return value


def _env_flag(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes", "on"}
