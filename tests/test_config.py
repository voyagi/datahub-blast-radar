"""Tests for environment-resolved settings.

The budgets get their own tests because a bad one is silent by nature: a scan
that never looked and a scan that found nothing produce the same empty cone,
and the empty cone reads as "safe to change".
"""

from __future__ import annotations

import pytest

from blast_radar.config import PLAINTEXT_OPT_OUT_VAR, Settings

BUDGETS = ("BLAST_RADAR_MAX_HOPS", "BLAST_RADAR_MAX_NODES", "BLAST_RADAR_RESULTS_PER_HOP")


@pytest.mark.parametrize("name", BUDGETS)
def test_a_budget_below_one_is_refused_and_names_itself(monkeypatch, name):
    """Whoever mistyped it needs the variable name, not a stack trace."""
    monkeypatch.setenv(name, "0")

    with pytest.raises(ValueError, match=f"{name} must be at least 1"):
        Settings.from_env()


def test_a_negative_budget_is_refused(monkeypatch):
    monkeypatch.setenv("BLAST_RADAR_MAX_NODES", "-5")

    with pytest.raises(ValueError, match="must be at least 1"):
        Settings.from_env()


def test_a_budget_that_is_not_a_number_says_which_one(monkeypatch):
    monkeypatch.setenv("BLAST_RADAR_MAX_HOPS", "five")

    with pytest.raises(ValueError, match="BLAST_RADAR_MAX_HOPS must be a whole number"):
        Settings.from_env()


def test_the_default_budgets_resolve(monkeypatch):
    for name in BUDGETS:
        monkeypatch.delenv(name, raising=False)

    settings = Settings.from_env()

    assert (settings.max_hops, settings.max_nodes, settings.results_per_hop) == (5, 400, 100)


@pytest.fixture
def clean_env(monkeypatch):
    """A run with nothing inherited from the developer's own shell."""
    for name in (*BUDGETS, "DATAHUB_GMS_URL", "DATAHUB_GMS_TOKEN", PLAINTEXT_OPT_OUT_VAR):
        monkeypatch.delenv(name, raising=False)
    return monkeypatch


@pytest.mark.parametrize(
    "url",
    [
        # The one that motivated the check: urlopen would have read this path
        # and handed back its bytes as though GMS had answered.
        "file:///etc/passwd",
        "ftp://catalog.internal/graphql",
        "gopher://catalog.internal",
        # A bare host with no scheme parses with an empty scheme, so it must be
        # refused rather than guessed at.
        "localhost:8080",
    ],
)
def test_a_gms_url_that_is_not_http_is_refused(clean_env, url):
    clean_env.setenv("DATAHUB_GMS_URL", url)

    with pytest.raises(ValueError, match="must start with http"):
        Settings.from_env()


def test_a_gms_url_with_no_host_is_refused(clean_env):
    clean_env.setenv("DATAHUB_GMS_URL", "http://")

    with pytest.raises(ValueError, match="no host"):
        Settings.from_env()


def test_a_token_is_not_sent_over_plaintext_to_a_remote_host(clean_env):
    """The credential leak this guard exists for.

    The token grants write access to the metadata graph, and a bearer header on
    plaintext http is readable by anything on the path.
    """
    clean_env.setenv("DATAHUB_GMS_URL", "http://datahub.example.com:8080")
    clean_env.setenv("DATAHUB_GMS_TOKEN", "pat-secret-value")

    with pytest.raises(ValueError, match="Refusing to send DATAHUB_GMS_TOKEN"):
        Settings.from_env()


def test_the_refusal_does_not_put_the_token_in_the_message(clean_env):
    """An error that quotes the credential just moves it into the logs."""
    clean_env.setenv("DATAHUB_GMS_URL", "http://datahub.example.com:8080")
    clean_env.setenv("DATAHUB_GMS_TOKEN", "pat-secret-value")

    with pytest.raises(ValueError, match="Refusing to send DATAHUB_GMS_TOKEN") as caught:
        Settings.from_env()

    assert "pat-secret-value" not in str(caught.value)


@pytest.mark.parametrize(
    "url",
    [
        # The documented local setup, which has to keep working.
        "http://localhost:8080",
        "http://127.0.0.1:8080",
        "http://[::1]:8080",
        # Remote is fine once the wire is encrypted.
        "https://datahub.example.com",
    ],
)
def test_a_token_is_allowed_where_it_cannot_leak(clean_env, url):
    clean_env.setenv("DATAHUB_GMS_URL", url)
    clean_env.setenv("DATAHUB_GMS_TOKEN", "pat-secret-value")

    assert Settings.from_env().gms_token == "pat-secret-value"


def test_plaintext_to_a_remote_host_is_fine_with_no_token(clean_env):
    """Nothing to leak, so nothing to refuse."""
    clean_env.setenv("DATAHUB_GMS_URL", "http://datahub.example.com:8080")

    assert Settings.from_env().gms_token is None


def test_the_opt_out_is_respected_and_has_to_be_set_deliberately(clean_env):
    """TLS terminating in front of GMS is a real deployment, so there is a way through."""
    clean_env.setenv("DATAHUB_GMS_URL", "http://datahub.internal:8080")
    clean_env.setenv("DATAHUB_GMS_TOKEN", "pat-secret-value")
    clean_env.setenv(PLAINTEXT_OPT_OUT_VAR, "1")

    assert Settings.from_env().gms_token == "pat-secret-value"


def test_a_hostname_that_merely_looks_local_is_not_treated_as_loopback(clean_env):
    """`localhost.attacker.example` is a remote host with a reassuring prefix."""
    clean_env.setenv("DATAHUB_GMS_URL", "http://localhost.attacker.example")
    clean_env.setenv("DATAHUB_GMS_TOKEN", "pat-secret-value")

    with pytest.raises(ValueError, match="Refusing to send DATAHUB_GMS_TOKEN"):
        Settings.from_env()


# --- What the MCP server subprocess is handed ------------------------------
#
# The child process gets a token and a write switch, so what is in that
# environment is a security question, not a plumbing one.


def test_the_write_switch_handed_to_the_server_is_the_resolved_one(clean_env):
    """An inherited TOOLS_IS_MUTATION_ENABLED=true must not survive as a yes.

    The child inherits the parent environment, so the switch has to be written
    on every run rather than only when it is on. Otherwise a shell that
    exported it once leaves write tools live in a run that resolved to off.
    """
    clean_env.setenv("TOOLS_IS_MUTATION_ENABLED", "true")
    enabled = Settings.from_env()
    clean_env.setenv("TOOLS_IS_MUTATION_ENABLED", "no")
    disabled = Settings.from_env()

    assert enabled.mcp_env()["TOOLS_IS_MUTATION_ENABLED"] == "true"
    assert disabled.mcp_env()["TOOLS_IS_MUTATION_ENABLED"] == "false"


def test_the_server_is_pointed_at_the_url_this_run_resolved(clean_env):
    clean_env.setenv("DATAHUB_GMS_URL", "https://datahub.example.com/")
    clean_env.setenv("DATAHUB_GMS_TOKEN", "pat-secret-value")

    env = Settings.from_env().mcp_env()

    # The trailing slash is stripped once, at the boundary, so the subprocess
    # and the GraphQL POST cannot disagree about the URL.
    assert env["DATAHUB_GMS_URL"] == "https://datahub.example.com"
    assert env["DATAHUB_GMS_TOKEN"] == "pat-secret-value"


def test_the_child_keeps_the_parent_environment(clean_env):
    """uvx needs PATH, and a corporate network needs the proxy variables."""
    clean_env.setenv("BLAST_RADAR_TEST_MARKER", "kept")

    assert Settings.from_env().mcp_env()["BLAST_RADAR_TEST_MARKER"] == "kept"


def test_no_token_means_none_is_added(clean_env):
    env = Settings.from_env().mcp_env()

    assert "DATAHUB_GMS_TOKEN" not in env
