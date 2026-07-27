"""Tests for the one-time GraphQL setup.

This is the only module that leaves the MCP surface, and it is the one that
carries the personal access token in a header, so the cases worth pinning are
the transport ones: which connection class is chosen, what the token is
attached to, and what happens when GMS answers with an error.

The connection is faked at the class the module imported, which is the same
seam the security fix relies on. `HTTPConnection` and `HTTPSConnection` take a
host and are picked here in code, so there is no URL for a `file://` scheme to
be smuggled through - and these tests fail if that ever goes back to a handler
chosen from the URL.
"""

from __future__ import annotations

import json
from typing import ClassVar

import pytest

from blast_radar import bootstrap
from blast_radar.bootstrap import (
    SCORED_ENTITY_TYPES,
    _graphql,
    ensure_risk_property,
    ensure_severity_tags,
)
from blast_radar.models import Severity
from blast_radar.scoring import CONSUMER_TYPE_WEIGHTS

GMS = "http://localhost:8080"
FAKE_PAT = "pat-secret-value"


class FakeResponse:
    def __init__(self, payload: str, status: int = 200, reason: str = "OK") -> None:
        self._payload = payload
        self.status = status
        self.reason = reason

    def read(self) -> bytes:
        return self._payload.encode("utf-8")


class FakeConnection:
    """Records what a request would have put on the wire."""

    # A registry rather than a return value: `ensure_severity_tags` opens one
    # connection per tag, so the assertions need all of them, not the last.
    instances: ClassVar[list[FakeConnection]] = []
    # Declared, not assigned: the factory in `install` sets it after
    # construction, so `getresponse` reads an attribute nothing on this class
    # ever writes.
    response: FakeResponse

    def __init__(self, host, port=None, timeout=None) -> None:
        self.host = host
        self.port = port
        self.timeout = timeout
        self.requests: list[tuple[str, str, bytes, dict[str, str]]] = []
        self.closed = False
        self.scheme = "unset"
        FakeConnection.instances.append(self)

    def request(self, method, path, body, headers) -> None:
        self.requests.append((method, path, body, headers))

    def getresponse(self) -> FakeResponse:
        return self.response

    def close(self) -> None:
        self.closed = True


def install(monkeypatch, *payloads: object, status: int = 200, reason: str = "OK"):
    """Answer each connection with the next payload in turn.

    A queue rather than one canned reply, because `ensure_severity_tags` opens
    a connection per tag and the interesting cases are the ones where the
    replies differ.
    """
    FakeConnection.instances = []
    queue = list(payloads)

    def build(scheme):
        def factory(host, port=None, timeout=None):
            connection = FakeConnection(host, port, timeout)
            connection.scheme = scheme
            body = queue.pop(0) if queue else {"data": {}}
            connection.response = FakeResponse(
                body if isinstance(body, str) else json.dumps(body), status, reason
            )
            return connection

        return factory

    monkeypatch.setattr(bootstrap, "HTTPConnection", build("http"))
    monkeypatch.setattr(bootstrap, "HTTPSConnection", build("https"))
    return FakeConnection.instances


@pytest.mark.parametrize(
    "url",
    [
        # The substitution the rewrite exists to make unrepresentable.
        "file:///etc/passwd",
        "ftp://catalog.internal/graphql",
        "gopher://catalog.internal",
        "localhost:8080",
    ],
)
def test_a_url_that_is_not_http_is_refused_before_anything_is_sent(monkeypatch, url):
    connections = install(monkeypatch)

    with pytest.raises(ValueError, match="http or https"):
        _graphql(url, "query {}", {}, None)

    assert connections == []


def test_a_url_with_no_host_is_refused(monkeypatch):
    install(monkeypatch)

    with pytest.raises(ValueError, match="no host"):
        _graphql("http:///api", "query {}", {}, None)


def test_https_and_http_pick_different_connection_classes(monkeypatch):
    install(monkeypatch, {"data": {}}, {"data": {}})

    _graphql("https://catalog.internal", "query {}", {}, None)
    _graphql("http://localhost:8080", "query {}", {}, None)

    assert [connection.scheme for connection in FakeConnection.instances] == ["https", "http"]


def test_the_token_travels_as_a_bearer_header_and_only_when_there_is_one(monkeypatch):
    install(monkeypatch, {"data": {}}, {"data": {}})

    _graphql(GMS, "query {}", {}, FAKE_PAT)
    _graphql(GMS, "query {}", {}, None)

    with_token, without_token = FakeConnection.instances
    assert with_token.requests[0][3]["Authorization"] == f"Bearer {FAKE_PAT}"
    assert "Authorization" not in without_token.requests[0][3]


def test_the_graphql_path_is_appended_to_whatever_prefix_gms_sits_behind(monkeypatch):
    install(monkeypatch, {"data": {}}, {"data": {}})

    _graphql("http://localhost:8080", "query {}", {}, None)
    _graphql("http://gateway.internal/datahub/", "query {}", {}, None)

    assert [connection.requests[0][1] for connection in FakeConnection.instances] == [
        "/api/graphql",
        "/datahub/api/graphql",
    ]


def test_the_query_and_variables_go_out_as_one_json_body(monkeypatch):
    install(monkeypatch, {"data": {}})

    _graphql(GMS, "mutation x", {"input": {"id": "risk"}}, None)

    method, _, body, headers = FakeConnection.instances[0].requests[0]
    assert method == "POST"
    assert headers["Content-Type"] == "application/json"
    assert json.loads(body) == {"query": "mutation x", "variables": {"input": {"id": "risk"}}}


def test_an_http_error_is_raised_rather_than_parsed_as_an_empty_reply(monkeypatch):
    """A 401 that decoded to `{}` would read like "nothing needed creating"."""
    install(monkeypatch, "unauthorized", status=401, reason="Unauthorized")

    with pytest.raises(RuntimeError, match="401 Unauthorized"):
        _graphql(GMS, "query {}", {}, FAKE_PAT)


def test_the_connection_is_closed_even_when_the_call_fails(monkeypatch):
    install(monkeypatch, "boom", status=500, reason="Server Error")

    with pytest.raises(RuntimeError):
        _graphql(GMS, "query {}", {}, FAKE_PAT)

    assert FakeConnection.instances[0].closed is True


def test_a_created_property_reports_the_urn_it_got_back(monkeypatch):
    install(
        monkeypatch,
        {"data": {"createStructuredProperty": {"urn": "urn:li:structuredProperty:blastRadar"}}},
    )

    result = ensure_risk_property(GMS, FAKE_PAT)

    assert result.created is True
    assert "urn:li:structuredProperty:blastRadar" in result.detail


def test_the_property_is_defined_for_every_type_the_scorer_can_rank(monkeypatch):
    """Two lists, one decision, and they drifted: the scorer ranked an ML
    feature table while `setup` defined the property without one, so the
    publish failed with "no valid property assignments remain".

    Asserted against `CONSUMER_TYPE_WEIGHTS` rather than against
    `SCORED_ENTITY_TYPES` itself. Comparing the constant to itself is how the
    drift survived a test that read like it was checking for it.
    """
    install(monkeypatch, {"data": {}})

    ensure_risk_property(GMS, None)

    defined = json.loads(FakeConnection.instances[0].requests[0][2])
    sent = defined["variables"]["input"]["entityTypes"]
    assert sent == list(SCORED_ENTITY_TYPES)

    covered = {entity_type.rsplit(".", 1)[-1].lower() for entity_type in sent}
    scored = {entity_type.lower() for entity_type in CONSUMER_TYPE_WEIGHTS}
    assert scored - covered == set()


@pytest.mark.parametrize("message", ["Property already exists", "CONFLICT on write"])
def test_re_running_setup_is_not_reported_as_a_failure(monkeypatch, message):
    install(monkeypatch, {"errors": [{"message": message}]})

    result = ensure_risk_property(GMS, None)

    assert result.created is False
    assert result.detail == "property already defined"


def test_a_real_error_keeps_its_message(monkeypatch):
    install(monkeypatch, {"errors": [{"message": "entityType does not exist"}]})

    result = ensure_risk_property(GMS, None)

    assert result.created is False
    assert "entityType does not exist" in result.detail


def test_an_error_with_no_message_field_still_says_something(monkeypatch):
    install(monkeypatch, {"errors": [{"code": 500}]})

    result = ensure_risk_property(GMS, None)

    assert result.created is False
    assert result.detail


def test_every_severity_gets_a_tag_with_a_description(monkeypatch):
    """DataHub refuses to attach a tag whose entity does not exist, so all four
    are created up front rather than implied by the first publish.

    Checked against the `Severity` enum, not against `TAG_DESCRIPTIONS`.
    Dropping a severity from that mapping shrinks the calls and the
    expectation together, so a self-referential version of this passes while
    a whole band goes untagged.
    """
    install(monkeypatch, *[{"data": {}} for _ in Severity])

    results = ensure_severity_tags(GMS, None)

    assert len(results) == len(list(Severity))
    assert all(result.created for result in results)

    sent = [
        json.loads(connection.requests[0][2])["variables"]["input"]
        for connection in FakeConnection.instances
    ]
    assert {payload["id"] for payload in sent} == {
        f"blast-radar-{severity.value}" for severity in Severity
    }
    # The description is what tells a reader in DataHub what the tag means, so
    # an empty one is a tag nobody can act on.
    assert all(payload["description"].strip() for payload in sent)


def test_tags_that_already_exist_are_reported_per_tag_and_the_rest_still_run(monkeypatch):
    install(
        monkeypatch,
        {"errors": [{"message": "already exists"}]},
        {"data": {}},
        {"errors": [{"message": "token expired"}]},
        {"data": {}},
    )

    results = ensure_severity_tags(GMS, None)

    assert [result.created for result in results] == [False, True, False, True]
    assert "already defined" in results[0].detail
    assert "token expired" in results[2].detail
