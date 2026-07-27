"""One-time setup of the structured property Blast Radar writes scores into.

This is the one place that talks to DataHub's GraphQL API instead of the MCP
server, for a specific reason: the MCP server can *assign* structured
properties but has no tool to *define* one. Writing a value for an undefined
property gets you a soft-defined property scoped to whatever entity type you
happened to write first - which is why, before this existed, scores landed on
datasets and were rejected on every chart and dashboard with "no valid property
assignments remain after removing values for non-existent properties".

Scanning still goes through MCP exclusively. This is setup, not runtime.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from http.client import HTTPConnection, HTTPSConnection
from urllib.parse import urlsplit

from .writeback import RISK_PROPERTY, SEVERITY_TAG_PREFIX, SEVERITY_TAG_SEPARATOR

# The tag URNs are built from a bare id, not the full URN, since createTag
# takes the id and DataHub assembles urn:li:tag:<id> itself.
TAG_PREFIX = SEVERITY_TAG_PREFIX.rsplit(":", 1)[-1]

# Every entity type a consumer can be. A property defined only for datasets is
# silently useless for the dashboards that matter most, and a type the scorer
# ranks but this list omits fails at publish time with "no valid property
# assignments remain after removing values for non-existent properties" - the
# exact error this module exists to prevent. `test_bootstrap` asserts this
# covers every type `scoring.CONSUMER_TYPE_WEIGHTS` names, because these two
# lists are one decision written down twice and they drifted once already.
SCORED_ENTITY_TYPES = (
    "urn:li:entityType:datahub.dataset",
    "urn:li:entityType:datahub.chart",
    "urn:li:entityType:datahub.dashboard",
    "urn:li:entityType:datahub.dataJob",
    "urn:li:entityType:datahub.dataFlow",
    "urn:li:entityType:datahub.mlModel",
    "urn:li:entityType:datahub.mlFeatureTable",
    "urn:li:entityType:datahub.notebook",
)

CREATE_PROPERTY = """
mutation createBlastRadarProperty($input: CreateStructuredPropertyInput!) {
  createStructuredProperty(input: $input) {
    urn
  }
}
"""


CREATE_TAG = """
mutation createBlastRadarTag($input: CreateTagInput!) {
  createTag(input: $input)
}
"""

# What each severity tag means, shown in DataHub next to the tag itself so the
# label is self-explanatory to someone who has never run this tool.
TAG_DESCRIPTIONS = {
    "critical": "Blast Radar scored this asset 75+ for a pending upstream change.",
    "high": "Blast Radar scored this asset 55-74 for a pending upstream change.",
    "moderate": "Blast Radar scored this asset 35-54 for a pending upstream change.",
    "low": "Blast Radar scored this asset under 35 for a pending upstream change.",
}


@dataclass(frozen=True)
class BootstrapResult:
    created: bool
    detail: str


def ensure_risk_property(gms_url: str, token: str | None = None) -> BootstrapResult:
    """Define the risk-score property, or report that it already exists."""
    property_id = RISK_PROPERTY.rsplit(":", 1)[-1]
    variables = {
        "input": {
            "id": property_id,
            "qualifiedName": property_id,
            "displayName": "Blast Radar risk score",
            "description": (
                "Downstream break-risk score (0-100) published by Blast Radar "
                "for the most recent scan that touched this asset."
            ),
            "valueType": "urn:li:dataType:datahub.number",
            "cardinality": "SINGLE",
            "entityTypes": list(SCORED_ENTITY_TYPES),
        }
    }

    payload = _graphql(gms_url, CREATE_PROPERTY, variables, token)
    errors = payload.get("errors") or []
    if errors:
        message = "; ".join(str(error.get("message", error)) for error in errors)
        # Re-running setup is normal and must not look like a failure.
        if "already exists" in message.lower() or "conflict" in message.lower():
            return BootstrapResult(created=False, detail="property already defined")
        return BootstrapResult(created=False, detail=message)

    urn = (payload.get("data") or {}).get("createStructuredProperty", {}).get("urn")
    return BootstrapResult(created=True, detail=f"created {urn or RISK_PROPERTY}")


def ensure_severity_tags(gms_url: str, token: str | None = None) -> list[BootstrapResult]:
    """Create the four severity tags.

    DataHub refuses to associate a tag whose entity does not exist yet
    ("Failed to validate label ... Urn does not exist"), so the tags have to be
    created before the first publish rather than implied by it.
    """
    results: list[BootstrapResult] = []
    for severity, description in TAG_DESCRIPTIONS.items():
        tag_id = f"{TAG_PREFIX}{SEVERITY_TAG_SEPARATOR}{severity}"
        payload = _graphql(
            gms_url,
            CREATE_TAG,
            {"input": {"id": tag_id, "name": tag_id, "description": description}},
            token,
        )
        errors = payload.get("errors") or []
        if errors:
            message = "; ".join(str(error.get("message", error)) for error in errors)
            already_there = "already exists" in message.lower() or "conflict" in message.lower()
            results.append(
                BootstrapResult(
                    created=False,
                    detail=f"{tag_id}: {'already defined' if already_there else message}",
                )
            )
            continue
        results.append(BootstrapResult(created=True, detail=f"created tag {tag_id}"))
    return results


def _graphql(gms_url: str, query: str, variables: dict, token: str | None) -> dict:
    """POST a GraphQL document to GMS and return the decoded reply.

    Built on `http.client` rather than `urlopen` on purpose. `urlopen` takes a
    URL and picks a handler from its scheme, so a `DATAHUB_GMS_URL` of
    `file:///etc/passwd` reads a local path and hands back the bytes as though a
    server had replied. `HTTPConnection` and `HTTPSConnection` take a host and
    are chosen here, in code, which makes that substitution unrepresentable
    instead of merely guarded against. `Settings.from_env` also rejects the
    scheme up front, but this function is reachable with a bare string, so it
    does not rely on that.
    """
    parsed = urlsplit(gms_url)
    if parsed.scheme not in ("http", "https"):
        raise ValueError(
            f"GMS URL must be http or https, got {gms_url!r}. "
            "Setup talks to DataHub's GraphQL endpoint, not to a local path."
        )
    if not parsed.hostname:
        raise ValueError(f"GMS URL has no host, got {gms_url!r}.")

    body = json.dumps({"query": query, "variables": variables}).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    connect = HTTPSConnection if parsed.scheme == "https" else HTTPConnection
    connection = connect(parsed.hostname, parsed.port, timeout=30)
    path = f"{parsed.path.rstrip('/')}/api/graphql"
    try:
        connection.request("POST", path, body=body, headers=headers)
        response = connection.getresponse()
        raw = response.read().decode("utf-8")
        if response.status >= 400:
            # Previously an HTTPError from urlopen. Kept as a raise so a 401 on
            # a missing token still fails loudly rather than parsing as an empty
            # reply and reading like "nothing needed creating".
            raise RuntimeError(
                f"GMS returned {response.status} {response.reason} for {path}. {raw[:200]}"
            )
        return json.loads(raw)
    finally:
        connection.close()
