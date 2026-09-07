"""The admin scope seed must authorize HTTP verbs on gateway-proxied entities.

The generic proxy authorizes a request by HTTP verb, and the resolver splits the
value spaces deliberately: for an HTTP verb the legacy ``all`` / ``*`` wildcard
does NOT grant. That guard exists so flipping an entity to ``is_proxied`` cannot
silently turn an existing MCP wildcard into DELETE/PUT authority
(``_HTTP_VERB_WILDCARD`` in ``auth_server/server.py``).

The consequence for the shipped seed is that an administrator holding
``methods: ["all"]`` is denied every proxied route. It surfaces as a 403 that
reads like a registration or routing failure rather than a missing grant, and it
cannot be repaired through the management API, which refuses wildcard
``server_access`` entries -- so ``scripts/registry-admins.json`` is the only
path, and a fresh install depends on it.

This drives the real resolver with the on-disk seed rather than asserting the
file's shape, so it keeps holding if the wildcard token or the split is reworked.
"""

import json
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest

pytestmark = pytest.mark.unit

# A proxied custom record's authz key is "<entity_type>/<path minus leading slash>",
# which for a UUID-keyed custom record reads doubled. Using the real shape keeps
# the test honest about what the gateway actually sends to /validate.
PROXIED_AUTHZ_KEY = "rest-endpoint/rest-endpoint/1a546ca6-4336-4164-8fd3-b5d7e6bccb56"
ADMIN_SEED = "scripts/registry-admins.json"
ADMIN_SCOPE = "registry-admins"


@pytest.fixture(scope="module")
def admin_server_access() -> list[dict]:
    """The shipped admin seed's server_access rules, read from the repository."""
    # tests/auth_server/unit/<this file> -> repository root is three levels up.
    repo_root = Path(__file__).resolve().parents[3]
    return json.loads((repo_root / ADMIN_SEED).read_text())["server_access"]


@pytest.mark.asyncio
@pytest.mark.parametrize("verb", ["GET", "POST", "PUT", "DELETE", "PATCH"])
async def test_admin_seed_authorizes_http_verbs_on_a_proxied_entity(
    admin_server_access: list[dict], verb: str
) -> None:
    from auth_server.server import validate_server_tool_access

    repo = AsyncMock()
    repo.get_server_scopes = AsyncMock(
        side_effect=lambda scope: admin_server_access if scope == ADMIN_SCOPE else []
    )

    with patch("auth_server.server.get_scope_repository", return_value=repo):
        allowed = await validate_server_tool_access(
            PROXIED_AUTHZ_KEY, verb, "", [ADMIN_SCOPE], is_http_verb=True
        )

    assert allowed, (
        f"{ADMIN_SEED} does not authorize {verb} on a proxied entity, so administrators "
        f"get a 403 on every gateway-proxied route on a fresh install. Add "
        f'{{"server": "*", "methods": ["http:*"], "tools": []}} to server_access.'
    )


@pytest.mark.asyncio
async def test_mcp_wildcard_alone_does_not_authorize_an_http_verb() -> None:
    """Pins the escalation guard the seed rule works around.

    If ``all`` ever starts granting HTTP verbs, the seed rule becomes redundant --
    and every non-admin group holding an MCP wildcard silently gains DELETE on
    each entity that is flipped to proxied. That is the failure this asserts
    against, so the guard cannot be relaxed without this test objecting.
    """
    from auth_server.server import validate_server_tool_access

    repo = AsyncMock()
    repo.get_server_scopes = AsyncMock(
        return_value=[{"server": "*", "methods": ["all"], "tools": ["all"]}]
    )

    with patch("auth_server.server.get_scope_repository", return_value=repo):
        allowed = await validate_server_tool_access(
            PROXIED_AUTHZ_KEY, "DELETE", "", ["some-group"], is_http_verb=True
        )

    assert not allowed, (
        "methods:['all'] authorized an HTTP verb. The value-space split is the guard "
        "that stops an MCP wildcard from becoming DELETE/PUT authority when an entity "
        "is flipped to is_proxied."
    )
