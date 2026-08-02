"""
Regression tests for the scope seed files' ``group_mappings`` arrays.

``group_mappings`` is a list of IdP *group* identifiers (Keycloak group names,
Entra ID Object IDs, PingFederate groups) that grant the scope named by ``_id``.
``map_cognito_groups_to_scopes`` inverts every scope's array into a
group -> scopes lookup, so an entry that names a *scope* instead of a group can
never match a token claim, and a scope with an empty array is unreachable for
every user.

The same arrays back the login-time session-group filter
(``auth_server/group_filter.py``), which intersects a user's IdP groups with the
union of these arrays. A group missing here is silently dropped from the
session, so a wrong entry degrades authorization in two places at once.

These are static assertions over the seed JSON, so they hold without a live
Keycloak or DocumentDB.
"""

import json
from pathlib import Path

import pytest


@pytest.fixture(scope="module")
def repo_root() -> Path:
    """Get repository root directory."""
    return Path(__file__).parent.parent.parent


# Seed files mounted into the mongodb-init container by docker-compose.yml and
# loaded verbatim by scripts/init-mongodb-ce.py.
SCOPE_SEED_FILES = [
    "scripts/registry-admins.json",
    "scripts/mcp-registry-admin.json",
    "scripts/mcp-servers-unrestricted-read.json",
    "scripts/mcp-servers-unrestricted-execute.json",
    "scripts/federation-service.json",
]

# The privileged registry scopes. Each must be reachable by at least one group,
# otherwise no operator can administer a fresh install and the documented
# quickstart cannot register a server.
ADMIN_SCOPES = [
    "registry-admins",
    "mcp-registry-admin",
]

# Scopes that gate MCP server access for administrators. The pre-DocumentDB
# auth_server/scopes.yml granted both to registry-admins and mcp-registry-admin.
SERVER_ACCESS_SCOPES = [
    "mcp-servers-unrestricted/read",
    "mcp-servers-unrestricted/execute",
]


def _load(repo_root: Path, rel_path: str) -> dict:
    return json.loads((repo_root / rel_path).read_text())


@pytest.mark.parametrize("seed_path", SCOPE_SEED_FILES)
def test_group_mappings_only_name_real_idp_groups(repo_root: Path, seed_path: str) -> None:
    """Every entry must be an identifier an IdP can actually emit.

    Most scopes here are deliberately named after the group that holds them
    (``registry-admins`` is both a Keycloak group and a scope ``_id``), so a name
    appearing in both roles is expected and correct. What is never valid is a
    *path-qualified* scope id such as 'mcp-servers-unrestricted/read': Keycloak
    groups cannot contain '/', so such an entry is inert -- the runtime inverts
    these arrays into a group -> scopes map and nothing ever matches it.
    """
    doc = _load(repo_root, seed_path)

    offenders = [g for g in doc.get("group_mappings", []) if "/" in g]

    assert not offenders, (
        f"{seed_path} lists path-qualified scope name(s) {offenders} in "
        f"group_mappings, which expects IdP group identifiers. These can never "
        f"match a token claim. To grant another scope to this group, add this "
        f"group to that scope's file instead."
    )


@pytest.mark.parametrize("scope_id", ADMIN_SCOPES)
def test_admin_scopes_are_reachable(repo_root: Path, scope_id: str) -> None:
    """A privileged scope with no groups cannot be held by anyone."""
    doc = next(
        _load(repo_root, p) for p in SCOPE_SEED_FILES if _load(repo_root, p)["_id"] == scope_id
    )

    assert doc.get("group_mappings"), (
        f"Scope '{scope_id}' has an empty group_mappings array, so no user can "
        f"ever hold it. A fresh install would have no administrator."
    )


@pytest.mark.parametrize("scope_id", SERVER_ACCESS_SCOPES)
def test_admin_groups_grant_server_access(repo_root: Path, scope_id: str) -> None:
    """Admin groups must reach the unrestricted server-access scopes.

    Both admin groups held these in the pre-DocumentDB scopes.yml. Without them
    an administrator authenticates but resolves to zero server access, which
    surfaces as a 403 on /api/servers/register.
    """
    doc = next(
        _load(repo_root, p) for p in SCOPE_SEED_FILES if _load(repo_root, p)["_id"] == scope_id
    )
    mapped = doc.get("group_mappings", [])

    missing = [g for g in ADMIN_SCOPES if g not in mapped]

    assert not missing, (
        f"Scope '{scope_id}' is not granted to admin group(s) {missing}. "
        f"group_mappings={mapped}. Administrators will resolve to zero scopes "
        f"and be denied server registration."
    )


def test_every_seed_group_mapping_is_a_plain_group_name(repo_root: Path) -> None:
    """No entry may contain '/', the scope-name separator.

    Guards the whole seed set against reintroducing the scope-vs-group confusion
    in a file this suite does not enumerate individually.
    """
    violations: list[str] = []
    for path in SCOPE_SEED_FILES:
        doc = _load(repo_root, path)
        for group in doc.get("group_mappings", []):
            if "/" in group:
                violations.append(f"{path}: {group!r}")

    assert not violations, (
        "group_mappings entries must be IdP group identifiers, but these "
        f"contain '/', the scope-name separator: {violations}"
    )
