"""Shared authorization-boundary constants.

This is a dependency-free leaf module so every layer can import it without
creating a cycle (the repository layer must not import the service layer, and
the auth layer must not import either). It holds the single source of truth for
the values that define "who is an admin" and "which scope writes are
privileged" -- values that MUST agree across the layers that enforce the same
security boundary:

- ``registry.auth.dependencies._user_is_admin`` derives per-request ``is_admin``
  from ``ADMIN_ACTION_PREFIXES`` + ``PRIVILEGED_GRANTS``.
- ``registry.services.scope_service`` and
  ``registry.repositories.documentdb.scope_repository`` use the same values
  (plus ``PRIVILEGED_SCOPE_NAMES``) as the defense-in-depth guard that blocks a
  non-admin from importing a group definition that would confer admin.

SECURITY BOUNDARY: changing any value here changes who is considered an admin
and which writes are gated. Previously these were copy-pasted across three
modules and kept aligned only by comments; centralizing them here means a new
mutating prefix (or privileged scope name) is picked up by every layer at once,
so the privileged-write guard cannot silently drift out of sync with the
admin-derivation rule.
"""

import re

# Mutating (management) UI-Scopes action prefixes. A user with any action whose
# name starts with one of these, granted for "all" resources, is an admin.
# Read-only prefixes (list_, get_, health_check_) are intentionally excluded.
#
# IMPORTANT: admin is conferred only by the literal "all" grant, NOT "*". A "*"
# grant on a mutating action grants access to every server WITHOUT admin (see
# issue #663), so it must not be treated as admin-conferring.
ADMIN_ACTION_PREFIXES: tuple[str, ...] = (
    "register_",
    "modify_",
    "toggle_",
    "delete_",
    "publish_",
    "create_",
)

# Grant value that makes a mutating action admin-conferring. See the note above
# on why "*" is deliberately excluded.
PRIVILEGED_GRANTS: frozenset[str] = frozenset({"all"})

# Per-type custom-entity mutation scopes (create_/modify_/delete_<type>_entity)
# match the mutating prefixes above, but they are NOT admin-conferring: they gate
# a single custom type's records, not registry management. Excluding them here
# lets an admin grant a non-admin group ``create_dataset_entity: ["all"]`` without
# silently promoting that user to full registry admin. This intentionally MODIFIES
# the prefix-only admin-derivation contract from PR #717/#663 -- ``list_`` scopes
# are read-only-prefixed and already excluded, so only the mutating forms need it.
#
# SECURITY BOUNDARY: this exclusion is the single source of truth for "a per-type
# entity scope is not admin". ``is_admin_conferring_action`` (below) is the ONLY
# supported way to apply the admin-derivation rule; both the per-request
# ``_user_is_admin`` (dependencies.py) and the privileged-write ``_grants_admin``
# (scope_repository.py) call it so they cannot drift apart.
_PER_TYPE_ENTITY_SCOPE_RE = re.compile(r"^(create|modify|delete)_.+_entity$")

# Skill-management scopes (publish_/modify_/delete_/toggle_skill) also match the
# mutating prefixes above, but they are NOT admin-conferring, for the same reason
# as the per-type entity scopes: they gate SKILL management, not registry
# management. Excluding them lets an admin grant a non-admin "skill-manager" group
# the full skill scope set (publish/modify/delete/toggle_skill: ["all"]) -- exactly
# what the admin seeds carry -- without silently promoting that group to full
# registry admin. ``list_skills`` is read-only-prefixed and already excluded, so
# only the four mutating forms are enumerated here. All four are excluded (not just
# modify/delete/toggle) so the full grantable set stays non-conferring; leaving
# ``publish_skill`` in would let a skill-manager holding the seeded set re-acquire
# admin and defeat the exclusion.
_SKILL_MANAGEMENT_SCOPES: frozenset[str] = frozenset(
    {"publish_skill", "modify_skill", "delete_skill", "toggle_skill"}
)


def is_admin_conferring_action(action: str) -> bool:
    """Return True if holding ``action`` for "all" resources confers admin.

    An action confers admin when it starts with a mutating management prefix
    (``ADMIN_ACTION_PREFIXES``) AND is not an excluded scope: a per-type
    custom-entity scope (``create_/modify_/delete_<type>_entity``) or a
    skill-management scope (``publish_/modify_/delete_/toggle_skill``). Read-only
    prefixed actions (``list_``/``get_``/``health_check_``) never match a mutating
    prefix and so are non-conferring by construction.

    This is the sole gate for the admin-derivation rule; both the per-request
    admin check and the privileged-write guard defer to it so the exclusion
    cannot be honored by one consumer and ignored by the other.

    Args:
        action: A UI-Scopes action name (e.g. ``register_service``,
            ``create_dataset_entity``, ``modify_skill``).

    Returns:
        True if the action is admin-conferring (subject to a "all" grant),
        False otherwise.
    """
    if not action.startswith(ADMIN_ACTION_PREFIXES):
        return False
    if _PER_TYPE_ENTITY_SCOPE_RE.match(action):
        return False
    if action in _SKILL_MANAGEMENT_SCOPES:
        return False
    return True


# Scope/group names that confer administrative access by membership. Naming a
# scope one of these, or mapping a group to one of these, elevates whoever holds
# it -- the original /api/servers/groups/import privesc vector.
PRIVILEGED_SCOPE_NAMES: frozenset[str] = frozenset(
    {
        "mcp-registry-admin",
        "mcp-registry-operator",
        "registry-admins",
        "mcp-servers-unrestricted/execute",
        "mcp-servers-unrestricted/read",
    }
)

# Group-name markers that confer registry-administrator privileges by membership.
# This is the single source of truth for "which group names mean admin" and MUST
# agree across every layer that makes an admin decision by group membership:
#
# - ``auth_server.server`` uses it for the A2A admin bypass and for rejecting a
#   token-mint request that claims a privileged group the session never held.
# - ``auth_server.mongodb_groups_enrichment`` uses it to audit when DB group
#   enrichment grants an admin group.
#
# These were previously two separate hardcoded frozensets kept aligned only by
# comments; centralizing here means an admin-group rename is picked up by every
# gate at once. This is a strict subset of ``PRIVILEGED_SCOPE_NAMES`` (which also
# covers scope-shaped names); keep the two consistent for the shared entries.
ADMIN_GROUP_MARKERS: frozenset[str] = frozenset({"mcp-registry-admin", "registry-admins"})
