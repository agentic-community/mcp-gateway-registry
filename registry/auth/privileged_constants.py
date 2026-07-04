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
