"""IdP group filtering applied at login, before the session store write.

Users with very large IdP group memberships (e.g. Entra ID accounts that belong
to hundreds or thousands of AD groups) cause two failures that scale with the
group count, on every authenticated request:

1. The ``X-Groups`` response header from ``/validate`` overflows nginx buffers,
   producing 500 errors on ``/scan`` and ``/ratings``.
2. ``map_groups_to_scopes`` issues one DocumentDB query per group.

This module filters the group list down to the groups the registry actually
grants access through, BEFORE it is persisted in the session. Because scopes are
only ever derived from groups that appear in a scope's ``group_mappings``,
dropping the other groups is lossless for authorization.

Group names are compared in their canonical form (surrounding whitespace and
slashes removed, see ``registry.utils.group_names``) on both sides, so a
Keycloak full-path claim such as ``/mcp-admins`` matches a mapping stored as
``mcp-admins``. The filtered list is returned in canonical form too, which is
what the session then carries (issue #1689).

Scope of this filter: it runs on the session-cookie login path only, at the
session write. A bearer-token request reads its groups live from the verified
token claim and does not pass through here, so the same user can carry a
different (unfiltered) group set on that path. That is by design: the filter
exists to bound what is persisted in the session, not to authorize; both paths
derive scopes from the same ``group_mappings`` afterwards.

Precedence (see .scratchpad/idp-group-filter-fix/lld.md):

- Design B (explicit allowlist) takes precedence: when ``ALLOWED_IDP_GROUPS`` is
  set, the user's groups are intersected with that exact-name list and the
  auto-derivation is skipped.
- Design C (default) runs only when the allowlist is unset: the user's groups
  are intersected with every group mapped to any scope.
- Fail-open applies ONLY when Design C cannot run at all (repository error or no
  scope mappings seeded): the full group list is stored and a warning is logged.
  A successful Design C run that yields an empty set for a particular user means
  that user is genuinely no-access; we store the empty set rather than chaining
  to the allowlist.
"""

import logging
import os
import re

from registry.utils.group_names import normalize_group_name, normalize_group_names

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Optional explicit allowlist (Design B). Empty means auto-derive (Design C).
ALLOWED_IDP_GROUPS_RAW: str = os.environ.get("ALLOWED_IDP_GROUPS", "")

# Parse comma-separated entries and validate each to prevent injection. Entra ID
# group Object IDs are GUIDs (hex + hyphens) and pass this pattern, as do plain
# group display names.
ALLOWED_IDP_GROUPS: list[str] = []
if ALLOWED_IDP_GROUPS_RAW:
    ALLOWED_IDP_GROUPS = [g.strip() for g in ALLOWED_IDP_GROUPS_RAW.split(",") if g.strip()]
    for _group in ALLOWED_IDP_GROUPS:
        if not re.match(r"^[a-zA-Z0-9\-_ ]+$", _group):
            raise ValueError(
                f"ALLOWED_IDP_GROUPS contains invalid characters in '{_group}'. "
                f"Only alphanumeric, hyphens, underscores, and spaces are allowed."
            )
    logger.info("ALLOWED_IDP_GROUPS allowlist active: %d entries", len(ALLOWED_IDP_GROUPS))


def _log_filter_result(
    mode: str,
    username_hash: str,
    groups: list[str],
    kept: list[str],
) -> None:
    """Log the outcome of a filter pass, naming any group that was dropped.

    The counts alone (``5 -> 1``) cannot tell an operator which group went
    missing, and once a group is gone from the session every later log line
    faithfully reports the reduced set. Naming the dropped groups here is the
    one place the information still exists (issue #1689). Group names are
    already logged at INFO elsewhere on this path (the ``/validate`` enrichment
    check), so this does not widen what the logs expose.

    Args:
        mode: Which filter ran, for the log line.
        username_hash: Hashed username for safe logging.
        groups: Canonical input groups.
        kept: Canonical groups that survived.
    """
    logger.info(
        "Group filter (%s) user=%s: %d -> %d",
        mode,
        username_hash,
        len(groups),
        len(kept),
    )
    kept_set = set(kept)
    dropped = [g for g in groups if g not in kept_set]
    if dropped:
        logger.warning(
            "Group filter (%s) user=%s dropped %d group(s) with no scope mapping: %s",
            mode,
            username_hash,
            len(dropped),
            dropped,
        )


async def _filter_by_scope_mappings(
    groups: list[str],
    username_hash: str,
) -> list[str]:
    """Design C: intersect the user's groups with all scope-mapped groups.

    Fails open (returns the original list) only when the mapped-group set cannot
    be determined (repository error) or is empty (no scope mappings seeded yet).

    Args:
        groups: Canonical group names from the claim.
        username_hash: Hashed username for safe logging.

    Returns:
        The canonical groups that map to at least one scope.
    """
    try:
        from registry.repositories.factory import get_scope_repository

        scope_repo = get_scope_repository()
        mapped = await scope_repo.get_all_mapped_group_names()
    except Exception as e:
        logger.warning(
            "Group filter could not derive scope-mapped groups (%s); storing "
            "full group list for user=%s (fail-open)",
            e,
            username_hash,
        )
        return groups

    if not mapped:
        logger.warning(
            "Group filter: no scope mappings found; storing full group list for "
            "user=%s (fail-open). Set ALLOWED_IDP_GROUPS to override.",
            username_hash,
        )
        return groups

    # Canonicalise the stored side too: a DocumentDB backend already returns
    # both forms, but an in-memory backend may return the raw names only.
    mapped_canonical = {normalize_group_name(m) for m in mapped}
    filtered = [g for g in groups if g in mapped_canonical]
    _log_filter_result("scope-derived", username_hash, groups, filtered)
    return filtered


async def filter_session_groups(
    groups: list[str],
    username_hash: str,
) -> list[str]:
    """Filter IdP groups to the scope-relevant subset for session storage.

    Args:
        groups: The user's full IdP group list (names or Object IDs), as
            emitted by the IdP. Full Keycloak paths (``/team``) are accepted.
        username_hash: Hashed username for safe logging.

    Returns:
        The filtered group list in canonical form. See the module docstring for
        precedence and fail-open semantics.
    """
    if not groups:
        return groups

    canonical_groups = normalize_group_names(groups)

    # Design B takes precedence when configured. The allowlist regex forbids
    # slashes, so its entries are already canonical.
    if ALLOWED_IDP_GROUPS:
        allowed = set(ALLOWED_IDP_GROUPS)
        filtered = [g for g in canonical_groups if g in allowed]
        _log_filter_result("allowlist", username_hash, canonical_groups, filtered)
        return filtered

    # Design C (default).
    return await _filter_by_scope_mappings(canonical_groups, username_hash)
