"""Canonical form for IdP group identifiers.

Group names arrive from an IdP claim and are compared against the
``group_mappings`` arrays in the scope documents. Server names in the same
comparisons go through ``_normalize_server_name`` (strip surrounding slashes),
but group names were compared verbatim everywhere. Keycloak's Group Membership
mapper emits full group paths (``/mcp-admins``) when *Full group path* is on,
so such a group matched nothing: it resolved to no scope and was dropped from
the session at login, with no error anywhere (issue #1689).

The normalisation is deliberately minimal and symmetric: surrounding whitespace
and surrounding slashes. It is applied to both sides of every comparison, so a
mapping written as ``/team`` still matches a claim of ``team`` and vice versa.
Case is preserved: Keycloak group names are case-sensitive and Entra Object IDs
are opaque, so folding case could merge two distinct groups.

A nested Keycloak path (``/parent/child``) normalises to ``parent/child``. It is
not reduced to ``child``, because two nested groups can share a leaf name and
collapsing them would grant one group's scopes to the other.
"""

from collections.abc import Iterable


def normalize_group_name(name: str) -> str:
    """Return the canonical form of one group identifier.

    Args:
        name: Raw group name or Object ID from an IdP claim or a scope document.

    Returns:
        The name with surrounding whitespace and surrounding slashes removed.
        An empty or non-string input returns an empty string.
    """
    if not isinstance(name, str):
        return ""
    return name.strip().strip("/").strip()


def normalize_group_names(groups: list[str]) -> list[str]:
    """Normalise a group list, dropping empties and duplicates.

    Args:
        groups: Raw group identifiers, in claim order.

    Returns:
        Canonical names in first-seen order. Duplicates that only differ by
        surrounding slashes or whitespace collapse to one entry.
    """
    seen: set[str] = set()
    result: list[str] = []
    for group in groups or []:
        canonical = normalize_group_name(group)
        if canonical and canonical not in seen:
            seen.add(canonical)
            result.append(canonical)
    return result


def _slash_forms(canonical: str) -> set[str]:
    """Return the canonical name with each combination of single outer slashes."""
    return {canonical, f"/{canonical}", f"{canonical}/", f"/{canonical}/"}


def group_name_variants(groups: list[str]) -> list[str]:
    """Return every form a stored mapping might use for the given groups.

    Scope documents written before normalisation existed may hold the raw
    claim, the canonical name, or a slash-decorated form (``/team``,
    ``team/``, ``/team/``), and a data migration is not worth the risk for a
    comparison this cheap. Querying with all of them matches any of them
    without touching stored data, so a bare ``team`` claim finds a mapping
    stored as ``/team`` or ``team/`` and a ``/team`` claim finds one stored as
    ``team``. That keeps scope lookup in agreement with the login filter,
    which canonicalises the stored names.

    A stored value with other decoration (doubled slashes, padding inside the
    slashes) is not matched. ``is_lookup_reachable`` tells the login filter
    which stored values those are, so it does not keep a group the lookup
    cannot resolve.

    Args:
        groups: Raw group identifiers.

    Returns:
        Sorted, de-duplicated list of the raw, canonical and slash-decorated
        forms, with empties removed. Sorted so a query built from it is
        deterministic.
    """
    variants: set[str] = set()
    for group in groups or []:
        canonical = normalize_group_name(group)
        if not canonical:
            # A value that is empty once canonicalised ("", "/", whitespace)
            # names no group in any form, so none is queried.
            continue
        variants.update(_slash_forms(canonical))
        variants.add(group.strip())
    return sorted(variants)


def is_lookup_reachable(stored: str) -> bool:
    """Return True when a lookup by the canonical name finds this stored value.

    The scope lookup queries ``group_name_variants`` of the canonical session
    name with an exact ``$in`` match, so it finds a stored value only if that
    value is one of the slash forms of its own canonical name. The login
    filter uses this to keep exactly the groups the lookup can resolve.

    Args:
        stored: A group identifier as stored in a scope document.

    Returns:
        True if ``group_name_variants([normalize_group_name(stored)])``
        contains ``stored`` unchanged.
    """
    canonical = normalize_group_name(stored)
    return bool(canonical) and stored in _slash_forms(canonical)


def groups_intersect(
    user_groups: Iterable[str] | None,
    allowed_groups: Iterable[str] | None,
) -> bool:
    """Return True when the two group lists share a group in canonical form.

    Group-restricted visibility compares the caller's groups with a record's
    ``allowed_groups``. The login filter stores canonical names in the session
    (``team``), while an operator may have written ``/team`` into
    ``allowed_groups`` (or the other way round), so an exact set intersection
    would hide the record. Both sides are canonicalised first. Names that are
    empty once canonicalised never match, so an empty or blank entry on either
    side grants nothing.

    Args:
        user_groups: The caller's groups, from the session or token claim.
        allowed_groups: The groups a record is restricted to.

    Returns:
        True if at least one group appears on both sides.
    """
    allowed = set(normalize_group_names(list(allowed_groups or [])))
    if not allowed:
        return False
    return any(group in allowed for group in normalize_group_names(list(user_groups or [])))
