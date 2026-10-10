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


def group_name_variants(groups: list[str]) -> list[str]:
    """Return every form a stored mapping might use for the given groups.

    Scope documents written before normalisation existed may hold either the
    raw or the canonical form, and a data migration is not worth the risk for
    a comparison this cheap. Querying with both forms matches either without
    touching stored data.

    Args:
        groups: Raw group identifiers.

    Returns:
        Sorted, de-duplicated list of the raw and canonical forms, with
        empties removed. Sorted so a query built from it is deterministic.
    """
    variants: set[str] = set()
    for group in groups or []:
        canonical = normalize_group_name(group)
        if not canonical:
            # A value that is empty once canonicalised ("", "/", whitespace)
            # names no group in either form, so neither form is queried.
            continue
        variants.add(canonical)
        variants.add(group.strip())
    return sorted(variants)
