from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Optional


def _split_whitespace_tokens(
    value: str,
) -> list[str]:
    return [token.strip() for token in value.split() if token.strip()]


def _normalize_string_list(
    value: Any,
) -> list[str]:
    if value is None:
        return []

    if isinstance(value, str):
        return _split_whitespace_tokens(value)

    if isinstance(value, list):
        tokens: list[str] = []
        for item in value:
            if isinstance(item, str):
                tokens.extend(_split_whitespace_tokens(item))
        return tokens

    return []


def _dedupe_preserve_order(
    items: list[str],
) -> list[str]:
    seen: set[str] = set()
    deduped: list[str] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        deduped.append(item)
    return deduped


def normalize_token_audiences(
    claims: Mapping[str, Any],
) -> list[str]:
    """Normalize JWT `aud` claim to a list of strings.

    Supports both OIDC forms:
    - `aud` as a string
    - `aud` as a list of strings
    """

    return _dedupe_preserve_order(_normalize_string_list(claims.get("aud")))


def is_audience_allowed(
    *,
    token_audiences: list[str],
    allowed_audiences: list[str],
) -> bool:
    """Return True if any token audience matches allowed audiences."""

    if not token_audiences or not allowed_audiences:
        return False

    allowed_set = set(allowed_audiences)
    return any(audience in allowed_set for audience in token_audiences)


def extract_claim_values_by_precedence(
    *,
    claims: Mapping[str, Any],
    claim_precedence: list[str],
) -> tuple[Optional[str], list[str]]:
    """Extract a list of values from claims using precedence.

    Args:
        claims: JWT claims mapping.
        claim_precedence: Claim names in precedence order.

    Returns:
        A tuple of (selected_claim_name, values). If no values found, returns (None, []).
    """

    for claim_name in claim_precedence:
        raw_value = claims.get(claim_name)
        values = _dedupe_preserve_order(_normalize_string_list(raw_value))
        if values:
            return claim_name, values

    return None, []


def extract_scopes(
    *,
    claims: Mapping[str, Any],
    scope_claims: list[str],
) -> list[str]:
    """Extract scopes from claims using configured precedence.

    Common shapes supported:
    - `scp`: list of strings
    - `scope`: space-delimited string (or list)
    - `permissions`: list or string
    """

    _, values = extract_claim_values_by_precedence(
        claims=claims,
        claim_precedence=scope_claims,
    )
    return values


def extract_roles_for_audit(
    *,
    claims: Mapping[str, Any],
    role_claims: list[str],
) -> list[str]:
    """Extract roles/groups from claims using configured precedence.

    Returned values are intended for audit/metadata only and must not be used
    to grant authorization.
    """

    _, values = extract_claim_values_by_precedence(
        claims=claims,
        claim_precedence=role_claims,
    )
    return values

