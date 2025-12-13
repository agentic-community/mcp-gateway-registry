"""
Unit tests for Stage 3.2 OIDC claim normalization helpers.
"""

import pytest

from auth_server.enforceai.oidc.claims import (
    extract_roles_for_audit,
    extract_scopes,
    is_audience_allowed,
    normalize_token_audiences,
)


@pytest.mark.unit
class TestOIDCAudienceNormalization:
    def test_aud_as_string(self):
        token_audiences = normalize_token_audiences({"aud": "mcp-registry"})
        assert token_audiences == ["mcp-registry"]
        assert is_audience_allowed(
            token_audiences=token_audiences,
            allowed_audiences=["mcp-registry"],
        )

    def test_aud_as_list(self):
        token_audiences = normalize_token_audiences(
            {
                "aud": [
                    "mcp-registry",
                    "mcp-gateway",
                ]
            }
        )
        assert token_audiences == ["mcp-registry", "mcp-gateway"]
        assert is_audience_allowed(
            token_audiences=token_audiences,
            allowed_audiences=["mcp-gateway"],
        )

    def test_aud_invalid_types_are_ignored(self):
        token_audiences = normalize_token_audiences({"aud": 123})
        assert token_audiences == []
        assert not is_audience_allowed(
            token_audiences=token_audiences,
            allowed_audiences=["mcp-registry"],
        )


@pytest.mark.unit
class TestOIDCScopeExtraction:
    def test_scopes_from_scp_list(self):
        scopes = extract_scopes(
            claims={"scp": ["read", "write"]},
            scope_claims=["scp", "scope", "permissions"],
        )
        assert scopes == ["read", "write"]

    def test_scopes_from_scope_string_space_delimited(self):
        scopes = extract_scopes(
            claims={"scope": "read write"},
            scope_claims=["scp", "scope", "permissions"],
        )
        assert scopes == ["read", "write"]

    def test_scopes_from_permissions_list(self):
        scopes = extract_scopes(
            claims={"permissions": ["read", "write"]},
            scope_claims=["scp", "scope", "permissions"],
        )
        assert scopes == ["read", "write"]

    def test_precedence_picks_first_non_empty_claim(self):
        scopes = extract_scopes(
            claims={
                "scp": [],
                "scope": "read write",
                "permissions": ["admin"],
            },
            scope_claims=["scp", "scope", "permissions"],
        )
        assert scopes == ["read", "write"]

    def test_whitespace_and_duplicates_are_dropped(self):
        scopes = extract_scopes(
            claims={
                "scope": "read   write  read",
            },
            scope_claims=["scope"],
        )
        assert scopes == ["read", "write"]


@pytest.mark.unit
class TestOIDCRoleExtraction:
    def test_roles_from_roles_list(self):
        roles = extract_roles_for_audit(
            claims={"roles": ["admin", "viewer"]},
            role_claims=["roles", "groups", "permissions"],
        )
        assert roles == ["admin", "viewer"]

    def test_roles_from_groups_list(self):
        roles = extract_roles_for_audit(
            claims={"groups": ["group-a", "group-b"]},
            role_claims=["roles", "groups", "permissions"],
        )
        assert roles == ["group-a", "group-b"]

    def test_roles_from_permissions_string(self):
        roles = extract_roles_for_audit(
            claims={"permissions": "role-a role-b"},
            role_claims=["roles", "groups", "permissions"],
        )
        assert roles == ["role-a", "role-b"]

    def test_roles_precedence_picks_first_non_empty_claim(self):
        roles = extract_roles_for_audit(
            claims={
                "roles": ["admin"],
                "groups": ["group-a"],
                "permissions": ["role-a"],
            },
            role_claims=["roles", "groups", "permissions"],
        )
        assert roles == ["admin"]

