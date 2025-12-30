"""
Unit tests for registry auth user context builder.
"""

from unittest.mock import patch

import pytest

from registry.auth.user_context import (
    build_registry_user_context,
)


@pytest.mark.unit
@pytest.mark.auth
class TestBuildRegistryUserContext:
    def test_build_registry_user_context_merges_extra_fields(self) -> None:
        with (
            patch(
                "registry.auth.user_context.get_ui_permissions_for_user",
                return_value={"list_service": ["all"]},
            ),
            patch(
                "registry.auth.user_context.get_user_accessible_servers",
                return_value=["*"],
            ),
            patch(
                "registry.auth.user_context.get_accessible_services_for_user",
                return_value=["all"],
            ),
            patch(
                "registry.auth.user_context.get_accessible_agents_for_user",
                return_value=["all"],
            ),
            patch(
                "registry.auth.user_context.user_can_modify_servers",
                return_value=True,
            ),
            patch(
                "registry.auth.user_context.user_has_wildcard_access",
                return_value=True,
            ),
        ):
            context = build_registry_user_context(
                username="alice",
                groups=["mcp-registry-admin"],
                scopes=["mcp-servers-unrestricted/read"],
                auth_method="password",
                provider="local",
                extra={"user_id": "local|alice", "email": "alice@example.com"},
            )

        assert context["username"] == "alice"
        assert context["user_id"] == "local|alice"
        assert context["email"] == "alice@example.com"
        assert context["ui_permissions"] == {"list_service": ["all"]}
        assert context["can_modify_servers"] is True
        assert context["is_admin"] is True

