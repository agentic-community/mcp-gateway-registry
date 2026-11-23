"""What a caller sees of an OpenAPI MCP server's tools (issue #2022, security review #8)."""

from datetime import (
    UTC,
    datetime,
)

import pytest

from registry.schemas.openapi_server_models import (
    OpenApiServer,
    OpenApiTarget,
    StoredTargetAuth,
)
from registry.services import openapi_server_views as views

NOW = datetime(2026, 11, 20, tzinfo=UTC)


@pytest.fixture
def server() -> OpenApiServer:
    target = OpenApiTarget(
        name="weather",
        spec_sha256="0" * 64,
        base_url="https://api.example.com",
        auth=StoredTargetAuth(),
        tool_names=["weather_forecast", "weather_admin_purge"],
        tool_name_overrides={"purgeAll": "weather_admin_purge"},
        description_overrides={"weather_admin_purge": "Deletes every station"},
        include_operations=["getForecast", "purgeAll"],
        created_by="admin",
        created_at=NOW,
        updated_at=NOW,
    )
    return OpenApiServer(
        slug="weather",
        name="Weather",
        path="/weather",
        targets=[target],
        created_by="admin",
        created_at=NOW,
        updated_at=NOW,
    )


def _user(
    tools: set[str],
) -> dict:
    return {
        "is_admin": False,
        "accessible_servers": ["weather"],
        "accessible_tools": {"weather": tools},
    }


class TestToolProjection:
    def test_a_caller_granted_one_tool_sees_only_that_tool(self, server):
        response = views.server_response(server, _user({"weather_forecast"}))

        assert response.targets[0].tool_names == ["weather_forecast"]

    def test_override_maps_and_filters_are_hidden_from_non_admins(self, server):
        target = views.target_response(server, server.targets[0], _user({"weather_forecast"}))

        assert (
            target.tool_name_overrides,
            target.description_overrides,
            target.include_operations,
        ) == ({}, {}, [])

    def test_an_empty_grant_sees_no_tools(self, server):
        target = views.target_response(server, server.targets[0], _user(set()))

        assert target.tool_names == []

    def test_an_admin_sees_everything(self, server):
        target = views.target_response(server, server.targets[0], {"is_admin": True})

        assert target.description_overrides == {"weather_admin_purge": "Deletes every station"}
        assert target.tool_names == ["weather_forecast", "weather_admin_purge"]

    def test_projection_leaves_the_stored_target_alone(self, server):
        views.target_response(server, server.targets[0], _user(set()))

        assert server.targets[0].tool_names == ["weather_forecast", "weather_admin_purge"]

    def test_a_non_admin_gets_a_generic_load_error(self, server):
        runtime = {"targets": {"weather": {"status": "error", "error": "operation purgeAll: bad"}}}

        target = views.target_response(server, server.targets[0], _user(set()), runtime)

        assert "purgeAll" not in (target.error or "")
