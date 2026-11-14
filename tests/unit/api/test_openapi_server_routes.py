"""Tests for /api/openapi-servers routes (issue #2022)."""

import json
from unittest.mock import AsyncMock

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.api import openapi_server_routes as routes
from registry.auth.csrf import verify_csrf_token_flexible
from registry.auth.dependencies import nginx_proxied_auth
from registry.schemas.openapi_server_models import (
    OpenApiServer,
    OpenApiTarget,
    StoredTargetAuth,
)
from registry.services import openapi_spec_checks
from registry.services.openapi_errors import unprocessable

ADMIN: dict = {
    "username": "admin",
    "is_admin": True,
    "groups": [],
    "scopes": [],
    "accessible_servers": ["all"],
}
USER: dict = {
    "username": "bob",
    "is_admin": False,
    "groups": ["eng"],
    "scopes": [],
    "accessible_servers": ["all"],
}
ENCRYPTED: str = "gAAAAABfakeEncryptedCredentialValue=="


def _server(
    visibility: str = "public",
    groups: list[str] | None = None,
) -> OpenApiServer:
    now = openapi_spec_checks.now_utc()
    target = OpenApiTarget(
        name="petstore",
        spec_url="https://internal.example.com/spec.json",
        spec_sha256="a" * 64,
        base_url="https://internal.example.com/api",
        tool_names=["petstore_getPet"],
        created_by="admin",
        created_at=now,
        updated_at=now,
        auth=StoredTargetAuth(
            scheme="api_key_header", header_name="api_key", credential_encrypted=ENCRYPTED
        ),
    )
    return OpenApiServer(
        slug="weather-apis",
        name="Weather APIs",
        path="/weather-apis",
        visibility=visibility,
        allowed_groups=groups or [],
        generation=2,
        targets=[target],
        created_by="admin",
        created_at=now,
        updated_at=now,
    )


@pytest.fixture
def make_client(monkeypatch):
    monkeypatch.setattr(
        routes.openapi_mcp_client, "status", AsyncMock(return_value={"servers": {}})
    )
    monkeypatch.setattr(routes, "user_can_access_server", AsyncMock(return_value=True))

    def _make(
        user: dict,
    ) -> TestClient:
        app = FastAPI()
        app.include_router(routes.router, prefix="/api")
        app.dependency_overrides[nginx_proxied_auth] = lambda: user
        app.dependency_overrides[verify_csrf_token_flexible] = lambda: None
        return TestClient(app)

    return _make


class TestWriteAuthorization:
    @pytest.mark.parametrize(
        ("method", "path", "body"),
        [
            ("POST", "/api/openapi-servers", {"name": "x"}),
            ("PATCH", "/api/openapi-servers/weather-apis", {"description": "d"}),
            ("DELETE", "/api/openapi-servers/weather-apis", None),
            (
                "POST",
                "/api/openapi-servers/weather-apis/targets",
                {"name": "pet", "spec_url": "https://a.example/s", "base_url": "https://a.example"},
            ),
            ("PATCH", "/api/openapi-servers/weather-apis/targets/petstore", {"description": "d"}),
            ("DELETE", "/api/openapi-servers/weather-apis/targets/petstore", None),
        ],
    )
    def test_non_admin_writes_are_403(self, make_client, method, path, body):
        response = make_client(USER).request(method, path, json=body)

        assert response.status_code == 403

    def test_execute_scope_is_not_admin(self, make_client):
        user = {**USER, "can_modify_servers": True, "scopes": ["mcp-servers-unrestricted/execute"]}

        response = make_client(user).post("/api/openapi-servers", json={"name": "x"})

        assert response.status_code == 403


class TestReads:
    def test_private_server_is_404_for_non_admin(self, make_client, monkeypatch):
        monkeypatch.setattr(
            routes.openapi_server_service,
            "get_server_or_404",
            AsyncMock(return_value=_server("private")),
        )

        response = make_client(USER).get("/api/openapi-servers/weather-apis")

        assert response.status_code == 404

    def test_group_restricted_needs_a_group(self, make_client, monkeypatch):
        monkeypatch.setattr(
            routes.openapi_server_service,
            "get_server_or_404",
            AsyncMock(return_value=_server("group-restricted", ["ops"])),
        )

        assert make_client(USER).get("/api/openapi-servers/weather-apis").status_code == 404

    def test_non_admin_gets_no_backend_urls(self, make_client, monkeypatch):
        monkeypatch.setattr(
            routes.openapi_server_service, "get_server_or_404", AsyncMock(return_value=_server())
        )

        body = make_client(USER).get("/api/openapi-servers/weather-apis").json()

        target = body["targets"][0]
        assert (target["spec_url"], target["base_url"]) == (None, None)
        assert target["auth"]["credential_set"] is True

    def test_admin_gets_backend_urls(self, make_client, monkeypatch):
        monkeypatch.setattr(
            routes.openapi_server_service,
            "get_server_or_404",
            AsyncMock(return_value=_server("private")),
        )

        body = make_client(ADMIN).get("/api/openapi-servers/weather-apis").json()

        assert body["targets"][0]["base_url"] == "https://internal.example.com/api"

    def test_list_filters_hidden_servers(self, make_client, monkeypatch):
        repo = type("Repo", (), {"list_all": AsyncMock(return_value=[_server("private")])})()
        monkeypatch.setattr(
            "registry.repositories.factory.get_openapi_server_repository", lambda: repo
        )

        assert make_client(USER).get("/api/openapi-servers").json() == {"servers": []}
        assert len(make_client(ADMIN).get("/api/openapi-servers").json()["servers"]) == 1

    def test_credential_absent_from_every_response(self, make_client, monkeypatch):
        monkeypatch.setattr(
            routes.openapi_server_service, "get_server_or_404", AsyncMock(return_value=_server())
        )
        repo = type("Repo", (), {"list_all": AsyncMock(return_value=[_server()])})()
        monkeypatch.setattr(
            "registry.repositories.factory.get_openapi_server_repository", lambda: repo
        )
        client = make_client(ADMIN)

        bodies = [
            client.get("/api/openapi-servers").text,
            client.get("/api/openapi-servers/weather-apis").text,
            client.get("/api/openapi-servers/weather-apis/targets/petstore").text,
        ]

        for text in bodies:
            assert ENCRYPTED not in text and "credential_encrypted" not in text


class TestErrorShape:
    def test_service_error_becomes_top_level_body(self, make_client, monkeypatch):
        error = unprocessable(
            "too_many_tools", "too many", {"count": 3, "limit": 2, "tags": {"pet": 3}}
        )
        monkeypatch.setattr(
            routes.openapi_server_service, "add_target", AsyncMock(side_effect=error)
        )
        body = {
            "name": "petstore",
            "spec_url": "https://a.example/s.json",
            "base_url": "https://a.example",
        }

        response = make_client(ADMIN).post("/api/openapi-servers/weather-apis/targets", json=body)

        assert response.status_code == 422
        assert json.loads(response.text) == {
            "error": "too_many_tools",
            "detail": "too many",
            "count": 3,
            "limit": 2,
            "tags": {"pet": 3},
        }

    def test_bad_auth_fields_are_validation_error(self, make_client):
        body = {
            "name": "petstore",
            "spec_url": "https://a.example/s.json",
            "base_url": "https://a.example",
            "auth": {"scheme": "api_key_header", "credential": "k"},
        }

        response = make_client(ADMIN).post("/api/openapi-servers/weather-apis/targets", json=body)

        assert response.status_code == 422
        assert "header_name" in response.text

    def test_bad_slug_in_path_is_rejected(self, make_client):
        assert make_client(ADMIN).get("/api/openapi-servers/Bad_Slug").status_code == 422

    def test_create_returns_201_with_mcp_url(self, make_client, monkeypatch):
        monkeypatch.setattr(
            routes.openapi_server_service,
            "create_server",
            AsyncMock(return_value=(_server(), "applied")),
        )

        response = make_client(ADMIN).post("/api/openapi-servers", json={"name": "Weather APIs"})

        assert response.status_code == 201
        assert response.json()["mcp_url"].endswith("/weather-apis/mcp")
        assert response.json()["sync_status"] == "applied"
