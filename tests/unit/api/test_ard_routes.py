"""Unit tests for the ARD catalog route and public record endpoints."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from fastapi import FastAPI
from fastapi.testclient import TestClient

from registry.api import public_record_routes, wellknown_routes
from registry.schemas.ard_models import (
    AICatalogManifest,
    ArdCatalogEntry,
    ArdHost,
)


def _client():
    app = FastAPI()
    app.include_router(wellknown_routes.router, prefix="/.well-known")
    app.include_router(public_record_routes.router, prefix="/api")
    return TestClient(app)


def _manifest():
    return AICatalogManifest(
        host=ArdHost(display_name="Test Registry"),
        entries=[
            ArdCatalogEntry(
                identifier="urn:air:registry.example.com:server:github",
                display_name="GitHub",
                type="application/mcp-server-card+json",
                url="https://registry.example.com/api/public/servers/github/server.json",
            )
        ],
    )


class TestCatalogRoute:
    """Tests for GET /.well-known/ai-catalog.json."""

    def test_returns_manifest_when_enabled(self):
        with (
            patch.object(wellknown_routes.settings, "enable_wellknown_discovery", True),
            patch.object(wellknown_routes.settings, "ard_catalog_enabled", True),
            patch.object(wellknown_routes.settings, "wellknown_cache_ttl", 300),
            patch.object(
                wellknown_routes.ard_service,
                "build_catalog",
                AsyncMock(return_value=_manifest()),
            ),
        ):
            resp = _client().get("/.well-known/ai-catalog.json", headers={"X-Real-IP": "192.0.2.1"})
        assert resp.status_code == 200
        body = resp.json()
        assert body["specVersion"] == "1.0"
        assert body["host"]["displayName"] == "Test Registry"
        assert len(body["entries"]) == 1
        # oneOf: url present, data absent.
        assert "url" in body["entries"][0]
        assert "data" not in body["entries"][0]
        assert resp.headers["cache-control"] == "public, max-age=300"

    def test_404_when_ard_disabled(self):
        with (
            patch.object(wellknown_routes.settings, "enable_wellknown_discovery", True),
            patch.object(wellknown_routes.settings, "ard_catalog_enabled", False),
        ):
            resp = _client().get("/.well-known/ai-catalog.json")
        assert resp.status_code == 404

    def test_404_when_wellknown_disabled(self):
        with (
            patch.object(wellknown_routes.settings, "enable_wellknown_discovery", False),
            patch.object(wellknown_routes.settings, "ard_catalog_enabled", True),
        ):
            resp = _client().get("/.well-known/ai-catalog.json")
        assert resp.status_code == 404


class TestPublicServer:
    """Tests for GET /api/public/servers/{leaf}/server.json.

    The endpoint scans the public+enabled set and matches by sanitized leaf, so
    find_with_filter already excludes private/disabled records (a non-match -> 404).
    """

    def test_public_enabled_returns_200(self):
        records = {
            "/github/": {
                "server_name": "GitHub",
                "is_enabled": True,
                "visibility": "public",
                "proxy_pass_url": "http://backend:9000",
            }
        }
        repo = SimpleNamespace(find_with_filter=AsyncMock(return_value=records))
        with (
            patch.object(public_record_routes, "get_server_repository", return_value=repo),
            patch.object(
                public_record_routes, "to_canonical", return_value=({"name": "github"}, False)
            ),
            patch.object(public_record_routes, "redact_backend_urls", side_effect=lambda d: d),
        ):
            resp = _client().get("/api/public/servers/github/server.json")
        assert resp.status_code == 200

    def test_recursively_strips_rating_details_and_nested_credentials(self):
        records = {
            "/github/": {
                "server_name": "GitHub",
                "is_enabled": True,
                "visibility": "public",
                "rating_details": [{"user": "top@example.com", "rating": 5}],
                "metadata": {
                    "mcp_registry_spec": {
                        "rating_details": [{"user": "nested@example.com", "rating": 4}],
                        "egress_auth": {"oauth": {"client_secret_encrypted": "nested-ciphertext"}},
                    }
                },
            }
        }
        repo = SimpleNamespace(find_with_filter=AsyncMock(return_value=records))

        def fake_to_canonical(record):
            return (
                {
                    "name": record["server_name"],
                    "_meta": record["metadata"]["mcp_registry_spec"],
                    "rating_details": record.get("rating_details"),
                },
                False,
            )

        with (
            patch.object(public_record_routes, "get_server_repository", return_value=repo),
            patch.object(public_record_routes, "to_canonical", side_effect=fake_to_canonical),
            patch.object(public_record_routes, "redact_backend_urls", side_effect=lambda d: d),
        ):
            resp = _client().get("/api/public/servers/github/server.json")

        assert resp.status_code == 200
        serialized = str(resp.json())
        assert "rating_details" not in serialized
        assert "top@example.com" not in serialized
        assert "nested@example.com" not in serialized
        assert "client_secret_encrypted" not in serialized
        assert "nested-ciphertext" not in serialized

    def test_non_public_server_returns_404(self):
        # find_with_filter returns only public+enabled, so a private/disabled
        # server is simply absent -> leaf never matches -> 404.
        repo = SimpleNamespace(find_with_filter=AsyncMock(return_value={}))
        with patch.object(public_record_routes, "get_server_repository", return_value=repo):
            resp = _client().get("/api/public/servers/secret/server.json")
        assert resp.status_code == 404

    def test_missing_server_returns_404(self):
        records = {"/other/": {"server_name": "Other", "is_enabled": True, "visibility": "public"}}
        repo = SimpleNamespace(find_with_filter=AsyncMock(return_value=records))
        with patch.object(public_record_routes, "get_server_repository", return_value=repo):
            resp = _client().get("/api/public/servers/nope/server.json")
        assert resp.status_code == 404


class TestPublicAgent:
    """Tests for GET /api/public/agents/{path}."""

    def test_strips_sensitive_fields(self):
        agents = {
            "/trav": {
                "name": "Trav",
                "visibility": "public",
                "is_enabled": True,
                "security_schemes": {"oauth2": {}},
                "allowed_groups": ["secret-group"],
                "registered_by": "alice",
                "num_stars": 4.5,
                "rating_count": 2,
                "rating_details": [
                    {"user": "alice@example.com", "rating": 5},
                    {"user": "bob@example.com", "rating": 4},
                ],
            }
        }
        repo = SimpleNamespace(find_with_filter=AsyncMock(return_value=agents))
        with patch.object(public_record_routes, "get_agent_repository", return_value=repo):
            resp = _client().get("/api/public/agents/trav")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "Trav"
        assert "security_schemes" not in body
        assert "allowed_groups" not in body
        assert "registered_by" not in body
        assert "rating_details" not in body
        assert body["num_stars"] == 4.5
        assert body["rating_count"] == 2

    def test_non_public_agent_404(self):
        # find_with_filter only returns public+enabled, so a private agent is absent.
        repo = SimpleNamespace(find_with_filter=AsyncMock(return_value={}))
        with patch.object(public_record_routes, "get_agent_repository", return_value=repo):
            resp = _client().get("/api/public/agents/secret")
        assert resp.status_code == 404


class TestPublicSkill:
    """Tests for GET /api/public/skills/{path}."""

    def test_returns_public_skill(self):
        skill = SimpleNamespace(
            path="/skills/pdf",
            model_dump=lambda mode="json": {
                "name": "pdf",
                "path": "/skills/pdf",
                "auth_credential_encrypted": "secret",
                "auth_scheme": "bearer",
                "auth_header_name": "Authorization",
                "credential_updated_at": "2026-07-29T00:00:00Z",
                "_identity_url_normalized": "https://private.example/skill.md",
                "owner": "alice",
                "num_stars": 4.0,
                "rating_count": 1,
                "rating_details": [{"user": "alice@example.com", "rating": 4}],
            },
        )
        repo = SimpleNamespace(list_filtered=AsyncMock(return_value=[skill]))
        with patch.object(public_record_routes, "get_skill_repository", return_value=repo):
            resp = _client().get("/api/public/skills/pdf")
        assert resp.status_code == 200
        body = resp.json()
        assert body["name"] == "pdf"
        assert "auth_credential_encrypted" not in body
        assert "auth_scheme" not in body
        assert "auth_header_name" not in body
        assert "credential_updated_at" not in body
        assert "_identity_url_normalized" not in body
        assert "owner" not in body
        assert "rating_details" not in body
        assert body["num_stars"] == 4.0
        assert body["rating_count"] == 1

    def test_missing_skill_404(self):
        repo = SimpleNamespace(list_filtered=AsyncMock(return_value=[]))
        with patch.object(public_record_routes, "get_skill_repository", return_value=repo):
            resp = _client().get("/api/public/skills/nope")
        assert resp.status_code == 404
