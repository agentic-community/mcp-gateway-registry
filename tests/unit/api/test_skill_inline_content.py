"""
Unit tests for inline content serving in the skill content endpoint.

Tests the get_skill_content endpoint behavior when a skill has
skill_md_content set (inline content) versus when it is None
(fallback to URL fetch).
"""

import base64
import logging
from typing import Any
from unittest.mock import (
    AsyncMock,
    MagicMock,
    patch,
)

import httpx
import pytest
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)


# =============================================================================
# CONSTANTS
# =============================================================================

INLINE_SKILL_PATH: str = "/skills/inline-test"
INLINE_SKILL_NAME: str = "inline-test"
INLINE_SKILL_DESCRIPTION: str = "A skill with inline content"
INLINE_SKILL_MD_CONTENT: str = "# Inline Skill\n\nThis content is stored in the database."
SKILL_MD_URL: str = "https://example.com/SKILL.md"
SKILL_MD_RAW_URL: str = "https://raw.example.com/SKILL.md"
URL_FETCHED_CONTENT: str = "# URL Skill\n\nThis content was fetched from a URL."
PNG_RESOURCE_PATH: str = "assets/logo.png"
PNG_SIZE_BYTES: int = 76_091
PNG_SIGNATURE: bytes = b"\x89PNG\r\n\x1a\n"


# =============================================================================
# HELPERS
# =============================================================================


def _make_mock_skill(
    path: str = INLINE_SKILL_PATH,
    name: str = INLINE_SKILL_NAME,
    description: str = INLINE_SKILL_DESCRIPTION,
    skill_md_content: str | None = None,
    skill_md_url: str = SKILL_MD_URL,
    skill_md_raw_url: str | None = SKILL_MD_RAW_URL,
    visibility: str = "public",
    owner: str = "testuser",
) -> MagicMock:
    """Create a mock SkillCard with configurable inline content.

    Args:
        path: Skill path
        name: Skill name
        description: Skill description
        skill_md_content: Inline SKILL.md content (None for URL fetch)
        skill_md_url: SKILL.md URL
        skill_md_raw_url: Raw SKILL.md URL
        visibility: Visibility setting
        owner: Skill owner

    Returns:
        MagicMock configured as a SkillCard
    """
    mock = MagicMock()
    mock.path = path
    mock.name = name
    mock.description = description
    mock.skill_md_content = skill_md_content
    mock.skill_md_url = skill_md_url
    mock.skill_md_raw_url = skill_md_raw_url
    mock.visibility = visibility
    mock.owner = owner
    mock.allowed_groups = []
    mock.tags = []
    # Drift-detection guard in get_skill_content() reads .content_integrity;
    # MagicMock's auto-attributes are truthy, which would trigger 409 Conflict.
    # Tests that exercise drift behaviour should override this explicitly.
    mock.content_integrity = None
    mock.resource_manifest = None
    # A real SkillCard always carries a health string; the content route reads it.
    mock.health_status = "healthy"
    return mock


def _make_png_bytes() -> bytes:
    """Build deterministic PNG-shaped bytes that are not valid UTF-8.

    Returns:
        Bytes beginning with the PNG signature and ``PNG_SIZE_BYTES`` long.
    """
    payload = bytes((index * 37 + 11) % 256 for index in range(PNG_SIZE_BYTES - len(PNG_SIGNATURE)))
    return PNG_SIGNATURE + payload


def _make_admin_user_context() -> dict[str, Any]:
    """Create admin user context for authentication.

    Returns:
        Dictionary with admin user context
    """
    return {
        "username": "admin",
        "is_admin": True,
        "groups": ["mcp-registry-admin"],
        "scopes": [],
        "accessible_servers": ["all"],
        "accessible_services": ["all"],
        "accessible_agents": ["all"],
        "auth_method": "session",
    }


def _create_test_client_with_mocks(
    mock_skill_service: MagicMock,
    user_context: dict[str, Any],
) -> TestClient:
    """Create a FastAPI test client with mocked skill service and auth.

    Args:
        mock_skill_service: Mocked skill service
        user_context: User context for authentication

    Returns:
        TestClient instance (as a context manager generator)
    """
    from registry.auth.dependencies import nginx_proxied_auth
    from registry.main import app

    app.dependency_overrides[nginx_proxied_auth] = lambda: user_context

    with (
        patch(
            "registry.api.skill_routes.get_skill_service",
            return_value=mock_skill_service,
        ),
        patch("registry.health.service.health_service", MagicMock()),
        patch("registry.core.nginx_service.nginx_service", MagicMock()),
    ):
        client = TestClient(app, cookies={"mcp_gateway_session": "test-session"})
        yield client

    app.dependency_overrides.clear()


# =============================================================================
# FIXTURES
# =============================================================================


@pytest.fixture
def admin_user_context() -> dict[str, Any]:
    """Create admin user context."""
    return _make_admin_user_context()


@pytest.fixture
def mock_skill_service() -> MagicMock:
    """Create a mock skill service.

    Returns:
        MagicMock configured as a skill service
    """
    service = MagicMock()
    service.get_skill = AsyncMock(return_value=None)
    service.list_skills_for_user = AsyncMock(return_value=[])
    return service


@pytest.fixture
def test_client(
    mock_settings,
    mock_skill_service,
    admin_user_context,
):
    """Create test client with admin auth and mocked skill service."""
    yield from _create_test_client_with_mocks(mock_skill_service, admin_user_context)


# =============================================================================
# TESTS
# =============================================================================


@pytest.mark.unit
class TestSkillInlineContent:
    """Tests for inline content serving in get_skill_content endpoint."""

    def test_inline_content_returned_when_skill_md_content_set(
        self,
        test_client,
        mock_skill_service,
    ):
        """When a skill has skill_md_content set, the endpoint returns it directly."""
        # Arrange
        mock_skill = _make_mock_skill(skill_md_content=INLINE_SKILL_MD_CONTENT)
        mock_skill_service.get_skill.return_value = mock_skill

        # Act
        response = test_client.get("/api/skills/inline-test/content")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == INLINE_SKILL_MD_CONTENT
        assert data["source"] == "inline"
        assert data["path"] == INLINE_SKILL_PATH

    def test_inline_content_response_has_no_url_field(
        self,
        test_client,
        mock_skill_service,
    ):
        """When inline content is served, the response should not contain a url field."""
        # Arrange
        mock_skill = _make_mock_skill(skill_md_content=INLINE_SKILL_MD_CONTENT)
        mock_skill_service.get_skill.return_value = mock_skill

        # Act
        response = test_client.get("/api/skills/inline-test/content")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "url" not in data

    def test_falls_through_to_url_fetch_when_skill_md_content_is_none(
        self,
        test_client,
        mock_skill_service,
    ):
        """When skill_md_content is None, the endpoint fetches from the URL."""
        # Arrange
        mock_skill = _make_mock_skill(skill_md_content=None)
        mock_skill_service.get_skill.return_value = mock_skill

        # Mock the httpx fetch and SSRF check (avoids DNS resolution in tests)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = URL_FETCHED_CONTENT
        mock_response.url = SKILL_MD_RAW_URL

        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("registry.services.skill_service._is_safe_url", return_value=True),
            patch(
                "registry.services.skill_service.guarded_async_client",
                return_value=mock_async_client,
            ),
        ):
            # Act
            response = test_client.get("/api/skills/inline-test/content")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == URL_FETCHED_CONTENT
        assert data["url"] == SKILL_MD_RAW_URL
        assert "source" not in data

    def test_falls_through_to_url_fetch_when_skill_md_content_is_empty_string(
        self,
        test_client,
        mock_skill_service,
    ):
        """When skill_md_content is an empty string (falsy), it falls through to URL fetch."""
        # Arrange
        mock_skill = _make_mock_skill(skill_md_content="")
        mock_skill_service.get_skill.return_value = mock_skill

        # Mock the httpx fetch and SSRF check (avoids DNS resolution in tests)
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.text = URL_FETCHED_CONTENT
        mock_response.url = SKILL_MD_RAW_URL

        mock_async_client = AsyncMock()
        mock_async_client.__aenter__ = AsyncMock(return_value=mock_async_client)
        mock_async_client.__aexit__ = AsyncMock(return_value=False)
        mock_async_client.get = AsyncMock(return_value=mock_response)

        with (
            patch("registry.services.skill_service._is_safe_url", return_value=True),
            patch(
                "registry.services.skill_service.guarded_async_client",
                return_value=mock_async_client,
            ),
        ):
            # Act
            response = test_client.get("/api/skills/inline-test/content")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["content"] == URL_FETCHED_CONTENT
        assert "source" not in data

    def test_inline_content_returns_404_when_skill_not_found(
        self,
        test_client,
        mock_skill_service,
    ):
        """When skill does not exist, the endpoint returns 404."""
        # Arrange
        mock_skill_service.get_skill.return_value = None

        # Act
        response = test_client.get("/api/skills/nonexistent/content")

        # Assert
        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()


def _resource_entry(path: str, resource_type: str) -> MagicMock:
    """Create a manifest entry with the given path and type."""
    entry = MagicMock()
    entry.path = path
    entry.type = resource_type
    return entry


def _make_skill_with_resources(*entries: MagicMock) -> MagicMock:
    """Create a mock skill whose manifest contains the given entries."""
    manifest = MagicMock()
    manifest.references = [e for e in entries if e.type == "reference"]
    manifest.scripts = [e for e in entries if e.type == "script"]
    manifest.agents = [e for e in entries if e.type == "agent"]
    manifest.assets = [e for e in entries if e.type == "asset"]

    skill = _make_mock_skill(skill_md_content=None)
    skill.resource_manifest = manifest
    return skill


def _fetch_resource_via_http(
    test_client: TestClient,
    mock_skill_service: MagicMock,
    skill: MagicMock,
    resource_path: str,
    upstream_bytes: bytes,
) -> httpx.Response:
    """GET a companion resource with the upstream fetch mocked."""
    mock_skill_service.get_skill.return_value = skill
    upstream_response = httpx.Response(200, content=upstream_bytes)
    with patch(
        "registry.api.skill_routes._fetch_authenticated_content",
        AsyncMock(return_value=upstream_response),
    ):
        return test_client.get(
            f"/api/skills/{INLINE_SKILL_NAME}/content",
            params={"resource": resource_path},
        )


@pytest.mark.unit
class TestAssetResourceContent:
    """Asset resources are always delivered as base64 JSON, never raw bytes."""

    def test_png_asset_round_trips_byte_for_byte(
        self,
        test_client,
        mock_skill_service,
    ):
        """A PNG asset comes back as base64 JSON with its original bytes."""
        # Arrange
        png_bytes = _make_png_bytes()
        skill = _make_skill_with_resources(_resource_entry(PNG_RESOURCE_PATH, "asset"))

        # Act
        response = _fetch_resource_via_http(
            test_client,
            mock_skill_service,
            skill,
            PNG_RESOURCE_PATH,
            png_bytes,
        )

        # Assert
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert data["encoding"] == "base64"
        decoded = base64.b64decode(data["content_base64"])
        assert decoded == png_bytes
        assert decoded.startswith(PNG_SIGNATURE)
        assert data["size_bytes"] == PNG_SIZE_BYTES
        assert data["path"] == PNG_RESOURCE_PATH
        assert data["type"] == "asset"
        assert data["url"].endswith(f"/{PNG_RESOURCE_PATH}")
        assert "content" not in data

    @pytest.mark.parametrize(
        ("path", "payload"),
        [
            ("assets/evil.html", b"<!-- \xff --><script>alert(1)</script>"),
            (
                "assets/evil.svg",
                b'<svg xmlns="http://www.w3.org/2000/svg">\xff<script>alert(1)</script></svg>',
            ),
        ],
    )
    def test_markup_assets_with_invalid_utf8_stay_json(
        self,
        test_client,
        mock_skill_service,
        path,
        payload,
    ):
        """Extension must not select an executable content type (stored XSS)."""
        # Arrange
        skill = _make_skill_with_resources(_resource_entry(path, "asset"))

        # Act
        response = _fetch_resource_via_http(
            test_client,
            mock_skill_service,
            skill,
            path,
            payload,
        )

        # Assert
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert data["encoding"] == "base64"
        assert base64.b64decode(data["content_base64"]) == payload
        assert data["type"] == "asset"
        assert "content" not in data


@pytest.mark.unit
class TestTextResourceContent:
    """Reference, script and agent resources always use the text path."""

    def test_latin1_reference_is_decoded_with_replacement(
        self,
        test_client,
        mock_skill_service,
    ):
        """A non-UTF-8 reference is text, not a binary asset."""
        # Arrange
        latin1_notes = b"caf\xe9 notes"
        skill = _make_skill_with_resources(_resource_entry("references/notes.md", "reference"))

        # Act
        response = _fetch_resource_via_http(
            test_client,
            mock_skill_service,
            skill,
            "references/notes.md",
            latin1_notes,
        )

        # Assert
        assert response.status_code == 200
        assert response.headers["content-type"] == "application/json"
        data = response.json()
        assert data["content"] == "caf� notes"
        assert data["type"] == "reference"
        assert data["path"] == "references/notes.md"
        assert "content_base64" not in data
        assert "encoding" not in data
