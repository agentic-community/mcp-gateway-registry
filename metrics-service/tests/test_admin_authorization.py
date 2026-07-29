"""Tests for privilege separation on the metrics-service /admin/* surface.

The metrics service authenticates two distinct classes of caller with the same
``X-API-Key`` header:

- INGEST clients (auth-server, registry, mcpgw) that POST metrics. They are
  verified by :func:`verify_api_key` against the API-key table.
- An ADMIN operator that can change retention policies, run cleanup, and read
  database stats via ``/admin/*``. Those routes are verified by
  :func:`verify_admin_api_key` against the dedicated ``METRICS_ADMIN_API_KEY``.

These tests assert the security PROPERTY that the two are separated: an
ingest-only key must never be able to perform admin operations, and the admin
surface must fail closed when no admin key is configured. They intentionally do
not assert the internal mechanism (constant-time compare, header name) beyond
what the property requires.
"""

import os
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient

from app.api.auth import verify_admin_api_key, verify_api_key
from app.config import _validate_admin_api_key
from app.main import app


# A strong, explicitly-configured admin key and a separate ingest key. These are
# deliberately DISTINCT to model correct operator configuration.
ADMIN_KEY = "admin-key-abcdef0123456789-strong-value"  # pragma: allowlist secret
INGEST_KEY = "ingest-key-9876543210fedcba-strong-value"  # pragma: allowlist secret

# Every admin route in the metrics service. Kept as a list so the uniformity
# tests exercise the WHOLE family, not just the retention endpoints that
# motivated the fix. (method, path)
ADMIN_ROUTES: list[tuple[str, str]] = [
    ("get", "/admin/retention/preview"),
    ("post", "/admin/retention/cleanup"),
    ("get", "/admin/retention/policies"),
    ("put", "/admin/retention/policies/metrics"),
    ("get", "/admin/database/stats"),
    ("get", "/admin/database/size"),
]


def _request(client: TestClient, method: str, path: str, headers: dict):
    """Issue a request for the given method against the test client."""
    return client.request(method, path, headers=headers)


class TestAdminKeyRequiredConfiguration:
    """The admin surface fails closed unless a strong admin key is configured."""

    async def test_missing_admin_key_denies_even_with_correct_looking_header(self):
        """With NO admin key configured, /admin/* denies (fail closed).

        This is the core fail-closed property: absence of configuration must
        never fall back to accepting a caller. Any presented key is rejected.
        """
        mock_request = MagicMock()
        mock_request.headers = {"X-API-Key": "anything-at-all"}

        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("METRICS_ADMIN_API_KEY", None)
            with pytest.raises(HTTPException) as exc_info:
                await verify_admin_api_key(mock_request)

        # 503: the admin surface is unusable until configured, distinct from a
        # normal 401 for a wrong key. It must NOT be a 2xx / accept.
        assert exc_info.value.status_code == 503

    async def test_empty_admin_key_denies(self):
        """An empty/whitespace admin key is treated as unconfigured (deny)."""
        mock_request = MagicMock()
        mock_request.headers = {"X-API-Key": "   "}

        with patch.dict(os.environ, {"METRICS_ADMIN_API_KEY": "   "}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_admin_api_key(mock_request)

        assert exc_info.value.status_code == 503


class TestAdminKeyStrengthValidation:
    """The admin key must be as strong as the pepper it mirrors.

    Property: the admin credential guards destructive operations, so a short or
    known-placeholder value must be rejected exactly like the HMAC pepper is --
    not merely a >=16-char length check that would accept "changemechangeme".
    """

    def test_short_16_char_key_rejected(self):
        # 16 chars was the old floor; a 16-char value must now fail the 32-char
        # floor. (A literal placeholder like "changemechangeme" is caught even
        # earlier by the weak-value check; here we use a non-placeholder value
        # to isolate the length floor.)
        sixteen_char = "aZ3kP9qR7tW1xY5v"
        assert len(sixteen_char) == 16
        with pytest.raises(ValueError, match="at least 32"):
            _validate_admin_api_key(sixteen_char)

    def test_placeholder_shorter_than_floor_caught_by_weak_check_first(self):
        # A 16-char known placeholder is rejected as weak BEFORE the length
        # check, giving the operator the most actionable message.
        assert len("changemechangeme") == 16
        with pytest.raises(ValueError, match="known-weak"):
            _validate_admin_api_key("changemechangeme")

    def test_known_weak_placeholder_rejected_even_if_long_enough(self):
        # Long enough to pass the length check, but an unmistakable placeholder.
        with pytest.raises(ValueError, match="known-weak"):
            _validate_admin_api_key("CHANGE-ME-generate-with-openssl-rand-hex-32")

    def test_embedded_placeholder_marker_rejected(self):
        with pytest.raises(ValueError, match="known-weak"):
            _validate_admin_api_key("internal-CHANGE-ME-and-this-is-long-enough-xxxx")

    def test_missing_admin_key_rejected(self):
        with pytest.raises(ValueError, match="required"):
            _validate_admin_api_key(None)

    def test_strong_32_char_key_accepted_and_stripped(self):
        strong = "admin-key-abcdef0123456789-strong-value"  # pragma: allowlist secret
        assert len(strong) >= 32
        assert _validate_admin_api_key(f"  {strong}  ") == strong


class TestAdminPrivilegeSeparation:
    """An ingest key cannot act as admin; the admin key can."""

    async def test_ingest_key_is_rejected_on_admin_dependency(self):
        """A valid-looking ingest key is REJECTED by the admin dependency.

        This is the whole point of the fix: holding an ingest credential must
        not grant admin authority. The admin key is configured and distinct, so
        the ingest key does not match and is denied.
        """
        mock_request = MagicMock()
        mock_request.headers = {"X-API-Key": INGEST_KEY}

        with patch.dict(os.environ, {"METRICS_ADMIN_API_KEY": ADMIN_KEY}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_admin_api_key(mock_request)

        assert exc_info.value.status_code == 401

    async def test_admin_key_is_accepted_by_admin_dependency(self):
        """The configured admin key is ACCEPTED by the admin dependency."""
        mock_request = MagicMock()
        mock_request.headers = {"X-API-Key": ADMIN_KEY}

        with patch.dict(os.environ, {"METRICS_ADMIN_API_KEY": ADMIN_KEY}):
            principal = await verify_admin_api_key(mock_request)

        assert principal == "admin"

    async def test_missing_header_is_rejected(self):
        """No X-API-Key header is rejected with 401 even when admin is set."""
        mock_request = MagicMock()
        mock_request.headers = {}

        with patch.dict(os.environ, {"METRICS_ADMIN_API_KEY": ADMIN_KEY}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_admin_api_key(mock_request)

        assert exc_info.value.status_code == 401


class TestAdminRoutesRejectIngestKeyUniformly:
    """Every /admin/* route rejects an ingest key (uniform across the family)."""

    @pytest.mark.parametrize("method,path", ADMIN_ROUTES)
    @patch("app.api.auth.MetricsStorage")
    def test_ingest_key_rejected_on_every_admin_route(self, mock_storage_class, method, path):
        """An ingest key that would pass verify_api_key still cannot reach any
        admin route.

        The ingest key here is backed by a mocked, ACTIVE api-key row -- so it
        would authenticate on the ingest path -- proving the rejection is due to
        privilege separation, not merely an unknown key. The check must hold for
        the entire admin family, not just retention.
        """
        mock_storage = AsyncMock()
        mock_storage.get_api_key.return_value = {
            "service_name": "ingest-client",
            "is_active": True,
            "rate_limit": 1000,
            "last_used_at": None,
        }
        mock_storage_class.return_value = mock_storage

        with patch.dict(os.environ, {"METRICS_ADMIN_API_KEY": ADMIN_KEY}):
            client = TestClient(app)
            response = _request(client, method, path, {"X-API-Key": INGEST_KEY})

        assert response.status_code == 401, (
            f"{method.upper()} {path} accepted an ingest key on an admin route"
        )

    @pytest.mark.parametrize("method,path", ADMIN_ROUTES)
    def test_admin_routes_deny_when_admin_key_unconfigured(self, method, path):
        """With no admin key configured, every admin route denies (fail closed)."""
        with patch.dict(os.environ, {}, clear=False):
            os.environ.pop("METRICS_ADMIN_API_KEY", None)
            client = TestClient(app)
            response = _request(client, method, path, {"X-API-Key": ADMIN_KEY})

        assert response.status_code == 503, (
            f"{method.upper()} {path} did not fail closed without an admin key"
        )


class TestAdminKeyAcceptedOnRoute:
    """The admin key reaches the handler on a representative admin route."""

    @patch("app.api.routes.retention_manager")
    def test_admin_key_reaches_retention_update_handler(self, mock_manager):
        """The admin key is accepted on PUT /admin/retention/policies/{table}.

        We patch the manager so the test isolates authorization from storage:
        reaching the handler (200) proves the admin key passed the dependency,
        and the ingest key in the sibling test proves it would not.
        """
        mock_manager.update_policy = AsyncMock(return_value=None)

        with patch.dict(os.environ, {"METRICS_ADMIN_API_KEY": ADMIN_KEY}):
            client = TestClient(app)
            response = client.put(
                "/admin/retention/policies/metrics?retention_days=30",
                headers={"X-API-Key": ADMIN_KEY},
            )

        assert response.status_code == 200
        mock_manager.update_policy.assert_awaited_once()


class TestAdminKeyNonAsciiHeader:
    """A non-ASCII X-API-Key must yield a clean 401, never an uncaught 500.

    Starlette decodes header values as latin-1, so a byte > 0x7F produces a
    non-ASCII str. hmac.compare_digest on two str would raise TypeError and
    surface as a 500 with a traceback. Comparing UTF-8 bytes keeps the result a
    clean 401 (still deny) for any mismatching key.
    """

    async def test_non_ascii_header_denies_with_401_not_typeerror(self):
        mock_request = MagicMock()
        # A latin-1 char above 0x7F, exactly what Starlette would hand us.
        mock_request.headers = {"X-API-Key": "éèê-not-the-admin-key"}

        with patch.dict(os.environ, {"METRICS_ADMIN_API_KEY": ADMIN_KEY}):
            with pytest.raises(HTTPException) as exc_info:
                await verify_admin_api_key(mock_request)

        # Clean deny, NOT a TypeError bubbling up as a 500.
        assert exc_info.value.status_code == 401

    async def test_non_ascii_admin_key_config_also_denies_cleanly(self):
        """A non-ASCII CONFIGURED admin key vs a mismatching presented key.

        Encoding both sides means the comparison never raises regardless of
        which side carries the non-ASCII byte; the mismatch is a clean 401.
        """
        mock_request = MagicMock()
        mock_request.headers = {"X-API-Key": "plain-ascii-but-wrong-key-value-xxxx"}

        with patch.dict(
            os.environ,
            {
                "METRICS_ADMIN_API_KEY": "admin-key-with-nöñ-ascii-bytes-xxxx"  # pragma: allowlist secret
            },
        ):
            with pytest.raises(HTTPException) as exc_info:
                await verify_admin_api_key(mock_request)

        assert exc_info.value.status_code == 401


class TestIngestPathNotRegressed:
    """The ingest routes still accept a valid ingest key (no regression)."""

    @patch("app.api.auth.MetricsStorage")
    async def test_ingest_key_still_verified_on_ingest_path(self, mock_storage_class):
        """verify_api_key still accepts a valid, active ingest key.

        The admin fix must not change ingest authentication. An admin key is
        configured to mirror production, but the ingest path does not consult
        it.
        """
        mock_storage = AsyncMock()
        mock_storage.get_api_key.return_value = {
            "service_name": "ingest-client",
            "is_active": True,
            "rate_limit": 1000,
            "last_used_at": None,
        }
        mock_storage.update_api_key_usage.return_value = None
        mock_storage_class.return_value = mock_storage

        mock_request = MagicMock()
        mock_request.headers = {"X-API-Key": INGEST_KEY}

        with patch.dict(os.environ, {"METRICS_ADMIN_API_KEY": ADMIN_KEY}):
            result = await verify_api_key(mock_request)

        assert result == "ingest-client"
