"""Tests for API-key hashing hardening and the /rate-limit oracle mitigation.

Covers:
- The API-key HMAC pepper is REQUIRED and fails closed (missing/empty/weak/short).
- Hashing is deterministic under a given pepper and changes when the pepper
  changes (defeats cross-deployment/offline brute force).
- Constant-time hash comparison helper.
- /rate-limit is throttled per client IP and does not leak key validity to
  unauthenticated callers beyond the standard 401.
- Retention admin error responses use static text (no allowlist leak).
"""

import pytest
from fastapi import HTTPException
from fastapi.testclient import TestClient
from unittest.mock import AsyncMock, MagicMock, patch

from app.config import _validate_pepper
from app.core.rate_limiter import IPThrottle
from app.main import app
from app.utils import helpers


_STRONG_PEPPER = "test-metrics-key-pepper-that-is-long-enough-1234567890"


@pytest.fixture
def strong_pepper(monkeypatch):
    """Ensure a strong pepper is set for the duration of a test."""
    monkeypatch.setenv("METRICS_KEY_PEPPER", _STRONG_PEPPER)
    return _STRONG_PEPPER


class TestPepperValidation:
    """The pepper must be present and strong; otherwise fail closed."""

    def test_missing_pepper_rejected(self):
        with pytest.raises(ValueError, match="required"):
            _validate_pepper(None)

    def test_empty_pepper_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            _validate_pepper("")

    def test_whitespace_pepper_rejected(self):
        with pytest.raises(ValueError, match="empty"):
            _validate_pepper("      ")

    def test_short_pepper_rejected(self):
        with pytest.raises(ValueError, match="at least 32"):
            _validate_pepper("too-short")

    def test_known_weak_pepper_rejected(self):
        # The historical hard-coded constant must be rejected even if set.
        with pytest.raises(ValueError, match="known-weak"):
            _validate_pepper("mcp-gateway-metrics-api-key-v1")

    def test_weak_check_runs_before_length_check(self):
        # "changeme" is short AND weak; the weak message must win.
        with pytest.raises(ValueError, match="known-weak"):
            _validate_pepper("changeme")

    def test_strong_pepper_accepted_and_stripped(self):
        assert _validate_pepper(f"  {_STRONG_PEPPER}  ") == _STRONG_PEPPER

    def test_former_env_example_placeholder_rejected(self):
        # .env.example ships METRICS_KEY_PEPPER unset (no default value) so there
        # is nothing to copy-paste-and-forget. This former placeholder must stay
        # rejected regardless, in case an operator copied it from an older
        # checkout: it is long enough to pass the length check but must still
        # fail closed rather than silently seed a predictable pepper.
        with pytest.raises(ValueError, match="known-weak"):
            _validate_pepper("CHANGE-ME-generate-with-openssl-rand-hex-32")

    def test_placeholder_prefix_rejected_case_insensitive(self):
        with pytest.raises(ValueError, match="known-weak"):
            _validate_pepper("ChAnGe-Me-this-is-still-a-placeholder-value")

    def test_embedded_placeholder_marker_rejected(self):
        # An operator who prepended to or embedded the example text (rather than
        # leaving it as a prefix) must still be rejected -- a start-only check
        # would let this through even though it is clearly the unedited example.
        with pytest.raises(ValueError, match="known-weak"):
            _validate_pepper("internal-CHANGE-ME-and-this-is-long-enough-xxxx")
        with pytest.raises(ValueError, match="known-weak"):
            _validate_pepper("prod-generate-with-openssl-rand-hex-32-placeholder")


class TestHashApiKeyFailsClosed:
    """hash_api_key must fail closed when the pepper is not usable."""

    def test_hash_raises_when_pepper_unset(self, monkeypatch):
        monkeypatch.delenv("METRICS_KEY_PEPPER", raising=False)
        with pytest.raises(ValueError):
            helpers.hash_api_key("some-key")

    def test_hash_raises_when_pepper_weak(self, monkeypatch):
        monkeypatch.setenv("METRICS_KEY_PEPPER", "changeme")
        with pytest.raises(ValueError):
            helpers.hash_api_key("some-key")

    def test_hash_deterministic_under_same_pepper(self, strong_pepper):
        assert helpers.hash_api_key("abc") == helpers.hash_api_key("abc")

    def test_hash_changes_with_pepper(self, monkeypatch):
        monkeypatch.setenv("METRICS_KEY_PEPPER", _STRONG_PEPPER)
        h1 = helpers.hash_api_key("abc")
        monkeypatch.setenv("METRICS_KEY_PEPPER", "a-completely-different-pepper-value-abcdef123456")
        h2 = helpers.hash_api_key("abc")
        assert h1 != h2

    def test_hash_not_plain_sha256(self, strong_pepper):
        # A leaked hash must not be a plain SHA-256 of the key (which would be
        # brute-forceable without the pepper).
        import hashlib

        assert helpers.hash_api_key("abc") != hashlib.sha256(b"abc").hexdigest()


class TestConstantTimeCompare:
    """The hash comparison helper must exist and be correct."""

    def test_equal_hashes(self, strong_pepper):
        h = helpers.hash_api_key("abc")
        assert helpers.hashes_equal(h, h) is True

    def test_unequal_hashes(self, strong_pepper):
        assert helpers.hashes_equal(helpers.hash_api_key("a"), helpers.hash_api_key("b")) is False


class TestIPThrottle:
    """The per-IP throttle must deny once the budget is exhausted."""

    async def test_allows_up_to_limit_then_denies(self):
        throttle = IPThrottle()
        # 3 allowed within the window.
        assert await throttle.allow("1.2.3.4", max_requests=3, window_seconds=60) is True
        assert await throttle.allow("1.2.3.4", max_requests=3, window_seconds=60) is True
        assert await throttle.allow("1.2.3.4", max_requests=3, window_seconds=60) is True
        # 4th denied.
        assert await throttle.allow("1.2.3.4", max_requests=3, window_seconds=60) is False

    async def test_separate_ips_have_separate_budgets(self):
        throttle = IPThrottle()
        assert await throttle.allow("1.1.1.1", max_requests=1, window_seconds=60) is True
        assert await throttle.allow("1.1.1.1", max_requests=1, window_seconds=60) is False
        # Different IP is unaffected.
        assert await throttle.allow("2.2.2.2", max_requests=1, window_seconds=60) is True

    async def test_window_resets_after_elapse(self):
        throttle = IPThrottle()
        assert await throttle.allow("9.9.9.9", max_requests=1, window_seconds=0) is True
        # window_seconds=0 means every call is a fresh window.
        assert await throttle.allow("9.9.9.9", max_requests=1, window_seconds=0) is True


class TestRateLimitEndpointThrottled:
    """/rate-limit must be throttled and must not be an unthrottled oracle."""

    @patch("app.api.auth.MetricsStorage")
    def test_invalid_key_throttled_after_budget(self, mock_storage_class, monkeypatch):
        monkeypatch.setenv("METRICS_KEY_PEPPER", _STRONG_PEPPER)
        # Reset throttle + config to a tiny budget so the test is fast.
        import app.api.routes as routes_module

        routes_module.ip_throttle = IPThrottle()
        monkeypatch.setattr(routes_module.settings, "RATE_LIMIT_ENDPOINT_MAX_REQUESTS", 2)
        monkeypatch.setattr(routes_module.settings, "RATE_LIMIT_ENDPOINT_WINDOW_SECONDS", 60)

        mock_storage = AsyncMock()
        mock_storage.get_api_key.return_value = None  # invalid key
        mock_storage_class.return_value = mock_storage

        client = TestClient(app)
        headers = {"X-API-Key": "guessed-key"}

        # First two attempts hit the (uniform) 401 for an unknown key.
        assert client.get("/rate-limit", headers=headers).status_code == 401
        assert client.get("/rate-limit", headers=headers).status_code == 401
        # Third is throttled: the endpoint can no longer be used as a fast
        # brute-force oracle.
        resp = client.get("/rate-limit", headers=headers)
        assert resp.status_code == 429
        assert "Retry-After" in resp.headers

    @patch("app.api.auth.MetricsStorage")
    def test_invalid_key_uniform_401(self, mock_storage_class, monkeypatch):
        monkeypatch.setenv("METRICS_KEY_PEPPER", _STRONG_PEPPER)
        import app.api.routes as routes_module

        routes_module.ip_throttle = IPThrottle()

        mock_storage = AsyncMock()
        mock_storage.get_api_key.return_value = None
        mock_storage_class.return_value = mock_storage

        client = TestClient(app)
        resp = client.get("/rate-limit", headers={"X-API-Key": "nope"})
        assert resp.status_code == 401
        # Response body must not echo the submitted key.
        assert "nope" not in resp.text

    @patch("app.api.auth.MetricsStorage")
    def test_inactive_key_indistinguishable_from_unknown(self, mock_storage_class, monkeypatch):
        # A known-but-inactive key must return the same 401 as an unknown key
        # so the endpoint does not confirm the key hash exists in the DB.
        monkeypatch.setenv("METRICS_KEY_PEPPER", _STRONG_PEPPER)
        import app.api.routes as routes_module

        routes_module.ip_throttle = IPThrottle()

        mock_storage = AsyncMock()
        mock_storage.get_api_key.return_value = {
            "service_name": "svc",
            "is_active": False,  # known but inactive
            "rate_limit": 1000,
            "last_used_at": None,
        }
        mock_storage_class.return_value = mock_storage

        client = TestClient(app)
        resp = client.get("/rate-limit", headers={"X-API-Key": "known-inactive"})
        assert resp.status_code == 401
        assert resp.json()["detail"] == "Invalid API key"


class TestRetentionErrorTextStatic:
    """Retention admin errors must not leak the table allowlist."""

    @patch("app.api.auth.MetricsStorage")
    @patch("app.api.routes.retention_manager")
    def test_cleanup_preview_invalid_table_static_text(
        self, mock_retention, mock_storage_class, monkeypatch
    ):
        monkeypatch.setenv("METRICS_KEY_PEPPER", _STRONG_PEPPER)

        # Authenticate: valid key.
        mock_storage = AsyncMock()
        mock_storage.get_api_key.return_value = {
            "service_name": "svc",
            "is_active": True,
            "rate_limit": 1000,
            "last_used_at": None,
        }
        mock_storage_class.return_value = mock_storage

        # retention_manager raises ValueError with the allowlist in the message.
        leaky_message = "Invalid table 'evil'. Allowed tables: metrics, auth_metrics, tool_metrics"
        mock_retention.get_cleanup_preview = AsyncMock(side_effect=ValueError(leaky_message))

        client = TestClient(app)
        resp = client.get(
            "/admin/retention/preview?table_name=evil",
            headers={"X-API-Key": "valid-key"},
        )
        assert resp.status_code == 400
        # The leaky allowlist must NOT appear in the response.
        assert "auth_metrics" not in resp.text
        assert "Allowed tables" not in resp.text
        assert resp.json()["detail"] == "Invalid table name"


class TestVerifyApiKeyFailsClosedOnMisconfig:
    """A missing pepper must deny requests, never accept."""

    @patch("app.api.auth.MetricsStorage")
    async def test_verify_denies_when_pepper_unset(self, mock_storage_class, monkeypatch):
        monkeypatch.delenv("METRICS_KEY_PEPPER", raising=False)

        from app.api.auth import verify_api_key

        mock_request = MagicMock()
        mock_request.headers = {"X-API-Key": "any-key"}

        with pytest.raises(HTTPException) as exc_info:
            await verify_api_key(mock_request)

        assert exc_info.value.status_code == 503
        # Storage must never be consulted when we cannot hash safely.
        mock_storage_class.assert_not_called()
