"""
Unit tests for Audit Middleware.

Validates: Requirements 4.1, 4.3
"""

import tempfile
from unittest.mock import MagicMock

import pytest

from registry.audit import AuditLogger, AuditMiddleware


class MockRequest:
    """Mock FastAPI Request object."""

    def __init__(
        self, path="/api/test", method="GET", headers=None, cookies=None, client_host="127.0.0.1"
    ):
        self.url = MagicMock()
        self.url.path = path
        self.method = method
        self._headers = headers or {}
        self._cookies = cookies or {}
        self.client = MagicMock()
        self.client.host = client_host
        self.state = MagicMock()
        self.state.user_context = None
        self.state.audit_action = None
        self.query_params = {}

    @property
    def headers(self):
        return self._headers

    @property
    def cookies(self):
        return self._cookies


class TestShouldLog:
    """Tests for _should_log method - health check and static asset exclusion."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.audit_logger = AuditLogger(log_dir=self.tmpdir)
        self.mock_app = MagicMock()

    def test_logs_regular_api_paths(self):
        """Regular API paths should be logged."""
        middleware = AuditMiddleware(self.mock_app, self.audit_logger)
        assert middleware._should_log("/api/servers") is True

    def test_excludes_health_checks_by_default(self):
        """Health check paths should NOT be logged by default."""
        middleware = AuditMiddleware(self.mock_app, self.audit_logger)
        assert middleware._should_log("/health") is False
        assert middleware._should_log("/api/health") is False

    def test_logs_health_checks_when_enabled(self):
        """Health check paths should be logged when enabled."""
        middleware = AuditMiddleware(self.mock_app, self.audit_logger, log_health_checks=True)
        assert middleware._should_log("/health") is True

    def test_excludes_static_assets_by_default(self):
        """Static asset paths should NOT be logged by default."""
        middleware = AuditMiddleware(self.mock_app, self.audit_logger)
        assert middleware._should_log("/static/app.js") is False
        assert middleware._should_log("/favicon.ico") is False


class TestCredentialDetection:
    """Tests for credential type and hint detection."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.audit_logger = AuditLogger(log_dir=self.tmpdir)
        self.middleware = AuditMiddleware(MagicMock(), self.audit_logger)

    def test_detects_session_cookie(self):
        """Session cookie should be detected."""
        # Use the actual configured cookie name from settings
        request = MockRequest(cookies={"mcp_gateway_session": "abc123"})
        assert self.middleware._get_credential_type(request) == "session_cookie"

    def test_detects_bearer_token(self):
        """Bearer token should be detected."""
        request = MockRequest(headers={"Authorization": "Bearer eyJhbGciOiJIUzI1NiJ9..."})
        assert self.middleware._get_credential_type(request) == "bearer_token"

    def test_detects_no_credential(self):
        """No credential should return 'none'."""
        request = MockRequest()
        assert self.middleware._get_credential_type(request) == "none"


class TestDispatch:
    """Tests for dispatch method."""

    def setup_method(self):
        self.tmpdir = tempfile.mkdtemp()
        self.audit_logger = AuditLogger(log_dir=self.tmpdir)
        self.middleware = AuditMiddleware(MagicMock(), self.audit_logger)

    @pytest.mark.asyncio
    async def test_captures_request_response(self):
        """Dispatch captures request and response details."""
        request = MockRequest(path="/api/servers", method="POST")
        request.state.user_context = {"username": "testuser", "auth_method": "oauth2"}

        response = MagicMock()
        response.status_code = 201
        response.headers = {}

        logged_events = []

        async def capture_log_event(record):
            logged_events.append(record)

        self.audit_logger.log_event = capture_log_event

        result = await self.middleware.dispatch(request, lambda r: self._async_return(response))

        assert result == response
        assert len(logged_events) == 1
        assert logged_events[0].request.method == "POST"
        assert logged_events[0].response.status_code == 201

    @pytest.mark.asyncio
    async def test_skips_excluded_paths(self):
        """Dispatch skips logging for excluded paths."""
        request = MockRequest(path="/health")
        response = MagicMock()
        response.status_code = 200

        log_called = []

        async def track_log(record):
            log_called.append(record)

        self.audit_logger.log_event = track_log

        await self.middleware.dispatch(request, lambda r: self._async_return(response))
        assert len(log_called) == 0

    async def _async_return(self, value):
        return value


class TestBestEffortSessionIdentity:
    """Tests for the session-cookie fallback used when no auth dep ran.

    These cover the /api/version case: a public endpoint with no auth
    dependency still gets logged under the caller's real username when
    the browser presents a valid session cookie.

    With server-side sessions, "valid cookie" means: signature checks out
    AND the resolved session_id has a record in the store. We mock the
    resolver to control both halves.
    """

    def setup_method(self):
        from registry.auth.dependencies import signer
        from registry.core.config import settings

        self.tmpdir = tempfile.mkdtemp()
        self.audit_logger = AuditLogger(log_dir=self.tmpdir)
        self.middleware = AuditMiddleware(MagicMock(), self.audit_logger)
        self.signer = signer
        self.cookie_name = settings.session_cookie_name

    def _signed_session_id_cookie(self, session_id: str = "b" * 64) -> str:
        return self.signer.dumps(session_id)

    @pytest.mark.asyncio
    async def test_fallback_recovers_username_from_valid_cookie(self, monkeypatch):
        async def _fake(_session_id):
            return {
                "session_id": "b" * 64,
                "username": "alice",
                "auth_method": "oauth2",
                "provider": "keycloak",
            }

        monkeypatch.setattr("registry.auth.dependencies._store_resolve_session", _fake)
        request = MockRequest(cookies={self.cookie_name: self._signed_session_id_cookie()})

        identity = await self.middleware._extract_identity(request)

        assert identity.username == "alice"
        assert identity.auth_method == "session-cookie-fallback"
        assert identity.provider == "keycloak"
        assert identity.credential_type == "session_cookie"

    @pytest.mark.asyncio
    async def test_fallback_anonymous_on_missing_cookie(self):
        request = MockRequest(cookies={})

        identity = await self.middleware._extract_identity(request)

        assert identity.username == "anonymous"
        assert identity.auth_method == "anonymous"

    @pytest.mark.asyncio
    async def test_fallback_anonymous_on_tampered_cookie(self):
        request = MockRequest(cookies={self.cookie_name: "garbage.value.here"})

        identity = await self.middleware._extract_identity(request)

        assert identity.username == "anonymous"
        assert identity.auth_method == "anonymous"

    @pytest.mark.asyncio
    async def test_auth_dependency_wins_over_fallback(self):
        """If a dep populated user_context, do NOT overwrite with fallback."""
        request = MockRequest(cookies={self.cookie_name: self._signed_session_id_cookie()})
        request.state.user_context = {
            "username": "bob",
            "auth_method": "oauth2",
            "provider": "keycloak",
        }

        identity = await self.middleware._extract_identity(request)

        assert identity.username == "bob"
        assert identity.auth_method == "oauth2"

    @pytest.mark.asyncio
    async def test_fallback_does_not_mutate_request_state(self, monkeypatch):
        async def _fake(_session_id):
            return {
                "session_id": "b" * 64,
                "username": "alice",
                "auth_method": "oauth2",
                "provider": "keycloak",
            }

        monkeypatch.setattr("registry.auth.dependencies._store_resolve_session", _fake)
        request = MockRequest(cookies={self.cookie_name: self._signed_session_id_cookie()})

        await self.middleware._extract_identity(request)

        assert request.state.user_context is None

    @pytest.mark.asyncio
    async def test_fallback_anonymous_when_session_missing_username(self, monkeypatch):
        """Store returns a record without a username — treated as anonymous."""

        async def _fake(_session_id):
            return {"session_id": "b" * 64, "auth_method": "oauth2"}

        monkeypatch.setattr("registry.auth.dependencies._store_resolve_session", _fake)
        request = MockRequest(cookies={self.cookie_name: self._signed_session_id_cookie()})

        identity = await self.middleware._extract_identity(request)

        assert identity.username == "anonymous"
        assert identity.auth_method == "anonymous"

    @pytest.mark.asyncio
    async def test_fallback_anonymous_when_cookie_missing_username(self, monkeypatch):
        # Backward-compat alias for the previous test name; same behavior.
        async def _fake(_session_id):
            return {"session_id": "b" * 64, "auth_method": "oauth2"}

        monkeypatch.setattr("registry.auth.dependencies._store_resolve_session", _fake)
        request = MockRequest(cookies={self.cookie_name: self._signed_session_id_cookie()})

        identity = await self.middleware._extract_identity(request)

        assert identity.username == "anonymous"
        assert identity.auth_method == "anonymous"

    @pytest.mark.asyncio
    async def test_fallback_anonymous_on_expired_cookie(self, monkeypatch):
        """max_age=-1 forces SignatureExpired on any cookie regardless of age."""
        from registry.core.config import settings

        request = MockRequest(cookies={self.cookie_name: self._signed_session_id_cookie()})

        monkeypatch.setattr(settings, "session_max_age_seconds", -1)
        identity = await self.middleware._extract_identity(request)

        assert identity.username == "anonymous"
        assert identity.auth_method == "anonymous"
