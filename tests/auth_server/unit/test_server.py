"""
Unit tests for auth_server/server.py

Tests cover token validation, session management, scope validation,
rate limiting, and helper functions.
"""

import logging
import time
from unittest.mock import AsyncMock, MagicMock, Mock, patch

import jwt
import pytest
from fastapi.testclient import TestClient

logger = logging.getLogger(__name__)


# Mark all tests in this file
pytestmark = [pytest.mark.unit, pytest.mark.auth]


# =============================================================================
# HELPER FUNCTION TESTS
# =============================================================================


class TestMaskingFunctions:
    """Tests for sensitive data masking functions."""

    def test_mask_sensitive_id_short(self):
        """Test masking short IDs."""
        from auth_server.server import mask_sensitive_id

        # Arrange
        short_id = "abc"

        # Act
        result = mask_sensitive_id(short_id)

        # Assert
        assert result == "***MASKED***"

    def test_mask_sensitive_id_normal(self):
        """Test masking normal length IDs."""
        from auth_server.server import mask_sensitive_id

        # Arrange
        normal_id = "us-east-1_ABCD12345"

        # Act
        result = mask_sensitive_id(normal_id)

        # Assert
        assert result.startswith("us-e")
        assert result.endswith("2345")
        assert "..." in result

    def test_mask_token(self):
        """mask_token emits no part of the value, not even a prefix."""
        from auth_server.server import mask_token

        # Arrange
        token = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.test"

        # Act
        result = mask_token(token)

        # Assert: no substring of the token leaks (a prefix is still sensitive
        # for opaque tokens / API keys); only a fixed marker is returned.
        assert result == "***MASKED***"
        assert "eyJh" not in result
        assert token[:4] not in result

    def test_mask_headers_masks_auth_credential_and_variants(self):
        """mask_headers redacts credential-bearing headers via substring match."""
        from auth_server.server import mask_headers

        headers = {
            "X-Auth-Credential": "super-secret-token",
            "Authorization": "Bearer eyJabc.def.ghi",
            "Cookie": "mcp_gateway_session=abc",
            "X-Api-Key": "k123456",
            "X-Access-Token": "t123456",
            "Accept": "application/json",
        }

        masked = mask_headers(headers)

        # The plaintext credential never appears in the masked output.
        assert masked["X-Auth-Credential"] == "***MASKED***"
        assert masked["Cookie"] == "***MASKED***"
        assert masked["X-Api-Key"] == "***MASKED***"
        assert masked["X-Access-Token"] == "***MASKED***"
        assert masked["Authorization"].startswith("Bearer ")
        assert "eyJabc.def.ghi" not in masked["Authorization"]
        # Non-sensitive headers pass through untouched.
        assert masked["Accept"] == "application/json"
        assert "super-secret-token" not in str(masked)

    def test_header_substrings_match_shared_redactor(self):
        """auth-server and registry header-substring sets must stay identical.

        The credential-bearing substring list is duplicated across two
        deployables (the auth server and the registry cannot import each
        other), so nothing at runtime catches the two drifting apart. If one
        gains a marker the other lacks, a credential header masked in one
        service would leak in the other. Pin them equal here so any edit to
        one copy without the other fails this test.
        """
        from auth_server.server import _SENSITIVE_HEADER_SUBSTRINGS
        from registry.common.log_redaction import SENSITIVE_HEADER_SUBSTRINGS

        assert set(_SENSITIVE_HEADER_SUBSTRINGS) == set(SENSITIVE_HEADER_SUBSTRINGS), (
            "auth_server._SENSITIVE_HEADER_SUBSTRINGS has drifted from "
            "registry.common.log_redaction.SENSITIVE_HEADER_SUBSTRINGS; keep them "
            "identical so a credential header is masked consistently in both services"
        )

    def test_anonymize_ip_ipv4(self):
        """Test IPv4 anonymization."""
        from auth_server.server import anonymize_ip

        # Arrange
        ipv4 = "192.168.1.100"

        # Act
        result = anonymize_ip(ipv4)

        # Assert
        assert result == "192.168.1.xxx"

    def test_anonymize_ip_ipv6(self):
        """Test IPv6 anonymization."""
        from auth_server.server import anonymize_ip

        # Arrange
        ipv6 = "2001:0db8:85a3:0000:0000:8a2e:0370:7334"

        # Act
        result = anonymize_ip(ipv6)

        # Assert
        assert result.endswith(":xxxx")
        assert "2001" in result

    def test_hash_username(self):
        """Test username hashing for privacy."""
        from auth_server.server import hash_username

        # Arrange
        username = "testuser"

        # Act
        result = hash_username(username)

        # Assert
        assert result.startswith("user_")
        assert len(result) > len(username)
        # Same input produces same hash
        assert hash_username(username) == result


class TestSafeIdentitySummary:
    """safe_identity_summary must never leak claim/user-info values."""

    def test_omits_email_name_and_group_values(self):
        from auth_server.server import safe_identity_summary

        claims = {
            "sub": "1234567890abcdef",
            "email": "alice@example.com",
            "name": "Alice Example",
            "preferred_username": "alice.example",
            "groups": ["engineering", "admins", "finance"],
            "aud": "my-client",
        }

        summary = safe_identity_summary(claims)
        rendered = str(summary)

        # PII / authz values must NOT appear anywhere in the summary.
        assert "alice@example.com" not in rendered
        assert "Alice Example" not in rendered
        assert "engineering" not in rendered
        assert "admins" not in rendered
        assert "finance" not in rendered
        # sub is masked, not raw.
        assert summary["sub"] != claims["sub"]
        assert "..." in summary["sub"]
        # Only counts and claim NAMES are exposed.
        assert summary["group_count"] == 3
        assert set(summary["claim_names"]) == set(claims.keys())
        assert "email" in summary["claim_names"]  # the NAME, not the value

    def test_counts_roles_when_groups_absent(self):
        from auth_server.server import safe_identity_summary

        claims = {"sub": "abcd1234efgh", "roles": ["r1", "r2"]}
        summary = safe_identity_summary(claims)
        assert summary["group_count"] == 2

    def test_handles_missing_sub_and_non_dict(self):
        from auth_server.server import safe_identity_summary

        assert safe_identity_summary({})["sub"] is None
        assert safe_identity_summary(None) == {"claims": "unavailable"}

    def test_mapped_user_dict_is_safe(self):
        """A mapped_user dict (email/name/groups) must not leak its values."""
        from auth_server.server import safe_identity_summary

        mapped_user = {
            "username": "bob@corp.com",
            "email": "bob@corp.com",
            "name": "Bob Corp",
            "groups": ["g1", "g2", "g3", "g4"],
        }
        rendered = str(safe_identity_summary(mapped_user))
        assert "bob@corp.com" not in rendered
        assert "Bob Corp" not in rendered
        assert "g1" not in rendered
        assert safe_identity_summary(mapped_user)["group_count"] == 4


class TestServerNameNormalization:
    """Tests for server name normalization and matching."""

    def test_normalize_server_name_with_trailing_slash(self):
        """Test removing trailing slash."""
        from auth_server.server import _normalize_server_name

        # Arrange
        name_with_slash = "test-server/"

        # Act
        result = _normalize_server_name(name_with_slash)

        # Assert
        assert result == "test-server"

    def test_normalize_server_name_without_trailing_slash(self):
        """Test name without trailing slash."""
        from auth_server.server import _normalize_server_name

        # Arrange
        name = "test-server"

        # Act
        result = _normalize_server_name(name)

        # Assert
        assert result == "test-server"

    def test_server_names_match_exact(self):
        """Test exact server name matching."""
        from auth_server.server import _server_names_match

        # Act & Assert
        assert _server_names_match("test-server", "test-server")

    def test_server_names_match_with_trailing_slash(self):
        """Test server name matching with trailing slash."""
        from auth_server.server import _server_names_match

        # Act & Assert
        assert _server_names_match("test-server/", "test-server")
        assert _server_names_match("test-server", "test-server/")

    def test_server_names_match_wildcard(self):
        """Test wildcard matching."""
        from auth_server.server import _server_names_match

        # Act & Assert
        assert _server_names_match("*", "any-server")
        assert _server_names_match("*", "another-server")


class TestIsRedirectWithinCookieDomain:
    """Tests for _is_redirect_within_cookie_domain same-origin validation.

    The registry's server-to-server logout hop (issue #1503) forwards a raw
    X-Forwarded-Host that may carry a port (e.g. "localhost:7860"), while the
    redirect_uri host is port-stripped by urlparse. These tests guard against
    the port causing a false same-origin rejection.
    """

    def _request(self, forwarded_host="", forwarded_proto="", url_host="localhost", scheme="http"):
        request = MagicMock()
        headers = {"x-forwarded-host": forwarded_host, "x-forwarded-proto": forwarded_proto}
        request.headers = MagicMock()
        request.headers.get = lambda key, default="": headers.get(key, default)
        request.url = MagicMock()
        request.url.hostname = url_host
        request.url.scheme = scheme
        return request

    def test_forwarded_host_with_port_matches_portless_redirect(self):
        """X-Forwarded-Host: localhost:7860 must match http://localhost:7860/logout."""
        from auth_server.server import _is_redirect_within_cookie_domain

        request = self._request(forwarded_host="localhost:7860", forwarded_proto="http")
        assert _is_redirect_within_cookie_domain("http://localhost:7860/logout", "", request)

    def test_forwarded_host_https_with_domain(self):
        """A real public host forwarded with https matches an https redirect."""
        from auth_server.server import _is_redirect_within_cookie_domain

        request = self._request(forwarded_host="app.example.com", forwarded_proto="https")
        assert _is_redirect_within_cookie_domain("https://app.example.com/logout", "", request)

    def test_different_host_rejected(self):
        """A redirect to a different host is rejected when no cookie domain covers it."""
        from auth_server.server import _is_redirect_within_cookie_domain

        request = self._request(forwarded_host="app.example.com", forwarded_proto="https")
        assert not _is_redirect_within_cookie_domain("https://evil.example.net/logout", "", request)

    def test_scheme_mismatch_rejected(self):
        """A proto mismatch (http forwarded vs https redirect) is rejected."""
        from auth_server.server import _is_redirect_within_cookie_domain

        request = self._request(forwarded_host="localhost:7860", forwarded_proto="http")
        assert not _is_redirect_within_cookie_domain("https://localhost:7860/logout", "", request)

    def test_configured_external_host_trusted_on_s2s_hop(self, monkeypatch):
        """Issue #1503: on the internal S2S logout hop the request scheme reconstructs
        as http and X-Forwarded-Host is absent, so the same-origin match fails for a
        legitimate https public redirect_uri. The redirect_uri host matching the
        deployment's own AUTH_SERVER_EXTERNAL_URL must still be accepted."""
        from auth_server.server import _is_redirect_within_cookie_domain

        monkeypatch.setenv("AUTH_SERVER_EXTERNAL_URL", "https://d2xl2zfuhgc4l0.cloudfront.net")
        # Reconstructed request origin is internal (http, no forwarded host).
        request = self._request(forwarded_host="", forwarded_proto="", url_host="127.0.0.1")
        assert _is_redirect_within_cookie_domain(
            "https://d2xl2zfuhgc4l0.cloudfront.net/logout", "", request
        )

    def test_configured_external_host_falls_back_to_registry_url(self, monkeypatch):
        """REGISTRY_URL is used when AUTH_SERVER_EXTERNAL_URL is unset."""
        from auth_server.server import _is_redirect_within_cookie_domain

        monkeypatch.delenv("AUTH_SERVER_EXTERNAL_URL", raising=False)
        monkeypatch.setenv("REGISTRY_URL", "https://app.example.com")
        request = self._request(forwarded_host="", forwarded_proto="", url_host="127.0.0.1")
        assert _is_redirect_within_cookie_domain("https://app.example.com/logout", "", request)

    def test_non_configured_host_still_rejected_on_s2s_hop(self, monkeypatch):
        """A redirect to a host that is NOT the configured external host is still
        rejected even on the internal hop (the fix does not open a general bypass)."""
        from auth_server.server import _is_redirect_within_cookie_domain

        monkeypatch.setenv("AUTH_SERVER_EXTERNAL_URL", "https://d2xl2zfuhgc4l0.cloudfront.net")
        request = self._request(forwarded_host="", forwarded_proto="", url_host="127.0.0.1")
        assert not _is_redirect_within_cookie_domain("https://evil.example.net/logout", "", request)


class TestGroupToScopeMapping:
    """Tests for mapping IdP groups to MCP scopes."""

    @pytest.mark.asyncio
    async def test_map_groups_to_scopes_basic(self, mock_scopes_config):
        """Test basic group to scope mapping."""
        from auth_server.server import map_groups_to_scopes

        # Arrange - Mock the repository to return the union of scopes in one
        # bulk call (map_groups_to_scopes resolves all groups in a single query).
        mock_repo = AsyncMock()
        mock_repo.get_group_mappings_bulk.return_value = [
            "read:servers",
            "read:tools",
            "write:servers",
        ]

        with patch("auth_server.server.get_scope_repository", return_value=mock_repo):
            groups = ["users", "developers"]

            # Act
            scopes = await map_groups_to_scopes(groups)

            # Assert
            assert "read:servers" in scopes
            assert "write:servers" in scopes
            assert "read:tools" in scopes
            mock_repo.get_group_mappings_bulk.assert_awaited_once_with(groups)

    @pytest.mark.asyncio
    async def test_map_groups_to_scopes_no_duplicates(self, mock_scopes_config):
        """Test that duplicate scopes are removed."""
        from auth_server.server import map_groups_to_scopes

        # Arrange - the bulk query returns the de-duplicated union; verify
        # map_groups_to_scopes preserves that (no duplicate re-introduced).
        mock_repo = AsyncMock()
        mock_repo.get_group_mappings_bulk.return_value = [
            "read:servers",
            "read:tools",
            "write:servers",
        ]

        with patch("auth_server.server.get_scope_repository", return_value=mock_repo):
            # Both groups have "read:servers"
            groups = ["users", "developers"]

            # Act
            scopes = await map_groups_to_scopes(groups)

            # Assert
            # Should only appear once (duplicates removed)
            assert scopes.count("read:servers") == 1
            assert "write:servers" in scopes
            assert "read:tools" in scopes

    @pytest.mark.asyncio
    async def test_map_groups_to_scopes_unknown_group(self, mock_scopes_config):
        """Test mapping with unknown group."""
        from auth_server.server import map_groups_to_scopes

        # Arrange - Mock repository to return empty list for unknown groups
        mock_repo = AsyncMock()
        mock_repo.get_group_mappings_bulk.return_value = []

        with patch("auth_server.server.get_scope_repository", return_value=mock_repo):
            groups = ["unknown-group"]

            # Act
            scopes = await map_groups_to_scopes(groups)

            # Assert
            assert len(scopes) == 0


class TestScopeValidation:
    """Tests for scope-based access validation."""

    @pytest.mark.asyncio
    async def test_validate_server_tool_access_allowed(self, mock_scope_repository_with_data):
        """Test access validation when allowed."""
        from auth_server.server import validate_server_tool_access

        # Arrange
        with patch(
            "auth_server.server.get_scope_repository", return_value=mock_scope_repository_with_data
        ):
            server_name = "test-server"
            method = "initialize"
            tool_name = None
            user_scopes = ["read:servers"]

            # Act
            result = await validate_server_tool_access(server_name, method, tool_name, user_scopes)

            # Assert
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_server_tool_access_denied(self, mock_scope_repository_with_data):
        """Test access validation when denied."""
        from auth_server.server import validate_server_tool_access

        # Arrange
        with patch(
            "auth_server.server.get_scope_repository", return_value=mock_scope_repository_with_data
        ):
            server_name = "other-server"
            method = "initialize"
            tool_name = None
            user_scopes = ["read:servers"]  # Only for test-server

            # Act
            result = await validate_server_tool_access(server_name, method, tool_name, user_scopes)

            # Assert
            assert result is False

    @pytest.mark.asyncio
    async def test_validate_server_tool_access_wildcard_server(
        self, mock_scope_repository_with_data
    ):
        """Test wildcard server access."""
        from auth_server.server import validate_server_tool_access

        # Arrange
        with patch(
            "auth_server.server.get_scope_repository", return_value=mock_scope_repository_with_data
        ):
            server_name = "any-server"
            method = "initialize"
            tool_name = None
            user_scopes = ["admin:all"]

            # Act
            result = await validate_server_tool_access(server_name, method, tool_name, user_scopes)

            # Assert
            assert result is True

    @pytest.mark.asyncio
    async def test_validate_server_tool_access_tools_call(self, mock_scope_repository_with_data):
        """Test access validation for tools/call method."""
        from auth_server.server import validate_server_tool_access

        # Arrange
        with patch(
            "auth_server.server.get_scope_repository", return_value=mock_scope_repository_with_data
        ):
            server_name = "test-server"
            method = "tools/call"
            tool_name = "test-tool"
            user_scopes = ["write:servers"]  # Has wildcard tools

            # Act
            result = await validate_server_tool_access(server_name, method, tool_name, user_scopes)

            # Assert
            assert result is True

    @pytest.mark.asyncio
    async def test_allow_does_not_log_scopes_or_config_at_info(
        self, mock_scope_repository_with_data, caplog
    ):
        """An allow does not leak user_scopes/scope_config at INFO.

        The verbose per-request trace (user scopes, the full scope_config
        authorization policy) must stay at DEBUG. Exactly one INFO line, the
        access decision, is emitted and it must not carry the scopes list.
        """
        from auth_server.server import validate_server_tool_access

        with patch(
            "auth_server.server.get_scope_repository",
            return_value=mock_scope_repository_with_data,
        ):
            caplog.set_level(logging.INFO, logger="auth_server.server")
            server_name = "test-server"
            method = "initialize"
            user_scopes = ["read:servers"]

            result = await validate_server_tool_access(server_name, method, None, user_scopes)

        assert result is True

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        info_text = "\n".join(r.getMessage() for r in info_records)

        # The full scope config (its dict repr) and the raw scopes list must not
        # appear at INFO.
        assert "'methods'" not in info_text
        assert str(user_scopes) not in info_text
        assert "User scopes" not in info_text

        # Exactly one INFO decision line, and it records the grant.
        decision_lines = [line for line in info_text.splitlines() if "Access " in line]
        assert len(decision_lines) == 1
        assert "Access granted" in decision_lines[0]
        assert "test-server" in decision_lines[0]

    @pytest.mark.asyncio
    async def test_deny_does_not_log_scopes_or_config_at_info(
        self, mock_scope_repository_with_data, caplog
    ):
        """A deny emits exactly one INFO decision line, no scopes dump."""
        from auth_server.server import validate_server_tool_access

        with patch(
            "auth_server.server.get_scope_repository",
            return_value=mock_scope_repository_with_data,
        ):
            caplog.set_level(logging.INFO, logger="auth_server.server")
            server_name = "other-server"
            method = "initialize"
            user_scopes = ["read:servers"]

            result = await validate_server_tool_access(server_name, method, None, user_scopes)

        assert result is False

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        info_text = "\n".join(r.getMessage() for r in info_records)

        assert "'methods'" not in info_text
        assert str(user_scopes) not in info_text

        decision_lines = [line for line in info_text.splitlines() if "Access " in line]
        assert len(decision_lines) == 1
        assert "Access denied" in decision_lines[0]

    @pytest.mark.asyncio
    async def test_scope_config_visible_only_at_debug(
        self, mock_scope_repository_with_data, caplog
    ):
        """The scope_config dump still exists, but only at DEBUG."""
        from auth_server.server import validate_server_tool_access

        with patch(
            "auth_server.server.get_scope_repository",
            return_value=mock_scope_repository_with_data,
        ):
            caplog.set_level(logging.DEBUG, logger="auth_server.server")
            await validate_server_tool_access("test-server", "initialize", None, ["read:servers"])

        debug_text = "\n".join(r.getMessage() for r in caplog.records if r.levelno == logging.DEBUG)
        # The full authorization policy trace lands at DEBUG.
        assert "'methods'" in debug_text
        assert "User scopes" in debug_text

    @pytest.mark.asyncio
    async def test_exception_path_denies_with_one_info_line_no_scopes(self, caplog):
        """The fail-closed exception branch denies, emits exactly one
        INFO decision line, and never leaks user_scopes at INFO."""
        from auth_server.server import validate_server_tool_access

        # A scope repo whose lookup raises drives the except branch.
        failing_repo = MagicMock()
        failing_repo.get_server_scopes = AsyncMock(side_effect=RuntimeError("boom"))

        with patch(
            "auth_server.server.get_scope_repository",
            return_value=failing_repo,
        ):
            caplog.set_level(logging.INFO, logger="auth_server.server")
            user_scopes = ["read:servers", "secret-scope-name"]

            result = await validate_server_tool_access(
                "test-server", "initialize", None, user_scopes
            )

        # Fail closed.
        assert result is False

        info_records = [r for r in caplog.records if r.levelno == logging.INFO]
        info_text = "\n".join(r.getMessage() for r in info_records)

        # No scope list leaked at INFO, and exactly one decision line (deny).
        assert str(user_scopes) not in info_text
        assert "secret-scope-name" not in info_text
        decision_lines = [line for line in info_text.splitlines() if "Access " in line]
        assert len(decision_lines) == 1
        assert "Access denied" in decision_lines[0]

    def test_validate_scope_subset_valid(self):
        """Test that requested scopes are subset of user scopes."""
        from auth_server.server import validate_scope_subset

        # Arrange
        user_scopes = ["read:servers", "write:servers", "admin:all"]
        requested_scopes = ["read:servers", "write:servers"]

        # Act
        result = validate_scope_subset(user_scopes, requested_scopes)

        # Assert
        assert result is True

    def test_validate_scope_subset_invalid(self):
        """Test that requested scopes exceed user scopes."""
        from auth_server.server import validate_scope_subset

        # Arrange
        user_scopes = ["read:servers"]
        requested_scopes = ["read:servers", "write:servers"]

        # Act
        result = validate_scope_subset(user_scopes, requested_scopes)

        # Assert
        assert result is False


class TestRateLimiting:
    """Tests for token generation rate limiting."""

    def test_check_rate_limit_under_limit(self):
        """Test rate limiting when under limit."""
        from auth_server.server import check_rate_limit, user_token_generation_counts

        # Arrange
        user_token_generation_counts.clear()
        username = "testuser"

        # Act
        result = check_rate_limit(username)

        # Assert
        assert result is True

    def test_check_rate_limit_exceeded(self, monkeypatch):
        """Test rate limiting when limit exceeded."""
        from auth_server.server import check_rate_limit, user_token_generation_counts

        # Arrange
        monkeypatch.setenv("MAX_TOKENS_PER_USER_PER_HOUR", "3")
        from auth_server import server

        server.MAX_TOKENS_PER_USER_PER_HOUR = 3

        user_token_generation_counts.clear()
        username = "testuser"

        # Generate tokens up to limit
        for _ in range(3):
            check_rate_limit(username)

        # Act - try one more
        result = check_rate_limit(username)

        # Assert
        assert result is False

    def test_check_rate_limit_cleanup_old_entries(self):
        """Test that old rate limit entries are cleaned up."""
        from auth_server.server import check_rate_limit, user_token_generation_counts

        # Arrange
        user_token_generation_counts.clear()
        username = "testuser"
        current_time = int(time.time())
        old_hour = (current_time // 3600) - 2  # 2 hours ago

        # Add old entry
        user_token_generation_counts[f"{username}:{old_hour}"] = 5

        # Act
        check_rate_limit(username)

        # Assert - old entry should be removed
        assert f"{username}:{old_hour}" not in user_token_generation_counts


# =============================================================================
# SESSION COOKIE VALIDATION TESTS
# =============================================================================


class TestSessionCookieValidation:
    """Tests for session cookie validation."""

    @pytest.mark.asyncio
    async def test_validate_session_cookie_valid(self, auth_env_vars, valid_session_cookie):
        """Test validating a valid session cookie.

        With server-side sessions, validation does signer.loads(cookie) →
        session_id (string), then store.resolve_session(session_id) →
        hydrated record. We mock the store lookup.
        """
        from itsdangerous import URLSafeTimedSerializer

        from auth_server.server import validate_session_cookie

        test_signer = URLSafeTimedSerializer(auth_env_vars["SECRET_KEY"])

        async def _fake_resolve(_session_id):
            return {
                "session_id": _session_id,
                "username": "testuser",
                "email": "testuser@example.com",
                "groups": ["users", "developers"],
                "provider": "cognito",
                "auth_method": "oauth2",
            }

        with (
            patch("auth_server.server.signer", test_signer),
            patch("session_store.resolve_session", _fake_resolve),
        ):
            result = await validate_session_cookie(valid_session_cookie)

            assert result["valid"] is True
            assert result["username"] == "testuser"
            assert result["method"] == "session_cookie"
            assert "users" in result["groups"]

    @pytest.mark.asyncio
    async def test_validate_session_cookie_expired(self, auth_env_vars):
        """Test validating an expired session cookie."""
        from itsdangerous import URLSafeTimedSerializer

        from auth_server.server import validate_session_cookie

        # Create signer with test key
        test_signer = URLSafeTimedSerializer(auth_env_vars["SECRET_KEY"])

        # Create cookie with far past timestamp
        old_data = {"username": "testuser", "groups": []}
        import time

        old_time = time.time() - 30000  # Way past max_age
        with patch("time.time", return_value=old_time):
            old_cookie = test_signer.dumps(old_data)

        # Patch the module's signer to use test key
        with patch("auth_server.server.signer", test_signer):
            # Act & Assert
            with pytest.raises(ValueError, match="expired"):
                await validate_session_cookie(old_cookie)

    @pytest.mark.asyncio
    async def test_validate_session_cookie_invalid_signature(self, auth_env_vars):
        """Test validating cookie with invalid signature."""
        from auth_server.server import validate_session_cookie

        # Arrange
        invalid_cookie = "invalid.signature.data"

        # Act & Assert
        with pytest.raises(ValueError, match="Invalid session cookie"):
            await validate_session_cookie(invalid_cookie)


# =============================================================================
# SIMPLIFIED COGNITO VALIDATOR TESTS
# =============================================================================


class TestSimplifiedCognitoValidator:
    """Tests for SimplifiedCognitoValidator class."""

    def test_validator_initialization(self):
        """Test validator initialization."""
        from auth_server.server import SimplifiedCognitoValidator

        # Act
        validator = SimplifiedCognitoValidator(region="us-west-2")

        # Assert
        assert validator.default_region == "us-west-2"
        assert validator._jwks_cache == {}

    @patch("auth_server.server.requests.get")
    def test_get_jwks_success(self, mock_get, mock_jwks_response):
        """Test successful JWKS retrieval."""
        from auth_server.server import SimplifiedCognitoValidator

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_response.raise_for_status.return_value = None
        mock_get.return_value = mock_response

        validator = SimplifiedCognitoValidator()
        user_pool_id = "us-east-1_TEST"
        region = "us-east-1"

        # Act
        jwks = validator._get_jwks(user_pool_id, region)

        # Assert
        assert "keys" in jwks
        assert len(jwks["keys"]) == 2
        mock_get.assert_called_once()

    @patch("auth_server.server.requests.get")
    def test_get_jwks_cached(self, mock_get, mock_jwks_response):
        """Test JWKS caching."""
        from auth_server.server import SimplifiedCognitoValidator

        # Arrange
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_get.return_value = mock_response

        validator = SimplifiedCognitoValidator()
        user_pool_id = "us-east-1_TEST"
        region = "us-east-1"

        # Act - call twice
        jwks1 = validator._get_jwks(user_pool_id, region)
        jwks2 = validator._get_jwks(user_pool_id, region)

        # Assert - should only call once due to caching
        assert mock_get.call_count == 1
        assert jwks1 == jwks2

    def test_validate_self_signed_token_valid(self, auth_env_vars, self_signed_token):
        """Test validating a valid self-signed token."""
        from auth_server.server import SimplifiedCognitoValidator

        # Arrange
        validator = SimplifiedCognitoValidator()

        # Patch SECRET_KEY at module level (loaded at import time before fixture sets env)
        with patch("auth_server.server.SECRET_KEY", auth_env_vars["SECRET_KEY"]):
            # Act
            result = validator.validate_self_signed_token(self_signed_token)

            # Assert
            assert result["valid"] is True
            assert result["method"] == "self_signed"
            assert result["username"] == "testuser"
            assert "read:servers" in result["scopes"]

    def test_validate_self_signed_token_expired(self, auth_env_vars):
        """Test validating an expired self-signed token."""
        from auth_server.server import SimplifiedCognitoValidator

        # Arrange
        validator = SimplifiedCognitoValidator()
        secret_key = auth_env_vars["SECRET_KEY"]
        now = int(time.time())

        # Create expired token
        payload = {
            "iss": "mcp-auth-server",
            "aud": "mcp-registry",
            "sub": "testuser",
            "exp": now - 3600,  # Expired 1 hour ago
            "iat": now - 7200,
            "token_use": "access",
        }
        expired_token = jwt.encode(payload, secret_key, algorithm="HS256")

        # Patch SECRET_KEY at module level (loaded at import time before fixture sets env)
        with patch("auth_server.server.SECRET_KEY", secret_key):
            # Act & Assert
            with pytest.raises(ValueError, match="expired"):
                validator.validate_self_signed_token(expired_token)


# =============================================================================
# FASTAPI ENDPOINT TESTS
# =============================================================================


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    @patch("auth_server.server.get_auth_provider")
    def test_health_check(self, mock_get_provider):
        """Test health check endpoint."""
        # Arrange - import after mocking
        import auth_server.server as server_module

        client = TestClient(server_module.app)

        # Act
        response = client.get("/health")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "healthy"
        assert data["service"] == "simplified-auth-server"


class TestValidateEndpoint:
    """Tests for /validate endpoint."""

    @patch("auth_server.server.get_auth_provider")
    def test_validate_with_valid_token(
        self,
        mock_get_provider,
        mock_cognito_provider,
        auth_env_vars,
        mock_scope_repository_with_data,
    ):
        """Test validation with valid JWT token."""
        # Arrange
        mock_get_provider.return_value = mock_cognito_provider

        import auth_server.server as server_module

        # Patch scope repository to return test data
        with patch(
            "auth_server.server.get_scope_repository", return_value=mock_scope_repository_with_data
        ):
            client = TestClient(server_module.app)

            # Act
            # URL format: /server-name/mcp-endpoint where endpoint is mcp, sse, or messages
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer test-token",
                    "X-Original-URL": "https://example.com/test-server/mcp",
                },
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["username"] == "testuser"

    @patch("auth_server.server.get_auth_provider")
    def test_validate_a2a_agent_request_allowed(
        self,
        mock_get_provider,
        mock_cognito_provider,
        auth_env_vars,
        mock_scope_repository_with_data,
    ):
        """An A2A agent proxy request with invoke access passes and skips MCP checks.

        The gateway credential is presented in X-Authorization (the A2A egress
        trust model): Authorization is reserved for the target-agent credential.
        """
        mock_get_provider.return_value = mock_cognito_provider

        import auth_server.server as server_module

        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.validate_a2a_agent_access",
                AsyncMock(return_value=True),
            ),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "X-Authorization": "Bearer test-token",
                    "X-Original-URL": "https://example.com/agent/travel/",
                },
            )

        assert response.status_code == 200
        assert response.json()["valid"] is True

    @patch("auth_server.server.get_auth_provider")
    def test_validate_a2a_agent_request_denied(
        self,
        mock_get_provider,
        mock_cognito_provider,
        auth_env_vars,
        mock_scope_repository_with_data,
    ):
        """An A2A agent proxy request without invoke access is rejected with 403."""
        mock_get_provider.return_value = mock_cognito_provider

        import auth_server.server as server_module

        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.validate_a2a_agent_access",
                AsyncMock(return_value=False),
            ),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "X-Authorization": "Bearer test-token",
                    "X-Original-URL": "https://example.com/agent/travel/",
                },
            )

        assert response.status_code == 403

    @patch("auth_server.server.get_auth_provider")
    def test_validate_a2a_no_authorization_fallback(
        self,
        mock_get_provider,
        mock_cognito_provider,
        auth_env_vars,
        mock_scope_repository_with_data,
    ):
        """On an agent path, Authorization is NOT accepted as the gateway credential.

        Authorization carries the target-agent credential (forwarded end-to-end),
        so a request with only Authorization and no X-Authorization must fail
        closed as unauthenticated rather than authenticate on -- and leak -- the
        target-agent credential.
        """
        mock_get_provider.return_value = mock_cognito_provider

        import auth_server.server as server_module

        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.validate_a2a_agent_access",
                AsyncMock(return_value=True),
            ),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer target-agent-token",
                    "X-Original-URL": "https://example.com/agent/travel/",
                },
            )

        assert response.status_code == 401

    @patch("auth_server.server.get_auth_provider")
    def test_validate_a2a_rejects_duplicate_gateway_credential(
        self,
        mock_get_provider,
        mock_cognito_provider,
        auth_env_vars,
        mock_scope_repository_with_data,
    ):
        """Duplicating the gateway token into Authorization is refused (fail closed).

        If Authorization equals the validated X-Authorization, the Authorization
        copy would be forwarded to the registrant-controlled agent backend and
        could be replayed against the registry. The request is rejected with 401.
        """
        mock_get_provider.return_value = mock_cognito_provider

        import auth_server.server as server_module

        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.validate_a2a_agent_access",
                AsyncMock(return_value=True),
            ),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "X-Authorization": "Bearer test-token",
                    "Authorization": "Bearer test-token",
                    "X-Original-URL": "https://example.com/agent/travel/",
                },
            )

        assert response.status_code == 401

    @patch("auth_server.server.get_auth_provider")
    def test_validate_a2a_rejects_duplicate_ignoring_scheme_prefix(
        self,
        mock_get_provider,
        mock_cognito_provider,
        auth_env_vars,
        mock_scope_repository_with_data,
    ):
        """The duplicate-token guard compares token VALUES, not raw headers.

        A caller that sends the same token but with a differing "Bearer " scheme
        prefix / whitespace in one header must still be refused, or the gateway
        credential would leak to the backend (PR #1434 finding SF-4).
        """
        mock_get_provider.return_value = mock_cognito_provider

        import auth_server.server as server_module

        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.validate_a2a_agent_access",
                AsyncMock(return_value=True),
            ),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "X-Authorization": "Bearer test-token",
                    # Same token value, no "Bearer " prefix: must still be caught.
                    "Authorization": "test-token",
                    "X-Original-URL": "https://example.com/agent/travel/",
                },
            )

        assert response.status_code == 401

    @patch("auth_server.server.get_auth_provider")
    def test_validate_uninspectable_body_fails_closed(
        self,
        mock_get_provider,
        mock_cognito_provider,
        auth_env_vars,
        mock_scope_repository_with_data,
    ):
        """A server-scoped request with an uninspectable (spilled) body is denied.

        When capture_body.lua could not buffer the body in memory it sets
        X-Body-Uninspectable=1 and no X-Body. /validate must not default the
        method to "initialize" and authorize -- it must fail closed so a
        privileged body cannot slip through unauthorized (TM-15 edge defense).
        """
        mock_get_provider.return_value = mock_cognito_provider

        import auth_server.server as server_module

        with patch(
            "auth_server.server.get_scope_repository",
            return_value=mock_scope_repository_with_data,
        ):
            client = TestClient(server_module.app)

            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer test-token",
                    "X-Original-URL": "https://example.com/test-server/mcp",
                    "X-Body-Uninspectable": "1",
                },
            )

        assert response.status_code == 413

    @patch("auth_server.server.get_auth_provider")
    def test_validate_unparseable_body_fails_closed(
        self,
        mock_get_provider,
        mock_cognito_provider,
        auth_env_vars,
        mock_scope_repository_with_data,
    ):
        """A server-scoped request whose X-Body is present but unparseable is denied.

        A non-empty body that cannot be parsed into a scope-relevant payload
        must not silently default to "initialize" -- the real method is unknown,
        so /validate fails closed rather than authorizing it (TM-15 edge defense).
        """
        mock_get_provider.return_value = mock_cognito_provider

        import auth_server.server as server_module

        with patch(
            "auth_server.server.get_scope_repository",
            return_value=mock_scope_repository_with_data,
        ):
            client = TestClient(server_module.app)

            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer test-token",
                    "X-Original-URL": "https://example.com/test-server/mcp",
                    "X-Body": "{not-valid-json",
                },
            )

        assert response.status_code == 400

    @patch("auth_server.server.get_auth_provider")
    def test_validate_missing_auth_header(self, mock_get_provider, auth_env_vars):
        """Test validation without Authorization header returns 401."""
        # Arrange
        import auth_server.server as server_module

        client = TestClient(server_module.app)

        # Act
        response = client.get("/validate")

        # Assert
        assert response.status_code == 401
        assert "Missing or invalid Authorization header" in response.json()["detail"]

    @patch("auth_server.server.get_auth_provider")
    def test_validate_with_session_cookie(
        self,
        mock_get_provider,
        auth_env_vars,
        valid_session_cookie,
        mock_scope_repository_with_data,
    ):
        """Test validation with valid session cookie."""
        from itsdangerous import URLSafeTimedSerializer

        import auth_server.server as server_module

        test_signer = URLSafeTimedSerializer(auth_env_vars["SECRET_KEY"])

        async def _fake_resolve(_session_id):
            return {
                "session_id": _session_id,
                "username": "testuser",
                "groups": ["users", "developers"],
                "provider": "cognito",
                "auth_method": "oauth2",
            }

        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch("auth_server.server.signer", test_signer),
            patch("session_store.resolve_session", _fake_resolve),
        ):
            client = TestClient(server_module.app)

            response = client.get(
                "/validate",
                headers={
                    "Cookie": f"mcp_gateway_session={valid_session_cookie}",
                    "X-Original-URL": "https://example.com/test-server/mcp",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True

    @patch("auth_server.server.get_auth_provider")
    def test_validate_token_failure_returns_401_not_500(
        self,
        mock_get_provider,
        auth_env_vars,
    ):
        """A ValueError from the provider's validate_token() must surface as
        401 with WWW-Authenticate, not 500. Pre-#989 this was a 500 which
        prevented MCP discovery clients from re-triggering OAuth on a stale
        token.

        Regression test for the auth_server fix shipped alongside #989.
        """
        from unittest.mock import MagicMock

        provider = MagicMock()
        provider.validate_token.side_effect = ValueError("Token missing 'kid' in header")
        mock_get_provider.return_value = provider

        import auth_server.server as server_module

        client = TestClient(server_module.app)

        response = client.get(
            "/validate",
            headers={
                "Authorization": "Bearer junk-token",
                "X-Original-URL": "https://example.com/test-server/mcp",
            },
        )

        assert response.status_code == 401
        assert "www-authenticate" in {k.lower() for k in response.headers.keys()}
        assert response.headers["www-authenticate"].startswith("Bearer")


class TestConfigEndpoint:
    """Tests for /config endpoint."""

    @patch("auth_server.server.get_auth_provider")
    def test_config_keycloak(self, mock_get_provider, mock_keycloak_provider):
        """Test config endpoint with Keycloak provider."""
        # Arrange
        mock_get_provider.return_value = mock_keycloak_provider

        import auth_server.server as server_module

        client = TestClient(server_module.app)

        # Act
        response = client.get("/config")

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert data["auth_type"] == "keycloak"


def _internal_auth_headers(auth_env_vars: dict) -> dict:
    """Build an Authorization header carrying a valid internal JWT.

    ``/internal/tokens`` and ``/internal/reload-scopes`` both require a
    Bearer JWT signed with the shared SECRET_KEY (see
    ``registry.auth.internal.generate_internal_token``). Tests that POST
    to either endpoint must attach this header.
    """
    from registry.auth.internal import generate_internal_token

    token = generate_internal_token(
        subject="test-suite",
        purpose="unit-test",
    )
    return {"Authorization": f"Bearer {token}"}


class TestGenerateTokenEndpoint:
    """Tests for /internal/tokens endpoint."""

    @patch("auth_server.server.get_auth_provider")
    def test_generate_token_success(self, mock_get_provider, auth_env_vars):
        """Test successful token generation using Keycloak M2M."""
        # Arrange
        import auth_server.server as server_module

        # Mock Keycloak provider
        mock_provider = Mock()
        mock_provider.get_provider_info.return_value = {"provider_type": "keycloak"}
        # M2M token uses fixed scopes for IdP compatibility, not user-requested scopes
        mock_provider.get_m2m_token.return_value = {
            "access_token": "mock_keycloak_m2m_token",
            "refresh_token": None,
            "expires_in": 28800,
            "refresh_expires_in": 0,
            "scope": "openid email profile",
        }
        mock_get_provider.return_value = mock_provider

        client = TestClient(server_module.app)

        request_data = {
            "user_context": {"username": "testuser", "scopes": ["read:servers", "write:servers"]},
            "requested_scopes": ["read:servers"],
            "expires_in_hours": 8,
            "description": "Test token",
        }

        # Act
        response = client.post(
            "/internal/tokens",
            json=request_data,
            headers=_internal_auth_headers(auth_env_vars),
        )

        # Assert
        assert response.status_code == 200
        data = response.json()
        assert "access_token" in data
        assert data["access_token"] == "mock_keycloak_m2m_token"
        assert data["token_type"] == "Bearer"
        # Scope in response comes from Keycloak M2M client configuration
        assert data["scope"] == "openid email profile"
        # Verify Keycloak M2M was called with IdP-compatible scopes
        mock_provider.get_m2m_token.assert_called_once_with(scope="openid email profile")

    @patch("auth_server.server.get_auth_provider")
    def test_generate_token_missing_username(self, mock_get_provider, auth_env_vars):
        """Test token generation without username."""
        # Arrange
        import auth_server.server as server_module

        client = TestClient(server_module.app)

        request_data = {
            "user_context": {"scopes": ["read:servers"]},
            "requested_scopes": ["read:servers"],
            "expires_in_hours": 8,
        }

        # Act
        response = client.post(
            "/internal/tokens",
            json=request_data,
            headers=_internal_auth_headers(auth_env_vars),
        )

        # Assert
        assert response.status_code == 400
        assert "Username is required" in response.json()["detail"]

    @patch("auth_server.server.get_auth_provider")
    def test_generate_token_invalid_scopes(self, mock_get_provider, auth_env_vars):
        """Test token generation with invalid scopes."""
        # Arrange
        import auth_server.server as server_module

        client = TestClient(server_module.app)

        request_data = {
            "user_context": {"username": "testuser", "scopes": ["read:servers"]},
            "requested_scopes": ["admin:all"],  # User doesn't have this
            "expires_in_hours": 8,
        }

        # Act
        response = client.post(
            "/internal/tokens",
            json=request_data,
            headers=_internal_auth_headers(auth_env_vars),
        )

        # Assert
        assert response.status_code == 403
        assert "exceed user permissions" in response.json()["detail"]

    @patch("auth_server.server.get_auth_provider")
    def test_generate_token_rate_limit(self, mock_get_provider, auth_env_vars, monkeypatch):
        """Test token generation rate limiting."""
        # Arrange
        monkeypatch.setenv("MAX_TOKENS_PER_USER_PER_HOUR", "2")

        import auth_server.server as server_module

        server_module.MAX_TOKENS_PER_USER_PER_HOUR = 2
        server_module.user_token_generation_counts.clear()

        # Mock Keycloak provider for successful token generation
        mock_provider = Mock()
        mock_provider.get_provider_info.return_value = {"provider_type": "keycloak"}
        mock_provider.get_m2m_token.return_value = {
            "access_token": "mock_keycloak_m2m_token",
            "refresh_token": None,
            "expires_in": 28800,
            "refresh_expires_in": 0,
            "scope": "read:servers",
        }
        mock_get_provider.return_value = mock_provider

        client = TestClient(server_module.app)

        request_data = {
            "user_context": {"username": "testuser", "scopes": ["read:servers"]},
            "requested_scopes": ["read:servers"],
            "expires_in_hours": 8,
        }

        # Act - generate tokens up to limit
        for _ in range(2):
            response = client.post(
                "/internal/tokens",
                json=request_data,
                headers=_internal_auth_headers(auth_env_vars),
            )
            assert response.status_code == 200

        # Try one more - should fail
        response = client.post(
            "/internal/tokens",
            json=request_data,
            headers=_internal_auth_headers(auth_env_vars),
        )

        # Assert
        assert response.status_code == 429
        assert "Rate limit exceeded" in response.json()["detail"]


class TestInternalRouterGate:
    """Meta-test: every route under the ``/internal/`` prefix on the
    auth-server must require the signed-Bearer internal-JWT gate.

    The router-level dependency in ``auth_server.server.internal_router``
    is the mechanism that provides this guarantee. This test enumerates
    the routes at runtime and asserts each one returns 401 when called
    without an ``Authorization`` header — so a future developer who
    adds a new ``/internal/*`` handler by accident on ``@app.post`` or
    without the router dependency will get a failing build instead of
    an unauthenticated privileged endpoint.
    """

    def _internal_routes(self, server_module) -> list:
        """Return every (path, method) on app.routes whose path starts
        with ``/internal/``. Filters out non-HTTP things like
        ``Mount``/``WebSocketRoute`` which don't have a ``methods``
        attribute.
        """
        collected: list[tuple[str, str]] = []

        def _walk(routes) -> None:
            for route in routes:
                original_router = getattr(route, "original_router", None)
                if original_router is not None:
                    _walk(original_router.routes)
                    continue
                path = getattr(route, "path", None)
                methods = getattr(route, "methods", None)
                if not path or not methods:
                    continue
                if not path.startswith("/internal/"):
                    continue
                for method in methods:
                    # HEAD/OPTIONS are auto-added and not interesting.
                    if method in ("HEAD", "OPTIONS"):
                        continue
                    collected.append((path, method))

        _walk(server_module.app.routes)
        return collected

    def test_at_least_the_known_endpoints_are_present(self, auth_env_vars):
        """Guard against the meta-test trivially passing when the
        router is empty. If someone deletes both endpoints this catches it."""
        import auth_server.server as server_module

        paths = {path for path, _ in self._internal_routes(server_module)}
        assert "/internal/tokens" in paths
        assert "/internal/reload-scopes" in paths

    def test_every_internal_route_rejects_unauthenticated_request(self, auth_env_vars):
        """For every /internal/* route, a request without Authorization
        must return 401. A future /internal/foo endpoint registered on
        ``@app.post`` (bypassing the router) will fail here because the
        handler runs without the gate and returns something other than
        401.
        """
        import auth_server.server as server_module

        client = TestClient(server_module.app)
        routes = self._internal_routes(server_module)
        assert routes, "expected at least one /internal/* route"

        failures: list[str] = []
        for path, method in routes:
            response = client.request(method, path, json={})
            if response.status_code != 401:
                failures.append(
                    f"  {method} {path} returned {response.status_code} "
                    f"(expected 401); body={response.text[:200]}"
                )
        if failures:
            raise AssertionError(
                "One or more /internal/* routes accept requests without the "
                "internal-JWT gate. This is almost always because a handler "
                "was registered directly on ``app`` (e.g. "
                "``@app.post('/internal/foo')``) instead of on the "
                "``internal_router`` defined in auth_server/server.py.\n"
                "\n"
                "Fix: decorate the handler with ``@internal_router.post(...)`` "
                "(and drop the ``/internal`` prefix from the path, since the "
                "router already provides it). This inherits the router-level "
                "``Depends(validate_internal_auth)`` so the handler cannot "
                "ship without the signed-Bearer check.\n"
                "\n"
                "Offending routes:\n" + "\n".join(failures)
            )


class TestGenerateTokenEndpointInternalAuth:
    """Regression coverage for the internal-JWT gate on /internal/tokens.

    The endpoint mints JWTs — any caller that can reach it can issue a
    token for any user. We require the caller to prove knowledge of the
    shared ``SECRET_KEY`` via a short-lived internal JWT signed the same
    way the existing ``/internal/reload-scopes`` handler does it.
    """

    def test_rejects_missing_authorization(self, auth_env_vars):
        import auth_server.server as server_module

        client = TestClient(server_module.app)
        response = client.post(
            "/internal/tokens",
            json={
                "user_context": {"username": "alice", "scopes": []},
                "requested_scopes": [],
                "expires_in_hours": 8,
            },
        )
        assert response.status_code == 401
        assert "Missing authorization header" in response.json()["detail"]

    def test_rejects_non_bearer_scheme(self, auth_env_vars):
        import auth_server.server as server_module

        client = TestClient(server_module.app)
        response = client.post(
            "/internal/tokens",
            json={
                "user_context": {"username": "alice", "scopes": []},
                "requested_scopes": [],
                "expires_in_hours": 8,
            },
            headers={"Authorization": "Basic YWxpY2U6cGFzcw=="},
        )
        assert response.status_code == 401

    def test_rejects_bearer_signed_with_wrong_key(self, auth_env_vars):
        # Sign a JWT with a DIFFERENT secret — identical shape, wrong key.
        # Models the realistic threat: an attacker on the internal network
        # who does not possess SECRET_KEY.
        import time as _time

        import auth_server.server as server_module

        wrong_key_token = jwt.encode(
            {
                "iss": "mcp-auth-server",
                "aud": "mcp-registry",
                "sub": "attacker",
                "purpose": "forged",
                "token_use": "access",
                "iat": int(_time.time()),
                "exp": int(_time.time()) + 60,
            },
            "not-the-real-secret",
            algorithm="HS256",
        )
        client = TestClient(server_module.app)
        response = client.post(
            "/internal/tokens",
            json={
                "user_context": {"username": "alice", "scopes": []},
                "requested_scopes": [],
                "expires_in_hours": 8,
            },
            headers={"Authorization": f"Bearer {wrong_key_token}"},
        )
        assert response.status_code == 401

    def test_rejects_expired_bearer(self, auth_env_vars):
        # A token correctly signed with SECRET_KEY but whose ``exp`` is in
        # the past must be rejected. Models the realistic threat of an
        # attacker who captured a valid internal JWT from an earlier
        # request and tries to replay it after the short TTL.
        import time as _time

        import auth_server.server as server_module

        secret = auth_env_vars["SECRET_KEY"]
        now = int(_time.time())
        expired_token = jwt.encode(
            {
                "iss": "mcp-auth-server",
                "aud": "mcp-registry",
                "sub": "registry-service",
                "purpose": "generate-token",
                "token_use": "access",
                # ``leeway=30`` on validation, so push exp well past that.
                "iat": now - 600,
                "exp": now - 120,
            },
            secret,
            algorithm="HS256",
        )
        client = TestClient(server_module.app)
        response = client.post(
            "/internal/tokens",
            json={
                "user_context": {"username": "alice", "scopes": []},
                "requested_scopes": [],
                "expires_in_hours": 8,
            },
            headers={"Authorization": f"Bearer {expired_token}"},
        )
        assert response.status_code == 401


class TestReloadScopesEndpoint:
    """Tests for /internal/reload-scopes endpoint."""

    @patch("registry.common.scopes_loader.reload_scopes_config")
    @patch("auth_server.server.get_auth_provider")
    def test_reload_scopes_success_with_jwt(
        self, mock_get_provider, mock_reload_scopes, auth_env_vars
    ):
        """Test successful scopes reload using an internal service JWT.

        The token is minted via ``generate_internal_token`` (not hand-rolled)
        so it carries the internal-service contract enforced by
        ``validate_internal_auth``: audience ``mcp-internal``, a
        ``token_kind=internal-service`` claim, and a signature made with the
        derived internal key rather than the raw SECRET_KEY (Security Finding 1).
        A user-shape token (aud=mcp-registry, raw-key signed) is now rejected.
        """
        # Arrange
        mock_reload_scopes.return_value = {"group_mappings": {}}

        import auth_server.server as server_module
        from registry.auth.internal import generate_internal_token

        # Patch module-level SECRET_KEY to match the test env var
        # (it may already be set to a different value from earlier test imports)
        secret_key = auth_env_vars["SECRET_KEY"]
        original_secret_key = server_module.SECRET_KEY
        server_module.SECRET_KEY = secret_key

        try:
            client = TestClient(server_module.app)

            # The auth_env_vars fixture sets SECRET_KEY in os.environ (monkeypatch),
            # which is what generate_internal_token reads to derive the signing key.
            token = generate_internal_token(
                subject="registry-service",
                purpose="reload-scopes",
            )

            # Act
            response = client.post(
                "/internal/reload-scopes", headers={"Authorization": f"Bearer {token}"}
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert "successfully" in data["message"]
        finally:
            server_module.SECRET_KEY = original_secret_key

    @patch("auth_server.server.get_auth_provider")
    def test_reload_scopes_no_auth(self, mock_get_provider):
        """Test scopes reload without authentication."""
        # Arrange
        import auth_server.server as server_module

        client = TestClient(server_module.app)

        # Act
        response = client.post("/internal/reload-scopes")

        # Assert
        assert response.status_code == 401

    @patch("auth_server.server.get_auth_provider")
    def test_reload_scopes_invalid_jwt(self, mock_get_provider, auth_env_vars):
        """Test scopes reload with an invalid JWT token."""
        # Arrange
        import auth_server.server as server_module

        client = TestClient(server_module.app)

        # Act
        response = client.post(
            "/internal/reload-scopes", headers={"Authorization": "Bearer invalid-token"}
        )

        # Assert
        assert response.status_code == 401

    @patch("registry.common.scopes_loader.reload_scopes_config")
    @patch("auth_server.server.get_auth_provider")
    def test_reload_scopes_basic_auth_rejected(self, mock_get_provider, auth_env_vars):
        """Test that Basic Auth is rejected (no longer supported)."""
        # Arrange
        import base64

        import auth_server.server as server_module

        client = TestClient(server_module.app)

        credentials = base64.b64encode(b"testadmin:testadminpass").decode()

        # Act
        response = client.post(
            "/internal/reload-scopes", headers={"Authorization": f"Basic {credentials}"}
        )

        # Assert - Basic Auth is no longer supported
        assert response.status_code == 401
        assert "Unsupported authentication scheme" in response.json()["detail"]


# =============================================================================
# NETWORK-TRUSTED MODE TESTS
# =============================================================================


class TestNetworkTrustedMode:
    """Tests for network-trusted auth bypass mode (issue #357)."""

    def test_network_trusted_bypasses_registry_api(self):
        """When enabled, registry API requests bypass JWT validation."""
        # Arrange
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("test-api-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer test-api-key",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["username"] == "network-user"
            assert data["client_id"] == "network-trusted"
            assert data["method"] == "network-trusted"
            assert "mcp-servers-unrestricted/read" in data["scopes"]
            assert "mcp-servers-unrestricted/execute" in data["scopes"]
            assert response.headers["X-Auth-Method"] == "network-trusted"
            assert response.headers["X-Username"] == "network-user"

    def test_network_trusted_missing_auth_falls_through_to_jwt(self):
        """Missing Authorization header falls through to JWT/session validation.

        Before issue #871 the static-token block terminated with a 401. After
        the fix the block falls through so Okta JWT / self-signed JWT callers
        still work. An absent Authorization header ultimately reaches the JWT
        block which returns 401 with a different detail message.
        """
        # Arrange
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("test-api-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act
            response = client.get(
                "/validate",
                headers={
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert: 401 comes from the downstream JWT block, not the static
            # token block. The detail text changed to the JWT-block message.
            assert response.status_code == 401
            assert "Missing or invalid Authorization header" in response.json()["detail"]

    @patch("auth_server.server.get_auth_provider")
    def test_network_trusted_does_not_bypass_mcp_gateway(
        self,
        mock_get_provider,
        auth_env_vars,
    ):
        """MCP server access still requires full validation even when bypass is enabled."""
        # Arrange
        import auth_server.server as server_module

        mock_provider = MagicMock()
        mock_provider.validate_token = AsyncMock(side_effect=ValueError("Invalid token"))
        mock_get_provider.return_value = mock_provider

        token_map = _make_legacy_token_map("test-api-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act - request to an MCP server path, not /api/ or /v0.1/
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer test-api-key",
                    "X-Original-URL": "https://example.com/mcpserver/messages",
                },
            )

            # Assert - should NOT be bypassed, falls through to normal validation
            assert response.status_code != 200 or response.json().get("method") != "network-trusted"

    def test_network_trusted_disabled_by_default(self, auth_env_vars):
        """Default behavior requires full authentication, no bypass."""
        # Arrange
        import auth_server.server as server_module

        with patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", False):
            client = TestClient(server_module.app)

            # Act
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer network-trusted",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert - should NOT return network-trusted response
            if response.status_code == 200:
                assert response.json().get("method") != "network-trusted"

    def test_network_trusted_bypasses_v01_api(self):
        """When enabled, /v0.1/* requests also bypass JWT validation."""
        # Arrange
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("test-api-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer test-api-key",
                    "X-Original-URL": "https://example.com/v0.1/servers",
                },
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["username"] == "network-user"
            assert data["method"] == "network-trusted"

    def test_network_trusted_valid_api_token(self):
        """When REGISTRY_API_TOKEN is set, matching Bearer token is accepted."""
        # Arrange
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("my-secret-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "my-secret-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer my-secret-key",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["method"] == "network-trusted"

    @patch("auth_server.server.get_auth_provider")
    def test_network_trusted_invalid_api_token_falls_through_to_jwt(
        self,
        mock_get_provider,
        auth_env_vars,
    ):
        """A mismatched Bearer now falls through to JWT validation (issue #871).

        Pre-#871 the static-token block returned 403 "Invalid API token". After
        #871 a mismatched bearer is handed to the JWT block. When the JWT
        provider rejects it, the final response does NOT contain the old
        static-token-block detail text.
        """
        # Arrange - provider returns an invalid-token result
        mock_provider = MagicMock()
        mock_provider.validate_token = MagicMock(side_effect=ValueError("Invalid token"))
        mock_get_provider.return_value = mock_provider

        import auth_server.server as server_module

        token_map = _make_legacy_token_map("my-secret-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "my-secret-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer wrong-key",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert: response is no longer the static-token block's 403 with
            # "Invalid API token". The terminal status depends on the JWT
            # provider's failure handling (pre-existing 500 path wraps
            # ValueError), but either way it must NOT be the old 403 body.
            assert response.status_code != 403
            assert "Invalid API token" not in response.json().get("detail", "")

    def test_network_trusted_disabled_when_no_token_configured(self):
        """When REGISTRY_API_TOKEN is empty, static token auth is disabled (falls back to JWT)."""
        # Arrange
        import auth_server.server as server_module

        # Simulate: enabled flag was set to False at startup because token was empty
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", False),
            patch.object(server_module, "REGISTRY_API_TOKEN", ""),
        ):
            client = TestClient(server_module.app)

            # Act
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer anything-goes",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert - should NOT return network-trusted (falls through to JWT validation)
            if response.status_code == 200:
                assert response.json().get("method") != "network-trusted"

    def test_network_trusted_skips_bypass_when_session_cookie_present(self):
        """When session cookie is present, bypass is skipped for normal cookie auth flow."""
        # Arrange
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("test-api-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act - send with session cookie but no Bearer token
            response = client.get(
                "/validate",
                headers={
                    "X-Original-URL": "https://example.com/api/servers",
                    "Cookie": "mcp_gateway_session=some-session-value",
                },
            )

            # Assert - should NOT get 401 from bypass (bypass was skipped)
            # It will fail session validation, but not with the bypass 401 message
            if response.status_code == 401:
                assert "Authorization header required" not in response.json().get("detail", "")

    def test_network_trusted_non_bearer_scheme_falls_through_to_jwt(self):
        """Non-Bearer scheme now falls through to JWT validation (issue #871).

        Before #871 the static-token block returned 401 with detail mentioning
        "Bearer scheme". After #871 the block falls through; the JWT block
        returns 401 with its own detail message.
        """
        # Arrange
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("test-api-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act - send Basic auth instead of Bearer
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Basic dXNlcjpwYXNz",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert: 401 from JWT block, not the old "Bearer scheme" detail
            assert response.status_code == 401
            assert "Bearer scheme" not in response.json()["detail"]

    @patch("auth_server.server.get_auth_provider")
    def test_network_trusted_empty_bearer_falls_through_to_jwt(
        self,
        mock_get_provider,
        auth_env_vars,
    ):
        """Empty Bearer token now falls through to JWT validation (issue #871)."""
        # Arrange - provider rejects empty token
        mock_provider = MagicMock()
        mock_provider.validate_token = MagicMock(side_effect=ValueError("Empty token"))
        mock_get_provider.return_value = mock_provider

        import auth_server.server as server_module

        token_map = _make_legacy_token_map("test-api-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "test-api-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            # Act - send Bearer with empty token
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer ",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert: fall-through → JWT block rejects → no longer the old 403
            # "Invalid API token" detail.
            assert response.status_code != 403
            assert "Invalid API token" not in response.json().get("detail", "")


# =============================================================================
# HELPER UNIT TESTS (issue #871)
# =============================================================================


def _make_legacy_token_map(token: str) -> dict[str, dict]:
    """Build a _STATIC_TOKEN_MAP with just the legacy entry for test helpers."""
    return {
        "legacy": {
            "key_bytes": token.encode("utf-8"),
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "username_override": "network-user",
            "client_id_override": "network-trusted",
        },
    }


class TestCheckRegistryStaticToken:
    """Unit tests for the _check_registry_static_token helper.

    Updated for issue #779 (multi-key map iteration).
    """

    def test_legacy_match_returns_network_trusted_identity(self):
        """Matching bearer for legacy key returns the back-compat identity dict."""
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("expected-token")
        with patch.object(server_module, "_STATIC_TOKEN_MAP", token_map):
            identity = server_module._check_registry_static_token("expected-token")

        assert identity is not None
        assert identity["username"] == "network-user"
        assert identity["client_id"] == "network-trusted"
        assert identity["groups"] == ["mcp-registry-admin"]
        assert "mcp-servers-unrestricted/read" in identity["scopes"]
        assert "mcp-servers-unrestricted/execute" in identity["scopes"]

    def test_mismatch_returns_none(self):
        """Non-matching bearer returns None (not an exception, not a falsy dict)."""
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("expected-token")
        with patch.object(server_module, "_STATIC_TOKEN_MAP", token_map):
            assert server_module._check_registry_static_token("something-else") is None

    def test_empty_bearer_returns_none(self):
        """Empty-string bearer must not match any configured token."""
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("expected-token")
        with patch.object(server_module, "_STATIC_TOKEN_MAP", token_map):
            assert server_module._check_registry_static_token("") is None

    def test_empty_map_returns_none(self):
        """When no keys are configured, any bearer returns None."""
        import auth_server.server as server_module

        with patch.object(server_module, "_STATIC_TOKEN_MAP", {}):
            assert server_module._check_registry_static_token("any-token") is None

    def test_uses_timing_safe_comparison(self):
        """Guard against regression: must use hmac.compare_digest, not ==."""
        import inspect

        import auth_server.server as server_module

        source = inspect.getsource(server_module._check_registry_static_token)
        assert "hmac.compare_digest" in source

    def test_multi_key_match_returns_correct_identity(self):
        """With multiple keys, the matched entry's identity is returned."""
        import auth_server.server as server_module

        token_map = {
            "monitoring": {
                "key_bytes": b"aaaa" * 8,
                "groups": ["mcp-readonly"],
                "scopes": ["mcp-readonly/read"],
            },
            "deploy": {
                "key_bytes": b"bbbb" * 8,
                "groups": ["mcp-registry-admin"],
                "scopes": ["mcp-servers-unrestricted/read"],
            },
        }
        with patch.object(server_module, "_STATIC_TOKEN_MAP", token_map):
            identity = server_module._check_registry_static_token("bbbb" * 8)

        assert identity is not None
        assert identity["username"] == "deploy"
        assert identity["client_id"] == "deploy"
        assert identity["groups"] == ["mcp-registry-admin"]

    def test_multi_key_no_match_returns_none(self):
        """With multiple keys, a non-matching bearer returns None."""
        import auth_server.server as server_module

        token_map = {
            "monitoring": {
                "key_bytes": b"aaaa" * 8,
                "groups": ["mcp-readonly"],
                "scopes": ["mcp-readonly/read"],
            },
        }
        with patch.object(server_module, "_STATIC_TOKEN_MAP", token_map):
            assert server_module._check_registry_static_token("wrong-token") is None

    def test_legacy_username_override_preserved(self):
        """Legacy entry uses username_override / client_id_override for back-compat."""
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("legacy-token")
        with patch.object(server_module, "_STATIC_TOKEN_MAP", token_map):
            identity = server_module._check_registry_static_token("legacy-token")

        assert identity["username"] == "network-user"
        assert identity["client_id"] == "network-trusted"

    def test_non_legacy_key_uses_name_as_username(self):
        """Non-legacy entries use the key name as username and client_id."""
        import auth_server.server as server_module

        token_map = {
            "ci-pipeline": {
                "key_bytes": b"x" * 32,
                "groups": ["mcp-registry-admin"],
                "scopes": ["admin/all"],
            },
        }
        with patch.object(server_module, "_STATIC_TOKEN_MAP", token_map):
            identity = server_module._check_registry_static_token("x" * 32)

        assert identity["username"] == "ci-pipeline"
        assert identity["client_id"] == "ci-pipeline"


# =============================================================================
# JWT / STATIC TOKEN COEXISTENCE TESTS (issue #871)
# =============================================================================


class TestStaticTokenFallthrough:
    """Tests verifying that static-token mode accepts Okta/self-signed JWTs
    as ADDITIONAL credentials, not as replacements. See issue #871.
    """

    @patch("auth_server.server.get_auth_provider")
    def test_valid_jwt_accepted_when_static_token_enabled(
        self,
        mock_get_provider,
        mock_cognito_provider,
        auth_env_vars,
        mock_scope_repository_with_data,
    ):
        """A valid IdP JWT must be accepted on /api/* even when static-token
        mode is on. Pre-#871 the static-token block returned 403 here.
        """
        # Arrange
        mock_get_provider.return_value = mock_cognito_provider

        import auth_server.server as server_module

        token_map = _make_legacy_token_map("static-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "static-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
        ):
            client = TestClient(server_module.app)

            # Act: send a non-matching Bearer that the JWT provider accepts
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer some-valid-idp-jwt",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert: JWT path wins; response is 200 but NOT network-trusted.
            assert response.status_code == 200
            data = response.json()
            assert data["valid"] is True
            assert data["method"] != "network-trusted"
            # The cognito mock returns method="cognito".
            assert data["username"] == "testuser"

    def test_static_token_match_still_returns_network_trusted(self):
        """The happy path for the static token is unchanged by #871."""
        import auth_server.server as server_module

        token_map = _make_legacy_token_map("static-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "static-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer static-key",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["method"] == "network-trusted"
            assert data["client_id"] == "network-trusted"
            assert response.headers["X-Auth-Method"] == "network-trusted"

    @patch("auth_server.server.get_auth_provider")
    def test_mismatched_bearer_and_invalid_jwt_returns_401(
        self,
        mock_get_provider,
        auth_env_vars,
    ):
        """Bearer that matches neither static token nor any valid JWT returns
        401 from the JWT block (previously 403 from static-token block).
        """
        # Arrange - provider rejects the token
        mock_provider = MagicMock()
        mock_provider.validate_token = MagicMock(side_effect=ValueError("Invalid token"))
        mock_get_provider.return_value = mock_provider

        import auth_server.server as server_module

        token_map = _make_legacy_token_map("static-key")
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "static-key"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)

            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer neither-static-nor-jwt",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            # Assert: the terminal rejection is no longer the static-token
            # block's 403 "Invalid API token". Downstream JWT failure
            # semantics (401 on empty / 500 on provider ValueError etc.) are
            # out of scope for #871; we only assert the removal of the old
            # static-token rejection.
            assert "Invalid API token" not in response.json().get("detail", "")


# =============================================================================
# OAUTH2 CALLBACK SESSION STORAGE INTEGRATION TESTS
# =============================================================================


class TestOAuth2CallbackTokenStorage:
    """The browser cookie always carries only an opaque session_id — never
    user data, groups, or tokens. id_token is always persisted server-side
    (encrypted) so SSO logout via id_token_hint keeps working.
    """

    # Cookie ceiling: we sign an opaque 32-byte session_id; the resulting
    # value comfortably fits well under 512 bytes. Lock that in — the whole
    # point of the server-side store is to keep the cookie small.
    COOKIE_SIZE_CEILING_BYTES = 512

    def _call_oauth2_callback(self) -> tuple[dict, str]:
        """Drive the real oauth2_callback endpoint with create_session mocked.

        Returns (kwargs, session_cookie_value) so tests can assert what would
        have landed in the persistent record AND can verify the cookie value
        bounds.
        """
        from auth_server.server import app, signer

        mock_token_data = {
            "access_token": "mock-access-token-value",
            "refresh_token": "mock-refresh-token-value",
            "expires_in": 3600,
            "id_token": "mock-id-token",
        }
        mock_user_info = {
            "sub": "testuser",
            "email": "test@example.com",
            "name": "Test User",
        }
        temp_session_data = {
            "state": "test-state",
            "provider": "github",
            "callback_uri": "http://localhost:8888/oauth2/callback/github",
            "nonce": "test-nonce",
            "code_verifier": "test-code-verifier",
        }
        temp_cookie = signer.dumps(temp_session_data)

        captured: dict = {}

        async def _fake_create_session(**kwargs):
            captured.update(kwargs)
            return "fake-session-id"

        client = TestClient(app, raise_server_exceptions=False)

        with (
            patch(
                "auth_server.server.exchange_code_for_token",
                new_callable=AsyncMock,
                return_value=mock_token_data,
            ),
            patch(
                "auth_server.server.get_user_info",
                new_callable=AsyncMock,
                return_value=mock_user_info,
            ),
            patch(
                "auth_server.server.map_user_info",
                return_value={
                    "username": "testuser",
                    "email": "test@example.com",
                    "name": "Test User",
                    "groups": [],
                },
            ),
            patch("session_store.create_session", _fake_create_session),
        ):
            response = client.get(
                "/oauth2/callback/github",
                params={"code": "test-code", "state": "test-state"},
                cookies={"oauth2_temp_session": temp_cookie},
                follow_redirects=False,
            )

        assert response.status_code == 302
        # Cookie payload is the signed opaque session_id, never user data.
        session_cookie = response.cookies.get("mcp_gateway_session")
        assert session_cookie is not None, "Session cookie not set in response"
        assert signer.loads(session_cookie) == "fake-session-id"

        return captured, session_cookie

    def test_id_token_persisted_in_session_store(self):
        """id_token is always passed to the session store (regression guard
        for SSO logout via id_token_hint). access_token / refresh_token are
        never stored.
        """
        kwargs, _cookie = self._call_oauth2_callback()
        assert kwargs["username"] == "testuser"
        assert kwargs["auth_method"] == "oauth2"
        assert kwargs["id_token"] == "mock-id-token"
        assert "access_token" not in kwargs
        assert "refresh_token" not in kwargs

    def test_session_cookie_stays_well_under_size_ceiling(self):
        """The whole point of the server-side store is a small cookie.

        Lock in the win: the cookie should be a bounded-size signed opaque
        session_id, not a serialized payload of user/groups/id_token. If a
        future change reintroduces inline user data, this test catches it.
        """
        _kwargs, cookie = self._call_oauth2_callback()
        assert len(cookie) < self.COOKIE_SIZE_CEILING_BYTES, (
            f"Session cookie is {len(cookie)} bytes; expected < "
            f"{self.COOKIE_SIZE_CEILING_BYTES}. The server-side session store "
            "should keep this small."
        )


class TestOAuth2CallbackIdTokenVerification:
    """The OAuth2 callback must verify a present id_token before trusting its
    claims. A forged/tampered id_token (verification failure) denies the login
    and never persists attacker-controlled identity or group claims.
    """

    def _drive_callback(self, provider: str, validate_side_effect):
        """Drive the callback for a JWKS provider with a mocked provider whose
        validate_id_token exhibits the given behaviour. Returns
        (response, create_session_called, captured_kwargs).
        """
        from auth_server import server as srv
        from auth_server.server import app, signer

        mock_token_data = {
            "access_token": "mock-access-token-value",
            "id_token": "attacker-supplied-id-token",
        }
        temp_session_data = {
            "state": "test-state",
            "provider": provider,
            "callback_uri": f"http://localhost:8888/oauth2/callback/{provider}",
            "nonce": "test-nonce",
            "code_verifier": "test-code-verifier",
        }
        temp_cookie = signer.dumps(temp_session_data)

        captured: dict = {}
        create_called = {"value": False}

        async def _fake_create_session(**kwargs):
            create_called["value"] = True
            captured.update(kwargs)
            return "fake-session-id"

        fake_provider = MagicMock()
        fake_provider.validate_id_token.side_effect = validate_side_effect

        patched_config = dict(srv.OAUTH2_CONFIG)
        patched_config["providers"] = dict(patched_config.get("providers", {}))
        patched_config["providers"][provider] = {
            "client_id": "gateway-web",
            "user_info_url": "https://idp.example.com/userinfo",
            "username_claim": "sub",
            "email_claim": "email",
            "name_claim": "name",
        }

        client = TestClient(app, raise_server_exceptions=False)

        with (
            patch.object(srv, "OAUTH2_CONFIG", patched_config),
            patch(
                "auth_server.server.exchange_code_for_token",
                new_callable=AsyncMock,
                return_value=mock_token_data,
            ),
            patch("auth_server.server.get_auth_provider", return_value=fake_provider),
            patch("session_store.create_session", _fake_create_session),
        ):
            response = client.get(
                f"/oauth2/callback/{provider}",
                params={"code": "test-code", "state": "test-state"},
                cookies={"oauth2_temp_session": temp_cookie},
                follow_redirects=False,
            )

        return response, create_called["value"], captured

    @pytest.mark.parametrize("provider", ["keycloak", "entra", "okta", "pingfederate"])
    def test_forged_id_token_denies_login_and_never_persists_groups(self, provider):
        """A present id_token that fails verification must fail closed: the
        session is never created, so forged group/identity claims cannot reach
        the session store.
        """
        # Import the error via the SAME module path the server uses at runtime
        # (auth_server/ is on sys.path, so the server imports `providers.base`).
        # Using a different path (`auth_server.providers.base`) would create a
        # distinct class object that the server's `except` would not catch.
        from providers.base import IdTokenVerificationError

        forged_claims_groups = ["mcp-registry-admin"]

        def _raise(_token, expected_nonce=None):
            # Simulate a forged token whose (unverified) claims assert admin;
            # verification rejects it before any claim is used.
            raise IdTokenVerificationError(f"forged token asserting {forged_claims_groups}")

        response, create_called, captured = self._drive_callback(provider, _raise)

        # Login denied (not a 302 success redirect to the registry).
        assert response.status_code == 401
        # Fail closed: no session persisted at all.
        assert create_called is False
        assert captured == {}

    def test_verified_id_token_allows_login(self):
        """A verified id_token proceeds to session creation with its claims."""
        verified = {
            "sub": "alice",
            "preferred_username": "alice",
            "email": "alice@example.com",
            "name": "Alice",
            "groups": ["mcp-registry-user"],
        }
        response, create_called, captured = self._drive_callback(
            "keycloak", lambda _t, expected_nonce=None: verified
        )

        assert response.status_code == 302
        assert create_called is True
        assert captured["username"] == "alice"


class TestOAuth2PkceAndNonce:
    """The OAuth2 flow must use PKCE (S256) and an OIDC nonce.

    - The authorization request carries a code_challenge (S256) and a nonce.
    - The callback fails closed when the PKCE code_verifier is absent.
    - The callback rejects an id_token whose nonce does not match the value
      bound to this login (replay/injection), even if the signature is valid.
    - The happy path (matching nonce + verifier present) succeeds and forwards
      the verifier to the token exchange.
    """

    def test_pkce_code_challenge_matches_rfc7636(self):
        """S256 challenge is BASE64URL(SHA256(verifier)) with padding stripped."""
        import base64
        import hashlib

        from auth_server.server import _pkce_code_challenge

        verifier = "a-high-entropy-code-verifier-value"
        expected = (
            base64.urlsafe_b64encode(hashlib.sha256(verifier.encode("ascii")).digest())
            .rstrip(b"=")
            .decode("ascii")
        )
        assert _pkce_code_challenge(verifier) == expected
        assert "=" not in _pkce_code_challenge(verifier)

    def test_authorization_request_includes_pkce_and_nonce(self):
        """/oauth2/login redirects with code_challenge=S256 and a nonce, and
        persists the verifier + nonce in the signed flow cookie."""
        import urllib.parse

        from auth_server import server as srv
        from auth_server.server import app, signer

        patched_config = dict(srv.OAUTH2_CONFIG)
        patched_config["providers"] = dict(patched_config.get("providers", {}))
        patched_config["providers"]["keycloak"] = {
            "enabled": True,
            "client_id": "gateway-web",
            "response_type": "code",
            "scopes": ["openid", "email", "profile"],
            "auth_url": "https://idp.example.com/authorize",
        }

        client = TestClient(app, raise_server_exceptions=False)
        with patch.object(srv, "OAUTH2_CONFIG", patched_config):
            response = client.get("/oauth2/login/keycloak", follow_redirects=False)

        assert response.status_code == 302
        location = response.headers["location"]
        query = dict(urllib.parse.parse_qsl(urllib.parse.urlparse(location).query))
        assert query.get("code_challenge_method") == "S256"
        assert query.get("code_challenge"), "code_challenge missing from auth request"
        assert query.get("nonce"), "nonce missing from auth request"

        # The verifier and nonce are persisted (integrity-protected) in the cookie.
        temp_cookie = response.cookies.get("oauth2_temp_session")
        assert temp_cookie is not None
        session_data = signer.loads(temp_cookie)
        assert session_data.get("code_verifier"), "code_verifier not persisted"
        assert session_data.get("nonce") == query.get("nonce")
        # The challenge on the wire is derived from the persisted verifier.
        assert srv._pkce_code_challenge(session_data["code_verifier"]) == query["code_challenge"]

    def _drive_callback(
        self,
        provider: str,
        temp_session_data: dict,
        validate_side_effect,
    ):
        """Drive the callback with a mocked provider. Returns
        (response, create_session_called, captured_kwargs, exchange_mock)."""
        from auth_server import server as srv
        from auth_server.server import app, signer

        mock_token_data = {
            "access_token": "mock-access-token-value",
            "id_token": "attacker-or-valid-id-token",
        }
        temp_cookie = signer.dumps(temp_session_data)

        captured: dict = {}
        create_called = {"value": False}

        async def _fake_create_session(**kwargs):
            create_called["value"] = True
            captured.update(kwargs)
            return "fake-session-id"

        fake_provider = MagicMock()
        fake_provider.validate_id_token.side_effect = validate_side_effect

        patched_config = dict(srv.OAUTH2_CONFIG)
        patched_config["providers"] = dict(patched_config.get("providers", {}))
        patched_config["providers"][provider] = {
            "client_id": "gateway-web",
            "user_info_url": "https://idp.example.com/userinfo",
            "username_claim": "sub",
            "email_claim": "email",
            "name_claim": "name",
        }

        exchange_mock = AsyncMock(return_value=mock_token_data)
        client = TestClient(app, raise_server_exceptions=False)
        with (
            patch.object(srv, "OAUTH2_CONFIG", patched_config),
            patch("auth_server.server.exchange_code_for_token", exchange_mock),
            patch("auth_server.server.get_auth_provider", return_value=fake_provider),
            patch("session_store.create_session", _fake_create_session),
        ):
            response = client.get(
                f"/oauth2/callback/{provider}",
                params={"code": "test-code", "state": "test-state"},
                cookies={"oauth2_temp_session": temp_cookie},
                follow_redirects=False,
            )

        return response, create_called["value"], captured, exchange_mock

    def test_callback_denies_when_code_verifier_absent(self):
        """A login flow with no persisted PKCE verifier fails closed."""
        temp_session_data = {
            "state": "test-state",
            "provider": "keycloak",
            "callback_uri": "http://localhost:8888/oauth2/callback/keycloak",
            "nonce": "test-nonce",
            # code_verifier deliberately absent
        }
        response, create_called, captured, exchange_mock = self._drive_callback(
            "keycloak",
            temp_session_data,
            lambda _t, expected_nonce=None: {"sub": "alice"},
        )

        assert response.status_code == 400
        assert create_called is False
        assert captured == {}
        # Fail closed BEFORE any token exchange.
        exchange_mock.assert_not_called()

    def test_callback_rejects_id_token_with_nonce_mismatch(self):
        """A validly signed id_token whose nonce != stored is rejected (replay)."""
        from providers.base import IdTokenVerificationError

        temp_session_data = {
            "state": "test-state",
            "provider": "keycloak",
            "callback_uri": "http://localhost:8888/oauth2/callback/keycloak",
            "nonce": "expected-nonce",
            "code_verifier": "test-code-verifier",
        }

        def _verify(_token, expected_nonce=None):
            # Faithful provider: enforces the nonce it was handed. The injected
            # token carries a DIFFERENT nonce, so verification fails closed.
            token_nonce = "attacker-nonce"
            if expected_nonce is not None and token_nonce != expected_nonce:
                raise IdTokenVerificationError("nonce mismatch")
            return {"sub": "alice", "groups": ["mcp-registry-admin"]}

        response, create_called, captured, _exchange = self._drive_callback(
            "keycloak", temp_session_data, _verify
        )

        assert response.status_code == 401
        assert create_called is False
        assert captured == {}

    def test_callback_happy_path_forwards_verifier_and_matches_nonce(self):
        """Matching nonce + present verifier: login succeeds and the verifier is
        forwarded to the token exchange."""
        verified = {
            "sub": "alice",
            "preferred_username": "alice",
            "email": "alice@example.com",
            "name": "Alice",
            "groups": ["mcp-registry-user"],
        }
        temp_session_data = {
            "state": "test-state",
            "provider": "keycloak",
            "callback_uri": "http://localhost:8888/oauth2/callback/keycloak",
            "nonce": "expected-nonce",
            "code_verifier": "the-code-verifier",
        }

        captured_nonce = {}

        def _verify(_token, expected_nonce=None):
            captured_nonce["value"] = expected_nonce
            return verified

        response, create_called, captured, exchange_mock = self._drive_callback(
            "keycloak", temp_session_data, _verify
        )

        assert response.status_code == 302
        assert create_called is True
        assert captured["username"] == "alice"
        # The stored nonce reached the verifier.
        assert captured_nonce["value"] == "expected-nonce"
        # The stored verifier was forwarded to the token exchange (PKCE proof).
        _args, kwargs = exchange_mock.call_args
        assert kwargs.get("code_verifier") == "the-code-verifier"


# =============================================================================
# MULTI-KEY STATIC TOKEN PARSER TESTS (issue #779)
# =============================================================================


class TestParseRegistryApiKeys:
    """Unit tests for _parse_registry_api_keys config parser."""

    def test_empty_string_returns_empty_list(self):
        """Empty raw string produces no entries."""
        import auth_server.server as server_module

        result = server_module._parse_registry_api_keys("")
        assert result == []

    def test_valid_single_entry(self):
        """A single valid entry parses correctly."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "deploy-pipeline": {
                    "key": "a" * 32,
                    "groups": ["mcp-registry-admin"],
                }
            }
        )
        result = server_module._parse_registry_api_keys(raw)
        assert len(result) == 1
        assert result[0].name == "deploy-pipeline"
        assert result[0].key == "a" * 32
        assert result[0].groups == ["mcp-registry-admin"]

    def test_valid_multiple_entries(self):
        """Multiple valid entries parse correctly."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "monitoring": {"key": "m" * 32, "groups": ["mcp-readonly"]},
                "deploy": {"key": "d" * 32, "groups": ["mcp-registry-admin"]},
            }
        )
        result = server_module._parse_registry_api_keys(raw)
        assert len(result) == 2
        names = {e.name for e in result}
        assert names == {"monitoring", "deploy"}

    def test_malformed_json_raises(self):
        """Non-JSON input raises ValueError."""
        import auth_server.server as server_module

        with pytest.raises(ValueError, match="not valid JSON"):
            server_module._parse_registry_api_keys("{bad json")

    def test_non_object_json_raises(self):
        """A JSON array (not object) raises ValueError."""
        import auth_server.server as server_module

        with pytest.raises(ValueError, match="must be a JSON object"):
            server_module._parse_registry_api_keys('[{"key":"abc"}]')

    def test_reserved_name_legacy_raises(self):
        """The name 'legacy' is reserved and must be rejected."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "legacy": {"key": "x" * 32, "groups": ["admin"]},
            }
        )
        with pytest.raises(ValueError, match="reserved"):
            server_module._parse_registry_api_keys(raw)

    def test_reserved_name_network_user_raises(self):
        """The name 'network-user' is reserved."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "network-user": {"key": "x" * 32, "groups": ["admin"]},
            }
        )
        with pytest.raises(ValueError, match="reserved"):
            server_module._parse_registry_api_keys(raw)

    def test_reserved_name_network_trusted_raises(self):
        """The name 'network-trusted' is reserved."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "network-trusted": {"key": "x" * 32, "groups": ["admin"]},
            }
        )
        with pytest.raises(ValueError, match="reserved"):
            server_module._parse_registry_api_keys(raw)

    def test_key_too_short_raises(self):
        """Key shorter than 32 chars raises."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "short-key": {"key": "abc", "groups": ["admin"]},
            }
        )
        with pytest.raises(ValueError, match="Invalid entry"):
            server_module._parse_registry_api_keys(raw)

    def test_empty_groups_raises(self):
        """Empty groups list raises."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "no-groups": {"key": "x" * 32, "groups": []},
            }
        )
        with pytest.raises(ValueError, match="Invalid entry"):
            server_module._parse_registry_api_keys(raw)

    def test_duplicate_key_value_raises(self):
        """Two entries with the same key value raises."""
        import json

        import auth_server.server as server_module

        same_key = "k" * 32
        raw = json.dumps(
            {
                "entry-a": {"key": same_key, "groups": ["g1"]},
                "entry-b": {"key": same_key, "groups": ["g2"]},
            }
        )
        with pytest.raises(ValueError, match="Duplicate key value"):
            server_module._parse_registry_api_keys(raw)

    def test_invalid_name_format_raises(self):
        """Name with uppercase or special chars raises."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "Invalid-Name!": {"key": "x" * 32, "groups": ["admin"]},
            }
        )
        with pytest.raises(ValueError, match="Invalid"):
            server_module._parse_registry_api_keys(raw)

    def test_entry_not_object_raises(self):
        """Entry value that is not a dict raises."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "bad-entry": "just-a-string",
            }
        )
        with pytest.raises(ValueError, match="must be an object"):
            server_module._parse_registry_api_keys(raw)

    def test_empty_object_returns_empty_list(self):
        """An empty JSON object '{}' returns an empty list."""
        import auth_server.server as server_module

        result = server_module._parse_registry_api_keys("{}")
        assert result == []


# =============================================================================
# MULTI-KEY BUILD TOKEN MAP TESTS (issue #779)
# =============================================================================


class TestBuildStaticTokenMap:
    """Unit tests for _build_static_token_map startup builder."""

    @pytest.mark.asyncio
    async def test_disabled_flag_does_nothing(self):
        """When REGISTRY_STATIC_TOKEN_AUTH_ENABLED is False, map stays empty."""
        import auth_server.server as server_module

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", False),
            patch.object(server_module, "_STATIC_TOKEN_MAP", {}),
        ):
            await server_module._build_static_token_map()
            assert server_module._STATIC_TOKEN_MAP == {}

    @pytest.mark.asyncio
    async def test_legacy_only_builds_single_entry(self):
        """With only REGISTRY_API_TOKEN set (no REGISTRY_API_KEYS), map has one legacy entry."""
        import auth_server.server as server_module

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "t" * 32),
            patch.object(server_module, "_REGISTRY_API_KEYS_RAW", ""),
            patch.object(server_module, "_STATIC_TOKEN_MAP", {}),
        ):
            await server_module._build_static_token_map()
            assert "legacy" in server_module._STATIC_TOKEN_MAP
            assert len(server_module._STATIC_TOKEN_MAP) == 1
            legacy = server_module._STATIC_TOKEN_MAP["legacy"]
            assert legacy["username_override"] == "network-user"
            assert legacy["client_id_override"] == "network-trusted"

    @pytest.mark.asyncio
    async def test_bad_json_disables_feature(self):
        """Malformed REGISTRY_API_KEYS disables static-token auth (fail-closed)."""
        import auth_server.server as server_module

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", ""),
            patch.object(server_module, "_REGISTRY_API_KEYS_RAW", "{bad json"),
            patch.object(server_module, "_STATIC_TOKEN_MAP", {}),
        ):
            await server_module._build_static_token_map()
            assert server_module.REGISTRY_STATIC_TOKEN_AUTH_ENABLED is False

    @pytest.mark.asyncio
    async def test_valid_keys_plus_legacy_merged(self):
        """Both REGISTRY_API_KEYS and REGISTRY_API_TOKEN produce merged map."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "monitoring": {"key": "m" * 32, "groups": ["mcp-readonly"]},
            }
        )

        mock_repo = AsyncMock()
        mock_repo.get_group_mappings.return_value = ["mcp-readonly/read"]
        mock_repo.get_group_mappings_bulk.return_value = ["mcp-readonly/read"]

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", "t" * 32),
            patch.object(server_module, "_REGISTRY_API_KEYS_RAW", raw),
            patch.object(server_module, "_STATIC_TOKEN_MAP", {}),
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_repo,
            ),
        ):
            await server_module._build_static_token_map()
            assert "monitoring" in server_module._STATIC_TOKEN_MAP
            assert "legacy" in server_module._STATIC_TOKEN_MAP
            assert len(server_module._STATIC_TOKEN_MAP) == 2

    @pytest.mark.asyncio
    async def test_zero_keys_warns_but_stays_enabled(self):
        """Empty REGISTRY_API_KEYS and empty REGISTRY_API_TOKEN logs warning."""
        import auth_server.server as server_module

        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "REGISTRY_API_TOKEN", ""),
            patch.object(server_module, "_REGISTRY_API_KEYS_RAW", ""),
            patch.object(server_module, "_STATIC_TOKEN_MAP", {}),
        ):
            await server_module._build_static_token_map()
            assert server_module._STATIC_TOKEN_MAP == {}
            # Feature stays enabled (callers just fall through to JWT)
            assert server_module.REGISTRY_STATIC_TOKEN_AUTH_ENABLED is True


# =============================================================================
# MULTI-KEY VALIDATE INTEGRATION TESTS (issue #779)
# =============================================================================


class TestMultiKeyStaticTokenValidate:
    """Integration tests for multi-key static token through /validate."""

    def test_named_key_returns_key_name_as_username(self):
        """A named key match returns the key name as X-Username."""
        import auth_server.server as server_module

        token_map = {
            "ci-runner": {
                "key_bytes": ("c" * 32).encode("utf-8"),
                "groups": ["mcp-registry-admin"],
                "scopes": ["mcp-servers-unrestricted/read"],
            },
        }
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {'c' * 32}",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["username"] == "ci-runner"
            assert data["client_id"] == "ci-runner"
            assert data["method"] == "network-trusted"
            assert response.headers["X-Username"] == "ci-runner"

    def test_readonly_key_gets_limited_scopes(self):
        """A read-only key gets only the scopes configured for its groups."""
        import auth_server.server as server_module

        token_map = {
            "readonly-monitor": {
                "key_bytes": ("r" * 32).encode("utf-8"),
                "groups": ["mcp-readonly"],
                "scopes": ["mcp-readonly/read"],
            },
        }
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {'r' * 32}",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["scopes"] == ["mcp-readonly/read"]
            assert data["groups"] == ["mcp-readonly"]

    def test_key_with_empty_scopes_still_matches(self):
        """A key whose groups map to no scopes still matches (but will 403 at registry)."""
        import auth_server.server as server_module

        token_map = {
            "empty-scope-key": {
                "key_bytes": ("e" * 32).encode("utf-8"),
                "groups": ["ghost-group"],
                "scopes": [],
            },
        }
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {'e' * 32}",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

            assert response.status_code == 200
            data = response.json()
            assert data["scopes"] == []
            assert data["username"] == "empty-scope-key"


# =============================================================================
# MCP PROXY RESPONSE-HEADER ALLOWLIST TESTS
# =============================================================================
#
# The /mcp-proxy/{server_name} hop introduced in Issue #1026 buffers the
# upstream MCP response and re-emits it via Starlette. Two regressions live
# on the same code path:
#
#   1. Earlier revisions silently dropped every upstream response header,
#      including Mcp-Session-Id which streamable-http MCP servers emit
#      during initialize and which the client requires on follow-up
#      requests. These tests pin the three return-site branches so that
#      cannot recur.
#
#   2. The auth-server sits on a trust boundary in front of arbitrary
#      upstream MCP servers. Forwarding every header (denylist) lets an
#      upstream dictate Set-Cookie, Location, Strict-Transport-Security,
#      Content-Security-Policy, Access-Control-Allow-*, Server, etc. The
#      handler must instead use an allowlist (_FORWARDED_RESPONSE_HEADERS)
#      that opts in the small set of MCP-essential headers only. These
#      tests pin the allowlist and assert dangerous headers are dropped.


def _build_mock_upstream_response(
    status_code: int,
    headers: dict[str, str],
    body: bytes,
) -> MagicMock:
    """Build a MagicMock that mimics the surface of httpx.Response used by
    auth_server.server.mcp_proxy: aiter_bytes(chunk_size=...), status_code,
    and headers.
    """
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.headers = headers

    async def _aiter(chunk_size: int = 64 * 1024):
        yield body

    mock_resp.aiter_bytes = _aiter
    return mock_resp


def _patch_httpx_async_client(mock_upstream_response: MagicMock):
    """Return a patch context manager that replaces
    ``auth_server.server.httpx.AsyncClient`` so the ``async with
    httpx.AsyncClient(...) as client: async with client.stream(...) as
    response:`` chain yields ``mock_upstream_response``.
    """
    mock_stream_cm = AsyncMock()
    mock_stream_cm.__aenter__ = AsyncMock(return_value=mock_upstream_response)
    mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(return_value=mock_stream_cm)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    return patch("auth_server.server.httpx.AsyncClient", return_value=mock_client)


class TestForwardedResponseHeadersAllowlist:
    """Tests for ``_select_forwarded_response_headers`` -- the function that
    enforces the upstream-response-header allowlist.

    Two responsibilities under test:
      1. Allowlisted headers (Mcp-Session-Id, X-Mcp-Session-Id, WWW-
         Authenticate, Retry-After) survive, case-insensitively.
      2. Everything else, including framing headers and headers in the
         review specifically called out as security-sensitive
         (Set-Cookie, Location, HSTS, CSP, ACAO, Server), is dropped.
    """

    def test_allowlist_membership_matches_constant(self):
        """Pin the allowlist contents. Adding to the set is a
        security-relevant change; this test forces the PR author to
        update it intentionally and the reviewer to re-evaluate.
        """
        from auth_server.server import _FORWARDED_RESPONSE_HEADERS

        assert _FORWARDED_RESPONSE_HEADERS == frozenset(
            {
                "mcp-session-id",
                "x-mcp-session-id",
                "www-authenticate",
                "retry-after",
            }
        )

    def test_keeps_mcp_session_id(self):
        """Mcp-Session-Id is the original symptom of issue #1096 -- it
        must survive the allowlist or streamable-http MCP clients lose
        their session.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {
            "content-type": "application/json",
            "mcp-session-id": "sess-abc-123",
        }

        forwarded = _select_forwarded_response_headers(upstream)

        assert forwarded == {"mcp-session-id": "sess-abc-123"}

    def test_keeps_x_mcp_session_id_legacy_alias(self):
        """Some MCP servers emit the legacy X-Mcp-Session-Id alias; the
        allowlist must include it for the same session-management reason.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {"x-mcp-session-id": "sess-legacy-007"}

        assert _select_forwarded_response_headers(upstream) == {
            "x-mcp-session-id": "sess-legacy-007",
        }

    def test_keeps_www_authenticate_for_prm_flow(self):
        """PR #1115 introduced the OAuth PRM / resource-metadata 401
        flow that depends on WWW-Authenticate reaching the MCP client.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {
            "www-authenticate": 'Bearer resource_metadata="https://idp.example/.well-known/oauth-protected-resource"'
        }

        assert _select_forwarded_response_headers(upstream) == upstream

    def test_keeps_retry_after_for_backoff(self):
        """Retry-After must be forwarded so MCP clients honor upstream
        429/503 backoff instead of retry-storming the upstream.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {"retry-after": "30"}

        assert _select_forwarded_response_headers(upstream) == {"retry-after": "30"}

    def test_match_is_case_insensitive_for_allowlist(self):
        """HTTP header names are case-insensitive (RFC 9110 §5.1). The
        upstream may emit any casing; the allowlist match must work
        regardless, and the helper must preserve the wire casing on
        return so the client sees what the MCP server actually sent.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {
            "Mcp-Session-Id": "sess-xyz",
            "WWW-Authenticate": "Bearer realm=mcp",
            "Retry-After": "5",
        }

        forwarded = _select_forwarded_response_headers(upstream)

        assert forwarded["Mcp-Session-Id"] == "sess-xyz"
        assert forwarded["WWW-Authenticate"] == "Bearer realm=mcp"
        assert forwarded["Retry-After"] == "5"

    def test_drops_set_cookie_and_location_regression(self):
        """Regression for review feedback on PR #1097: an upstream MCP
        server must not be able to set cookies or redirect targets on
        the gateway response. These are the headers the reviewer
        specifically called out -- they are the canonical "trust
        boundary leak" pair and the most likely abuse vector.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {
            "Set-Cookie": "session=evil; Path=/; HttpOnly",
            "Location": "https://attacker.example/oauth/authorize",
            "Mcp-Session-Id": "sess-keep-me",
        }

        forwarded = _select_forwarded_response_headers(upstream)

        assert "Set-Cookie" not in forwarded
        assert "Location" not in forwarded
        # And the allowlisted header is still there so the test is not
        # vacuously asserting an empty dict.
        assert forwarded["Mcp-Session-Id"] == "sess-keep-me"

    def test_drops_security_policy_headers(self):
        """Other reviewer-cited headers -- HSTS, CSP, ACAO, Server -- let
        an upstream override the gateway's security posture. Drop them.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {
            "Strict-Transport-Security": "max-age=0",
            "Content-Security-Policy": "default-src *",
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Credentials": "true",
            "Server": "evil-upstream/1.0",
        }

        assert _select_forwarded_response_headers(upstream) == {}

    def test_drops_framing_headers(self):
        """Framing / encoding headers describe the upstream wire
        message; Starlette must recompute them for the body it actually
        serializes (which may be httpx-decoded when the upstream used
        Content-Encoding: gzip). They are not on the allowlist, so the
        allowlist drops them for free.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {
            "content-length": "1024",
            "content-encoding": "gzip",
            "transfer-encoding": "chunked",
            "connection": "keep-alive",
        }

        assert _select_forwarded_response_headers(upstream) == {}

    def test_drops_arbitrary_x_headers(self):
        """The reviewer wants opt-in, not opt-out. Custom X-* headers
        from the upstream must not survive even if they look innocuous.
        """
        from auth_server.server import _select_forwarded_response_headers

        upstream = {
            "X-Frame-Options": "DENY",
            "X-Custom-Tracking": "abc123",
            "X-Powered-By": "MCP/0.1",
        }

        assert _select_forwarded_response_headers(upstream) == {}

    def test_empty_input_returns_empty(self):
        """Defensive: an empty upstream-header dict must produce an
        empty forwarded dict, never a None or an error.
        """
        from auth_server.server import _select_forwarded_response_headers

        assert _select_forwarded_response_headers({}) == {}


def _mcp_proxy_token_headers(
    server_name: str = "office-docs",
    upstream_url: str = "https://upstream.example/mcp",
    scopes: list[str] | None = None,
) -> dict:
    """Build the X-Internal-Token nginx would forward to /mcp-proxy.

    The verify_mcp_proxy_token dependency reads SECRET_KEY from the env (set
    process-wide by the test conftest), and mint_mcp_proxy_token reads the same,
    so a token minted here verifies in-process. Identity/scopes/upstream are read
    from these claims; the handler ignores the inbound X-User/X-Scopes/X-Upstream-Url.

    The handler now re-authorizes the forwarded body against these scopes, so the
    default carries a wildcard scope (``admin:all``); pair it with
    ``_patch_scope_repo_allow_all`` so the re-auth passes. Pass an explicit
    ``scopes`` (e.g. ``[]``) to exercise the forwarded-body denial path.
    """
    from auth_server.internal_request_token import mint_mcp_proxy_token

    token = mint_mcp_proxy_token(
        subject="test-user",
        scopes=["admin:all"] if scopes is None else scopes,
        server_name=server_name,
        upstream_url=upstream_url,
    )
    return {"X-Internal-Token": token}


def _obo_ingress_jwt(sub: str = "test-user") -> str:
    """Build a decodable ingress JWT whose principal matches the internal token.

    The obo branch binds the OBO subject token to the /validate-authorized
    principal (_obo_subject_matches_principal): the subject token's
    ``preferred_username``/``sub`` must equal the internal mcp-proxy token's
    ``sub``. _mcp_proxy_token_headers mints the internal token with sub="test-user",
    so the raw ingress JWT the client presents must carry the same principal. The
    signature is irrelevant here (the binding check does an unverified decode; the
    real signature was already checked by /validate), so a throwaway secret is fine.
    """
    import jwt as _jwt

    return _jwt.encode({"sub": sub, "preferred_username": sub}, "test-secret", algorithm="HS256")


def _patch_scope_repo_allow_all():
    """Patch get_scope_repository so ``admin:all`` grants any server/method.

    The /mcp-proxy handler re-authorizes the forwarded body via
    validate_server_tool_access, which consults the scope repository. These
    header-passthrough tests care about response-header handling, not the
    allowlist itself, so they run with a wildcard-granting repository and an
    ``admin:all`` token.
    """
    repo = AsyncMock()

    async def _get_server_scopes(scope_name: str):
        if scope_name == "admin:all":
            return [{"server": "*", "methods": ["*"], "tools": ["*"]}]
        return []

    repo.get_server_scopes.side_effect = _get_server_scopes
    return patch("auth_server.server.get_scope_repository", return_value=repo)


class TestMcpProxyEndpointHeaderPassthrough:
    """End-to-end-style tests that drive the FastAPI /mcp-proxy/... route
    with a mocked httpx upstream and assert the client-visible response
    headers include the upstream Mcp-Session-Id.

    Each request carries a valid X-Internal-Token (minted as nginx's /validate
    hop would) so it passes the verify_mcp_proxy_token dependency; the upstream
    URL is bound in the token, not the inbound header.
    """

    def test_session_id_forwarded_on_passthrough_branch(self):
        """SSE / non-tools-list response: the early-return branch at
        ``if not should_filter`` must include upstream Mcp-Session-Id on
        the Starlette Response sent to the client.
        """
        import auth_server.server as server_module

        upstream_resp = _build_mock_upstream_response(
            status_code=200,
            headers={
                "content-type": "text/event-stream",
                "mcp-session-id": "sess-passthrough-001",
                "content-length": "12",
            },
            body=b"data: hello\n\n",
        )

        with (
            _patch_httpx_async_client(upstream_resp),
            _patch_scope_repo_allow_all(),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/office-docs",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                headers=_mcp_proxy_token_headers(),
            )

        assert response.status_code == 200
        assert response.headers.get("mcp-session-id") == "sess-passthrough-001"
        # Framing header must not survive: Starlette sets its own.
        assert response.headers.get("content-length") != "12"

    def test_session_id_forwarded_on_filtered_tools_list_branch(self):
        """tools/list with JSON body and filter enabled: the filtered
        JSONResponse branch must also include upstream Mcp-Session-Id.
        """
        import auth_server.server as server_module

        upstream_body = b'{"jsonrpc":"2.0","id":1,"result":{"tools":[{"name":"t1"},{"name":"t2"}]}}'
        upstream_resp = _build_mock_upstream_response(
            status_code=200,
            headers={
                "content-type": "application/json",
                "mcp-session-id": "sess-filtered-002",
            },
            body=upstream_body,
        )

        async def _no_filter(server_name, user_scopes, tools):
            return tools

        with (
            _patch_httpx_async_client(upstream_resp),
            _patch_scope_repo_allow_all(),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=True),
            patch.object(
                server_module,
                "filter_tools_list_response",
                side_effect=_no_filter,
            ),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/office-docs",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers=_mcp_proxy_token_headers(),
            )

        assert response.status_code == 200
        assert response.headers.get("mcp-session-id") == "sess-filtered-002"

    def test_session_id_forwarded_on_tools_list_non_json_fallback(self):
        """tools/list whose upstream body fails JSON parsing falls into
        the ``_safe_parse_body`` JSONResponse branch; that branch must
        still forward upstream Mcp-Session-Id.
        """
        import auth_server.server as server_module

        upstream_resp = _build_mock_upstream_response(
            status_code=200,
            headers={
                "content-type": "application/json",
                "mcp-session-id": "sess-malformed-003",
            },
            body=b"not-json-at-all",
        )

        with (
            _patch_httpx_async_client(upstream_resp),
            _patch_scope_repo_allow_all(),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=True),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/office-docs",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers=_mcp_proxy_token_headers(),
            )

        assert response.status_code == 200
        assert response.headers.get("mcp-session-id") == "sess-malformed-003"

    def test_session_id_preserved_when_filter_flag_is_false(self):
        """Regression guard for the workaround question: setting
        MCP_TOOLS_LIST_FILTER_ENABLED=false alone is NOT enough --
        nginx still routes through this endpoint, and the response
        builder must forward Mcp-Session-Id regardless of the flag.
        """
        import auth_server.server as server_module

        upstream_resp = _build_mock_upstream_response(
            status_code=200,
            headers={
                "content-type": "application/json",
                "mcp-session-id": "sess-flag-off-004",
            },
            body=b'{"jsonrpc":"2.0","id":1,"result":{"tools":[]}}',
        )

        with (
            _patch_httpx_async_client(upstream_resp),
            _patch_scope_repo_allow_all(),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/office-docs",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/list"},
                headers=_mcp_proxy_token_headers(),
            )

        assert response.status_code == 200
        assert response.headers.get("mcp-session-id") == "sess-flag-off-004"

    def test_missing_internal_token_rejected(self):
        """A request with no X-Internal-Token is rejected by the
        verify_mcp_proxy_token dependency (default enforce) BEFORE the handler
        runs.
        """
        import auth_server.server as server_module

        client = TestClient(server_module.app)
        response = client.post(
            "/mcp-proxy/office-docs",
            json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
            headers={"X-Upstream-Url": "https://attacker.example/collect"},
        )

        assert response.status_code == 401

    def test_forwarded_body_reauthorized_denies_unscoped_caller(self):
        """A token with no scopes is denied at the proxy hop before any
        outbound call -- the forwarded body is re-authorized here, not only
        at /validate on a separately-captured copy (TM-15).
        """
        import auth_server.server as server_module

        # No upstream patch: if the guard fails open we would attempt the
        # outbound call and this test would surface a different failure.
        with (
            _patch_scope_repo_allow_all(),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/office-docs",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "privileged-tool"},
                },
                headers=_mcp_proxy_token_headers(scopes=[]),
            )

        assert response.status_code == 403

    def test_egress_consent_emits_iserror_baseline_with_connect_url(self):
        """DEFAULT consent delivery: when egress is on and the user has no token,
        a tools/call gets a SUCCESSFUL JSON-RPC result with isError=true whose
        text carries the connect URL. This baseline works on every MCP client
        (no -32042 support needed). Elicitation (-32042) is opt-in via
        egress_consent_use_elicitation (covered separately below).
        """
        import auth_server.server as server_module

        vend = {
            "consent_required": True,
            "connect_url": "https://gw.example.com/oauth2/egress/connect?server=%2Fgithub",
            "request_state": "AEAD-blob",
            "provider": "github",
        }

        async def _consent_vend(token, server):
            return vend

        with (
            patch.object(server_module.settings, "egress_auth_enabled", True),
            patch.object(server_module.settings, "egress_consent_use_elicitation", False),
            patch.object(server_module, "_vend_egress_token", _consent_vend),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/github",
                json={"jsonrpc": "2.0", "id": 7, "method": "tools/call"},
                headers=_mcp_proxy_token_headers(server_name="github"),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == 7
        assert "error" not in body
        result = body["result"]
        assert result["isError"] is True
        text = result["content"][0]["text"]
        assert vend["connect_url"] in text
        assert "github" in text.lower()

    def test_egress_consent_emits_url_elicitation_when_enabled(self):
        """With egress_consent_use_elicitation=True, a tools/call for a tokenless
        egress server returns the 2025-11-25 URLElicitationRequiredError (-32042)
        whose data.elicitations[] carries a mode:url elicitation with an
        elicitationId and the connect URL.
        """
        import auth_server.server as server_module

        vend = {
            "consent_required": True,
            "connect_url": "https://gw.example.com/oauth2/egress/connect?server=%2Fgithub",
            "request_state": "AEAD-blob",
            "provider": "github",
        }

        async def _consent_vend(token, server):
            return vend

        with (
            patch.object(server_module.settings, "egress_auth_enabled", True),
            patch.object(server_module.settings, "egress_consent_use_elicitation", True),
            patch.object(server_module, "_vend_egress_token", _consent_vend),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/github",
                json={"jsonrpc": "2.0", "id": 7, "method": "tools/call"},
                headers=_mcp_proxy_token_headers(server_name="github"),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == 7
        assert "result" not in body
        err = body["error"]
        assert err["code"] == -32042
        elicitations = err["data"]["elicitations"]
        assert len(elicitations) == 1
        e = elicitations[0]
        assert e["mode"] == "url"
        assert e["elicitationId"]  # present and non-empty
        # connect URL is preserved and carries the elicitationId for correlation
        assert e["url"].startswith(vend["connect_url"])
        assert "elicitationId=" in e["url"]
        assert "github" in e["message"]

    def test_egress_consent_answers_initialize_locally(self):
        """For an egress server with no vaulted token, the gateway must answer
        initialize LOCALLY (not proxy it). The upstream (e.g. GitHub) is itself an
        OAuth RS that 401s every call including initialize; proxying it tokenless
        would 401 the handshake and a legacy client could never reach tools/call.
        initialize is capability negotiation with the gateway, so it is answered
        here -- the handshake completes and the client proceeds to the
        token-requiring methods where consent is surfaced.
        """
        import auth_server.server as server_module

        vend = {
            "consent_required": True,
            "connect_url": "https://gw.example.com/oauth2/egress/connect?server=%2Fgithub",
            "provider": "github",
        }

        async def _consent_vend(token, server):
            return vend

        with (
            patch.object(server_module.settings, "egress_auth_enabled", True),
            patch.object(server_module, "_vend_egress_token", _consent_vend),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/github",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "initialize",
                    "params": {"protocolVersion": "2025-11-25"},
                },
                headers=_mcp_proxy_token_headers(server_name="github"),
            )

        # Local initialize result -- no upstream proxied, no consent error.
        assert response.status_code == 200
        body = response.json()
        assert body["id"] == 1
        assert "error" not in body
        result = body["result"]
        assert result["protocolVersion"] == "2025-11-25"
        assert "capabilities" in result
        assert result["serverInfo"]["name"] == "mcp-gateway-registry"

    def test_egress_consent_acks_notifications_locally(self):
        """notifications/* carry no result; for a tokenless egress server the
        gateway acks them locally (202) rather than proxying to the 401-ing
        upstream."""
        import auth_server.server as server_module

        vend = {
            "consent_required": True,
            "connect_url": "https://gw.example.com/oauth2/egress/connect?server=%2Fgithub",
            "provider": "github",
        }

        async def _consent_vend(token, server):
            return vend

        with (
            patch.object(server_module.settings, "egress_auth_enabled", True),
            patch.object(server_module, "_vend_egress_token", _consent_vend),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/github",
                json={"jsonrpc": "2.0", "method": "notifications/initialized"},
                headers=_mcp_proxy_token_headers(server_name="github"),
            )

        assert response.status_code == 202

    def test_egress_consent_tools_list_returns_empty(self):
        """tools/list must NOT error (erroring dead-ends clients -- they mark the
        server failed and never call a tool). The upstream tool list needs the
        token, so for a tokenless egress server the gateway answers LOCALLY with
        an EMPTY tool list. The user connects out of band via the Connected
        Accounts UI; once the token is vaulted, the vend HITs and the real
        upstream tools are proxied.
        """
        import auth_server.server as server_module

        vend = {
            "consent_required": True,
            "connect_url": "https://gw.example.com/oauth2/egress/connect?server=%2Fgithub",
            "provider": "github",
        }

        async def _consent_vend(token, server):
            return vend

        with (
            patch.object(server_module.settings, "egress_auth_enabled", True),
            patch.object(server_module, "_vend_egress_token", _consent_vend),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/github",
                json={"jsonrpc": "2.0", "id": 9, "method": "tools/list"},
                headers=_mcp_proxy_token_headers(server_name="github"),
            )

        assert response.status_code == 200
        body = response.json()
        assert "error" not in body
        assert body["result"]["tools"] == []

    def test_set_cookie_and_location_dropped_end_to_end(self):
        """End-to-end regression: even if an upstream MCP server emits
        Set-Cookie and Location, the Starlette response the gateway
        sends back to the client must not carry them. This is the
        regression test the reviewer explicitly asked for on the
        allowlist conversion.
        """
        import auth_server.server as server_module

        upstream_resp = _build_mock_upstream_response(
            status_code=200,
            headers={
                "content-type": "application/json",
                "mcp-session-id": "sess-allowlist-100",
                "set-cookie": "session=evil; Path=/; HttpOnly",
                "location": "https://attacker.example/oauth/authorize",
                "strict-transport-security": "max-age=0",
                "access-control-allow-origin": "*",
            },
            body=b'{"jsonrpc":"2.0","id":1,"result":{}}',
        )

        with (
            _patch_httpx_async_client(upstream_resp),
            _patch_scope_repo_allow_all(),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/office-docs",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                headers=_mcp_proxy_token_headers(),
            )

        assert response.status_code == 200
        # Allowlisted header survives.
        assert response.headers.get("mcp-session-id") == "sess-allowlist-100"
        # The reviewer-cited headers must not reach the client.
        assert "set-cookie" not in {k.lower() for k in response.headers.keys()}
        assert response.headers.get("location") is None
        assert response.headers.get("strict-transport-security") is None
        assert response.headers.get("access-control-allow-origin") is None

    def test_www_authenticate_and_retry_after_forwarded(self):
        """The PRM/OAuth flow (PR #1115) needs WWW-Authenticate on 401s,
        and rate-limit-aware MCP clients honor Retry-After. Both must
        round-trip through the gateway.
        """
        import auth_server.server as server_module

        upstream_resp = _build_mock_upstream_response(
            status_code=401,
            headers={
                "content-type": "application/json",
                "www-authenticate": 'Bearer resource_metadata="https://idp.example/.well-known/oauth-protected-resource"',
                "retry-after": "15",
            },
            body=b'{"error":"unauthorized"}',
        )

        with (
            _patch_httpx_async_client(upstream_resp),
            _patch_scope_repo_allow_all(),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/office-docs",
                json={"jsonrpc": "2.0", "id": 1, "method": "initialize"},
                headers=_mcp_proxy_token_headers(),
            )

        assert response.status_code == 401
        assert response.headers.get("www-authenticate", "").startswith("Bearer")
        assert response.headers.get("retry-after") == "15"


# =============================================================================
# OBO EXCHANGE EGRESS MODE TESTS (Phase 3)
# =============================================================================


def _capture_upstream_headers():
    """Patch httpx.AsyncClient.stream to record the headers forwarded upstream.

    Returns (patch_cm, captured) where captured['headers'] holds the dict passed
    to client.stream(...) once a request reaches the upstream-call branch.
    """
    captured: dict = {}
    upstream_resp = _build_mock_upstream_response(
        status_code=200,
        headers={"content-type": "application/json"},
        body=b'{"jsonrpc":"2.0","id":1,"result":{}}',
    )

    mock_stream_cm = AsyncMock()
    mock_stream_cm.__aenter__ = AsyncMock(return_value=upstream_resp)
    mock_stream_cm.__aexit__ = AsyncMock(return_value=False)

    def _stream(method, url, **kwargs):
        captured["headers"] = kwargs.get("headers", {})
        captured["url"] = url
        return mock_stream_cm

    mock_client = AsyncMock()
    mock_client.stream = MagicMock(side_effect=_stream)
    mock_client.__aenter__ = AsyncMock(return_value=mock_client)
    mock_client.__aexit__ = AsyncMock(return_value=False)

    return patch("auth_server.server.httpx.AsyncClient", return_value=mock_client), captured


class _FakeEntraProvider:
    client_id = "gw-client"
    client_secret = "gw-secret"
    token_url = "https://login.microsoftonline.com/t/oauth2/v2.0/token"


class TestMcpProxyOboExchange:
    """Phase 3 seam: obo_exchange branch in mcp_proxy.

    The registry vend returns an OBO DIRECTIVE (mode + target_audience), not a
    token; auth_server runs the exchange locally and injects the result, after
    stripping the user's gateway credentials.
    """

    @staticmethod
    async def _obo_directive_vend(token, server):
        return {
            "mode": "obo_exchange",
            "obo_target_audience": "api://outlook-mcp-server",
            "obo_scopes": [],
        }

    def test_obo_success_strips_creds_and_injects_exchanged_token(self, monkeypatch):
        import auth_server.server as server_module

        ingress_jwt = _obo_ingress_jwt("test-user")

        async def _fake_exchange(provider, subject_token, target_audience, scopes=None):
            assert subject_token == ingress_jwt
            assert target_audience == "api://outlook-mcp-server"
            return "exchanged-obo-token"

        patch_httpx, captured = _capture_upstream_headers()
        with (
            patch.object(server_module.settings, "egress_auth_enabled", True),
            patch.object(server_module, "_vend_egress_token", self._obo_directive_vend),
            patch.object(server_module, "get_auth_provider", lambda *a, **k: _FakeEntraProvider()),
            patch.object(server_module, "obo_exchange", _fake_exchange),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
            _patch_scope_repo_allow_all(),
            patch_httpx,
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/outlook",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "read_inbox"},
                },
                headers={
                    **_mcp_proxy_token_headers(server_name="outlook"),
                    "X-Authorization": f"Bearer {ingress_jwt}",
                    "Cookie": "session=secret",
                },
            )

        assert response.status_code == 200
        sent = {k.lower(): v for k, v in captured["headers"].items()}
        # Exchanged token injected.
        assert sent["authorization"] == "Bearer exchanged-obo-token"
        # User gateway creds / internal identity stripped.
        assert "x-authorization" not in sent
        assert "cookie" not in sent
        assert "x-internal-token" not in sent

    def test_obo_no_bearer_jwt_is_terminal_no_consent(self, monkeypatch):
        """Session-cookie / M2M caller (no bearer ingress JWT) -> terminal error,
        never a consent affordance."""
        import auth_server.server as server_module

        with (
            patch.object(server_module.settings, "egress_auth_enabled", True),
            patch.object(server_module, "_vend_egress_token", self._obo_directive_vend),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/outlook",
                json={
                    "jsonrpc": "2.0",
                    "id": 7,
                    "method": "tools/call",
                    "params": {"name": "read_inbox"},
                },
                # No X-Authorization / Authorization bearer on the request.
                headers=_mcp_proxy_token_headers(server_name="outlook"),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == 7
        assert body["error"]["message"] == "obo_exchange_failed"
        # No URL-mode elicitation / consent affordance.
        assert body["error"]["code"] != -32042
        assert "elicitations" not in body["error"].get("data", {})

    def test_obo_subject_token_principal_mismatch_is_rejected(self, monkeypatch):
        """If the raw ingress JWT's principal differs from the /validate-authorized
        principal (internal token sub=test-user), the exchange is refused terminally
        and no exchange/forward happens -- closing the subject-token binding gap."""
        import auth_server.server as server_module

        exchange_called = {"n": 0}

        async def _fake_exchange(provider, subject_token, target_audience, scopes=None):
            exchange_called["n"] += 1
            return "should-not-be-reached"

        # A JWT for a DIFFERENT principal than the internal token's sub (test-user).
        mismatched_jwt = _obo_ingress_jwt("attacker-user")

        patch_httpx, captured = _capture_upstream_headers()
        with (
            patch.object(server_module.settings, "egress_auth_enabled", True),
            patch.object(server_module, "_vend_egress_token", self._obo_directive_vend),
            patch.object(server_module, "get_auth_provider", lambda *a, **k: _FakeEntraProvider()),
            patch.object(server_module, "obo_exchange", _fake_exchange),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
            _patch_scope_repo_allow_all(),
            patch_httpx,
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/outlook",
                json={
                    "jsonrpc": "2.0",
                    "id": 3,
                    "method": "tools/call",
                    "params": {"name": "read_inbox"},
                },
                headers={
                    **_mcp_proxy_token_headers(server_name="outlook"),
                    "X-Authorization": f"Bearer {mismatched_jwt}",
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["error"]["message"] == "obo_exchange_failed"
        # The exchange must NOT have run and NOTHING was forwarded upstream.
        assert exchange_called["n"] == 0
        assert "headers" not in captured

    def test_obo_exchange_failure_is_terminal_no_consent(self, monkeypatch):
        import auth_server.server as server_module

        # Raise via the same OboExchangeError identity server.py imported (the
        # module is import-path sensitive; OboReauthRequired is a subclass of it).
        async def _failing_exchange(provider, subject_token, target_audience, scopes=None):
            raise server_module.OboExchangeError("token expired")

        with (
            patch.object(server_module.settings, "egress_auth_enabled", True),
            patch.object(server_module, "_vend_egress_token", self._obo_directive_vend),
            patch.object(server_module, "get_auth_provider", lambda *a, **k: _FakeEntraProvider()),
            patch.object(server_module, "obo_exchange", _failing_exchange),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/outlook",
                json={
                    "jsonrpc": "2.0",
                    "id": 9,
                    "method": "tools/call",
                    "params": {"name": "read_inbox"},
                },
                headers={
                    **_mcp_proxy_token_headers(server_name="outlook"),
                    "X-Authorization": f"Bearer {_obo_ingress_jwt('test-user')}",
                },
            )

        assert response.status_code == 200
        body = response.json()
        assert body["error"]["message"] == "obo_exchange_failed"
        assert "token expired" in body["error"]["data"]["detail"]
        assert body["error"]["code"] != -32042

    def test_mode_none_does_not_inject_obo_token(self):
        """Egress feature OFF: the obo branch never fires (no exchanged token is
        injected). Client ingress auth headers are still stripped on egress by the
        unconditional ingress-only strip (#1369), so the upstream sees neither the
        raw ingress JWT nor an obo token."""
        import auth_server.server as server_module

        async def _disabled_vend(token, server):
            return {"consent_required": True}

        patch_httpx, captured = _capture_upstream_headers()
        with (
            # Egress feature OFF entirely -> the whole egress block is skipped.
            patch.object(server_module.settings, "egress_auth_enabled", False),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
            _patch_scope_repo_allow_all(),
            patch_httpx,
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/plain",
                json={"jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": {"name": "x"}},
                headers={
                    **_mcp_proxy_token_headers(server_name="plain"),
                    "X-Authorization": "Bearer raw-ingress-jwt",
                },
            )

        assert response.status_code == 200
        sent = {k.lower(): v for k, v in captured["headers"].items()}
        # Ingress-only auth headers are stripped on egress (#1369): the raw
        # ingress JWT is never forwarded to the upstream MCP server...
        assert "x-authorization" not in sent
        # ...and with the feature off, the obo branch never injects a token.
        assert "authorization" not in sent

    def test_obo_integration_real_exchange_only_idp_mocked(self, monkeypatch):
        """Full pipeline through mcp_proxy with the REAL obo_exchange engine;
        only the IdP token HTTP endpoint is mocked. Proves the wiring: directive
        -> subject extraction -> exchange -> strip -> inject -> forward.

        The engine (httpx .post) and the upstream proxy (httpx .stream) share the
        global httpx.AsyncClient, so a SINGLE unified mock client serves both and
        httpx is patched exactly once (two patches would collide on the same name).
        """
        from contextlib import asynccontextmanager

        import auth_server.server as server_module

        class _IdpResp:
            status_code = 200

            def json(self):
                return {"access_token": "real-exchanged-token"}

        idp_post = AsyncMock(return_value=_IdpResp())

        upstream_resp = _build_mock_upstream_response(
            status_code=200,
            headers={"content-type": "application/json"},
            body=b'{"jsonrpc":"2.0","id":1,"result":{}}',
        )
        upstream_cm = AsyncMock()
        upstream_cm.__aenter__ = AsyncMock(return_value=upstream_resp)
        upstream_cm.__aexit__ = AsyncMock(return_value=False)
        captured: dict = {}

        def _stream(method, url, **kwargs):
            captured["headers"] = kwargs.get("headers", {})
            return upstream_cm

        @asynccontextmanager
        async def _unified_client(*a, **k):
            c = MagicMock()
            c.post = idp_post  # engine's IdP token call
            c.stream = MagicMock(side_effect=_stream)  # upstream proxy call
            yield c

        ingress_jwt = _obo_ingress_jwt("test-user")

        # The engine's IdP token POST now goes through the SSRF-guarded client
        # (registry.utils.url_guard.guarded_async_client), imported lazily inside
        # egress_obo.obo_exchange, so patch it at its source module. The upstream
        # proxy hop still uses server_module.httpx.AsyncClient. Both are pointed at
        # the same unified mock client so a single fake serves the two calls.
        with (
            patch.object(server_module.settings, "egress_auth_enabled", True),
            patch.object(server_module, "_vend_egress_token", self._obo_directive_vend),
            patch.object(server_module, "get_auth_provider", lambda *a, **k: _FakeEntraProvider()),
            patch.object(server_module.httpx, "AsyncClient", _unified_client),
            patch("registry.utils.url_guard.guarded_async_client", _unified_client),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
            _patch_scope_repo_allow_all(),
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/outlook",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "read_inbox"},
                },
                headers={
                    **_mcp_proxy_token_headers(server_name="outlook"),
                    "X-Authorization": f"Bearer {ingress_jwt}",
                },
            )

        assert response.status_code == 200
        # The engine built the Entra jwt-bearer body from the directive + subject.
        idp_body = idp_post.call_args.kwargs["data"]
        assert idp_body["grant_type"] == "urn:ietf:params:oauth:grant-type:jwt-bearer"
        assert idp_body["assertion"] == ingress_jwt
        assert idp_body["scope"] == "api://outlook-mcp-server/.default"
        # The exchanged token reached the upstream Authorization header.
        sent = {k.lower(): v for k, v in captured["headers"].items()}
        assert sent["authorization"] == "Bearer real-exchanged-token"


class TestMcpProxyPatMode:
    """pat egress mode in mcp_proxy.

    A vended PAT arrives as {access_token: <PAT>} and flows through the SAME
    generic inject branch as a vaulted OAuth token (no auth_server change for the
    hit path). A miss arrives as {mode: "pat"} with no access_token/connect_url
    and is answered by the terminal _pat_missing_response.
    """

    def test_pat_missing_response_shape(self):
        import auth_server.server as server_module

        resp = server_module._pat_missing_response("github", "tools/call", 7)
        assert resp.status_code == 200
        import json as _json

        body = _json.loads(resp.body)
        assert body["id"] == 7
        # Terminal tool result (isError), NOT a -32042 URL elicitation.
        assert "error" not in body
        assert body["result"]["isError"] is True
        text = body["result"]["content"][0]["text"]
        assert "No PAT configured" in text
        assert "Connected Accounts" in text
        # No connect/authorize URL is offered (pat is not interactive).
        assert "http" not in text.lower()

    def test_pat_hit_injects_and_strips_ingress_creds(self):
        import auth_server.server as server_module

        async def _pat_vend(token, server):
            return {"access_token": "ghp_the_pat"}

        patch_httpx, captured = _capture_upstream_headers()
        with (
            patch.object(server_module.settings, "egress_auth_enabled", True),
            patch.object(server_module, "_vend_egress_token", _pat_vend),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
            _patch_scope_repo_allow_all(),
            patch_httpx,
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/github",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "list_repos"},
                },
                headers={
                    **_mcp_proxy_token_headers(server_name="github"),
                    "X-Authorization": "Bearer raw-ingress-jwt",
                    "Cookie": "session=secret",
                },
            )

        assert response.status_code == 200
        sent = {k.lower(): v for k, v in captured["headers"].items()}
        # The PAT is injected via the generic access_token branch.
        assert sent["authorization"] == "Bearer ghp_the_pat"
        # Ingress gateway creds / internal identity are stripped before inject.
        assert "x-authorization" not in sent
        assert "cookie" not in sent
        assert "x-internal-token" not in sent

    def test_pat_hit_injects_custom_header_and_prefix(self):
        """The vend response carries the inject header derived from the server's
        Backend Auth scheme (api_key -> PRIVATE-TOKEN with an empty prefix -> a
        bare token in a custom header). mcp_proxy injects it and strips any
        client-supplied copy so only the gateway-injected value reaches upstream."""
        import auth_server.server as server_module

        async def _pat_vend(token, server):
            return {
                "access_token": "glpat_the_pat",
                "pat_header_name": "PRIVATE-TOKEN",
                "pat_value_prefix": "",
            }

        patch_httpx, captured = _capture_upstream_headers()
        with (
            patch.object(server_module.settings, "egress_auth_enabled", True),
            patch.object(server_module, "_vend_egress_token", _pat_vend),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
            _patch_scope_repo_allow_all(),
            patch_httpx,
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/gitlab",
                json={
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": "list_projects"},
                },
                headers={
                    **_mcp_proxy_token_headers(server_name="gitlab"),
                    # A hostile/stale client copy of the custom header must not survive.
                    "PRIVATE-TOKEN": "attacker-supplied",
                    "X-Authorization": "Bearer raw-ingress-jwt",
                },
            )

        assert response.status_code == 200
        sent = {k.lower(): v for k, v in captured["headers"].items()}
        # Bare token in the custom header (no "Bearer " prefix).
        assert sent["private-token"] == "glpat_the_pat"
        # The client-supplied copy of the custom header was dropped, not merged.
        assert "attacker-supplied" not in sent.get("private-token", "")
        # No stray Authorization was injected for a custom-header pat server.
        assert "authorization" not in sent
        assert "x-authorization" not in sent

    def test_pat_miss_is_terminal_no_forward(self):
        import auth_server.server as server_module

        async def _pat_miss_vend(token, server):
            return {"consent_required": True, "mode": "pat"}

        patch_httpx, captured = _capture_upstream_headers()
        with (
            patch.object(server_module.settings, "egress_auth_enabled", True),
            patch.object(server_module, "_vend_egress_token", _pat_miss_vend),
            patch.object(server_module, "_read_mcp_filter_enabled", return_value=False),
            _patch_scope_repo_allow_all(),
            patch_httpx,
        ):
            client = TestClient(server_module.app)
            response = client.post(
                "/mcp-proxy/github",
                json={"jsonrpc": "2.0", "id": 9, "method": "tools/call"},
                headers=_mcp_proxy_token_headers(server_name="github"),
            )

        assert response.status_code == 200
        body = response.json()
        assert body["id"] == 9
        assert body["result"]["isError"] is True
        assert "No PAT configured" in body["result"]["content"][0]["text"]
        # Nothing was forwarded upstream (terminal miss).
        assert "headers" not in captured


# =============================================================================
# SCOPES STARTUP LOGGING TESTS (issue #1248)
# =============================================================================


class TestLogScopesLoaded:
    """Tests for _log_scopes_loaded empty-vs-populated startup logging.

    The auth-server reconfigures the root logger with custom handlers
    (registry.utils.logging_setup), which makes pytest's caplog fixture
    unreliable here. Patch the module logger directly and assert on the calls.
    """

    def test_empty_scopes_emits_warning_with_remediation(self):
        """0 group mappings -> WARNING with actionable seeding hint."""
        from auth_server import server as server_module

        with patch.object(server_module, "logger") as mock_logger:
            server_module._log_scopes_loaded({"group_mappings": {}})

        mock_logger.warning.assert_called_once()
        mock_logger.info.assert_not_called()
        msg = mock_logger.warning.call_args[0][0]
        assert "EMPTY" in msg
        assert "read-only" in msg
        assert "load-scopes.py" in msg

    def test_missing_group_mappings_key_treated_as_empty(self):
        """A config dict with no group_mappings key still warns."""
        from auth_server import server as server_module

        with patch.object(server_module, "logger") as mock_logger:
            server_module._log_scopes_loaded({})

        mock_logger.warning.assert_called_once()

    def test_populated_scopes_logs_info_no_warning(self):
        """Non-empty group mappings -> INFO only, no warning (unchanged behaviour)."""
        from auth_server import server as server_module

        with patch.object(server_module, "logger") as mock_logger:
            server_module._log_scopes_loaded({"group_mappings": {"mcp-registry-admin": ["admin"]}})

        mock_logger.warning.assert_not_called()
        mock_logger.info.assert_called_once()
        assert "1 group mappings" in mock_logger.info.call_args[0][0]


# =============================================================================
# TOKEN LIFETIME ENFORCEMENT (#889)
# =============================================================================


class TestTokenLifetimeEnforcement:
    """Verify that expires_in_hours is honoured and clamped to
    MAX_TOKEN_LIFETIME_HOURS (#889).
    """

    def _generate_self_signed_token(
        self,
        auth_env_vars: dict,
        expires_in_hours: int = 8,
    ) -> dict:
        """Helper: call /internal/tokens with an OAuth user context so
        the self-signed JWT path is taken, and return the decoded claims.
        """
        import auth_server.server as server_module

        server_module.user_token_generation_counts.clear()

        client = TestClient(server_module.app)
        body = {
            "user_context": {
                "username": "alice",
                "scopes": ["mcp-servers/read"],
                "groups": ["mcp-registry-user"],
                "auth_method": "oauth2",
                "provider": "keycloak",
            },
            "requested_scopes": ["mcp-servers/read"],
            "expires_in_hours": expires_in_hours,
            "description": "lifetime test",
        }
        response = client.post(
            "/internal/tokens",
            json=body,
            headers=_internal_auth_headers(auth_env_vars),
        )
        assert response.status_code == 200, response.text
        data = response.json()
        token = data["access_token"]
        claims = jwt.decode(
            token,
            server_module.SECRET_KEY,
            algorithms=["HS256"],
            audience="mcp-registry",
        )
        return {**data, "claims": claims}

    def test_default_lifetime_is_8_hours(self, auth_env_vars):
        """Omitting expires_in_hours defaults to 8 h."""
        result = self._generate_self_signed_token(auth_env_vars)
        assert result["expires_in"] == 8 * 3600

    def test_custom_lifetime_honoured(self, auth_env_vars):
        """A caller-requested 4 h lifetime must be respected (#889)."""
        result = self._generate_self_signed_token(auth_env_vars, expires_in_hours=4)
        assert result["expires_in"] == 4 * 3600

    def test_lifetime_clamped_to_max(self, auth_env_vars):
        """Requesting > MAX_TOKEN_LIFETIME_HOURS (24) must be clamped."""
        result = self._generate_self_signed_token(auth_env_vars, expires_in_hours=48)
        # MAX_TOKEN_LIFETIME_HOURS = 24
        assert result["expires_in"] == 24 * 3600

    def test_lifetime_floor_is_one_hour(self, auth_env_vars):
        """Requesting 0 or negative hours must be clamped to 1 h."""
        result = self._generate_self_signed_token(auth_env_vars, expires_in_hours=0)
        assert result["expires_in"] == 1 * 3600


# =============================================================================
# FORWARDED-BODY RE-AUTHORIZATION TESTS (MCP proxy hop)
# =============================================================================


class TestRegisteredServerFromProxyPath:
    """Tests for _registered_server_from_proxy_path scope-key derivation.

    The scope key must match what /validate authorizes against so both hops
    check the identical server. Only a trailing MCP transport segment is
    stripped; federated "peer/server" keys are preserved.
    """

    def test_local_server_no_transport(self):
        """A bare server name is returned unchanged."""
        from auth_server.server import _registered_server_from_proxy_path

        assert _registered_server_from_proxy_path("currenttime") == "currenttime"

    def test_local_server_strips_trailing_transport(self):
        """A trailing mcp/sse/messages segment is stripped."""
        from auth_server.server import _registered_server_from_proxy_path

        assert _registered_server_from_proxy_path("currenttime/mcp") == "currenttime"
        assert _registered_server_from_proxy_path("currenttime/sse") == "currenttime"
        assert _registered_server_from_proxy_path("currenttime/messages") == "currenttime"

    def test_federated_peer_server_preserved(self):
        """A federated peer/server key is preserved (not truncated to peer)."""
        from auth_server.server import _registered_server_from_proxy_path

        assert (
            _registered_server_from_proxy_path("peer-registry-lob-1/cloudflare-docs")
            == "peer-registry-lob-1/cloudflare-docs"
        )

    def test_federated_peer_server_strips_trailing_transport(self):
        """peer/server/mcp -> peer/server (only the transport tail is stripped)."""
        from auth_server.server import _registered_server_from_proxy_path

        assert (
            _registered_server_from_proxy_path("peer-registry-lob-1/cloudflare-docs/mcp")
            == "peer-registry-lob-1/cloudflare-docs"
        )

    def test_leading_and_trailing_slashes_ignored(self):
        """Surrounding slashes do not change the derived key."""
        from auth_server.server import _registered_server_from_proxy_path

        assert _registered_server_from_proxy_path("/currenttime/mcp/") == "currenttime"


class TestAuthorizeForwardedMcpBody:
    """Tests for _authorize_forwarded_mcp_body (TM-15 forwarded-body re-auth).

    The proxy hop must re-authorize the EXACT forwarded body, independently of
    the separately-captured X-Body /validate saw, and must fail closed on any
    body it cannot parse well enough to determine the scope-relevant method.
    """

    @staticmethod
    def _body(method: str, tool: str | None = None) -> bytes:
        """Build a JSON-RPC request body as raw bytes."""
        import json

        payload: dict = {"jsonrpc": "2.0", "id": 1, "method": method}
        if tool is not None:
            payload["params"] = {"name": tool}
        return json.dumps(payload).encode("utf-8")

    @pytest.mark.asyncio
    async def test_divergent_tools_call_rejected_for_readonly_scope(
        self, mock_scope_repository_with_data
    ):
        """A tools/call forwarded body is rejected for a read-only scope.

        This is the concrete TM-15 bypass: /validate saw no X-Body (or an
        "initialize" default) and passed, but the forwarded body is a
        privileged tools/call. read:servers only allows initialize/tools/list
        on test-server, so re-authorizing the real body must deny.
        """
        from fastapi import HTTPException

        from auth_server.server import _authorize_forwarded_mcp_body

        with patch(
            "auth_server.server.get_scope_repository",
            return_value=mock_scope_repository_with_data,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _authorize_forwarded_mcp_body(
                    "test-server/mcp",
                    self._body("tools/call", tool="danger-tool"),
                    ["read:servers"],
                )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_tools_call_allowed_for_write_scope(self, mock_scope_repository_with_data):
        """A tools/call body is allowed when the scope permits it (no raise)."""
        from auth_server.server import _authorize_forwarded_mcp_body

        with patch(
            "auth_server.server.get_scope_repository",
            return_value=mock_scope_repository_with_data,
        ):
            # write:servers allows tools/call with wildcard tools on test-server.
            await _authorize_forwarded_mcp_body(
                "test-server/mcp",
                self._body("tools/call", tool="any-tool"),
                ["write:servers"],
            )

    @pytest.mark.asyncio
    async def test_small_initialize_body_allowed(self, mock_scope_repository_with_data):
        """A normal small initialize body still authorizes for a valid scope."""
        from auth_server.server import _authorize_forwarded_mcp_body

        with patch(
            "auth_server.server.get_scope_repository",
            return_value=mock_scope_repository_with_data,
        ):
            await _authorize_forwarded_mcp_body(
                "test-server/mcp",
                self._body("initialize"),
                ["read:servers"],
            )

    @pytest.mark.asyncio
    async def test_empty_body_treated_as_initialize(self, mock_scope_repository_with_data):
        """An empty forwarded body is authorized as the initialize method."""
        from auth_server.server import _authorize_forwarded_mcp_body

        with patch(
            "auth_server.server.get_scope_repository",
            return_value=mock_scope_repository_with_data,
        ):
            # Allowed for a scope that grants initialize on the server ...
            await _authorize_forwarded_mcp_body("test-server/mcp", b"", ["read:servers"])

    @pytest.mark.asyncio
    async def test_empty_body_denied_without_matching_scope(self, mock_scope_repository_with_data):
        """Empty body (initialize) is denied when no scope grants the server."""
        from fastapi import HTTPException

        from auth_server.server import _authorize_forwarded_mcp_body

        with patch(
            "auth_server.server.get_scope_repository",
            return_value=mock_scope_repository_with_data,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _authorize_forwarded_mcp_body("other-server/mcp", b"", ["read:servers"])
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_unparseable_body_fails_closed(self, mock_scope_repository_with_data):
        """A non-JSON forwarded body is rejected (cannot determine method)."""
        from fastapi import HTTPException

        from auth_server.server import _authorize_forwarded_mcp_body

        with patch(
            "auth_server.server.get_scope_repository",
            return_value=mock_scope_repository_with_data,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _authorize_forwarded_mcp_body(
                    "test-server/mcp",
                    b"\xff\xfe not json at all {",
                    ["write:servers"],
                )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_non_object_json_fails_closed(self, mock_scope_repository_with_data):
        """A well-formed JSON value that is not a JSON-RPC object is rejected."""
        from fastapi import HTTPException

        from auth_server.server import _authorize_forwarded_mcp_body

        with patch(
            "auth_server.server.get_scope_repository",
            return_value=mock_scope_repository_with_data,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _authorize_forwarded_mcp_body(
                    "test-server/mcp",
                    b'["tools/call", "danger"]',
                    ["write:servers"],
                )
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_no_scopes_denied(self, mock_scope_repository_with_data):
        """A caller with no scopes is denied regardless of body."""
        from fastapi import HTTPException

        from auth_server.server import _authorize_forwarded_mcp_body

        with patch(
            "auth_server.server.get_scope_repository",
            return_value=mock_scope_repository_with_data,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _authorize_forwarded_mcp_body("test-server/mcp", self._body("initialize"), [])
        assert exc_info.value.status_code == 403

    @pytest.mark.asyncio
    async def test_tools_call_missing_tool_name_denied_for_readonly(
        self, mock_scope_repository_with_data
    ):
        """tools/call with no tool name is denied under a read-only scope."""
        from fastapi import HTTPException

        from auth_server.server import _authorize_forwarded_mcp_body

        with patch(
            "auth_server.server.get_scope_repository",
            return_value=mock_scope_repository_with_data,
        ):
            with pytest.raises(HTTPException) as exc_info:
                await _authorize_forwarded_mcp_body(
                    "test-server/mcp",
                    self._body("tools/call"),
                    ["read:servers"],
                )
        assert exc_info.value.status_code == 403


class TestSessionCookieSecureDefault:
    """Verify the session cookie Secure flag resolves fail-closed.

    The callback handler computes the Secure flag as
    ``OAUTH2_CONFIG.get("session", {}).get("secure", True) and is_https``.
    These tests pin the two properties that matter for security: a missing
    ``session.secure`` config key defaults to Secure (True), and the flag is
    never emitted over plain HTTP (a browser would reject a Secure cookie set
    on an HTTP response).
    """

    @staticmethod
    def _resolve_cookie_secure(session_config: dict, is_https: bool) -> bool:
        """Mirror the auth_server callback's Secure-flag resolution."""
        cookie_secure_config = session_config.get("secure", True)
        return cookie_secure_config and is_https

    def test_missing_secure_key_defaults_to_secure_over_https(self) -> None:
        """A config with no session.secure key must default to Secure=True."""
        assert self._resolve_cookie_secure({}, is_https=True) is True

    def test_explicit_false_disables_secure_for_dev(self) -> None:
        """An operator can still opt out for a plain-HTTP dev stack."""
        assert self._resolve_cookie_secure({"secure": False}, is_https=True) is False

    def test_secure_never_set_over_plain_http(self) -> None:
        """Even secure-by-default must not emit Secure over plain HTTP."""
        assert self._resolve_cookie_secure({}, is_https=False) is False
        assert self._resolve_cookie_secure({"secure": True}, is_https=False) is False


class TestForwardHeadersIngressStrip:
    """Egress ingress-auth policy (issue #1266): client auth headers are ingress
    credentials, stripped on egress. X-Authorization and Cookie are ALWAYS
    stripped; Authorization is stripped unless relay_authorization=True (set only
    for the built-in internal registry-tools server). No general relay mode --
    upstream creds come from the egress vault.
    """

    def _forward(self, incoming, relay=False):
        from auth_server.server import _forward_headers

        return _forward_headers(incoming, relay_authorization=relay)

    def test_strips_authorization_by_default(self):
        """Non-relay (default): Authorization stripped."""
        out = self._forward({"Authorization": "Bearer a", "Accept": "x"})
        assert "authorization" not in {k.lower() for k in out}
        assert out.get("Accept") == "x"

    def test_strips_x_authorization_always_even_when_relaying(self):
        """X-Authorization is never forwarded, even for the internal relay server."""
        out = self._forward({"X-Authorization": "Bearer x"}, relay=True)
        assert "x-authorization" not in {k.lower() for k in out}

    def test_strips_cookie_always_even_when_relaying(self):
        """Cookie is never forwarded, even for the internal relay server."""
        out = self._forward({"Cookie": "mcp_gateway_session=abc"}, relay=True)
        assert "cookie" not in {k.lower() for k in out}

    def test_relays_authorization_only_when_flag_set(self):
        """relay_authorization=True keeps Authorization (internal relay server)."""
        out = self._forward({"Authorization": "Bearer a"}, relay=True)
        assert out.get("Authorization") == "Bearer a"

    def test_relay_flag_does_not_readmit_x_authorization_or_cookie(self):
        """With relay on, only Authorization is relayed; X-Authorization/Cookie stay stripped."""
        out = self._forward(
            {"Authorization": "Bearer a", "X-Authorization": "Bearer x", "Cookie": "s=1"},
            relay=True,
        )
        assert out.get("Authorization") == "Bearer a"
        assert "x-authorization" not in {k.lower() for k in out}
        assert "cookie" not in {k.lower() for k in out}

    def test_case_insensitive(self):
        """Lowercase header keys are handled identically."""
        out = self._forward(
            {"authorization": "Bearer a", "x-authorization": "Bearer x", "cookie": "s=1"},
            relay=False,
        )
        assert "authorization" not in {k.lower() for k in out}
        assert "x-authorization" not in {k.lower() for k in out}
        assert "cookie" not in {k.lower() for k in out}

    def test_still_strips_hop_by_hop_and_x_upstream_url(self):
        """Regression: existing exclusions (hop-by-hop, X-Upstream-Url) still apply."""
        out = self._forward(
            {"X-Upstream-Url": "http://x", "Connection": "keep-alive", "Accept": "y"},
        )
        assert "x-upstream-url" not in {k.lower() for k in out}
        assert "connection" not in {k.lower() for k in out}
        assert out.get("Accept") == "y"

    def test_preserves_non_auth_headers(self):
        """Regression: non-auth headers pass through untouched."""
        out = self._forward(
            {"Content-Type": "application/json", "Mcp-Session-Id": "vs-abc", "Accept": "z"},
        )
        assert out.get("Content-Type") == "application/json"
        assert out.get("Mcp-Session-Id") == "vs-abc"
        assert out.get("Accept") == "z"

    def test_internal_relay_server_set_is_airegistry_tools(self):
        """The internal relay allowlist is the single hardcoded registry-tools server."""
        from auth_server.server import _INTERNAL_INGRESS_RELAY_SERVERS

        assert _INTERNAL_INGRESS_RELAY_SERVERS == frozenset({"airegistry-tools"})

    def test_proxy_authorization_stripped(self):
        """Proxy-Authorization (hop-by-hop and an auth header) never reaches upstream."""
        out = self._forward({"Proxy-Authorization": "Bearer p"}, relay=True)
        assert "proxy-authorization" not in {k.lower() for k in out}

    def test_empty_headers_no_crash(self):
        """Empty input yields empty output without error."""
        assert self._forward({}) == {}

    def test_relay_false_strips_all_three_auth_headers_together(self):
        """The core leak-closure: a request carrying all three auth headers to a
        non-internal server forwards none of them."""
        out = self._forward(
            {
                "Authorization": "Bearer a",
                "X-Authorization": "Bearer x",
                "Cookie": "mcp_gateway_session=s",
                "Accept": "application/json",
            },
            relay=False,
        )
        assert set(k.lower() for k in out) == {"accept"}

    def test_bearer_value_not_needed_key_only(self):
        """Stripping is by header name, independent of value shape."""
        out = self._forward({"Authorization": "Basic Zm9vOmJhcg=="}, relay=False)
        assert "authorization" not in {k.lower() for k in out}


class TestInternalRelayDecision:
    """The mcp_proxy relay decision keys on the verified `server` claim (first
    path segment), matches the hardcoded internal set exactly, and normalizes
    case. Mirrors the inline logic in mcp_proxy (issue #1266).
    """

    def _decides_relay(self, server_claim):
        from auth_server.server import _INTERNAL_INGRESS_RELAY_SERVERS

        # Mirror mcp_proxy: registered_server = (claims.get("server") or "").lower()
        registered_server = (server_claim or "").lower()
        return registered_server in _INTERNAL_INGRESS_RELAY_SERVERS

    def test_exact_internal_server_relays(self):
        assert self._decides_relay("airegistry-tools") is True

    def test_case_insensitive_match(self):
        assert self._decides_relay("AiRegistry-Tools") is True

    def test_missing_claim_does_not_relay(self):
        assert self._decides_relay("") is False
        assert self._decides_relay(None) is False

    def test_similar_but_not_exact_name_does_not_relay(self):
        """No substring/prefix match: only the exact internal name relays."""
        for name in (
            "airegistry-tools-evil",
            "evil-airegistry-tools",
            "airegistry",
            "airegistry_tools",
            "ai-registry",
        ):
            assert self._decides_relay(name) is False, name

    def test_federated_prefixed_name_does_not_relay(self):
        """A federated copy (e.g. server claim 'ai-registry' from /ai-registry/...)
        is a different first path segment and must NOT relay."""
        assert self._decides_relay("ai-registry") is False


# =============================================================================
# A2A AGENT PROXY ACCESS TESTS
# =============================================================================


class TestGetA2AAgentPath:
    """Tests for _get_a2a_agent_path URL parsing."""

    def test_none_url_returns_none(self):
        from auth_server.server import _get_a2a_agent_path

        assert _get_a2a_agent_path(None) is None

    def test_agent_jsonrpc_url(self):
        from auth_server.server import _get_a2a_agent_path

        assert _get_a2a_agent_path("https://mcp.example.com/agent/travel/") == "/travel"

    def test_agent_card_url(self):
        from auth_server.server import _get_a2a_agent_path

        url = "https://mcp.example.com/agent/flight-booking-agent/.well-known/agent-card.json"
        assert _get_a2a_agent_path(url) == "/flight-booking-agent"

    def test_non_agent_url_returns_none(self):
        from auth_server.server import _get_a2a_agent_path

        assert _get_a2a_agent_path("https://mcp.example.com/currenttime/mcp") is None

    def test_api_url_returns_none(self):
        from auth_server.server import _get_a2a_agent_path

        assert _get_a2a_agent_path("https://mcp.example.com/api/agents") is None

    def test_bare_agent_prefix_without_segment_returns_none(self):
        from auth_server.server import _get_a2a_agent_path

        assert _get_a2a_agent_path("https://mcp.example.com/agent/") is None

    def test_multi_segment_agent_path(self):
        from auth_server.server import _get_a2a_agent_path

        assert _get_a2a_agent_path("https://mcp.example.com/agent/lob1/travel/") == "/lob1/travel"

    def test_multi_segment_agent_card_url(self):
        from auth_server.server import _get_a2a_agent_path

        url = "https://mcp.example.com/agent/lob1/travel/.well-known/agent-card.json"
        assert _get_a2a_agent_path(url) == "/lob1/travel"

    def test_registry_root_path_prefix_is_stripped(self):
        """When the registry is hosted on a sub-path, the prefix is stripped."""
        import auth_server.server as server_module

        with patch.object(server_module, "REGISTRY_ROOT_PATH", "/registry"):
            assert (
                server_module._get_a2a_agent_path("https://mcp.example.com/registry/agent/travel/")
                == "/travel"
            )

    def test_agent_card_at_root_returns_none(self):
        """A card discovery URL with no agent segment resolves to None."""
        from auth_server.server import _get_a2a_agent_path

        url = "https://mcp.example.com/agent/.well-known/agent-card.json"
        assert _get_a2a_agent_path(url) is None

    def test_empty_agent_segment_returns_none(self):
        """An empty path segment (…/agent/lob1//travel/) is rejected."""
        from auth_server.server import _get_a2a_agent_path

        assert _get_a2a_agent_path("https://mcp.example.com/agent/lob1//travel/") is None


class TestValidateA2AAgentAccess:
    """Tests for validate_a2a_agent_access structured per-agent gating.

    The function resolves each caller scope via the scope repository and looks
    for a per-agent rule ``{"agent": "<path or *>", "actions": [...]}`` whose
    ``agent`` matches and whose ``actions`` include ``invoke_agent`` (or a
    wildcard). The rule shape mirrors a server rule (``agent`` like ``server``,
    ``actions`` like ``methods``). The repository is mocked to return scope ->
    server_access config, mirroring the MCP validate_server_tool_access tests.
    """

    @staticmethod
    def _repo(scope_config: dict[str, list]):
        """Build a mock scope repository returning the given scope -> config map."""
        repo = AsyncMock()

        async def get_server_scopes(scope_name: str):
            return scope_config.get(scope_name, [])

        async def get_server_scopes_bulk(scope_names: list[str]):
            # Mirror the real bulk contract: one round-trip returning
            # {scope_name: rules} for the requested scopes.
            return {name: scope_config.get(name, []) for name in scope_names}

        repo.get_server_scopes.side_effect = get_server_scopes
        repo.get_server_scopes_bulk.side_effect = get_server_scopes_bulk
        return repo

    @staticmethod
    def _invoke_scope(agent: str) -> list:
        """A server_access list granting invoke_agent on the given agent (path or *)."""
        return [{"agent": agent, "actions": ["invoke_agent"]}]

    async def test_admin_scope_allows_regardless_of_doc_shape(self):
        """An admin is allowed invoke even when their scope doc has NO agent rule
        (legacy nested shape backwards compat -- no re-seed required)."""
        from auth_server.server import validate_a2a_agent_access

        # Legacy nested shape: no {agent, actions} rule, so the flattener yields
        # nothing invoke-relevant; the admin marker must still grant access.
        repo = self._repo({"registry-admins": []})
        with patch("auth_server.server.get_scope_repository", return_value=repo):
            assert await validate_a2a_agent_access("/travel", ["registry-admins"]) is True

    async def test_admin_group_marker_allows(self):
        """The admin marker is honored when it arrives as a GROUP, not a scope."""
        from auth_server.server import validate_a2a_agent_access

        repo = self._repo({})
        with patch("auth_server.server.get_scope_repository", return_value=repo):
            assert (
                await validate_a2a_agent_access("/travel", [], user_groups=["mcp-registry-admin"])
                is True
            )

    async def test_non_admin_legacy_shape_still_denied(self):
        """A non-admin whose doc lacks a {agent, actions} invoke rule is denied
        (admin bypass must not leak to ordinary users)."""
        from auth_server.server import validate_a2a_agent_access

        repo = self._repo({"public-mcp-users": []})
        with patch("auth_server.server.get_scope_repository", return_value=repo):
            assert await validate_a2a_agent_access("/travel", ["public-mcp-users"]) is False

    async def test_invoke_wildcard_agent_allows(self):
        from auth_server.server import validate_a2a_agent_access

        repo = self._repo({"a2a-invoker": self._invoke_scope("*")})
        with patch("auth_server.server.get_scope_repository", return_value=repo):
            assert await validate_a2a_agent_access("/travel", ["a2a-invoker"]) is True

    async def test_invoke_all_agent_keyword_allows(self):
        """The ``all`` keyword works as a wildcard for the agent identifier too."""
        from auth_server.server import validate_a2a_agent_access

        repo = self._repo({"a2a-invoker": self._invoke_scope("all")})
        with patch("auth_server.server.get_scope_repository", return_value=repo):
            assert await validate_a2a_agent_access("/travel", ["a2a-invoker"]) is True

    async def test_invoke_exact_path_allows(self):
        from auth_server.server import validate_a2a_agent_access

        repo = self._repo({"a2a-travel": self._invoke_scope("/travel")})
        with patch("auth_server.server.get_scope_repository", return_value=repo):
            assert await validate_a2a_agent_access("/travel", ["a2a-travel"]) is True

    async def test_invoke_different_path_denied(self):
        from auth_server.server import validate_a2a_agent_access

        repo = self._repo({"a2a-hr": self._invoke_scope("/hr")})
        with patch("auth_server.server.get_scope_repository", return_value=repo):
            assert await validate_a2a_agent_access("/travel", ["a2a-hr"]) is False

    async def test_sibling_path_not_matched(self):
        """An exact-path rule for /travel-extended must NOT grant /travel."""
        from auth_server.server import validate_a2a_agent_access

        repo = self._repo({"a2a-ext": self._invoke_scope("/travel-extended")})
        with patch("auth_server.server.get_scope_repository", return_value=repo):
            assert await validate_a2a_agent_access("/travel", ["a2a-ext"]) is False

    async def test_actions_wildcard_allows(self):
        """A rule whose actions include the ``all`` wildcard grants invoke."""
        from auth_server.server import validate_a2a_agent_access

        repo = self._repo({"a2a-admin": [{"agent": "/travel", "actions": ["all"]}]})
        with patch("auth_server.server.get_scope_repository", return_value=repo):
            assert await validate_a2a_agent_access("/travel", ["a2a-admin"]) is True

    async def test_non_invoke_action_denied(self):
        """A rule granting only agent CRUD (no invoke_agent) is denied."""
        from auth_server.server import validate_a2a_agent_access

        crud = [{"agent": "*", "actions": ["get_agent", "list_agents"]}]
        repo = self._repo({"a2a-reader": crud})
        with patch("auth_server.server.get_scope_repository", return_value=repo):
            assert await validate_a2a_agent_access("/travel", ["a2a-reader"]) is False

    async def test_mcp_only_scope_denied(self):
        """A pure MCP server scope (no agents block) is denied."""
        from auth_server.server import validate_a2a_agent_access

        mcp = [{"server": "*", "methods": ["all"], "tools": ["all"]}]
        repo = self._repo({"mcp-servers-unrestricted/read": mcp})
        with patch("auth_server.server.get_scope_repository", return_value=repo):
            assert (
                await validate_a2a_agent_access("/travel", ["mcp-servers-unrestricted/read"])
                is False
            )

    async def test_empty_scopes_denied(self):
        from auth_server.server import validate_a2a_agent_access

        assert await validate_a2a_agent_access("/travel", []) is False

    async def test_scope_resolution_error_is_skipped_and_denied(self):
        """A repository lookup that raises is not fatal; access is denied (fail closed)."""
        from auth_server.server import validate_a2a_agent_access

        repo = AsyncMock()
        repo.get_server_scopes_bulk.side_effect = RuntimeError("scope backend down")
        with patch("auth_server.server.get_scope_repository", return_value=repo):
            assert await validate_a2a_agent_access("/travel", ["a2a-invoker"]) is False

    async def test_unknown_scope_with_empty_config_denied(self):
        """A scope that resolves to an empty config is skipped (denied)."""
        from auth_server.server import validate_a2a_agent_access

        repo = self._repo({})
        with patch("auth_server.server.get_scope_repository", return_value=repo):
            assert await validate_a2a_agent_access("/travel", ["missing-scope"]) is False


# =============================================================================
# LEGACY STATIC ADMIN TOKEN STRENGTH VALIDATION AT STARTUP
# =============================================================================


class TestLegacyRegistryTokenStrengthValidation:
    """Startup strength validation for the legacy REGISTRY_API_TOKEN.

    When set, REGISTRY_API_TOKEN is promoted to an unrestricted admin entry, so
    it grants the highest privilege in the system. It must therefore clear the
    same strength bar as the application signing secret: an unset token is fine
    (the feature simply has no legacy entry), but a token that is present must
    be strong (non-empty after stripping, at least the minimum length, and not a
    known-weak placeholder). A present-but-weak value must fail closed at
    startup rather than silently arm a weak admin credential.

    These tests reload the server module under a patched environment so the
    real module-level validation runs, then restore a known-good module state
    for the rest of the suite.
    """

    _STRONG = "x" * 40
    _RESTORE_ENV = {
        "SECRET_KEY": "test-secret-key-that-is-definitely-long-enough-32b",
        "DOCUMENTDB_HOST": "localhost",
    }

    def _reload_with_token(self, token_value):
        """Reload auth_server.server with REGISTRY_API_TOKEN set to token_value.

        A ``None`` token_value means the variable is unset entirely. Returns the
        freshly reloaded module. Raises whatever the module raises at import.
        """
        import importlib
        import os

        import auth_server.server as server_module

        env = dict(self._RESTORE_ENV)
        if token_value is not None:
            env["REGISTRY_API_TOKEN"] = token_value

        # patch.dict(clear=False) plus explicit pop keeps unrelated env intact
        # while giving us precise control over REGISTRY_API_TOKEN.
        with patch.dict(os.environ, env, clear=False):
            if token_value is None:
                os.environ.pop("REGISTRY_API_TOKEN", None)
            return importlib.reload(server_module)

    def teardown_method(self):
        """Restore a valid module state so later tests see a sane module."""
        import importlib
        import os

        import auth_server.server as server_module

        with patch.dict(os.environ, self._RESTORE_ENV, clear=False):
            os.environ.pop("REGISTRY_API_TOKEN", None)
            importlib.reload(server_module)

    def test_unset_token_is_accepted_and_empty(self):
        """An unset token is fine: no legacy admin credential, no raise."""
        reloaded = self._reload_with_token(None)
        assert reloaded.REGISTRY_API_TOKEN == ""

    def test_strong_token_is_accepted(self):
        """A sufficiently long, non-placeholder token is accepted verbatim."""
        reloaded = self._reload_with_token(self._STRONG)
        assert reloaded.REGISTRY_API_TOKEN == self._STRONG

    def test_short_token_fails_closed(self):
        """A present but too-short token must raise at startup."""
        with pytest.raises(RuntimeError):
            self._reload_with_token("short")

    def test_whitespace_only_token_is_treated_as_unset(self):
        """A whitespace-only token is equivalent to unset: no admin credential.

        The canonical validator treats a whitespace-only value as unset when the
        secret is optional, which is the fail-closed outcome here: no legacy
        admin entry is armed. A whitespace-only value is never accepted as a
        usable credential (it strips to empty), so no weak admin token results.
        """
        reloaded = self._reload_with_token("   " * 20)
        assert reloaded.REGISTRY_API_TOKEN == ""

    def test_known_weak_literal_token_fails_closed(self):
        """A present but known-weak placeholder literal must raise at startup."""
        with pytest.raises(RuntimeError):
            self._reload_with_token("change-this-immediately-use-a-strong-random-key-in-production")


# =============================================================================
# FEDERATION STATIC TOKEN IS LEAST-PRIVILEGE READ-ONLY
# =============================================================================


class TestFederationStaticTokenReadOnly:
    """The federation static token grants read-only access.

    The federation static token is a long-lived, non-expiring credential meant
    for federation data sync. It must be least-privilege: it grants only
    ``federation/read`` and must NOT carry a peer/federation management scope.
    Peer management stays behind a real admin credential.
    """

    _TOKEN = "f" * 40

    def test_validate_grants_only_read_scope(self):
        """A matching federation token yields scopes == ['federation/read']."""
        import auth_server.server as server_module

        with (
            patch.object(server_module, "FEDERATION_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "FEDERATION_STATIC_TOKEN", self._TOKEN),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {self._TOKEN}",
                    "X-Original-URL": "https://example.com/api/federation/peers",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["method"] == "federation-static"
        assert data["scopes"] == ["federation/read"]

    def test_validate_does_not_grant_peer_management_scope(self):
        """The federation token must not carry the peer-management scope."""
        import auth_server.server as server_module

        with (
            patch.object(server_module, "FEDERATION_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "FEDERATION_STATIC_TOKEN", self._TOKEN),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {self._TOKEN}",
                    "X-Original-URL": "https://example.com/api/federation/peers",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert "federation/peers" not in data["scopes"]
        assert "federation/peers" not in response.headers.get("X-Scopes", "")


# =============================================================================
# FEDERATION STATIC TOKEN STRENGTH VALIDATION (FAIL CLOSED ON WEAK TOKEN)
# =============================================================================


class TestFederationStaticTokenStrengthValidation:
    """Startup strength validation for FEDERATION_STATIC_TOKEN.

    The federation static token bypasses IdP JWT validation when armed, so it
    must clear the same weak-value bar as every other privilege-granting
    credential. When the operator explicitly enables the feature, the token is
    required and must be strong: a short OR known-weak placeholder value must
    NOT be armed. Because this is an optional feature, a weak token degrades
    gracefully -- the feature is DISABLED (fail closed) rather than crashing the
    process -- mirroring the missing-token branch. Warn-only is not fail closed.

    These tests reload the server module under a patched environment so the real
    module-level validation runs, then restore a known-good module state for the
    rest of the suite.
    """

    _STRONG = "f" * 40
    _RESTORE_ENV = {
        "SECRET_KEY": "test-secret-key-that-is-definitely-long-enough-32b",
        "DOCUMENTDB_HOST": "localhost",
    }

    def _reload_with_federation_token(self, token_value):
        """Reload auth_server.server with the feature enabled and a given token.

        A ``None`` token_value means FEDERATION_STATIC_TOKEN is unset entirely.
        Returns the freshly reloaded module.
        """
        import importlib
        import os

        import auth_server.server as server_module

        env = dict(self._RESTORE_ENV)
        env["FEDERATION_STATIC_TOKEN_AUTH_ENABLED"] = "true"
        if token_value is not None:
            env["FEDERATION_STATIC_TOKEN"] = token_value

        with patch.dict(os.environ, env, clear=False):
            if token_value is None:
                os.environ.pop("FEDERATION_STATIC_TOKEN", None)
            return importlib.reload(server_module)

    def teardown_method(self):
        """Restore a valid module state (feature disabled) for later tests."""
        import importlib
        import os

        import auth_server.server as server_module

        with patch.dict(os.environ, self._RESTORE_ENV, clear=False):
            os.environ.pop("FEDERATION_STATIC_TOKEN", None)
            os.environ.pop("FEDERATION_STATIC_TOKEN_AUTH_ENABLED", None)
            importlib.reload(server_module)

    def test_strong_token_stays_enabled_and_armed(self):
        """A strong token keeps the feature enabled and arms the token."""
        reloaded = self._reload_with_federation_token(self._STRONG)
        assert reloaded.FEDERATION_STATIC_TOKEN_AUTH_ENABLED is True
        assert reloaded.FEDERATION_STATIC_TOKEN == self._STRONG

    def test_short_token_disables_feature(self):
        """A short token disables the feature (fail closed), does not raise."""
        reloaded = self._reload_with_federation_token("short")
        assert reloaded.FEDERATION_STATIC_TOKEN_AUTH_ENABLED is False

    def test_known_weak_literal_disables_feature(self):
        """A known-weak >=32-char placeholder disables the feature."""
        reloaded = self._reload_with_federation_token(
            "change-this-immediately-use-a-strong-random-key-in-production"
        )
        assert reloaded.FEDERATION_STATIC_TOKEN_AUTH_ENABLED is False

    def test_unset_token_disables_feature(self):
        """Enabling the feature without a token disables it (fail closed)."""
        reloaded = self._reload_with_federation_token(None)
        assert reloaded.FEDERATION_STATIC_TOKEN_AUTH_ENABLED is False

    def test_validate_does_not_authenticate_weak_token(self):
        """A weak token is not armed: the /validate federation path rejects it."""
        weak = "short"
        reloaded = self._reload_with_federation_token(weak)
        assert reloaded.FEDERATION_STATIC_TOKEN_AUTH_ENABLED is False

        client = TestClient(reloaded.app)
        response = client.get(
            "/validate",
            headers={
                "Authorization": f"Bearer {weak}",
                "X-Original-URL": "https://example.com/api/federation/peers",
            },
        )
        # The weak token is not armed, so it never authenticates via the
        # federation-static path (it falls through to standard JWT validation,
        # which rejects a non-JWT bearer).
        assert response.status_code != 200 or response.json().get("method") != ("federation-static")


# =============================================================================
# REGISTRY_API_KEYS ENTRY WEAK-VALUE REJECTION (FAIL CLOSED)
# =============================================================================


class TestRegistryApiKeyEntryStrengthValidation:
    """Per-key REGISTRY_API_KEYS entries must reject weak key values.

    A keyed entry grants the scopes mapped from its groups (which may include
    admin), so its key bypasses IdP JWT validation and must clear the same
    weak-value bar as every other privilege-granting credential. The Pydantic
    ``min_length=32`` constraint alone accepts a >=32-char known placeholder, so
    the key is additionally routed through the canonical validator, which
    rejects short AND known-weak literals. A weak key must fail closed: the
    entry is rejected and the parse path disables the feature rather than arming
    a weak keyed credential.
    """

    _STRONG = "x" * 40

    def test_strong_key_validates(self):
        """A strong, non-placeholder key builds a valid entry."""
        import auth_server.server as server_module

        entry = server_module._RegistryApiKeyEntry(
            name="deploy-pipeline",
            key=self._STRONG,
            groups=["mcp-registry-admin"],
        )
        assert entry.key == self._STRONG

    def test_short_key_raises(self):
        """A key shorter than the minimum raises a validation error."""
        import pydantic

        import auth_server.server as server_module

        with pytest.raises((pydantic.ValidationError, ValueError)):
            server_module._RegistryApiKeyEntry(
                name="deploy-pipeline",
                key="short",
                groups=["mcp-registry-admin"],
            )

    def test_known_weak_literal_key_raises(self):
        """A >=32-char known-weak placeholder key raises a validation error."""
        import pydantic

        import auth_server.server as server_module

        with pytest.raises((pydantic.ValidationError, ValueError)):
            server_module._RegistryApiKeyEntry(
                name="deploy-pipeline",
                key="change-this-immediately-use-a-strong-random-key-in-production",
                groups=["mcp-registry-admin"],
            )

    def test_parse_rejects_weak_key_entry(self):
        """A >=32-char weak-literal key fails the parser (fail closed)."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "deploy-pipeline": {
                    "key": "change-this-immediately-use-a-strong-random-key-in-production",
                    "groups": ["mcp-registry-admin"],
                }
            }
        )
        with pytest.raises(ValueError, match="Invalid entry"):
            server_module._parse_registry_api_keys(raw)

    async def test_build_static_token_map_disabled_on_weak_key(self):
        """A weak keyed entry disables static-token auth (matches malformed-JSON)."""
        import json

        import auth_server.server as server_module

        raw = json.dumps(
            {
                "deploy-pipeline": {
                    "key": "change-this-immediately-use-a-strong-random-key-in-production",
                    "groups": ["mcp-registry-admin"],
                }
            }
        )
        with (
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "_REGISTRY_API_KEYS_RAW", raw),
            patch.object(server_module, "REGISTRY_API_TOKEN", ""),
            patch.object(server_module, "_STATIC_TOKEN_MAP", {}),
        ):
            await server_module._build_static_token_map()
            assert server_module.REGISTRY_STATIC_TOKEN_AUTH_ENABLED is False
            assert server_module._STATIC_TOKEN_MAP == {}


# =============================================================================
# RUNTIME FEDERATION-TOKEN ROTATION MUST ENFORCE THE SAME STRENGTH BAR
# =============================================================================


class TestFederationTokenRotationStrength:
    """The runtime rotation endpoint must reject weak new tokens.

    Rotating the federation static token arms the same privileged credential as
    startup, so the rotation endpoint must clear the same weak-value bar: a short
    OR known-weak placeholder value must be rejected with 400 and must NOT arm
    the token, otherwise an admin could rotate to a long-but-well-known
    placeholder and silently undo the startup hardening.
    """

    _ADMIN = "a" * 40
    _STRONG = "f" * 40

    def _client_and_module(self):
        import auth_server.server as server_module

        return TestClient(server_module.app), server_module

    def test_short_new_token_rejected(self):
        """A short rotation token is rejected (400) and does not arm the token."""
        client, server_module = self._client_and_module()
        with (
            patch.object(server_module, "REGISTRY_API_TOKEN", self._ADMIN),
            patch.object(server_module, "FEDERATION_STATIC_TOKEN", ""),
            patch.object(server_module, "FEDERATION_STATIC_TOKEN_AUTH_ENABLED", False),
        ):
            response = client.post(
                "/admin/federation-token",
                headers={"Authorization": f"Bearer {self._ADMIN}"},
                json={"new_token": "short"},
            )
            assert response.status_code == 400
            assert server_module.FEDERATION_STATIC_TOKEN_AUTH_ENABLED is False
            assert server_module.FEDERATION_STATIC_TOKEN == ""

    def test_known_weak_literal_new_token_rejected(self):
        """A long-but-known-placeholder rotation token is rejected (400)."""
        client, server_module = self._client_and_module()
        with (
            patch.object(server_module, "REGISTRY_API_TOKEN", self._ADMIN),
            patch.object(server_module, "FEDERATION_STATIC_TOKEN", ""),
            patch.object(server_module, "FEDERATION_STATIC_TOKEN_AUTH_ENABLED", False),
        ):
            response = client.post(
                "/admin/federation-token",
                headers={"Authorization": f"Bearer {self._ADMIN}"},
                json={"new_token": "change-this-immediately-use-a-strong-random-key-in-production"},
            )
            assert response.status_code == 400
            assert server_module.FEDERATION_STATIC_TOKEN_AUTH_ENABLED is False
            assert server_module.FEDERATION_STATIC_TOKEN == ""

    def test_strong_new_token_rotates(self):
        """A strong rotation token is accepted and arms the feature."""
        client, server_module = self._client_and_module()
        with (
            patch.object(server_module, "REGISTRY_API_TOKEN", self._ADMIN),
            patch.object(server_module, "FEDERATION_STATIC_TOKEN", ""),
            patch.object(server_module, "FEDERATION_STATIC_TOKEN_AUTH_ENABLED", False),
        ):
            response = client.post(
                "/admin/federation-token",
                headers={"Authorization": f"Bearer {self._ADMIN}"},
                json={"new_token": self._STRONG},
            )
            assert response.status_code == 200
            assert response.json()["action"] == "rotated"
            assert server_module.FEDERATION_STATIC_TOKEN == self._STRONG
            assert server_module.FEDERATION_STATIC_TOKEN_AUTH_ENABLED is True


class TestEntraLogoutQueryStringGuard:
    """Entra logout must not breach the logout-URL length limit (AADSTS90015).

    Large ID tokens (users in many groups) push the full logout URL past Entra's
    limit. The handler drops the optional id_token_hint when the composed URL
    would exceed MAX_LOGOUT_URL_LENGTH; Entra still processes the logout without
    it.
    """

    _PROVIDER = "entra"
    _LOGOUT_URL = "https://login.microsoftonline.com/tenant/oauth2/v2.0/logout"

    def _run_logout(self, id_token_hint):
        import asyncio
        import urllib.parse

        import auth_server.server as server_module

        config = {
            "providers": {
                self._PROVIDER: {
                    "client_id": "app-client-id",
                    "logout_url": self._LOGOUT_URL,
                }
            }
        }
        request = Mock()
        request.headers = {}

        with (
            patch.object(server_module, "OAUTH2_CONFIG", config),
            patch.dict("os.environ", {"REGISTRY_URL": "https://gw.example.com"}),
        ):
            response = asyncio.run(
                server_module.oauth2_logout(
                    provider=self._PROVIDER,
                    request=request,
                    redirect_uri="/login",
                    id_token_hint=id_token_hint,
                )
            )

        parsed = urllib.parse.urlparse(response.headers["location"])
        return urllib.parse.parse_qs(parsed.query)

    def _hint_for_url_length(self, target_url_len):
        """Build an id_token_hint that makes the composed logout URL exactly
        target_url_len chars, so boundary behavior can be tested precisely."""
        import urllib.parse

        base_params = {"post_logout_redirect_uri": "https://gw.example.com/login"}
        # Length of the URL with an empty hint value appended.
        empty_hint_url_len = len(
            f"{self._LOGOUT_URL}?" + urllib.parse.urlencode({**base_params, "id_token_hint": ""})
        )
        # "a" is not percent-encoded, so each char adds exactly one to the URL.
        return "a" * (target_url_len - empty_hint_url_len)

    def test_small_token_keeps_id_token_hint(self):
        """A short id_token_hint stays in the logout query string."""
        query = self._run_logout("short-token")
        assert query["id_token_hint"] == ["short-token"]
        assert query["post_logout_redirect_uri"] == ["https://gw.example.com/login"]

    def test_no_hint_provided_omits_id_token_hint(self):
        """With no id_token_hint the handler must not inject one; logout still
        redirects with the post_logout_redirect_uri."""
        query = self._run_logout(None)
        assert "id_token_hint" not in query
        assert query["post_logout_redirect_uri"] == ["https://gw.example.com/login"]

    def test_hint_at_limit_is_kept(self):
        """A hint whose composed URL is exactly at the limit is kept (boundary)."""
        from auth_server.server import MAX_LOGOUT_URL_LENGTH

        hint = self._hint_for_url_length(MAX_LOGOUT_URL_LENGTH)
        query = self._run_logout(hint)
        assert query["id_token_hint"] == [hint]

    def test_hint_one_over_limit_is_dropped(self):
        """A hint whose composed URL is one char over the limit is dropped."""
        from auth_server.server import MAX_LOGOUT_URL_LENGTH

        hint = self._hint_for_url_length(MAX_LOGOUT_URL_LENGTH + 1)
        query = self._run_logout(hint)
        assert "id_token_hint" not in query
        assert query["post_logout_redirect_uri"] == ["https://gw.example.com/login"]

    def test_oversized_token_drops_id_token_hint(self):
        """An id_token_hint that would breach the limit is omitted; logout still
        redirects with the post_logout_redirect_uri so Entra completes it."""
        from auth_server.server import MAX_LOGOUT_URL_LENGTH

        oversized = "a" * (MAX_LOGOUT_URL_LENGTH + 100)
        query = self._run_logout(oversized)
        assert "id_token_hint" not in query
        assert query["post_logout_redirect_uri"] == ["https://gw.example.com/login"]
