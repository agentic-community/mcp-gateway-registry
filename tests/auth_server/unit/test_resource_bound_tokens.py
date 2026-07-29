"""Tests for resource-bound token support in auth_server.

Covers:
    - URL classification helpers used by the /validate guard.
    - Blocked-endpoint enforcement for resource-bound tokens.
    - JWT claims emitted by /internal/tokens with and without a resource.
    - Legacy-token deprecation warning scoped to self-signed tokens only.
    - /validate edge enforcement (integration): match, mismatch, blocked
      path, user-token unrestricted, legacy token accepted.
"""

from __future__ import annotations

import logging
import time
from unittest.mock import patch

import jwt
import pytest

pytestmark = [pytest.mark.unit, pytest.mark.auth]


def _internal_auth_headers() -> dict:
    """Build an Authorization header carrying a valid internal JWT.

    ``/internal/tokens`` requires a Bearer JWT signed with the shared
    SECRET_KEY (see ``registry.auth.internal.generate_internal_token``).
    Tests that POST to the endpoint must attach this header.
    """
    from registry.auth.internal import generate_internal_token

    token = generate_internal_token(
        subject="test-suite",
        purpose="unit-test",
    )
    return {"Authorization": f"Bearer {token}"}


def _mint_self_signed(
    secret_key: str,
    *,
    token_kind: str | None = "user",
    resource_type: str | None = None,
    resource_id: str | None = None,
    username: str = "alice",
    scope: str = "mcp-servers/read mcp-servers/execute",
) -> str:
    """Create a self-signed JWT matching auth_server.server's format.

    We build the token directly rather than calling /internal/tokens because
    these tests want to exercise the /validate edge enforcement in
    isolation from the mint path.
    """
    now = int(time.time())
    claims = {
        "iss": "mcp-auth-server",
        "aud": "mcp-registry",
        "sub": username,
        "preferred_username": username,
        "email": f"{username}@example.com",
        "groups": [],
        "scope": scope,
        "token_use": "access",
        "auth_method": "oauth2",
        "provider": "keycloak",
        "iat": now,
        "exp": now + 3600,
    }
    if token_kind is not None:
        claims["token_kind"] = token_kind
    if resource_type:
        claims["resource_type"] = resource_type
    if resource_id:
        claims["resource_id"] = resource_id
    return jwt.encode(claims, secret_key, algorithm="HS256")


class TestAuthServerUsesSharedHelpers:
    """The auth_server must re-export the shared classify/allow helpers so
    /validate calls the exact functions the registry's pre-mint check uses.
    """

    @pytest.fixture(autouse=True)
    def _server_module(self, auth_env_vars):
        import auth_server.server as server_module

        self.server_module = server_module

    def test_classify_and_allow_functions_shared(self):
        from registry.auth import resource_binding

        assert self.server_module.classify_request_url is resource_binding.classify_request_url
        assert (
            self.server_module.check_resource_token_allowed
            is resource_binding.check_resource_token_allowed
        )


class TestMintResourceBoundToken:
    """Verify JWT claims emitted by /internal/tokens."""

    def _mint(self, resource=None, auth_env_vars=None):
        from fastapi.testclient import TestClient

        import auth_server.server as server_module

        # Clear rate limits between test instances
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
            "expires_in_hours": 1,
            "description": "test",
        }
        if resource is not None:
            body["resource"] = resource
        response = client.post("/internal/tokens", json=body, headers=_internal_auth_headers())
        return response, server_module.SECRET_KEY

    def test_user_token_has_token_kind_user(self, auth_env_vars):
        response, secret = self._mint(auth_env_vars=auth_env_vars)
        assert response.status_code == 200, response.text
        token = response.json()["access_token"]
        claims = jwt.decode(token, secret, algorithms=["HS256"], audience="mcp-registry")
        assert claims["token_kind"] == "user"
        assert "resource_type" not in claims
        assert "resource_id" not in claims

    def test_user_token_stamps_egress_user_from_session_subject(self, auth_env_vars):
        # A session-backed mint stamps the session's OIDC sub as egress_user so
        # the vend keys the vault on the SAME id the browser-consent path wrote.
        # The token's `sub` stays the login username (unchanged).
        from unittest.mock import AsyncMock
        from unittest.mock import patch as _patch

        from fastapi.testclient import TestClient

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
                "session_id": "s1",
            },
            "requested_scopes": ["mcp-servers/read"],
            "expires_in_hours": 1,
            "description": "test",
        }
        session = {
            "username": "alice",
            "groups": ["mcp-registry-user"],
            "subject": "oidc-sub-uuid",
        }
        with (
            _patch("session_store.resolve_session", AsyncMock(return_value=session)),
            _patch(
                "auth_server.server.map_groups_to_scopes",
                AsyncMock(return_value=["mcp-servers/read"]),
            ),
        ):
            response = client.post("/internal/tokens", json=body, headers=_internal_auth_headers())
        assert response.status_code == 200, response.text
        claims = jwt.decode(
            response.json()["access_token"],
            server_module.SECRET_KEY,
            algorithms=["HS256"],
            audience="mcp-registry",
        )
        assert claims["egress_user"] == "oidc-sub-uuid"
        assert claims["sub"] == "alice"

    def test_resource_token_has_claims(self, auth_env_vars):
        response, secret = self._mint(
            resource={"type": "server", "id": "/cloudflare-docs"},
            auth_env_vars=auth_env_vars,
        )
        assert response.status_code == 200, response.text
        token = response.json()["access_token"]
        claims = jwt.decode(token, secret, algorithms=["HS256"], audience="mcp-registry")
        assert claims["token_kind"] == "resource"
        assert claims["resource_type"] == "server"
        # Leading slash is normalized away.
        assert claims["resource_id"] == "cloudflare-docs"

    def test_resource_token_invalid_type_rejected(self, auth_env_vars):
        response, _ = self._mint(
            resource={"type": "not-a-type", "id": "foo"},
            auth_env_vars=auth_env_vars,
        )
        assert response.status_code == 422  # Pydantic validation

    def test_resource_token_accepts_all_four_types(self, auth_env_vars):
        for rtype, rid in [
            ("server", "cloudflare-docs"),
            ("virtual_server", "virtual/my-agg"),
            ("agent", "code-reviewer"),
            ("skill", "python-linter"),
        ]:
            response, secret = self._mint(
                resource={"type": rtype, "id": rid},
                auth_env_vars=auth_env_vars,
            )
            assert response.status_code == 200, (rtype, response.text)
            claims = jwt.decode(
                response.json()["access_token"],
                secret,
                algorithms=["HS256"],
                audience="mcp-registry",
            )
            assert claims["token_kind"] == "resource"
            assert claims["resource_type"] == rtype
            assert claims["resource_id"] == rid


class TestLegacyTokenWarningScoping:
    """External IdP tokens (Cognito, Keycloak, Entra, Okta, Auth0) legitimately
    do not carry a ``token_kind`` claim. The deprecation warning must only
    fire for self-signed tokens; otherwise every external IdP request
    triggers noise on every /validate call.
    """

    def test_external_idp_token_does_not_log_legacy_warning(self, auth_env_vars, caplog):
        from fastapi.testclient import TestClient

        import auth_server.server as server_module

        # Simulate what a Cognito validation result looks like: method
        # "cognito", "data" is raw IdP claims with no token_kind.
        validation_result = {
            "valid": True,
            "method": "cognito",
            "username": "alice",
            "email": "alice@example.com",
            "groups": [],
            "scopes": [],
            "client_id": "test-client",
            "data": {"cognito:username": "alice", "sub": "alice"},
        }
        provider = type(
            "P",
            (),
            {
                "validate_token": staticmethod(lambda *_a, **_k: validation_result),
                "get_provider_info": staticmethod(
                    lambda: {"provider_type": "cognito", "region": "us-east-1"}
                ),
            },
        )()

        caplog.set_level(logging.WARNING, logger="auth_server.server")
        with patch("auth_server.server.get_auth_provider", return_value=provider):
            client = TestClient(server_module.app)
            client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer external-token",
                    "X-Original-URL": "https://example.com/api/agents/foo",
                },
            )

        # The key invariant: no legacy warning for the external IdP token.
        assert not any(
            "Legacy" in record.message and "token_kind" in record.message
            for record in caplog.records
        ), [r.message for r in caplog.records]


class TestValidateEdgeEnforcement:
    """End-to-end /validate enforcement for resource-bound tokens.

    These tests catch regressions that the pure-Python helper tests can't:
    * The enforcement block being moved inside the ``if server_name:`` branch
      (which would skip /api/agents/* and /api/skills/* entirely).
    * Claim-vs-URL mismatch no longer returning 403.
    * Blocked endpoints silently allowing resource-bound tokens.
    * User tokens being restricted by resource-binding rules they don't have.
    """

    def _make_provider(self, module):
        """Build a minimal provider that delegates straight to the
        self-signed JWT validator. The default cognito provider in
        ``auth_env_vars`` would try to reach JWKS in the real Cognito
        service, which is not what these tests are exercising.
        """
        validator = module.SimplifiedCognitoValidator()

        class _SelfSignedOnlyProvider:
            def validate_token(self, token: str) -> dict:
                return validator.validate_self_signed_token(token)

            def get_provider_info(self) -> dict:
                return {"provider_type": "self_signed", "region": "us-east-1"}

        return _SelfSignedOnlyProvider()

    def _client_and_secret(self, auth_env_vars):
        from fastapi.testclient import TestClient

        import auth_server.server as server_module

        # Module-level SECRET_KEY was loaded at import time before
        # auth_env_vars ran; patch it so the self-signed validator accepts
        # our freshly-minted test tokens.
        server_module.SECRET_KEY = auth_env_vars["SECRET_KEY"]
        return TestClient(server_module.app), server_module.SECRET_KEY, server_module

    def test_resource_token_on_matching_agent_url_passes(
        self, auth_env_vars, mock_scope_repository_with_data
    ):
        # /api/* paths skip the scope-validation server_name branch.
        # The enforcement block must still fire and accept the match.
        client, secret, module = self._client_and_secret(auth_env_vars)
        token = _mint_self_signed(
            secret,
            token_kind="resource",
            resource_type="agent",
            resource_id="code-reviewer",
        )
        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.get_auth_provider",
                return_value=self._make_provider(module),
            ),
        ):
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Original-URL": "https://example.com/api/agents/code-reviewer",
                },
            )
        assert response.status_code == 200, response.text

    def test_resource_token_on_mismatched_agent_url_returns_403(
        self, auth_env_vars, mock_scope_repository_with_data
    ):
        client, secret, module = self._client_and_secret(auth_env_vars)
        token = _mint_self_signed(
            secret,
            token_kind="resource",
            resource_type="agent",
            resource_id="code-reviewer",
        )
        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.get_auth_provider",
                return_value=self._make_provider(module),
            ),
        ):
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Original-URL": "https://example.com/api/agents/other-agent",
                },
            )
        assert response.status_code == 403, response.text
        # Body must say "does not permit this request" (generic; no
        # binding disclosure). The substring "bound to" would also match
        # the word "Resource-bound token" in the body, so it tests nothing
        # specific — use the unambiguous phrase instead.
        assert "does not permit this request" in response.json()["detail"]

    def test_resource_token_on_different_type_returns_403(
        self, auth_env_vars, mock_scope_repository_with_data
    ):
        # Token bound to server:test-server hitting a virtual server must 403.
        client, secret, module = self._client_and_secret(auth_env_vars)
        token = _mint_self_signed(
            secret,
            token_kind="resource",
            resource_type="server",
            resource_id="test-server",
        )
        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.get_auth_provider",
                return_value=self._make_provider(module),
            ),
        ):
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Original-URL": "https://example.com/virtual/test-server/mcp",
                },
            )
        assert response.status_code == 403, response.text

    def test_resource_token_on_tokens_generate_blocked(
        self, auth_env_vars, mock_scope_repository_with_data
    ):
        client, secret, module = self._client_and_secret(auth_env_vars)
        token = _mint_self_signed(
            secret,
            token_kind="resource",
            resource_type="agent",
            resource_id="code-reviewer",
        )
        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.get_auth_provider",
                return_value=self._make_provider(module),
            ),
        ):
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Original-URL": "https://example.com/api/tokens/generate",
                },
            )
        assert response.status_code == 403, response.text
        assert (
            "cannot access this endpoint" in response.json()["detail"].lower()
            or "bound" in response.json()["detail"].lower()
        )

    def test_resource_token_on_auth_me_allowed(
        self, auth_env_vars, mock_scope_repository_with_data
    ):
        # /api/auth/me is carved out of the deny-list so bound tokens can
        # still call the introspection endpoint.
        client, secret, module = self._client_and_secret(auth_env_vars)
        token = _mint_self_signed(
            secret,
            token_kind="resource",
            resource_type="agent",
            resource_id="code-reviewer",
        )
        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.get_auth_provider",
                return_value=self._make_provider(module),
            ),
        ):
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Original-URL": "https://example.com/api/auth/me",
                },
            )
        # /api/auth/me is an introspection path: it's not a classifiable
        # resource but every token (including resource-bound ones) must
        # be able to hit it to verify itself. The allow-list bypasses
        # both the deny-list AND the classification requirement.
        assert response.status_code == 200, response.text

    def test_user_token_unrestricted_on_api_paths(
        self, auth_env_vars, mock_scope_repository_with_data
    ):
        # User-kind tokens must NOT be subject to the resource-binding
        # checks — they should continue to work on any path their scopes
        # allow. This is the regression test for "I moved the enforcement
        # block and now user tokens are blocked too."
        client, secret, module = self._client_and_secret(auth_env_vars)
        token = _mint_self_signed(secret, token_kind="user")
        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.get_auth_provider",
                return_value=self._make_provider(module),
            ),
        ):
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Original-URL": "https://example.com/api/agents/code-reviewer",
                },
            )
        assert response.status_code == 200, response.text

    def test_self_signed_token_without_kind_rejected(
        self, auth_env_vars, mock_scope_repository_with_data
    ):
        # Every self-signed token minted emits token_kind.
        # A self-signed JWT without the claim is either a prior artifact
        # that outlived its rollout window or a forgery attempt — reject
        # hard rather than silently accepting as a user token.
        client, secret, module = self._client_and_secret(auth_env_vars)
        token = _mint_self_signed(secret, token_kind=None)
        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.get_auth_provider",
                return_value=self._make_provider(module),
            ),
        ):
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Original-URL": "https://example.com/api/agents/code-reviewer",
                },
            )
        assert response.status_code == 403, response.text
        assert "token_kind" in response.json()["detail"]

    def test_external_idp_token_without_kind_accepted(
        self, auth_env_vars, mock_scope_repository_with_data
    ):
        # External IdP tokens (Cognito/Keycloak/Entra/Okta/Auth0) and
        # session cookies never carry token_kind — that's normal, not
        # legacy. They must continue to flow through the unrestricted
        # user-token path. Only self-signed tokens are gated on token_kind.
        from fastapi.testclient import TestClient

        import auth_server.server as server_module

        # Build a provider that mimics an external IdP: it returns
        # method != self_signed and data without token_kind.
        validation_result = {
            "valid": True,
            "method": "cognito",
            "username": "alice",
            "email": "alice@example.com",
            "groups": [],
            "scopes": [],
            "client_id": "test-client",
            "data": {"cognito:username": "alice", "sub": "alice"},
        }

        class _ExternalProvider:
            def validate_token(self, _token):
                return validation_result

            def get_provider_info(self):
                return {"provider_type": "cognito", "region": "us-east-1"}

        client = TestClient(server_module.app)
        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.get_auth_provider",
                return_value=_ExternalProvider(),
            ),
        ):
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer external-opaque-token",
                    "X-Original-URL": "https://example.com/api/agents/code-reviewer",
                },
            )
        assert response.status_code == 200, response.text

    @pytest.mark.parametrize("bad_kind", ["admin", "superuser", "", "0", "garbage"])
    def test_unknown_token_kind_rejected(
        self, auth_env_vars, mock_scope_repository_with_data, bad_kind
    ):
        # Any token_kind we don't explicitly recognize must be rejected
        # rather than silently treated as a user token. Defence against
        # future claim types that haven't been taught to this enforcer
        # and against forgery attempts (which already need SECRET_KEY,
        # but we don't want to rely on that as the only boundary).
        client, secret, module = self._client_and_secret(auth_env_vars)
        token = _mint_self_signed(secret, token_kind=bad_kind)
        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.get_auth_provider",
                return_value=self._make_provider(module),
            ),
        ):
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Original-URL": "https://example.com/api/agents/foo",
                },
            )
        assert response.status_code == 403, response.text

    def test_resource_token_with_whitespace_only_claims_rejected(
        self, auth_env_vars, mock_scope_repository_with_data
    ):
        # A whitespace-only resource_type or resource_id must be caught
        # as "missing required claims", not fall through and produce a
        # misleading "does not permit this request" 403.
        client, secret, module = self._client_and_secret(auth_env_vars)
        token = _mint_self_signed(
            secret,
            token_kind="resource",
            resource_type="agent",
            resource_id="   ",
        )
        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.get_auth_provider",
                return_value=self._make_provider(module),
            ),
        ):
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Original-URL": "https://example.com/api/agents/foo",
                },
            )
        assert response.status_code == 403
        assert "missing required claims" in response.json()["detail"]

    def test_resource_token_traversal_in_original_url_blocked(
        self, auth_env_vars, mock_scope_repository_with_data
    ):
        # urlparse does NOT resolve '..'. An attacker submitting
        # /api/auth/me/../tokens/generate as X-Original-URL keeps the raw
        # path; it won't match the introspection exact-match check and
        # falls through to the deny-list, which catches /api/auth and
        # /api/tokens. Explicit regression test so a future "normalize
        # the URL before checking" refactor can't accidentally resolve
        # the traversal into a permitted path.
        client, secret, module = self._client_and_secret(auth_env_vars)
        token = _mint_self_signed(
            secret,
            token_kind="resource",
            resource_type="agent",
            resource_id="code-reviewer",
        )
        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.get_auth_provider",
                return_value=self._make_provider(module),
            ),
        ):
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Original-URL": ("https://example.com/api/auth/me/../tokens/generate"),
                },
            )
        assert response.status_code == 403, response.text

    def test_resource_token_without_original_url_rejected(
        self, auth_env_vars, mock_scope_repository_with_data
    ):
        # If the edge did not forward X-Original-URL, the guard cannot
        # determine the requested resource and must fail-closed — a silent
        # skip would let a bound token reach any path.
        client, secret, module = self._client_and_secret(auth_env_vars)
        token = _mint_self_signed(
            secret,
            token_kind="resource",
            resource_type="agent",
            resource_id="code-reviewer",
        )
        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.get_auth_provider",
                return_value=self._make_provider(module),
            ),
        ):
            response = client.get(
                "/validate",
                headers={"Authorization": f"Bearer {token}"},
            )
        assert response.status_code == 403, response.text

    def test_mismatch_error_body_does_not_leak_binding(
        self, auth_env_vars, mock_scope_repository_with_data
    ):
        # A stolen resource-bound token should not have its binding
        # disclosed to the holder via the 403 body. Specifics stay in the
        # server-side log.
        client, secret, module = self._client_and_secret(auth_env_vars)
        token = _mint_self_signed(
            secret,
            token_kind="resource",
            resource_type="agent",
            resource_id="secret-agent",
        )
        with (
            patch(
                "auth_server.server.get_scope_repository",
                return_value=mock_scope_repository_with_data,
            ),
            patch(
                "auth_server.server.get_auth_provider",
                return_value=self._make_provider(module),
            ),
        ):
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Original-URL": "https://example.com/api/agents/other-agent",
                },
            )
        assert response.status_code == 403
        body = response.json()["detail"]
        assert "secret-agent" not in body
        assert "other-agent" not in body

    def test_resource_token_with_traversal_rejected_at_mint(self, auth_env_vars):
        # The Pydantic validator should refuse to mint
        # a token whose resource.id contains '..' or '%'.
        from fastapi.testclient import TestClient

        import auth_server.server as server_module

        server_module.user_token_generation_counts.clear()
        client = TestClient(server_module.app)
        body = {
            "user_context": {
                "username": "alice",
                "scopes": ["mcp-servers/read"],
                "groups": [],
                "auth_method": "oauth2",
                "provider": "keycloak",
            },
            "requested_scopes": ["mcp-servers/read"],
            "expires_in_hours": 1,
            "resource": {"type": "server", "id": "../admin"},
        }
        response = client.post("/internal/tokens", json=body, headers=_internal_auth_headers())
        assert response.status_code == 422, response.text

    def test_non_string_resource_id_rejected_at_mint(self, auth_env_vars):
        # Pydantic v2 defaults to lax mode and would coerce ``{"id": 123}``
        # to the string ``"123"``. We enable strict mode on
        # ``ResourceBinding`` so numeric/list/dict ids are rejected with a
        # 422 rather than silently coerced into a (likely-useless) token.
        from fastapi.testclient import TestClient

        import auth_server.server as server_module

        server_module.user_token_generation_counts.clear()
        client = TestClient(server_module.app)
        base_body = {
            "user_context": {
                "username": "alice",
                "scopes": ["mcp-servers/read"],
                "groups": [],
                "auth_method": "oauth2",
                "provider": "keycloak",
            },
            "requested_scopes": ["mcp-servers/read"],
            "expires_in_hours": 1,
        }
        for bad in (123, True, ["foo"], {"nested": "foo"}):
            body = {**base_body, "resource": {"type": "server", "id": bad}}
            response = client.post("/internal/tokens", json=body, headers=_internal_auth_headers())
            assert response.status_code == 422, (bad, response.text)

    @pytest.mark.parametrize(
        "bad_id",
        [
            "foo\x00bar",  # null byte — truncates in C parsers
            "foo\nbar",  # LF — CRLF-injection territory
            "foo\rbar",  # CR
            "foo\tbar",  # tab
            "foo\x7fbar",  # DEL
        ],
    )
    def test_resource_id_with_control_characters_rejected_at_mint(self, auth_env_vars, bad_id):
        # Control characters in resource ids could cause truncation in
        # C-backed URL parsers downstream. Reject at the Pydantic model.
        from fastapi.testclient import TestClient

        import auth_server.server as server_module

        server_module.user_token_generation_counts.clear()
        client = TestClient(server_module.app)
        body = {
            "user_context": {
                "username": "alice",
                "scopes": ["mcp-servers/read"],
                "groups": [],
                "auth_method": "oauth2",
                "provider": "keycloak",
            },
            "requested_scopes": ["mcp-servers/read"],
            "expires_in_hours": 1,
            "resource": {"type": "server", "id": bad_id},
        }
        response = client.post("/internal/tokens", json=body, headers=_internal_auth_headers())
        assert response.status_code == 422, response.text
