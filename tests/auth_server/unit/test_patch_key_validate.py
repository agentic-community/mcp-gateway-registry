"""Unit tests for the patch-key path in auth_server /validate (wire-platform-v1 2.1).

Covers the acceptance flow mint -> authenticate -> revoke -> 401 at the
/validate boundary, plus the two non-interference guarantees:

- a `wgk-` bearer NEVER reaches the self-signed JWT decoder or the IdP
  provider validator;
- JWT / static REGISTRY_API_KEYS credentials behave exactly as before with
  the patch-key branch compiled in.

The patch_keys collection is faked via a dict-backed service mock; the scope
repository is the shared in-memory fixture so group->scope mapping works.
"""

import logging
from datetime import datetime
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from registry.schemas.patch_key import (
    PATCH_KEY_STATUS_ACTIVE,
    PATCH_KEY_STATUS_REVOKED,
    PatchKeyInfo,
)

logger = logging.getLogger(__name__)

pytestmark = [pytest.mark.unit, pytest.mark.auth]

# A representative minted key (shape: "wgk-" + token_urlsafe(32) == 43 chars).
_PLAINTEXT_KEY = "wgk-A1b2C3d4E5f6G7h8I9j0K1l2M3n4O5p5Q6r7S8t9U0v"


def _key_info(status: str = PATCH_KEY_STATUS_ACTIVE) -> PatchKeyInfo:
    return PatchKeyInfo(
        key_id="keyid001",
        name="ci-runner",
        key_prefix=_PLAINTEXT_KEY[:12],
        username="alice",
        email="alice@example.com",
        provider="oauth2",
        groups=["developers", "users"],
        status=status,
        created_at=datetime.utcnow(),
        last_used_at=None,
        revoked_at=None,
    )


class _FakePatchKeyService:
    """Dict-backed stand-in for PatchKeyService: one active key."""

    def __init__(self, info: PatchKeyInfo | None, plaintext: str) -> None:
        self._info = info
        self._plaintext = plaintext
        self.last_used_calls: list[str] = []

    async def verify_key(self, plaintext: str) -> PatchKeyInfo | None:
        # mirrors the real service: only ACTIVE keys verify
        if (
            self._info is not None
            and plaintext == self._plaintext
            and self._info.status == PATCH_KEY_STATUS_ACTIVE
        ):
            return self._info
        return None

    async def touch_last_used(self, key_id: str) -> None:
        self.last_used_calls.append(key_id)


def _patched_service(service: _FakePatchKeyService):
    """Patch the lazily-imported service getter used inside _validate_patch_key_token."""
    return patch(
        "registry.services.patch_key_service.get_patch_key_service",
        new=AsyncMock(return_value=service),
    )


@pytest.fixture
def scope_repo(mock_scope_repository_with_data):
    with patch(
        "auth_server.server.get_scope_repository",
        return_value=mock_scope_repository_with_data,
    ):
        yield mock_scope_repository_with_data


class TestPatchKeyHappyPath:
    """/validate accepts an active key on both registry API and MCP paths."""

    def test_registry_api_path_accepted(self, scope_repo, auth_env_vars):
        import auth_server.server as server_module

        service = _FakePatchKeyService(_key_info(), _PLAINTEXT_KEY)
        with _patched_service(service):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {_PLAINTEXT_KEY}",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

        assert response.status_code == 200
        data = response.json()
        assert data["valid"] is True
        assert data["method"] == "patch-key"
        assert data["username"] == "alice"
        assert data["groups"] == ["developers", "users"]
        # scopes are resolved from the CURRENT group mappings at call time
        assert "read:servers" in data["scopes"]
        assert "write:servers" in data["scopes"]
        assert response.headers["X-Auth-Method"] == "patch-key"
        assert response.headers["X-Username"] == "alice"
        assert "developers users" == response.headers["X-Groups"]
        # last-used metadata refresh fired
        assert service.last_used_calls == ["keyid001"]

    def test_mcp_server_path_accepted_and_scope_checked(self, scope_repo, auth_env_vars):
        """MCP proxy path: the key authenticates AND per-server scope
        validation still runs (the key holder is treated like the user
        holding a JWT, not like a blanket-bypass static token)."""
        import auth_server.server as server_module

        service = _FakePatchKeyService(_key_info(), _PLAINTEXT_KEY)
        with _patched_service(service):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {_PLAINTEXT_KEY}",
                    "X-Original-URL": "https://example.com/test-server/mcp",
                },
            )

        assert response.status_code == 200
        assert response.json()["method"] == "patch-key"
        assert response.json()["server_name"] == "test-server"

    def test_mcp_server_path_denied_without_scope(self, scope_repo, auth_env_vars):
        """A key whose owner lacks the server scope gets 403 -- same as a JWT."""
        import auth_server.server as server_module

        info = _key_info()
        # "users" group maps only to read scopes for test-server; the
        # unknown-server is covered by no scope -> fail closed.
        info = info.model_copy(update={"groups": ["users"]})
        service = _FakePatchKeyService(info, _PLAINTEXT_KEY)
        with _patched_service(service):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {_PLAINTEXT_KEY}",
                    "X-Original-URL": "https://example.com/unknown-server/mcp",
                },
            )

        assert response.status_code == 403

    def test_registry_ui_token_minted_for_api_hop(self, scope_repo, auth_env_vars):
        """With nginx's registry-API marker present, /validate also mints the
        X-Internal-Token-Registry hop token so /api/ requests work behind nginx."""
        import auth_server.server as server_module

        service = _FakePatchKeyService(_key_info(), _PLAINTEXT_KEY)
        with _patched_service(service):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {_PLAINTEXT_KEY}",
                    "X-Original-URL": "https://example.com/api/servers",
                    "X-Registry-Api-Auth": "1",
                },
            )

        assert response.status_code == 200
        assert response.headers.get("X-Internal-Token-Registry")


class TestPatchKeyRejection:
    """Unknown / revoked keys 401 immediately (revocation is next-call)."""

    def test_unknown_key_401(self, scope_repo, auth_env_vars):
        import auth_server.server as server_module

        service = _FakePatchKeyService(None, _PLAINTEXT_KEY)
        with _patched_service(service):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer wgk-not-a-real-key-value-0000000000000",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )

        assert response.status_code == 401
        assert response.headers.get("WWW-Authenticate") == "Bearer"

    def test_revoked_key_401_immediately(self, scope_repo, auth_env_vars):
        """mint -> 200, revoke (status flip in the store) -> next call 401."""
        import auth_server.server as server_module

        service = _FakePatchKeyService(_key_info(), _PLAINTEXT_KEY)
        with _patched_service(service):
            client = TestClient(server_module.app)
            ok = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {_PLAINTEXT_KEY}",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )
            assert ok.status_code == 200

            # Revoke: the store now reports the key as revoked/absent.
            service._info = _key_info(PATCH_KEY_STATUS_REVOKED)

            denied = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {_PLAINTEXT_KEY}",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )
        assert denied.status_code == 401

    def test_disabled_feature_falls_through_to_jwt_and_401s(self, scope_repo, auth_env_vars):
        import auth_server.server as server_module

        service = _FakePatchKeyService(_key_info(), _PLAINTEXT_KEY)
        mock_provider = MagicMock()
        mock_provider.validate_token = MagicMock(side_effect=ValueError("Invalid token"))
        with (
            _patched_service(service),
            patch.object(server_module, "PATCH_KEY_AUTH_ENABLED", False),
            patch("auth_server.server.get_auth_provider", return_value=mock_provider),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {_PLAINTEXT_KEY}",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )
        assert response.status_code == 401
        # the service was never consulted
        assert service.last_used_calls == []

    def test_datastore_error_fails_closed(self, scope_repo, auth_env_vars):
        import auth_server.server as server_module

        broken = MagicMock()
        broken.verify_key = AsyncMock(side_effect=RuntimeError("db down"))
        with patch(
            "registry.services.patch_key_service.get_patch_key_service",
            new=AsyncMock(return_value=broken),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {_PLAINTEXT_KEY}",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )
        assert response.status_code == 401


class TestPatchKeyLogSafety:
    """The plaintext never reaches any log line (DEBUG included)."""

    def test_success_path_never_logs_plaintext(self, scope_repo, auth_env_vars, caplog):
        import auth_server.server as server_module

        service = _FakePatchKeyService(_key_info(), _PLAINTEXT_KEY)
        with _patched_service(service):
            client = TestClient(server_module.app)
            with caplog.at_level(logging.DEBUG):
                response = client.get(
                    "/validate",
                    headers={
                        "Authorization": f"Bearer {_PLAINTEXT_KEY}",
                        "X-Original-URL": "https://example.com/api/servers",
                    },
                )
        assert response.status_code == 200
        assert _PLAINTEXT_KEY not in caplog.text
        # not even a fragment of the body after the prefix
        assert _PLAINTEXT_KEY[4:20] not in caplog.text

    def test_rejection_path_never_logs_plaintext(self, scope_repo, auth_env_vars, caplog):
        import auth_server.server as server_module

        bad_key = "wgk-RejectedKeyValueThatMustNotAppearAnywhere0000"
        service = _FakePatchKeyService(None, bad_key)
        with _patched_service(service):
            client = TestClient(server_module.app)
            with caplog.at_level(logging.DEBUG):
                response = client.get(
                    "/validate",
                    headers={
                        "Authorization": f"Bearer {bad_key}",
                        "X-Original-URL": "https://example.com/api/servers",
                    },
                )
        assert response.status_code == 401
        assert bad_key not in caplog.text
        assert bad_key[4:20] not in caplog.text


class TestPatchKeyNonInterference:
    """JWT and static-key credentials behave exactly as before."""

    def test_self_signed_jwt_still_validated(self, scope_repo, auth_env_vars):
        import time as _time

        # Mint a self-signed token with the module's ACTUAL signing key (the
        # shared fixture signs with a different env key than the one the
        # module bound at import time).
        import jwt as pyjwt

        import auth_server.server as server_module

        now = int(_time.time())
        token = pyjwt.encode(
            {
                "iss": server_module.JWT_ISSUER,
                "aud": server_module._USER_JWT_AUDIENCE,
                "sub": "alice",
                "scope": "read:servers",
                "groups": ["developers"],
                "exp": now + 3600,
                "iat": now,
                "token_use": "access",
                "client_id": "user-generated",
                # required by the self-signed edge guard on every gateway JWT
                "token_kind": "user",
            },
            server_module.SECRET_KEY,
            algorithm="HS256",
        )

        service = _FakePatchKeyService(_key_info(), _PLAINTEXT_KEY)
        with _patched_service(service):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )
        assert response.status_code == 200
        assert response.json()["method"] == "self_signed"
        # the patch-key service was never consulted for a JWT bearer
        assert service.last_used_calls == []

    def test_static_registry_key_still_validated(self, scope_repo, auth_env_vars):
        """The REGISTRY_API_TOKEN / REGISTRY_API_KEYS mechanism is untouched."""
        import auth_server.server as server_module

        service = _FakePatchKeyService(_key_info(), _PLAINTEXT_KEY)
        token_map = {
            "legacy": {
                "key_bytes": b"test-api-key-static-32-chars-minimum",
                "groups": ["mcp-registry-admin"],
                "scopes": ["mcp-registry-admin"],
            }
        }
        with (
            _patched_service(service),
            patch.object(server_module, "REGISTRY_STATIC_TOKEN_AUTH_ENABLED", True),
            patch.object(server_module, "_STATIC_TOKEN_MAP", token_map),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer test-api-key-static-32-chars-minimum",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )
        assert response.status_code == 200
        assert response.json()["method"] == "network-trusted"
        assert service.last_used_calls == []

    def test_non_wgk_garbage_bearer_reaches_provider_as_before(self, scope_repo, auth_env_vars):
        """A non-wgk, non-JWT bearer still falls through to the provider
        validator (pre-existing behavior) and 401s there."""
        import auth_server.server as server_module

        mock_provider = MagicMock()
        mock_provider.validate_token = MagicMock(side_effect=ValueError("Invalid token"))
        service = _FakePatchKeyService(_key_info(), _PLAINTEXT_KEY)
        with (
            _patched_service(service),
            patch("auth_server.server.get_auth_provider", return_value=mock_provider),
        ):
            client = TestClient(server_module.app)
            response = client.get(
                "/validate",
                headers={
                    "Authorization": "Bearer not-a-jwt-not-a-wgk-key",
                    "X-Original-URL": "https://example.com/api/servers",
                },
            )
        assert response.status_code == 401
        mock_provider.validate_token.assert_called()
