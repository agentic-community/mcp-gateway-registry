"""
Unit tests for EnforceAI IdentityResolver orchestration (Stage 4.5).
"""

from __future__ import annotations

import base64
import hashlib
import hmac
import sqlite3
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import jwt
import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

from auth_server.enforceai.auth.resolver import (
    IdentityResolver,
)
from auth_server.enforceai.config import (
    OIDCIssuerConfig,
)
from auth_server.enforceai.crypto.keyring import (
    GatewayKeyring,
)
from auth_server.enforceai.db.migrations import (
    upgrade_to_latest,
)
from auth_server.enforceai.errors import (
    DependencyUnavailableError,
    ForbiddenError,
    UnauthorizedError,
)
from auth_server.enforceai.oidc.jwks import (
    JWKSCache,
)
from auth_server.enforceai.oidc.verify import (
    OIDCVerifier,
)
from auth_server.enforceai.providers.api_key import (
    ApiKeyProvider,
)
from auth_server.enforceai.providers.gateway_token import (
    GatewayTokenProvider,
)
from auth_server.enforceai.providers.oidc import (
    OidcProvider,
)
from auth_server.enforceai.stores.sqlite.agent_store import (
    SqliteAgentStore,
)
from auth_server.enforceai.stores.sqlite.api_key_store import (
    SqliteApiKeyStore,
)
from auth_server.enforceai.stores.sqlite.revocation_store import (
    SqliteRevocationStore,
)
from auth_server.enforceai.tokens.mint import (
    mint_gateway_token,
)


def _migrate_db(
    db_path: Path,
) -> None:
    connection = sqlite3.connect(db_path)
    try:
        upgrade_to_latest(connection)
    finally:
        connection.close()


def _b64url_uint(
    value: int,
) -> str:
    data = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(data).decode("ascii").rstrip("=")


def _jwk_from_public_key(
    *,
    public_key: rsa.RSAPublicKey,
    kid: str,
) -> dict[str, Any]:
    numbers = public_key.public_numbers()
    return {
        "kty": "RSA",
        "kid": kid,
        "use": "sig",
        "alg": "RS256",
        "n": _b64url_uint(numbers.n),
        "e": _b64url_uint(numbers.e),
    }


def _compute_api_key_hash(
    *,
    pepper: bytes,
    secret: str,
) -> str:
    return hmac.new(
        pepper,
        secret.encode("utf-8"),
        hashlib.sha256,
    ).hexdigest()


def _issuer_config(
    *,
    jwks_uri: str,
    audiences: list[str],
) -> OIDCIssuerConfig:
    return OIDCIssuerConfig.model_validate(
        {
            "jwks_uri": jwks_uri,
            "audiences": audiences,
            "jwks_cache_ttl_seconds": 9999,
            "clock_skew_seconds": 0,
        }
    )


@pytest.mark.unit
class TestIdentityResolver:
    async def test_mixed_routes_api_key_gateway_token_and_oidc(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
        enforceai_gateway_key_files,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        api_key_store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)
        revocation_store = SqliteRevocationStore(db_path=enforceai_sqlite_db_path)

        now_epoch_seconds = int(time.time())
        now = datetime.fromtimestamp(now_epoch_seconds, tz=timezone.utc).replace(microsecond=0)

        gateway_issuer = "enforceai-gateway"
        oidc_issuer = "https://issuer.example"
        oidc_jwks_uri = "https://issuer.example/jwks.json"
        oidc_aud = "mcp-registry"
        oidc_kid = "kid-1"

        oidc_private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        oidc_private_pem = oidc_private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        oidc_public_key = oidc_private_key.public_key()
        assert isinstance(oidc_public_key, rsa.RSAPublicKey)

        jwks_payload = {"keys": [_jwk_from_public_key(public_key=oidc_public_key, kid=oidc_kid)]}

        async def fetcher(uri: str) -> dict[str, Any]:
            assert uri == oidc_jwks_uri
            return jwks_payload

        oidc_verifier = OIDCVerifier(
            issuers={
                oidc_issuer: _issuer_config(
                    jwks_uri=oidc_jwks_uri,
                    audiences=[oidc_aud],
                )
            },
            jwks_cache=JWKSCache(fetcher=fetcher),
            now=lambda: now_epoch_seconds,
        )

        oidc_provider = OidcProvider(
            verifier=oidc_verifier,
            agent_store=agent_store,
        )

        pepper = b"pepper-1"
        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(pepper)
        api_key_provider = ApiKeyProvider(
            api_key_store=api_key_store,
            agent_store=agent_store,
            pepper_path=pepper_path,
        )

        key_files = enforceai_gateway_key_files
        gateway_provider = GatewayTokenProvider(
            agent_store=agent_store,
            revocation_store=revocation_store,
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
            expected_issuer=gateway_issuer,
        )

        resolver = IdentityResolver(
            auth_provider="mixed",
            oidc_provider=oidc_provider,
            api_key_provider=api_key_provider,
            gateway_token_provider=gateway_provider,
            gateway_issuer=gateway_issuer,
            oidc_issuers={oidc_issuer},
        )

        user_subject = "user-1"
        oidc_user_id = f"{oidc_issuer}|{user_subject}"

        agent_id_oidc = str(uuid.uuid4())
        agent_store.create_agent(
            user_id=oidc_user_id,
            agent_id=agent_id_oidc,
            scopes=[
                "agent-oidc-1",
            ],
        )

        oidc_token = jwt.encode(
            {
                "iss": oidc_issuer,
                "sub": user_subject,
                "aud": oidc_aud,
                "iat": now_epoch_seconds,
                "exp": now_epoch_seconds + 300,
                "scope": "idp-scope-1",
                "roles": ["idp-role-1"],
            },
            key=oidc_private_pem,
            algorithm="RS256",
            headers={"kid": oidc_kid},
        )

        resolved_oidc = await resolver.resolve_identity(
            headers={
                "Authorization": f"Bearer {oidc_token}",
                "X-Agent-Id": agent_id_oidc,
            }
        )
        assert resolved_oidc.provider == "oidc"
        assert resolved_oidc.user_id == oidc_user_id
        assert resolved_oidc.agent_id == agent_id_oidc
        assert resolved_oidc.scopes == ["agent-oidc-1"]

        user_id_gateway = "https://issuer.example|sub-1"
        agent_id_gateway = str(uuid.uuid4())
        agent_store.create_agent(
            user_id=user_id_gateway,
            agent_id=agent_id_gateway,
            scopes=[
                "a",
                "b",
            ],
        )

        gateway_keyring = GatewayKeyring.load(
            private_key_path=key_files.private_key_path,
            public_keys_dir=key_files.public_keys_dir,
            active_kid=key_files.active_kid,
        )
        gateway_token = mint_gateway_token(
            keyring=gateway_keyring,
            issuer=gateway_issuer,
            user_id=user_id_gateway,
            agent_id=agent_id_gateway,
            scopes=[
                "b",
                "c",
            ],
            issued_at=now,
            ttl_seconds=3600,
            jti="jti-1",
        )

        resolved_gateway = await resolver.resolve_identity(
            headers={
                "Authorization": f"Bearer {gateway_token}",
            }
        )
        assert resolved_gateway.provider == "gateway-token"
        assert resolved_gateway.user_id == user_id_gateway
        assert resolved_gateway.agent_id == agent_id_gateway
        assert resolved_gateway.scopes == ["b"]

        agent_id_api_key = str(uuid.uuid4())
        agent_store.create_agent(
            user_id=user_id_gateway,
            agent_id=agent_id_api_key,
            scopes=[
                "s1",
                "s2",
            ],
        )

        api_key_secret = "secret-1"
        api_key_store.create_key(
            key_id="key-1",
            secret_hash=_compute_api_key_hash(
                pepper=pepper,
                secret=api_key_secret,
            ),
            user_id=user_id_gateway,
            agent_id=agent_id_api_key,
            scopes=[
                "s2",
            ],
        )

        resolved_api_key = await resolver.resolve_identity(
            headers={
                "X-API-Key": f"eak_key-1.{api_key_secret}",
            }
        )
        assert resolved_api_key.provider == "api-key"
        assert resolved_api_key.user_id == user_id_gateway
        assert resolved_api_key.agent_id == agent_id_api_key
        assert resolved_api_key.scopes == ["s2"]

    async def test_oidc_mode_requires_x_agent_id(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)

        verifier = OIDCVerifier(
            issuers={},
            jwks_cache=JWKSCache(fetcher=lambda _uri: pytest.fail("fetcher called")),
        )
        oidc_provider = OidcProvider(
            verifier=verifier,
            agent_store=agent_store,
        )

        resolver = IdentityResolver(
            auth_provider="oidc",
            oidc_provider=oidc_provider,
        )

        with pytest.raises(ForbiddenError, match="Missing X-Agent-Id"):
            await resolver.resolve_identity(
                headers={
                    "Authorization": "Bearer dummy",
                }
            )

    async def test_api_key_mode_rejects_bearer(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)
        api_key_store = SqliteApiKeyStore(db_path=enforceai_sqlite_db_path)

        pepper_path = tmp_path / "pepper"
        pepper_path.write_bytes(b"pepper")
        api_key_provider = ApiKeyProvider(
            api_key_store=api_key_store,
            agent_store=agent_store,
            pepper_path=pepper_path,
        )

        resolver = IdentityResolver(
            auth_provider="api-key",
            api_key_provider=api_key_provider,
        )

        with pytest.raises(UnauthorizedError):
            await resolver.resolve_identity(
                headers={
                    "Authorization": "Bearer dummy",
                }
            )

    async def test_mixed_rejects_unknown_bearer_issuer(
        self,
    ) -> None:
        token = jwt.encode(
            {
                "iss": "https://unknown-issuer.example",
                "sub": "user-1",
                "aud": "mcp-registry",
                "iat": 1,
                "exp": 2,
            },
            key="shared-secret",
            algorithm="HS256",
        )

        resolver = IdentityResolver(
            auth_provider="mixed",
            gateway_issuer="enforceai-gateway",
            oidc_issuers={"https://issuer.example"},
        )

        with pytest.raises(UnauthorizedError):
            await resolver.resolve_identity(
                headers={
                    "Authorization": f"Bearer {token}",
                }
            )

    async def test_dependency_failure_is_503(
        self,
        tmp_path: Path,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)

        now_epoch_seconds = int(time.time())
        issuer = "https://issuer.example"
        jwks_uri = "https://issuer.example/jwks.json"
        aud = "mcp-registry"
        kid = "kid-1"

        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )
        public_key = private_key.public_key()
        assert isinstance(public_key, rsa.RSAPublicKey)

        async def fetcher(uri: str) -> dict[str, Any]:
            assert uri == jwks_uri
            return {"keys": [_jwk_from_public_key(public_key=public_key, kid=kid)]}

        verifier = OIDCVerifier(
            issuers={
                issuer: _issuer_config(
                    jwks_uri=jwks_uri,
                    audiences=[aud],
                )
            },
            jwks_cache=JWKSCache(fetcher=fetcher),
            now=lambda: now_epoch_seconds,
        )
        oidc_provider = OidcProvider(
            verifier=verifier,
            agent_store=agent_store,
        )

        def _boom(*args: object, **kwargs: object) -> object:
            raise RuntimeError("boom")

        agent_store.get_agent_by_id = _boom  # type: ignore[method-assign]

        resolver = IdentityResolver(
            auth_provider="oidc",
            oidc_provider=oidc_provider,
        )

        token = jwt.encode(
            {
                "iss": issuer,
                "sub": "user-1",
                "aud": aud,
                "iat": now_epoch_seconds,
                "exp": now_epoch_seconds + 300,
            },
            key=private_pem,
            algorithm="RS256",
            headers={"kid": kid},
        )

        with pytest.raises(DependencyUnavailableError):
            await resolver.resolve_identity(
                headers={
                    "Authorization": f"Bearer {token}",
                    "X-Agent-Id": str(uuid.uuid4()),
                }
            )
