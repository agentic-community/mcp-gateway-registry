"""
Unit tests for EnforceAI OIDC provider (Stage 4.4).
"""

from __future__ import annotations

import base64
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

from auth_server.enforceai.config import (
    OIDCIssuerConfig,
)
from auth_server.enforceai.db.migrations import (
    upgrade_to_latest,
)
from auth_server.enforceai.errors import (
    ForbiddenError,
    UnauthorizedError,
)
from auth_server.enforceai.oidc.jwks import (
    JWKSCache,
)
from auth_server.enforceai.oidc.verify import (
    OIDCVerifier,
)
from auth_server.enforceai.providers.oidc import (
    OidcProvider,
)
from auth_server.enforceai.stores.sqlite.agent_store import (
    SqliteAgentStore,
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


def _make_rs256_token(
    *,
    private_pem: bytes,
    kid: str,
    issuer: str,
    subject: str,
    audience: str,
    now_epoch_seconds: int,
    scopes: list[str],
    roles: list[str],
) -> str:
    return jwt.encode(
        {
            "iss": issuer,
            "sub": subject,
            "aud": audience,
            "iat": now_epoch_seconds,
            "exp": now_epoch_seconds + 300,
            "scope": " ".join(scopes),
            "roles": roles,
        },
        key=private_pem,
        algorithm="RS256",
        headers={"kid": kid},
    )


@pytest.mark.unit
class TestOidcProvider:
    async def test_missing_x_agent_id_is_forbidden(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)

        verifier = OIDCVerifier(
            issuers={},
            jwks_cache=JWKSCache(fetcher=lambda _uri: pytest.fail("fetcher called")),
        )
        provider = OidcProvider(
            verifier=verifier,
            agent_store=agent_store,
        )

        with pytest.raises(ForbiddenError, match="Missing X-Agent-Id"):
            await provider.resolve_identity(
                bearer_token="dummy",
                agent_id_header=None,
            )

    async def test_invalid_x_agent_id_is_forbidden(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)

        verifier = OIDCVerifier(
            issuers={},
            jwks_cache=JWKSCache(fetcher=lambda _uri: pytest.fail("fetcher called")),
        )
        provider = OidcProvider(
            verifier=verifier,
            agent_store=agent_store,
        )

        with pytest.raises(ForbiddenError, match="Invalid X-Agent-Id"):
            await provider.resolve_identity(
                bearer_token="dummy",
                agent_id_header="not-a-uuid",
            )

    async def test_happy_path_binds_agent_and_uses_agent_scopes_only(
        self,
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

        jwks_payload = {"keys": [_jwk_from_public_key(public_key=public_key, kid=kid)]}

        async def fetcher(uri: str) -> dict[str, Any]:
            assert uri == jwks_uri
            return jwks_payload

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

        subject = "user-1"
        user_id = f"{issuer}|{subject}"
        agent_id = str(uuid.uuid4())
        agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=[
                "agent-scope-1",
                "agent-scope-2",
            ],
        )

        token = _make_rs256_token(
            private_pem=private_pem,
            kid=kid,
            issuer=issuer,
            subject=subject,
            audience=aud,
            now_epoch_seconds=now_epoch_seconds,
            scopes=["idp-scope-1"],
            roles=["idp-role-1"],
        )

        provider = OidcProvider(
            verifier=verifier,
            agent_store=agent_store,
        )

        identity = await provider.resolve_identity(
            bearer_token=token,
            agent_id_header=agent_id,
            now=datetime.fromtimestamp(now_epoch_seconds, tz=timezone.utc),
        )
        assert identity.provider == "oidc"
        assert identity.user_id == user_id
        assert identity.agent_id == agent_id
        assert identity.scopes == [
            "agent-scope-1",
            "agent-scope-2",
        ]
        assert identity.user_roles == ["idp-role-1"]
        assert identity.metadata is not None
        assert identity.metadata["issuer"] == issuer
        assert identity.metadata["oidc_scopes"] == ["idp-scope-1"]
        assert identity.metadata["oidc_roles"] == ["idp-role-1"]

    async def test_agent_not_found_denied(
        self,
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

        jwks_payload = {"keys": [_jwk_from_public_key(public_key=public_key, kid=kid)]}

        async def fetcher(uri: str) -> dict[str, Any]:
            assert uri == jwks_uri
            return jwks_payload

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

        token = _make_rs256_token(
            private_pem=private_pem,
            kid=kid,
            issuer=issuer,
            subject="user-1",
            audience=aud,
            now_epoch_seconds=now_epoch_seconds,
            scopes=[],
            roles=[],
        )

        provider = OidcProvider(
            verifier=verifier,
            agent_store=agent_store,
        )

        with pytest.raises(ForbiddenError, match="Agent not found"):
            await provider.resolve_identity(
                bearer_token=token,
                agent_id_header=str(uuid.uuid4()),
            )

    async def test_agent_wrong_owner_denied(
        self,
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

        jwks_payload = {"keys": [_jwk_from_public_key(public_key=public_key, kid=kid)]}

        async def fetcher(uri: str) -> dict[str, Any]:
            assert uri == jwks_uri
            return jwks_payload

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

        agent_id = str(uuid.uuid4())
        agent_store.create_agent(
            user_id=f"{issuer}|other-user",
            agent_id=agent_id,
            scopes=["agent-scope-1"],
        )

        token = _make_rs256_token(
            private_pem=private_pem,
            kid=kid,
            issuer=issuer,
            subject="user-1",
            audience=aud,
            now_epoch_seconds=now_epoch_seconds,
            scopes=[],
            roles=[],
        )

        provider = OidcProvider(
            verifier=verifier,
            agent_store=agent_store,
        )

        with pytest.raises(ForbiddenError, match="ownership mismatch"):
            await provider.resolve_identity(
                bearer_token=token,
                agent_id_header=agent_id,
            )

    async def test_agent_revoked_denied(
        self,
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

        jwks_payload = {"keys": [_jwk_from_public_key(public_key=public_key, kid=kid)]}

        async def fetcher(uri: str) -> dict[str, Any]:
            assert uri == jwks_uri
            return jwks_payload

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

        subject = "user-1"
        user_id = f"{issuer}|{subject}"
        agent_id = str(uuid.uuid4())
        agent_store.create_agent(
            user_id=user_id,
            agent_id=agent_id,
            scopes=["agent-scope-1"],
        )
        agent_store.revoke_agent(agent_id=agent_id)

        token = _make_rs256_token(
            private_pem=private_pem,
            kid=kid,
            issuer=issuer,
            subject=subject,
            audience=aud,
            now_epoch_seconds=now_epoch_seconds,
            scopes=[],
            roles=[],
        )

        provider = OidcProvider(
            verifier=verifier,
            agent_store=agent_store,
        )

        with pytest.raises(ForbiddenError, match="Agent revoked"):
            await provider.resolve_identity(
                bearer_token=token,
                agent_id_header=agent_id,
            )

    async def test_unknown_issuer_is_unauthorized(
        self,
        enforceai_sqlite_db_path: Path,
    ) -> None:
        _migrate_db(enforceai_sqlite_db_path)
        agent_store = SqliteAgentStore(db_path=enforceai_sqlite_db_path)

        verifier = OIDCVerifier(
            issuers={},
            jwks_cache=JWKSCache(fetcher=lambda _uri: pytest.fail("fetcher called")),
        )
        provider = OidcProvider(
            verifier=verifier,
            agent_store=agent_store,
        )

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
            headers={"kid": "kid-ignored"},
        )

        with pytest.raises(UnauthorizedError):
            await provider.resolve_identity(
                bearer_token=token,
                agent_id_header=str(uuid.uuid4()),
            )

