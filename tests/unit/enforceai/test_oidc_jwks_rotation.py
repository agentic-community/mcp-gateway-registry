"""
Unit tests for Stage 3.4 JWKS rotation behavior (refresh-on-missing-kid).
"""

from __future__ import annotations

import base64
import time
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import jwt

from auth_server.enforceai.config import OIDCIssuerConfig
from auth_server.enforceai.errors import UnauthorizedError
from auth_server.enforceai.oidc.jwks import JWKSCache
from auth_server.enforceai.oidc.verify import OIDCVerifier


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
        }
    )


@pytest.mark.unit
class TestOIDCJWKSRotation:
    async def test_missing_kid_triggers_single_refresh(self):
        now_epoch_seconds = int(time.time())
        issuer = "https://issuer.example"
        jwks_uri = "https://issuer.example/jwks.json"
        aud = "mcp-registry"

        kid = "kid-rotated"
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

        initial_jwks = {"keys": [{"kty": "RSA", "kid": "old-kid", "n": "x", "e": "AQAB"}]}
        refreshed_jwks = {"keys": [_jwk_from_public_key(public_key=public_key, kid=kid)]}

        calls: list[str] = []

        async def fetcher(uri: str) -> dict[str, Any]:
            calls.append(uri)
            if len(calls) == 1:
                return initial_jwks
            return refreshed_jwks

        issuers = {
            issuer: _issuer_config(
                jwks_uri=jwks_uri,
                audiences=[aud],
            )
        }

        verifier = OIDCVerifier(
            issuers=issuers,
            jwks_cache=JWKSCache(fetcher=fetcher),
            now=lambda: now_epoch_seconds,
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

        verified = await verifier.verify_bearer_token(token)
        assert verified.user_id == f"{issuer}|user-1"
        assert calls == [
            jwks_uri,
            jwks_uri,
        ]

    async def test_missing_kid_after_refresh_is_unauthorized(self):
        now_epoch_seconds = int(time.time())
        issuer = "https://issuer.example"
        jwks_uri = "https://issuer.example/jwks.json"
        aud = "mcp-registry"

        kid = "kid-missing"
        private_key = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        private_pem = private_key.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        jwks_payload = {"keys": [{"kty": "RSA", "kid": "other-kid", "n": "x", "e": "AQAB"}]}
        calls: list[str] = []

        async def fetcher(uri: str) -> dict[str, Any]:
            calls.append(uri)
            return jwks_payload

        issuers = {
            issuer: _issuer_config(
                jwks_uri=jwks_uri,
                audiences=[aud],
            )
        }

        verifier = OIDCVerifier(
            issuers=issuers,
            jwks_cache=JWKSCache(fetcher=fetcher),
            now=lambda: now_epoch_seconds,
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

        with pytest.raises(UnauthorizedError):
            await verifier.verify_bearer_token(token)

        assert calls == [
            jwks_uri,
            jwks_uri,
        ]

