"""
Hardening tests for Stage 3.5 OIDC components.

These tests ensure malformed inputs map to Unauthorized and that sensitive
inputs (raw tokens, JWKS payloads) are not leaked via exception messages.
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
from auth_server.enforceai.errors import (
    DependencyUnavailableError,
    UnauthorizedError,
)
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
            "clock_skew_seconds": 0,
        }
    )


@pytest.mark.unit
class TestOIDCHardening:
    async def test_malformed_token_is_unauthorized_and_not_leaked(self):
        token = "not-a-jwt"
        verifier = OIDCVerifier(
            issuers={},
            jwks_cache=JWKSCache(fetcher=lambda _uri: pytest.fail("fetcher called")),
        )

        with pytest.raises(UnauthorizedError) as exc:
            await verifier.verify_bearer_token(token)

        assert token not in str(exc.value)

    async def test_missing_iss_is_unauthorized_and_does_not_fetch(self):
        token = jwt.encode(
            {
                "sub": "user-1",
                "aud": "mcp-registry",
                "iat": 1,
                "exp": 2,
            },
            key="shared-secret",
            algorithm="HS256",
        )

        verifier = OIDCVerifier(
            issuers={},
            jwks_cache=JWKSCache(fetcher=lambda _uri: pytest.fail("fetcher called")),
        )

        with pytest.raises(UnauthorizedError, match="iss"):
            await verifier.verify_bearer_token(token)

    async def test_missing_sub_is_unauthorized_and_does_not_fetch(self):
        token = jwt.encode(
            {
                "iss": "https://issuer.example",
                "aud": "mcp-registry",
                "iat": 1,
                "exp": 2,
            },
            key="shared-secret",
            algorithm="HS256",
        )

        verifier = OIDCVerifier(
            issuers={},
            jwks_cache=JWKSCache(fetcher=lambda _uri: pytest.fail("fetcher called")),
        )

        with pytest.raises(UnauthorizedError, match="sub"):
            await verifier.verify_bearer_token(token)

    async def test_missing_aud_claim_is_unauthorized(self):
        now_epoch_seconds = int(time.time())
        issuer = "https://issuer.example"
        jwks_uri = "https://issuer.example/jwks.json"
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
                    audiences=["mcp-registry"],
                )
            },
            jwks_cache=JWKSCache(fetcher=fetcher),
            now=lambda: now_epoch_seconds,
        )

        token = jwt.encode(
            {
                "iss": issuer,
                "sub": "user-1",
                "iat": now_epoch_seconds,
                "exp": now_epoch_seconds + 300,
            },
            key=private_pem,
            algorithm="RS256",
            headers={"kid": kid},
        )

        with pytest.raises(UnauthorizedError):
            await verifier.verify_bearer_token(token)

    async def test_jwks_payload_is_not_leaked_in_dependency_errors(self):
        marker = "DO_NOT_LEAK_THIS_JWKS_PAYLOAD"
        issuer = "https://issuer.example"
        jwks_uri = "https://issuer.example/jwks.json"

        async def fetcher(uri: str) -> dict[str, Any]:
            assert uri == jwks_uri
            return {
                "keys": [
                    {
                        "kty": "RSA",
                        "kid": "kid-1",
                        "n": marker,
                        "e": "AQAB",
                    }
                ]
            }

        verifier = OIDCVerifier(
            issuers={
                issuer: _issuer_config(
                    jwks_uri=jwks_uri,
                    audiences=["mcp-registry"],
                )
            },
            jwks_cache=JWKSCache(fetcher=fetcher),
            now=lambda: 1,
        )

        token = jwt.encode(
            {
                "iss": issuer,
                "sub": "user-1",
                "aud": "mcp-registry",
                "iat": 1,
                "exp": 2,
            },
            key="shared-secret",
            algorithm="HS256",
            headers={"kid": "kid-1"},
        )

        with pytest.raises((UnauthorizedError, DependencyUnavailableError)) as exc:
            await verifier.verify_bearer_token(token)

        assert marker not in str(exc.value)
        assert marker not in getattr(exc.value, "public_message", "")

