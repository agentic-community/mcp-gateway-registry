"""
Unit tests for Stage 3.4 OIDC verifier error semantics.
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
            "clock_skew_seconds": 0,
        }
    )


@pytest.mark.unit
class TestOIDCVerifyErrors:
    async def test_expired_token_is_unauthorized(self):
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

        token = jwt.encode(
            {
                "iss": issuer,
                "sub": "user-1",
                "aud": aud,
                "iat": now_epoch_seconds - 20,
                "exp": now_epoch_seconds - 10,
            },
            key=private_pem,
            algorithm="RS256",
            headers={"kid": kid},
        )

        with pytest.raises(UnauthorizedError):
            await verifier.verify_bearer_token(token)

    async def test_tampered_token_is_unauthorized(self):
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

        parts = token.split(".")
        assert len(parts) == 3
        parts[1] = parts[1][::-1]
        tampered = ".".join(parts)

        with pytest.raises(UnauthorizedError):
            await verifier.verify_bearer_token(tampered)

    async def test_algorithm_mismatch_is_unauthorized(self):
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

        token = jwt.encode(
            {
                "iss": issuer,
                "sub": "user-1",
                "aud": aud,
                "iat": now_epoch_seconds,
                "exp": now_epoch_seconds + 300,
            },
            key="shared-secret",
            algorithm="HS256",
            headers={"kid": kid},
        )

        with pytest.raises(UnauthorizedError):
            await verifier.verify_bearer_token(token)

    async def test_iat_too_far_in_future_is_unauthorized(self):
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

        token = jwt.encode(
            {
                "iss": issuer,
                "sub": "user-1",
                "aud": aud,
                "iat": now_epoch_seconds + 10,
                "exp": now_epoch_seconds + 300,
            },
            key=private_pem,
            algorithm="RS256",
            headers={"kid": kid},
        )

        with pytest.raises(UnauthorizedError, match="iat"):
            await verifier.verify_bearer_token(token)
