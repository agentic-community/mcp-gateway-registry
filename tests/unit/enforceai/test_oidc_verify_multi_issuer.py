"""
Unit tests for Stage 3.4 OIDC verification pipeline (multi-issuer selection).
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


def _generate_rsa_private_key_pem() -> bytes:
    private_key = rsa.generate_private_key(
        public_exponent=65537,
        key_size=2048,
    )
    return private_key.private_bytes(
        encoding=serialization.Encoding.PEM,
        format=serialization.PrivateFormat.PKCS8,
        encryption_algorithm=serialization.NoEncryption(),
    )


def _issuer_config(
    *,
    jwks_uri: str,
    audiences: list[str],
    ttl_seconds: int = 60,
) -> OIDCIssuerConfig:
    return OIDCIssuerConfig.model_validate(
        {
            "jwks_uri": jwks_uri,
            "audiences": audiences,
            "jwks_cache_ttl_seconds": ttl_seconds,
        }
    )


def _make_token(
    *,
    private_key_pem: bytes,
    kid: str,
    issuer: str,
    subject: str,
    audience: str,
    now_epoch_seconds: int,
    exp_offset_seconds: int = 300,
) -> str:
    payload = {
        "iss": issuer,
        "sub": subject,
        "aud": audience,
        "iat": now_epoch_seconds,
        "exp": now_epoch_seconds + exp_offset_seconds,
    }
    return jwt.encode(
        payload,
        key=private_key_pem,
        algorithm="RS256",
        headers={"kid": kid},
    )


@pytest.mark.unit
class TestOIDCVerifyMultiIssuer:
    async def test_unknown_issuer_is_unauthorized(self):
        verifier = OIDCVerifier(
            issuers={},
            jwks_cache=JWKSCache(fetcher=lambda _uri: pytest.fail("fetcher called")),
        )

        token = "not-a-real-token"
        with pytest.raises(UnauthorizedError):
            await verifier.verify_bearer_token(token)

    async def test_unknown_issuer_valid_token_is_unauthorized_without_fetch(self):
        verifier = OIDCVerifier(
            issuers={},
            jwks_cache=JWKSCache(fetcher=lambda _uri: pytest.fail("fetcher called")),
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

        with pytest.raises(UnauthorizedError, match="Unknown issuer"):
            await verifier.verify_bearer_token(token)

    async def test_wrong_audience_is_unauthorized(self):
        now_epoch_seconds = int(time.time())
        issuer = "https://issuer.example"
        kid = "kid-1"
        jwks_uri = "https://issuer.example/jwks.json"

        private_key_pem = _generate_rsa_private_key_pem()
        public_key = serialization.load_pem_private_key(
            private_key_pem,
            password=None,
        ).public_key()
        assert isinstance(public_key, rsa.RSAPublicKey)

        jwks_payload = {"keys": [_jwk_from_public_key(public_key=public_key, kid=kid)]}

        async def fetcher(uri: str) -> dict[str, Any]:
            assert uri == jwks_uri
            return jwks_payload

        issuers = {
            issuer: _issuer_config(
                jwks_uri=jwks_uri,
                audiences=["mcp-registry"],
            )
        }
        verifier = OIDCVerifier(
            issuers=issuers,
            jwks_cache=JWKSCache(fetcher=fetcher),
            now=lambda: now_epoch_seconds,
        )

        token = _make_token(
            private_key_pem=private_key_pem,
            kid=kid,
            issuer=issuer,
            subject="user-1",
            audience="wrong-audience",
            now_epoch_seconds=now_epoch_seconds,
        )

        with pytest.raises(UnauthorizedError):
            await verifier.verify_bearer_token(token)

    async def test_multi_issuer_routing_works(self):
        now_epoch_seconds = int(time.time())

        issuer_one = "https://issuer-one.example"
        issuer_two = "https://issuer-two.example"

        kid_one = "kid-one"
        kid_two = "kid-two"

        jwks_uri_one = "https://issuer-one.example/jwks.json"
        jwks_uri_two = "https://issuer-two.example/jwks.json"

        private_one = _generate_rsa_private_key_pem()
        private_two = _generate_rsa_private_key_pem()

        public_one = serialization.load_pem_private_key(private_one, password=None).public_key()
        public_two = serialization.load_pem_private_key(private_two, password=None).public_key()
        assert isinstance(public_one, rsa.RSAPublicKey)
        assert isinstance(public_two, rsa.RSAPublicKey)

        jwks_one = {"keys": [_jwk_from_public_key(public_key=public_one, kid=kid_one)]}
        jwks_two = {"keys": [_jwk_from_public_key(public_key=public_two, kid=kid_two)]}

        async def fetcher(uri: str) -> dict[str, Any]:
            if uri == jwks_uri_one:
                return jwks_one
            if uri == jwks_uri_two:
                return jwks_two
            raise AssertionError(f"Unexpected JWKS URI: {uri}")

        issuers = {
            issuer_one: _issuer_config(
                jwks_uri=jwks_uri_one,
                audiences=["aud-one"],
            ),
            issuer_two: _issuer_config(
                jwks_uri=jwks_uri_two,
                audiences=["aud-two"],
            ),
        }
        verifier = OIDCVerifier(
            issuers=issuers,
            jwks_cache=JWKSCache(fetcher=fetcher),
            now=lambda: now_epoch_seconds,
        )

        token_one = _make_token(
            private_key_pem=private_one,
            kid=kid_one,
            issuer=issuer_one,
            subject="user-1",
            audience="aud-one",
            now_epoch_seconds=now_epoch_seconds,
        )
        token_two = _make_token(
            private_key_pem=private_two,
            kid=kid_two,
            issuer=issuer_two,
            subject="user-2",
            audience="aud-two",
            now_epoch_seconds=now_epoch_seconds,
        )

        verified_one = await verifier.verify_bearer_token(token_one)
        verified_two = await verifier.verify_bearer_token(token_two)

        assert verified_one.user_id == f"{issuer_one}|user-1"
        assert verified_two.user_id == f"{issuer_two}|user-2"
