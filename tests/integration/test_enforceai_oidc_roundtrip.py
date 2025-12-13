"""
Integration-lite test for EnforceAI OIDC verify + JWKS cache roundtrip.

No FastAPI wiring and no network access; validates OIDC verifier and JWKS cache
work together with realistic RS256-signed tokens and multi-issuer configs.
"""

from __future__ import annotations

import base64
from typing import Any

import pytest
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa

import jwt

from auth_server.enforceai.config import OIDCIssuerConfig
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
            "jwks_cache_ttl_seconds": 600,
            "clock_skew_seconds": 60,
        }
    )


@pytest.mark.integration
class TestEnforceAIOIDCRoundtrip:
    async def test_verify_roundtrip_and_jwks_cache_prevents_refetch(
        self,
        enforceai_rsa_keypair_pem: tuple[bytes, bytes],
    ) -> None:
        private_pem_one, _public_pem_one = enforceai_rsa_keypair_pem
        private_key_two = rsa.generate_private_key(
            public_exponent=65537,
            key_size=2048,
        )
        private_pem_two = private_key_two.private_bytes(
            encoding=serialization.Encoding.PEM,
            format=serialization.PrivateFormat.PKCS8,
            encryption_algorithm=serialization.NoEncryption(),
        )

        issuer_one = "https://issuer-one.example"
        issuer_two = "https://issuer-two.example"

        jwks_uri_one = "https://issuer-one.example/jwks.json"
        jwks_uri_two = "https://issuer-two.example/jwks.json"

        kid_one = "kid-one"
        kid_two = "kid-two"

        public_one = serialization.load_pem_private_key(private_pem_one, password=None).public_key()
        public_two = serialization.load_pem_private_key(private_pem_two, password=None).public_key()
        assert isinstance(public_one, rsa.RSAPublicKey)
        assert isinstance(public_two, rsa.RSAPublicKey)

        jwks_one = {"keys": [_jwk_from_public_key(public_key=public_one, kid=kid_one)]}
        jwks_two = {"keys": [_jwk_from_public_key(public_key=public_two, kid=kid_two)]}

        calls: dict[str, int] = {}

        async def fetcher(uri: str) -> dict[str, Any]:
            calls[uri] = calls.get(uri, 0) + 1
            if uri == jwks_uri_one:
                return jwks_one
            if uri == jwks_uri_two:
                return jwks_two
            raise AssertionError(f"Unexpected JWKS URI: {uri}")

        cache = JWKSCache(fetcher=fetcher)
        verifier = OIDCVerifier(
            issuers={
                issuer_one: _issuer_config(
                    jwks_uri=jwks_uri_one,
                    audiences=["aud-one"],
                ),
                issuer_two: _issuer_config(
                    jwks_uri=jwks_uri_two,
                    audiences=["aud-two"],
                ),
            },
            jwks_cache=cache,
            now=lambda: 1,
        )

        token_one = jwt.encode(
            {
                "iss": issuer_one,
                "sub": "user-1",
                "aud": "aud-one",
                "iat": 1,
                "exp": 3600,
                "scp": ["read"],
                "groups": ["group-a"],
            },
            key=private_pem_one,
            algorithm="RS256",
            headers={"kid": kid_one},
        )
        verified_one = await verifier.verify_bearer_token(token_one)
        assert verified_one.user_id == f"{issuer_one}|user-1"
        assert verified_one.scopes == ["read"]
        assert verified_one.roles == ["group-a"]

        verified_one_again = await verifier.verify_bearer_token(token_one)
        assert verified_one_again.user_id == f"{issuer_one}|user-1"
        assert calls[jwks_uri_one] == 1

        token_two = jwt.encode(
            {
                "iss": issuer_two,
                "sub": "user-2",
                "aud": "aud-two",
                "iat": 1,
                "exp": 3600,
                "scope": "write",
            },
            key=private_pem_two,
            algorithm="RS256",
            headers={"kid": kid_two},
        )
        verified_two = await verifier.verify_bearer_token(token_two)
        assert verified_two.user_id == f"{issuer_two}|user-2"
        assert verified_two.scopes == ["write"]
        assert calls[jwks_uri_two] == 1
