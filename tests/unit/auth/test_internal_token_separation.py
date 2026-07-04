"""Unit tests for internal/user token separation (Security Finding 1).

Internal service tokens and end-user tokens used to be cryptographically
interchangeable HS256 JWTs (same SECRET_KEY, same issuer, same audience),
so a low-privilege user token was accepted by ``validate_internal_auth`` on
every ``/api/internal/*`` route. These tests pin the three independent
separation controls that close that hole:

1. Distinct audience: internal tokens use ``mcp-internal``; user tokens use
   ``mcp-registry``.
2. Mandatory ``token_kind`` claim: internal tokens carry
   ``internal-service``; the validator rejects anything else.
3. Separate signing key: internal tokens are signed with a key derived from
   ``SECRET_KEY`` via HMAC-SHA256, so a user token signed with the raw
   ``SECRET_KEY`` fails signature verification on the internal path.

The tests hit ``_validate_authorization_header`` (the header helper behind the
public ``validate_internal_auth`` dependency) so a regression fails a focused
unit test rather than an HTTP meta-test.
"""

import os
import time
from unittest.mock import patch

import jwt as pyjwt
import pytest
from fastapi import HTTPException

from registry.auth.internal import (
    _INTERNAL_JWT_AUDIENCE,
    _INTERNAL_JWT_ISSUER,
    _INTERNAL_TOKEN_KIND,
    _derive_internal_signing_key,
    _validate_authorization_header,
    generate_internal_token,
)

_SECRET_KEY: str = "x" * 40  # >= 32 bytes so the config-level guard is satisfied
_USER_AUDIENCE: str = "mcp-registry"


def _make_token(
    secret_key: str,
    audience: str,
    token_kind: str | None,
    use_derived_key: bool,
) -> str:
    """Build a JWT for a specific separation scenario.

    Args:
        secret_key: The shared application secret.
        audience: The ``aud`` claim to stamp.
        token_kind: The ``token_kind`` claim, or None to omit it.
        use_derived_key: When True, sign with the derived internal key;
            when False, sign with the raw ``secret_key`` (the user-token key).

    Returns:
        An encoded HS256 JWT.
    """
    now = int(time.time())
    claims = {
        "iss": _INTERNAL_JWT_ISSUER,
        "aud": audience,
        "sub": "test-subject",
        "token_use": "access",
        "iat": now,
        "exp": now + 60,
    }
    if token_kind is not None:
        claims["token_kind"] = token_kind

    signing_key = _derive_internal_signing_key(secret_key) if use_derived_key else secret_key
    return pyjwt.encode(claims, signing_key, algorithm="HS256")


class TestInternalTokenSeparation:
    """Separation invariants between internal and user tokens."""

    def test_internal_token_accepted(self) -> None:
        """A token from generate_internal_token passes the gate and returns sub."""
        with patch.dict(os.environ, {"SECRET_KEY": _SECRET_KEY}):
            token = generate_internal_token(subject="registry-service", purpose="test")
            caller = _validate_authorization_header(f"Bearer {token}")
        assert caller == "registry-service"

    def test_user_token_rejected_by_audience(self) -> None:
        """A user-shape token (aud=mcp-registry, raw-key signed) is rejected."""
        with patch.dict(os.environ, {"SECRET_KEY": _SECRET_KEY}):
            token = _make_token(
                secret_key=_SECRET_KEY,
                audience=_USER_AUDIENCE,
                token_kind="user",
                use_derived_key=False,
            )
            with pytest.raises(HTTPException) as exc_info:
                _validate_authorization_header(f"Bearer {token}")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid token"

    def test_user_token_rejected_by_signature(self) -> None:
        """Right audience but signed with the raw key (not derived) is rejected.

        This isolates the signing-key control: even if an attacker could set
        aud=mcp-internal, a token signed with the raw SECRET_KEY fails the
        signature check against the derived internal key.
        """
        with patch.dict(os.environ, {"SECRET_KEY": _SECRET_KEY}):
            token = _make_token(
                secret_key=_SECRET_KEY,
                audience=_INTERNAL_JWT_AUDIENCE,
                token_kind=_INTERNAL_TOKEN_KIND,
                use_derived_key=False,
            )
            with pytest.raises(HTTPException) as exc_info:
                _validate_authorization_header(f"Bearer {token}")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid token"

    def test_wrong_token_kind_rejected(self) -> None:
        """Correct audience and derived key but token_kind=user is rejected.

        This isolates the token_kind control (the derived-key and audience
        controls are both satisfied here).
        """
        with patch.dict(os.environ, {"SECRET_KEY": _SECRET_KEY}):
            token = _make_token(
                secret_key=_SECRET_KEY,
                audience=_INTERNAL_JWT_AUDIENCE,
                token_kind="user",
                use_derived_key=True,
            )
            with pytest.raises(HTTPException) as exc_info:
                _validate_authorization_header(f"Bearer {token}")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid token"

    def test_missing_token_kind_rejected(self) -> None:
        """Correct audience and derived key but no token_kind claim is rejected."""
        with patch.dict(os.environ, {"SECRET_KEY": _SECRET_KEY}):
            token = _make_token(
                secret_key=_SECRET_KEY,
                audience=_INTERNAL_JWT_AUDIENCE,
                token_kind=None,
                use_derived_key=True,
            )
            with pytest.raises(HTTPException) as exc_info:
                _validate_authorization_header(f"Bearer {token}")
        assert exc_info.value.status_code == 401
        assert exc_info.value.detail == "Invalid token"

    def test_expired_internal_token_rejected(self) -> None:
        """An otherwise-valid internal token whose exp is in the past is rejected."""
        with patch.dict(os.environ, {"SECRET_KEY": _SECRET_KEY}):
            now = int(time.time())
            claims = {
                "iss": _INTERNAL_JWT_ISSUER,
                "aud": _INTERNAL_JWT_AUDIENCE,
                "sub": "registry-service",
                "token_kind": _INTERNAL_TOKEN_KIND,
                "token_use": "access",
                "iat": now - 120,
                "exp": now - 60,
            }
            token = pyjwt.encode(
                claims, _derive_internal_signing_key(_SECRET_KEY), algorithm="HS256"
            )
            with pytest.raises(HTTPException) as exc_info:
                _validate_authorization_header(f"Bearer {token}")
        assert exc_info.value.status_code == 401


class TestDerivedSigningKey:
    """Properties of the internal signing-key derivation."""

    def test_derived_key_differs_from_secret(self) -> None:
        """The derived key is 32 bytes and not equal to the raw secret bytes."""
        derived = _derive_internal_signing_key(_SECRET_KEY)
        assert isinstance(derived, bytes)
        assert len(derived) == 32
        assert derived != _SECRET_KEY.encode()

    def test_derivation_is_deterministic(self) -> None:
        """Both services must derive the same key from the same secret."""
        assert _derive_internal_signing_key(_SECRET_KEY) == _derive_internal_signing_key(
            _SECRET_KEY
        )

    def test_different_secret_yields_different_key(self) -> None:
        """A different SECRET_KEY yields a different derived key."""
        assert _derive_internal_signing_key("a" * 40) != _derive_internal_signing_key("b" * 40)


class TestAudienceInvariant:
    """The two audiences must never collapse back into one."""

    def test_internal_audience_is_distinct_from_user_audience(self) -> None:
        """Guard against a future re-merge of the internal and user audiences."""
        assert _INTERNAL_JWT_AUDIENCE == "mcp-internal"
        assert _INTERNAL_JWT_AUDIENCE != _USER_AUDIENCE
