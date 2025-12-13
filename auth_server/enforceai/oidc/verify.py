from __future__ import annotations

import json
import time
from typing import Any, Callable, Optional

import jwt
from jwt import (
    DecodeError,
    InvalidAlgorithmError,
    InvalidAudienceError,
    InvalidIssuerError,
    InvalidSignatureError,
    MissingRequiredClaimError,
)

from auth_server.enforceai.config import (
    OIDCIssuerConfig,
)
from auth_server.enforceai.errors import (
    DependencyUnavailableError,
    UnauthorizedError,
)
from auth_server.enforceai.oidc.claims import (
    extract_roles_for_audit,
    extract_scopes,
    is_audience_allowed,
    normalize_token_audiences,
)
from auth_server.enforceai.oidc.jwks import (
    JWKSCache,
)
from auth_server.enforceai.oidc.models import (
    OIDCValidatedToken,
)

NowCallable = Callable[[], int]


def _now_epoch_seconds() -> int:
    return int(time.time())


def _get_unverified_header(
    token: str,
) -> dict[str, Any]:
    try:
        return jwt.get_unverified_header(token)
    except Exception as exc:  # noqa: BLE001
        raise UnauthorizedError(
            "Failed to parse JWT header",
            public_message="Unauthorized",
        ) from exc


def _get_unverified_claims(
    token: str,
) -> dict[str, Any]:
    try:
        return jwt.decode(
            token,
            options={
                "verify_signature": False,
                "verify_aud": False,
            },
        )
    except Exception as exc:  # noqa: BLE001
        raise UnauthorizedError(
            "Failed to parse JWT claims",
            public_message="Unauthorized",
        ) from exc


def _select_jwk_for_token(
    *,
    jwks: dict[str, Any],
    kid: Optional[str],
) -> Optional[dict[str, Any]]:
    keys = jwks.get("keys")
    if not isinstance(keys, list):
        return None

    if kid:
        for key in keys:
            if isinstance(key, dict) and key.get("kid") == kid:
                return key
        return None

    jwk_candidates = [key for key in keys if isinstance(key, dict)]
    if len(jwk_candidates) == 1:
        return jwk_candidates[0]

    return None


def _public_key_from_jwk(
    jwk: dict[str, Any],
) -> Any:
    try:
        jwk_json = json.dumps(jwk)
        return jwt.algorithms.RSAAlgorithm.from_jwk(jwk_json)
    except Exception as exc:  # noqa: BLE001
        raise DependencyUnavailableError(
            "Failed to parse RSA public key from JWKS",
            public_message="OIDC key set unavailable",
        ) from exc


def _validate_iat(
    *,
    claims: dict[str, Any],
    now_epoch_seconds: int,
    clock_skew_seconds: int,
) -> None:
    iat = claims.get("iat")
    if iat is None:
        return

    if not isinstance(iat, int):
        raise UnauthorizedError(
            "Invalid iat claim type",
            public_message="Unauthorized",
        )

    if iat > now_epoch_seconds + clock_skew_seconds:
        raise UnauthorizedError(
            "iat is in the future beyond allowed clock skew",
            public_message="Unauthorized",
        )


def _validate_exp(
    *,
    claims: dict[str, Any],
    now_epoch_seconds: int,
    clock_skew_seconds: int,
) -> None:
    exp = claims.get("exp")
    if not isinstance(exp, int):
        raise UnauthorizedError(
            "Invalid exp claim type",
            public_message="Unauthorized",
        )

    if exp < now_epoch_seconds - clock_skew_seconds:
        raise UnauthorizedError(
            "Token expired",
            public_message="Unauthorized",
        )


def _build_audit_claims_subset(
    verified_claims: dict[str, Any],
) -> dict[str, Any]:
    subset_keys = {
        "iss",
        "sub",
        "aud",
        "exp",
        "iat",
        "jti",
    }
    subset: dict[str, Any] = {}
    for key in subset_keys:
        if key in verified_claims:
            subset[key] = verified_claims[key]
    return subset


class OIDCVerifier:
    """Generic OIDC JWT verifier with multi-issuer config and JWKS caching."""

    def __init__(
        self,
        *,
        issuers: dict[str, OIDCIssuerConfig],
        jwks_cache: Optional[JWKSCache] = None,
        now: Optional[NowCallable] = None,
    ) -> None:
        self._issuers = issuers
        self._jwks_cache = jwks_cache or JWKSCache()
        self._now = now or _now_epoch_seconds

    async def verify_bearer_token(
        self,
        token: str,
    ) -> OIDCValidatedToken:
        """Verify an OIDC JWT and return a normalized, verified output model.

        Raises:
            UnauthorizedError: invalid/expired/malformed token, unknown issuer, missing kid, or aud mismatch.
            DependencyUnavailableError: JWKS cannot be fetched/parsed when required to verify.
        """

        header = _get_unverified_header(token)
        kid = header.get("kid")
        if kid is not None and not isinstance(kid, str):
            raise UnauthorizedError(
                "Invalid kid header type",
                public_message="Unauthorized",
            )

        unverified = _get_unverified_claims(token)
        issuer = unverified.get("iss")
        subject = unverified.get("sub")
        if not isinstance(issuer, str) or not issuer.strip():
            raise UnauthorizedError(
                "Missing or invalid iss claim",
                public_message="Unauthorized",
            )
        if not isinstance(subject, str) or not subject.strip():
            raise UnauthorizedError(
                "Missing or invalid sub claim",
                public_message="Unauthorized",
            )

        issuer = issuer.strip()
        issuer_config = self._issuers.get(issuer)
        if issuer_config is None:
            raise UnauthorizedError(
                f"Unknown issuer: {issuer}",
                public_message="Unauthorized",
            )

        jwks = await self._jwks_cache.get_jwks(
            issuer=issuer,
            issuer_config=issuer_config,
        )
        jwk = _select_jwk_for_token(
            jwks=jwks,
            kid=kid,
        )

        if jwk is None and kid:
            jwks = await self._jwks_cache.refresh_jwks(
                issuer=issuer,
                issuer_config=issuer_config,
            )
            jwk = _select_jwk_for_token(
                jwks=jwks,
                kid=kid,
            )

        if jwk is None:
            raise UnauthorizedError(
                "No matching JWKS key for token",
                public_message="Unauthorized",
            )

        public_key = _public_key_from_jwk(jwk)

        try:
            verified_claims = jwt.decode(
                token,
                key=public_key,
                algorithms=issuer_config.algorithms,
                audience=issuer_config.audiences,
                issuer=issuer,
                options={
                    "require": [
                        "exp",
                        "iss",
                        "sub",
                    ],
                    "verify_exp": False,
                    "verify_iat": False,
                    "verify_nbf": False,
                },
            )
        except (
            DecodeError,
            InvalidAlgorithmError,
            InvalidAudienceError,
            InvalidIssuerError,
            InvalidSignatureError,
            MissingRequiredClaimError,
        ) as exc:
            raise UnauthorizedError(
                "OIDC token verification failed",
                public_message="Unauthorized",
            ) from exc
        except Exception as exc:  # noqa: BLE001
            raise UnauthorizedError(
                "Unexpected OIDC token verification failure",
                public_message="Unauthorized",
            ) from exc

        token_audiences = normalize_token_audiences(verified_claims)
        if not is_audience_allowed(
            token_audiences=token_audiences,
            allowed_audiences=issuer_config.audiences,
        ):
            raise UnauthorizedError(
                "Audience mismatch",
                public_message="Unauthorized",
            )

        now_epoch_seconds = self._now()
        _validate_exp(
            claims=verified_claims,
            now_epoch_seconds=now_epoch_seconds,
            clock_skew_seconds=issuer_config.clock_skew_seconds,
        )
        _validate_iat(
            claims=verified_claims,
            now_epoch_seconds=now_epoch_seconds,
            clock_skew_seconds=issuer_config.clock_skew_seconds,
        )

        verified_issuer = verified_claims.get("iss")
        verified_subject = verified_claims.get("sub")
        if not isinstance(verified_issuer, str) or not verified_issuer.strip():
            raise UnauthorizedError(
                "Verified token missing iss",
                public_message="Unauthorized",
            )
        if not isinstance(verified_subject, str) or not verified_subject.strip():
            raise UnauthorizedError(
                "Verified token missing sub",
                public_message="Unauthorized",
            )

        verified_issuer = verified_issuer.strip()
        verified_subject = verified_subject.strip()
        user_id = f"{verified_issuer}|{verified_subject}"

        return OIDCValidatedToken(
            issuer=verified_issuer,
            subject=verified_subject,
            user_id=user_id,
            audiences=token_audiences,
            scopes=extract_scopes(
                claims=verified_claims,
                scope_claims=issuer_config.scope_claims,
            ),
            roles=extract_roles_for_audit(
                claims=verified_claims,
                role_claims=issuer_config.role_claims,
            ),
            claims=_build_audit_claims_subset(verified_claims),
        )
