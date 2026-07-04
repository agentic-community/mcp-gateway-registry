"""Registry-side verifier for the /validate-minted registry-UI internal token.

The auth-server ``/validate`` endpoint mints a short-lived HS256 JWT (audience
``mcp-registry-ui``) signed with the shared ``SECRET_KEY`` whenever nginx forwards
the ``X-Registry-Api-Auth`` marker on a registry ``/api/`` request. nginx forwards
the token to the registry on the ``X-Internal-Token-Registry`` header. The registry
verifies it here and treats the verified claims as the source of truth for identity,
ignoring the forgeable inbound ``X-User`` / ``X-Scopes`` / ``X-Groups`` headers.

The token is a *thin identity assertion*: it binds who the caller is
(``sub``/``session_id``/``groups``/``auth_method``/``client_id``), NOT their resolved
entitlements. ``nginx_proxied_auth`` derives groups->scopes->permissions server-side
(mirroring the cookie path), so the token stays a constant size regardless of how many
groups a user has.

This module is the registry's own verifier (mirroring ``registry/auth/internal.py``);
it deliberately does NOT import from ``auth_server/`` -- the two are separate services
that share only the ``SECRET_KEY`` and the JWT contract.

``NGINX_DISABLE_API_AUTH_REQUEST`` is a soft deployment-mode gate, NOT a security
boundary: in disable mode the registry receives no token and falls back to the session
cookie / real bearer path -- the forgeable inbound identity headers are ignored either
way. See ``_api_auth_request_enabled``.
"""

import logging
import os

import jwt as pyjwt
from fastapi import HTTPException, status

logger = logging.getLogger(__name__)

# Must match auth_server/internal_request_token.py.
_ISSUER: str = "mcp-auth-server"
_AUDIENCE: str = "mcp-registry-ui"
_TOKEN_USE: str = "mcp-registry-ui"


def _leeway_seconds() -> int:
    """Clock-skew leeway on exp/iat checks. Mirrors the auth-server minter's read
    of the same env var so mint and verify agree on the tolerance."""
    raw = os.environ.get("INTERNAL_TOKEN_LEEWAY_SECONDS", "5")
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(f"Invalid INTERNAL_TOKEN_LEEWAY_SECONDS={raw!r}; using default 5")
        return 5


def _api_auth_request_enabled() -> bool:
    """Whether nginx fronts ``/api/`` with ``auth_request /validate`` (and therefore
    mints/forwards the registry-UI token).

    Reads the SAME ``NGINX_DISABLE_API_AUTH_REQUEST`` flag that
    ``registry/core/nginx_service.py`` reads when generating the nginx config, so the
    registry's "reject missing token" vs "fall back to cookie" decision can never drift
    from what nginx actually emitted.

    This is a soft deployment-mode gate, not a security boundary: an attacker who can
    set this env already owns the container, and in disable mode the inbound identity
    headers are still ignored (the only fallback is the session cookie).
    """
    return os.environ.get("NGINX_DISABLE_API_AUTH_REQUEST", "false").lower() not in (
        "1",
        "true",
        "yes",
        "on",
    )


def verify_registry_ui_token(token: str) -> dict:
    """Decode and validate the registry-UI internal token.

    Args:
        token: The raw JWT from the ``X-Internal-Token-Registry`` header.

    Returns:
        The verified claims: ``sub``, ``session_id``, ``groups``, ``auth_method``,
        ``client_id`` (plus standard JWT claims).

    Raises:
        HTTPException: 500 if ``SECRET_KEY`` is unset (configuration error); 401 on
            any token failure (missing/garbage/expired/wrong-audience/wrong-issuer/
            wrong-token_use/tampered).
    """
    secret_key = os.environ.get("SECRET_KEY")
    if not secret_key:
        logger.error("SECRET_KEY not set, cannot verify registry-ui token")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server configuration error",
        )

    try:
        claims = pyjwt.decode(
            token,
            secret_key,
            algorithms=["HS256"],
            issuer=_ISSUER,
            audience=_AUDIENCE,
            leeway=_leeway_seconds(),
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_iss": True,
                "verify_aud": True,
            },
        )
    except pyjwt.ExpiredSignatureError:
        logger.warning("registry-ui token expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Internal token expired",
        )
    except pyjwt.InvalidTokenError as exc:
        logger.warning(f"registry-ui token invalid: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal token",
        )

    if claims.get("token_use") != _TOKEN_USE:
        logger.warning("registry-ui token has wrong token_use")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal token",
        )

    return claims


# Must match auth_server/internal_request_token.py (MCP_PROXY_AUDIENCE/USE).
_MCP_PROXY_AUDIENCE: str = "mcp-proxy"
_MCP_PROXY_TOKEN_USE: str = "mcp-proxy"


def verify_mcp_proxy_token(token: str) -> dict:
    """Decode and validate the /mcp-proxy internal token, registry-side.

    The registry deliberately does NOT import auth_server's verifier (see the
    module docstring); this mirrors ``verify_registry_ui_token`` with the
    ``mcp-proxy`` audience/token_use so the egress vend endpoint can independently
    re-derive ``sub``/``auth_method`` from the SECRET_KEY-signed token rather than
    trusting an asserted body field.

    Returns the verified claims: ``sub``, ``auth_method``, ``scopes``, ``server``,
    ``upstream_url`` (plus standard JWT claims).

    Raises:
        HTTPException: 500 if ``SECRET_KEY`` is unset; 401 on any token failure
            (missing/garbage/expired/wrong-audience/wrong-issuer/wrong-token_use/
            tampered/missing-upstream-binding).
    """
    secret_key = os.environ.get("SECRET_KEY")
    if not secret_key:
        logger.error("SECRET_KEY not set, cannot verify mcp-proxy token")
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server configuration error",
        )

    try:
        claims = pyjwt.decode(
            token,
            secret_key,
            algorithms=["HS256"],
            issuer=_ISSUER,
            audience=_MCP_PROXY_AUDIENCE,
            leeway=_leeway_seconds(),
            options={
                "verify_signature": True,
                "verify_exp": True,
                "verify_iat": True,
                "verify_iss": True,
                "verify_aud": True,
            },
        )
    except pyjwt.ExpiredSignatureError:
        logger.warning("mcp-proxy token expired")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Internal token expired",
        )
    except pyjwt.InvalidTokenError as exc:
        logger.warning(f"mcp-proxy token invalid: {exc}")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal token",
        )

    if claims.get("token_use") != _MCP_PROXY_TOKEN_USE:
        logger.warning("mcp-proxy token has wrong token_use")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal token",
        )

    # The token MUST carry an upstream binding -- without it the vend endpoint
    # cannot run the upstream cross-check; fail closed.
    if not claims.get("upstream_url"):
        logger.warning("mcp-proxy token missing upstream binding")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid internal token",
        )

    return claims
