"""Mint and verify short-lived internal JWTs for nginx->backend hops.

The auth-server ``/validate`` endpoint is the only component that holds both the
validated caller identity/scopes and (once nginx forwards ``$backend_url``) the
resolved upstream URL. It mints a short-lived HS256 JWT over those fields, signed
with the shared ``SECRET_KEY`` (the same key/pattern as
``registry/auth/internal.py``). nginx forwards the token to the backend route via
the existing ``auth_request_set`` plumbing. The backend route verifies the token
and treats the claims as the source of truth for identity/scopes/destination,
ignoring inbound ``X-User`` / ``X-Scopes`` / ``X-Upstream-Url`` headers entirely.
"""

import logging
import os
import time

import jwt as pyjwt
from fastapi import HTTPException, Request, status

logger = logging.getLogger(__name__)

# Reuse the existing internal issuer (matches registry/auth/internal.py).
_ISSUER: str = "mcp-auth-server"

# Audience for the /mcp-proxy hop. Distinct from the "mcp-registry" audience used
# by registry/auth/internal.py service-to-service tokens, so one cannot be
# replayed for the other (PyJWT verify_aud rejects audience mismatches).
MCP_PROXY_AUDIENCE: str = "mcp-proxy"
MCP_PROXY_TOKEN_USE: str = "mcp-proxy"

# Audience for the registry /api/ hop. Distinct from both "mcp-proxy" (above) and
# the "mcp-registry" service-to-service audience, so a registry-UI token cannot be
# replayed as an mcp-proxy token, a service token, or vice-versa.
MCP_REGISTRY_UI_AUDIENCE: str = "mcp-registry-ui"
MCP_REGISTRY_UI_TOKEN_USE: str = "mcp-registry-ui"

# Audience for the generic reverse-proxy hop (/proxy/{entity_type}/{path}), used
# for non-MCP proxied entities (skills, agents, custom). Distinct from every
# audience above so a generic-proxy token can't be replayed at /mcp-proxy (which
# has no tool-list filtering) or the registry API, or vice-versa.
GENERIC_PROXY_AUDIENCE: str = "generic-proxy"
GENERIC_PROXY_TOKEN_USE: str = "generic-proxy"

# TTL is clamped to this floor so a misconfigured TTL of 0/negative cannot
# combine with the leeway into a confusing always-valid window.
_MIN_TTL_SECONDS: int = 5


def _ttl_seconds() -> int:
    raw = os.environ.get("INTERNAL_TOKEN_TTL_SECONDS", "30")
    try:
        candidate = int(raw)
    except ValueError:
        logger.warning(f"Invalid INTERNAL_TOKEN_TTL_SECONDS={raw!r}; using default 30")
        return 30
    if candidate < _MIN_TTL_SECONDS:
        logger.warning(
            f"INTERNAL_TOKEN_TTL_SECONDS={candidate} below floor; clamping to {_MIN_TTL_SECONDS}"
        )
        return _MIN_TTL_SECONDS
    return candidate


def _generic_ttl_seconds() -> int:
    """TTL for the generic-proxy token (GENERIC_PROXY_TOKEN_TTL_SECONDS).

    Falls back to the shared INTERNAL_TOKEN_TTL_SECONDS when unset, so a single
    knob covers both hops unless an operator tunes the generic hop separately.
    Clamped to the same floor as the mcp-proxy TTL.
    """
    raw = os.environ.get("GENERIC_PROXY_TOKEN_TTL_SECONDS")
    if raw is None:
        return _ttl_seconds()
    try:
        candidate = int(raw)
    except ValueError:
        logger.warning(f"Invalid GENERIC_PROXY_TOKEN_TTL_SECONDS={raw!r}; using default 30")
        return 30
    if candidate < _MIN_TTL_SECONDS:
        logger.warning(
            f"GENERIC_PROXY_TOKEN_TTL_SECONDS={candidate} below floor; "
            f"clamping to {_MIN_TTL_SECONDS}"
        )
        return _MIN_TTL_SECONDS
    return candidate


def _leeway_seconds() -> int:
    raw = os.environ.get("INTERNAL_TOKEN_LEEWAY_SECONDS", "5")
    try:
        return max(0, int(raw))
    except ValueError:
        logger.warning(f"Invalid INTERNAL_TOKEN_LEEWAY_SECONDS={raw!r}; using default 5")
        return 5


def _get_secret_key() -> str:
    secret_key = os.environ.get("SECRET_KEY")
    if not secret_key:
        raise ValueError("SECRET_KEY environment variable not set")
    return secret_key


def _mint_internal_token(
    audience: str,
    subject: str,
    scopes: list[str],
    extra_claims: dict | None = None,
    ttl_seconds: int | None = None,
) -> str:
    """Mint a short-lived HS256 internal JWT signed with SECRET_KEY.

    Args:
        audience: The intended verifier (e.g. "mcp-proxy").
        subject: The validated caller identity (becomes ``sub``).
        scopes: The validated entitlements (becomes ``scopes``, a JSON array).
        extra_claims: Audience-specific claims (e.g. ``server``/``upstream_url``).
        ttl_seconds: Explicit TTL; defaults to ``_ttl_seconds()`` (the shared
            INTERNAL_TOKEN_TTL_SECONDS). The generic hop passes its own knob.

    Returns:
        Encoded JWT string.

    Raises:
        ValueError: If ``SECRET_KEY`` is unset or ``subject`` is empty.
    """
    if not subject:
        # An empty subject would mint an anonymous-but-valid token. Refuse, so
        # /validate returns 200 without a token, nginx forwards none, and the
        # backend route rejects (fail-closed) rather than trusting "".
        raise ValueError("Cannot mint internal token with empty subject")
    secret_key = _get_secret_key()
    now = int(time.time())
    base_claims: dict = {
        "iss": _ISSUER,
        "aud": audience,
        "sub": subject,
        "scopes": list(scopes or []),
        "iat": now,
        "exp": now + (ttl_seconds if ttl_seconds is not None else _ttl_seconds()),
    }
    # Merge extra first, THEN base — so a caller's extra_claims can never override
    # a reserved base claim (aud/exp/iss/sub/...). Callers pass server-derived
    # constants today, but this makes the invariant structural, not a convention.
    claims: dict = {**(extra_claims or {}), **base_claims}
    return pyjwt.encode(claims, secret_key, algorithm="HS256")


def _decode_internal_token(
    token: str,
    audience: str,
) -> dict:
    """Decode and validate an internal JWT. Raises ``pyjwt`` errors on failure."""
    secret_key = _get_secret_key()
    return pyjwt.decode(
        token,
        secret_key,
        algorithms=["HS256"],
        issuer=_ISSUER,
        audience=audience,
        leeway=_leeway_seconds(),
        options={
            "verify_signature": True,
            "verify_exp": True,
            "verify_iat": True,
            "verify_iss": True,
            "verify_aud": True,
        },
    )


# --------------------------------------------------------------------------- #
# mcp-proxy wrappers (pin audience + required claims)
# --------------------------------------------------------------------------- #


def mint_mcp_proxy_token(
    subject: str,
    scopes: list[str],
    server_name: str,
    upstream_url: str,
    auth_method: str = "",
    egress_user: str = "",
) -> str:
    """Mint the per-request /mcp-proxy token in /validate's 200 path.

    ``upstream_url`` is the resolved upstream **before** mcp_proxy's sub-path
    append; mcp_proxy applies the append itself so the bound claim and /validate's
    view agree exactly. ``server`` is the first path segment, used as a
    path-traversal guard by the verifier.

    ``auth_method`` is the CANONICAL egress principal method (see
    ``EgressAuthService.canonical_auth_method``): the per-user egress vault keys
    on it, so it MUST match what the consent/list paths resolve for the same
    user (cookie users are ``oauth2``, never ``session_cookie``). It also lets
    the registry vend endpoint reject non-per-user callers.

    ``egress_user`` is the CANONICAL per-user vault id (the OIDC ``sub``; see
    ``_canonical_egress_user``). The egress vend keys the vault on this, so it
    MUST match what the consent-write path resolved for the same human — both
    derive it identically. Empty for non-per-user callers.
    """
    return _mint_internal_token(
        audience=MCP_PROXY_AUDIENCE,
        subject=subject,
        scopes=scopes,
        extra_claims={
            "server": server_name.split("/", 1)[0] if server_name else "",
            "upstream_url": upstream_url,
            "auth_method": auth_method,
            "egress_user": egress_user or "",
            "token_use": MCP_PROXY_TOKEN_USE,
        },
    )


# --------------------------------------------------------------------------- #
# generic-proxy wrappers (pin audience; bind entity_type + FULL registered path)
# --------------------------------------------------------------------------- #


def mint_generic_proxy_token(
    subject: str,
    scopes: list[str],
    entity_type: str,
    registered_path: str,
    upstream_url: str,
    streaming: bool = False,
) -> str:
    """Mint the per-request /proxy/{entity_type}/{path} token in /validate's 200 path.

    Binds BOTH the entity_type AND the FULL registered path as two distinct
    claims (unlike mcp-proxy, which binds only the first path segment): registered
    paths are multi-segment (e.g. ``skills/proxy-demo``), so binding only the
    first segment would let a token for one skill be replayed against a sibling.
    The verifier checks entity_type exactly and requires the route's entity path
    to equal the bound path or be a sub-path UNDER it (no sibling escape).

    ``upstream_url`` is the resolved backend the generic hop forwards to,
    cryptographically pinned so the handler ignores forgeable headers.

    ``streaming`` binds the entity's proxy_streaming opt-in into the token so the
    hop streams (chunk-forward, no response-size cap) only on the signed claim,
    never on a forgeable inbound header.
    """
    return _mint_internal_token(
        audience=GENERIC_PROXY_AUDIENCE,
        subject=subject,
        scopes=scopes,
        extra_claims={
            "entity_type": entity_type,
            "server": registered_path,  # FULL registered path, not first-segment
            "upstream_url": upstream_url,
            "streaming": bool(streaming),
            "token_use": GENERIC_PROXY_TOKEN_USE,
        },
        ttl_seconds=_generic_ttl_seconds(),
    )


# --------------------------------------------------------------------------- #
# registry-UI wrapper (pin audience; thin identity assertion)
# --------------------------------------------------------------------------- #


def mint_registry_ui_token(
    subject: str,
    session_id: str,
    groups: list[str],
    auth_method: str,
    client_id: str,
    egress_user: str = "",
) -> str:
    """Mint the per-request registry /api/ token in /validate's 200 path.

    A thin identity assertion: it binds *who* the caller is, NOT their resolved
    entitlements. The registry derives groups->scopes->permissions server-side
    (mirroring its cookie path), so no scopes are encoded and the token stays a
    constant size regardless of group count.

    ``session_id`` is the opaque server-side session identifier for browser/
    session-backed callers (the registry resolves live groups via the session
    store); it is empty for bearer/IdP-JWT and static-token callers, which have
    no session row and instead rely on the ``groups`` claim. ``groups`` is the
    fallback for those non-session callers (small, machine-identity group sets).

    ``egress_user`` is the CANONICAL per-user egress vault id (OIDC ``sub``). The
    registry's consent-write path keys the vault on this; it must match the value
    the vend path reads from the mcp-proxy token so one human maps to one bucket.

    ``_mint_internal_token`` refuses an empty subject (fail-closed): if minting
    raises, /validate attaches no token and the registry rejects the request
    rather than trusting unsigned headers.
    """
    return _mint_internal_token(
        audience=MCP_REGISTRY_UI_AUDIENCE,
        subject=subject,
        scopes=[],
        extra_claims={
            "session_id": session_id or "",
            "groups": list(groups or []),
            "auth_method": auth_method or "",
            "client_id": client_id or "",
            "egress_user": egress_user or "",
            "token_use": MCP_REGISTRY_UI_TOKEN_USE,
        },
    )


async def verify_mcp_proxy_token(request: Request) -> None:
    """FastAPI dependency on mcp_proxy.

    Always fail-closed: a request to /mcp-proxy must carry a valid
    /validate-minted ``X-Internal-Token``. On success the verified claims are
    stashed on ``request.state.mcp_proxy_claims`` (the handler reads
    identity/scopes/upstream from there and ignores the forgeable inbound
    ``X-User`` / ``X-Scopes`` / ``X-Upstream-Url`` headers). Any failure --
    missing, tampered, expired, wrong-audience, wrong token_use, missing upstream
    binding, or server-claim/path mismatch -- raises 401 before any outbound call.
    """
    try:
        # Presence check; _decode_internal_token re-reads SECRET_KEY itself.
        _get_secret_key()
    except ValueError:
        logger.error("SECRET_KEY not set, cannot verify mcp-proxy token")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server configuration error",
        )

    token = request.headers.get("X-Internal-Token")
    if not token:
        logger.warning("mcp_proxy: missing X-Internal-Token (rejecting)")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Missing internal proxy token")

    try:
        claims = _decode_internal_token(token, audience=MCP_PROXY_AUDIENCE)
    except pyjwt.ExpiredSignatureError:
        logger.warning("mcp_proxy: expired internal token")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Internal proxy token expired")
    except pyjwt.InvalidTokenError as exc:
        logger.warning(f"mcp_proxy: invalid internal token: {exc}")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid internal proxy token")

    if claims.get("token_use") != MCP_PROXY_TOKEN_USE:
        logger.warning("mcp_proxy: wrong token_use in internal token")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid internal proxy token")
    if not claims.get("upstream_url"):
        logger.warning("mcp_proxy: internal token missing upstream binding")
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="Internal proxy token missing upstream"
        )

    # Path-traversal guard: the bound server must match the route's first segment.
    path_server = (request.path_params.get("server_name") or "").split("/", 1)[0]
    if claims.get("server") != path_server:
        logger.warning(
            f"mcp_proxy: server claim/path mismatch (claim={claims.get('server')!r} path={path_server!r})"
        )
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Server claim/path mismatch")

    request.state.mcp_proxy_claims = claims


def _norm_path(value: str) -> str:
    """Normalize a route/claim path for comparison: strip leading/trailing '/'."""
    return (value or "").strip("/")


async def verify_generic_proxy_token(request: Request) -> None:
    """FastAPI dependency on the generic_proxy handler (/proxy/{entity_type}/{path}).

    Fail-closed, mirroring verify_mcp_proxy_token, EXCEPT the traversal guard
    binds the entity_type AND the FULL registered path (both claims set by
    mint_generic_proxy_token), read from the two route params ``entity_type`` and
    ``entity_path``. Rejects (401) unless:
      - a valid, unexpired, GENERIC_PROXY_AUDIENCE / GENERIC_PROXY_TOKEN_USE token;
      - an upstream binding is present;
      - claims["entity_type"] == route entity_type (blocks cross-type replay,
        e.g. a skill token used on an /a2a_agent/ route); AND
      - the route entity_path equals the bound registered path OR is a sub-path
        strictly UNDER it (``<registered>/...``) — a sub-resource of the bound
        entity is allowed, a sibling/escape is not; ``..`` segments are rejected.
    On success the claims are stashed on ``request.state.generic_proxy_claims``.

    SECURITY ASSUMPTIONS (enforced by nginx, not this function):
    - nginx MUST strip any client-supplied ``X-Internal-Token*`` header and set
      ``X-Internal-Token-Generic`` from the /validate-minted value only. The
      entire fail-closed model collapses if a client can inject this header.
    - the downstream generic handler MUST normalize the post-bound path remainder
      before appending it to the pinned ``upstream_url`` (this guard rejects ``..``
      segments, but path normalization on the upstream append is the handler's job).
    """
    try:
        _get_secret_key()
    except ValueError:
        logger.error("SECRET_KEY not set, cannot verify generic-proxy token")
        raise HTTPException(
            status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server configuration error",
        )

    token = request.headers.get("X-Internal-Token-Generic")
    if not token:
        logger.warning("generic_proxy: missing X-Internal-Token-Generic (rejecting)")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Missing internal proxy token")

    try:
        claims = _decode_internal_token(token, audience=GENERIC_PROXY_AUDIENCE)
    except pyjwt.ExpiredSignatureError:
        logger.warning("generic_proxy: expired internal token")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Internal proxy token expired")
    except pyjwt.InvalidTokenError as exc:
        logger.warning(f"generic_proxy: invalid internal token: {exc}")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid internal proxy token")

    if claims.get("token_use") != GENERIC_PROXY_TOKEN_USE:
        logger.warning("generic_proxy: wrong token_use in internal token")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Invalid internal proxy token")
    if not claims.get("upstream_url"):
        logger.warning("generic_proxy: internal token missing upstream binding")
        raise HTTPException(
            status.HTTP_401_UNAUTHORIZED, detail="Internal proxy token missing upstream"
        )

    # Cross-type replay guard: the bound entity_type must match the route.
    route_entity_type = request.path_params.get("entity_type") or ""
    if claims.get("entity_type") != route_entity_type:
        logger.warning(
            "generic_proxy: entity_type claim/path mismatch "
            f"(claim={claims.get('entity_type')!r} path={route_entity_type!r})"
        )
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Entity-type claim/path mismatch")

    # Path-traversal guard: route path must be the bound registered path, or a
    # sub-path strictly under it. Compared on normalized, segment boundaries so
    # "skills/proxy-demo" does NOT match "skills/proxy-demo-evil".
    bound = _norm_path(claims.get("server", ""))
    route_path = _norm_path(request.path_params.get("entity_path") or "")
    # Reject dot-dot segments outright: the prefix check alone would let
    # "<bound>/../../x" pass (it literally starts with "<bound>/"). This guard is
    # sibling-escape-safe but NOT dot-dot-safe without this, and the downstream
    # handler must ALSO normalize before appending to the pinned upstream.
    if ".." in route_path.split("/"):
        logger.warning(f"generic_proxy: dot-dot segment in entity path {route_path!r}")
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Illegal path segment")
    if not bound or not (route_path == bound or route_path.startswith(bound + "/")):
        logger.warning(
            f"generic_proxy: entity path/claim mismatch (claim={bound!r} path={route_path!r})"
        )
        raise HTTPException(status.HTTP_401_UNAUTHORIZED, detail="Entity path claim/path mismatch")

    request.state.generic_proxy_claims = claims
