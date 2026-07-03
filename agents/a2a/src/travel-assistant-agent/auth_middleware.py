"""Reusable JWT-authentication middleware for A2A agent servers.

The A2A mount and the custom ``/api/*`` endpoints drive an LLM tool loop that can
call registry operations, book flights, and move money. Left unauthenticated, any
network-reachable caller could trigger those actions. This middleware enforces a
bearer-JWT check in front of every request so only callers holding a token minted
by the trusted Keycloak realm reach the agent.

Design notes (fail closed):
    - Deny by default. A request without a valid token is rejected with 401; a
      request whose token fails signature/issuer/expiry/audience validation is
      rejected. Only an explicit allowlist of unauthenticated paths (health
      probes) is exempt.
    - If authentication is not configured (no issuer/JWKS reachable), the
      middleware still denies protected paths rather than falling open. Set
      ``AGENT_AUTH_DISABLED=true`` to intentionally run without auth in a trusted
      local sandbox — this is logged loudly and must never be used in a shared or
      internet-reachable deployment.
    - Signature verification uses PyJWT's ``PyJWKClient`` against the realm JWKS
      endpoint (keys are cached and rotated by the client), so we never trust
      unverified claims.

This module is intentionally self-contained so it can be copied into any A2A agent
without pulling in registry internals.
"""

import logging
import os

import jwt
from jwt import PyJWKClient
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse
from starlette.types import ASGIApp

logger = logging.getLogger(__name__)

# Paths that must remain reachable without a token (container health probes).
# Keep this list minimal; everything else is authenticated.
_PUBLIC_PATHS: frozenset[str] = frozenset({"/ping", "/api/health"})

_SIGNING_ALGORITHMS: tuple[str, ...] = ("RS256", "RS384", "RS512", "ES256", "ES384")


class AuthConfigurationError(RuntimeError):
    """Raised when authentication is required but cannot be configured."""


def _env_flag(
    name: str,
    default: bool = False,
) -> bool:
    """Read a boolean environment flag.

    Args:
        name: Environment variable name.
        default: Value returned when the variable is unset.

    Returns:
        The parsed boolean value.
    """
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _build_issuer(
    keycloak_url: str,
    realm: str,
) -> str:
    """Build the expected token issuer URL for a Keycloak realm."""
    return f"{keycloak_url.rstrip('/')}/realms/{realm}"


class JWTAuthMiddleware(BaseHTTPMiddleware):
    """Starlette/FastAPI middleware that validates inbound bearer JWTs.

    Every request except the public health paths must carry an
    ``Authorization: Bearer <jwt>`` header whose token validates against the
    configured Keycloak realm. Validation failures and unconfigured auth both
    result in a 401/503 denial — the middleware never falls open.
    """

    def __init__(
        self,
        app: ASGIApp,
        keycloak_url: str,
        realm: str,
        audience: str | None = None,
        auth_disabled: bool = False,
    ) -> None:
        """Initialize the middleware.

        Args:
            app: The wrapped ASGI application.
            keycloak_url: Base URL of the Keycloak server.
            realm: Keycloak realm whose JWKS signs the accepted tokens.
            audience: Expected ``aud`` claim. When None, audience is not checked
                (Keycloak access tokens often carry ``account``); prefer setting
                it in production.
            auth_disabled: When True, authentication is bypassed entirely. Only
                for trusted local sandboxes; logged as a loud warning.
        """
        super().__init__(app)
        self._auth_disabled = auth_disabled
        self._audience = audience
        self._issuer = _build_issuer(keycloak_url, realm)
        self._jwks_client: PyJWKClient | None = None

        if auth_disabled:
            logger.warning(
                "AGENT AUTH IS DISABLED (AGENT_AUTH_DISABLED). All A2A and /api endpoints "
                "are UNAUTHENTICATED. Never use this in a shared or internet-reachable deployment."
            )
            return

        jwks_uri = f"{self._issuer}/protocol/openid-connect/certs"
        try:
            # PyJWKClient caches and rotates signing keys for us.
            self._jwks_client = PyJWKClient(jwks_uri)
            logger.info("JWT auth middleware configured (issuer=%s)", self._issuer)
        except Exception as exc:  # noqa: BLE001 - fail closed on any config error
            # Do not fall open: leave jwks_client None so protected paths are denied.
            logger.error("Failed to initialize JWKS client for %s: %s", jwks_uri, exc)

    def _is_public(
        self,
        path: str,
    ) -> bool:
        """Return True if the path may be reached without authentication."""
        return path in _PUBLIC_PATHS

    def _validate_token(
        self,
        token: str,
    ) -> dict:
        """Verify a bearer token against the realm JWKS.

        Args:
            token: The raw JWT string.

        Returns:
            The decoded, verified claims.

        Raises:
            AuthConfigurationError: If the JWKS client is unavailable.
            jwt.PyJWTError: If the token is invalid.
        """
        if self._jwks_client is None:
            raise AuthConfigurationError("JWKS client not initialized")

        signing_key = self._jwks_client.get_signing_key_from_jwt(token)
        options = {"require": ["exp", "iss"]}
        decode_kwargs: dict = {
            "algorithms": _SIGNING_ALGORITHMS,
            "issuer": self._issuer,
            "options": options,
        }
        if self._audience is not None:
            decode_kwargs["audience"] = self._audience
        else:
            options["verify_aud"] = False

        return jwt.decode(token, signing_key.key, **decode_kwargs)

    async def dispatch(
        self,
        request: Request,
        call_next,
    ):
        """Authenticate the request before dispatching, or deny it."""
        if self._auth_disabled or self._is_public(request.url.path):
            return await call_next(request)

        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return JSONResponse(
                {"error": "Missing or malformed Authorization header"},
                status_code=401,
            )

        token = auth_header[len("bearer ") :].strip()
        if not token:
            return JSONResponse({"error": "Empty bearer token"}, status_code=401)

        try:
            claims = self._validate_token(token)
        except AuthConfigurationError:
            # Auth required but not configured: deny (do not fall open).
            logger.error("Rejecting request: authentication is not configured")
            return JSONResponse(
                {"error": "Authentication is not available"},
                status_code=503,
            )
        except jwt.PyJWTError as exc:
            logger.warning("Rejecting request: token validation failed: %s", exc)
            return JSONResponse({"error": "Invalid token"}, status_code=401)

        # Attach verified identity for downstream handlers; never log the token.
        request.state.auth_claims = claims
        return await call_next(request)


# Bind addresses that expose the agent beyond the local host.
_EXPOSED_BIND_ADDRESSES: frozenset[str] = frozenset({"0.0.0.0", "::", "*"})  # nosec B104 - matched, not bound


def install_agent_auth(
    app: ASGIApp,
    keycloak_url: str,
    realm: str,
    audience: str | None = None,
    bind_host: str | None = None,
) -> None:
    """Install JWT authentication on an A2A agent FastAPI app.

    Reads ``AGENT_AUTH_DISABLED`` to allow an explicit, logged opt-out for trusted
    local sandboxes. When enabled (the default), unauthenticated callers are
    rejected before reaching the LLM tool loop.

    Args:
        app: The FastAPI application to protect. Must be called before the app
            starts serving (middleware cannot be added after startup).
        keycloak_url: Base URL of the Keycloak server.
        realm: Keycloak realm whose JWKS signs accepted tokens.
        audience: Optional expected ``aud`` claim.
        bind_host: The address the server will bind to. When authentication is
            disabled AND the server would listen on all interfaces, startup is
            refused — that combination exposes an unauthenticated LLM tool loop to
            the network.

    Raises:
        AuthConfigurationError: If auth is disabled while binding to a
            network-exposed address.
    """
    auth_disabled = _env_flag("AGENT_AUTH_DISABLED", default=False)
    if auth_disabled and bind_host is not None and bind_host in _EXPOSED_BIND_ADDRESSES:
        raise AuthConfigurationError(
            "Refusing to start: AGENT_AUTH_DISABLED is set while binding to "
            f"'{bind_host}'. An unauthenticated agent must not listen on all "
            "interfaces. Bind to 127.0.0.1 or enable authentication."
        )
    app.add_middleware(
        JWTAuthMiddleware,
        keycloak_url=keycloak_url,
        realm=realm,
        audience=audience,
        auth_disabled=auth_disabled,
    )
