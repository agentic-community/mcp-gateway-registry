"""Microsoft Entra ID (Azure AD) authentication provider implementation."""

import asyncio
import hashlib
import logging
import os
import time
from collections import OrderedDict
from typing import Any
from urllib.parse import urlencode, urlsplit, urlunsplit

import httpx
import jwt
import requests

from registry.exceptions import UrlValidationError
from registry.utils.url_guard import (
    ENTRA_GRAPH_PROFILE,
    guarded_async_client,
    guarded_client,
    validate_url,
)

try:
    from egress_obo import OboConfigError, OboConsentRequired, OboExchangeError, obo_exchange
except ImportError:
    from auth_server.egress_obo import (
        OboConfigError,
        OboConsentRequired,
        OboExchangeError,
        obo_exchange,
    )

from .base import AuthProvider, normalize_groups_claim

# Constants for self-signed token validation
JWT_ISSUER = os.environ.get("JWT_ISSUER", "mcp-auth-server")
JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "mcp-registry")
# SECRET_KEY is enforced at process startup by auth_server/server.py and
# registry/core/config.py; we read it at import time but do not provide a
# fallback. Self-signed token validation (which consumes this constant)
# raises if it is missing rather than silently using a known-bad value.
SECRET_KEY = os.environ.get("SECRET_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Public and sovereign login bases map to their own Microsoft Graph resources.
# https://learn.microsoft.com/en-us/graph/deployments
DEFAULT_ENTRA_LOGIN_BASE_URL = "https://login.microsoftonline.com"
DEFAULT_ENTRA_GRAPH_BASE_URL = "https://graph.microsoft.com"
_LOGIN_TO_GRAPH_HOST: dict[str, str] = {
    "login.microsoftonline.com": "https://graph.microsoft.com",
    "login.microsoftonline.us": "https://graph.microsoft.us",
    "login.chinacloudapi.cn": "https://microsoftgraph.chinacloudapi.cn",
    "login.partner.microsoftonline.cn": "https://microsoftgraph.chinacloudapi.cn",
}

# Only Microsoft's native Graph OAuth resources may be token audiences. The
# fetch endpoint may instead be an operator's reverse proxy.
_NATIVE_GRAPH_RESOURCES = frozenset(
    {*_LOGIN_TO_GRAPH_HOST.values(), "https://dod-graph.microsoft.us"}
)

# Memberships are cached per auth-server process; bounded TTL limits stale grants.
GRAPH_GROUP_CACHE_MAX_ENTRIES = 512
GRAPH_GROUP_LOOKUP_TIMEOUT_SECONDS = 8.0
GRAPH_GROUP_PAGE_SIZE = 999  # Microsoft Graph transitiveMemberOf maximum $top.
GRAPH_GROUP_NEGATIVE_TTL_SECONDS = 30
GRAPH_GROUP_NEGATIVE_MAX_ENTRIES = 512
_CACHEABLE_GROUP_FAILURES = frozenset({"consent", "configuration", "authorization"})


def _group_recovery_enabled() -> bool:
    """Read the operator switch; reject malformed values rather than guessing."""
    value = os.environ.get("ENTRA_GROUP_RECOVERY_ENABLED", "true").lower()
    if value not in {"true", "false"}:
        raise ValueError("ENTRA_GROUP_RECOVERY_ENABLED must be true or false")
    return value == "true"


def _group_cache_ttl_seconds() -> int:
    """Bound the membership revocation window configured for this process."""
    raw = os.environ.get("ENTRA_GROUP_CACHE_TTL_SECONDS", "300")
    try:
        ttl = int(raw)
    except ValueError as exc:
        raise ValueError("ENTRA_GROUP_CACHE_TTL_SECONDS must be 30..3600") from exc
    if not 30 <= ttl <= 3600:
        raise ValueError("ENTRA_GROUP_CACHE_TTL_SECONDS must be 30..3600")
    return ttl


# nginx's 16 KiB auth_request response contains both X-Groups and a registry
# token that base64-encodes the same list, plus other auth headers. Keep the
# group portion under 4 KiB so the combined headers have headroom.
GRAPH_GROUP_HEADER_MAX_BYTES = 4096


class GraphGroupLookupError(Exception):
    """Microsoft Graph could not establish the user's complete group membership."""

    def __init__(self, message: str, *, reason: str = "unknown") -> None:
        super().__init__(message)
        # A finite, non-sensitive label for structured request logging.
        self.reason = (
            reason
            if reason
            in {
                "configuration",
                "consent",
                "authorization",
                "timeout",
                "pagination",
                "transport",
                "malformed",
                "limit",
                "identity",
                "token_exchange",
                "unknown",
            }
            else "unknown"
        )


def _entra_login_base() -> str:
    """Use a consistent login origin for browser and bearer flows."""
    return (os.environ.get("ENTRA_LOGIN_BASE_URL") or DEFAULT_ENTRA_LOGIN_BASE_URL).rstrip("/")


def _infer_graph_base_url(login_base_url: str) -> str | None:
    """Return the Graph resource for a recognized Entra login cloud.

    A validated but unknown login origin has no implicit public-Graph resource.
    """
    try:
        parsed = urlsplit(login_base_url)
        port = parsed.port
    except ValueError as exc:
        raise GraphGroupLookupError(
            "Invalid Entra login cloud configuration", reason="configuration"
        ) from exc
    if (
        parsed.scheme != "https"
        or not parsed.hostname
        or parsed.username
        or parsed.password
        or parsed.query
        or parsed.fragment
        or parsed.path not in ("", "/")
        or port not in (None, 443)
    ):
        raise GraphGroupLookupError(
            "Invalid Entra login cloud configuration", reason="configuration"
        )
    return _LOGIN_TO_GRAPH_HOST.get(parsed.hostname or "")


def _validated_graph_resource_url(login_resource: str | None) -> str | None:
    """Select an official Graph OAuth resource, independently of its fetch URL."""
    configured = os.environ.get("ENTRA_GRAPH_RESOURCE_URL")
    if configured is None or configured == "":
        return login_resource
    try:
        parsed = urlsplit(configured)
        if (
            configured != configured.strip()
            or parsed.scheme != "https"
            or not parsed.hostname
            or parsed.username
            or parsed.password
            or parsed.path not in ("", "/")
            or parsed.query
            or parsed.fragment
            or parsed.port not in (None, 443)
        ):
            raise GraphGroupLookupError(
                "Invalid Microsoft Graph resource configuration", reason="configuration"
            )
    except ValueError as exc:
        raise GraphGroupLookupError(
            "Invalid Microsoft Graph resource configuration", reason="configuration"
        ) from exc
    resource = f"https://{parsed.hostname}"
    if configured not in {resource, f"{resource}/", f"{resource}:443", f"{resource}:443/"}:
        raise GraphGroupLookupError(
            "Invalid Microsoft Graph resource configuration", reason="configuration"
        )
    if resource not in _NATIVE_GRAPH_RESOURCES:
        raise GraphGroupLookupError(
            "Microsoft Graph resource must be a native Graph origin", reason="configuration"
        )
    if login_resource is not None:
        permitted = {login_resource}
        if login_resource == "https://graph.microsoft.us":
            permitted.add("https://dod-graph.microsoft.us")
        if resource not in permitted:
            raise GraphGroupLookupError(
                "Graph resource does not match Entra login cloud", reason="configuration"
            )
    return resource


def _validated_graph_override(login_resource: str | None) -> str | None:
    """Validate an explicit Graph fetch origin and reject cross-cloud Graph hosts."""
    configured = os.environ.get("ENTRA_GRAPH_BASE_URL")
    if not configured:
        return None
    override = configured.rstrip("/")
    try:
        origin = urlsplit(override)
        if (
            origin.scheme != "https"
            or not origin.hostname
            or origin.username
            or origin.password
            or origin.path
            or origin.query
            or origin.fragment
            or origin.port not in (None, 443)
        ):
            raise GraphGroupLookupError(
                "Invalid Microsoft Graph endpoint configuration", reason="configuration"
            )
        validate_url(override, profile=ENTRA_GRAPH_PROFILE, resolve=False)
    except (UrlValidationError, ValueError) as exc:
        raise GraphGroupLookupError(
            "Invalid Microsoft Graph endpoint configuration", reason="configuration"
        ) from exc
    known_graph_hosts = {urlsplit(base).hostname for base in _NATIVE_GRAPH_RESOURCES}
    if origin.hostname in known_graph_hosts and login_resource != f"https://{origin.hostname}":
        raise GraphGroupLookupError(
            "Graph endpoint does not match Entra login cloud", reason="configuration"
        )
    return override


def _graph_resource_url() -> str:
    """Return the trusted native Graph OAuth audience, never the fetch proxy."""
    resource = _validated_graph_resource_url(_infer_graph_base_url(_entra_login_base()))
    _validated_graph_override(resource)
    if resource is None:
        raise GraphGroupLookupError(
            "Unknown Entra cloud: cannot determine Graph token resource", reason="configuration"
        )
    return resource


def _post_entra_token(
    url: str,
    data: dict[str, str],
) -> httpx.Response:
    """Send an Entra token request through the HTTPS-only, pinned OAuth client."""
    from registry.utils.url_guard import CREDENTIALED_OAUTH_PROFILE

    validate_url(url, profile=CREDENTIALED_OAUTH_PROFILE, resolve=False)
    with guarded_client(
        profile=CREDENTIALED_OAUTH_PROFILE, timeout=10.0, follow_redirects=False
    ) as client:
        return client.post(url, data=data)


def _delegated_scopes(claims: dict[str, Any]) -> list[str]:
    """Read signed Entra user scopes, retaining the legacy scope claim."""
    raw = claims.get("scp", claims.get("scope", ""))
    if not isinstance(raw, str):
        raise ValueError("Entra scp claim must be a string")
    return raw.split()


class EntraIdProvider(AuthProvider):
    """Microsoft Entra ID (Azure AD) authentication provider.

    This provider implements OAuth2/OIDC authentication using Microsoft Entra ID
    (formerly Azure Active Directory). It supports:
    - User authentication via OAuth2 authorization code flow
    - Machine-to-machine authentication via client credentials flow
    - JWT token validation using Azure AD JWKS
    - Group-based authorization with Azure AD security groups
    """

    # OIDC scopes Entra accepts bare on /authorize in BOTH v1 and v2. These are
    # never prefixed with the api://<app-id> resource URI: Entra rejects
    # `api://<app>/openid` with AADSTS650053. Only custom resource scopes (e.g.
    # `mcp.read`, `user_impersonation`) take the v1 prefix.
    STANDARD_OIDC_SCOPES: frozenset[str] = frozenset(
        {"openid", "profile", "email", "offline_access", "address", "phone"}
    )

    def __init__(
        self,
        tenant_id: str,
        client_id: str,
        client_secret: str,
        scope_format: str | None = None,
        application_id_uri: str | None = None,
    ):
        """Initialize Entra ID provider.

        Args:
            tenant_id: Azure AD tenant ID (GUID)
            client_id: App registration client ID (GUID)
            client_secret: App registration client secret
            scope_format: Entra scope advertisement form, ``"v1"`` or ``"v2"``.
                v1 requires custom resource scopes to be requested as
                ``api://<app-id-or-uri>/<scope>`` on ``/authorize`` (Entra
                rejects the bare form with AADSTS650053); v2 accepts the bare
                fragment. Defaults to the ``ENTRA_SCOPE_FORMAT`` env var, then
                ``"v2"`` (backward-compatible with existing deployments).
            application_id_uri: The Application ID URI registered on the Entra
                app (e.g. ``api://<app-id>`` or a custom ``api://<uri>``). Used
                verbatim as the v1 scope prefix and accepted as a token
                audience. Defaults to the ``ENTRA_APPLICATION_ID_URI`` env var.
        """
        self.tenant_id = tenant_id
        self.client_id = client_id
        self.client_secret = client_secret

        # Scope-advertisement format and Application ID URI. Read from explicit
        # constructor args when provided (the factory passes registry config),
        # otherwise from the environment so direct construction still works.
        # Normalized to lower-case; anything other than "v1" means v2 (bare).
        raw_scope_format = scope_format or os.environ.get("ENTRA_SCOPE_FORMAT", "v2")
        self.scope_format = (raw_scope_format or "v2").strip().lower()
        raw_app_id_uri = (
            application_id_uri
            if application_id_uri is not None
            else os.environ.get("ENTRA_APPLICATION_ID_URI")
        )
        self.application_id_uri = raw_app_id_uri.rstrip("/") if raw_app_id_uri else None

        # JWKS cache
        self._jwks_cache: dict[str, Any] | None = None
        self._jwks_cache_time: float = 0
        self._jwks_cache_ttl: int = 3600  # 1 hour

        # The fetch endpoint can be a private TLS reverse proxy; the OAuth
        # resource is always a separate, explicitly trusted native Graph origin.
        login_base_url = _entra_login_base()
        login_resource = _infer_graph_base_url(login_base_url)
        graph_resource = _validated_graph_resource_url(login_resource)
        self.graph_base_url = _validated_graph_override(graph_resource)
        if self.graph_base_url is None:
            if login_resource is None:
                raise GraphGroupLookupError(
                    "Unknown Entra cloud: configure Graph endpoint", reason="configuration"
                )
            self.graph_base_url = graph_resource

        # Entra ID endpoints
        base_url = f"{login_base_url}/{tenant_id}"
        self.auth_url = f"{base_url}/oauth2/v2.0/authorize"
        self.token_url = f"{base_url}/oauth2/v2.0/token"
        self.userinfo_url = f"{self.graph_base_url}/oidc/userinfo"
        self.jwks_url = f"{base_url}/discovery/v2.0/keys"
        self.logout_url = f"{base_url}/oauth2/v2.0/logout"

        # Entra ID supports two issuer formats:
        # v2.0 endpoint: https://login.microsoftonline.com/{tenant}/v2.0
        # v1.0/M2M endpoint: https://sts.windows.net/{tenant}/
        self.issuer_v2 = f"{base_url}/v2.0"
        self.issuer_v1 = f"https://sts.windows.net/{tenant_id}/"
        self.valid_issuers = [self.issuer_v2, self.issuer_v1]

        logger.debug(f"Initialized Entra ID provider for tenant '{tenant_id}'")

    def accepted_audiences(
        self,
        extra_audiences: list[str] | None = None,
    ) -> list[str]:
        """Return the closed allowlist of audiences accepted for this app.

        Entra v1-issued access tokens may carry the ``aud`` claim as either the
        bare client-id GUID (``<app-id>``) or the URI form (``api://<app-id>``);
        both are equivalent and must be accepted (dual-audience normalization).
        The list also includes the operator-configured Application ID URI (which
        for a gateway may be a custom ``api://<uri>``) and any per-server OBO
        resource audiences the caller passes for the server being accessed
        (RFC 8707).

        This is a closed allowlist -- only these statically-known and
        caller-provided, registry-derived audiences are accepted, never a
        wildcard. ``verify_aud`` is always enforced by the caller.

        Args:
            extra_audiences: Per-server OBO resource audiences to also accept
                (e.g. ``https://gw/<server>/mcp``). Each is trailing-slash
                stripped; empty values are ignored.

        Returns:
            Ordered, de-duplicated list of accepted ``aud`` values.
        """
        # Bare GUID + default Application ID URI: the two forms Entra v1 mints
        # for a token audienced to this app.
        audiences: list[str] = [self.client_id, f"api://{self.client_id}"]

        # Operator-configured Application ID URI (may be a custom api://<uri>).
        if self.application_id_uri:
            audiences.append(self.application_id_uri)

        # Per-server OBO resource audiences (caller-provided, registry-derived).
        for extra in extra_audiences or []:
            if extra:
                audiences.append(extra.rstrip("/"))

        # De-duplicate while preserving order.
        seen: set[str] = set()
        deduped: list[str] = []
        for aud in audiences:
            if aud and aud not in seen:
                seen.add(aud)
                deduped.append(aud)
        return deduped

    @staticmethod
    def _reject_non_access_token(claims: dict[str, Any]) -> None:
        """Reject an Entra id_token presented as an access token (fail closed).

        The gateway's data plane consumes **access tokens**. An Entra id_token is
        signed by the same tenant JWKS, carries a valid issuer, and has
        ``aud == client_id`` -- which accepted_audiences() accepts (the bare GUID
        is also a valid v2 access-token audience). So audience + issuer +
        signature do NOT distinguish the two, and a client could replay the
        id_token it receives from the same auth-code exchange as the bearer,
        getting authenticated with authz derived from the id_token ``groups``.
        This mirrors the self-signed path's ``token_use == "access"`` check.

        Discriminator: reject on the presence of id_token-only claims. Entra
        id_tokens issued via the authorization-code flow carry ``nonce`` (bound
        to the login), and OIDC id_tokens carry ``at_hash`` / ``c_hash`` when an
        access/authorization code is co-issued -- none of which appear in an
        Entra access token. We deliberately do NOT require ``scp``/``roles`` to
        be PRESENT: a valid client-credentials access token for an app with no
        assigned app roles carries neither, so requiring them would false-reject
        legitimate M2M tokens. Rejecting on id_token-only claims closes the
        replay vector without that false-positive.

        KNOWN RESIDUAL (accepted, not a believed-closed gap): an id_token minted
        by an authorization-code flow in which the client sent NO ``nonce``
        carries none of ``nonce``/``at_hash``/``c_hash`` (the latter two only
        appear in hybrid/implicit flows), so a claim-presence discriminator lets
        it through. Closing it fully requires a positive access-token signal
        (e.g. requiring ``scp``/``roles``/a resource-scoped ``aud``), which
        reopens the roleless-M2M false-reject above -- an explicit trade-off.
        The residual is narrow: the gateway's own login ALWAYS sends a per-login
        ``nonce`` (see auth_server/server.py), so every id_token our real flows
        produce is rejected; exploiting the gap requires an attacker to drive
        their own nonce-less code flow against the gateway's app registration,
        from which they could equally obtain an access token. Tracked as a known
        limitation rather than silently trusted.

        Raises:
            ValueError: if the token is an id_token rather than an access token.
        """
        for id_token_only in ("nonce", "at_hash", "c_hash"):
            if id_token_only in claims:
                raise ValueError(
                    "Token is an id_token (carries id_token-only claim "
                    f"'{id_token_only}'), not an access token"
                )

    def _scope_prefix(self) -> str | None:
        """Return the ``api://<app-id-or-uri>`` prefix for v1 custom scopes.

        Prefers the operator-configured Application ID URI (which may be a
        custom ``api://<uri>`` registered on the app), falling back to the
        default ``api://<client-id>`` form. Returns None only if no client id
        is available (should not happen for a configured provider).
        """
        if self.application_id_uri:
            return self.application_id_uri
        if self.client_id:
            return f"api://{self.client_id}"
        return None

    def format_advertised_scopes(
        self,
        scopes_supported: list[str],
    ) -> list[str]:
        """Format PRM ``scopes_supported`` for the configured Entra scope form.

        Entra v1 requires custom resource scopes to be requested on
        ``/authorize`` as ``api://<app-id-or-uri>/<scope>``; the bare form is
        rejected with ``AADSTS650053``. Entra v2 accepts the bare fragment. This
        method rewrites each caller-supplied scope for the configured form so
        the PRM advertises exactly what the client must send.

        Standard OIDC scopes (openid/profile/email/offline_access/...) are
        always emitted bare -- Entra rejects ``api://<app>/openid`` even under
        v1 -- as are scopes that are already URI-qualified (contain ``://``,
        e.g. the per-server OBO PRM's ``https://gw/<server>/mcp/...`` resource
        scope, which the caller already fully resolved).

        Args:
            scopes_supported: The scope strings the gateway recognizes, as
                derived by the discovery route.

        Returns:
            The scope list rewritten for the configured Entra scope format.
            Order is preserved; a v2 (or non-v1) provider returns the input
            unchanged.
        """
        if self.scope_format != "v1":
            return list(scopes_supported)

        prefix = self._scope_prefix()
        formatted: list[str] = []
        for scope in scopes_supported:
            if not scope:
                continue
            # Leave OIDC scopes and already URI-qualified scopes untouched.
            if scope in self.STANDARD_OIDC_SCOPES or "://" in scope:
                formatted.append(scope)
                continue
            if prefix:
                formatted.append(f"{prefix}/{scope}")
            else:
                # No resource prefix available: advertise verbatim rather than
                # drop the scope (fail loud in logs, not silent).
                logger.warning(
                    "Entra v1 scope format configured but no Application ID URI "
                    "or client_id available; advertising scope '%s' unqualified",
                    scope,
                )
                formatted.append(scope)
        return formatted

    def protected_resource_metadata(
        self,
        resource: str,
        scopes_supported: list[str],
        resource_documentation: str | None = None,
    ) -> dict[str, Any]:
        """Build the RFC 9728 PRM document, formatting scopes for Entra.

        Overrides the base implementation to apply Entra scope-format rules
        (see :meth:`format_advertised_scopes`) so the advertised
        ``scopes_supported`` are exactly the strings a spec-compliant client
        must send on ``/authorize``. All other fields match the base shape.
        """
        return super().protected_resource_metadata(
            resource=resource,
            scopes_supported=self.format_advertised_scopes(scopes_supported),
            resource_documentation=resource_documentation,
        )

    def validate_token(self, token: str, **kwargs: Any) -> dict[str, Any]:
        """Validate Entra ID JWT token.

        Args:
            token: The JWT access token to validate
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary containing:
                - valid: True if token is valid
                - username: User's preferred_username or sub claim
                - email: User's email address
                - groups: List of Azure AD group Object IDs
                - scopes: List of token scopes
                - client_id: Client ID that issued the token
                - method: 'entra'
                - data: Raw token claims

        Raises:
            ValueError: If token validation fails
        """
        try:
            logger.debug("Validating Entra ID JWT token")

            # First check if this is a self-signed token from our auth server
            try:
                unverified_claims = jwt.decode(token, options={"verify_signature": False})
                if unverified_claims.get("iss") == JWT_ISSUER:
                    logger.debug("Token appears to be self-signed, validating...")
                    return self._validate_self_signed_token(token)
            except Exception as e:
                logger.debug(f"Not a self-signed token: {e}")

            # Get JWKS for validation
            jwks = self.get_jwks()

            # Decode token header to get key ID
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")

            if not kid:
                raise ValueError("Token missing 'kid' in header")

            # Find matching key
            signing_key = None
            for key in jwks.get("keys", []):
                if key.get("kid") == kid:
                    from jwt import PyJWK

                    signing_key = PyJWK(key).key
                    break

            if not signing_key:
                raise ValueError(f"No matching key found for kid: {kid}")

            # First, decode without validation to check issuer
            unverified_claims = jwt.decode(token, options={"verify_signature": False})
            token_issuer = unverified_claims.get("iss")

            # Check if issuer is valid (v1.0 or v2.0)
            if token_issuer not in self.valid_issuers:
                raise ValueError(
                    f"Invalid issuer: {token_issuer}. Expected one of: {self.valid_issuers}"
                )

            # Validate and decode token with the correct issuer. Audience
            # acceptance (bare GUID + api:// forms + per-server OBO resources)
            # is centralized in accepted_audiences() so every entrypoint uses
            # the same closed allowlist. Never a wildcard.
            accepted_audiences = self.accepted_audiences(
                extra_audiences=kwargs.get("extra_audiences")
            )
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                issuer=token_issuer,
                audience=accepted_audiences,
                options={"verify_exp": True, "verify_iat": True, "verify_aud": True},
            )

            # Reject an id_token presented as an access token (token-type
            # confusion): an Entra id_token shares the JWKS/issuer and has
            # aud == client_id, which accepted_audiences() accepts, so only a
            # claim-level discriminator separates them. See the method docstring.
            self._reject_non_access_token(claims)

            logger.debug(
                f"Token validation successful for user: {claims.get('preferred_username', 'unknown')}"
            )

            # Extract group memberships. M2M tokens carry them in 'roles',
            # user tokens in 'groups'. Whichever applies, the claim is strictly
            # validated as a JSON array of strings.
            raw_groups = claims.get("groups")
            if not raw_groups and "roles" in claims:
                raw_groups = claims.get("roles")
            groups = normalize_groups_claim(raw_groups, source="entra groups")

            scopes = _delegated_scopes(claims)
            return {
                "valid": True,
                "username": claims.get("preferred_username", claims.get("sub")),
                "email": claims.get("email"),
                "groups": groups,
                "scopes": scopes,
                "client_id": claims.get("azp", self.client_id),
                "method": "entra",
                "data": claims,
            }

        except jwt.ExpiredSignatureError:
            logger.warning("Token validation failed: Token has expired")
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token validation failed: Invalid token - {e}")
            raise ValueError(f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"Entra ID token validation error: {e}")
            raise ValueError(f"Token validation failed: {e}")

    def validate_id_token(
        self,
        id_token: str,
        expected_nonce: str | None = None,
    ) -> dict[str, Any]:
        """Verify an Entra ID OIDC id_token and return its verified claims.

        Verifies the RS256 signature against the tenant JWKS and enforces
        issuer (Entra v2.0 or v1.0 tenant issuer), audience (the gateway's
        client_id — the id_token ``aud`` for Entra), and expiry before any
        claim is trusted. When ``expected_nonce`` is supplied, the token's
        ``nonce`` claim must match it. Fails closed.

        Args:
            id_token: The raw id_token string from the token endpoint.
            expected_nonce: The nonce bound to this login (replay protection).

        Returns:
            The verified id_token claim set.

        Raises:
            IdTokenVerificationError: If verification fails.
        """
        # Entra id_tokens carry the app registration's client_id as 'aud'
        # (not the api:// Application ID URI, which applies to access tokens).
        accepted_audiences = [self.client_id]
        return self._verify_id_token_with_jwks(
            id_token, self.valid_issuers, accepted_audiences, expected_nonce=expected_nonce
        )

    def _validate_self_signed_token(self, token: str) -> dict[str, Any]:
        """Validate a self-signed JWT token generated by our auth server.

        Self-signed tokens are generated for OAuth users to use for programmatic
        API access. They contain the user's identity, groups, and scopes.

        Args:
            token: The self-signed JWT token to validate

        Returns:
            Dictionary containing validation results

        Raises:
            ValueError: If token validation fails
        """
        try:
            if not SECRET_KEY:
                raise ValueError("SECRET_KEY is required for self-signed token validation")
            claims = jwt.decode(
                token,
                SECRET_KEY,
                algorithms=["HS256"],
                audience=JWT_AUDIENCE,
                issuer=JWT_ISSUER,
                options={"verify_exp": True, "verify_iat": True, "verify_aud": True},
            )

            # Check token_use claim
            token_use = claims.get("token_use")
            if token_use != "access":  # nosec B105 - OAuth2 token type validation per RFC 6749, not a password
                raise ValueError(f"Invalid token_use: {token_use}")

            # Extract scopes from claims
            scopes = []
            if "scope" in claims:
                scope_value = claims["scope"]
                if isinstance(scope_value, str):
                    scopes = scope_value.split() if scope_value else []
                elif isinstance(scope_value, list):
                    scopes = scope_value

            # Extract groups from claims
            groups = normalize_groups_claim(claims.get("groups"), source="entra self-signed groups")

            logger.info(
                f"Successfully validated self-signed token for user: {claims.get('sub')}, "
                f"groups: {groups}, scopes: {scopes}"
            )

            return {
                "valid": True,
                "method": "self_signed",
                "data": claims,
                "client_id": claims.get("client_id", "user-generated"),
                "username": claims.get("sub", ""),
                "email": claims.get("email", ""),
                "expires_at": claims.get("exp"),
                "scopes": scopes,
                "groups": groups,
                "token_type": "user_generated",
            }

        except jwt.ExpiredSignatureError:
            logger.warning("Self-signed token validation failed: Token has expired")
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Self-signed token validation failed: {e}")
            raise ValueError(f"Invalid self-signed token: {e}")
        except Exception as e:
            logger.error(f"Self-signed token validation error: {e}")
            raise ValueError(f"Self-signed token validation failed: {e}")

    def get_jwks(self) -> dict[str, Any]:
        """Get JSON Web Key Set from Entra ID with caching.

        Returns:
            Dictionary containing the JWKS data

        Raises:
            ValueError: If JWKS cannot be retrieved
        """
        current_time = time.time()

        # Check if cache is still valid
        if self._jwks_cache and (current_time - self._jwks_cache_time) < self._jwks_cache_ttl:
            logger.debug("Using cached JWKS")
            return self._jwks_cache

        try:
            logger.debug(f"Fetching JWKS from {self.jwks_url}")
            response = requests.get(self.jwks_url, timeout=10)
            response.raise_for_status()

            self._jwks_cache = response.json()
            self._jwks_cache_time = current_time

            logger.debug("JWKS fetched and cached successfully")
            return self._jwks_cache

        except Exception as e:
            logger.error(f"Failed to retrieve JWKS from Entra ID: {e}")
            raise ValueError(f"Cannot retrieve JWKS: {e}")

    def exchange_code_for_token(self, code: str, redirect_uri: str) -> dict[str, Any]:
        """Exchange authorization code for access token.

        Args:
            code: Authorization code from OAuth2 flow
            redirect_uri: Redirect URI used in the authorization request

        Returns:
            Dictionary containing token response:
                - access_token: The access token
                - id_token: The ID token
                - refresh_token: The refresh token (if available)
                - token_type: "Bearer"
                - expires_in: Token expiration time in seconds

        Raises:
            ValueError: If code exchange fails
        """
        try:
            logger.debug("Exchanging authorization code for token")

            data = {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri": redirect_uri,
            }

            response = _post_entra_token(self.token_url, data)
            response.raise_for_status()

            token_data = response.json()
            logger.debug("Token exchange successful")

            return token_data

        except (httpx.HTTPError, UrlValidationError) as e:
            logger.warning("Entra code token exchange failed; type=%s", type(e).__name__)
            raise ValueError("Token exchange failed") from e

    # Graph supports direct and nested memberships with delegated User.Read.
    GRAPH_MEMBEROF_PATH: str = "/v1.0/me/transitiveMemberOf"
    GROUP_FETCH_HARD_CAP: int = 1000
    GROUP_FETCH_PAGE_HARD_CAP: int = 100
    _group_cache: OrderedDict[tuple[str, str, str, str | None], tuple[float, tuple[str, ...]]] = (
        OrderedDict()
    )
    _group_inflight: dict[tuple[str, str, str, str | None, bytes], asyncio.Task[list[str]]] = {}
    _group_denials: OrderedDict[tuple[str, str, str, str | None, bytes], tuple[float, str]] = (
        OrderedDict()
    )
    _group_verified: OrderedDict[tuple[str, str, str, str | None, bytes], float] = OrderedDict()

    @staticmethod
    def has_group_overage(claims: dict[str, Any]) -> bool:
        """Detect Entra group-overage indicators in an ID token.

        Entra signals overage in two ways:
        - `hasgroups` claim set to True (v1.0 endpoint behavior)
        - `_claim_names` dict containing key `groups` pointing to a Graph
          endpoint (v2.0 endpoint behavior)

        Either form means the inline `groups` claim is unreliable and the
        caller should fall back to Microsoft Graph.
        """
        if claims.get("hasgroups") is True:
            return True
        claim_names = claims.get("_claim_names")
        if isinstance(claim_names, dict) and "groups" in claim_names:
            return True
        return False

    @classmethod
    def _graph_memberof_url(cls) -> str:
        """Build the guarded Graph fetch endpoint, distinct from its OAuth resource."""
        login_resource = _infer_graph_base_url(_entra_login_base())
        resource = _validated_graph_resource_url(login_resource)
        base = _validated_graph_override(resource)
        if base is None:
            if login_resource is None:
                raise GraphGroupLookupError(
                    "Unknown Entra cloud: configure Graph endpoint", reason="configuration"
                )
            base = resource
        return f"{base}{cls.GRAPH_MEMBEROF_PATH}"

    @classmethod
    async def resolve_groups(
        cls,
        claims: dict[str, Any],
        access_token: str,
        *,
        graph_requested: bool = False,
    ) -> list[str]:
        """Use verified inline groups or recover when Graph was requested.

        Raises:
            GraphGroupLookupError: Graph did not establish complete membership.
            ValueError: A present group or role claim has an invalid type.
        """
        inline = normalize_groups_claim(claims.get("groups"), source="entra groups")
        roles = normalize_groups_claim(claims.get("roles"), source="entra roles")
        groups = inline or roles
        overage = cls.has_group_overage(claims)
        if not graph_requested:
            return roles if overage else groups
        if not _group_recovery_enabled():
            return roles if overage else groups
        if not overage and groups:
            return groups

        tid = claims.get("tid")
        oid = claims.get("oid")
        try:
            async with asyncio.timeout(GRAPH_GROUP_LOOKUP_TIMEOUT_SECONDS):
                graph_groups = await cls.fetch_groups_via_graph(
                    access_token,
                    tenant_id=tid if isinstance(tid, str) else None,
                    object_id=oid if isinstance(oid, str) else None,
                )
        except TimeoutError as e:
            raise GraphGroupLookupError(
                "Microsoft Graph group lookup timed out", reason="timeout"
            ) from e
        return list(dict.fromkeys([*roles, *graph_groups]))

    @classmethod
    async def _lookup_bearer_groups(
        cls,
        key: tuple[str, str, str, str | None],
        access_token: str,
        idp_provider: "EntraIdProvider",
    ) -> list[str]:
        """Exchange one delegated user assertion before querying Graph."""
        graph_resource = _graph_resource_url()
        try:
            # nginx closes /validate at ten seconds; cap the whole OBO and
            # paginated Graph chain to leave room for scope mapping and signing.
            async with asyncio.timeout(GRAPH_GROUP_LOOKUP_TIMEOUT_SECONDS):
                graph_token = await obo_exchange(
                    idp_provider,
                    subject_token=access_token,
                    target_audience=graph_resource,
                    scopes=[f"{graph_resource}/User.Read"],
                )
                return await cls.fetch_groups_via_graph(
                    graph_token, tenant_id=key[1], object_id=key[2]
                )
        except OboConsentRequired as e:
            raise GraphGroupLookupError(
                "Microsoft Graph token exchange failed", reason="consent"
            ) from e
        except OboConfigError as e:
            raise GraphGroupLookupError(
                "Microsoft Graph token exchange failed", reason="configuration"
            ) from e
        except OboExchangeError as e:
            raise GraphGroupLookupError(
                "Microsoft Graph token exchange failed", reason="token_exchange"
            ) from e
        except TimeoutError as e:
            raise GraphGroupLookupError(
                "Microsoft Graph group lookup timed out", reason="timeout"
            ) from e
        except (ValueError, AttributeError, TypeError) as e:
            logger.warning("Entra Graph token exchange returned malformed response")
            raise GraphGroupLookupError(
                "Microsoft Graph token exchange failed", reason="malformed"
            ) from e

    @staticmethod
    def is_delegated_user(validation_result: dict[str, Any]) -> bool:
        """Identify a verified Entra user token eligible for group recovery."""
        claims = validation_result.get("data")
        return (
            validation_result.get("method") == "entra"
            and validation_result.get("valid") is True
            and isinstance(claims, dict)
            and bool(_delegated_scopes(claims))
        )

    @classmethod
    def _finish_group_lookup(
        cls,
        key: tuple[str, str, str, str | None, bytes],
        task: asyncio.Task[list[str]],
    ) -> None:
        """Release the single-flight slot after a lookup finishes."""
        if cls._group_inflight.get(key) is task:
            cls._group_inflight.pop(key, None)
        if not task.cancelled():
            task.exception()

    async def recover_bearer_groups(
        self,
        validation_result: dict[str, Any],
        access_token: str,
    ) -> list[str] | None:
        """Recover a verified Entra user's missing or overage group claims.

        Returns:
            Group IDs (including a confirmed empty result), or None if inline
            claims apply or the token is not a delegated Entra user token.

        Raises:
            GraphGroupLookupError: No reliable membership could be resolved.
        """
        if validation_result.get("method") != "entra":
            return None
        if validation_result.get("valid") is not True:
            raise GraphGroupLookupError("Entra token was not verified", reason="identity")
        claims = validation_result.get("data")
        if not isinstance(claims, dict):
            raise GraphGroupLookupError("Verified Entra claims unavailable", reason="identity")
        if not _delegated_scopes(claims):
            return None  # Client-credential tokens have roles, not user scopes.
        inline = normalize_groups_claim(claims.get("groups"), source="entra groups")
        roles = normalize_groups_claim(claims.get("roles"), source="entra roles")
        overage = self.has_group_overage(claims)
        if not overage and (inline or roles):
            return None
        if not _group_recovery_enabled():
            return roles if overage else []

        _group_cache_ttl_seconds()
        tid, oid = claims.get("tid"), claims.get("oid")
        if not isinstance(tid, str) or not tid or not isinstance(oid, str) or not oid:
            raise GraphGroupLookupError(
                "Verified Entra user identity unavailable", reason="identity"
            )
        resource = _graph_resource_url()  # Validate before cache access.
        key = (self._graph_memberof_url(), tid, oid, resource)
        cached = self._group_cache.get(key)
        denial_key = (*key, hashlib.sha256(access_token.encode("utf-8")).digest())
        denied = self._group_denials.get(denial_key)
        if denied and time.monotonic() < denied[0]:
            self._group_denials.move_to_end(denial_key)
            raise GraphGroupLookupError(
                "Microsoft Graph group lookup unavailable", reason=denied[1]
            )
        self._group_denials.pop(denial_key, None)

        verified_until = self._group_verified.get(denial_key)
        if (
            cached
            and time.monotonic() < cached[0]
            and verified_until
            and time.monotonic() < verified_until
        ):
            self._group_cache.move_to_end(key)
            self._group_verified.move_to_end(denial_key)
            graph_groups = list(cached[1])
            return list(dict.fromkeys([*roles, *graph_groups])) if graph_groups else roles
        self._group_verified.pop(denial_key, None)

        task = self._group_inflight.get(denial_key)
        if task is None:
            task = asyncio.create_task(self._lookup_bearer_groups(key, access_token, self))
            self._group_inflight[denial_key] = task
            task.add_done_callback(lambda done: self._finish_group_lookup(denial_key, done))
        try:
            graph_groups = list(await asyncio.shield(task))
        except GraphGroupLookupError as exc:
            if exc.reason in _CACHEABLE_GROUP_FAILURES:
                self._group_denials[denial_key] = (
                    time.monotonic() + GRAPH_GROUP_NEGATIVE_TTL_SECONDS,
                    exc.reason,
                )
                self._group_denials.move_to_end(denial_key)
                while len(self._group_denials) > GRAPH_GROUP_NEGATIVE_MAX_ENTRIES:
                    self._group_denials.popitem(last=False)
            raise
        cached = self._group_cache.get(key)
        if cached and time.monotonic() < cached[0]:
            self._group_verified[denial_key] = cached[0]
            self._group_verified.move_to_end(denial_key)
            while len(self._group_verified) > GRAPH_GROUP_CACHE_MAX_ENTRIES:
                self._group_verified.popitem(last=False)
        return list(dict.fromkeys([*roles, *graph_groups])) if graph_groups else roles

    @classmethod
    async def fetch_groups_via_graph(
        cls,
        access_token: str,
        tenant_id: str | None = None,
        object_id: str | None = None,
    ) -> list[str]:
        """Fetch complete transitive group IDs; cache successful results only.

        Raises:
            GraphGroupLookupError: Graph failed or returned an incomplete response.
        """
        cache_ttl = _group_cache_ttl_seconds()
        url = cls._graph_memberof_url()
        resource = _validated_graph_resource_url(_infer_graph_base_url(_entra_login_base()))
        try:
            origin = urlsplit(url)
            if (
                origin.scheme != "https"
                or not origin.hostname
                or origin.username
                or origin.password
                or origin.query
                or origin.fragment
                or origin.path != cls.GRAPH_MEMBEROF_PATH
                or origin.port not in (None, 443)
            ):
                raise GraphGroupLookupError(
                    "Invalid Microsoft Graph endpoint configuration", reason="configuration"
                )
            # Validate the proxy even on cache hits. The guarded transport
            # resolves and pins each uncached network request separately.
            validate_url(url, profile=ENTRA_GRAPH_PROFILE, resolve=False)
        except (UrlValidationError, ValueError) as e:
            raise GraphGroupLookupError(
                "Invalid Microsoft Graph endpoint configuration", reason="configuration"
            ) from e

        cache_key = (
            (url, tenant_id, object_id, resource)
            if isinstance(tenant_id, str) and tenant_id and isinstance(object_id, str) and object_id
            else None
        )
        if cache_key:
            cached = cls._group_cache.get(cache_key)
            if cached and time.monotonic() < cached[0]:
                cls._group_cache.move_to_end(cache_key)
                return list(cached[1])
            cls._group_cache.pop(cache_key, None)

        ids: list[str] = []
        seen: set[str] = set()
        visited: set[str] = set()
        page = 0
        try:
            async with guarded_async_client(
                profile=ENTRA_GRAPH_PROFILE, timeout=10.0, follow_redirects=False
            ) as client:
                while url:
                    target = urlsplit(url)
                    if (
                        target.scheme != origin.scheme
                        or target.netloc != origin.netloc
                        or target.path != origin.path
                        or target.username
                        or target.password
                        or target.fragment
                        or url in visited
                        or page >= cls.GROUP_FETCH_PAGE_HARD_CAP
                    ):
                        raise GraphGroupLookupError(
                            "Invalid Microsoft Graph pagination link", reason="pagination"
                        )
                    visited.add(url)
                    page += 1
                    response = await client.get(
                        url,
                        headers={"Authorization": f"Bearer {access_token}"},
                        params={"$top": GRAPH_GROUP_PAGE_SIZE} if page == 1 else None,
                    )
                    response.raise_for_status()
                    if response.status_code != 200:
                        raise GraphGroupLookupError(
                            "Microsoft Graph group request was not successful", reason="transport"
                        )
                    body = response.json()
                    if not isinstance(body, dict) or not isinstance(body.get("value"), list):
                        raise GraphGroupLookupError(
                            "Malformed Microsoft Graph group response", reason="malformed"
                        )

                    for item in body["value"]:
                        if not isinstance(item, dict):
                            raise GraphGroupLookupError(
                                "Malformed Microsoft Graph group object", reason="malformed"
                            )
                        if item.get("@odata.type") != "#microsoft.graph.group":
                            continue
                        gid = item.get("id")
                        if not isinstance(gid, str) or not gid:
                            raise GraphGroupLookupError(
                                "Microsoft Graph group missing object ID", reason="malformed"
                            )
                        if gid in seen:
                            continue
                        if len(ids) >= cls.GROUP_FETCH_HARD_CAP:
                            raise GraphGroupLookupError(
                                "Microsoft Graph group limit exceeded", reason="limit"
                            )
                        seen.add(gid)
                        ids.append(gid)

                    next_link = body.get("@odata.nextLink")
                    if next_link is not None and (not isinstance(next_link, str) or not next_link):
                        raise GraphGroupLookupError(
                            "Malformed Microsoft Graph pagination link", reason="pagination"
                        )
                    if next_link and os.environ.get("ENTRA_GRAPH_BASE_URL"):
                        # Graph may return its native origin even when the
                        # request went through an operator's reverse proxy.
                        # Rewrite only the selected OAuth resource's exact
                        # origin/path, never a different cloud or arbitrary URL.
                        resource = _validated_graph_resource_url(
                            _infer_graph_base_url(_entra_login_base())
                        )
                        if resource:
                            link = urlsplit(next_link)
                            graph = urlsplit(resource)
                            if (
                                link.scheme == graph.scheme
                                and link.netloc == graph.netloc
                                and link.path == origin.path
                                and not link.username
                                and not link.password
                                and not link.fragment
                            ):
                                next_link = urlunsplit(
                                    (origin.scheme, origin.netloc, link.path, link.query, "")
                                )
                    url = next_link
        except GraphGroupLookupError:
            raise
        except httpx.HTTPStatusError as e:
            logger.warning("Entra Graph transitiveMemberOf returned %d", e.response.status_code)
            raise GraphGroupLookupError(
                "Microsoft Graph group request failed",
                reason="authorization" if e.response.status_code == 403 else "transport",
            ) from e
        except (httpx.HTTPError, UrlValidationError, ValueError, TypeError) as e:
            logger.warning("Entra Graph transitiveMemberOf request or response failed")
            raise GraphGroupLookupError(
                "Microsoft Graph group request failed", reason="transport"
            ) from e

        logger.info("Resolved %d Entra group IDs via Graph across %d page(s)", len(ids), page)
        if cache_key:
            cls._group_cache[cache_key] = (
                time.monotonic() + cache_ttl,
                tuple(ids),
            )
            cls._group_cache.move_to_end(cache_key)
            while len(cls._group_cache) > GRAPH_GROUP_CACHE_MAX_ENTRIES:
                cls._group_cache.popitem(last=False)
        return ids

    def get_user_info(self, access_token: str) -> dict[str, Any]:
        """Get user information from Entra ID.

        Args:
            access_token: Valid access token

        Returns:
            Dictionary containing user information:
                - username: User's preferred_username
                - email: User's email
                - groups: User's group memberships (Object IDs)

        Raises:
            ValueError: If user info cannot be retrieved
        """
        try:
            login_resource = _infer_graph_base_url(_entra_login_base())
            graph_resource = _validated_graph_resource_url(login_resource)
            graph_origin = _validated_graph_override(graph_resource) or graph_resource
            if self.userinfo_url != f"{graph_origin}/oidc/userinfo":
                raise ValueError("Invalid Microsoft Graph userinfo endpoint")
            validate_url(self.userinfo_url, profile=ENTRA_GRAPH_PROFILE, resolve=False)

            with guarded_client(
                profile=ENTRA_GRAPH_PROFILE, timeout=10.0, follow_redirects=False
            ) as client:
                response = client.get(
                    self.userinfo_url, headers={"Authorization": f"Bearer {access_token}"}
                )
                response.raise_for_status()
                user_info = response.json()
            if not isinstance(user_info, dict):
                raise ValueError("Invalid Microsoft Graph userinfo response")
            logger.debug("User info retrieved from Entra ID")
            return user_info
        except (
            GraphGroupLookupError,
            httpx.HTTPError,
            UrlValidationError,
            ValueError,
            TypeError,
        ):
            logger.warning("Failed to get user info from Entra ID")
            raise ValueError("User info retrieval failed") from None

    def get_auth_url(self, redirect_uri: str, state: str, scope: str | None = None) -> str:
        """Get Entra ID authorization URL.

        Args:
            redirect_uri: URI to redirect to after authorization
            state: State parameter for CSRF protection
            scope: Optional scope parameter (defaults to openid email profile)

        Returns:
            Full authorization URL
        """
        logger.debug(f"Generating auth URL with redirect_uri: {redirect_uri}")

        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "scope": scope or "openid email profile",
            "redirect_uri": redirect_uri,
            "state": state,
        }

        auth_url = f"{self.auth_url}?{urlencode(params)}"
        logger.debug(f"Generated auth URL: {auth_url}")

        return auth_url

    def get_logout_url(self, redirect_uri: str) -> str:
        """Get Entra ID logout URL.

        Args:
            redirect_uri: URI to redirect to after logout

        Returns:
            Full logout URL
        """
        logger.debug(f"Generating logout URL with redirect_uri: {redirect_uri}")

        params = {"client_id": self.client_id, "post_logout_redirect_uri": redirect_uri}

        logout_url = f"{self.logout_url}?{urlencode(params)}"
        logger.debug(f"Generated logout URL: {logout_url}")

        return logout_url

    def refresh_token(self, refresh_token: str) -> dict[str, Any]:
        """Refresh an access token using a refresh token.

        Args:
            refresh_token: The refresh token

        Returns:
            Dictionary containing new token response

        Raises:
            ValueError: If token refresh fails
        """
        try:
            logger.debug("Refreshing access token")

            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }

            response = _post_entra_token(self.token_url, data)
            response.raise_for_status()

            token_data = response.json()
            logger.debug("Token refresh successful")

            return token_data

        except (httpx.HTTPError, UrlValidationError) as e:
            logger.warning("Entra refresh token exchange failed; type=%s", type(e).__name__)
            raise ValueError("Token refresh failed") from e

    def validate_m2m_token(self, token: str) -> dict[str, Any]:
        """Validate a machine-to-machine token.

        Args:
            token: The M2M access token to validate

        Returns:
            Dictionary containing validation result

        Raises:
            ValueError: If token validation fails
        """
        return self.validate_token(token)

    def get_m2m_token(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Get machine-to-machine token using client credentials.

        This method is used for AI agent authentication using Azure AD service principals.
        Each AI agent should have its own service principal (app registration) in Azure AD.

        Args:
            client_id: Optional client ID (uses default if not provided)
            client_secret: Optional client secret (uses default if not provided)
            scope: Optional scope for the token (defaults to .default)

        Returns:
            Dictionary containing token response:
                - access_token: The M2M access token
                - token_type: "Bearer"
                - expires_in: Token expiration time in seconds

        Raises:
            ValueError: If token generation fails
        """
        try:
            logger.debug("Requesting M2M token using client credentials")

            # Default scope for Entra ID M2M tokens
            if not scope:
                scope = f"api://{client_id or self.client_id}/.default"

            data = {
                "grant_type": "client_credentials",
                "client_id": client_id or self.client_id,
                "client_secret": client_secret or self.client_secret,
                "scope": scope,
            }

            response = _post_entra_token(self.token_url, data)
            response.raise_for_status()

            token_data = response.json()
            logger.debug("M2M token generation successful")

            return token_data

        except (httpx.HTTPError, UrlValidationError) as e:
            logger.warning("Entra M2M token exchange failed; type=%s", type(e).__name__)
            raise ValueError("M2M token generation failed") from e

    def initiate_device_code_flow(self, scope: str | None = None) -> dict[str, Any]:
        """Initiate device code flow for user authentication.

        This allows CLI applications to authenticate users by displaying a code
        that the user enters at a browser URL. The user logs in with their
        credentials and the CLI receives a token on their behalf.

        Args:
            scope: OAuth scopes to request (defaults to openid profile email)

        Returns:
            Dictionary containing:
                - device_code: Code for polling
                - user_code: Code for user to enter
                - verification_uri: URL for user to visit
                - expires_in: Seconds until codes expire
                - interval: Polling interval in seconds
                - message: User-friendly instruction message

        Raises:
            ValueError: If device code request fails
        """
        try:
            logger.info("Initiating device code flow")

            # Default scopes for user authentication
            if not scope:
                scope = f"api://{self.client_id}/user_impersonation openid profile email"

            data = {"client_id": self.client_id, "scope": scope}

            # Derive only the sibling endpoint from this provider's token URL.
            device_code_url = self.token_url.replace("/token", "/devicecode")
            response = _post_entra_token(device_code_url, data)
            response.raise_for_status()

            result = response.json()
            logger.info("Device code flow initiated")

            return result

        except (httpx.HTTPError, UrlValidationError) as e:
            logger.warning("Entra device code request failed; type=%s", type(e).__name__)
            raise ValueError("Device code flow initiation failed") from e

    def poll_device_code_token(
        self, device_code: str, interval: int = 5, timeout: int = 300
    ) -> dict[str, Any]:
        """Poll for token after user completes device code authentication.

        Args:
            device_code: The device code from initiate_device_code_flow
            interval: Polling interval in seconds (default 5)
            timeout: Maximum time to wait in seconds (default 300)

        Returns:
            Dictionary containing token response:
                - access_token: The user's access token
                - token_type: "Bearer"
                - expires_in: Token expiration time in seconds
                - refresh_token: Token for refreshing access
                - id_token: OpenID Connect ID token

        Raises:
            ValueError: If polling times out or fails
        """
        try:
            logger.info("Polling for device code token")

            data = {
                "grant_type": "urn:ietf:params:oauth:grant-type:device_code",
                "client_id": self.client_id,
                "device_code": device_code,
            }

            # Each poll checks the destination before sending the device assertion.

            start_time = time.time()

            while (time.time() - start_time) < timeout:
                response = _post_entra_token(self.token_url, data)

                if response.status_code == 200:
                    token_data = response.json()
                    logger.info("Device code authentication successful")
                    return token_data

                error_data = response.json()
                error = error_data.get("error", "")

                if error == "authorization_pending":
                    # User hasn't completed auth yet, keep polling
                    logger.debug("Authorization pending, continuing to poll")
                    time.sleep(interval)
                    continue
                elif error == "slow_down":
                    # Polling too fast, increase interval
                    interval += 5
                    logger.debug(f"Slowing down, new interval: {interval}s")
                    time.sleep(interval)
                    continue
                elif error == "expired_token":
                    raise ValueError("Device code expired. Please start over.")
                elif error == "access_denied":
                    raise ValueError("User denied the authorization request.")
                else:
                    raise ValueError(
                        f"Token request failed: {error_data.get('error_description', error)}"
                    )

            raise ValueError("Device code authentication timed out")

        except (httpx.HTTPError, UrlValidationError) as e:
            logger.warning("Entra device token poll failed; type=%s", type(e).__name__)
            raise ValueError("Device code token polling failed") from e

    def authorization_server_metadata(self) -> dict[str, Any]:
        """Return Entra ID's RFC 8414 metadata for the v2.0 endpoint.

        This emits the Entra v2 OIDC AS metadata (issuer, endpoints, PKCE). The
        v1 issuer (`https://sts.windows.net/{tenant}/`) is a valid token source
        recognized in validate_token but is not advertised as a separate AS
        document. Entra v1 `api://<app-id>/<scope>` verbatim scope-format
        support (sub-issue F, #990) lives on the PRM ``scopes_supported`` array
        via :meth:`format_advertised_scopes`, not here -- the AS metadata's
        static ``scopes_supported`` remains the OIDC-universal set both v1 and
        v2 accept bare.
        """
        return {
            "issuer": self.issuer_v2,
            "authorization_endpoint": self.auth_url,
            "token_endpoint": self.token_url,
            "userinfo_endpoint": self.userinfo_url,
            "jwks_uri": self.jwks_url,
            "end_session_endpoint": self.logout_url,
            "response_types_supported": ["code", "id_token", "code id_token"],
            "grant_types_supported": [
                "authorization_code",
                "refresh_token",
                "client_credentials",
            ],
            "token_endpoint_auth_methods_supported": [
                "client_secret_basic",
                "client_secret_post",
                "private_key_jwt",
            ],
            "code_challenge_methods_supported": ["S256"],
            "subject_types_supported": ["pairwise"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "scopes_supported": ["openid", "email", "profile", "offline_access"],
        }

    def get_provider_info(self) -> dict[str, Any]:
        """Get provider-specific information.

        Returns:
            Dictionary containing provider configuration and endpoints
        """
        return {
            "provider_type": "entra",
            "tenant_id": self.tenant_id,
            "client_id": self.client_id,
            "scope_format": self.scope_format,
            "application_id_uri": self.application_id_uri,
            "endpoints": {
                "auth": self.auth_url,
                "token": self.token_url,
                "userinfo": self.userinfo_url,
                "jwks": self.jwks_url,
                "logout": self.logout_url,
            },
            "issuers": {"v2": self.issuer_v2, "v1": self.issuer_v1},
        }
