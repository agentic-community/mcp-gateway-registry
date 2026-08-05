"""Microsoft Entra ID (Azure AD) authentication provider implementation."""

import logging
import os
import time
from typing import Any
from urllib.parse import urlencode

import httpx
import jwt
import requests

from .base import AuthProvider

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

# Default Entra ID login base URL. Sovereign clouds override via env:
#   Public (default): https://login.microsoftonline.com
#   US Gov:           https://login.microsoftonline.us
#   China:            https://login.partner.microsoftonline.cn
DEFAULT_ENTRA_LOGIN_BASE_URL = "https://login.microsoftonline.com"

# Default Microsoft Graph base URL. Inferred from the login base URL via the
# fixed sovereign-cloud mapping below; ENTRA_GRAPH_BASE_URL can override for
# edge cases (e.g. a Graph proxy in front of the cluster). Operators on the
# three documented sovereign clouds only need to set ENTRA_LOGIN_BASE_URL.
DEFAULT_ENTRA_GRAPH_BASE_URL = "https://graph.microsoft.com"

# Fixed Microsoft mapping from login host -> Graph host across sovereign clouds.
# Documented at https://learn.microsoft.com/en-us/graph/deployments
_LOGIN_TO_GRAPH_HOST: dict[str, str] = {
    "login.microsoftonline.com": "https://graph.microsoft.com",
    "login.microsoftonline.us": "https://graph.microsoft.us",
    "login.partner.microsoftonline.cn": "https://microsoftgraph.chinacloudapi.cn",
}


def _infer_graph_base_url(login_base_url: str) -> str:
    """Infer the Graph base URL from the configured login base URL.

    Returns the matching Graph host for one of Microsoft's three documented
    sovereign clouds. Unknown login hosts fall back to the public Graph
    endpoint; operators should set ENTRA_GRAPH_BASE_URL explicitly in that
    case (e.g. air-gapped or proxied deployments).
    """
    from urllib.parse import urlparse

    host = urlparse(login_base_url).hostname or ""
    return _LOGIN_TO_GRAPH_HOST.get(host, DEFAULT_ENTRA_GRAPH_BASE_URL)


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

        # Login base URL: explicit env override, defaulting to the public
        # cloud. Graph base URL: explicit override (rare — for proxied
        # deployments) takes precedence; otherwise inferred from the login
        # base URL via the documented sovereign-cloud mapping.
        login_base_url = os.environ.get("ENTRA_LOGIN_BASE_URL", DEFAULT_ENTRA_LOGIN_BASE_URL)
        graph_override = os.environ.get("ENTRA_GRAPH_BASE_URL")
        self.graph_base_url = (
            graph_override.rstrip("/") if graph_override else _infer_graph_base_url(login_base_url)
        )

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

            # Extract user info from claims
            # For M2M tokens, group memberships are in 'roles' claim instead of 'groups'
            # For user tokens, they're in 'groups' claim
            groups = claims.get("groups", [])
            if not groups and "roles" in claims:
                # M2M token - use roles claim as groups
                groups = claims.get("roles", [])
                # Count only: role/group names reveal the internal authz structure.
                logger.debug("M2M token detected, using %d roles as groups", len(groups))

            return {
                "valid": True,
                "username": claims.get("preferred_username", claims.get("sub")),
                "email": claims.get("email"),
                "groups": groups,
                "scopes": claims.get("scope", "").split() if claims.get("scope") else [],
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
            groups = claims.get("groups", [])
            if isinstance(groups, str):
                groups = [groups]

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

            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            response = requests.post(self.token_url, data=data, headers=headers, timeout=10)
            response.raise_for_status()

            token_data = response.json()
            logger.debug("Token exchange successful")

            return token_data

        except requests.RequestException as e:
            logger.error(f"Failed to exchange code for token: {e}")
            raise ValueError(f"Token exchange failed: {e}")

    # Path for the user's direct group memberships. Combined with the
    # tenant's Graph base URL (varies by sovereign cloud) at call time. We
    # use /me/memberOf rather than /me/getMemberObjects because memberOf
    # works with the User.Read scope already granted by 'openid email profile';
    # getMemberObjects requires Directory.Read.All / GroupMember.Read.All
    # which would force a tenant admin to grant new consent.
    GRAPH_MEMBEROF_PATH: str = "/v1.0/me/memberOf?$select=id"

    # Hard cap so a misconfigured tenant cannot pull an unbounded list. 1000
    # covers any realistic user; pages past this are dropped with a warning.
    GROUP_FETCH_HARD_CAP: int = 1000

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
        """Build the full Graph /me/memberOf URL.

        Resolves the base URL at call time using the same precedence as
        __init__ (explicit ENTRA_GRAPH_BASE_URL override, otherwise inferred
        from ENTRA_LOGIN_BASE_URL). Resolved per-call so sovereign-cloud
        overrides set after module import are honored.
        """
        graph_override = os.environ.get("ENTRA_GRAPH_BASE_URL")
        if graph_override:
            base = graph_override.rstrip("/")
        else:
            login_base = os.environ.get("ENTRA_LOGIN_BASE_URL", DEFAULT_ENTRA_LOGIN_BASE_URL)
            base = _infer_graph_base_url(login_base)
        return f"{base}{cls.GRAPH_MEMBEROF_PATH}"

    @classmethod
    async def fetch_groups_via_graph(cls, access_token: str) -> list[str]:
        """Fetch the user's direct group object IDs from Microsoft Graph.

        Used when the ID token signals group overage (see has_group_overage).
        Calls GET /me/memberOf, follows @odata.nextLink for pagination, and
        returns deduplicated group object IDs only (filters out directoryRole
        and other directory-object types).

        Returns [] on any HTTP/network failure so the caller can degrade
        gracefully — the user ends up with whatever groups were inline (often
        none in the overage case), which is the same as today's behavior.
        """
        ids: list[str] = []
        seen: set[str] = set()
        url: str | None = cls._graph_memberof_url()

        try:
            async with httpx.AsyncClient(timeout=10.0) as client:
                page = 0
                while url:
                    page += 1
                    response = await client.get(
                        url, headers={"Authorization": f"Bearer {access_token}"}
                    )
                    response.raise_for_status()
                    body = response.json()

                    for item in body.get("value", []):
                        if item.get("@odata.type") != "#microsoft.graph.group":
                            continue
                        gid = item.get("id")
                        if not gid or gid in seen:
                            continue
                        seen.add(gid)
                        ids.append(gid)
                        if len(ids) >= cls.GROUP_FETCH_HARD_CAP:
                            logger.warning(
                                "Entra Graph group fetch hit hard cap "
                                f"({cls.GROUP_FETCH_HARD_CAP}); truncating"
                            )
                            return ids

                    url = body.get("@odata.nextLink")

                logger.info(
                    f"Resolved {len(ids)} Entra group IDs via Graph memberOf across {page} page(s)"
                )
        except httpx.HTTPStatusError as e:
            logger.warning(
                f"Entra Graph memberOf returned {e.response.status_code}; "
                "falling back to inline groups (may be empty)"
            )
            return []
        except httpx.HTTPError as e:
            logger.warning(f"Entra Graph memberOf request failed: {e}")
            return []
        except Exception as e:
            logger.warning(f"Unexpected error during Entra Graph memberOf fetch: {e}")
            return []

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
            logger.debug("Fetching user info from Entra ID")

            headers = {"Authorization": f"Bearer {access_token}"}
            response = requests.get(self.userinfo_url, headers=headers, timeout=10)
            response.raise_for_status()

            user_info = response.json()
            logger.debug(
                f"User info retrieved for: {user_info.get('preferred_username', 'unknown')}"
            )

            return user_info

        except requests.RequestException as e:
            logger.error(f"Failed to get user info: {e}")
            raise ValueError(f"User info retrieval failed: {e}")

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

            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            response = requests.post(self.token_url, data=data, headers=headers, timeout=10)
            response.raise_for_status()

            token_data = response.json()
            logger.debug("Token refresh successful")

            return token_data

        except requests.RequestException as e:
            logger.error(f"Failed to refresh token: {e}")
            raise ValueError(f"Token refresh failed: {e}")

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

            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            response = requests.post(self.token_url, data=data, headers=headers, timeout=10)
            response.raise_for_status()

            token_data = response.json()
            logger.debug("M2M token generation successful")

            return token_data

        except requests.RequestException as e:
            logger.error(f"Failed to get M2M token: {e}")
            raise ValueError(f"M2M token generation failed: {e}")

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

            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            # Device code endpoint
            device_code_url = self.token_url.replace("/token", "/devicecode")

            response = requests.post(device_code_url, data=data, headers=headers, timeout=10)
            response.raise_for_status()

            result = response.json()
            logger.info(f"Device code flow initiated, user_code: {result.get('user_code')}")

            return result

        except requests.RequestException as e:
            logger.error(f"Failed to initiate device code flow: {e}")
            raise ValueError(f"Device code flow initiation failed: {e}")

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

            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            start_time = time.time()

            while (time.time() - start_time) < timeout:
                response = requests.post(self.token_url, data=data, headers=headers, timeout=10)

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

        except requests.RequestException as e:
            logger.error(f"Failed to poll device code token: {e}")
            raise ValueError(f"Device code token polling failed: {e}")

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
