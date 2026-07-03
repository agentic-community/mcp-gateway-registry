"""PingFederate authentication provider implementation."""

import logging
import os
import time
from functools import lru_cache
from typing import Any
from urllib.parse import urlencode

import jwt
import requests

from .base import AuthProvider

JWT_ISSUER = os.environ.get("JWT_ISSUER", "mcp-auth-server")
JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "mcp-registry")
# SECRET_KEY is enforced at process startup by auth_server/server.py and
# registry/core/config.py; we read it at import time but do not provide a
# fallback. Self-signed token validation raises if it is missing rather
# than silently using a known-bad value.
SECRET_KEY = os.environ.get("SECRET_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


class PingFederateProvider(AuthProvider):
    """PingFederate authentication provider implementation.

    This provider implements OAuth2/OIDC authentication using PingFederate.
    It supports:
    - User authentication via OAuth2 authorization code flow
    - Machine-to-machine authentication via client credentials flow
    - JWT token validation using PingFederate JWKS
    - Group-based authorization (requires custom ATM attribute contract)

    Endpoint URLs are resolved lazily from PingFederate's RFC 8414
    discovery document (/.well-known/openid-configuration), cached for
    the process lifetime.
    """

    def __init__(
        self,
        base_url: str,
        client_id: str,
        client_secret: str,
        m2m_client_id: str | None = None,
        m2m_client_secret: str | None = None,
        application_id_uri: str | None = None,
        groups_claim: str = "groups",
        m2m_allowed_audiences: list[str] | None = None,
    ):
        """Initialize PingFederate provider.

        Args:
            base_url: PingFederate runtime base URL (e.g., https://pf-host:9031)
            client_id: OAuth2 client ID for web authentication
            client_secret: OAuth2 client secret
            m2m_client_id: Optional separate M2M client ID
            m2m_client_secret: Optional separate M2M client secret
            application_id_uri: Optional resource-server identifier accepted as aud
            groups_claim: JWT claim name for group memberships (default: groups)
            m2m_allowed_audiences: Explicit allowlist of ``aud`` values accepted
                on M2M (client-credentials) access tokens whose audience is a
                resource identifier rather than a client_id. Audience validation
                is ALWAYS enforced against the union of the client ids,
                ``application_id_uri``, and this allowlist — an ``aud`` not
                present is rejected. If empty, only the configured client ids and
                ``application_id_uri`` are accepted, so a token minted for a
                different resource in the same PingFederate tenant fails closed.
        """
        self.base_url = base_url.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.m2m_client_id = m2m_client_id or client_id
        self.m2m_client_secret = m2m_client_secret or client_secret
        self.application_id_uri = application_id_uri
        self.groups_claim = groups_claim
        # Explicit, config-driven allowlist of accepted M2M audiences. Never
        # derived from token claims.
        self.m2m_allowed_audiences = list(m2m_allowed_audiences or [])

        # Discovery URL — all other endpoint URLs are resolved lazily from
        # the cached discovery document on first use.
        self.config_url = f"{self.base_url}/.well-known/openid-configuration"

        # JWKS cache with TTL
        self._jwks_cache: dict[str, Any] | None = None
        self._jwks_cache_time: float = 0.0
        self._jwks_cache_ttl: int = 3600  # 1 hour

    @property
    def token_url(self) -> str:
        """Token endpoint URL from discovery."""
        return self._get_openid_configuration()["token_endpoint"]

    @property
    def jwks_url(self) -> str:
        """JWKS endpoint URL from discovery."""
        return self._get_openid_configuration()["jwks_uri"]

    @property
    def auth_url(self) -> str:
        """Authorization endpoint URL from discovery."""
        return self._get_openid_configuration()["authorization_endpoint"]

    @property
    def userinfo_url(self) -> str:
        """UserInfo endpoint URL from discovery."""
        config = self._get_openid_configuration()
        url = config.get("userinfo_endpoint")
        if not url:
            raise ValueError(
                "PingFederate discovery document does not advertise a userinfo_endpoint"
            )
        return url

    @property
    def logout_url(self) -> str:
        """End-session endpoint URL from discovery."""
        config = self._get_openid_configuration()
        url = config.get("end_session_endpoint")
        if not url:
            raise ValueError(
                "PingFederate discovery document does not advertise an end_session_endpoint. "
                "Verify that SLO is enabled in the PingFederate server configuration."
            )
        return url

    @property
    def issuer(self) -> str:
        """Issuer identifier from discovery."""
        return self._get_openid_configuration()["issuer"]

    @lru_cache(maxsize=1)
    def _get_openid_configuration(self) -> dict[str, Any]:
        """Fetch and cache the OpenID Connect discovery document.

        Uses @lru_cache for process-lifetime caching (matches Keycloak pattern).
        """
        try:
            logger.debug(f"Fetching OpenID configuration from {self.config_url}")
            response = requests.get(self.config_url, timeout=10)
            response.raise_for_status()

            config = response.json()
            logger.debug("OpenID configuration retrieved successfully")

            return config

        except requests.RequestException as e:
            logger.error(f"Failed to get OpenID configuration: {e}")
            raise ValueError(f"OpenID configuration retrieval failed: {e}")

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
            if token_use != "access":  # nosec B105 - OAuth2 token type validation per RFC 6749
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

    def validate_token(self, token: str, **kwargs: Any) -> dict[str, Any]:
        """Validate PingFederate JWT token.

        Checks for self-signed tokens first (iss == mcp-auth-server), then
        validates against PingFederate JWKS using RS256.

        Args:
            token: The JWT access token to validate
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary containing validation results with valid=True,
            username, email, groups, scopes, client_id, method, and data.

        Raises:
            ValueError: If token validation fails
        """
        try:
            logger.debug("Validating PingFederate JWT token")

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

            # Build the explicit set of accepted audiences: the web/M2M client
            # ids, the resource-server identifier (application_id_uri), PLUS any
            # operator-configured M2M audiences. Audience verification is ALWAYS
            # enforced against this allowlist. We never derive verify_aud from
            # unverified claims — a genuinely-signed token from this issuer whose
            # "aud" is not in the allowlist (e.g. one minted for a different
            # resource in the same tenant) is rejected.
            valid_audiences = [self.client_id]
            if self.m2m_client_id and self.m2m_client_id != self.client_id:
                valid_audiences.append(self.m2m_client_id)
            if self.application_id_uri and self.application_id_uri not in valid_audiences:
                valid_audiences.append(self.application_id_uri)
            for aud in self.m2m_allowed_audiences:
                if aud and aud not in valid_audiences:
                    valid_audiences.append(aud)

            # Validate and decode token — audience is always verified against
            # the allowlist above (fail closed).
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                issuer=self.issuer,
                audience=valid_audiences,
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": True,
                },
            )

            logger.debug(f"Token validation successful for user: {claims.get('sub', 'unknown')}")

            # Extract groups using configurable claim name
            groups = claims.get(self.groups_claim, [])
            if not isinstance(groups, list):
                groups = [groups] if groups else []
            if not all(isinstance(g, str) for g in groups):
                raise ValueError("Invalid groups claim format: must contain only strings")

            if not groups:
                logger.warning(
                    "PingFederate token has no '%s' claim for sub=%s. "
                    "Verify JWT ATM extended attribute contract maps groups. "
                    "See docs/idp/pingfederate.md 'Configure custom groups scope'.",
                    self.groups_claim,
                    claims.get("sub", "<unknown>"),
                )

            # Extract scopes — PingFederate uses 'scope' (space-delimited string)
            scope_claim = claims.get("scope", "")
            if isinstance(scope_claim, list):
                scopes = scope_claim
            else:
                scopes = scope_claim.split() if scope_claim else []

            return {
                "valid": True,
                "username": claims.get("sub", claims.get("preferred_username", "")),
                "email": claims.get("email", ""),
                "groups": groups,
                "scopes": scopes,
                "client_id": claims.get("client_id", claims.get("cid", self.client_id)),
                "method": "pingfederate",
                "data": claims,
            }

        except jwt.ExpiredSignatureError:
            logger.warning("Token validation failed: Token has expired")
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token validation failed: Invalid token - {e}")
            raise ValueError(f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"PingFederate token validation error: {e}")
            raise ValueError(f"Token validation failed: {e}")

    def validate_id_token(
        self,
        id_token: str,
        expected_nonce: str | None = None,
    ) -> dict[str, Any]:
        """Verify a PingFederate OIDC id_token and return its verified claims.

        Verifies the RS256 signature against the discovered JWKS and enforces
        issuer (the discovery ``issuer``), audience (the gateway's client_id —
        the id_token ``aud`` for PingFederate), and expiry before any claim is
        trusted. When ``expected_nonce`` is supplied, the token's ``nonce``
        claim must match it. Fails closed.

        Args:
            id_token: The raw id_token string from the token endpoint.
            expected_nonce: The nonce bound to this login (replay protection).

        Returns:
            The verified id_token claim set.

        Raises:
            IdTokenVerificationError: If verification fails.
        """
        accepted_audiences = [self.client_id]
        if self.m2m_client_id and self.m2m_client_id != self.client_id:
            accepted_audiences.append(self.m2m_client_id)
        if self.application_id_uri:
            accepted_audiences.append(self.application_id_uri.rstrip("/"))
        return self._verify_id_token_with_jwks(
            id_token, [self.issuer], accepted_audiences, expected_nonce=expected_nonce
        )

    def get_jwks(self) -> dict[str, Any]:
        """Get JSON Web Key Set from PingFederate with caching.

        Returns cached JWKS if still valid (within TTL), otherwise fetches
        fresh data. Retries once on failure and falls back to stale cache
        if available.

        Returns:
            JWKS dictionary containing keys for token verification

        Raises:
            ValueError: If JWKS cannot be retrieved and no cache exists
        """
        current_time = time.time()

        # Check if cache is still valid
        if self._jwks_cache and (current_time - self._jwks_cache_time) < self._jwks_cache_ttl:
            logger.debug("Using cached JWKS")
            return self._jwks_cache

        # Try to fetch fresh JWKS with retry
        # Do not introduce a verify=False parameter; if TLS verification fails
        # on dev containers, fix the CA bundle (see REQUESTS_CA_BUNDLE in
        # docs/idp/pingfederate.md).
        max_retries = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.debug(f"Fetching JWKS (attempt {attempt + 1})")
                response = requests.get(self.jwks_url, timeout=10)
                response.raise_for_status()

                self._jwks_cache = response.json()
                self._jwks_cache_time = current_time

                logger.debug("JWKS fetched and cached successfully")
                return self._jwks_cache

            except Exception as e:
                last_error = e
                logger.warning(f"JWKS fetch attempt {attempt + 1} failed: {e}")
                if attempt < max_retries - 1:
                    time.sleep(1)

        # Graceful degradation: use stale cache if available
        if self._jwks_cache:
            cache_age = current_time - self._jwks_cache_time
            logger.warning(
                f"JWKS fetch failed after {max_retries} attempts, "
                f"using stale cache (age: {cache_age:.0f}s): {last_error}"
            )
            return self._jwks_cache

        # No cache available, must fail
        logger.error(
            f"Failed to retrieve JWKS from PingFederate (no cache available): {last_error}"
        )
        raise ValueError(f"Cannot retrieve JWKS: {last_error}")

    def exchange_code_for_token(self, code: str, redirect_uri: str) -> dict[str, Any]:
        """Exchange authorization code for access token.

        Args:
            code: Authorization code from PingFederate callback
            redirect_uri: The redirect URI used in the authorization request

        Returns:
            Token response dictionary containing access_token, id_token, etc.

        Raises:
            ValueError: If the token exchange request fails
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
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            }
            response = requests.post(self.token_url, data=data, headers=headers, timeout=10)
            response.raise_for_status()
            token_data = response.json()
            logger.debug("Token exchange successful")
            return token_data
        except requests.RequestException as e:
            logger.error(f"Failed to exchange code for token: {e}")
            raise ValueError(f"Token exchange failed: {e}")

    def get_user_info(self, access_token: str) -> dict[str, Any]:
        """Get user information from PingFederate.

        Args:
            access_token: Valid PingFederate access token

        Returns:
            User info dictionary from PingFederate userinfo endpoint

        Raises:
            ValueError: If the userinfo request fails
        """
        try:
            logger.debug("Fetching user info from PingFederate")
            headers = {"Authorization": f"Bearer {access_token}"}
            response = requests.get(self.userinfo_url, headers=headers, timeout=10)
            response.raise_for_status()
            user_info = response.json()
            logger.debug(f"User info retrieved for: {user_info.get('sub', 'unknown')}")
            return user_info
        except requests.RequestException as e:
            logger.error(f"Failed to get user info: {e}")
            raise ValueError(f"User info retrieval failed: {e}")

    def get_auth_url(self, redirect_uri: str, state: str, scope: str | None = None) -> str:
        """Get PingFederate authorization URL.

        Args:
            redirect_uri: The redirect URI after authentication
            state: CSRF protection state parameter
            scope: OAuth2 scopes (defaults to 'openid email profile groups')

        Returns:
            Authorization URL string
        """
        logger.debug(f"Generating auth URL with redirect_uri: {redirect_uri}")
        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "scope": scope or "openid email profile groups",
            "redirect_uri": redirect_uri,
            "state": state,
        }
        auth_url = f"{self.auth_url}?{urlencode(params)}"
        logger.debug(f"Generated auth URL for endpoint: {self.auth_url}")
        return auth_url

    def get_logout_url(self, redirect_uri: str) -> str:
        """Get PingFederate logout URL.

        Args:
            redirect_uri: URI to redirect to after logout

        Returns:
            Full logout URL with client_id and post_logout_redirect_uri params

        Raises:
            ValueError: If end_session_endpoint is not in the discovery document
        """
        logger.debug(f"Generating logout URL with redirect_uri: {redirect_uri}")

        params = {
            "client_id": self.client_id,
            "post_logout_redirect_uri": redirect_uri,
        }

        logout_url = f"{self.logout_url}?{urlencode(params)}"
        logger.debug(f"Generated logout URL for endpoint: {self.logout_url}")

        return logout_url

    def refresh_token(self, refresh_token: str) -> dict[str, Any]:
        """Refresh an access token using a refresh token.

        Args:
            refresh_token: The refresh token from a previous token response

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

            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            }

            response = requests.post(
                self.token_url,
                data=data,
                headers=headers,
                timeout=10,
            )
            response.raise_for_status()

            token_data = response.json()
            logger.debug("Token refresh successful")

            return token_data

        except requests.RequestException as e:
            logger.error(f"Failed to refresh token: {e}")
            raise ValueError(f"Token refresh failed: {e}")

    def validate_m2m_token(self, token: str) -> dict[str, Any]:
        """Validate a machine-to-machine token.

        Delegates to the standard validate_token() method since M2M tokens
        use the same JWT validation logic as user tokens.

        Args:
            token: JWT token string to validate

        Returns:
            Validated token information dictionary

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

        Args:
            client_id: Optional override client ID (defaults to configured M2M client ID)
            client_secret: Optional override client secret
            scope: Optional scope string (defaults to 'openid')

        Returns:
            Token response dictionary containing access_token, etc.

        Raises:
            ValueError: If the M2M token request fails
        """
        try:
            logger.debug("Requesting M2M token using client credentials")
            data = {
                "grant_type": "client_credentials",
                "client_id": client_id or self.m2m_client_id,
                "client_secret": client_secret or self.m2m_client_secret,
                "scope": scope or "openid",
            }
            headers = {
                "Content-Type": "application/x-www-form-urlencoded",
                "Accept": "application/json",
            }
            response = requests.post(self.token_url, data=data, headers=headers, timeout=10)
            response.raise_for_status()
            token_data = response.json()
            logger.debug("M2M token generation successful")
            return token_data
        except requests.RequestException as e:
            logger.error(f"Failed to get M2M token: {e}")
            raise ValueError(f"M2M token generation failed: {e}")

    def authorization_server_metadata(self) -> dict[str, Any]:
        """Return PingFederate's RFC 8414 metadata.

        Returns the cached discovery document directly since PingFederate's
        /.well-known/openid-configuration is already RFC 8414-compliant.
        """
        return self._get_openid_configuration()

    def get_provider_info(self) -> dict[str, Any]:
        """Get provider-specific information.

        Returns:
            Dictionary containing provider configuration and endpoints
        """
        try:
            config = self._get_openid_configuration()
            return {
                "provider_type": "pingfederate",
                "base_url": self.base_url,
                "client_id": self.client_id,
                "endpoints": {
                    "auth": config.get("authorization_endpoint"),
                    "token": config.get("token_endpoint"),
                    "userinfo": config.get("userinfo_endpoint"),
                    "jwks": config.get("jwks_uri"),
                    "logout": config.get("end_session_endpoint"),
                    "config": self.config_url,
                },
                "issuer": config.get("issuer"),
            }
        except ValueError:
            return {
                "provider_type": "pingfederate",
                "base_url": self.base_url,
                "client_id": self.client_id,
                "status": "discovery_unavailable",
            }
