"""Logto authentication provider implementation.

Logto (https://logto.io) is an OIDC-compliant identity provider. Endpoint
dialect: authorization ``/oidc/auth``, token ``/oidc/token``, userinfo
``/oidc/me``, JWKS ``/oidc/jwks``, end-session ``/oidc/session/end``. The
token issuer is the configured public endpoint plus ``/oidc`` (e.g.
``https://auth.example.com/oidc``).

Server-to-server calls (token, userinfo, JWKS, discovery) use the internal
base URL; browser-facing endpoints (authorization, end-session) use the
external URL - the same split the Keycloak provider uses.

Machine-to-machine: Logto requires a ``resource`` (API indicator) on
client_credentials token requests; the token's ``aud`` is that resource.
"""

import logging
import os
import time
from functools import lru_cache
from typing import Any
from urllib.parse import urlencode

import jwt
import requests

from .base import AuthProvider

# Constants for self-signed token validation
JWT_ISSUER = os.environ.get("JWT_ISSUER", "mcp-auth-server")
JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "mcp-registry")
# SECRET_KEY is enforced at process startup by auth_server/server.py and
# registry/core/config.py; we read it at import time but do not provide
# a fallback. Self-signed token validation (which consumes this constant)
# raises if it is missing rather than silently using a known-bad value.
SECRET_KEY = os.environ.get("SECRET_KEY")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


class LogtoProvider(AuthProvider):
    """Logto authentication provider implementation."""

    def __init__(
        self,
        logto_url: str,
        client_id: str,
        client_secret: str,
        logto_external_url: str | None = None,
        m2m_client_id: str | None = None,
        m2m_client_secret: str | None = None,
        m2m_resource: str | None = None,
    ):
        """Initialize Logto provider.

        Args:
            logto_url: Base URL of the Logto instance for server-to-server
                communication (e.g. ``http://idp-logto:3001``).
            client_id: OAuth2 client ID of the gateway's web application.
            client_secret: OAuth2 client secret of the web application.
            logto_external_url: External URL for browser redirects
                (e.g. ``https://auth.example.com``); defaults to logto_url.
            m2m_client_id: Optional M2M application client ID.
            m2m_client_secret: Optional M2M application client secret.
            m2m_resource: API resource indicator required by Logto on
                client_credentials requests (the resulting token ``aud``).
        """
        self.logto_url = logto_url.rstrip("/")
        self.logto_external_url = (logto_external_url or logto_url).rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.m2m_client_id = m2m_client_id or client_id
        self.m2m_client_secret = m2m_client_secret or client_secret
        self.m2m_resource = m2m_resource

        # Cache for JWKS and configuration
        self._jwks_cache: dict[str, Any] | None = None
        self._jwks_cache_time: float = 0
        self._jwks_cache_ttl: int = 3600  # 1 hour

        # Logto endpoints - internal URL for server-to-server, external for browser
        self.oidc_base = f"{self.logto_url}/oidc"
        self.external_oidc_base = f"{self.logto_external_url}/oidc"
        # Logto derives the issuer from its configured public endpoint.
        self.issuer = self.external_oidc_base
        self.token_url = f"{self.oidc_base}/token"
        self.userinfo_url = f"{self.oidc_base}/me"
        self.jwks_url = f"{self.oidc_base}/jwks"
        self.config_url = f"{self.oidc_base}/.well-known/openid-configuration"
        self.auth_url = f"{self.external_oidc_base}/auth"
        self.logout_url = f"{self.external_oidc_base}/session/end"

        logger.debug(
            f"Initialized Logto provider at {logto_url} (external: {self.logto_external_url})"
        )

    def validate_token(self, token: str, **kwargs: Any) -> dict[str, Any]:
        """Validate Logto JWT token."""
        try:
            logger.debug("Validating Logto JWT token")

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

            # Logto has a single issuer: the public endpoint + /oidc.
            valid_issuers = [self.issuer]

            # Accepted audiences: only audiences that identify THIS gateway.
            #   - client_id / m2m_client_id (the gateway's own applications)
            #   - m2m_resource (Logto client_credentials tokens carry the
            #     API resource indicator as their audience)
            accepted_audiences = [
                self.client_id,
                self.m2m_client_id,
                self.m2m_resource,
            ]

            claims = None
            last_error = None
            for issuer in valid_issuers:
                try:
                    claims = jwt.decode(
                        token,
                        signing_key,
                        algorithms=["ES384", "RS256"],
                        issuer=issuer,
                        audience=accepted_audiences,
                        options={"verify_exp": True, "verify_iat": True, "verify_aud": True},
                    )
                    logger.debug(f"Token validation successful with issuer: {issuer}")
                    break
                except jwt.InvalidIssuerError as e:
                    last_error = e
                    continue

            if claims is None:
                raise last_error or ValueError("Token validation failed with all valid issuers")

            logger.debug(
                f"Token validation successful for subject: {claims.get('sub', 'unknown')}"
            )

            # Logto exposes role names in the `roles` claim (also in userinfo).
            roles = claims.get("roles", [])
            if isinstance(roles, str):
                roles = [roles]

            # Extract user info from claims
            return {
                "valid": True,
                "username": claims.get("username", claims.get("sub")),
                "email": claims.get("email"),
                "groups": roles,
                "scopes": claims.get("scope", "").split() if claims.get("scope") else [],
                "client_id": claims.get("client_id", claims.get("aud", self.client_id)),
                "method": "logto",
                "data": claims,
            }

        except jwt.ExpiredSignatureError:
            logger.warning("Token validation failed: Token has expired")
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token validation failed: Invalid token - {e}")
            raise ValueError(f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"Logto token validation error: {e}")
            raise ValueError(f"Token validation failed: {e}")

    def validate_id_token(
        self,
        id_token: str,
        expected_nonce: str | None = None,
    ) -> dict[str, Any]:
        """Verify a Logto OIDC id_token and return its verified claims.

        Verifies the RS256 signature against the Logto JWKS and enforces
        issuer (the public endpoint + /oidc), audience (the gateway's web
        client_id, which is the id_token ``aud``), and expiry before any
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
        valid_issuers = [self.issuer]
        # Logto sets the id_token 'aud' to the client that requested it.
        accepted_audiences = [self.client_id, self.m2m_client_id]
        return self._verify_id_token_with_jwks(
            id_token,
            valid_issuers,
            accepted_audiences,
            expected_nonce=expected_nonce,
            algorithms=["ES384", "RS256"],
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

            # Counts only: group names are organizational PII and the scope list
            # reveals the authz model. The subject is masked (may be an email).
            _sub = str(claims.get("sub") or "")
            _masked_sub = f"{_sub[:4]}***" if _sub else "unknown"
            logger.info(
                "Successfully validated self-signed token for user %s (groups=%d, scopes=%d)",
                _masked_sub,
                len(groups),
                len(scopes),
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
        """Get JSON Web Key Set from Logto with caching."""
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
            logger.error(f"Failed to retrieve JWKS from Logto: {e}")
            raise ValueError(f"Cannot retrieve JWKS: {e}")

    def exchange_code_for_token(self, code: str, redirect_uri: str) -> dict[str, Any]:
        """Exchange authorization code for access token."""
        try:
            logger.debug("Exchanging authorization code for token")

            data = {
                "grant_type": "authorization_code",
                "code": code,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
                "redirect_uri": redirect_uri,
            }

            response = requests.post(self.token_url, data=data, timeout=10)
            response.raise_for_status()

            token_data = response.json()
            logger.debug("Token exchange successful")

            return token_data

        except requests.RequestException as e:
            logger.error(f"Failed to exchange code for token: {e}")
            raise ValueError(f"Token exchange failed: {e}")

    def get_user_info(self, access_token: str) -> dict[str, Any]:
        """Get user information from Logto."""
        try:
            logger.debug("Fetching user info from Logto")

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
        """Get Logto authorization URL."""
        logger.debug(f"Generating auth URL with redirect_uri: {redirect_uri}")

        params = {
            "client_id": self.client_id,
            "response_type": "code",
            "scope": scope or "openid offline_access profile email roles",
            "redirect_uri": redirect_uri,
            "state": state,
        }

        auth_url = f"{self.auth_url}?{urlencode(params)}"
        logger.debug(f"Generated auth URL: {auth_url}")

        return auth_url

    def get_logout_url(self, redirect_uri: str) -> str:
        """Get Logto end-session URL."""
        logger.debug(f"Generating logout URL with redirect_uri: {redirect_uri}")

        params = {"client_id": self.client_id, "post_logout_redirect_uri": redirect_uri}

        logout_url = f"{self.logout_url}?{urlencode(params)}"
        logger.debug(f"Generated logout URL: {logout_url}")

        return logout_url

    def refresh_token(self, refresh_token: str) -> dict[str, Any]:
        """Refresh an access token using a refresh token."""
        try:
            logger.debug("Refreshing access token")

            data = {
                "grant_type": "refresh_token",
                "refresh_token": refresh_token,
                "client_id": self.client_id,
                "client_secret": self.client_secret,
            }

            response = requests.post(self.token_url, data=data, timeout=10)
            response.raise_for_status()

            token_data = response.json()
            logger.debug("Token refresh successful")

            return token_data

        except requests.RequestException as e:
            logger.error(f"Failed to refresh token: {e}")
            raise ValueError(f"Token refresh failed: {e}")

    def validate_m2m_token(self, token: str) -> dict[str, Any]:
        """Validate a machine-to-machine token."""
        # M2M tokens use the same validation as regular tokens
        return self.validate_token(token)

    def get_m2m_token(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Get machine-to-machine token using client credentials.

        Logto requires a ``resource`` (API indicator) on client_credentials
        requests; the resulting access token's ``aud`` is that resource.
        """
        try:
            logger.debug("Requesting M2M token using client credentials")

            data = {
                "grant_type": "client_credentials",
                "client_id": client_id or self.m2m_client_id,
                "client_secret": client_secret or self.m2m_client_secret,
                "resource": self.m2m_resource,
            }
            if scope:
                data["scope"] = scope

            response = requests.post(self.token_url, data=data, timeout=10)
            response.raise_for_status()

            token_data = response.json()
            logger.debug("M2M token generation successful")

            return token_data

        except requests.RequestException as e:
            logger.error(f"Failed to get M2M token: {e}")
            raise ValueError(f"M2M token generation failed: {e}")

    def authorization_server_metadata(self) -> dict[str, Any]:
        """Return Logto's OIDC discovery document, internal hostnames rewritten.

        We fetch discovery from the internal cluster URL but rewrite any
        browser-facing endpoints onto the external URL so a discovery client
        lands on the correct host.
        """
        config = self._get_openid_configuration()
        if self.logto_url == self.logto_external_url:
            return dict(config)

        rewritten: dict[str, Any] = dict(config)
        for field in (
            "issuer",
            "authorization_endpoint",
            "token_endpoint",
            "userinfo_endpoint",
            "jwks_uri",
            "end_session_endpoint",
            "introspection_endpoint",
            "registration_endpoint",
            "revocation_endpoint",
            "device_authorization_endpoint",
        ):
            value = rewritten.get(field)
            if isinstance(value, str) and value.startswith(self.logto_url):
                rewritten[field] = value.replace(self.logto_url, self.logto_external_url, 1)
        return rewritten

    @lru_cache(maxsize=1)
    def _get_openid_configuration(self) -> dict[str, Any]:
        """Get OpenID Connect discovery document from Logto."""
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

    def _check_logto_health(self) -> bool:
        """Check if Logto is healthy and accessible (serves discovery)."""
        try:
            response = requests.get(self.config_url, timeout=5)
            return response.status_code == 200
        except Exception:
            return False

    def get_provider_info(self) -> dict[str, Any]:
        """Get provider-specific information."""
        return {
            "provider_type": "logto",
            "logto_url": self.logto_url,
            "logto_external_url": self.logto_external_url,
            "client_id": self.client_id,
            "endpoints": {
                "auth": self.auth_url,
                "token": self.token_url,
                "userinfo": self.userinfo_url,
                "jwks": self.jwks_url,
                "logout": self.logout_url,
                "config": self.config_url,
            },
            "healthy": self._check_logto_health(),
        }
