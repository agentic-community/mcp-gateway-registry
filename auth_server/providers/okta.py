"""Okta authentication provider implementation."""

import logging
import os
import re
import time
from typing import Any
from urllib.parse import urlencode

import jwt
import requests

from .base import AuthProvider

# Constants for self-signed token validation
JWT_ISSUER = os.environ.get("JWT_ISSUER", "mcp-auth-server")
JWT_AUDIENCE = os.environ.get("JWT_AUDIENCE", "mcp-registry")
SECRET_KEY = os.environ.get("SECRET_KEY", "development-secret-key")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


class OktaProvider(AuthProvider):
    """Okta authentication provider implementation.

    This provider implements OAuth2/OIDC authentication using Okta.
    It supports:
    - User authentication via OAuth2 authorization code flow
    - Machine-to-machine authentication via client credentials flow
    - JWT token validation using Okta JWKS
    - Group-based authorization with Okta groups
    """

    def __init__(
        self,
        okta_domain: str,
        client_id: str,
        client_secret: str,
        m2m_client_id: str | None = None,
        m2m_client_secret: str | None = None,
    ):
        """Initialize Okta provider.

        Args:
            okta_domain: Okta org domain (e.g., dev-123456.okta.com)
            client_id: OAuth2 client ID for web authentication
            client_secret: OAuth2 client secret
            m2m_client_id: Optional separate M2M client ID
            m2m_client_secret: Optional separate M2M client secret
        """
        # Normalize domain (remove https:// if present)
        self.okta_domain = okta_domain.replace("https://", "").rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.m2m_client_id = m2m_client_id or client_id
        self.m2m_client_secret = m2m_client_secret or client_secret

        # Validate Okta domain format (security: warn on non-standard domains)
        standard_okta_pattern = r"^[a-zA-Z0-9-]+\.(okta\.com|oktapreview\.com|okta-emea\.com)$"
        if not re.match(standard_okta_pattern, self.okta_domain):
            logger.warning(
                f"Non-standard Okta domain: {self.okta_domain}. "
                f"Expected format: *.okta.com, *.oktapreview.com, or *.okta-emea.com"
            )

        # JWKS cache
        self._jwks_cache: dict[str, Any] | None = None
        self._jwks_cache_time: float = 0
        self._jwks_cache_ttl: int = 3600  # 1 hour

        # Check for custom authorization server
        auth_server_id = os.environ.get("OKTA_AUTH_SERVER_ID", "")

        # Okta endpoints (org or custom authorization server)
        base_url = f"https://{self.okta_domain}"
        if auth_server_id:
            # Custom authorization server endpoints
            oauth2_base = f"{base_url}/oauth2/{auth_server_id}/v1"
            self.auth_url = f"{oauth2_base}/authorize"
            self.token_url = f"{oauth2_base}/token"
            self.userinfo_url = f"{oauth2_base}/userinfo"
            self.jwks_url = f"{oauth2_base}/keys"
            self.logout_url = f"{oauth2_base}/logout"
            self.issuer = f"{base_url}/oauth2/{auth_server_id}"
            logger.info(
                f"Initialized Okta provider with custom authorization server '{auth_server_id}'"
            )
        else:
            # Default org authorization server endpoints
            self.auth_url = f"{base_url}/oauth2/v1/authorize"
            self.token_url = f"{base_url}/oauth2/v1/token"
            self.userinfo_url = f"{base_url}/oauth2/v1/userinfo"
            self.jwks_url = f"{base_url}/oauth2/v1/keys"
            self.logout_url = f"{base_url}/oauth2/v1/logout"
            self.issuer = base_url
            logger.info(f"Initialized Okta provider for domain '{self.okta_domain}'")

    def validate_token(self, token: str, **kwargs: Any) -> dict[str, Any]:
        """Validate Okta JWT token.

        Checks for self-signed tokens first (iss == mcp-auth-server), then
        validates against Okta JWKS using RS256.

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
            logger.debug("Validating Okta JWT token")

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

            # Accept both web client_id and M2M client_id as valid audiences
            valid_audiences = [self.client_id]
            if self.m2m_client_id and self.m2m_client_id != self.client_id:
                valid_audiences.append(self.m2m_client_id)

            # For custom authorization servers, M2M tokens use API identifier as audience
            # Decode without audience validation first to check token type
            unverified_claims = jwt.decode(token, options={"verify_signature": False})

            # Check if this is an M2M token (has cid but audience is not client_id)
            is_m2m_token = "cid" in unverified_claims
            aud_claim = unverified_claims.get("aud", "")
            aud_is_client_id = aud_claim in valid_audiences

            # For M2M tokens with custom auth server, skip audience validation
            # since Okta uses API identifier (e.g., "api://ai-registry") as audience
            verify_audience = not (is_m2m_token and not aud_is_client_id)

            # Validate and decode token
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                issuer=self.issuer,
                audience=valid_audiences if verify_audience else None,
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": verify_audience,
                },
            )

            logger.debug(f"Token validation successful for user: {claims.get('sub', 'unknown')}")

            # Extract and validate groups claim (must be list of strings)
            groups = claims.get("groups", [])
            if not isinstance(groups, list):
                groups = [groups] if groups else []
            if not all(isinstance(g, str) for g in groups):
                raise ValueError("Invalid groups claim format: must contain only strings")

            # Extract scopes - Okta uses 'scp' for scopes in access tokens
            scope_claim = claims.get("scp") or claims.get("scope", "")
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
                "client_id": claims.get("cid", self.client_id),
                "method": "okta",
                "data": claims,
            }

        except jwt.ExpiredSignatureError:
            logger.warning("Token validation failed: Token has expired")
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token validation failed: Invalid token - {e}")
            raise ValueError(f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"Okta token validation error: {e}")
            raise ValueError(f"Token validation failed: {e}")

    def _validate_self_signed_token(self, token: str) -> dict[str, Any]:
        """Validate a self-signed JWT token generated by our auth server.

        Self-signed tokens are generated for OAuth users to use for programmatic
        API access. They contain the user's identity, groups, and scopes.

        Args:
            token: The self-signed JWT token to validate

        Returns:
            Dictionary containing validation results with method="self_signed"

        Raises:
            ValueError: If token validation fails
        """
        try:
            claims = jwt.decode(
                token,
                SECRET_KEY,
                algorithms=["HS256"],
                audience=JWT_AUDIENCE,
                issuer=JWT_ISSUER,
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": True,
                },
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

    def get_jwks(self) -> dict[str, Any]:
        """Get JSON Web Key Set from Okta with caching.

        Returns cached JWKS if still valid (within TTL), otherwise fetches
        fresh data from Okta. Retries once on failure and falls back to
        stale cache if available.

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
        max_retries = 2
        last_error = None

        for attempt in range(max_retries):
            try:
                logger.debug(f"Fetching JWKS from {self.jwks_url} (attempt {attempt + 1})")
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
                    time.sleep(1)  # Brief delay before retry

        # Graceful degradation: use stale cache if available
        if self._jwks_cache:
            cache_age = current_time - self._jwks_cache_time
            logger.warning(
                f"JWKS fetch failed after {max_retries} attempts, "
                f"using stale cache (age: {cache_age:.0f}s): {last_error}"
            )
            return self._jwks_cache

        # No cache available, must fail
        logger.error(f"Failed to retrieve JWKS from Okta (no cache available): {last_error}")
        raise ValueError(f"Cannot retrieve JWKS: {last_error}")

    def exchange_code_for_token(self, code: str, redirect_uri: str) -> dict[str, Any]:
        """Exchange authorization code for access token.

        Args:
            code: Authorization code from Okta callback
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
        """Get user information from Okta.

        Args:
            access_token: Valid Okta access token

        Returns:
            User info dictionary from Okta userinfo endpoint

        Raises:
            ValueError: If the userinfo request fails
        """
        try:
            logger.debug("Fetching user info from Okta")
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
        """Get Okta authorization URL.

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
        logger.debug(f"Generated auth URL: {auth_url}")
        return auth_url

    def get_logout_url(self, redirect_uri: str) -> str:
        """Get Okta logout URL.

        Args:
            redirect_uri: URI to redirect to after logout

        Returns:
            Full logout URL with client_id and post_logout_redirect_uri params
        """
        logger.debug(f"Generating logout URL with redirect_uri: {redirect_uri}")

        params = {
            "client_id": self.client_id,
            "post_logout_redirect_uri": redirect_uri,
        }

        logout_url = f"{self.logout_url}?{urlencode(params)}"
        logger.debug(f"Generated logout URL: {logout_url}")

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
            client_secret: Optional override client secret (defaults to configured M2M client secret)
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

    def get_provider_info(self) -> dict[str, Any]:
        """Get provider-specific information.

        Returns:
            Dictionary containing provider configuration and endpoints
        """
        return {
            "provider_type": "okta",
            "okta_domain": self.okta_domain,
            "client_id": self.client_id,
            "endpoints": {
                "auth": self.auth_url,
                "token": self.token_url,
                "userinfo": self.userinfo_url,
                "jwks": self.jwks_url,
                "logout": self.logout_url,
            },
            "issuer": self.issuer,
        }
