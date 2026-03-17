"""Auth0 authentication provider implementation."""

import logging
import os
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


class Auth0Provider(AuthProvider):
    """Auth0 authentication provider implementation.

    This provider implements OAuth2/OIDC authentication using Auth0.
    It supports:
    - User authentication via OAuth2 authorization code flow
    - Machine-to-machine authentication via client credentials flow
    - JWT token validation using Auth0 JWKS
    - Group-based authorization via custom claims or Auth0 Organizations
    """

    def __init__(
        self,
        domain: str,
        client_id: str,
        client_secret: str,
        audience: str | None = None,
        m2m_client_id: str | None = None,
        m2m_client_secret: str | None = None,
        groups_claim: str = "https://mcp-gateway/groups",
    ):
        """Initialize Auth0 provider.

        Args:
            domain: Auth0 domain (e.g., 'your-tenant.auth0.com')
            client_id: OAuth2 client ID for web authentication
            client_secret: OAuth2 client secret for web authentication
            audience: API audience identifier for access tokens
            m2m_client_id: Optional M2M client ID (defaults to client_id)
            m2m_client_secret: Optional M2M client secret (defaults to client_secret)
            groups_claim: Custom claim name for groups in the ID/access token.
                Auth0 requires a namespaced claim via a Rule/Action
                (e.g., 'https://mcp-gateway/groups'). Defaults to
                'https://mcp-gateway/groups'.
        """
        self.domain = domain.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.audience = audience
        self.m2m_client_id = m2m_client_id or client_id
        self.m2m_client_secret = m2m_client_secret or client_secret
        self.groups_claim = groups_claim

        # JWKS cache
        self._jwks_cache: dict[str, Any] | None = None
        self._jwks_cache_time: float = 0
        self._jwks_cache_ttl: int = 3600  # 1 hour

        # Auth0 endpoints
        base_url = f"https://{self.domain}"
        self.auth_url = f"{base_url}/authorize"
        self.token_url = f"{base_url}/oauth/token"
        self.userinfo_url = f"{base_url}/userinfo"
        self.jwks_url = f"{base_url}/.well-known/jwks.json"
        self.logout_url = f"{base_url}/v2/logout"
        self.issuer = f"{base_url}/"

        logger.debug(f"Initialized Auth0 provider for domain '{domain}'")

    def validate_token(self, token: str, **kwargs: Any) -> dict[str, Any]:
        """Validate Auth0 JWT token.

        Args:
            token: The JWT access token to validate
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary containing:
                - valid: True if token is valid
                - username: User's sub or nickname claim
                - email: User's email address
                - groups: List of group memberships from custom claim
                - scopes: List of token scopes
                - client_id: Client ID that issued the token
                - method: 'auth0'
                - data: Raw token claims

        Raises:
            ValueError: If token validation fails
        """
        try:
            logger.debug("Validating Auth0 JWT token")

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

            # Build audience list for validation
            valid_audiences = [self.client_id]
            if self.audience:
                valid_audiences.append(self.audience)

            # Validate and decode token
            claims = jwt.decode(
                token,
                signing_key,
                algorithms=["RS256"],
                issuer=self.issuer,
                audience=valid_audiences,
                options={"verify_exp": True, "verify_iat": True, "verify_aud": True},
            )

            logger.debug(
                f"Token validation successful for user: "
                f"{claims.get('nickname', claims.get('sub', 'unknown'))}"
            )

            # Extract groups from custom namespaced claim
            groups = claims.get(self.groups_claim, [])
            if not groups:
                # Fallback: check permissions claim (Auth0 RBAC)
                groups = claims.get("permissions", [])

            return {
                "valid": True,
                "username": claims.get("nickname", claims.get("sub")),
                "email": claims.get("email"),
                "groups": groups,
                "scopes": claims.get("scope", "").split() if claims.get("scope") else [],
                "client_id": claims.get("azp", self.client_id),
                "method": "auth0",
                "data": claims,
            }

        except jwt.ExpiredSignatureError as e:
            logger.warning("Token validation failed: Token has expired")
            raise ValueError("Token has expired") from e
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token validation failed: Invalid token - {e}")
            raise ValueError(f"Invalid token: {e}") from e
        except Exception as e:
            logger.error(f"Auth0 token validation error: {e}")
            raise ValueError(f"Token validation failed: {e}") from e

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

        except jwt.ExpiredSignatureError as e:
            logger.warning("Self-signed token validation failed: Token has expired")
            raise ValueError("Token has expired") from e
        except jwt.InvalidTokenError as e:
            logger.warning(f"Self-signed token validation failed: {e}")
            raise ValueError(f"Invalid self-signed token: {e}") from e
        except Exception as e:
            logger.error(f"Self-signed token validation error: {e}")
            raise ValueError(f"Self-signed token validation failed: {e}") from e

    def get_jwks(self) -> dict[str, Any]:
        """Get JSON Web Key Set from Auth0 with caching.

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
            logger.error(f"Failed to retrieve JWKS from Auth0: {e}")
            raise ValueError(f"Cannot retrieve JWKS: {e}")

    def exchange_code_for_token(self, code: str, redirect_uri: str) -> dict[str, Any]:
        """Exchange authorization code for access token.

        Args:
            code: Authorization code from OAuth2 flow
            redirect_uri: Redirect URI used in the authorization request

        Returns:
            Dictionary containing token response

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

    def get_user_info(self, access_token: str) -> dict[str, Any]:
        """Get user information from Auth0.

        Args:
            access_token: Valid access token

        Returns:
            Dictionary containing user information

        Raises:
            ValueError: If user info cannot be retrieved
        """
        try:
            logger.debug("Fetching user info from Auth0")

            headers = {"Authorization": f"Bearer {access_token}"}
            response = requests.get(self.userinfo_url, headers=headers, timeout=10)
            response.raise_for_status()

            user_info = response.json()
            logger.debug(
                f"User info retrieved for: {user_info.get('nickname', 'unknown')}"
            )

            return user_info

        except requests.RequestException as e:
            logger.error(f"Failed to get user info: {e}")
            raise ValueError(f"User info retrieval failed: {e}")

    def get_auth_url(self, redirect_uri: str, state: str, scope: str | None = None) -> str:
        """Get Auth0 authorization URL.

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

        # Include audience if configured (required for API access tokens)
        if self.audience:
            params["audience"] = self.audience

        auth_url = f"{self.auth_url}?{urlencode(params)}"
        logger.debug(f"Generated auth URL: {auth_url}")

        return auth_url

    def get_logout_url(self, redirect_uri: str) -> str:
        """Get Auth0 logout URL.

        Auth0 uses 'returnTo' parameter and requires client_id.

        Args:
            redirect_uri: URI to redirect to after logout

        Returns:
            Full logout URL
        """
        logger.debug(f"Generating logout URL with redirect_uri: {redirect_uri}")

        params = {"client_id": self.client_id, "returnTo": redirect_uri}

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

        Auth0 M2M tokens require an audience parameter to specify which API
        the token is intended for.

        Args:
            client_id: Optional client ID (uses M2M default if not provided)
            client_secret: Optional client secret (uses M2M default if not provided)
            scope: Optional scope for the token

        Returns:
            Dictionary containing token response

        Raises:
            ValueError: If token generation fails
        """
        try:
            logger.debug("Requesting M2M token using client credentials")

            data: dict[str, str] = {
                "grant_type": "client_credentials",
                "client_id": client_id or self.m2m_client_id,
                "client_secret": client_secret or self.m2m_client_secret,
            }

            # Auth0 requires audience for M2M tokens
            if self.audience:
                data["audience"] = self.audience

            if scope:
                data["scope"] = scope

            headers = {"Content-Type": "application/x-www-form-urlencoded"}

            response = requests.post(self.token_url, data=data, headers=headers, timeout=10)
            response.raise_for_status()

            token_data = response.json()
            logger.debug("M2M token generation successful")

            return token_data

        except requests.RequestException as e:
            logger.error(f"Failed to get M2M token: {e}")
            raise ValueError(f"M2M token generation failed: {e}")

    def extract_user_from_tokens(self, token_data: dict[str, Any]) -> dict[str, Any]:
        """Extract user information from Auth0 token response.

        Parses the ID token from the OAuth2 token exchange response to extract
        user identity and group memberships. The ID token is validated for
        issuer and audience claims to prevent token forgery.

        Groups are extracted from a custom namespaced claim (e.g.,
        'https://mcp-gateway/groups') which must be configured via an
        Auth0 Action or Rule. If no groups are found, falls back to the
        'permissions' claim from Auth0 RBAC.

        Args:
            token_data: Token response from Auth0 containing 'id_token'
                and 'access_token' keys

        Returns:
            Dictionary containing:
                - username: User's nickname, email, or sub claim
                - email: User's email address
                - name: User's display name
                - groups: List of group memberships

        Raises:
            ValueError: If ID token is missing or cannot be parsed
        """
        if "id_token" not in token_data:
            raise ValueError("Missing ID token in Auth0 response")

        try:
            # Validate issuer and audience claims on the ID token.
            # Signature verification is skipped because this token was received
            # directly from Auth0's token endpoint over TLS (OIDC Core 3.1.3.7).
            id_token_claims = jwt.decode(
                token_data["id_token"],
                options={
                    "verify_signature": False,
                    "verify_iss": True,
                    "verify_aud": True,
                    "verify_exp": True,
                },
                issuer=self.issuer,
                audience=self.client_id,
            )
            logger.info(f"Auth0 ID token claims decoded for sub: {id_token_claims.get('sub')}")

            # Extract groups from custom namespaced claim.
            # Requires an Auth0 Action or Rule to add groups to the ID token.
            # Example Action: api.idToken.setCustomClaim("https://mcp-gateway/groups", event.user.groups)
            groups = id_token_claims.get(self.groups_claim, [])
            if not groups:
                # Fallback: check permissions claim (Auth0 RBAC)
                groups = id_token_claims.get("permissions", [])

            return {
                "username": id_token_claims.get("nickname")
                or id_token_claims.get("email")
                or id_token_claims.get("sub"),
                "email": id_token_claims.get("email"),
                "name": id_token_claims.get("name")
                or id_token_claims.get("given_name"),
                "groups": groups,
            }

        except jwt.InvalidTokenError as e:
            logger.warning(f"Auth0 ID token parsing failed: {e}")
            raise ValueError(f"Failed to parse Auth0 ID token: {e}") from e

    def get_provider_info(self) -> dict[str, Any]:
        """Get provider-specific information.

        Returns:
            Dictionary containing provider configuration and endpoints
        """
        return {
            "provider_type": "auth0",
            "domain": self.domain,
            "client_id": self.client_id,
            "audience": self.audience,
            "endpoints": {
                "auth": self.auth_url,
                "token": self.token_url,
                "userinfo": self.userinfo_url,
                "jwks": self.jwks_url,
                "logout": self.logout_url,
            },
            "issuer": self.issuer,
        }
