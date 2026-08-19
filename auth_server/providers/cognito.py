"""AWS Cognito authentication provider implementation."""

import logging
from typing import Any
from urllib.parse import urlencode

import jwt
import requests

from .base import AuthProvider
from .jwks_cache import JwksCache

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


class CognitoProvider(AuthProvider):
    """AWS Cognito authentication provider implementation."""

    def __init__(
        self,
        user_pool_id: str,
        client_id: str,
        client_secret: str,
        region: str,
        domain: str | None = None,
        ide_oauth_client_id: str | None = None,
        m2m_client_ids: list[str] | None = None,
    ):
        """Initialize Cognito provider.

        Args:
            user_pool_id: AWS Cognito User Pool ID
            client_id: OAuth2 client ID (the web/confidential client)
            client_secret: OAuth2 client secret
            region: AWS region
            domain: Optional custom domain name
            ide_oauth_client_id: Optional public IDE client_id (IDE_OAUTH_CLIENT_ID).
                Cognito access tokens minted by this client are also accepted, so
                the IDE OAuth login flow works alongside the web client.
            m2m_client_ids: Optional allowlist of Cognito app-client ids that mint
                machine (``client_credentials``) access tokens the gateway should
                accept (COGNITO_M2M_CLIENT_IDS). Cognito access tokens are not
                audience-bound, so an M2M client's ``client_id`` claim must be
                explicitly listed here or its token is rejected. Config-driven and
                default-empty (fail closed): an unlisted client is never accepted,
                so a token from a rogue/other client in the same pool cannot reach
                the gateway. A machine token carries no ``cognito:groups`` /
                ``username``; its authorization comes from the token's own
                ``scope`` claim (Cognito resource-server scopes = registry scope
                names), exactly like every other caller's resolved scopes.

                The single value ``"*"`` is a wildcard scoped to M2M tokens ONLY:
                accept ANY client_id in this user pool, but only for machine
                tokens (no ``username`` claim). User/login tokens stay restricted
                to the web + IDE clients regardless, so the wildcard cannot be
                abused to forge a user login. Use ``"*"`` only when this pool is
                dedicated to the gateway (with Pattern B, one M2M client per
                agent, ``"*"`` avoids listing every agent's client_id); enumerate
                explicitly when the pool is shared with other applications.
        """
        self.user_pool_id = user_pool_id
        self.client_id = client_id
        self.client_secret = client_secret
        self.region = region
        self.domain = domain

        # Client ids whose Cognito ACCESS tokens are accepted. Access tokens are
        # not audience-bound (no "aud"), so we validate the "client_id" claim
        # against this allowlist instead. The web client plus the optional public
        # IDE client (IDE_OAUTH_CLIENT_ID) are trusted; both live in the same
        # user pool and the user's groups (which drive authorization) come from
        # the pool, not the client. M2M client ids (COGNITO_M2M_CLIENT_IDS) are
        # appended so machine client_credentials tokens validate; the list is
        # config-driven and default-empty (fail closed).
        self.accepted_client_ids = [client_id]
        if ide_oauth_client_id and ide_oauth_client_id != client_id:
            self.accepted_client_ids.append(ide_oauth_client_id)
        # "*" = accept ANY client_id, but only for M2M (no-username) tokens; see
        # docstring. Explicit ids are still appended for the non-wildcard case.
        requested_m2m = list(m2m_client_ids or [])
        self.m2m_accept_any = "*" in requested_m2m
        self.m2m_client_ids = [c for c in requested_m2m if c and c != "*"]
        for m2m_id in self.m2m_client_ids:
            if m2m_id not in self.accepted_client_ids:
                self.accepted_client_ids.append(m2m_id)
        if self.m2m_accept_any:
            logger.warning(
                "COGNITO_M2M_CLIENT_IDS='*': accepting M2M (client_credentials) "
                "tokens from ANY client in user pool %s. Safe only if this pool is "
                "dedicated to the gateway; user/login tokens remain restricted.",
                user_pool_id,
            )

        # Shared JWKS cache: TTL cache, bounded stale-fallback, and coalesced
        # + negative-cached unknown-kid refetch.
        self._jwks = JwksCache(provider_name="cognito")

        # Cognito endpoints
        if domain:
            self.cognito_domain = f"https://{domain}.auth.{region}.amazoncognito.com"
        else:
            user_pool_id_clean = user_pool_id.replace("_", "")
            self.cognito_domain = f"https://{user_pool_id_clean}.auth.{region}.amazoncognito.com"

        self.token_url = f"{self.cognito_domain}/oauth2/token"
        self.auth_url = f"{self.cognito_domain}/oauth2/authorize"
        self.userinfo_url = f"{self.cognito_domain}/oauth2/userInfo"
        self.jwks_url = (
            f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}/.well-known/jwks.json"
        )
        self.logout_url = f"{self.cognito_domain}/logout"
        self.issuer = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"

        logger.debug(
            f"Initialized Cognito provider for user pool '{user_pool_id}' in region '{region}'"
        )

    def validate_token(self, token: str, **kwargs: Any) -> dict[str, Any]:
        """Validate Cognito JWT token."""
        try:
            logger.debug("Validating Cognito JWT token")

            # Resolve the signing key via the shared, hardened JWKS cache
            # (TTL cache, bounded stale-fallback, coalesced + negative-cached
            # unknown-kid refetch). RS256 pinning + claim checks stay below.
            unverified_header = jwt.get_unverified_header(token)
            kid = unverified_header.get("kid")

            if not kid:
                raise ValueError("Token missing 'kid' in header")

            signing_key = self._jwks.get_signing_key(self.jwks_url, kid)

            # Cognito issues two token types and they carry the client binding
            # DIFFERENTLY:
            #   - id token:     has an "aud" claim equal to the client_id
            #   - access token: has NO "aud" claim; the client is in "client_id"
            #     (token_use == "access"). MCP clients send the ACCESS token to
            #     the resource server, so validating "aud" rejects every access
            #     token ("missing aud claim"). Branch on token_use and verify the
            #     right claim instead.
            unverified = jwt.decode(token, options={"verify_signature": False})
            is_access_token = unverified.get("token_use") == "access"

            if is_access_token:
                # Access token: no aud; verify signature/issuer/expiry, then check
                # the client_id claim against the allowlist (web + IDE clients).
                # Cognito access tokens are not audience-bound, so jwt can't do
                # this for us.
                claims = jwt.decode(
                    token,
                    signing_key,
                    algorithms=["RS256"],
                    issuer=self.issuer,
                    options={"verify_exp": True, "verify_iat": True, "verify_aud": False},
                )
                token_client_id = claims.get("client_id")
                # A machine (client_credentials) token has no end-user "username"
                # claim. The "*" wildcard accepts any client_id but ONLY for such
                # machine tokens, so it can never widen which clients may mint a
                # USER/login token (those stay restricted to web + IDE clients).
                is_machine_token = "username" not in claims
                m2m_wildcard_ok = self.m2m_accept_any and is_machine_token
                if token_client_id not in self.accepted_client_ids and not m2m_wildcard_ok:
                    raise ValueError(
                        f"Access token client_id '{token_client_id}' is not in the "
                        f"accepted client list {self.accepted_client_ids}"
                        + (
                            " (M2M wildcard applies to no-username tokens only)"
                            if self.m2m_accept_any
                            else ""
                        )
                    )
            else:
                # ID token: audience-bound to the client_id.
                claims = jwt.decode(
                    token,
                    signing_key,
                    algorithms=["RS256"],
                    issuer=self.issuer,
                    audience=self.client_id,
                    options={"verify_exp": True, "verify_iat": True, "verify_aud": True},
                )

            logger.debug(
                f"Token validation successful for user: {claims.get('username', 'unknown')}"
            )

            # Extract user info from claims
            return {
                "valid": True,
                "username": claims.get("username", claims.get("sub")),
                "email": claims.get("email"),
                "groups": claims.get("cognito:groups", []),
                "scopes": claims.get("scope", "").split() if claims.get("scope") else [],
                "client_id": claims.get("client_id", self.client_id),
                "method": "cognito",
                "data": claims,
            }

        except jwt.ExpiredSignatureError:
            logger.warning("Token validation failed: Token has expired")
            raise ValueError("Token has expired")
        except jwt.InvalidTokenError as e:
            logger.warning(f"Token validation failed: Invalid token - {e}")
            raise ValueError(f"Invalid token: {e}")
        except Exception as e:
            logger.error(f"Cognito token validation error: {e}")
            raise ValueError(f"Token validation failed: {e}")

    def get_jwks(self) -> dict[str, Any]:
        """Get JSON Web Key Set from Cognito via the shared JWKS cache."""
        return self._jwks.get_jwks(self.jwks_url)

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
        """Get user information from Cognito."""
        try:
            logger.debug("Fetching user info from Cognito")

            headers = {"Authorization": f"Bearer {access_token}"}
            response = requests.get(self.userinfo_url, headers=headers, timeout=10)
            response.raise_for_status()

            user_info = response.json()
            logger.debug(f"User info retrieved for: {user_info.get('username', 'unknown')}")

            return user_info

        except requests.RequestException as e:
            logger.error(f"Failed to get user info: {e}")
            raise ValueError(f"User info retrieval failed: {e}")

    def get_auth_url(self, redirect_uri: str, state: str, scope: str | None = None) -> str:
        """Get Cognito authorization URL."""
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
        """Get Cognito logout URL."""
        logger.debug(f"Generating logout URL with redirect_uri: {redirect_uri}")

        params = {"client_id": self.client_id, "logout_uri": redirect_uri}

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
        """Validate a machine-to-machine token."""
        # M2M tokens use the same validation as regular tokens in Cognito
        return self.validate_token(token)

    def get_m2m_token(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Get machine-to-machine token using client credentials."""
        try:
            logger.debug("Requesting M2M token using client credentials")

            data = {
                "grant_type": "client_credentials",
                "client_id": client_id or self.client_id,
                "client_secret": client_secret or self.client_secret,
            }

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

    def authorization_server_metadata(self) -> dict[str, Any]:
        """Return Cognito's RFC 8414 metadata.

        Cognito splits its OAuth surface across two hosts: the issuer lives on
        `cognito-idp.{region}.amazonaws.com/{userPoolId}` (where the JWKS is
        served) but the `/authorize`, `/token`, `/userInfo`, and `/logout`
        endpoints live on the `cognito-domain` host. We build the RFC 8414
        document directly from values the provider already holds, which
        rehomes those endpoints onto the canonical RFC 8414 path
        `/.well-known/oauth-authorization-server` from a discovery client's
        perspective.
        """
        return {
            "issuer": self.issuer,
            "authorization_endpoint": self.auth_url,
            "token_endpoint": self.token_url,
            "userinfo_endpoint": self.userinfo_url,
            "jwks_uri": self.jwks_url,
            "end_session_endpoint": self.logout_url,
            "response_types_supported": ["code"],
            "grant_types_supported": [
                "authorization_code",
                "refresh_token",
                "client_credentials",
            ],
            "token_endpoint_auth_methods_supported": [
                "client_secret_basic",
                "client_secret_post",
            ],
            "code_challenge_methods_supported": ["S256"],
            "subject_types_supported": ["public"],
            "id_token_signing_alg_values_supported": ["RS256"],
            "scopes_supported": ["openid", "email", "profile"],
        }

    def get_provider_info(self) -> dict[str, Any]:
        """Get provider-specific information."""
        return {
            "provider_type": "cognito",
            "user_pool_id": self.user_pool_id,
            "region": self.region,
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
