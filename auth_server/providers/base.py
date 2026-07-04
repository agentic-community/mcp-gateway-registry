"""Base authentication provider interface."""

import logging
from abc import ABC, abstractmethod
from typing import Any

import jwt

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


class IdTokenVerificationError(Exception):
    """Raised when an OIDC id_token cannot be cryptographically verified.

    Signals that a present id_token failed signature, issuer, audience, or
    expiry verification. Callers MUST fail closed on this error (deny the
    login) rather than fall back to an unverified claim source, because a
    verification failure indicates the token may have been forged or tampered.
    """


class AuthProvider(ABC):
    """Abstract base class for authentication providers."""

    def _verify_id_token_with_jwks(
        self,
        id_token: str,
        valid_issuers: list[str],
        accepted_audiences: list[str],
        leeway_seconds: int = 0,
        expected_nonce: str | None = None,
    ) -> dict[str, Any]:
        """Cryptographically verify an OIDC id_token against the provider JWKS.

        This is the single shared verification path for id_tokens received from
        an OAuth2 token endpoint. It verifies the RS256 signature against the
        provider's published JWKS, and enforces issuer, audience, and expiry
        BEFORE any claim is returned. It fails closed: any missing key,
        unknown issuer, bad audience, expired token, or malformed input raises
        ``IdTokenVerificationError`` and no claims are returned.

        When ``expected_nonce`` is supplied, the token's ``nonce`` claim is
        enforced AFTER signature verification: the claim must be present and
        equal the value bound to this login. This binds a (validly signed)
        id_token to the specific authorization request it was issued for,
        defeating replay/injection of a token minted for a different login.

        Args:
            id_token: The raw compact-serialized JWT id_token string.
            valid_issuers: Issuers the token's ``iss`` claim may match. The
                token issuer is matched against this allowlist before signature
                verification so PyJWT validates against the exact expected value.
            accepted_audiences: Audiences the token's ``aud`` claim may match.
                For id_tokens this is normally the OAuth2 client_id only.
            leeway_seconds: Optional clock-skew leeway for expiry checks.
            expected_nonce: The nonce bound to this login (persisted in the
                OAuth2 flow cookie). When not ``None`` the verified token's
                ``nonce`` claim MUST match it exactly; a missing or mismatched
                nonce fails closed. ``None`` skips the nonce check (used only
                where no nonce was issued, e.g. a legacy in-flight login).

        Returns:
            The verified claim set as a dict. Safe to trust for identity and
            authorization decisions.

        Raises:
            IdTokenVerificationError: If the token is absent, malformed, signed
                by an unknown key, from an unexpected issuer/audience, expired,
                nonce-mismatched, or otherwise fails verification.
        """
        if not id_token or not isinstance(id_token, str):
            raise IdTokenVerificationError("id_token is missing or not a string")

        if not valid_issuers:
            raise IdTokenVerificationError("No valid issuers configured for id_token verification")

        if not accepted_audiences:
            raise IdTokenVerificationError(
                "No accepted audiences configured for id_token verification"
            )

        try:
            unverified_header = jwt.get_unverified_header(id_token)
        except jwt.InvalidTokenError as e:
            raise IdTokenVerificationError(f"Malformed id_token header: {e}") from e

        kid = unverified_header.get("kid")
        if not kid:
            raise IdTokenVerificationError("id_token header missing 'kid'")

        alg = unverified_header.get("alg")
        if alg != "RS256":
            # OIDC id_tokens from the supported IdPs are RS256. Reject anything
            # else (notably 'none' and HS256, which would let an attacker sign
            # with a public value) instead of trusting the header's algorithm.
            raise IdTokenVerificationError(f"Unsupported id_token signing algorithm: {alg}")

        # Determine the expected issuer without trusting a signature yet: read
        # the unverified 'iss' only to pick which allowlisted issuer PyJWT will
        # enforce. The subsequent jwt.decode still verifies the signature and
        # re-checks iss against this exact value, so a spoofed iss cannot pass.
        try:
            unverified_claims = jwt.decode(id_token, options={"verify_signature": False})
        except jwt.InvalidTokenError as e:
            raise IdTokenVerificationError(f"Malformed id_token payload: {e}") from e

        token_issuer = unverified_claims.get("iss")
        if token_issuer not in valid_issuers:
            raise IdTokenVerificationError(
                f"id_token issuer '{token_issuer}' is not in the expected issuer allowlist"
            )

        try:
            jwks = self.get_jwks()
        except Exception as e:
            # Fail closed: if we cannot reach the JWKS, we cannot verify.
            raise IdTokenVerificationError(f"Unable to retrieve JWKS for verification: {e}") from e

        signing_key = None
        for key in jwks.get("keys", []):
            if key.get("kid") == kid:
                from jwt import PyJWK

                signing_key = PyJWK(key).key
                break

        if signing_key is None:
            raise IdTokenVerificationError(f"No JWKS key matches id_token 'kid': {kid}")

        try:
            claims: dict[str, Any] = jwt.decode(
                id_token,
                signing_key,
                algorithms=["RS256"],
                issuer=token_issuer,
                audience=accepted_audiences,
                leeway=leeway_seconds,
                options={
                    "verify_signature": True,
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_aud": True,
                    "require": ["exp", "iss", "aud"],
                },
            )
        except jwt.InvalidTokenError as e:
            raise IdTokenVerificationError(f"id_token verification failed: {e}") from e

        # Bind the (now cryptographically verified) token to THIS login. Done
        # after signature verification so a forged nonce on an unsigned token
        # can never reach this check. A signed token whose nonce does not match
        # the one bound to this authorization request is a replay/injection.
        if expected_nonce is not None:
            token_nonce = claims.get("nonce")
            if not token_nonce or token_nonce != expected_nonce:
                raise IdTokenVerificationError(
                    "id_token nonce does not match the value bound to this login"
                )

        return claims

    def validate_id_token(
        self,
        id_token: str,
        expected_nonce: str | None = None,
    ) -> dict[str, Any]:
        """Verify an OIDC id_token and return its verified claims.

        Providers that issue an id_token during the authorization-code flow MUST
        override this to verify the token's signature (against their JWKS),
        issuer, audience (the gateway's client_id), and expiry before returning
        any claim. The default implementation refuses to trust an id_token,
        failing closed for providers that have not opted in.

        Args:
            id_token: The raw id_token string from the token endpoint.
            expected_nonce: The nonce bound to this login. When not ``None`` the
                verified token's ``nonce`` claim must match it exactly (replay
                protection). Overriding implementations MUST forward this to
                ``_verify_id_token_with_jwks``.

        Returns:
            The verified claim set.

        Raises:
            IdTokenVerificationError: Always, unless a subclass overrides this
                with a real JWKS-backed verification.
        """
        raise IdTokenVerificationError(
            f"{type(self).__name__} does not support verified id_token extraction"
        )

    @abstractmethod
    def validate_token(self, token: str, **kwargs: Any) -> dict[str, Any]:
        """Validate an access token and return user info.

        Args:
            token: The access token to validate
            **kwargs: Additional provider-specific arguments

        Returns:
            Dictionary containing:
                - valid: Boolean indicating if token is valid
                - username: User's username
                - email: User's email address
                - groups: List of group memberships
                - scopes: List of token scopes
                - client_id: Client ID that issued the token
                - method: Authentication method used
                - data: Raw token claims/data

        Raises:
            ValueError: If token validation fails
        """
        pass

    @abstractmethod
    def get_jwks(self) -> dict[str, Any]:
        """Get JSON Web Key Set for token validation.

        Returns:
            Dictionary containing the JWKS data

        Raises:
            ValueError: If JWKS cannot be retrieved
        """
        pass

    @abstractmethod
    def exchange_code_for_token(self, code: str, redirect_uri: str) -> dict[str, Any]:
        """Exchange authorization code for access token.

        Args:
            code: Authorization code from OAuth2 flow
            redirect_uri: Redirect URI used in the authorization request

        Returns:
            Dictionary containing token response:
                - access_token: The access token
                - id_token: The ID token (if available)
                - refresh_token: The refresh token (if available)
                - token_type: Type of token (usually "Bearer")
                - expires_in: Token expiration time in seconds

        Raises:
            ValueError: If code exchange fails
        """
        pass

    @abstractmethod
    def get_user_info(self, access_token: str) -> dict[str, Any]:
        """Get user information from access token.

        Args:
            access_token: Valid access token

        Returns:
            Dictionary containing user information:
                - username: User's username
                - email: User's email
                - groups: User's group memberships
                - Additional provider-specific fields

        Raises:
            ValueError: If user info cannot be retrieved
        """
        pass

    @abstractmethod
    def get_auth_url(self, redirect_uri: str, state: str, scope: str | None = None) -> str:
        """Get authorization URL for OAuth2 flow.

        Args:
            redirect_uri: URI to redirect to after authorization
            state: State parameter for CSRF protection
            scope: Optional scope parameter (defaults to provider's default)

        Returns:
            Full authorization URL
        """
        pass

    @abstractmethod
    def get_logout_url(self, redirect_uri: str) -> str:
        """Get logout URL.

        Args:
            redirect_uri: URI to redirect to after logout

        Returns:
            Full logout URL
        """
        pass

    @abstractmethod
    def refresh_token(self, refresh_token: str) -> dict[str, Any]:
        """Refresh an access token using a refresh token.

        Args:
            refresh_token: The refresh token

        Returns:
            Dictionary containing new token response

        Raises:
            ValueError: If token refresh fails
        """
        pass

    @abstractmethod
    def validate_m2m_token(self, token: str) -> dict[str, Any]:
        """Validate a machine-to-machine token.

        Args:
            token: The M2M access token to validate

        Returns:
            Dictionary containing validation result

        Raises:
            ValueError: If token validation fails
        """
        pass

    @abstractmethod
    def get_m2m_token(
        self,
        client_id: str | None = None,
        client_secret: str | None = None,
        scope: str | None = None,
    ) -> dict[str, Any]:
        """Get a machine-to-machine token using client credentials.

        Args:
            client_id: Optional client ID (uses default if not provided)
            client_secret: Optional client secret (uses default if not provided)
            scope: Optional scope for the token

        Returns:
            Dictionary containing token response

        Raises:
            ValueError: If token generation fails
        """
        pass

    @abstractmethod
    def authorization_server_metadata(self) -> dict[str, Any]:
        """Return the IdP's RFC 8414 Authorization Server Metadata document.

        Implementations should fetch the upstream OIDC/OAuth metadata, normalize
        any provider-specific quirks (e.g. rehoming Cognito's split endpoints onto
        the cognito-domain host), and return a stable RFC 8414-shaped dict.

        Returns:
            Dictionary conforming to RFC 8414 (issuer, authorization_endpoint,
            token_endpoint, jwks_uri, response_types_supported, ...).

        Raises:
            ValueError: If upstream metadata cannot be fetched or normalized.
        """
        pass

    def authorization_server_issuer(self) -> str:
        """Return the canonical issuer URL of the configured authorization server.

        Used as the `authorization_servers` entry in the gateway's RFC 9728
        Protected Resource Metadata document. Default implementation reads the
        `issuer` field from authorization_server_metadata(); providers can
        override for cheaper lookups.
        """
        metadata = self.authorization_server_metadata()
        issuer = metadata.get("issuer")
        if not issuer:
            raise ValueError("Authorization server metadata missing 'issuer' field")
        return issuer

    def protected_resource_metadata(
        self,
        resource: str,
        scopes_supported: list[str],
        resource_documentation: str | None = None,
    ) -> dict[str, Any]:
        """Build an RFC 9728 Protected Resource Metadata document for this gateway.

        The shape is identical across IdPs; the only per-provider input is the
        authorization server issuer (filled in via authorization_server_issuer()).
        Subclasses generally do not need to override this.

        Args:
            resource: Canonical URL of the MCP resource server (gateway).
                Must be HTTPS in non-local environments and have no trailing slash.
            scopes_supported: List of scope strings the gateway recognizes.
                Caller is responsible for sorting/dedup; this method preserves order.
            resource_documentation: Optional URL to operator-facing OAuth docs.

        Returns:
            Dictionary conforming to RFC 9728 §3.
        """
        document: dict[str, Any] = {
            "resource": resource,
            "authorization_servers": [self.authorization_server_issuer()],
            "scopes_supported": list(scopes_supported),
            "bearer_methods_supported": ["header"],
        }
        if resource_documentation:
            document["resource_documentation"] = resource_documentation
        return document
