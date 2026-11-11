"""Base authentication provider interface."""

import copy
import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, ClassVar

import jwt
import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# A token whose `kid` is not in the cached JWKS triggers at most one refetch per
# JWKS URL in this window. That picks up an IdP key rotation within seconds
# while stopping a stream of tokens with made-up `kid` values from turning every
# request into an outbound fetch.
JWKS_KID_MISS_REFRESH_INTERVAL_SECONDS: int = 60


@dataclass
class _SharedJwksEntry:
    """JWKS state for one JWKS URL, shared by every provider instance using it."""

    jwks: dict[str, Any] | None = None
    fetched_at: float = 0.0
    last_kid_miss_refresh: float = 0.0


def _match_jwks_key(
    jwks: dict[str, Any],
    kid: str,
) -> Any | None:
    """Return the public key for `kid` from a JWKS document, or None."""
    for key in jwks.get("keys", []):
        if key.get("kid") == kid:
            return jwt.PyJWK(key).key
    return None


class IdTokenVerificationError(Exception):
    """Raised when an OIDC id_token cannot be cryptographically verified.

    Signals that a present id_token failed signature, issuer, audience, or
    expiry verification. Callers MUST fail closed on this error (deny the
    login) rather than fall back to an unverified claim source, because a
    verification failure indicates the token may have been forged or tampered.
    """


def normalize_groups_claim(value: Any, *, source: str) -> list[str]:
    """Validate a token's ``groups`` claim as a JSON array of strings.

    Fail closed: accept ONLY a list whose every element is a string. An
    absent or null claim is the normal machine-to-machine case and maps to an
    empty list.

    Args:
        value: The raw claim value decoded from the already signature-verified
            JWT (or an equivalent verified userinfo/id-token field).
        source: Short label naming the claim/provider for the rejection log.

    Returns:
        The validated list of group names (possibly empty).

    Raises:
        ValueError: When the claim is present but is not a JSON array of
            strings.
    """
    if value is None:
        return []
    if not isinstance(value, list) or not all(isinstance(group, str) for group in value):
        logger.warning(
            "Rejecting token: %s claim is not a JSON array of strings (got %s)",
            source,
            type(value).__name__,
        )
        raise ValueError(f"Invalid {source} claim: expected a JSON array of strings")
    return value


class AuthProvider(ABC):
    """Abstract base class for authentication providers.

    The JWKS cache lives on the class, keyed by ``jwks_url``, not on the
    instance. ``get_auth_provider()`` builds a new provider per request, so a
    per-instance cache was always empty and every token validation refetched
    the JWKS (issue #2012). Subclasses keep reading and writing
    ``self._jwks_cache`` / ``self._jwks_cache_time``; the properties below route
    those to the shared entry for the provider's ``jwks_url``.
    """

    _shared_jwks: ClassVar[dict[str, _SharedJwksEntry]] = {}
    _shared_jwks_lock: ClassVar[threading.Lock] = threading.Lock()
    _shared_discovery: ClassVar[dict[str, dict[str, Any]]] = {}
    _shared_discovery_lock: ClassVar[threading.Lock] = threading.Lock()

    @classmethod
    def _clear_shared_discovery_cache(cls) -> None:
        """Drop every cached discovery document (test helper and operational reset)."""
        with cls._shared_discovery_lock:
            AuthProvider._shared_discovery.clear()

    def _fetch_shared_openid_configuration(
        self,
        config_url: str,
    ) -> dict[str, Any]:
        """Fetch an OpenID Connect discovery document, cached for the process.

        The cache is on the class, keyed by `config_url`. ``get_auth_provider()``
        builds a new provider per request, so a cache on the instance (an
        ``@lru_cache`` on a method, keyed on ``self``) refetched discovery on
        every request. A failed fetch is not cached. Each caller gets its own
        deep copy, so a caller that edits the result cannot change it for
        later requests.

        Args:
            config_url: The provider's ``/.well-known/openid-configuration`` URL.

        Returns:
            A private copy of the discovery document.

        Raises:
            ValueError: If the discovery document cannot be retrieved.
        """
        with self._shared_discovery_lock:
            cached = AuthProvider._shared_discovery.get(config_url)
        if cached is not None:
            return copy.deepcopy(cached)

        try:
            logger.debug(f"Fetching OpenID configuration from {config_url}")
            response = requests.get(config_url, timeout=10)
            response.raise_for_status()
            config: dict[str, Any] = response.json()
        except requests.RequestException as e:
            logger.error(f"Failed to get OpenID configuration: {e}")
            raise ValueError(f"OpenID configuration retrieval failed: {e}") from e

        logger.debug("OpenID configuration retrieved successfully")
        with self._shared_discovery_lock:
            AuthProvider._shared_discovery[config_url] = config
        return copy.deepcopy(config)

    @classmethod
    def _clear_shared_jwks_cache(cls) -> None:
        """Drop every shared JWKS entry (test helper and operational reset)."""
        with cls._shared_jwks_lock:
            AuthProvider._shared_jwks.clear()

    def _jwks_cache_key(self) -> str:
        """Key for this provider's shared JWKS entry; the JWKS URL by default.

        Override when reading ``jwks_url`` is itself expensive (for example,
        when it comes from an OIDC discovery fetch).
        """
        return str(self.jwks_url)  # type: ignore[attr-defined]

    def _shared_jwks_entry(self) -> _SharedJwksEntry:
        """Return the shared JWKS entry for this provider's cache key."""
        key = self._jwks_cache_key()
        with self._shared_jwks_lock:
            entry = AuthProvider._shared_jwks.get(key)
            if entry is None:
                entry = _SharedJwksEntry()
                AuthProvider._shared_jwks[key] = entry
            return entry

    @property
    def _jwks_cache(self) -> dict[str, Any] | None:
        return self._shared_jwks_entry().jwks

    @_jwks_cache.setter
    def _jwks_cache(self, value: dict[str, Any] | None) -> None:
        self._shared_jwks_entry().jwks = value

    @property
    def _jwks_cache_time(self) -> float:
        return self._shared_jwks_entry().fetched_at

    @_jwks_cache_time.setter
    def _jwks_cache_time(self, value: float) -> None:
        self._shared_jwks_entry().fetched_at = value

    def _claim_kid_miss_refresh(self) -> bool:
        """Reserve this URL's kid-miss refresh slot; False if used within the window."""
        entry = self._shared_jwks_entry()
        now = time.time()
        with self._shared_jwks_lock:
            if now - entry.last_kid_miss_refresh < JWKS_KID_MISS_REFRESH_INTERVAL_SECONDS:
                return False
            entry.last_kid_miss_refresh = now
            return True

    def _refetch_jwks_for_kid(
        self,
        kid: str,
    ) -> Any | None:
        """Force one JWKS refetch and look up `kid` again.

        Expires the shared entry so ``get_jwks()`` fetches, then restores the
        previous fetch time if the fetch did not store a new document. Without
        that restore, a failed refetch would leave the entry expired and every
        later request would hit the IdP.
        """
        previous_fetch_time = self._jwks_cache_time
        self._jwks_cache_time = 0.0
        try:
            jwks = self.get_jwks()
        except ValueError as e:
            logger.warning(f"JWKS refetch for unknown kid failed: {e}")
            return None
        finally:
            if self._jwks_cache_time == 0.0:
                self._jwks_cache_time = previous_fetch_time
        return _match_jwks_key(jwks, kid)

    def _find_jwks_signing_key(
        self,
        kid: str,
    ) -> Any | None:
        """Return the JWKS public key for `kid`, refetching once on a miss.

        A miss usually means the IdP rotated its signing keys since the JWKS was
        cached. The refetch is rate-limited per JWKS URL by
        ``JWKS_KID_MISS_REFRESH_INTERVAL_SECONDS``. A key that is still absent
        returns None, and the caller rejects the token.

        Args:
            kid: The ``kid`` from the token's (unverified) header.

        Returns:
            The public key object for PyJWT, or None when no key matches.

        Raises:
            ValueError: If the initial JWKS retrieval fails.
        """
        signing_key = _match_jwks_key(self.get_jwks(), kid)
        if signing_key is not None:
            return signing_key

        if not self._claim_kid_miss_refresh():
            logger.warning(f"No JWKS key for kid {kid!r}; refetch is rate-limited, rejecting")
            return None

        logger.info(f"No JWKS key for kid {kid!r}; refetching JWKS once")
        return self._refetch_jwks_for_kid(kid)

    def _verify_id_token_with_jwks(
        self,
        id_token: str,
        valid_issuers: list[str],
        accepted_audiences: list[str],
        leeway_seconds: int = 0,
        expected_nonce: str | None = None,
        algorithms: list[str] | None = None,
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
            signing_key = self._find_jwks_signing_key(kid)
        except Exception as e:
            # Fail closed: if we cannot reach the JWKS, we cannot verify.
            raise IdTokenVerificationError(f"Unable to retrieve JWKS for verification: {e}") from e

        if signing_key is None:
            raise IdTokenVerificationError(f"No JWKS key matches id_token 'kid': {kid}")

        try:
            claims: dict[str, Any] = jwt.decode(
                id_token,
                signing_key,
                algorithms=algorithms or ["RS256"],
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

    async def recover_bearer_groups(
        self,
        validation_result: dict[str, Any],
        access_token: str,
    ) -> list[str] | None:
        """Return recovered groups, or None when this provider uses inline claims.

        A confirmed empty membership is [], not None. Providers that need an
        external lookup override this method and raise on lookup failure.

        Args:
            validation_result: The provider's verified token result.
            access_token: The bearer token presented to the gateway.

        Returns:
            Recovered memberships, or None to leave the claims unchanged.
        """
        return None

    def is_delegated_user(self, validation_result: dict[str, Any]) -> bool:
        """Whether the verified result is a user token, not an M2M client.

        Args:
            validation_result: The provider's verified token result.

        Returns:
            False unless a provider implements delegated-user classification.
        """
        return False

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
