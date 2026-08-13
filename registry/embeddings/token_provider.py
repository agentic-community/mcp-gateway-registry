"""OAuth2 client-credentials token provider for IdP-gated embedding endpoints.

Generic implementation that works with any OAuth2-compliant IdP (Keycloak,
Microsoft Entra ID, Okta, Auth0, PingFederate, etc.). Mirrors the pattern
in registry/services/federation/federation_auth.py: thread-safe cache with
an expiry buffer and automatic refresh.

Injected into LiteLLMClient so a fresh bearer is used on every embedding call.
"""

import logging
from datetime import UTC, datetime, timedelta
from threading import Lock
from urllib.parse import urlparse

import httpx

logger = logging.getLogger(__name__)

TOKEN_REFRESH_BUFFER_SECONDS: int = 60
DEFAULT_EXPIRES_IN_SECONDS: int = 3600
# http:// is permitted only for these loopback hosts (local dev), never a remote host.
_LOOPBACK_HOSTS: frozenset[str] = frozenset({"localhost", "127.0.0.1", "::1"})


def _validate_token_endpoint(
    token_endpoint: str,
    allow_insecure: bool,
) -> None:
    """Enforce https on the token endpoint (the client secret is POSTed to it).

    Single source of truth used by both ``_require_idp_settings`` (pre-flight) and
    ``EmbeddingsTokenProvider.__init__`` (constructor guard). http:// is permitted
    ONLY when ``allow_insecure`` is set AND the host is loopback (local dev). A
    remote http:// endpoint would transmit the client secret in cleartext, so it is
    rejected even with ``allow_insecure`` (the "explicit flag + localhost guard").
    """
    if token_endpoint.lower().startswith("https://"):
        return
    if not allow_insecure:
        raise ValueError(
            "EMBEDDINGS_IDP_TOKEN_ENDPOINT must use https:// "
            "(the client secret is sent to this endpoint). "
            "Set EMBEDDINGS_IDP_ALLOW_INSECURE=true for local development only."
        )
    host = (urlparse(token_endpoint).hostname or "").lower()
    if host not in _LOOPBACK_HOSTS:
        raise ValueError(
            "EMBEDDINGS_IDP_ALLOW_INSECURE only permits http:// for a loopback host "
            f"(localhost/127.0.0.1/::1); got host '{host}'. A remote http:// endpoint "
            "would transmit the client secret in cleartext."
        )
    logger.warning(
        "EMBEDDINGS_IDP_ALLOW_INSECURE=true: token endpoint is http:// (loopback). "
        "DO NOT use this in production."
    )


def _require_idp_settings(
    token_endpoint: str | None,
    client_id: str | None,
    client_secret: str | None,
    allow_insecure: bool = False,
) -> None:
    """Validate IdP settings when embeddings_auth_mode == 'idp'. Fail hard."""
    missing = [
        name
        for name, value in (
            ("EMBEDDINGS_IDP_TOKEN_ENDPOINT", token_endpoint),
            ("EMBEDDINGS_IDP_CLIENT_ID", client_id),
            ("EMBEDDINGS_IDP_CLIENT_SECRET", client_secret),
        )
        if not value
    ]
    if missing:
        raise ValueError("EMBEDDINGS_AUTH_MODE=idp requires these settings: " + ", ".join(missing))
    _validate_token_endpoint(token_endpoint, allow_insecure)


class EmbeddingsTokenProvider:
    """Fetches and caches an OAuth2 client-credentials access token.

    Works with any standard OAuth2 token endpoint (Keycloak, Entra, Okta, etc.).
    Thread-safe: multiple concurrent encode() calls share one cached token.
    """

    def __init__(
        self,
        token_endpoint: str,
        client_id: str,
        client_secret: str,
        scope: str | None = None,
        timeout_seconds: int = 30,
        allow_insecure: bool = False,
    ) -> None:
        _validate_token_endpoint(token_endpoint, allow_insecure)
        self._token_endpoint = token_endpoint
        self._client_id = client_id
        self._client_secret = client_secret
        self._scope = scope
        self._access_token: str | None = None
        self._token_expiry: datetime | None = None
        self._lock = Lock()
        self._http_client = httpx.Client(timeout=timeout_seconds)
        logger.info(
            "EmbeddingsTokenProvider initialized (endpoint=%s, client_id=%s)",
            token_endpoint,
            client_id,
        )

    def _is_token_valid(self) -> bool:
        if not self._access_token or not self._token_expiry:
            return False
        now = datetime.now(UTC)
        return now < (self._token_expiry - timedelta(seconds=TOKEN_REFRESH_BUFFER_SECONDS))

    def _refresh_token(self) -> str:
        data: dict[str, str] = {
            "grant_type": "client_credentials",
            "client_id": self._client_id,
            "client_secret": self._client_secret,
        }
        if self._scope:
            data["scope"] = self._scope
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Accept": "application/json",
        }
        response = self._http_client.post(self._token_endpoint, data=data, headers=headers)
        response.raise_for_status()
        token_data = response.json()
        access_token = token_data.get("access_token")
        if not access_token:
            raise ValueError("Token response missing 'access_token'")
        expires_in = int(token_data.get("expires_in", DEFAULT_EXPIRES_IN_SECONDS))
        self._access_token = access_token
        self._token_expiry = datetime.now(UTC) + timedelta(seconds=expires_in)
        logger.info("Obtained embeddings IdP token (expires in %ds)", expires_in)
        return access_token

    def get_token(self) -> str:
        """Return a valid bearer token, refreshing if near expiry.

        Raises:
            RuntimeError: if the token cannot be obtained.
        """
        from registry.observability.meters import embeddings_idp_token_refresh_total

        with self._lock:
            if self._is_token_valid():
                logger.debug("Using cached embeddings IdP token")
                return self._access_token
            try:
                token = self._refresh_token()
                embeddings_idp_token_refresh_total.labels(result="success").inc()
                return token
            except httpx.HTTPStatusError as exc:
                embeddings_idp_token_refresh_total.labels(result="failure").inc()
                raise RuntimeError(
                    f"IdP token request failed with status {exc.response.status_code}. "
                    "Check EMBEDDINGS_IDP_CLIENT_ID / _CLIENT_SECRET / _SCOPE."
                ) from exc
            except httpx.RequestError as exc:
                embeddings_idp_token_refresh_total.labels(result="failure").inc()
                raise RuntimeError(f"Network error contacting IdP token endpoint: {exc}") from exc
            except (ValueError, KeyError) as exc:
                # Malformed token response: non-JSON body, non-numeric expires_in,
                # or missing access_token. Count it (so the failure metric is
                # accurate) and surface the actionable domain error.
                embeddings_idp_token_refresh_total.labels(result="failure").inc()
                raise RuntimeError(f"Malformed IdP token response: {exc}") from exc

    def close(self) -> None:
        """Close the underlying HTTP client."""
        self._http_client.close()

    def __del__(self) -> None:
        """Best-effort close of the HTTP client (mirrors FederationAuthManager)."""
        client = getattr(self, "_http_client", None)
        if client is not None:
            try:
                client.close()
            except Exception:  # nosec B110 - best-effort cleanup in __del__; never raise during GC/shutdown
                pass
