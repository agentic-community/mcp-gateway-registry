"""
Base federation client interface.

Provides common functionality for all federation clients.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any

import httpx

from ...common.log_redaction import redact_url
from ...utils.url_guard import FEDERATION_PROFILE, guarded_client

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


class BaseFederationClient(ABC):
    """Base class for federation clients."""

    def __init__(self, endpoint: str, timeout_seconds: int = 30, retry_attempts: int = 3):
        """
        Initialize federation client.

        Args:
            endpoint: Base URL for the federation API
            timeout_seconds: HTTP request timeout
            retry_attempts: Number of retry attempts for failed requests
        """
        self.endpoint = endpoint.rstrip("/")
        self.timeout_seconds = timeout_seconds
        self.retry_attempts = retry_attempts
        # SSRF-safe client: every request (and redirect hop) is resolved,
        # validated, and pinned to a public IP inside the transport, so there is
        # no DNS-rebinding window between validation and connect. The FEDERATION
        # profile grants NO allowlist bypass, matching the write-time endpoint
        # guard's empty allowlist. Federation requests carry a bearer credential,
        # so this atomic guard — not a bypassable pre-check — is what keeps the
        # token from ever reaching a private/loopback/metadata address.
        self.client = guarded_client(
            profile=FEDERATION_PROFILE,
            timeout=timeout_seconds,
        )

    def __del__(self):
        """Clean up HTTP client."""
        if hasattr(self, "client"):
            self.client.close()

    @abstractmethod
    def fetch_server(self, server_name: str, **kwargs) -> dict[str, Any] | None:
        """
        Fetch a single server from the federated registry.

        Args:
            server_name: Name of the server to fetch
            **kwargs: Additional parameters specific to the federation source

        Returns:
            Server data dictionary or None if fetch fails
        """
        pass

    @abstractmethod
    def fetch_all_servers(self, server_names: list[str], **kwargs) -> list[dict[str, Any]]:
        """
        Fetch multiple servers from the federated registry.

        Args:
            server_names: List of server names to fetch
            **kwargs: Additional parameters specific to the federation source

        Returns:
            List of server data dictionaries
        """
        pass

    def _make_request(
        self,
        url: str,
        method: str = "GET",
        headers: dict[str, str] | None = None,
        params: dict[str, Any] | None = None,
        data: dict[str, Any] | None = None,
    ) -> dict[str, Any] | None:
        """
        Make HTTP request with retry logic.

        Args:
            url: Full URL to request
            method: HTTP method (GET, POST, etc.)
            headers: HTTP headers
            params: Query parameters
            data: Request body data

        Returns:
            Response JSON or None if request fails
        """
        # Fail-closed SSRF pre-check at the egress chokepoint, using the same
        # FEDERATION profile (no allowlist bypass) as the pinned transport below,
        # so an early rejection is logged with a clear message. This is a
        # convenience gate only: the authoritative, rebinding-safe block happens
        # inside the guarded transport, which re-resolves, validates, and pins
        # every request (and redirect hop) to a public IP at connect time — there
        # is no TOCTOU window it can miss. Federation requests carry a bearer
        # credential in ``headers``, so the token can never reach a
        # private/loopback/link-local/metadata address.
        from ...exceptions import UrlValidationError
        from ...utils.url_guard import FEDERATION_PROFILE, validate_url

        try:
            validate_url(url, profile=FEDERATION_PROFILE)
        except UrlValidationError as exc:
            logger.error(f"Refusing federation request to unsafe URL {redact_url(url)}: {exc}")
            return None

        for attempt in range(self.retry_attempts):
            try:
                logger.debug(
                    f"Making {method} request to {redact_url(url)} "
                    f"(attempt {attempt + 1}/{self.retry_attempts})"
                )

                response = self.client.request(
                    method=method, url=url, headers=headers, params=params, json=data
                )

                response.raise_for_status()
                return response.json()

            except httpx.HTTPStatusError as e:
                logger.error(f"HTTP error {e.response.status_code} for {redact_url(url)}: {e}")
                if e.response.status_code in [404, 401, 403]:
                    # Don't retry for these errors
                    return None
                if attempt == self.retry_attempts - 1:
                    return None

            except httpx.RequestError as e:
                logger.error(f"Request error for {redact_url(url)}: {e}")
                if attempt == self.retry_attempts - 1:
                    return None

            except Exception as e:
                logger.error(f"Unexpected error for {redact_url(url)}: {e}")
                if attempt == self.retry_attempts - 1:
                    return None

        return None
