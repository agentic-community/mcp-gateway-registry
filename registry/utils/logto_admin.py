"""Minimal async client for the Logto Management API.

Auth model: a machine-to-machine application (client credentials) holding the
default ``Logto Management API access`` role, exchanged at the tenant's OIDC
token endpoint. Two hard requirements (verified live 2026-09-18): the token
request must carry BOTH ``resource`` and ``scope=all`` — a token without the
scope claim gets ``auth.forbidden`` (403) from every management endpoint even
when the role is assigned to the application.

Group semantics: this deployment maps Logto *roles* onto IAM groups (the
fork's Logto auth provider emits ``groups = claims["roles"]`` in user JWTs),
so the IAM manager consumes the roles endpoints of this API.
"""

import logging
import os
import time
from typing import Any

import httpx

logger = logging.getLogger(__name__)

LOGTO_ENDPOINT: str = os.environ.get("LOGTO_ENDPOINT", "http://logto:3001").rstrip("/")
LOGTO_MANAGEMENT_M2M_CLIENT_ID: str = os.environ.get("LOGTO_MANAGEMENT_M2M_CLIENT_ID", "")
LOGTO_MANAGEMENT_M2M_CLIENT_SECRET: str = os.environ.get("LOGTO_MANAGEMENT_M2M_CLIENT_SECRET", "")
# Resource indicator of the Management API. Self-hosted default tenant ships
# the fixed identifier below; override only for custom tenant setups.
LOGTO_MANAGEMENT_RESOURCE: str = os.environ.get(
    "LOGTO_MANAGEMENT_RESOURCE", "https://default.logto.app/api"
)

MANAGEMENT_SCOPE = "all"
_REQUEST_TIMEOUT = 15.0
_TOKEN_EXPIRY_MARGIN_SECS = 60


class LogtoAdminError(Exception):
    """A Management API call failed (non-2xx or transport error).

    The message deliberately embeds the HTTP status word ("HTTP 403",
    "HTTP 404") so ``iam_errors.looks_forbidden``/``looks_not_found`` can
    classify it for the generic 502/404 translation in management routes.
    """

    def __init__(self, message: str, status_code: int | None = None):
        super().__init__(message)
        self.status_code = status_code


class LogtoAdminClient:
    """Token-cached client for ``{LOGTO_ENDPOINT}/api/*`` endpoints."""

    def __init__(
        self,
        endpoint: str = LOGTO_ENDPOINT,
        client_id: str = LOGTO_MANAGEMENT_M2M_CLIENT_ID,
        client_secret: str = LOGTO_MANAGEMENT_M2M_CLIENT_SECRET,
        resource: str = LOGTO_MANAGEMENT_RESOURCE,
    ):
        self.endpoint = endpoint.rstrip("/")
        self.client_id = client_id
        self.client_secret = client_secret
        self.resource = resource
        self._token: str | None = None
        self._token_expires_at: float = 0.0

    def configured(self) -> bool:
        return bool(self.client_id and self.client_secret)

    async def _get_token(self, force_refresh: bool = False) -> str:
        if not self.configured():
            raise LogtoAdminError(
                "Logto management client is not configured "
                "(LOGTO_MANAGEMENT_M2M_CLIENT_ID/SECRET missing)"
            )
        now = time.monotonic()
        if self._token and not force_refresh and now < self._token_expires_at:
            return self._token

        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as http:
            response = await http.post(
                f"{self.endpoint}/oidc/token",
                data={
                    "grant_type": "client_credentials",
                    "client_id": self.client_id,
                    "client_secret": self.client_secret,
                    "resource": self.resource,
                    "scope": MANAGEMENT_SCOPE,
                },
                headers={"Content-Type": "application/x-www-form-urlencoded"},
            )
        if response.status_code != 200:
            raise LogtoAdminError(
                f"Failed to obtain Logto management token: HTTP {response.status_code}",
                status_code=response.status_code,
            )
        payload = response.json()
        self._token = payload.get("access_token")
        if not self._token:
            raise LogtoAdminError("Logto token endpoint returned no access_token")
        self._token_expires_at = now + max(payload.get("expires_in", 3600) - _TOKEN_EXPIRY_MARGIN_SECS, 30)
        return self._token

    async def request(
        self,
        method: str,
        path: str,
        json: Any | None = None,
        params: dict[str, Any] | None = None,
        _retry_on_401: bool = True,
    ) -> Any:
        token = await self._get_token()
        url = f"{self.endpoint}{path if path.startswith('/') else '/' + path}"
        async with httpx.AsyncClient(timeout=_REQUEST_TIMEOUT) as http:
            response = await http.request(
                method,
                url,
                json=json,
                params=params,
                headers={"Authorization": f"Bearer {token}"},
            )
        if response.status_code == 401 and _retry_on_401:
            # Token revoked/expired early: refresh once and retry.
            await self._get_token(force_refresh=True)
            return await self.request(method, path, json=json, params=params, _retry_on_401=False)
        if response.status_code >= 400:
            raise LogtoAdminError(
                f"Logto management API {method} {path} failed: HTTP {response.status_code}",
                status_code=response.status_code,
            )
        if response.status_code == 204 or not response.content:
            return None
        return response.json()

    async def get(self, path: str, params: dict[str, Any] | None = None) -> Any:
        return await self.request("GET", path, params=params)

    async def post(self, path: str, json: Any | None = None) -> Any:
        return await self.request("POST", path, json=json)

    async def patch(self, path: str, json: Any | None = None) -> Any:
        return await self.request("PATCH", path, json=json)

    async def delete(self, path: str) -> Any:
        return await self.request("DELETE", path)

    async def get_paged(
        self,
        path: str,
        params: dict[str, Any] | None = None,
        page_size: int = 100,
        max_items: int = 1000,
    ) -> list[Any]:
        """Walk Logto's 1-based pagination until a short page or max_items."""
        items: list[Any] = []
        page = 1
        while len(items) < max_items:
            merged = dict(params or {})
            merged["page"] = page
            merged["page_size"] = page_size
            batch = await self.get(path, params=merged)
            if not isinstance(batch, list):
                raise LogtoAdminError(f"Logto management API {path} returned a non-list payload")
            items.extend(batch)
            if len(batch) < page_size:
                break
            page += 1
        return items[:max_items]


_client: LogtoAdminClient | None = None


def get_logto_admin() -> LogtoAdminClient:
    """Process-wide singleton (env is read once at import, like keycloak_manager)."""
    global _client
    if _client is None:
        _client = LogtoAdminClient()
    return _client
