from __future__ import annotations

from dataclasses import dataclass
from datetime import (
    datetime,
    timedelta,
    timezone,
)
from typing import (
    Any,
    Optional,
)

import httpx


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _normalize_token_type(
    value: Optional[str],
) -> str:
    if value is None:
        return "Bearer"
    stripped = value.strip()
    return stripped or "Bearer"


def _parse_scopes(
    value: object,
) -> Optional[list[str]]:
    if value is None:
        return None

    if isinstance(value, str):
        normalized = [item.strip() for item in value.split(" ") if item.strip()]
        return sorted(set(normalized)) if normalized else None

    if isinstance(value, list):
        normalized = [str(item).strip() for item in value if str(item).strip()]
        return sorted(set(normalized)) if normalized else None

    return None


def _require_non_empty_str(
    value: object,
    *,
    label: str,
) -> str:
    if not isinstance(value, str):
        raise ValueError(f"Token response missing {label}")
    stripped = value.strip()
    if not stripped:
        raise ValueError(f"Token response missing {label}")
    return stripped


@dataclass(frozen=True)
class OAuthTokenSet:
    access_token: str
    token_type: str
    refresh_token: Optional[str]
    id_token: Optional[str]
    expires_at: Optional[datetime]
    scopes: Optional[list[str]]


class OAuthTokenClientError(Exception):
    def __init__(
        self,
        *,
        message: str,
    ) -> None:
        super().__init__(message)
        self.message = message


class OAuthTokenClient:
    def __init__(
        self,
        *,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        timeout_seconds: float = 10.0,
    ) -> None:
        self._transport = transport
        self._timeout = httpx.Timeout(timeout_seconds)

    async def exchange_authorization_code(
        self,
        *,
        token_endpoint: str,
        client_id: str,
        client_secret: str,
        code: str,
        redirect_uri: str,
        code_verifier: str,
    ) -> OAuthTokenSet:
        form = {
            "grant_type": "authorization_code",
            "code": code,
            "redirect_uri": redirect_uri,
            "code_verifier": code_verifier,
        }
        return await self._post_token_request(
            token_endpoint=token_endpoint,
            client_id=client_id,
            client_secret=client_secret,
            form=form,
        )

    async def refresh_token(
        self,
        *,
        token_endpoint: str,
        client_id: str,
        client_secret: str,
        refresh_token: str,
    ) -> OAuthTokenSet:
        form = {
            "grant_type": "refresh_token",
            "refresh_token": refresh_token,
        }
        return await self._post_token_request(
            token_endpoint=token_endpoint,
            client_id=client_id,
            client_secret=client_secret,
            form=form,
        )

    async def _post_token_request(
        self,
        *,
        token_endpoint: str,
        client_id: str,
        client_secret: str,
        form: dict[str, str],
    ) -> OAuthTokenSet:
        headers = {
            "Accept": "application/json",
        }

        async with httpx.AsyncClient(
            transport=self._transport,
            timeout=self._timeout,
            follow_redirects=False,
        ) as client:
            try:
                response = await client.post(
                    token_endpoint,
                    data=form,
                    auth=(client_id, client_secret),
                    headers=headers,
                )
            except httpx.HTTPError as exc:
                raise OAuthTokenClientError(message="Token endpoint request failed") from exc

        if response.status_code >= 400:
            raise OAuthTokenClientError(
                message=f"Token endpoint returned HTTP {response.status_code}",
            )

        try:
            data: Any = response.json()
        except ValueError as exc:
            raise OAuthTokenClientError(message="Token endpoint returned invalid JSON") from exc

        if not isinstance(data, dict):
            raise OAuthTokenClientError(message="Token endpoint returned invalid JSON")

        access_token = _require_non_empty_str(data.get("access_token"), label="access_token")
        token_type = _normalize_token_type(
            data.get("token_type") if isinstance(data.get("token_type"), str) else None
        )

        refresh_token: Optional[str] = None
        if isinstance(data.get("refresh_token"), str) and data["refresh_token"].strip():
            refresh_token = data["refresh_token"].strip()

        id_token: Optional[str] = None
        if isinstance(data.get("id_token"), str) and data["id_token"].strip():
            id_token = data["id_token"].strip()

        expires_at: Optional[datetime] = None
        expires_in_raw = data.get("expires_in")
        if isinstance(expires_in_raw, (int, float)):
            expires_in = int(expires_in_raw)
            if expires_in > 0:
                expires_at = _utc_now() + timedelta(seconds=expires_in)

        scopes = _parse_scopes(data.get("scope"))

        return OAuthTokenSet(
            access_token=access_token,
            token_type=token_type,
            refresh_token=refresh_token,
            id_token=id_token,
            expires_at=expires_at,
            scopes=scopes,
        )

