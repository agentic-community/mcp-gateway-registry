"""PingFederate admin API manager for M2M (service account) operations.

This module exposes a single async helper that creates or updates an OAuth2
``CLIENT_CREDENTIALS`` client in PingFederate via its admin API. It is the
PingFederate equivalent of ``keycloak_manager.create_service_account_client``.

Design note (groups):
    PingFederate's ``client_credentials`` access tokens do NOT carry group
    claims by default. The registry stores M2M client groups in the
    ``idp_m2m_clients`` MongoDB collection, and the auth-server enriches M2M
    tokens from that collection at validation time. So the helper here does
    NOT attach groups inside PingFederate; it only creates the OAuth client
    and returns the credentials. The caller (``management_routes.py``) is
    responsible for writing the groups into ``idp_m2m_clients``.

The PingFederate admin API contract (verb, path, headers, payload shape)
matches the bash bootstrap at ``pingfederate/setup/init-pingfederate.sh``
Step 6 (lines 151-175).
"""

import logging
import os
import re
import secrets
from typing import Any

import httpx

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


# Configuration -- mirrors registry.api.iam_user_groups_routes
_PF_ADMIN_URL: str = os.environ.get("PF_ADMIN_URL", "https://pingfederate:9999")
_PF_ADMIN_USER: str = os.environ.get("PF_ADMIN_USER", "administrator")
_PF_ADMIN_PASS: str = os.environ.get("PF_ADMIN_PASS", "2FederateM0re")
_PF_ADMIN_API_BASE: str = "/pf-admin-api/v1"
_PF_HTTP_TIMEOUT: float = 10.0

# clientId allowed character set per PingFederate admin API:
# alphanumerics, dash, underscore, period, length 1-256.
_CLIENT_ID_PATTERN = re.compile(r"^[A-Za-z0-9_\-.]{1,256}$")


class PingFederateAdminError(Exception):
    """Raised when an interaction with the PingFederate admin API fails."""


def _generate_client_secret() -> str:
    """Generate a 32-byte URL-safe random secret."""
    return secrets.token_urlsafe(32)


def _validate_client_id(
    client_id: str,
) -> None:
    """Validate that ``client_id`` matches PingFederate's accepted charset."""
    if not _CLIENT_ID_PATTERN.match(client_id):
        raise ValueError(
            f"Invalid client_id '{client_id}'. "
            "Must match ^[A-Za-z0-9_\\-.]{1,256}$ (alphanumerics, dash, underscore, period)."
        )


def _build_client_payload(
    client_id: str,
    client_secret: str,
    description: str | None,
) -> dict[str, Any]:
    """Build the PingFederate OAuth client payload.

    grantTypes is CLIENT_CREDENTIALS only. PingFederate rejects REFRESH_TOKEN
    for clients that don't issue refresh tokens (auth-code, implicit, RO-pwd).
    """
    return {
        "clientId": client_id,
        "name": description or f"M2M client: {client_id}",
        "clientAuth": {"type": "SECRET", "secret": client_secret},
        "grantTypes": ["CLIENT_CREDENTIALS"],
        "redirectUris": [],
        "enabled": True,
        "defaultAccessTokenManagerRef": {"id": "jwt"},
    }


def _pf_auth() -> tuple[str, str]:
    """Return basic-auth tuple for PingFederate admin API."""
    return (_PF_ADMIN_USER, _PF_ADMIN_PASS)


def _pf_headers() -> dict[str, str]:
    """Return required headers for PingFederate admin API."""
    return {
        "X-XSRF-Header": "PingFederate",
        "Content-Type": "application/json",
    }


async def _client_exists(
    client: httpx.AsyncClient,
    client_id: str,
) -> bool:
    """
    Check whether an OAuth client with ``client_id`` already exists in PingFederate.

    Returns True on HTTP 200, False on HTTP 404. Any other status is a hard error
    so the caller does not blindly attempt to create or update.
    """
    url = f"{_PF_ADMIN_URL}{_PF_ADMIN_API_BASE}/oauth/clients/{client_id}"
    response = await client.get(url, auth=_pf_auth(), headers=_pf_headers())

    if response.status_code == 200:
        return True
    if response.status_code == 404:
        return False

    logger.error(
        "PingFederate existence check for client '%s' failed: status_code=%s",
        client_id,
        response.status_code,
    )
    raise PingFederateAdminError(
        f"PingFederate existence check failed (status_code={response.status_code})"
    )


async def _create_or_update_client(
    client: httpx.AsyncClient,
    client_id: str,
    payload: dict[str, Any],
) -> None:
    """
    Create or update an OAuth client in PingFederate.

    Performs a GET to determine existence, then PUTs (update) or POSTs (create).
    Raises ``PingFederateAdminError`` on any non-2xx response.
    """
    exists = await _client_exists(client, client_id)

    if exists:
        url = f"{_PF_ADMIN_URL}{_PF_ADMIN_API_BASE}/oauth/clients/{client_id}"
        response = await client.put(
            url, auth=_pf_auth(), headers=_pf_headers(), json=payload
        )
        action = "update"
    else:
        url = f"{_PF_ADMIN_URL}{_PF_ADMIN_API_BASE}/oauth/clients"
        response = await client.post(
            url, auth=_pf_auth(), headers=_pf_headers(), json=payload
        )
        action = "create"

    if not (200 <= response.status_code < 300):
        # Log PF's response body server-side to aid debugging (PF returns a
        # validation-error JSON document on 422). Do NOT echo the body to
        # the caller: it can leak admin-API context.
        try:
            response_body = response.text
        except Exception:
            response_body = "<unreadable>"
        logger.error(
            "PingFederate client %s for '%s' failed: status_code=%s body=%s",
            action,
            client_id,
            response.status_code,
            response_body,
        )
        raise PingFederateAdminError(
            f"PingFederate client {action} failed (status_code={response.status_code})"
        )

    logger.info("PingFederate client '%s' %sd successfully", client_id, action)


async def create_pingfederate_service_account_client(
    client_id: str,
    group_names: list[str],
    description: str | None = None,
) -> dict[str, Any]:
    """
    Create or update an OAuth2 client_credentials client in PingFederate.

    Idempotent: if the client already exists in PingFederate, its grant
    types/redirect URIs/secret are reset to the values below. The client
    secret is regenerated on every call.

    Note on groups: PingFederate's client_credentials access tokens do NOT
    carry group claims by default. The registry stores M2M client groups in
    the ``idp_m2m_clients`` MongoDB collection, and the auth-server enriches
    M2M tokens from that collection at validation time. This function does
    NOT attach groups inside PingFederate; it only creates the OAuth client
    and returns the credentials. The caller (``management_routes.py``) is
    responsible for writing the groups into ``idp_m2m_clients``.

    Args:
        client_id: OAuth client_id to create/update.
        group_names: Groups to attach to this M2M client (used by caller for
            MongoDB write; not used inside this function).
        description: Optional human-readable description (becomes PF client name).

    Returns:
        Dict with keys:
        - client_id: str (echoes input)
        - client_secret: str (newly generated)
        - groups: list[str] (echoes input -- caller persists)

    Raises:
        ValueError: client_id fails validation.
        PingFederateAdminError: PF admin API call failed.
    """
    _validate_client_id(client_id)

    client_secret = _generate_client_secret()
    payload = _build_client_payload(
        client_id=client_id,
        client_secret=client_secret,
        description=description,
    )

    try:
        async with httpx.AsyncClient(verify=False, timeout=_PF_HTTP_TIMEOUT) as client:  # nosec B501 - PF admin uses self-signed cert in baseline profile
            await _create_or_update_client(client, client_id, payload)
    except PingFederateAdminError:
        raise
    except httpx.HTTPStatusError as exc:
        logger.error(
            "PingFederate admin API HTTP error for client '%s': %s",
            client_id,
            type(exc).__name__,
        )
        raise PingFederateAdminError(
            f"PingFederate admin API HTTP error ({type(exc).__name__})"
        ) from exc
    except httpx.RequestError as exc:
        logger.error(
            "PingFederate admin API request error for client '%s': %s",
            client_id,
            type(exc).__name__,
        )
        raise PingFederateAdminError(
            f"PingFederate admin API request error ({type(exc).__name__})"
        ) from exc
    except Exception as exc:
        logger.error(
            "PingFederate admin API unexpected error for client '%s': %s",
            client_id,
            type(exc).__name__,
        )
        raise PingFederateAdminError(
            f"PingFederate admin API unexpected error ({type(exc).__name__})"
        ) from exc

    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "groups": list(group_names),
    }
