"""
Microsoft Entra ID group and user management utilities.

This module provides functions to manage users and groups in Entra ID
via the Microsoft Graph API. It handles authentication, user/group CRUD
operations, and integrates with the registry.
"""

import asyncio
import logging
import os
import re
import secrets
import string
from typing import Any

import httpx

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Environment variables for Entra ID management
ENTRA_TENANT_ID: str = os.environ.get("ENTRA_TENANT_ID", "")
ENTRA_CLIENT_ID: str = os.environ.get("ENTRA_CLIENT_ID", "")
ENTRA_CLIENT_SECRET: str = os.environ.get("ENTRA_CLIENT_SECRET", "")

GRAPH_BASE_URL: str = "https://graph.microsoft.com/v1.0"


class EntraAdminError(RuntimeError):
    """Raised when Entra ID Graph API operations fail."""


def _is_guid(value: str) -> bool:
    """Check if a string looks like a GUID."""
    guid_pattern = re.compile(
        r"^[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$", re.I
    )
    return bool(guid_pattern.match(value))


def _generate_temp_password() -> str:
    """Generate a temporary password meeting Entra ID requirements."""
    # Entra ID password requirements: 8+ chars, 3 of 4 categories
    # (upper, lower, digit, special)
    alphabet = string.ascii_letters + string.digits + "!@#$%^&*()"
    password = "".join(secrets.choice(alphabet) for _ in range(16))
    return password


def _auth_headers(token: str) -> dict[str, str]:
    """Build auth headers for Graph API calls."""
    return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}


async def _get_entra_admin_token() -> str:
    """
    Get admin access token from Entra ID for Graph API calls.

    Uses client credentials flow with the app registration credentials.

    Returns:
        Access token string for Graph API

    Raises:
        EntraAdminError: If authentication fails
    """
    if not ENTRA_CLIENT_SECRET:
        raise EntraAdminError("ENTRA_CLIENT_SECRET environment variable not set")

    if not ENTRA_TENANT_ID:
        raise EntraAdminError("ENTRA_TENANT_ID environment variable not set")

    if not ENTRA_CLIENT_ID:
        raise EntraAdminError("ENTRA_CLIENT_ID environment variable not set")

    token_url = f"https://login.microsoftonline.com/{ENTRA_TENANT_ID}/oauth2/v2.0/token"

    data = {
        "grant_type": "client_credentials",
        "client_id": ENTRA_CLIENT_ID,
        "client_secret": ENTRA_CLIENT_SECRET,
        "scope": "https://graph.microsoft.com/.default",
    }

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.post(
                token_url, data=data, headers={"Content-Type": "application/x-www-form-urlencoded"}
            )
            response.raise_for_status()

            token_data = response.json()
            access_token = token_data.get("access_token")

            if not access_token:
                raise EntraAdminError("No access token in Entra ID response")

            logger.info("Successfully obtained Entra ID Graph API admin token")
            return access_token

    except httpx.HTTPStatusError as e:
        logger.error(f"Failed to authenticate with Entra ID: HTTP {e.response.status_code}")
        raise EntraAdminError(
            f"Entra ID authentication failed: HTTP {e.response.status_code}"
        ) from e
    except Exception as e:
        logger.error(f"Error getting Entra ID admin token: {e}")
        raise EntraAdminError(f"Failed to authenticate with Entra ID: {e}") from e


async def _get_default_domain(token: str) -> str:
    """Get the default verified domain for the tenant."""
    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{GRAPH_BASE_URL}/organization",
            headers=_auth_headers(token),
            params={"$select": "verifiedDomains"},
        )
        response.raise_for_status()

        data = response.json()
        orgs = data.get("value", [])

        if orgs:
            domains = orgs[0].get("verifiedDomains", [])
            for domain in domains:
                if domain.get("isDefault"):
                    return domain.get("name", "")
            if domains:
                return domains[0].get("name", "")

        raise EntraAdminError("Unable to determine default domain for tenant")


async def _find_group_id_by_name(
    client: httpx.AsyncClient, token: str, group_name: str
) -> str | None:
    """Find a group's object ID by display name."""
    response = await client.get(
        f"{GRAPH_BASE_URL}/groups",
        headers=_auth_headers(token),
        params={"$filter": f"displayName eq '{group_name}'", "$select": "id"},
    )
    response.raise_for_status()

    data = response.json()
    groups = data.get("value", [])

    if groups:
        return groups[0].get("id")
    return None


async def _find_user_id(
    client: httpx.AsyncClient, token: str, username_or_email: str
) -> str | None:
    """Find a user's object ID by userPrincipalName or email.

    Args:
        client: HTTP client
        token: Admin token
        username_or_email: User principal name or email address

    Returns:
        User's object ID if found, None otherwise
    """
    # Try to find by userPrincipalName first
    response = await client.get(
        f"{GRAPH_BASE_URL}/users",
        headers=_auth_headers(token),
        params={
            "$filter": f"userPrincipalName eq '{username_or_email}'",
            "$select": "id",
        },
    )
    if response.status_code == 200:
        data = response.json()
        users = data.get("value", [])
        if users:
            return users[0].get("id")

    # Try by mail if not found by UPN
    response = await client.get(
        f"{GRAPH_BASE_URL}/users",
        headers=_auth_headers(token),
        params={
            "$filter": f"mail eq '{username_or_email}'",
            "$select": "id",
        },
    )
    if response.status_code == 200:
        data = response.json()
        users = data.get("value", [])
        if users:
            return users[0].get("id")

    return None


async def _get_user_groups(client: httpx.AsyncClient, token: str, user_id: str) -> list[str]:
    """Fetch group names for a user in Entra ID."""
    try:
        response = await client.get(
            f"{GRAPH_BASE_URL}/users/{user_id}/memberOf",
            headers=_auth_headers(token),
            params={"$select": "id,displayName"},
        )
        response.raise_for_status()

        data = response.json()
        groups = data.get("value", [])

        # Return group display names
        return [
            g.get("displayName", "")
            for g in groups
            if g.get("@odata.type") == "#microsoft.graph.group"
        ]

    except Exception as e:
        logger.warning(f"Failed to get groups for user {user_id}: {e}")
        return []


async def _add_user_to_group_by_name(
    client: httpx.AsyncClient, token: str, user_id: str, group_name: str
) -> None:
    """Add a user to a group by group display name."""
    group_id = await _find_group_id_by_name(client, token, group_name)
    if not group_id:
        raise EntraAdminError(f"Group '{group_name}' not found")

    payload = {"@odata.id": f"{GRAPH_BASE_URL}/directoryObjects/{user_id}"}

    response = await client.post(
        f"{GRAPH_BASE_URL}/groups/{group_id}/members/$ref",
        headers=_auth_headers(token),
        json=payload,
    )

    # 204 = success, 400 with "already exist" = also acceptable
    if response.status_code not in (204, 400):
        raise EntraAdminError(
            f"Failed to add user to group '{group_name}' (HTTP {response.status_code})"
        )


async def _remove_user_from_group_by_name(
    client: httpx.AsyncClient, token: str, user_id: str, group_name: str
) -> None:
    """Remove a user from a group by group display name."""
    group_id = await _find_group_id_by_name(client, token, group_name)
    if not group_id:
        logger.warning(f"Group '{group_name}' not found, skipping removal")
        return

    response = await client.delete(
        f"{GRAPH_BASE_URL}/groups/{group_id}/members/{user_id}/$ref",
        headers=_auth_headers(token),
    )

    # 204 = success, 404 = user not in group (also acceptable)
    if response.status_code not in (204, 404):
        raise EntraAdminError(
            f"Failed to remove user from group '{group_name}' (HTTP {response.status_code})"
        )


async def _add_service_principal_to_group(
    client: httpx.AsyncClient, token: str, sp_id: str, group_name: str
) -> None:
    """Add a service principal to a group by group display name.

    Includes retry logic to handle Entra ID eventual consistency where
    the service principal may not be immediately available after creation.
    """
    logger.debug(f"Looking up group '{group_name}' for SP assignment")
    group_id = await _find_group_id_by_name(client, token, group_name)
    if not group_id:
        logger.warning(f"Group '{group_name}' not found in Entra ID, skipping assignment")
        return

    logger.debug(f"Found group '{group_name}' with ID: {group_id}")

    payload = {"@odata.id": f"{GRAPH_BASE_URL}/directoryObjects/{sp_id}"}

    # Retry logic for eventual consistency - SP may not be available immediately
    max_retries = 5
    retry_delay = 2.0

    for attempt in range(max_retries):
        response = await client.post(
            f"{GRAPH_BASE_URL}/groups/{group_id}/members/$ref",
            headers=_auth_headers(token),
            json=payload,
        )

        # 204 = success, 400 with "already exist" = also acceptable
        if response.status_code == 204:
            logger.info(f"Successfully added service principal {sp_id} to group '{group_name}'")
            return

        if response.status_code == 400:
            # Check if already a member (acceptable)
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", "")
            if "already exist" in error_msg.lower():
                logger.info(
                    f"Service principal {sp_id} is already a member of group '{group_name}'"
                )
                return
            logger.warning(f"Failed to add SP to group '{group_name}': {error_msg}")
            return

        if response.status_code == 404:
            # Could be eventual consistency - SP not yet propagated
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", "")
            logger.debug(f"HTTP 404 response: {error_msg}")

            if attempt < max_retries - 1:
                logger.warning(
                    f"Service principal not yet available for group assignment "
                    f"(attempt {attempt + 1}/{max_retries}), retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 1.5
                continue

        # Other error status codes
        logger.warning(
            f"Failed to add service principal to group '{group_name}': HTTP {response.status_code}"
        )
        try:
            error_detail = response.json()
            logger.debug(f"Error details: {error_detail}")
        except Exception as e:
            logger.warning(f"Could not parse error response from Entra: {e}")
        return

    logger.warning(
        f"Failed to add service principal {sp_id} to group '{group_name}' "
        f"after {max_retries} retries"
    )


# ==================== USER MANAGEMENT ====================


async def list_entra_users(
    search: str | None = None, max_results: int = 500, include_groups: bool = True
) -> list[dict[str, Any]]:
    """
    List users in Entra ID tenant.

    Args:
        search: Optional search filter (filters on displayName, userPrincipalName)
        max_results: Maximum number of results to return
        include_groups: Whether to include group memberships (slower)

    Returns:
        List of user dictionaries with id, username, email, etc.
    """
    admin_token = await _get_entra_admin_token()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Build query parameters
        params: dict[str, Any] = {
            "$top": max_results,
            "$select": "id,displayName,userPrincipalName,mail,givenName,surname,accountEnabled",
        }

        if search:
            # Graph API filter syntax
            params["$filter"] = (
                f"startswith(displayName,'{search}') or startswith(userPrincipalName,'{search}')"
            )

        response = await client.get(
            f"{GRAPH_BASE_URL}/users", headers=_auth_headers(admin_token), params=params
        )
        response.raise_for_status()

        data = response.json()
        users = data.get("value", [])

        # Transform to match Keycloak format
        result = []
        for user in users:
            user_entry = {
                "id": user.get("id", ""),
                "username": user.get("userPrincipalName", ""),
                "email": user.get("mail"),
                "firstName": user.get("givenName"),
                "lastName": user.get("surname"),
                "enabled": user.get("accountEnabled", True),
                "groups": [],
            }

            # Optionally fetch group memberships
            if include_groups:
                user_entry["groups"] = await _get_user_groups(client, admin_token, user["id"])

            result.append(user_entry)

        return result


async def create_entra_human_user(
    username: str,
    email: str,
    first_name: str,
    last_name: str,
    groups: list[str],
    password: str | None = None,
) -> dict[str, Any]:
    """
    Create a human user in Entra ID.

    Args:
        username: User principal name (must include @domain.com)
        email: Email address
        first_name: Given name
        last_name: Surname
        groups: List of group display names to add user to
        password: Initial password (if None, a random password is generated)

    Returns:
        User dictionary with id, username, etc.
    """
    admin_token = await _get_entra_admin_token()

    # Entra ID requires userPrincipalName to include domain
    # If username doesn't have @, append the default domain
    if "@" not in username:
        # Get default domain from tenant
        default_domain = await _get_default_domain(admin_token)
        username = f"{username}@{default_domain}"

    user_payload = {
        "accountEnabled": True,
        "displayName": f"{first_name} {last_name}",
        "givenName": first_name,
        "surname": last_name,
        "userPrincipalName": username,
        "mail": email,
        "mailNickname": username.split("@")[0],
        "passwordProfile": {
            "forceChangePasswordNextSignIn": password is None,
            "password": password or _generate_temp_password(),
        },
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{GRAPH_BASE_URL}/users", headers=_auth_headers(admin_token), json=user_payload
        )

        if response.status_code == 409:
            raise EntraAdminError(f"User '{username}' already exists")

        response.raise_for_status()
        user_data = response.json()

        # Add user to groups
        user_id = user_data["id"]
        for group_name in groups:
            await _add_user_to_group_by_name(client, admin_token, user_id, group_name)

        return {
            "id": user_data.get("id"),
            "username": user_data.get("userPrincipalName"),
            "email": user_data.get("mail"),
            "firstName": user_data.get("givenName"),
            "lastName": user_data.get("surname"),
            "enabled": user_data.get("accountEnabled", True),
            "groups": groups,
        }


async def delete_entra_user(username_or_id: str) -> bool:
    """
    Delete a user from Entra ID.

    Args:
        username_or_id: User principal name or object ID

    Returns:
        True if successful
    """
    admin_token = await _get_entra_admin_token()

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.delete(
            f"{GRAPH_BASE_URL}/users/{username_or_id}", headers=_auth_headers(admin_token)
        )

        if response.status_code == 404:
            raise EntraAdminError(f"User '{username_or_id}' not found")

        if response.status_code != 204:
            raise EntraAdminError(f"Failed to delete user (HTTP {response.status_code})")

        logger.info(f"Deleted Entra ID user: {username_or_id}")
        return True


# ==================== GROUP MANAGEMENT ====================


async def list_entra_groups() -> list[dict[str, Any]]:
    """
    List all groups in Entra ID tenant.

    Returns:
        List of group dictionaries
    """
    admin_token = await _get_entra_admin_token()

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.get(
            f"{GRAPH_BASE_URL}/groups",
            headers=_auth_headers(admin_token),
            params={"$select": "id,displayName,description,securityEnabled"},
        )
        response.raise_for_status()

        data = response.json()
        groups = data.get("value", [])

        return [
            {
                "id": g.get("id", ""),
                "name": g.get("displayName", ""),
                "path": f"/{g.get('displayName', '')}",  # Emulate Keycloak path format
                "attributes": {
                    "description": [g.get("description", "")],
                    "securityEnabled": g.get("securityEnabled", True),
                },
            }
            for g in groups
        ]


async def create_entra_group(group_name: str, description: str = "") -> dict[str, Any]:
    """
    Create a security group in Entra ID.

    Args:
        group_name: Display name for the group
        description: Optional description

    Returns:
        Group dictionary with id, name, path
    """
    admin_token = await _get_entra_admin_token()

    group_payload = {
        "displayName": group_name,
        "description": description,
        "mailEnabled": False,
        "mailNickname": group_name.replace(" ", "-").lower(),
        "securityEnabled": True,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{GRAPH_BASE_URL}/groups", headers=_auth_headers(admin_token), json=group_payload
        )

        if response.status_code == 400:
            error_data = response.json()
            error_msg = error_data.get("error", {}).get("message", "")
            if "already exists" in error_msg.lower():
                raise EntraAdminError(f"Group '{group_name}' already exists")
            raise EntraAdminError(f"Failed to create group: {error_msg}")

        response.raise_for_status()
        group_data = response.json()

        logger.info(f"Created Entra ID group: {group_name}")

        return {
            "id": group_data.get("id", ""),
            "name": group_data.get("displayName", ""),
            "path": f"/{group_data.get('displayName', '')}",
            "attributes": {"description": [description]},
        }


async def delete_entra_group(group_name_or_id: str) -> bool:
    """
    Delete a group from Entra ID.

    Args:
        group_name_or_id: Group display name or object ID

    Returns:
        True if successful
    """
    admin_token = await _get_entra_admin_token()

    async with httpx.AsyncClient(timeout=10.0) as client:
        # If it looks like a name (not a GUID), find the group ID first
        group_id = group_name_or_id
        if not _is_guid(group_name_or_id):
            group_id = await _find_group_id_by_name(client, admin_token, group_name_or_id)
            if not group_id:
                raise EntraAdminError(f"Group '{group_name_or_id}' not found")

        response = await client.delete(
            f"{GRAPH_BASE_URL}/groups/{group_id}", headers=_auth_headers(admin_token)
        )

        if response.status_code == 404:
            raise EntraAdminError(f"Group '{group_name_or_id}' not found")

        if response.status_code != 204:
            raise EntraAdminError(f"Failed to delete group (HTTP {response.status_code})")

        logger.info(f"Deleted Entra ID group: {group_name_or_id}")
        return True


async def update_entra_group(
    group_name_or_id: str,
    description: str,
) -> dict[str, Any]:
    """
    Update a group's description in Entra ID.

    Args:
        group_name_or_id: Group display name or object ID
        description: New description for the group

    Returns:
        Dictionary with updated group info (id, name, description)

    Raises:
        EntraAdminError: If group not found or update fails
    """
    admin_token = await _get_entra_admin_token()

    async with httpx.AsyncClient(timeout=10.0) as client:
        # If it looks like a name (not a GUID), find the group ID first
        group_id = group_name_or_id
        group_display_name = group_name_or_id

        if not _is_guid(group_name_or_id):
            found_id = await _find_group_id_by_name(client, admin_token, group_name_or_id)
            if not found_id:
                raise EntraAdminError(f"Group '{group_name_or_id}' not found")
            group_id = found_id
        else:
            # If GUID provided, fetch the display name
            get_response = await client.get(
                f"{GRAPH_BASE_URL}/groups/{group_id}",
                headers=_auth_headers(admin_token),
                params={"$select": "displayName"},
            )
            if get_response.status_code == 404:
                raise EntraAdminError(f"Group '{group_name_or_id}' not found")
            get_response.raise_for_status()
            group_data = get_response.json()
            group_display_name = group_data.get("displayName", group_name_or_id)

        # Update group description via PATCH request
        update_payload = {"description": description}

        response = await client.patch(
            f"{GRAPH_BASE_URL}/groups/{group_id}",
            headers=_auth_headers(admin_token),
            json=update_payload,
        )

        if response.status_code == 404:
            raise EntraAdminError(f"Group '{group_name_or_id}' not found")

        if response.status_code != 204:
            raise EntraAdminError(f"Failed to update group (HTTP {response.status_code})")

        logger.info(f"Updated Entra ID group: {group_display_name}")

        return {
            "id": group_id,
            "name": group_display_name,
            "description": description,
        }


# ==================== SERVICE PRINCIPAL (M2M) MANAGEMENT ====================


async def create_service_principal_client(
    client_id_name: str, group_names: list[str], description: str | None = None
) -> dict[str, Any]:
    """
    Create or update a service principal (app registration) with group assignments.

    For Entra ID M2M authentication, this creates:
    1. An App Registration
    2. A Service Principal
    3. A client secret
    4. Assigns app roles or groups

    Args:
        client_id_name: Name for the application
        group_names: List of group names to assign (via app roles or group membership)
        description: Optional description

    Returns:
        Dictionary with client_id, client_secret, groups
    """
    admin_token = await _get_entra_admin_token()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # 1. Create App Registration
        app_payload = {
            "displayName": client_id_name,
            "description": description or f"Service account for {client_id_name}",
            "signInAudience": "AzureADMyOrg",
            "api": {"requestedAccessTokenVersion": 2},
        }

        app_response = await client.post(
            f"{GRAPH_BASE_URL}/applications", headers=_auth_headers(admin_token), json=app_payload
        )

        if app_response.status_code == 400:
            error_data = app_response.json()
            error_msg = error_data.get("error", {}).get("message", "")
            raise EntraAdminError(f"Failed to create app registration: {error_msg}")

        app_response.raise_for_status()
        app_data = app_response.json()

        app_id = app_data["appId"]  # This is the client_id
        app_object_id = app_data["id"]  # Object ID for managing the app

        # 2. Create Service Principal for the app (with retry for eventual consistency)
        sp_payload = {"appId": app_id}
        sp_max_retries = 5
        sp_retry_delay = 2.0
        sp_object_id = None

        for sp_attempt in range(sp_max_retries):
            sp_response = await client.post(
                f"{GRAPH_BASE_URL}/servicePrincipals",
                headers=_auth_headers(admin_token),
                json=sp_payload,
            )

            if sp_response.status_code in (200, 201):
                sp_data = sp_response.json()
                sp_object_id = sp_data["id"]
                logger.info(f"Created service principal: {sp_object_id}")
                break

            if sp_response.status_code == 400:
                error_data = sp_response.json()
                error_msg = error_data.get("error", {}).get("message", "")

                # Check if it's an eventual consistency issue
                if "does not reference a valid application object" in error_msg:
                    if sp_attempt < sp_max_retries - 1:
                        logger.warning(
                            f"App not yet propagated for SP creation "
                            f"(attempt {sp_attempt + 1}/{sp_max_retries}), "
                            f"retrying in {sp_retry_delay}s..."
                        )
                        await asyncio.sleep(sp_retry_delay)
                        sp_retry_delay *= 1.5
                        continue

                # Check if SP already exists
                logger.warning(f"Service principal creation returned 400: {error_msg}")
                find_sp_response = await client.get(
                    f"{GRAPH_BASE_URL}/servicePrincipals",
                    headers=_auth_headers(admin_token),
                    params={"$filter": f"appId eq '{app_id}'"},
                )
                find_sp_response.raise_for_status()
                find_sp_data = find_sp_response.json()
                existing_sps = find_sp_data.get("value", [])

                if existing_sps:
                    sp_object_id = existing_sps[0]["id"]
                    logger.info(f"Found existing service principal: {sp_object_id}")
                    break

                raise EntraAdminError(f"Failed to create service principal: {error_msg}")

            sp_response.raise_for_status()

        if not sp_object_id:
            raise EntraAdminError(
                f"Failed to create service principal after {sp_max_retries} retries"
            )

        # 3. Create client secret (with retry for eventual consistency)
        secret_payload = {
            "passwordCredential": {
                "displayName": f"{client_id_name}-secret",
                "endDateTime": "2099-12-31T23:59:59Z",  # Long-lived for M2M
            }
        }

        # Retry logic for eventual consistency in Entra ID
        max_retries = 3
        retry_delay = 2.0
        client_secret = None

        for attempt in range(max_retries):
            secret_response = await client.post(
                f"{GRAPH_BASE_URL}/applications/{app_object_id}/addPassword",
                headers=_auth_headers(admin_token),
                json=secret_payload,
            )

            if secret_response.status_code == 200:
                secret_data = secret_response.json()
                client_secret = secret_data["secretText"]
                break
            elif secret_response.status_code == 404 and attempt < max_retries - 1:
                # App not yet available due to eventual consistency
                logger.warning(
                    f"App not ready for password creation (attempt {attempt + 1}/{max_retries}), "
                    f"retrying in {retry_delay}s..."
                )
                await asyncio.sleep(retry_delay)
                retry_delay *= 2  # Exponential backoff
            else:
                secret_response.raise_for_status()

        if not client_secret:
            raise EntraAdminError("Failed to create client secret after retries")

        # 4. Add service principal to groups
        for group_name in group_names:
            await _add_service_principal_to_group(client, admin_token, sp_object_id, group_name)

        logger.info(f"Created Entra ID service principal: {client_id_name}")

        return {
            "client_id": app_id,
            "client_uuid": app_object_id,
            "service_principal_id": sp_object_id,
            "client_secret": client_secret,
            "groups": group_names,
        }


async def update_entra_user_groups(
    username_or_id: str,
    groups: list[str],
) -> dict[str, Any]:
    """
    Update group memberships for an Entra ID user or service principal.

    Calculates the diff between current and desired groups, then adds/removes
    groups as needed.

    Args:
        username_or_id: User principal name, email, or object ID
        groups: List of group names the user should belong to

    Returns:
        Dict with username and updated groups list

    Raises:
        EntraAdminError: If user not found or group operations fail
    """
    admin_token = await _get_entra_admin_token()

    async with httpx.AsyncClient(timeout=30.0) as client:
        # Try to find as a regular user first
        user_id = await _find_user_id(client, admin_token, username_or_id)
        is_service_principal = False

        if not user_id:
            # Try to find as a service principal by display name
            sp_response = await client.get(
                f"{GRAPH_BASE_URL}/servicePrincipals",
                headers=_auth_headers(admin_token),
                params={"$filter": f"displayName eq '{username_or_id}'"},
            )
            if sp_response.status_code == 200:
                sp_data = sp_response.json()
                sp_list = sp_data.get("value", [])
                if sp_list:
                    user_id = sp_list[0].get("id")
                    is_service_principal = True

        if not user_id:
            raise EntraAdminError(f"User or service principal '{username_or_id}' not found")

        # Get current groups
        current_groups_data = await _get_user_groups(client, admin_token, user_id)
        current_groups = set(current_groups_data)
        desired_groups = set(groups)

        # Calculate diff
        groups_to_add = desired_groups - current_groups
        groups_to_remove = current_groups - desired_groups

        # Apply changes
        for group_name in groups_to_add:
            if is_service_principal:
                await _add_service_principal_to_group(client, admin_token, user_id, group_name)
            else:
                await _add_user_to_group_by_name(client, admin_token, user_id, group_name)

        for group_name in groups_to_remove:
            await _remove_user_from_group_by_name(client, admin_token, user_id, group_name)

        logger.info(
            "Updated groups for %s '%s': added=%s, removed=%s",
            "service principal" if is_service_principal else "user",
            username_or_id,
            list(groups_to_add),
            list(groups_to_remove),
        )

        return {
            "username": username_or_id,
            "groups": list(desired_groups),
            "added": list(groups_to_add),
            "removed": list(groups_to_remove),
        }
