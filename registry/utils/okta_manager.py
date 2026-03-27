"""Okta Admin API manager for user and group operations.

This module provides async functions for managing users and groups
in Okta using the Okta Admin API.
"""

import logging
import os
from typing import Any

import httpx

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Configuration from environment
OKTA_DOMAIN: str = os.environ.get("OKTA_DOMAIN", "")
OKTA_API_TOKEN: str = os.environ.get("OKTA_API_TOKEN", "")


def _get_api_headers() -> dict[str, str]:
    """Get headers for Okta Admin API requests."""
    if not OKTA_API_TOKEN:
        raise ValueError(
            "OKTA_API_TOKEN is not set. "
            "Create an API token in Okta Admin Console → Security → API → Tokens."
        )
    return {
        "Authorization": f"SSWS {OKTA_API_TOKEN}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _get_base_url() -> str:
    """Get Okta Admin API base URL."""
    domain = OKTA_DOMAIN.replace("https://", "").rstrip("/")
    return f"https://{domain}/api/v1"


def _check_rate_limit(response: httpx.Response) -> None:
    """Check for Okta rate limiting and raise appropriate error.

    Args:
        response: HTTP response to check

    Raises:
        ValueError: If rate limited, includes retry delay info
    """
    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 60))
        rate_limit_remaining = response.headers.get("X-Rate-Limit-Remaining", "0")
        logger.warning(
            f"Okta rate limit exceeded. "
            f"Remaining: {rate_limit_remaining}, Retry after: {retry_after}s"
        )
        raise ValueError(
            f"Okta API rate limited. Retry after {retry_after} seconds. "
            f"Consider reducing request frequency."
        )


async def list_okta_users(
    search: str | None = None,
    max_results: int = 500,
    include_groups: bool = True,
) -> list[dict[str, Any]]:
    """List users from Okta.

    Args:
        search: Optional search filter
        max_results: Maximum number of results to return
        include_groups: Whether to include group memberships

    Returns:
        List of user dictionaries
    """
    base_url = _get_base_url()
    headers = _get_api_headers()

    params: dict[str, Any] = {"limit": min(max_results, 200)}
    if search:
        params["search"] = f'profile.login sw "{search}" or profile.email sw "{search}"'

    users: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        url: str | None = f"{base_url}/users"

        while url and len(users) < max_results:
            response = await client.get(url, headers=headers, params=params)
            _check_rate_limit(response)
            response.raise_for_status()

            page_users = response.json()
            users.extend(page_users)

            url = response.links.get("next", {}).get("url")
            params = {}

        # Transform to common format
        result = []
        for user in users[:max_results]:
            user_data: dict[str, Any] = {
                "id": user.get("id"),
                "username": user.get("profile", {}).get("login"),
                "email": user.get("profile", {}).get("email"),
                "first_name": user.get("profile", {}).get("firstName"),
                "last_name": user.get("profile", {}).get("lastName"),
                "status": user.get("status"),
                "created": user.get("created"),
                "groups": [],
            }

            if include_groups:
                groups_url = f"{base_url}/users/{user['id']}/groups"
                groups_response = await client.get(groups_url, headers=headers)
                if groups_response.status_code == 200:
                    user_data["groups"] = [
                        g.get("profile", {}).get("name") for g in groups_response.json()
                    ]

            result.append(user_data)

    logger.info(f"Retrieved {len(result)} users from Okta")
    return result


async def create_okta_human_user(
    username: str,
    email: str,
    first_name: str,
    last_name: str,
    groups: list[str],
    password: str | None = None,
) -> dict[str, Any]:
    """Create a human user in Okta.

    Args:
        username: Username (login) for the account
        email: Email address
        first_name: First name
        last_name: Last name
        groups: List of group names to assign
        password: Optional initial password

    Returns:
        Dictionary with created user details
    """
    base_url = _get_base_url()
    headers = _get_api_headers()

    user_data: dict[str, Any] = {
        "profile": {
            "login": username,
            "email": email,
            "firstName": first_name,
            "lastName": last_name,
        }
    }

    if password:
        user_data["credentials"] = {"password": {"value": password}}

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{base_url}/users",
            headers=headers,
            json=user_data,
            params={"activate": "true" if password else "false"},
        )
        if response.status_code >= 400:
            try:
                error_body = response.json()
            except Exception:
                error_body = response.text
            logger.error(f"Okta user creation failed ({response.status_code}): {error_body}")
            raise ValueError(f"Okta user creation failed: {error_body}")
        created_user = response.json()

        # Assign to groups
        for group_name in groups:
            groups_response = await client.get(
                f"{base_url}/groups",
                headers=headers,
                params={"q": group_name},
            )
            groups_response.raise_for_status()
            matching_groups = groups_response.json()

            for group in matching_groups:
                if group.get("profile", {}).get("name") == group_name:
                    await client.put(
                        f"{base_url}/groups/{group['id']}/users/{created_user['id']}",
                        headers=headers,
                    )
                    break

    logger.info(f"Created Okta user: {username}")
    return {
        "id": created_user.get("id"),
        "username": username,
        "email": email,
        "groups": groups,
    }


async def delete_okta_user(username_or_id: str) -> bool:
    """Delete a user from Okta (deactivate then delete).

    Args:
        username_or_id: Username (login) or user ID

    Returns:
        True if successful

    Raises:
        ValueError: If user not found
    """
    base_url = _get_base_url()
    headers = _get_api_headers()

    async with httpx.AsyncClient(timeout=10.0) as client:
        # If it looks like a login, resolve to user ID
        if "@" in username_or_id or "." in username_or_id:
            response = await client.get(
                f"{base_url}/users/{username_or_id}",
                headers=headers,
            )
            if response.status_code == 200:
                user_id = response.json().get("id")
            else:
                raise ValueError(f"User not found: {username_or_id}")
        else:
            user_id = username_or_id

        # Deactivate user first (required before deletion)
        await client.post(
            f"{base_url}/users/{user_id}/lifecycle/deactivate",
            headers=headers,
        )

        # Delete user
        delete_response = await client.delete(
            f"{base_url}/users/{user_id}",
            headers=headers,
        )
        delete_response.raise_for_status()

    logger.info(f"Deleted Okta user: {username_or_id}")
    return True


async def list_okta_groups() -> list[dict[str, Any]]:
    """List all groups from Okta.

    Returns:
        List of group dictionaries with id, name, description, type
    """
    base_url = _get_base_url()
    headers = _get_api_headers()

    groups: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        url: str | None = f"{base_url}/groups"
        params: dict[str, Any] = {"limit": 200}

        while url:
            response = await client.get(url, headers=headers, params=params)
            response.raise_for_status()

            page_groups = response.json()
            groups.extend(page_groups)

            url = response.links.get("next", {}).get("url")
            params = {}

    result = [
        {
            "id": g.get("id"),
            "name": g.get("profile", {}).get("name"),
            "description": g.get("profile", {}).get("description", ""),
            "type": g.get("type"),
        }
        for g in groups
    ]

    logger.info(f"Retrieved {len(result)} groups from Okta")
    return result


async def create_okta_group(
    group_name: str,
    description: str = "",
) -> dict[str, Any]:
    """Create a group in Okta.

    Args:
        group_name: Name of the group
        description: Optional description

    Returns:
        Dictionary with created group details
    """
    base_url = _get_base_url()
    headers = _get_api_headers()

    group_data = {
        "profile": {
            "name": group_name,
            "description": description,
        }
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{base_url}/groups",
            headers=headers,
            json=group_data,
        )
        response.raise_for_status()
        created_group = response.json()

    logger.info(f"Created Okta group: {group_name}")
    return {
        "id": created_group.get("id"),
        "name": group_name,
        "description": description,
    }


async def delete_okta_group(group_name_or_id: str) -> bool:
    """Delete a group from Okta by name or ID.

    Resolves group name to ID if needed before deletion.

    Args:
        group_name_or_id: Group name or ID

    Returns:
        True if successful

    Raises:
        ValueError: If group not found
    """
    base_url = _get_base_url()
    headers = _get_api_headers()

    async with httpx.AsyncClient(timeout=10.0) as client:
        # If not a UUID-like string, search by name
        if "-" not in group_name_or_id or len(group_name_or_id) < 20:
            response = await client.get(
                f"{base_url}/groups",
                headers=headers,
                params={"q": group_name_or_id},
            )
            response.raise_for_status()
            groups = response.json()

            group_id = None
            for g in groups:
                if g.get("profile", {}).get("name") == group_name_or_id:
                    group_id = g.get("id")
                    break

            if not group_id:
                raise ValueError(f"Group not found: {group_name_or_id}")
        else:
            group_id = group_name_or_id

        delete_response = await client.delete(
            f"{base_url}/groups/{group_id}",
            headers=headers,
        )
        delete_response.raise_for_status()

    logger.info(f"Deleted Okta group: {group_name_or_id}")
    return True


async def create_okta_service_account(
    client_id_name: str,
    group_names: list[str],
    description: str | None = None,
) -> dict[str, Any]:
    """Create an OAuth2 service application (service account) in Okta.

    Creates an OIDC service app with client_credentials grant type
    and assigns it to the specified groups.

    Args:
        client_id_name: Name for the OAuth2 application
        group_names: List of group names to assign
        description: Optional description

    Returns:
        Dictionary with client_id and client_secret
    """
    base_url = _get_base_url()
    headers = _get_api_headers()

    app_data = {
        "name": "oidc_client",
        "label": client_id_name,
        "signOnMode": "OPENID_CONNECT",
        "credentials": {
            "oauthClient": {
                "token_endpoint_auth_method": "client_secret_basic",
            }
        },
        "settings": {
            "oauthClient": {
                "client_uri": None,
                "logo_uri": None,
                "redirect_uris": [],
                "response_types": ["token"],
                "grant_types": ["client_credentials"],
                "application_type": "service",
            }
        },
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{base_url}/apps",
            headers=headers,
            json=app_data,
        )
        response.raise_for_status()
        created_app = response.json()

        client_id = created_app.get("credentials", {}).get("oauthClient", {}).get("client_id")
        client_secret = (
            created_app.get("credentials", {}).get("oauthClient", {}).get("client_secret")
        )

        # Assign application to groups
        for group_name in group_names:
            groups_response = await client.get(
                f"{base_url}/groups",
                headers=headers,
                params={"q": group_name},
            )
            groups_response.raise_for_status()

            for group in groups_response.json():
                if group.get("profile", {}).get("name") == group_name:
                    await client.put(
                        f"{base_url}/apps/{created_app['id']}/groups/{group['id']}",
                        headers=headers,
                    )
                    break

    logger.info(f"Created Okta OAuth2 application: {client_id_name}")
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "groups": group_names,
        "okta_app_id": created_app.get("id"),  # Include Okta app ID
    }


async def update_okta_user_groups(
    username_or_id: str,
    groups: list[str],
) -> dict[str, Any]:
    """Update group memberships for an Okta user.

    Replaces the user's current group memberships with the specified groups.

    Args:
        username_or_id: Username (login) or user ID
        groups: List of group names to assign

    Returns:
        Dictionary with updated user info
    """
    base_url = _get_base_url()
    headers = _get_api_headers()

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Resolve user ID
        if "@" in username_or_id or "." in username_or_id:
            response = await client.get(
                f"{base_url}/users/{username_or_id}",
                headers=headers,
            )
            if response.status_code == 200:
                user_id = response.json().get("id")
            else:
                raise ValueError(f"User not found: {username_or_id}")
        else:
            user_id = username_or_id

        # Get current groups
        current_groups_resp = await client.get(
            f"{base_url}/users/{user_id}/groups",
            headers=headers,
        )
        current_groups_resp.raise_for_status()
        current_groups = {
            g.get("profile", {}).get("name"): g.get("id")
            for g in current_groups_resp.json()
            if g.get("type") == "OKTA_GROUP"
        }

        # Resolve target group names to IDs
        all_groups_resp = await client.get(
            f"{base_url}/groups",
            headers=headers,
            params={"limit": 200},
        )
        all_groups_resp.raise_for_status()
        all_groups = {g.get("profile", {}).get("name"): g.get("id") for g in all_groups_resp.json()}

        target_names = set(groups)

        # Remove from groups not in target
        for name, gid in current_groups.items():
            if name not in target_names:
                await client.delete(
                    f"{base_url}/groups/{gid}/users/{user_id}",
                    headers=headers,
                )

        # Add to groups in target but not current
        for name in target_names:
            if name not in current_groups and name in all_groups:
                await client.put(
                    f"{base_url}/groups/{all_groups[name]}/users/{user_id}",
                    headers=headers,
                )

    logger.info(f"Updated groups for Okta user {username_or_id}: {groups}")
    return {"username": username_or_id, "groups": groups}


async def update_okta_group(
    group_name_or_id: str,
    description: str = "",
) -> dict[str, Any]:
    """Update a group's properties in Okta.

    Args:
        group_name_or_id: Group name or ID
        description: New description for the group

    Returns:
        Dictionary with updated group info

    Raises:
        ValueError: If group not found
    """
    base_url = _get_base_url()
    headers = _get_api_headers()

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Resolve group ID if needed
        if "-" not in group_name_or_id or len(group_name_or_id) < 20:
            response = await client.get(
                f"{base_url}/groups",
                headers=headers,
                params={"q": group_name_or_id},
            )
            response.raise_for_status()
            matched = [
                g for g in response.json() if g.get("profile", {}).get("name") == group_name_or_id
            ]
            if not matched:
                raise ValueError(f"Group not found: {group_name_or_id}")
            group_id = matched[0].get("id")
            group_name = group_name_or_id
        else:
            group_id = group_name_or_id
            group_name = group_name_or_id

        update_resp = await client.put(
            f"{base_url}/groups/{group_id}",
            headers=headers,
            json={"profile": {"name": group_name, "description": description}},
        )
        update_resp.raise_for_status()

    logger.info(f"Updated Okta group: {group_name_or_id}")
    return {"name": group_name, "description": description}
