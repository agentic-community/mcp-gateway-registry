"""Auth0 Management API manager for user and role operations.

This module provides async functions for managing users and roles
in Auth0 using the Auth0 Management API.

Note: Auth0 uses "roles" terminology, but we map them to "groups"
for consistency with the MCP Gateway IAM interface.
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
AUTH0_DOMAIN: str = os.environ.get("AUTH0_DOMAIN", "")
AUTH0_M2M_CLIENT_ID: str = os.environ.get("AUTH0_M2M_CLIENT_ID", "")
AUTH0_M2M_CLIENT_SECRET: str = os.environ.get("AUTH0_M2M_CLIENT_SECRET", "")
AUTH0_MANAGEMENT_API_TOKEN: str = os.environ.get("AUTH0_MANAGEMENT_API_TOKEN", "")


async def _get_management_api_token() -> str:
    """Get Auth0 Management API access token using M2M credentials.

    Returns:
        Access token for Management API

    Raises:
        ValueError: If credentials are not configured or token request fails
    """
    # If static management API token is provided, use it
    if AUTH0_MANAGEMENT_API_TOKEN:
        return AUTH0_MANAGEMENT_API_TOKEN

    # Otherwise, get token using M2M client credentials
    if not AUTH0_M2M_CLIENT_ID or not AUTH0_M2M_CLIENT_SECRET:
        raise ValueError(
            "Auth0 Management API access not configured. "
            "Set AUTH0_M2M_CLIENT_ID and AUTH0_M2M_CLIENT_SECRET, "
            "or AUTH0_MANAGEMENT_API_TOKEN environment variables."
        )

    domain = AUTH0_DOMAIN.replace("https://", "").rstrip("/")
    token_url = f"https://{domain}/oauth/token"

    token_data = {
        "client_id": AUTH0_M2M_CLIENT_ID,
        "client_secret": AUTH0_M2M_CLIENT_SECRET,
        "audience": f"https://{domain}/api/v2/",
        "grant_type": "client_credentials",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(token_url, json=token_data)
        if response.status_code != 200:
            error_msg = f"Failed to get Auth0 Management API token: {response.text}"
            logger.error(error_msg)
            raise ValueError(error_msg)

        token_response = response.json()
        return token_response.get("access_token", "")


async def _get_api_headers() -> dict[str, str]:
    """Get headers for Auth0 Management API requests."""
    token = await _get_management_api_token()
    return {
        "Authorization": f"Bearer {token}",
        "Accept": "application/json",
        "Content-Type": "application/json",
    }


def _get_base_url() -> str:
    """Get Auth0 Management API base URL."""
    domain = AUTH0_DOMAIN.replace("https://", "").rstrip("/")
    return f"https://{domain}/api/v2"


def _check_rate_limit(response: httpx.Response) -> None:
    """Check for Auth0 rate limiting and raise appropriate error.

    Args:
        response: HTTP response to check

    Raises:
        ValueError: If rate limited, includes retry delay info
    """
    if response.status_code == 429:
        retry_after = int(response.headers.get("Retry-After", 60))
        rate_limit_remaining = response.headers.get("X-RateLimit-Remaining", "0")
        logger.warning(
            f"Auth0 rate limit exceeded. "
            f"Remaining: {rate_limit_remaining}, Retry after: {retry_after}s"
        )
        raise ValueError(
            f"Auth0 API rate limited. Retry after {retry_after} seconds. "
            f"Consider reducing request frequency."
        )


async def list_auth0_users(
    search: str | None = None,
    max_results: int = 500,
    include_groups: bool = True,
) -> list[dict[str, Any]]:
    """List users from Auth0.

    Args:
        search: Optional search filter (email or username)
        max_results: Maximum number of results to return
        include_groups: Whether to include role (group) memberships

    Returns:
        List of user dictionaries
    """
    base_url = _get_base_url()
    headers = await _get_api_headers()

    params: dict[str, Any] = {"per_page": min(max_results, 100), "page": 0}
    if search:
        params["q"] = f'email:"{search}*" OR username:"{search}*"'
        params["search_engine"] = "v3"

    users: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        while len(users) < max_results:
            response = await client.get(f"{base_url}/users", headers=headers, params=params)
            _check_rate_limit(response)
            response.raise_for_status()

            page_users = response.json()
            if not page_users:
                break

            users.extend(page_users)
            params["page"] += 1

        # Transform to common format
        result = []
        for user in users[:max_results]:
            user_data: dict[str, Any] = {
                "id": user.get("user_id"),
                "username": user.get("username") or user.get("email", "").split("@")[0],
                "email": user.get("email"),
                "first_name": user.get("given_name", ""),
                "last_name": user.get("family_name", ""),
                "status": "active" if not user.get("blocked") else "blocked",
                "created": user.get("created_at"),
                "groups": [],
            }

            if include_groups:
                # Get user's roles (which we map to groups)
                roles_url = f"{base_url}/users/{user['user_id']}/roles"
                roles_response = await client.get(roles_url, headers=headers)
                if roles_response.status_code == 200:
                    user_data["groups"] = [r.get("name") for r in roles_response.json()]

            result.append(user_data)

    logger.info(f"Retrieved {len(result)} users from Auth0")
    return result


async def create_auth0_human_user(
    username: str,
    email: str,
    first_name: str,
    last_name: str,
    groups: list[str],
    password: str | None = None,
) -> dict[str, Any]:
    """Create a human user in Auth0.

    Args:
        username: Username for the account
        email: Email address
        first_name: First name
        last_name: Last name
        groups: List of role names to assign (mapped to groups terminology)
        password: Optional initial password

    Returns:
        Dictionary with created user details
    """
    base_url = _get_base_url()
    headers = await _get_api_headers()

    user_data: dict[str, Any] = {
        "email": email,
        "given_name": first_name,
        "family_name": last_name,
        "name": f"{first_name} {last_name}",
        "connection": "Username-Password-Authentication",  # Auth0 default database connection
        "email_verified": False,
    }

    if password:
        user_data["password"] = password

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{base_url}/users",
            headers=headers,
            json=user_data,
        )
        if response.status_code >= 400:
            try:
                error_body = response.json()
            except Exception:
                error_body = response.text
            logger.error(f"Auth0 user creation failed ({response.status_code}): {error_body}")
            raise ValueError(f"Auth0 user creation failed: {error_body}")
        created_user = response.json()
        user_id = created_user.get("user_id")

        # Assign to roles (groups)
        if groups:
            # Get all roles to find IDs
            roles_response = await client.get(f"{base_url}/roles", headers=headers)
            roles_response.raise_for_status()
            all_roles = {r.get("name"): r.get("id") for r in roles_response.json()}

            # Assign user to matching roles
            role_ids = [all_roles[group] for group in groups if group in all_roles]
            if role_ids:
                await client.post(
                    f"{base_url}/users/{user_id}/roles",
                    headers=headers,
                    json={"roles": role_ids},
                )

    logger.info(f"Created Auth0 user: {username}")
    return {
        "id": user_id,
        "username": username,
        "email": email,
        "groups": groups,
    }


async def delete_auth0_user(username_or_id: str) -> bool:
    """Delete a user from Auth0.

    Args:
        username_or_id: Username (email) or user ID

    Returns:
        True if successful

    Raises:
        ValueError: If user not found
    """
    base_url = _get_base_url()
    headers = await _get_api_headers()

    async with httpx.AsyncClient(timeout=10.0) as client:
        # If it looks like an email, search for user
        if "@" in username_or_id:
            response = await client.get(
                f"{base_url}/users-by-email",
                headers=headers,
                params={"email": username_or_id},
            )
            if response.status_code == 200:
                users = response.json()
                if users:
                    user_id = users[0].get("user_id")
                else:
                    raise ValueError(f"User not found: {username_or_id}")
            else:
                raise ValueError(f"User not found: {username_or_id}")
        else:
            user_id = username_or_id

        # Delete user
        delete_response = await client.delete(
            f"{base_url}/users/{user_id}",
            headers=headers,
        )
        delete_response.raise_for_status()

    logger.info(f"Deleted Auth0 user: {username_or_id}")
    return True


async def list_auth0_groups() -> list[dict[str, Any]]:
    """List all roles from Auth0 (mapped to groups terminology).

    Returns:
        List of role dictionaries with id, name, description
    """
    base_url = _get_base_url()
    headers = await _get_api_headers()

    roles: list[dict[str, Any]] = []
    async with httpx.AsyncClient(timeout=10.0) as client:
        params: dict[str, Any] = {"per_page": 100, "page": 0}

        while True:
            response = await client.get(f"{base_url}/roles", headers=headers, params=params)
            response.raise_for_status()

            page_roles = response.json()
            if not page_roles:
                break

            roles.extend(page_roles)
            params["page"] += 1

    result = [
        {
            "id": r.get("id"),
            "name": r.get("name"),
            "description": r.get("description", ""),
            "type": "AUTH0_ROLE",
            "path": f"/{r.get('name')}",
        }
        for r in roles
    ]

    logger.info(f"Retrieved {len(result)} roles (groups) from Auth0")
    return result


async def create_auth0_group(
    group_name: str,
    description: str = "",
) -> dict[str, Any]:
    """Create a role in Auth0 (mapped to group terminology).

    Args:
        group_name: Name of the role
        description: Optional description

    Returns:
        Dictionary with created role details
    """
    base_url = _get_base_url()
    headers = await _get_api_headers()

    role_data = {
        "name": group_name,
        "description": description,
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{base_url}/roles",
            headers=headers,
            json=role_data,
        )
        response.raise_for_status()
        created_role = response.json()

    logger.info(f"Created Auth0 role (group): {group_name}")
    return {
        "id": created_role.get("id"),
        "name": group_name,
        "description": description,
    }


async def delete_auth0_group(group_name_or_id: str) -> bool:
    """Delete a role from Auth0 by name or ID.

    Args:
        group_name_or_id: Role name or ID

    Returns:
        True if successful

    Raises:
        ValueError: If role not found
    """
    base_url = _get_base_url()
    headers = await _get_api_headers()

    async with httpx.AsyncClient(timeout=10.0) as client:
        # If not an ID format, search by name
        if not group_name_or_id.startswith("rol_"):
            response = await client.get(
                f"{base_url}/roles",
                headers=headers,
                params={"name_filter": group_name_or_id},
            )
            response.raise_for_status()
            roles = response.json()

            role_id = None
            for r in roles:
                if r.get("name") == group_name_or_id:
                    role_id = r.get("id")
                    break

            if not role_id:
                raise ValueError(f"Role (group) not found: {group_name_or_id}")
        else:
            role_id = group_name_or_id

        delete_response = await client.delete(
            f"{base_url}/roles/{role_id}",
            headers=headers,
        )
        delete_response.raise_for_status()

    logger.info(f"Deleted Auth0 role (group): {group_name_or_id}")
    return True


async def create_auth0_service_account(
    client_id_name: str,
    group_names: list[str],
    description: str | None = None,
) -> dict[str, Any]:
    """Create an M2M application (service account) in Auth0.

    Creates an M2M application with client_credentials grant type.
    Note: Auth0 M2M applications don't directly have roles - roles are
    assigned to users, not applications.

    Args:
        client_id_name: Name for the M2M application
        group_names: List of role names (for documentation - not directly assigned)
        description: Optional description

    Returns:
        Dictionary with client_id and client_secret
    """
    base_url = _get_base_url()
    headers = await _get_api_headers()

    app_data = {
        "name": client_id_name,
        "description": description or f"M2M service account for {client_id_name}",
        "app_type": "non_interactive",  # M2M application
        "grant_types": ["client_credentials"],
        "token_endpoint_auth_method": "client_secret_post",
    }

    async with httpx.AsyncClient(timeout=10.0) as client:
        response = await client.post(
            f"{base_url}/clients",
            headers=headers,
            json=app_data,
        )
        response.raise_for_status()
        created_app = response.json()

        client_id = created_app.get("client_id")
        client_secret = created_app.get("client_secret")

    logger.info(f"Created Auth0 M2M application: {client_id_name}")
    logger.warning(
        f"Auth0 M2M applications don't have roles. "
        f"Configure API permissions in Auth0 dashboard for {client_id_name}."
    )
    return {
        "client_id": client_id,
        "client_secret": client_secret,
        "groups": group_names,
        "auth0_client_id": client_id,
    }


async def update_auth0_user_groups(
    username_or_id: str,
    groups: list[str],
) -> dict[str, Any]:
    """Update role memberships for an Auth0 user.

    Replaces the user's current role (group) memberships with the specified roles.

    Args:
        username_or_id: Username (email) or user ID
        groups: List of role names to assign

    Returns:
        Dictionary with updated user info
    """
    base_url = _get_base_url()
    headers = await _get_api_headers()

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Resolve user ID
        if "@" in username_or_id:
            response = await client.get(
                f"{base_url}/users-by-email",
                headers=headers,
                params={"email": username_or_id},
            )
            if response.status_code == 200:
                users = response.json()
                if users:
                    user_id = users[0].get("user_id")
                else:
                    raise ValueError(f"User not found: {username_or_id}")
            else:
                raise ValueError(f"User not found: {username_or_id}")
        else:
            user_id = username_or_id

        # Get current roles
        current_roles_resp = await client.get(
            f"{base_url}/users/{user_id}/roles",
            headers=headers,
        )
        current_roles_resp.raise_for_status()
        current_role_ids = [r.get("id") for r in current_roles_resp.json()]

        # Get all available roles
        all_roles_resp = await client.get(
            f"{base_url}/roles",
            headers=headers,
        )
        all_roles_resp.raise_for_status()
        all_roles = {r.get("name"): r.get("id") for r in all_roles_resp.json()}

        target_role_ids = [all_roles[group] for group in groups if group in all_roles]

        # Remove current roles
        if current_role_ids:
            await client.delete(
                f"{base_url}/users/{user_id}/roles",
                headers=headers,
                json={"roles": current_role_ids},
            )

        # Add target roles
        if target_role_ids:
            await client.post(
                f"{base_url}/users/{user_id}/roles",
                headers=headers,
                json={"roles": target_role_ids},
            )

    logger.info(f"Updated roles (groups) for Auth0 user {username_or_id}: {groups}")
    return {"username": username_or_id, "groups": groups}


async def update_auth0_group(
    group_name_or_id: str,
    description: str = "",
) -> dict[str, Any]:
    """Update a role's properties in Auth0.

    Args:
        group_name_or_id: Role name or ID
        description: New description for the role

    Returns:
        Dictionary with updated role info

    Raises:
        ValueError: If role not found
    """
    base_url = _get_base_url()
    headers = await _get_api_headers()

    async with httpx.AsyncClient(timeout=10.0) as client:
        # Resolve role ID if needed
        if not group_name_or_id.startswith("rol_"):
            response = await client.get(
                f"{base_url}/roles",
                headers=headers,
                params={"name_filter": group_name_or_id},
            )
            response.raise_for_status()
            matched = [r for r in response.json() if r.get("name") == group_name_or_id]
            if not matched:
                raise ValueError(f"Role (group) not found: {group_name_or_id}")
            role_id = matched[0].get("id")
            role_name = group_name_or_id
        else:
            role_id = group_name_or_id
            # Get current role name
            role_resp = await client.get(f"{base_url}/roles/{role_id}", headers=headers)
            role_resp.raise_for_status()
            role_name = role_resp.json().get("name")

        update_resp = await client.patch(
            f"{base_url}/roles/{role_id}",
            headers=headers,
            json={"description": description},
        )
        update_resp.raise_for_status()

    logger.info(f"Updated Auth0 role (group): {group_name_or_id}")
    return {"name": role_name, "description": description}
