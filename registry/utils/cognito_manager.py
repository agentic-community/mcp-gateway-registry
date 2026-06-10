"""
Amazon Cognito IAM helper functions.

Read-only group and user listing against a Cognito User Pool, used by the
CognitoIAMManager. Cognito's boto3 client is synchronous, so each call is run
in a worker thread to keep the public interface async and non-blocking.

Write operations (create/delete group, create user, etc.) are intentionally
not implemented here yet; see CognitoIAMManager for the supported surface.
"""

import asyncio
import logging
import os
from typing import Any

import boto3

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

# Cognito caps these list calls at 60 per page; paginate to get everything.
_MAX_PAGE_SIZE: int = 60


def _get_region() -> str:
    """Return the AWS region for the Cognito User Pool."""
    return os.environ.get("AWS_REGION", "us-east-1")


def _get_user_pool_id() -> str:
    """Return the configured Cognito User Pool ID.

    Raises:
        ValueError: If COGNITO_USER_POOL_ID is not set.
    """
    pool_id = os.environ.get("COGNITO_USER_POOL_ID")
    if not pool_id:
        raise ValueError("COGNITO_USER_POOL_ID is not set; cannot manage Cognito IAM")
    return pool_id


def _get_client() -> Any:
    """Create a boto3 cognito-idp client for the configured region."""
    return boto3.client("cognito-idp", region_name=_get_region())


def _extract_email(
    attributes: list[dict[str, str]],
) -> str | None:
    """Pull the email value out of a Cognito user's attribute list."""
    for attr in attributes:
        if attr.get("Name") == "email":
            return attr.get("Value")
    return None


def _list_groups_sync() -> list[dict[str, Any]]:
    """List all groups in the User Pool (synchronous, paginated)."""
    client = _get_client()
    pool_id = _get_user_pool_id()

    groups: list[dict[str, Any]] = []
    next_token: str | None = None

    while True:
        kwargs: dict[str, Any] = {"UserPoolId": pool_id, "Limit": _MAX_PAGE_SIZE}
        if next_token:
            kwargs["NextToken"] = next_token

        response = client.list_groups(**kwargs)
        for group in response.get("Groups", []):
            groups.append(
                {
                    "id": group.get("GroupName"),
                    "name": group.get("GroupName"),
                    "description": group.get("Description", ""),
                    "path": group.get("GroupName"),
                }
            )

        next_token = response.get("NextToken")
        if not next_token:
            break

    logger.info(f"Retrieved {len(groups)} groups from Cognito")
    return groups


def _list_users_sync(
    max_results: int,
    include_groups: bool,
) -> list[dict[str, Any]]:
    """List users in the User Pool (synchronous, paginated)."""
    client = _get_client()
    pool_id = _get_user_pool_id()

    users: list[dict[str, Any]] = []
    pagination_token: str | None = None

    while len(users) < max_results:
        kwargs: dict[str, Any] = {"UserPoolId": pool_id, "Limit": _MAX_PAGE_SIZE}
        if pagination_token:
            kwargs["PaginationToken"] = pagination_token

        response = client.list_users(**kwargs)
        for user in response.get("Users", []):
            attributes = user.get("Attributes", [])
            username = user.get("Username")
            user_data: dict[str, Any] = {
                "id": username,
                "username": username,
                "email": _extract_email(attributes),
                "status": user.get("UserStatus"),
                "enabled": user.get("Enabled", True),
                "groups": [],
            }

            if include_groups:
                user_data["groups"] = _list_groups_for_user_sync(client, pool_id, username)

            users.append(user_data)

        pagination_token = response.get("PaginationToken")
        if not pagination_token:
            break

    logger.info(f"Retrieved {len(users)} users from Cognito")
    return users[:max_results]


def _list_groups_for_user_sync(
    client: Any,
    pool_id: str,
    username: str,
) -> list[str]:
    """List the group names a single user belongs to (synchronous, paginated)."""
    group_names: list[str] = []
    next_token: str | None = None

    while True:
        kwargs: dict[str, Any] = {
            "UserPoolId": pool_id,
            "Username": username,
            "Limit": _MAX_PAGE_SIZE,
        }
        if next_token:
            kwargs["NextToken"] = next_token

        response = client.admin_list_groups_for_user(**kwargs)
        for group in response.get("Groups", []):
            group_names.append(group.get("GroupName"))

        next_token = response.get("NextToken")
        if not next_token:
            break

    return group_names


async def list_cognito_groups() -> list[dict[str, Any]]:
    """List all groups from the Cognito User Pool.

    Returns:
        List of group dictionaries with id, name, description, and path.
    """
    return await asyncio.to_thread(_list_groups_sync)


async def list_cognito_users(
    max_results: int = 500,
    include_groups: bool = True,
) -> list[dict[str, Any]]:
    """List users from the Cognito User Pool.

    Args:
        max_results: Maximum number of users to return.
        include_groups: Whether to look up each user's group memberships.

    Returns:
        List of user dictionaries.
    """
    return await asyncio.to_thread(_list_users_sync, max_results, include_groups)
