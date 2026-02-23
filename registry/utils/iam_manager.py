"""
IAM Manager factory for multi-provider support.

This module provides a unified interface for IAM operations across
different identity providers (Keycloak, Entra ID).
"""

import logging
import os
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

AUTH_PROVIDER: str = os.environ.get("AUTH_PROVIDER", "keycloak")


@runtime_checkable
class IAMManager(Protocol):
    """Protocol defining the IAM manager interface."""

    async def list_users(
        self, search: str | None = None, max_results: int = 500, include_groups: bool = True
    ) -> list[dict[str, Any]]:
        """
        List users from the identity provider.

        Args:
            search: Optional search filter
            max_results: Maximum number of results to return
            include_groups: Whether to include group memberships

        Returns:
            List of user dictionaries
        """
        ...

    async def create_human_user(
        self,
        username: str,
        email: str,
        first_name: str,
        last_name: str,
        groups: list[str],
        password: str | None = None,
    ) -> dict[str, Any]:
        """
        Create a human user account.

        Args:
            username: Username for the account
            email: Email address
            first_name: First name
            last_name: Last name
            groups: List of group names to assign
            password: Optional initial password

        Returns:
            User dictionary with created user details
        """
        ...

    async def delete_user(self, username: str) -> bool:
        """
        Delete a user by username.

        Args:
            username: Username or identifier of the user to delete

        Returns:
            True if successful
        """
        ...

    async def list_groups(self) -> list[dict[str, Any]]:
        """
        List all groups from the identity provider.

        Returns:
            List of group dictionaries
        """
        ...

    async def create_group(self, group_name: str, description: str = "") -> dict[str, Any]:
        """
        Create a group in the identity provider.

        Args:
            group_name: Name of the group to create
            description: Optional description

        Returns:
            Group dictionary with created group details
        """
        ...

    async def delete_group(self, group_name: str) -> bool:
        """
        Delete a group from the identity provider.

        Args:
            group_name: Name or identifier of the group to delete

        Returns:
            True if successful
        """
        ...

    async def group_exists(self, group_name: str) -> bool:
        """
        Check if a group exists in the identity provider.

        Args:
            group_name: Name of the group to check

        Returns:
            True if group exists, False otherwise
        """
        ...

    async def create_service_account(
        self, client_id: str, groups: list[str], description: str | None = None
    ) -> dict[str, Any]:
        """
        Create a service account (M2M) in the identity provider.

        Args:
            client_id: Client ID for the service account
            groups: List of group names to assign
            description: Optional description

        Returns:
            Dictionary with client_id, client_secret, and groups
        """
        ...

    async def update_user_groups(self, username: str, groups: list[str]) -> dict[str, Any]:
        """
        Update group memberships for a user or service account.

        Args:
            username: Username or client ID of the user/service account
            groups: List of group names the user should belong to

        Returns:
            Dictionary with username, groups, added, and removed lists
        """
        ...

    async def update_group(
        self,
        group_name: str,
        description: str = "",
    ) -> dict[str, Any]:
        """
        Update a group's properties in the identity provider.

        Args:
            group_name: Name of the group to update
            description: New description for the group

        Returns:
            Dictionary with updated group details (id, name, path, attributes)
        """
        ...


class KeycloakIAMManager:
    """Keycloak IAM manager implementation."""

    async def list_users(
        self, search: str | None = None, max_results: int = 500, include_groups: bool = True
    ) -> list[dict[str, Any]]:
        """List users from Keycloak."""
        from .keycloak_manager import list_keycloak_users

        return await list_keycloak_users(
            search=search, max_results=max_results, include_groups=include_groups
        )

    async def create_human_user(
        self,
        username: str,
        email: str,
        first_name: str,
        last_name: str,
        groups: list[str],
        password: str | None = None,
    ) -> dict[str, Any]:
        """Create a human user in Keycloak."""
        from .keycloak_manager import create_human_user_account

        return await create_human_user_account(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name,
            groups=groups,
            password=password,
        )

    async def delete_user(self, username: str) -> bool:
        """Delete a user from Keycloak."""
        from .keycloak_manager import delete_keycloak_user

        return await delete_keycloak_user(username=username)

    async def list_groups(self) -> list[dict[str, Any]]:
        """List all groups from Keycloak."""
        from .keycloak_manager import list_keycloak_groups

        return await list_keycloak_groups()

    async def create_group(self, group_name: str, description: str = "") -> dict[str, Any]:
        """Create a group in Keycloak."""
        from .keycloak_manager import create_keycloak_group

        return await create_keycloak_group(group_name=group_name, description=description)

    async def delete_group(self, group_name: str) -> bool:
        """Delete a group from Keycloak."""
        from .keycloak_manager import delete_keycloak_group

        return await delete_keycloak_group(group_name=group_name)

    async def group_exists(self, group_name: str) -> bool:
        """Check if a group exists in Keycloak."""
        from .keycloak_manager import group_exists_in_keycloak

        return await group_exists_in_keycloak(group_name)

    async def create_service_account(
        self, client_id: str, groups: list[str], description: str | None = None
    ) -> dict[str, Any]:
        """Create a service account client in Keycloak."""
        from .keycloak_manager import create_service_account_client

        return await create_service_account_client(
            client_id=client_id, group_names=groups, description=description
        )

    async def update_user_groups(self, username: str, groups: list[str]) -> dict[str, Any]:
        """Update group memberships for a Keycloak user or service account."""
        from .keycloak_manager import update_keycloak_user_groups

        return await update_keycloak_user_groups(username=username, groups=groups)

    async def update_group(
        self,
        group_name: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Update a group's properties in Keycloak."""
        from .keycloak_manager import update_keycloak_group

        return await update_keycloak_group(group_name=group_name, description=description)


class EntraIAMManager:
    """Entra ID IAM manager implementation."""

    async def list_users(
        self, search: str | None = None, max_results: int = 500, include_groups: bool = True
    ) -> list[dict[str, Any]]:
        """List users from Entra ID."""
        from .entra_manager import list_entra_users

        return await list_entra_users(
            search=search, max_results=max_results, include_groups=include_groups
        )

    async def create_human_user(
        self,
        username: str,
        email: str,
        first_name: str,
        last_name: str,
        groups: list[str],
        password: str | None = None,
    ) -> dict[str, Any]:
        """Create a human user in Entra ID."""
        from .entra_manager import create_entra_human_user

        return await create_entra_human_user(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name,
            groups=groups,
            password=password,
        )

    async def delete_user(self, username: str) -> bool:
        """Delete a user from Entra ID."""
        from .entra_manager import delete_entra_user

        return await delete_entra_user(username_or_id=username)

    async def list_groups(self) -> list[dict[str, Any]]:
        """List all groups from Entra ID."""
        from .entra_manager import list_entra_groups

        return await list_entra_groups()

    async def create_group(self, group_name: str, description: str = "") -> dict[str, Any]:
        """Create a group in Entra ID."""
        from .entra_manager import create_entra_group

        return await create_entra_group(group_name=group_name, description=description)

    async def delete_group(self, group_name: str) -> bool:
        """Delete a group from Entra ID."""
        from .entra_manager import delete_entra_group

        return await delete_entra_group(group_name_or_id=group_name)

    async def group_exists(self, group_name: str) -> bool:
        """Check if a group exists in Entra ID."""
        from .entra_manager import list_entra_groups

        try:
            groups = await list_entra_groups()
            return any(g.get("name", "").lower() == group_name.lower() for g in groups)
        except Exception:
            return False

    async def create_service_account(
        self, client_id: str, groups: list[str], description: str | None = None
    ) -> dict[str, Any]:
        """Create a service principal (app registration) in Entra ID."""
        from .entra_manager import create_service_principal_client

        return await create_service_principal_client(
            client_id_name=client_id, group_names=groups, description=description
        )

    async def update_user_groups(self, username: str, groups: list[str]) -> dict[str, Any]:
        """Update group memberships for an Entra ID user or service principal."""
        from .entra_manager import update_entra_user_groups

        return await update_entra_user_groups(username_or_id=username, groups=groups)

    async def update_group(
        self,
        group_name: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Update a group's properties in Entra ID."""
        from .entra_manager import update_entra_group

        return await update_entra_group(group_name_or_id=group_name, description=description)


def get_iam_manager() -> IAMManager:
    """
    Factory function to get the appropriate IAM manager based on AUTH_PROVIDER.

    Returns:
        IAMManager implementation for the configured provider
    """
    provider = AUTH_PROVIDER.lower()

    if provider == "keycloak":
        logger.debug("Using Keycloak IAM manager")
        return KeycloakIAMManager()

    elif provider == "entra":
        logger.debug("Using Entra ID IAM manager")
        return EntraIAMManager()

    else:
        logger.warning(f"Unknown AUTH_PROVIDER '{provider}', defaulting to Keycloak")
        return KeycloakIAMManager()
