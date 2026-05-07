"""
IAM Manager factory for multi-provider support.

This module provides a unified interface for IAM operations across
different identity providers (Keycloak, Entra ID, Okta, Auth0).
"""

import logging
import os
import re
from typing import (
    Any,
    Protocol,
    runtime_checkable,
)

from .iam_errors import wrap_idp_admin_error

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)

AUTH_PROVIDER: str = os.environ.get("AUTH_PROVIDER", "keycloak")

# IdP group filtering -- applies to all identity providers
IDP_GROUP_FILTER_PREFIX: str = os.environ.get("IDP_GROUP_FILTER_PREFIX", "")

# Parse comma-separated prefixes and validate each one to prevent injection
IDP_GROUP_FILTER_PREFIXES: list[str] = []
if IDP_GROUP_FILTER_PREFIX:
    IDP_GROUP_FILTER_PREFIXES = [p.strip() for p in IDP_GROUP_FILTER_PREFIX.split(",") if p.strip()]
    for _prefix in IDP_GROUP_FILTER_PREFIXES:
        if not re.match(r"^[a-zA-Z0-9\-_ ]+$", _prefix):
            raise ValueError(
                f"IDP_GROUP_FILTER_PREFIX contains invalid characters in "
                f"prefix '{_prefix}'. "
                f"Only alphanumeric, hyphens, underscores, and spaces are allowed."
            )
    logger.info("IdP group filter prefixes: %s", IDP_GROUP_FILTER_PREFIXES)


def _filter_groups_by_prefix(
    groups: list[dict[str, Any]],
    prefixes: list[str],
) -> list[dict[str, Any]]:
    """
    Filter groups by display name prefix (client-side fallback).

    Used when the IdP API does not support server-side prefix filtering.

    Args:
        groups: List of group dictionaries with a 'name' key
        prefixes: List of allowed prefixes

    Returns:
        Filtered list of groups whose name starts with any prefix
    """
    if not prefixes:
        return groups

    return [g for g in groups if any(g.get("name", "").startswith(prefix) for prefix in prefixes)]


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
        """List groups from Keycloak, filtered by IDP_GROUP_FILTER_PREFIX if set."""
        from .keycloak_manager import list_keycloak_groups

        groups = await list_keycloak_groups()
        return _filter_groups_by_prefix(groups, IDP_GROUP_FILTER_PREFIXES)

    async def create_group(self, group_name: str, description: str = "") -> dict[str, Any]:
        """Create a group in Keycloak."""
        from .keycloak_manager import create_keycloak_group

        return await create_keycloak_group(group_name=group_name, description=description)

    async def delete_group(self, group_name: str) -> bool:
        """Delete a group from Keycloak."""
        from .keycloak_manager import delete_keycloak_group

        try:
            return await delete_keycloak_group(group_name=group_name)
        except Exception as exc:
            typed = wrap_idp_admin_error(exc)
            if typed is not exc:
                raise typed from exc
            raise

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

        try:
            return await update_keycloak_group(
                group_name=group_name,
                description=description,
            )
        except Exception as exc:
            typed = wrap_idp_admin_error(exc)
            if typed is not exc:
                raise typed from exc
            raise


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

        try:
            return await delete_entra_group(group_name_or_id=group_name)
        except Exception as exc:
            typed = wrap_idp_admin_error(exc)
            if typed is not exc:
                raise typed from exc
            raise

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

        try:
            return await update_entra_group(
                group_name_or_id=group_name,
                description=description,
            )
        except Exception as exc:
            typed = wrap_idp_admin_error(exc)
            if typed is not exc:
                raise typed from exc
            raise


class OktaIAMManager:
    """Okta IAM manager implementation."""

    async def list_users(
        self, search: str | None = None, max_results: int = 500, include_groups: bool = True
    ) -> list[dict[str, Any]]:
        """List users from Okta."""
        from .okta_manager import list_okta_users

        return await list_okta_users(
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
        """Create a human user in Okta."""
        from .okta_manager import create_okta_human_user

        return await create_okta_human_user(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name,
            groups=groups,
            password=password,
        )

    async def delete_user(self, username: str) -> bool:
        """Delete a user from Okta."""
        from .okta_manager import delete_okta_user

        return await delete_okta_user(username_or_id=username)

    async def list_groups(self) -> list[dict[str, Any]]:
        """List groups from Okta, filtered by IDP_GROUP_FILTER_PREFIX if set."""
        from .okta_manager import list_okta_groups

        groups = await list_okta_groups()
        return _filter_groups_by_prefix(groups, IDP_GROUP_FILTER_PREFIXES)

    async def create_group(self, group_name: str, description: str = "") -> dict[str, Any]:
        """Create a group in Okta."""
        from .okta_manager import create_okta_group

        return await create_okta_group(group_name=group_name, description=description)

    async def delete_group(self, group_name: str) -> bool:
        """Delete a group from Okta."""
        from .okta_manager import delete_okta_group

        try:
            return await delete_okta_group(group_name_or_id=group_name)
        except Exception as exc:
            typed = wrap_idp_admin_error(exc)
            if typed is not exc:
                raise typed from exc
            raise

    async def group_exists(self, group_name: str) -> bool:
        """Check if a group exists in Okta."""
        from .okta_manager import list_okta_groups

        try:
            groups = await list_okta_groups()
            return any(g.get("name", "").lower() == group_name.lower() for g in groups)
        except Exception:
            return False

    async def create_service_account(
        self, client_id: str, groups: list[str], description: str | None = None
    ) -> dict[str, Any]:
        """Create an OAuth2 service application in Okta."""
        from .okta_manager import create_okta_service_account

        return await create_okta_service_account(
            client_id_name=client_id, group_names=groups, description=description
        )

    async def update_user_groups(self, username: str, groups: list[str]) -> dict[str, Any]:
        """Update group memberships for an Okta user."""
        from .okta_manager import update_okta_user_groups

        return await update_okta_user_groups(username_or_id=username, groups=groups)

    async def update_group(
        self,
        group_name: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Update a group's properties in Okta."""
        from .okta_manager import update_okta_group

        try:
            return await update_okta_group(
                group_name_or_id=group_name,
                description=description,
            )
        except Exception as exc:
            typed = wrap_idp_admin_error(exc)
            if typed is not exc:
                raise typed from exc
            raise


class Auth0IAMManager:
    """Auth0 IAM manager implementation."""

    async def list_users(
        self, search: str | None = None, max_results: int = 500, include_groups: bool = True
    ) -> list[dict[str, Any]]:
        """List users from Auth0."""
        from .auth0_manager import list_auth0_users

        return await list_auth0_users(
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
        """Create a human user in Auth0."""
        from .auth0_manager import create_auth0_human_user

        return await create_auth0_human_user(
            username=username,
            email=email,
            first_name=first_name,
            last_name=last_name,
            groups=groups,
            password=password,
        )

    async def delete_user(self, username: str) -> bool:
        """Delete a user from Auth0."""
        from .auth0_manager import delete_auth0_user

        return await delete_auth0_user(username_or_id=username)

    async def list_groups(self) -> list[dict[str, Any]]:
        """List roles (groups) from Auth0, filtered by IDP_GROUP_FILTER_PREFIX if set."""
        from .auth0_manager import list_auth0_groups

        groups = await list_auth0_groups()
        return _filter_groups_by_prefix(groups, IDP_GROUP_FILTER_PREFIXES)

    async def create_group(self, group_name: str, description: str = "") -> dict[str, Any]:
        """Create a role (group) in Auth0."""
        from .auth0_manager import create_auth0_group

        return await create_auth0_group(group_name=group_name, description=description)

    async def delete_group(self, group_name: str) -> bool:
        """Delete a role (group) from Auth0."""
        from .auth0_manager import delete_auth0_group

        try:
            return await delete_auth0_group(group_name_or_id=group_name)
        except Exception as exc:
            typed = wrap_idp_admin_error(exc)
            if typed is not exc:
                raise typed from exc
            raise

    async def group_exists(self, group_name: str) -> bool:
        """Check if a role (group) exists in Auth0."""
        from .auth0_manager import list_auth0_groups

        try:
            groups = await list_auth0_groups()
            return any(g.get("name", "").lower() == group_name.lower() for g in groups)
        except Exception:
            return False

    async def create_service_account(
        self, client_id: str, groups: list[str], description: str | None = None
    ) -> dict[str, Any]:
        """Create an M2M application (service account) in Auth0."""
        from .auth0_manager import create_auth0_service_account

        return await create_auth0_service_account(
            client_id_name=client_id, group_names=groups, description=description
        )

    async def update_user_groups(self, username: str, groups: list[str]) -> dict[str, Any]:
        """Update role (group) memberships for an Auth0 user."""
        from .auth0_manager import update_auth0_user_groups

        return await update_auth0_user_groups(username_or_id=username, groups=groups)

    async def update_group(
        self,
        group_name: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Update a role's (group's) properties in Auth0."""
        from .auth0_manager import update_auth0_group

        try:
            return await update_auth0_group(
                group_name_or_id=group_name,
                description=description,
            )
        except Exception as exc:
            typed = wrap_idp_admin_error(exc)
            if typed is not exc:
                raise typed from exc
            raise


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

    elif provider == "okta":
        logger.debug("Using Okta IAM manager")
        return OktaIAMManager()

    elif provider == "auth0":
        logger.debug("Using Auth0 IAM manager")
        return Auth0IAMManager()

    else:
        logger.warning(f"Unknown AUTH_PROVIDER '{provider}', defaulting to Keycloak")
        return KeycloakIAMManager()
