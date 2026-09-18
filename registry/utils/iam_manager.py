"""
IAM Manager factory for multi-provider support.

This module provides a unified interface for IAM operations across
different identity providers (Keycloak, Entra ID, Okta, Auth0, PingFederate,
Amazon Cognito, and Logto).
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


class PingFederateIAMManager:
    """PingFederate IAM manager implementation.

    Design note: PingFederate's admin API does not expose a first-class
    "groups" concept the way Keycloak/Entra/Okta do. In this deployment,
    groups live in the registry's MongoDB ``mcp_scopes_default`` collection
    and user-to-group mappings live in ``idp_user_groups`` (issue #1127).
    The Groups IAM tab's route at management_routes.py merges IdP-returned
    groups with MongoDB scope docs, so returning an empty list here lets
    the route fall through to the MongoDB-only view.

    User-side operations (list_users, create_human_user, update_user_groups)
    are routed through the User Groups IAM tab instead of the Users tab,
    and raise NotImplementedError if invoked directly so callers fail fast.
    """

    async def list_users(
        self, search: str | None = None, max_results: int = 500, include_groups: bool = True
    ) -> list[dict[str, Any]]:
        raise NotImplementedError(
            "PingFederate user listing is not implemented. "
            "Use the User Groups IAM tab to manage user-to-group mappings."
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
        raise NotImplementedError(
            "PingFederate human user creation is not implemented here. "
            "Use the User Groups IAM tab's 'Also create in PingFederate' checkbox instead."
        )

    async def delete_user(self, username: str) -> bool:
        raise NotImplementedError("PingFederate user deletion is not implemented.")

    async def list_groups(self) -> list[dict[str, Any]]:
        # Groups live in MongoDB mcp_scopes_default; the route merges them in.
        return []

    async def create_group(self, group_name: str, description: str = "") -> dict[str, Any]:
        # Groups live in MongoDB; the route's local-only path persists them.
        # Return a synthesized record so callers that pass create_in_idp=True
        # still get a valid response shape.
        return {
            "id": group_name,
            "name": group_name,
            "path": f"/{group_name}",
            "attributes": {"description": [description]} if description else None,
        }

    async def delete_group(self, group_name: str) -> bool:
        # No-op: deletion happens in MongoDB scopes only.
        return True

    async def group_exists(self, group_name: str) -> bool:
        # PF has no group concept here; defer to MongoDB scope checks upstream.
        return False

    async def create_service_account(
        self, client_id: str, groups: list[str], description: str | None = None
    ) -> dict[str, Any]:
        """Create an OAuth2 client_credentials client in PingFederate."""
        from .pingfederate_manager import create_pingfederate_service_account_client

        return await create_pingfederate_service_account_client(
            client_id=client_id, group_names=groups, description=description
        )

    async def update_user_groups(self, username: str, groups: list[str]) -> dict[str, Any]:
        raise NotImplementedError(
            "PingFederate user-group updates are not implemented here. "
            "Use the User Groups IAM tab to update user-to-group mappings."
        )

    async def update_group(
        self,
        group_name: str,
        description: str = "",
    ) -> dict[str, Any]:
        # No-op: updates happen on MongoDB scope docs.
        return {
            "id": group_name,
            "name": group_name,
            "path": f"/{group_name}",
            "attributes": {"description": [description]} if description else None,
        }


# Message shared by every write method that Cognito IAM management does not yet
# support from the UI. Group/user administration for Cognito is currently done
# out-of-band (AWS console or `aws cognito-idp ...`); see docs/idp/cognito.md.
_COGNITO_WRITE_UNSUPPORTED: str = (
    "Cognito IAM write operations are not supported from the registry UI yet. "
    "Manage Cognito groups and users via the AWS console or the AWS CLI "
    "(aws cognito-idp ...). See docs/idp/cognito.md."
)


class CognitoIAMManager:
    """Amazon Cognito IAM manager (read-only).

    Listing groups and users is implemented against the Cognito User Pool.
    Write operations are not yet supported and raise NotImplementedError with a
    clear message pointing operators to the AWS console / CLI.
    """

    async def list_users(
        self, search: str | None = None, max_results: int = 500, include_groups: bool = True
    ) -> list[dict[str, Any]]:
        """List users from the Cognito User Pool."""
        from .cognito_manager import list_cognito_users

        return await list_cognito_users(max_results=max_results, include_groups=include_groups)

    async def list_groups(self) -> list[dict[str, Any]]:
        """List groups from the Cognito User Pool, filtered by IDP_GROUP_FILTER_PREFIX if set."""
        from .cognito_manager import list_cognito_groups

        groups = await list_cognito_groups()
        return _filter_groups_by_prefix(groups, IDP_GROUP_FILTER_PREFIXES)

    async def group_exists(self, group_name: str) -> bool:
        """Check if a group exists in the Cognito User Pool."""
        from .cognito_manager import list_cognito_groups

        try:
            groups = await list_cognito_groups()
            return any(g.get("name", "").lower() == group_name.lower() for g in groups)
        except Exception:
            return False

    async def create_human_user(
        self,
        username: str,
        email: str,
        first_name: str,
        last_name: str,
        groups: list[str],
        password: str | None = None,
    ) -> dict[str, Any]:
        """Not supported yet for Cognito."""
        raise NotImplementedError(_COGNITO_WRITE_UNSUPPORTED)

    async def delete_user(self, username: str) -> bool:
        """Not supported yet for Cognito."""
        raise NotImplementedError(_COGNITO_WRITE_UNSUPPORTED)

    async def create_group(self, group_name: str, description: str = "") -> dict[str, Any]:
        """Not supported yet for Cognito."""
        raise NotImplementedError(_COGNITO_WRITE_UNSUPPORTED)

    async def delete_group(self, group_name: str) -> bool:
        """Not supported yet for Cognito."""
        raise NotImplementedError(_COGNITO_WRITE_UNSUPPORTED)

    async def create_service_account(
        self, client_id: str, groups: list[str], description: str | None = None
    ) -> dict[str, Any]:
        """Not supported yet for Cognito."""
        raise NotImplementedError(_COGNITO_WRITE_UNSUPPORTED)

    async def update_user_groups(self, username: str, groups: list[str]) -> dict[str, Any]:
        """Not supported yet for Cognito."""
        raise NotImplementedError(_COGNITO_WRITE_UNSUPPORTED)

    async def update_group(
        self,
        group_name: str,
        description: str = "",
    ) -> dict[str, Any]:
        """Not supported yet for Cognito."""
        raise NotImplementedError(_COGNITO_WRITE_UNSUPPORTED)


class LogtoIAMManager:
    """Logto IAM manager implementation.

    Group semantics: Logto *roles* are the IAM groups. The fork's Logto auth
    provider emits ``groups = claims["roles"]`` in user JWTs, so every group
    created here (a Logto role) automatically reaches user tokens, and the
    group→scope mapping consumes the same names. M2M-type roles appear too —
    the IAM group list is intentionally the full role list; use
    ``IDP_GROUP_FILTER_PREFIX`` to narrow it.

    Uses the Management API via an M2M application (client credentials with
    the ``Logto Management API access`` role); see ``logto_admin.py`` for the
    token recipe (``resource`` + ``scope=all`` are both mandatory).
    """

    def __init__(self, client=None):
        # Client injectable for tests; defaults to the process singleton.
        if client is None:
            from .logto_admin import get_logto_admin

            client = get_logto_admin()
        self._client = client

    # ── mapping helpers ─────────────────────────────────────────────────────

    @staticmethod
    def _role_to_group(role: dict[str, Any]) -> dict[str, Any]:
        """Map a Logto role onto the Keycloak-shaped group dict the routes expect."""
        name = str(role.get("name", ""))
        return {
            "id": str(role.get("id", "")),
            "name": name,
            "path": f"/{name}" if name else "",
            "attributes": {"description": [role.get("description") or ""]},
        }

    @staticmethod
    def _not_found(what: str) -> Exception:
        # Message carries "HTTP 404" so iam_errors.looks_not_found classifies it
        # and management routes translate it to a 404 instead of a 502.
        from .logto_admin import LogtoAdminError

        return LogtoAdminError(f"{what} not found in Logto (HTTP 404)", status_code=404)

    async def _find_role(self, role_name: str) -> dict[str, Any] | None:
        roles = await self._client.get_paged("/api/roles")
        return next((r for r in roles if r.get("name") == role_name), None)

    async def _find_user(self, username: str) -> dict[str, Any] | None:
        users = await self._client.get_paged(
            "/api/users", params={"search": username}, max_items=200
        )
        return next(
            (
                u
                for u in users
                if u.get("username") == username or u.get("primaryEmail") == username
            ),
            None,
        )

    async def _user_role_names(self, user_id: str) -> list[str]:
        roles = await self._client.get_paged(f"/api/users/{user_id}/roles")
        return [str(r.get("name", "")) for r in roles if r.get("name")]

    @staticmethod
    def _user_summary(user: dict[str, Any], groups: list[str]) -> dict[str, Any]:
        full_name = str(user.get("name") or "").strip()
        first, _, last = full_name.partition(" ")
        return {
            "id": str(user.get("id", "")),
            "username": user.get("username") or user.get("primaryEmail") or "",
            "email": user.get("primaryEmail"),
            "firstName": first or None,
            "lastName": last or None,
            "enabled": not user.get("isSuspended", False),
            "groups": groups,
        }

    async def _assign_user_roles(self, user_id: str, role_ids: list[str]) -> None:
        if role_ids:
            await self._client.post(f"/api/users/{user_id}/roles", {"roleIds": role_ids})

    # ── groups ───────────────────────────────────────────────────────────────

    async def list_groups(self) -> list[dict[str, Any]]:
        """List IAM groups (all Logto roles), filtered by IDP_GROUP_FILTER_PREFIX if set."""
        roles = await self._client.get_paged("/api/roles")
        groups = [self._role_to_group(r) for r in roles]
        return _filter_groups_by_prefix(groups, IDP_GROUP_FILTER_PREFIXES)

    async def create_group(self, group_name: str, description: str = "") -> dict[str, Any]:
        """Create a group (Logto role)."""
        role = await self._client.post("/api/roles", {"name": group_name, "description": description})
        return self._role_to_group(role)

    async def delete_group(self, group_name: str) -> bool:
        """Delete a group. Cascades in Logto: the role is removed from every holder."""
        role = await self._find_role(group_name)
        if role is None:
            raise self._not_found(f"Logto role '{group_name}'")
        await self._client.delete(f"/api/roles/{role['id']}")
        return True

    async def group_exists(self, group_name: str) -> bool:
        """Check whether a group (Logto role) exists."""
        return await self._find_role(group_name) is not None

    async def update_group(self, group_name: str, description: str = "") -> dict[str, Any]:
        """Update a group's description."""
        role = await self._find_role(group_name)
        if role is None:
            raise self._not_found(f"Logto role '{group_name}'")
        updated = await self._client.patch(f"/api/roles/{role['id']}", {"description": description})
        return self._role_to_group(updated or {**role, "description": description})

    # ── users ────────────────────────────────────────────────────────────────

    async def list_users(
        self, search: str | None = None, max_results: int = 500, include_groups: bool = True
    ) -> list[dict[str, Any]]:
        """List human users. M2M accounts are not IdP users here; the management
        route merges MongoDB-registered M2M clients into the user list itself."""
        params = {"search": search} if search else None
        users = await self._client.get_paged("/api/users", params=params, max_items=max_results)
        summaries = []
        for user in users:
            groups: list[str] = []
            if include_groups and user.get("id"):
                groups = await self._user_role_names(user["id"])
            summaries.append(self._user_summary(user, groups))
        return summaries

    async def create_human_user(
        self,
        username: str,
        email: str,
        first_name: str,
        last_name: str,
        groups: list[str],
        password: str | None = None,
    ) -> dict[str, Any]:
        """Create a user in Logto and assign the requested groups (roles must exist)."""
        payload: dict[str, Any] = {
            "username": username,
            "primaryEmail": email,
            "name": " ".join(p for p in (first_name, last_name) if p),
        }
        if password:
            payload["password"] = password
        user = await self._client.post("/api/users", payload)
        if groups:
            role_ids = []
            for group_name in groups:
                role = await self._find_role(group_name)
                if role is None:
                    logger.warning(
                        "create_human_user: group '%s' has no Logto role; skipping assignment",
                        group_name,
                    )
                    continue
                role_ids.append(role["id"])
            await self._assign_user_roles(user["id"], role_ids)
        return self._user_summary(user, groups)

    async def delete_user(self, username: str) -> bool:
        """Delete a user by username or email."""
        user = await self._find_user(username)
        if user is None:
            raise self._not_found(f"Logto user '{username}'")
        await self._client.delete(f"/api/users/{user['id']}")
        return True

    async def update_user_groups(self, username: str, groups: list[str]) -> dict[str, Any]:
        """Set a user's groups (roles) to exactly the requested list."""
        user = await self._find_user(username)
        if user is None:
            raise self._not_found(f"Logto user '{username}'")
        current = await self._client.get_paged(f"/api/users/{user['id']}/roles")
        current_by_name = {r.get("name"): r for r in current}
        desired = set(groups)

        add_ids = []
        for group_name in desired - set(current_by_name):
            role = await self._find_role(group_name)
            if role is None:
                logger.warning(
                    "update_user_groups: group '%s' has no Logto role; skipping", group_name
                )
                continue
            add_ids.append(role["id"])
        await self._assign_user_roles(user["id"], add_ids)

        for name in set(current_by_name) - desired:
            await self._client.delete(f"/api/users/{user['id']}/roles/{current_by_name[name]['id']}")

        return {"username": username, "groups": sorted(desired)}

    # ── service accounts ─────────────────────────────────────────────────────

    async def create_service_account(
        self, client_id: str, groups: list[str], description: str | None = None
    ) -> dict[str, Any]:
        """Create a machine-to-machine application in Logto and assign groups (roles).

        Returns the Logto application id as ``client_id`` plus the generated
        ``secret`` (only exposed at creation time).
        """
        app = await self._client.post(
            "/api/applications",
            {
                "name": client_id,
                "description": description or "",
                "type": "machine-to-machine",
            },
        )
        role_ids = []
        for group_name in groups:
            role = await self._find_role(group_name)
            if role is None:
                logger.warning(
                    "create_service_account: group '%s' has no Logto role; skipping", group_name
                )
                continue
            role_ids.append(role["id"])
        if role_ids:
            await self._client.post(f"/api/applications/{app['id']}/roles", {"roleIds": role_ids})
        return {
            "client_id": app["id"],
            "secret": app.get("secret"),
            "name": client_id,
            "groups": groups,
        }


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

    elif provider == "pingfederate":
        logger.debug("Using PingFederate IAM manager")
        return PingFederateIAMManager()

    elif provider == "cognito":
        logger.debug("Using Cognito IAM manager")
        return CognitoIAMManager()

    elif provider == "logto":
        logger.debug("Using Logto IAM manager")
        return LogtoIAMManager()

    else:
        logger.warning(f"Unknown AUTH_PROVIDER '{provider}', defaulting to Keycloak")
        return KeycloakIAMManager()
