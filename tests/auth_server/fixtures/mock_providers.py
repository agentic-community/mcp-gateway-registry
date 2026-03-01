"""
Mock authentication provider implementations for testing.

This module provides mock implementations of authentication providers
(Cognito, Keycloak, Entra ID) for testing the auth server.
"""

import logging
from typing import Any

logger = logging.getLogger(__name__)


class MockKeycloakProvider:
    """
    Mock Keycloak authentication provider for testing.

    Simulates the Keycloak provider interface without requiring
    a real Keycloak server.
    """

    def __init__(
        self,
        realm: str = "test-realm",
        server_url: str = "http://localhost:8080",
        client_id: str = "test-client",
    ):
        """
        Initialize mock Keycloak provider.

        Args:
            realm: Keycloak realm name
            server_url: Keycloak server URL
            client_id: Client ID
        """
        self.realm = realm
        self.server_url = server_url
        self.client_id = client_id
        self._valid_tokens: dict[str, dict[str, Any]] = {}

    def register_token(
        self,
        token: str,
        username: str,
        groups: list[str] | None = None,
        roles: list[str] | None = None,
    ) -> None:
        """
        Register a valid token for testing.

        Args:
            token: JWT token string
            username: Username
            groups: List of groups
            roles: List of roles
        """
        self._valid_tokens[token] = {
            "username": username,
            "groups": groups or [],
            "roles": roles or [],
        }
        logger.debug(f"Registered token for {username} in mock Keycloak")

    def validate_token(self, access_token: str) -> dict[str, Any]:
        """
        Validate a JWT token.

        Args:
            access_token: JWT token to validate

        Returns:
            Validation result dictionary

        Raises:
            ValueError: If token is invalid
        """
        if access_token in self._valid_tokens:
            token_info = self._valid_tokens[access_token]

            return {
                "valid": True,
                "method": "keycloak",
                "username": token_info["username"],
                "groups": token_info["groups"],
                "scopes": [],  # Keycloak uses groups/roles, not scopes
                "client_id": self.client_id,
                "data": token_info,
            }

        raise ValueError("Invalid Keycloak token")

    def get_provider_info(self) -> dict[str, Any]:
        """
        Get provider information.

        Returns:
            Provider info dictionary
        """
        return {
            "provider_type": "keycloak",
            "realm": self.realm,
            "server_url": self.server_url,
            "client_id": self.client_id,
        }


class MockCognitoValidator:
    """
    Mock Cognito validator for testing.

    Simulates AWS Cognito token validation without requiring
    actual AWS Cognito.
    """

    def __init__(
        self,
        region: str = "us-east-1",
        user_pool_id: str = "us-east-1_TEST12345",
        client_id: str = "test-client-id",
    ):
        """
        Initialize mock Cognito validator.

        Args:
            region: AWS region
            user_pool_id: Cognito User Pool ID
            client_id: Client ID
        """
        self.region = region
        self.user_pool_id = user_pool_id
        self.client_id = client_id
        self._valid_tokens: dict[str, dict[str, Any]] = {}

    def register_token(
        self, token: str, username: str, groups: list[str] | None = None, email: str | None = None
    ) -> None:
        """
        Register a valid token for testing.

        Args:
            token: JWT token string
            username: Username (Cognito sub)
            groups: List of Cognito groups
            email: User email
        """
        self._valid_tokens[token] = {
            "username": username,
            "groups": groups or [],
            "email": email or f"{username}@example.com",
            "email_verified": True,
        }
        logger.debug(f"Registered token for {username} in mock Cognito")

    def validate_token(
        self, access_token: str, user_pool_id: str, client_id: str, region: str | None = None
    ) -> dict[str, Any]:
        """
        Validate a Cognito JWT token.

        Args:
            access_token: JWT token to validate
            user_pool_id: User Pool ID
            client_id: Client ID
            region: AWS region

        Returns:
            Validation result dictionary

        Raises:
            ValueError: If token is invalid
        """
        if access_token in self._valid_tokens:
            token_info = self._valid_tokens[access_token]

            return {
                "valid": True,
                "method": "jwt",
                "username": token_info["username"],
                "groups": token_info["groups"],
                "scopes": [],
                "client_id": client_id,
                "data": {
                    "cognito:username": token_info["username"],
                    "cognito:groups": token_info["groups"],
                    "email": token_info["email"],
                },
            }

        raise ValueError("Invalid Cognito token")

    def get_provider_info(self) -> dict[str, Any]:
        """
        Get provider information.

        Returns:
            Provider info dictionary
        """
        return {
            "provider_type": "cognito",
            "region": self.region,
            "user_pool_id": self.user_pool_id,
            "client_id": self.client_id,
        }


def create_mock_provider(provider_type: str = "cognito", **kwargs: Any) -> Any:
    """
    Factory function to create mock authentication providers.

    Args:
        provider_type: Type of provider (cognito, keycloak, entra)
        **kwargs: Provider-specific configuration

    Returns:
        Mock provider instance

    Raises:
        ValueError: If provider type is not supported
    """
    if provider_type == "cognito":
        return MockCognitoValidator(**kwargs)
    elif provider_type == "keycloak":
        return MockKeycloakProvider(**kwargs)
    else:
        raise ValueError(f"Unsupported provider type: {provider_type}")
