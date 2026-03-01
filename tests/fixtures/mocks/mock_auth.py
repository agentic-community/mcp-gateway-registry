"""
Mock authentication implementations for testing.

This module provides mock implementations of authentication components.
"""

import logging
import time
from typing import Any

import jwt

logger = logging.getLogger(__name__)


class MockJWTValidator:
    """
    Mock JWT token validator for testing.

    Provides a simple JWT validation implementation that doesn't
    require actual authentication providers.
    """

    def __init__(self, secret_key: str = "test-secret-key", algorithm: str = "HS256"):
        """
        Initialize mock JWT validator.

        Args:
            secret_key: Secret key for JWT signing/validation
            algorithm: JWT algorithm
        """
        self.secret_key = secret_key
        self.algorithm = algorithm

    def create_token(
        self,
        username: str,
        groups: list[str] | None = None,
        scopes: list[str] | None = None,
        expires_in: int = 3600,
        extra_claims: dict[str, Any] | None = None,
    ) -> str:
        """
        Create a test JWT token.

        Args:
            username: Username for the token
            groups: List of groups
            scopes: List of scopes
            expires_in: Token expiration time in seconds
            extra_claims: Additional claims to include

        Returns:
            JWT token string
        """
        now = int(time.time())

        payload = {
            "sub": username,
            "username": username,
            "iat": now,
            "exp": now + expires_in,
            "token_use": "access",
        }

        if groups:
            payload["cognito:groups"] = groups
            payload["groups"] = groups

        if scopes:
            payload["scope"] = " ".join(scopes)

        if extra_claims:
            payload.update(extra_claims)

        token = jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
        logger.debug(f"Created mock JWT token for {username}")
        return token

    def validate_token(self, token: str) -> dict[str, Any]:
        """
        Validate a JWT token.

        Args:
            token: JWT token string

        Returns:
            Token payload dictionary

        Raises:
            jwt.InvalidTokenError: If token is invalid
        """
        payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
        logger.debug(f"Validated mock JWT token for {payload.get('username')}")
        return payload


class MockSessionValidator:
    """
    Mock session validator for testing cookie-based sessions.
    """

    def __init__(self, secret_key: str = "test-secret-key"):
        """
        Initialize mock session validator.

        Args:
            secret_key: Secret key for session signing
        """
        self.secret_key = secret_key

    def create_session(
        self, username: str, groups: list[str] | None = None, **extra_data: Any
    ) -> str:
        """
        Create a test session cookie value.

        Args:
            username: Username
            groups: List of groups
            **extra_data: Additional session data

        Returns:
            Session cookie value
        """
        from itsdangerous import URLSafeTimedSerializer

        serializer = URLSafeTimedSerializer(self.secret_key)

        data = {"username": username, "groups": groups or []}
        data.update(extra_data)

        session_value = serializer.dumps(data)
        logger.debug(f"Created mock session for {username}")
        return session_value

    def validate_session(self, session_value: str, max_age: int = 28800) -> dict[str, Any]:
        """
        Validate a session cookie.

        Args:
            session_value: Session cookie value
            max_age: Maximum age in seconds

        Returns:
            Session data dictionary

        Raises:
            Exception: If session is invalid or expired
        """
        from itsdangerous import URLSafeTimedSerializer

        serializer = URLSafeTimedSerializer(self.secret_key)
        data = serializer.loads(session_value, max_age=max_age)

        logger.debug(f"Validated mock session for {data.get('username')}")
        return data


def create_mock_auth_headers(
    token: str | None = None, username: str | None = None, scopes: list[str] | None = None
) -> dict[str, str]:
    """
    Create mock authentication headers for testing.

    Args:
        token: JWT token
        username: Username (if not using token)
        scopes: List of scopes (if not using token)

    Returns:
        Dictionary of HTTP headers
    """
    headers = {}

    if token:
        headers["Authorization"] = f"Bearer {token}"
    elif username:
        # Create a simple mock token
        validator = MockJWTValidator()
        token = validator.create_token(username, scopes=scopes)
        headers["Authorization"] = f"Bearer {token}"

    return headers


def create_mock_cognito_user_attributes(
    username: str, email: str | None = None, groups: list[str] | None = None
) -> list[dict[str, str]]:
    """
    Create mock Cognito user attributes.

    Args:
        username: Username
        email: Email address
        groups: List of groups

    Returns:
        List of attribute dictionaries
    """
    attributes = [
        {"Name": "sub", "Value": username},
        {"Name": "email", "Value": email or f"{username}@example.com"},
        {"Name": "email_verified", "Value": "true"},
    ]

    if groups:
        attributes.append({"Name": "cognito:groups", "Value": ",".join(groups)})

    return attributes
