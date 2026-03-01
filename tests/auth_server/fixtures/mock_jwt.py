"""
Mock JWT utilities for auth server testing.

This module provides utilities for creating and validating mock JWT tokens
in auth server tests.
"""

import logging
import time
from typing import Any

import jwt

logger = logging.getLogger(__name__)


def create_mock_jwt_token(
    username: str,
    secret_key: str = "test-secret-key",
    algorithm: str = "HS256",
    groups: list[str] | None = None,
    scopes: list[str] | None = None,
    expires_in: int = 3600,
    token_use: str = "access",
    client_id: str = "test-client-id",
    **extra_claims: Any,
) -> str:
    """
    Create a mock JWT token for testing.

    Args:
        username: Username for the token
        secret_key: Secret key for signing
        algorithm: JWT algorithm
        groups: List of user groups
        scopes: List of user scopes
        expires_in: Token expiration time in seconds
        token_use: Token use type (access, id, refresh)
        client_id: Client ID
        **extra_claims: Additional claims to include

    Returns:
        JWT token string
    """
    now = int(time.time())

    payload = {
        "sub": username,
        "username": username,
        "iat": now,
        "exp": now + expires_in,
        "token_use": token_use,
        "client_id": client_id,
        "iss": "test-issuer",
        "aud": "test-audience",
    }

    if groups:
        payload["cognito:groups"] = groups
        payload["groups"] = groups

    if scopes:
        payload["scope"] = " ".join(scopes)

    # Add extra claims
    payload.update(extra_claims)

    token = jwt.encode(payload, secret_key, algorithm=algorithm)
    logger.debug(f"Created mock JWT token for {username} with groups={groups}, scopes={scopes}")

    return token


def decode_mock_jwt_token(
    token: str, secret_key: str = "test-secret-key", algorithm: str = "HS256", verify: bool = True
) -> dict[str, Any]:
    """
    Decode a mock JWT token.

    Args:
        token: JWT token string
        secret_key: Secret key for verification
        algorithm: JWT algorithm
        verify: Whether to verify the signature

    Returns:
        Token payload dictionary

    Raises:
        jwt.InvalidTokenError: If token is invalid
    """
    options = {} if verify else {"verify_signature": False}

    payload = jwt.decode(token, secret_key, algorithms=[algorithm], options=options)

    logger.debug(f"Decoded mock JWT token for {payload.get('username')}")
    return payload


def create_expired_jwt_token(
    username: str, secret_key: str = "test-secret-key", algorithm: str = "HS256"
) -> str:
    """
    Create an expired JWT token for testing expiration handling.

    Args:
        username: Username for the token
        secret_key: Secret key for signing
        algorithm: JWT algorithm

    Returns:
        Expired JWT token string
    """
    now = int(time.time())

    payload = {
        "sub": username,
        "username": username,
        "iat": now - 7200,  # Issued 2 hours ago
        "exp": now - 3600,  # Expired 1 hour ago
        "token_use": "access",
    }

    token = jwt.encode(payload, secret_key, algorithm=algorithm)
    logger.debug(f"Created expired mock JWT token for {username}")

    return token


def create_malformed_jwt_token() -> str:
    """
    Create a malformed JWT token for testing error handling.

    Returns:
        Malformed token string
    """
    return "not.a.valid.jwt.token.format"
