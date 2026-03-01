"""
Conftest for auth server tests.

Provides fixtures specific to authentication server testing including
mock JWT tokens, JWKS endpoints, and authentication providers.
"""

import logging
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import jwt
import pytest

from tests.auth_server.fixtures.mock_jwt import (
    create_expired_jwt_token,
    create_malformed_jwt_token,
    create_mock_jwt_token,
)
from tests.fixtures.mocks.mock_auth import MockJWTValidator, MockSessionValidator

logger = logging.getLogger(__name__)


# =============================================================================
# AUTO-MOCKING FOR AUTH SERVER DEPENDENCIES
# =============================================================================


def _setup_auth_server_mocks() -> None:
    """
    Set up automatic mocking for auth server dependencies.

    This must run BEFORE importing auth_server modules to avoid
    missing dependency errors.
    """
    # Add auth_server to Python path
    auth_server_path = Path(__file__).parent.parent.parent / "auth_server"
    if str(auth_server_path) not in sys.path:
        sys.path.insert(0, str(auth_server_path))
        logger.info(f"Added auth_server to Python path: {auth_server_path}")

    # Mock metrics_middleware
    mock_metrics = MagicMock()
    mock_metrics.add_auth_metrics_middleware = MagicMock()
    sys.modules["metrics_middleware"] = mock_metrics
    logger.info("Auto-mocked: metrics_middleware")


# Execute auto-mocking setup
_setup_auth_server_mocks()


# =============================================================================
# MOCK JWKS FIXTURES
# =============================================================================


@pytest.fixture
def mock_jwks_response() -> dict:
    """
    Create a mock JWKS response with RSA public keys.

    Returns:
        Dictionary containing JWKS data
    """
    return {
        "keys": [
            {
                "kid": "test-key-id-1",
                "kty": "RSA",
                "alg": "RS256",
                "use": "sig",
                "n": "xGOr-H7A-PWgGZ8J0lYnBQTJHQLIvFKvSfBbQddPn8A",
                "e": "AQAB",
            },
            {
                "kid": "test-key-id-2",
                "kty": "RSA",
                "alg": "RS256",
                "use": "sig",
                "n": "yHPr-I8B-QXhHa9K1mZoCRUKIHRMJwGLwGTcTgeQo9B",
                "e": "AQAB",
            },
        ]
    }


@pytest.fixture
def mock_requests_get(mock_jwks_response):
    """
    Mock requests.get for JWKS endpoint calls.

    Args:
        mock_jwks_response: JWKS response fixture

    Yields:
        Mock requests.get function
    """
    with patch("requests.get") as mock_get:
        mock_response = MagicMock()
        mock_response.json.return_value = mock_jwks_response
        mock_response.raise_for_status.return_value = None
        mock_response.status_code = 200
        mock_get.return_value = mock_response

        logger.debug("Mocked requests.get for JWKS endpoints")
        yield mock_get


# =============================================================================
# JWT TOKEN FIXTURES
# =============================================================================


@pytest.fixture
def valid_jwt_token() -> str:
    """
    Create a valid JWT token for testing.

    Returns:
        Valid JWT token string
    """
    return create_mock_jwt_token(
        username="testuser",
        groups=["users", "developers"],
        scopes=["read:servers", "write:servers"],
        expires_in=3600,
    )


@pytest.fixture
def expired_jwt_token() -> str:
    """
    Create an expired JWT token for testing.

    Returns:
        Expired JWT token string
    """
    return create_expired_jwt_token(username="testuser")


@pytest.fixture
def malformed_jwt_token() -> str:
    """
    Create a malformed JWT token for testing.

    Returns:
        Malformed token string
    """
    return create_malformed_jwt_token()


@pytest.fixture
def self_signed_token(auth_env_vars) -> str:
    """
    Create a self-signed JWT token using the auth server's secret key.

    Args:
        auth_env_vars: Environment variables fixture

    Returns:
        Self-signed JWT token
    """
    secret_key = auth_env_vars["SECRET_KEY"]
    now = int(time.time())

    payload = {
        "iss": "mcp-auth-server",
        "aud": "mcp-registry",
        "sub": "testuser",
        "scope": "read:servers write:servers",
        "exp": now + 3600,
        "iat": now,
        "token_use": "access",
        "client_id": "user-generated",
    }

    return jwt.encode(payload, secret_key, algorithm="HS256")


@pytest.fixture
def m2m_token() -> str:
    """
    Create a machine-to-machine JWT token for testing.

    Returns:
        M2M JWT token string
    """
    return create_mock_jwt_token(
        username="service-account",
        scopes=["admin:all"],
        token_use="access",
        client_id="m2m-client",
        azp="m2m-client",
    )


# =============================================================================
# MOCK JWT VALIDATOR FIXTURES
# =============================================================================


@pytest.fixture
def mock_jwt_validator() -> MockJWTValidator:
    """
    Create a mock JWT validator for testing.

    Returns:
        MockJWTValidator instance
    """
    return MockJWTValidator(secret_key="test-jwt-secret")


@pytest.fixture
def mock_session_validator() -> MockSessionValidator:
    """
    Create a mock session validator for testing.

    Returns:
        MockSessionValidator instance
    """
    return MockSessionValidator(secret_key="test-session-secret")


# =============================================================================
# ENVIRONMENT FIXTURES
# =============================================================================


@pytest.fixture
def auth_env_vars(monkeypatch) -> dict[str, str]:
    """
    Set up environment variables for auth server testing.

    Args:
        monkeypatch: Pytest monkeypatch fixture

    Returns:
        Dictionary of environment variables set
    """
    env_vars = {
        "SECRET_KEY": "test-secret-key-for-auth-testing-do-not-use-in-prod",
        "ADMIN_USER": "testadmin",
        "ADMIN_PASSWORD": "testadminpass",
        "AUTH_PROVIDER": "cognito",
        "COGNITO_USER_POOL_ID": "us-east-1_TEST12345",
        "COGNITO_CLIENT_ID": "test-client-id",
        "COGNITO_CLIENT_SECRET": "test-client-secret",
        "AWS_REGION": "us-east-1",
        "MAX_TOKENS_PER_USER_PER_HOUR": "100",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    logger.debug(f"Set up {len(env_vars)} auth environment variables")
    return env_vars


@pytest.fixture
def keycloak_env_vars(monkeypatch) -> dict[str, str]:
    """
    Set up Keycloak environment variables for testing.

    Args:
        monkeypatch: Pytest monkeypatch fixture

    Returns:
        Dictionary of environment variables set
    """
    env_vars = {
        "AUTH_PROVIDER": "keycloak",
        "KEYCLOAK_URL": "http://localhost:8080",
        "KEYCLOAK_EXTERNAL_URL": "https://keycloak.example.com",
        "KEYCLOAK_REALM": "test-realm",
        "KEYCLOAK_CLIENT_ID": "test-client",
        "KEYCLOAK_CLIENT_SECRET": "test-secret",
        "KEYCLOAK_M2M_CLIENT_ID": "m2m-client",
        "KEYCLOAK_M2M_CLIENT_SECRET": "m2m-secret",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    logger.debug(f"Set up {len(env_vars)} Keycloak environment variables")
    return env_vars


@pytest.fixture
def entra_env_vars(monkeypatch) -> dict[str, str]:
    """
    Set up Entra ID environment variables for testing.

    Args:
        monkeypatch: Pytest monkeypatch fixture

    Returns:
        Dictionary of environment variables set
    """
    env_vars = {
        "AUTH_PROVIDER": "entra",
        "ENTRA_TENANT_ID": "test-tenant-id",
        "ENTRA_CLIENT_ID": "test-client-id",
        "ENTRA_CLIENT_SECRET": "test-client-secret",
    }

    for key, value in env_vars.items():
        monkeypatch.setenv(key, value)

    logger.debug(f"Set up {len(env_vars)} Entra ID environment variables")
    return env_vars


# =============================================================================
# MOCK PROVIDER FIXTURES
# =============================================================================


@pytest.fixture
def mock_cognito_provider():
    """
    Create a mock Cognito provider for testing.

    Returns:
        Mock Cognito provider
    """
    provider = MagicMock()
    provider.validate_token = MagicMock(
        return_value={
            "valid": True,
            "method": "cognito",
            "username": "testuser",
            "email": "testuser@example.com",
            "groups": ["users", "developers"],
            "scopes": [],
            "client_id": "test-client-id",
            "data": {
                "cognito:username": "testuser",
                "cognito:groups": ["users", "developers"],
                "email": "testuser@example.com",
            },
        }
    )
    provider.get_provider_info = MagicMock(
        return_value={
            "provider_type": "cognito",
            "region": "us-east-1",
            "user_pool_id": "us-east-1_TEST12345",
            "client_id": "test-client-id",
        }
    )
    provider.get_jwks = MagicMock(return_value={"keys": [{"kid": "test-key", "kty": "RSA"}]})

    return provider


@pytest.fixture
def mock_keycloak_provider():
    """
    Create a mock Keycloak provider for testing.

    Returns:
        Mock Keycloak provider
    """
    provider = MagicMock()
    provider.validate_token = MagicMock(
        return_value={
            "valid": True,
            "method": "keycloak",
            "username": "testuser",
            "email": "testuser@example.com",
            "groups": ["users", "admins"],
            "scopes": ["openid", "profile"],
            "client_id": "test-client",
            "data": {
                "preferred_username": "testuser",
                "email": "testuser@example.com",
                "groups": ["users", "admins"],
            },
        }
    )
    provider.get_provider_info = MagicMock(
        return_value={
            "provider_type": "keycloak",
            "realm": "test-realm",
            "keycloak_url": "http://localhost:8080",
            "client_id": "test-client",
        }
    )
    provider.get_jwks = MagicMock(return_value={"keys": [{"kid": "test-key", "kty": "RSA"}]})

    return provider


@pytest.fixture
def mock_entra_provider():
    """
    Create a mock Entra ID provider for testing.

    Returns:
        Mock Entra ID provider
    """
    provider = MagicMock()
    provider.validate_token = MagicMock(
        return_value={
            "valid": True,
            "method": "entra",
            "username": "testuser@example.com",
            "email": "testuser@example.com",
            "groups": ["group-id-1", "group-id-2"],
            "scopes": ["openid", "profile", "email"],
            "client_id": "test-client-id",
            "data": {
                "preferred_username": "testuser@example.com",
                "email": "testuser@example.com",
                "groups": ["group-id-1", "group-id-2"],
            },
        }
    )
    provider.get_provider_info = MagicMock(
        return_value={
            "provider_type": "entra",
            "tenant_id": "test-tenant-id",
            "client_id": "test-client-id",
        }
    )
    provider.get_jwks = MagicMock(return_value={"keys": [{"kid": "test-key", "kty": "RSA"}]})

    return provider


# =============================================================================
# SESSION COOKIE FIXTURES
# =============================================================================


@pytest.fixture
def valid_session_cookie(auth_env_vars) -> str:
    """
    Create a valid session cookie for testing.

    Args:
        auth_env_vars: Environment variables fixture

    Returns:
        Encrypted session cookie string
    """
    from itsdangerous import URLSafeTimedSerializer

    secret_key = auth_env_vars["SECRET_KEY"]
    signer = URLSafeTimedSerializer(secret_key)

    session_data = {
        "username": "testuser",
        "email": "testuser@example.com",
        "groups": ["users", "developers"],
        "provider": "cognito",
        "auth_method": "oauth2",
    }

    return signer.dumps(session_data)


@pytest.fixture
def expired_session_cookie() -> str:
    """
    Create an expired session cookie for testing.

    Returns:
        Expired session cookie string (with invalid signature)
    """
    # Return a cookie with bad signature to simulate expiration
    return "invalid.signature.cookie"


# =============================================================================
# SCOPES CONFIGURATION FIXTURES
# =============================================================================


@pytest.fixture
def mock_scopes_config() -> dict:
    """
    Create a mock scopes configuration for testing.

    Returns:
        Dictionary containing scopes configuration
    """
    return {
        "group_mappings": {
            "users": ["read:servers", "read:tools"],
            "developers": ["read:servers", "write:servers", "read:tools", "tools:call"],
            "admins": ["admin:all"],
        },
        "read:servers": [
            {"server": "test-server", "methods": ["initialize", "tools/list"], "tools": []}
        ],
        "write:servers": [
            {
                "server": "test-server",
                "methods": ["initialize", "tools/list", "tools/call"],
                "tools": ["*"],
            }
        ],
        "admin:all": [{"server": "*", "methods": ["*"], "tools": ["*"]}],
    }


@pytest.fixture
def mock_scopes_config_file(tmp_path, mock_scopes_config):
    """
    Create a temporary scopes.yml file for testing.

    Args:
        tmp_path: Pytest temporary path fixture
        mock_scopes_config: Mock scopes configuration

    Returns:
        Path to temporary scopes.yml file
    """
    import yaml

    scopes_file = tmp_path / "scopes.yml"
    with open(scopes_file, "w") as f:
        yaml.dump(mock_scopes_config, f)

    logger.debug(f"Created mock scopes config file: {scopes_file}")
    return scopes_file


@pytest.fixture
def mock_scope_repository_with_data(mock_scopes_config):
    """
    Create a mocked scope repository that returns data from mock_scopes_config.

    Args:
        mock_scopes_config: Mock scopes configuration fixture

    Returns:
        AsyncMock scope repository with get_server_scopes method
    """
    mock_repo = AsyncMock()

    # Mock get_server_scopes to return the scope data from mock_scopes_config
    async def get_server_scopes_side_effect(scope_name: str):
        """Return server access rules for a scope from mock_scopes_config."""
        # Return the scope data if it exists, otherwise empty list
        return mock_scopes_config.get(scope_name, [])

    # Mock get_group_mappings to return scopes for a group from mock_scopes_config
    async def get_group_mappings_side_effect(group_name: str):
        """Return scopes for a group from mock_scopes_config."""
        group_mappings = mock_scopes_config.get("group_mappings", {})
        return group_mappings.get(group_name, [])

    mock_repo.get_server_scopes.side_effect = get_server_scopes_side_effect
    mock_repo.get_group_mappings.side_effect = get_group_mappings_side_effect
    mock_repo.load_all = AsyncMock()
    mock_repo.list_groups.return_value = {}
    mock_repo.get_group.return_value = None
    mock_repo.get_scope_definition.return_value = None
    mock_repo.list_scope_definitions.return_value = []

    return mock_repo


# =============================================================================
# RATE LIMITING FIXTURES
# =============================================================================


@pytest.fixture
def mock_rate_limiter():
    """
    Create a mock rate limiter that tracks token generation.

    Returns:
        Dictionary to track rate limit state
    """
    return {"counts": {}, "limit": 100}
