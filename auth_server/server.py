"""
Simplified Authentication server that validates JWT tokens against Amazon Cognito.
Configuration is passed via headers instead of environment variables.
"""

import argparse
import hashlib
import hmac
import json
import logging
import os
import secrets

# Import shared scopes loader and repository factory from registry common module
import sys
import time
import urllib.parse
from contextlib import asynccontextmanager
from datetime import datetime
from pathlib import Path
from string import Template
from typing import Any
from urllib.parse import urlparse

import boto3
import httpx
import jwt
import requests
import uvicorn
import yaml
from botocore.exceptions import ClientError
from fastapi import Cookie, FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from jwt.api_jwk import PyJWK

# Import metrics middleware
from metrics_middleware import add_auth_metrics_middleware

# Import provider factory
from providers.factory import get_auth_provider
from pydantic import BaseModel

sys.path.insert(0, "/app")
# Import MCP audit logging components
from registry.audit.mcp_logger import MCPLogger
from registry.audit.models import Identity, MCPServer
from registry.audit.service import AuditLogger
from registry.common.scopes_loader import reload_scopes_config
from registry.core.config import settings
from registry.repositories.factory import get_scope_repository
from registry.utils.request_utils import get_client_ip

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# Import JWT constants from shared internal auth module
from registry.auth.internal import (
    _INTERNAL_JWT_ISSUER as JWT_ISSUER,
    _INTERNAL_JWT_AUDIENCE as JWT_AUDIENCE,
)
MAX_TOKEN_LIFETIME_HOURS = 24
DEFAULT_TOKEN_LIFETIME_HOURS = 8

# Rate limiting for token generation (simple in-memory counter)
user_token_generation_counts = {}
MAX_TOKENS_PER_USER_PER_HOUR = int(os.environ.get("MAX_TOKENS_PER_USER_PER_HOUR", "100"))

# Global scopes configuration (will be loaded during FastAPI startup)
SCOPES_CONFIG = {}

# Static token auth: use static API key instead of IdP JWT for Registry API
_registry_static_token_requested: bool = (
    os.environ.get("REGISTRY_STATIC_TOKEN_AUTH_ENABLED", "false").lower() == "true"
)

# Static API key for Registry API (must match Bearer token value when enabled)
REGISTRY_API_TOKEN: str = os.environ.get("REGISTRY_API_TOKEN", "")

# OAuth token storage in session cookies (disable for IdPs with large tokens)
# Default: false - tokens are not used functionally and storing them risks cookie size limits
OAUTH_STORE_TOKENS_IN_SESSION: bool = (
    os.environ.get("OAUTH_STORE_TOKENS_IN_SESSION", "false").lower() == "true"
)

logging.info(f"OAUTH_STORE_TOKENS_IN_SESSION={OAUTH_STORE_TOKENS_IN_SESSION}")

# Validate configuration: static token auth requires REGISTRY_API_TOKEN to be set
if _registry_static_token_requested and not REGISTRY_API_TOKEN:
    logging.error(
        "REGISTRY_STATIC_TOKEN_AUTH_ENABLED=true but REGISTRY_API_TOKEN is not set. "
        "Static token auth is DISABLED. Set REGISTRY_API_TOKEN or disable the feature. "
        "Falling back to standard IdP JWT validation."
    )
    REGISTRY_STATIC_TOKEN_AUTH_ENABLED: bool = False
else:
    REGISTRY_STATIC_TOKEN_AUTH_ENABLED: bool = _registry_static_token_requested

# Get ROOT_PATH for path-based routing (auth server's own path, e.g. /auth-server)
ROOT_PATH = os.environ.get("ROOT_PATH", "").rstrip("/")

# REGISTRY_ROOT_PATH is the registry's base path (e.g. /registry) used for matching
# X-Original-URL paths that come from the registry's nginx. Falls back to ROOT_PATH
# for backward compatibility when both services share the same root path.
REGISTRY_ROOT_PATH = os.environ.get("REGISTRY_ROOT_PATH", ROOT_PATH).rstrip("/")

# Registry API path patterns that use static token auth when enabled
# REGISTRY_ROOT_PATH is prepended so pattern matching works when hosted on a base path (e.g. /registry/api/)
REGISTRY_API_PATTERNS: list = [
    f"{REGISTRY_ROOT_PATH}/api/",
    f"{REGISTRY_ROOT_PATH}/v0.1/",
]

# Federation static token auth: scoped token for federation endpoints only
_federation_static_token_requested: bool = (
    os.environ.get("FEDERATION_STATIC_TOKEN_AUTH_ENABLED", "false").lower() == "true"
)

FEDERATION_STATIC_TOKEN: str = os.environ.get("FEDERATION_STATIC_TOKEN", "")

if _federation_static_token_requested and not FEDERATION_STATIC_TOKEN:
    logging.error(
        "FEDERATION_STATIC_TOKEN_AUTH_ENABLED=true but FEDERATION_STATIC_TOKEN is not set. "
        "Federation static token auth is DISABLED. Set FEDERATION_STATIC_TOKEN or disable the feature. "
        "Falling back to standard IdP JWT validation."
    )
    FEDERATION_STATIC_TOKEN_AUTH_ENABLED: bool = False
else:
    FEDERATION_STATIC_TOKEN_AUTH_ENABLED: bool = _federation_static_token_requested

# Warn if token is too short (weak entropy)
MIN_FEDERATION_TOKEN_LENGTH: int = 32
if (
    FEDERATION_STATIC_TOKEN_AUTH_ENABLED
    and len(FEDERATION_STATIC_TOKEN) < MIN_FEDERATION_TOKEN_LENGTH
):
    logging.warning(
        f"FEDERATION_STATIC_TOKEN is only {len(FEDERATION_STATIC_TOKEN)} characters. "
        f"Recommended minimum is {MIN_FEDERATION_TOKEN_LENGTH} characters. "
        'Generate a stronger token with: python3 -c "import secrets; print(secrets.token_urlsafe(32))"'
    )

# Federation endpoint path patterns (scoped access for federation static token)
# REGISTRY_ROOT_PATH is prepended so pattern matching works when hosted on a base path
FEDERATION_API_PATTERNS: list = [
    f"{REGISTRY_ROOT_PATH}/api/federation/",
    f"{REGISTRY_ROOT_PATH}/api/peers/",
    "/api/peers",  # exact match for list peers (no trailing slash)
]

# Utility functions for GDPR/SOX compliance


def is_request_https(request) -> bool:
    """
    Detect if the original request was HTTPS.

    Priority order:
    1. X-Cloudfront-Forwarded-Proto header (CloudFront deployments)
    2. x-forwarded-proto header (ALB/custom domain deployments)
    3. Request URL scheme (direct access)

    Args:
        request: FastAPI Request object

    Returns:
        True if the original request was HTTPS
    """
    # Check CloudFront header first (ALB won't overwrite this)
    cloudfront_proto = request.headers.get("x-cloudfront-forwarded-proto", "")
    if cloudfront_proto.lower() == "https":
        return True

    # Fall back to standard x-forwarded-proto
    x_forwarded_proto = request.headers.get("x-forwarded-proto", "")
    if x_forwarded_proto.lower() == "https":
        return True

    # Finally check request scheme
    return request.url.scheme == "https"


def mask_sensitive_id(value: str) -> str:
    """Mask sensitive IDs showing only first and last 4 characters."""
    if not value or len(value) <= 8:
        return "***MASKED***"
    return f"{value[:4]}...{value[-4:]}"


def hash_username(username: str) -> str:
    """Hash username for privacy compliance."""
    if not username:
        return "anonymous"
    return f"user_{hashlib.sha256(username.encode()).hexdigest()[:8]}"


def anonymize_ip(ip_address: str) -> str:
    """Anonymize IP address by masking last octet for IPv4."""
    if not ip_address or ip_address == "unknown":
        return ip_address
    if "." in ip_address:  # IPv4
        parts = ip_address.split(".")
        if len(parts) == 4:
            return f"{'.'.join(parts[:3])}.xxx"
    elif ":" in ip_address:  # IPv6
        # Mask last segment
        parts = ip_address.split(":")
        if len(parts) > 1:
            parts[-1] = "xxxx"
            return ":".join(parts)
    return ip_address


def mask_token(token: str) -> str:
    """Mask JWT token showing only first 4 characters followed by ellipsis."""
    if not token:
        return "***EMPTY***"
    if len(token) > 8:
        return f"{token[:4]}..."
    return "***MASKED***"


def _mask_sensitive_dict(
    data: dict,
    sensitive_keys: tuple = ("access_token", "refresh_token", "token", "secret", "password"),
) -> dict:
    """
    Recursively mask sensitive fields in a dictionary for safe logging.

    Args:
        data: Dictionary to process
        sensitive_keys: Tuple of key names to mask

    Returns:
        New dictionary with sensitive fields masked
    """
    if not isinstance(data, dict):
        return data

    masked = {}
    for key, value in data.items():
        key_lower = key.lower()
        if any(sensitive in key_lower for sensitive in sensitive_keys):
            if isinstance(value, str) and value:
                masked[key] = mask_token(value)
            else:
                masked[key] = "***MASKED***"
        elif isinstance(value, dict):
            masked[key] = _mask_sensitive_dict(value, sensitive_keys)
        elif isinstance(value, list):
            masked[key] = [
                _mask_sensitive_dict(item, sensitive_keys) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            masked[key] = value
    return masked


def mask_headers(headers: dict) -> dict:
    """Mask sensitive headers for logging compliance."""
    masked = {}
    for key, value in headers.items():
        key_lower = key.lower()
        if key_lower in ["x-authorization", "authorization", "cookie"]:
            if "bearer" in str(value).lower():
                # Extract token part and mask it
                parts = str(value).split(" ", 1)
                if len(parts) == 2:
                    masked[key] = f"Bearer {mask_token(parts[1])}"
                else:
                    masked[key] = mask_token(value)
            else:
                masked[key] = "***MASKED***"
        elif key_lower in ["x-user-pool-id", "x-client-id"]:
            masked[key] = mask_sensitive_id(value)
        else:
            masked[key] = value
    return masked


async def map_groups_to_scopes(groups: list[str]) -> list[str]:
    """
    Map identity provider groups to MCP scopes by querying DocumentDB directly.

    Args:
        groups: List of group names from identity provider (Cognito, Keycloak, etc.)

    Returns:
        List of MCP scopes
    """
    scopes = []

    # Query DocumentDB directly for group mappings
    try:
        scope_repo = get_scope_repository()

        for group in groups:
            # Query DocumentDB for this group's scope mappings
            group_scopes = await scope_repo.get_group_mappings(group)
            if group_scopes:
                scopes.extend(group_scopes)
                logger.debug(f"Mapped group '{group}' to scopes: {group_scopes}")
            else:
                logger.debug(f"No scope mapping found for group: {group}")
    except Exception as e:
        logger.error(f"Error querying group mappings from DocumentDB: {e}", exc_info=True)
        # Fall back to in-memory config if DocumentDB query fails
        group_mappings = SCOPES_CONFIG.get("group_mappings", {})
        for group in groups:
            if group in group_mappings:
                group_scopes = group_mappings[group]
                scopes.extend(group_scopes)
                logger.debug(f"Mapped group '{group}' to scopes (fallback): {group_scopes}")

    # Remove duplicates while preserving order
    seen = set()
    unique_scopes = []
    for scope in scopes:
        if scope not in seen:
            seen.add(scope)
            unique_scopes.append(scope)

    logger.info(f"Final mapped scopes: {unique_scopes}")
    return unique_scopes


async def validate_session_cookie(cookie_value: str) -> dict[str, any]:
    """
    Validate session cookie using itsdangerous serializer.

    Args:
        cookie_value: The session cookie value

    Returns:
        Dict containing validation results matching JWT validation format:
        {
            'valid': True,
            'username': str,
            'scopes': List[str],
            'method': 'session_cookie',
            'groups': List[str]
        }

    Raises:
        ValueError: If cookie is invalid or expired
    """
    # Use global signer initialized at startup
    global signer
    if not signer:
        logger.warning("Global signer not configured for session cookie validation")
        raise ValueError("Session cookie validation not configured")

    try:
        # Decrypt cookie (max_age=28800 for 8 hours)
        data = signer.loads(cookie_value, max_age=28800)

        # Extract user info
        username = data.get("username")
        groups = data.get("groups", [])

        # Map groups to scopes (async call to query DocumentDB)
        scopes = await map_groups_to_scopes(groups)

        logger.info(f"Session cookie validated for user: {hash_username(username)}")

        return {
            "valid": True,
            "username": username,
            "scopes": scopes,
            "method": "session_cookie",
            "groups": groups,
            "client_id": "",  # Not applicable for session
            "data": data,  # Include full data for consistency
        }
    except SignatureExpired:
        logger.warning("Session cookie has expired")
        raise ValueError("Session cookie has expired")
    except BadSignature:
        logger.warning("Invalid session cookie signature")
        raise ValueError("Invalid session cookie")
    except Exception as e:
        logger.error(f"Session cookie validation error: {e}")
        raise ValueError(f"Session cookie validation failed: {e}")


def parse_server_and_tool_from_url(original_url: str) -> tuple[str | None, str | None]:
    """
    Parse server name and tool name from the original URL and request payload.

    Args:
        original_url: The original URL from X-Original-URL header

    Returns:
        Tuple of (server_name, tool_name) or (None, None) if parsing fails
    """
    try:
        # Extract path from URL (remove query parameters and fragments)
        from urllib.parse import urlparse

        parsed_url = urlparse(original_url)
        path = parsed_url.path.strip("/")

        # The path should be in format: /server_name/...
        # Extract the first path component as server name
        path_parts = path.split("/") if path else []
        server_name = path_parts[0] if path_parts else None

        logger.debug(f"Parsed server name '{server_name}' from URL path: {path}")
        return server_name, None  # Tool name would need to be extracted from request payload

    except Exception as e:
        logger.error(f"Failed to parse server/tool from URL {original_url}: {e}")
        return None, None


def _normalize_server_name(name: str) -> str:
    """
    Normalize server name by removing leading and trailing slashes for comparison.

    This handles cases where a server is registered with a leading or trailing
    slash but accessed without one (or vice versa). Scope configs from the UI
    store server names with a leading slash (e.g. '/cloudflare-docs') while the
    URL extraction produces names without one (e.g. 'cloudflare-docs').

    Args:
        name: Server name to normalize

    Returns:
        Normalized server name (without leading or trailing slashes)
    """
    return name.strip("/") if name else name


def _server_names_match(name1: str, name2: str) -> bool:
    """
    Compare two server names, normalizing for trailing slashes.
    Supports wildcard matching with '*'.

    Args:
        name1: First server name (can be '*' for wildcard)
        name2: Second server name

    Returns:
        True if names match (ignoring trailing slashes) or if name1 is '*', False otherwise
    """
    normalized_name1 = _normalize_server_name(name1)
    if normalized_name1 == "*":
        return True
    return normalized_name1 == _normalize_server_name(name2)


async def validate_server_tool_access(
    server_name: str, method: str, tool_name: str, user_scopes: list[str]
) -> bool:
    """
    Validate if the user has access to the specified server method/tool based on scopes.

    Args:
        server_name: Name of the MCP server
        method: Name of the method being accessed (e.g., 'initialize', 'notifications/initialized', 'tools/list')
        tool_name: Name of the specific tool being accessed (optional, for tools/call)
        user_scopes: List of user scopes from token

    Returns:
        True if access is allowed, False otherwise
    """
    try:
        # Verbose logging: Print input parameters
        logger.info("=== VALIDATE_SERVER_TOOL_ACCESS START ===")
        logger.info(f"Requested server: '{server_name}'")
        logger.info(f"Requested method: '{method}'")
        logger.info(f"Requested tool: '{tool_name}'")
        logger.info(f"User scopes: {user_scopes}")

        # Query DocumentDB directly for server access rules
        scope_repo = get_scope_repository()

        # Check each user scope to see if it grants access
        for scope in user_scopes:
            logger.info(f"--- Checking scope: '{scope}' ---")

            # Query DocumentDB for this scope's server access rules
            scope_config = await scope_repo.get_server_scopes(scope)

            if not scope_config:
                logger.info(f"Scope '{scope}' not found in DocumentDB")
                continue

            logger.info(f"Scope '{scope}' config: {scope_config}")

            # The scope_config is directly a list of server configurations
            # since the permission type is already encoded in the scope name
            for server_config in scope_config:
                logger.info(f"  Examining server config: {server_config}")
                server_config_name = server_config.get("server")
                logger.info(
                    f"  Server name in config: '{server_config_name}' vs requested: '{server_name}'"
                )

                if _server_names_match(server_config_name, server_name):
                    logger.info("  ✓ Server name matches!")

                    # Check methods first
                    allowed_methods = server_config.get("methods", [])
                    logger.info(f"  Allowed methods for server '{server_name}': {allowed_methods}")
                    logger.info(f"  Checking if method '{method}' is in allowed methods...")

                    # Check if all methods are allowed (wildcard support)
                    has_wildcard_methods = "all" in allowed_methods or "*" in allowed_methods

                    # for all methods except tools/call we are good if the method is allowed
                    # for tools/call we need to do an extra validation to check if the tool
                    # itself is allowed or not
                    if (
                        method in allowed_methods or has_wildcard_methods
                    ) and method != "tools/call":
                        logger.info(f"  ✓ Method '{method}' found in allowed methods!")
                        logger.info(
                            f"Access granted: scope '{scope}' allows access to {server_name}.{method}"
                        )
                        logger.info("=== VALIDATE_SERVER_TOOL_ACCESS END: GRANTED ===")
                        return True

                    # Check tools if method not found in methods
                    allowed_tools = server_config.get("tools", [])
                    logger.info(f"  Allowed tools for server '{server_name}': {allowed_tools}")

                    # Check if all tools are allowed (wildcard support)
                    has_wildcard_tools = "all" in allowed_tools or "*" in allowed_tools

                    # For tools/call, check if the specific tool is allowed
                    if method == "tools/call" and tool_name:
                        logger.info(
                            f"  Checking if tool '{tool_name}' is in allowed tools for tools/call..."
                        )
                        if tool_name in allowed_tools or has_wildcard_tools:
                            logger.info(f"  ✓ Tool '{tool_name}' found in allowed tools!")
                            logger.info(
                                f"Access granted: scope '{scope}' allows access to {server_name}.{method} for tool {tool_name}"
                            )
                            logger.info("=== VALIDATE_SERVER_TOOL_ACCESS END: GRANTED ===")
                            return True
                        else:
                            logger.info(f"  ✗ Tool '{tool_name}' NOT found in allowed tools")
                    else:
                        # For other methods, check if method is in tools list (backward compatibility)
                        logger.info(f"  Checking if method '{method}' is in allowed tools...")
                        if method in allowed_tools or has_wildcard_tools:
                            logger.info(f"  ✓ Method '{method}' found in allowed tools!")
                            logger.info(
                                f"Access granted: scope '{scope}' allows access to {server_name}.{method}"
                            )
                            logger.info("=== VALIDATE_SERVER_TOOL_ACCESS END: GRANTED ===")
                            return True
                        else:
                            logger.info(f"  ✗ Method '{method}' NOT found in allowed tools")
                else:
                    logger.info("  ✗ Server name does not match")

        logger.warning(
            f"Access denied: no scope allows access to {server_name}.{method} (tool: {tool_name}) for user scopes: {user_scopes}"
        )
        logger.info("=== VALIDATE_SERVER_TOOL_ACCESS END: DENIED ===")
        return False

    except Exception as e:
        logger.error(f"Error validating server/tool access: {e}")
        logger.info("=== VALIDATE_SERVER_TOOL_ACCESS END: ERROR ===")
        return False  # Deny access on error


def validate_scope_subset(user_scopes: list[str], requested_scopes: list[str]) -> bool:
    """
    Validate that requested scopes are a subset of user's current scopes.

    Args:
        user_scopes: List of scopes the user currently has
        requested_scopes: List of scopes being requested for the token

    Returns:
        True if requested scopes are valid (subset of user scopes), False otherwise
    """
    if not requested_scopes:
        return True  # Empty request is valid

    user_scope_set = set(user_scopes)
    requested_scope_set = set(requested_scopes)

    is_valid = requested_scope_set.issubset(user_scope_set)

    if not is_valid:
        invalid_scopes = requested_scope_set - user_scope_set
        logger.warning(f"Invalid scopes requested: {invalid_scopes}")

    return is_valid


def check_rate_limit(username: str) -> bool:
    """
    Check if user has exceeded token generation rate limit.

    Args:
        username: Username to check

    Returns:
        True if under rate limit, False if exceeded
    """
    current_time = int(time.time())
    current_hour = current_time // 3600

    # Clean up old entries (older than 1 hour)
    keys_to_remove = []
    for key in user_token_generation_counts.keys():
        stored_hour = int(key.split(":")[1])
        if current_hour - stored_hour > 1:
            keys_to_remove.append(key)

    for key in keys_to_remove:
        del user_token_generation_counts[key]

    # Check current hour count
    rate_key = f"{username}:{current_hour}"
    current_count = user_token_generation_counts.get(rate_key, 0)

    if current_count >= MAX_TOKENS_PER_USER_PER_HOUR:
        logger.warning(
            f"Rate limit exceeded for user {hash_username(username)}: {current_count} tokens this hour"
        )
        return False

    # Increment counter
    user_token_generation_counts[rate_key] = current_count + 1
    return True


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Lifespan context manager for FastAPI application."""
    # Startup: Load scopes configuration
    global SCOPES_CONFIG
    try:
        SCOPES_CONFIG = await reload_scopes_config()
        logger.info(
            f"Loaded scopes configuration on startup with {len(SCOPES_CONFIG.get('group_mappings', {}))} group mappings"
        )
    except Exception as e:
        logger.error(f"Failed to load scopes configuration on startup: {e}", exc_info=True)
        # Fall back to empty config
        SCOPES_CONFIG = {"group_mappings": {}}

    yield

    # Shutdown: Add cleanup code here if needed in the future
    logger.info("Shutting down auth server")


# Create FastAPI app
app = FastAPI(
    title="Simplified Auth Server",
    description="Authentication server for validating JWT tokens against Amazon Cognito with header-based configuration",
    version="0.1.0",
    lifespan=lifespan,
    root_path=ROOT_PATH,
)


@app.on_event("startup")
async def startup_event():
    """Load scopes configuration on startup."""
    global SCOPES_CONFIG
    try:
        SCOPES_CONFIG = await reload_scopes_config()
        logger.info(
            f"Loaded scopes configuration on startup with {len(SCOPES_CONFIG.get('group_mappings', {}))} group mappings"
        )
    except Exception as e:
        logger.error(f"Failed to load scopes configuration on startup: {e}", exc_info=True)
        # Fall back to empty config
        SCOPES_CONFIG = {"group_mappings": {}}


# Add metrics collection middleware
add_auth_metrics_middleware(app)


class TokenValidationResponse(BaseModel):
    """Response model for token validation"""

    valid: bool
    scopes: list[str] = []
    error: str | None = None
    method: str | None = None
    client_id: str | None = None
    username: str | None = None


class GenerateTokenRequest(BaseModel):
    """Request model for token generation"""

    user_context: dict[str, Any]
    requested_scopes: list[str] = []
    expires_in_hours: int = DEFAULT_TOKEN_LIFETIME_HOURS
    description: str | None = None


class GenerateTokenResponse(BaseModel):
    """Response model for token generation"""

    access_token: str
    refresh_token: str | None = None
    token_type: str = "Bearer"
    expires_in: int
    refresh_expires_in: int | None = None
    scope: str
    issued_at: int
    description: str | None = None


class SimplifiedCognitoValidator:
    """
    Simplified Cognito token validator that doesn't rely on environment variables
    """

    def __init__(self, region: str = "us-east-1"):
        """
        Initialize with minimal configuration

        Args:
            region: Default AWS region
        """
        self.default_region = region
        self._cognito_clients = {}  # Cache boto3 clients by region
        self._jwks_cache = {}  # Cache JWKS by user pool

    def _get_cognito_client(self, region: str):
        """Get or create boto3 cognito client for region"""
        if region not in self._cognito_clients:
            self._cognito_clients[region] = boto3.client("cognito-idp", region_name=region)
        return self._cognito_clients[region]

    def _get_jwks(self, user_pool_id: str, region: str) -> dict:
        """
        Get JSON Web Key Set (JWKS) from Cognito with caching
        """
        cache_key = f"{region}:{user_pool_id}"

        if cache_key not in self._jwks_cache:
            try:
                issuer = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"
                jwks_url = f"{issuer}/.well-known/jwks.json"

                response = requests.get(jwks_url, timeout=10)
                response.raise_for_status()
                jwks = response.json()

                self._jwks_cache[cache_key] = jwks
                logger.debug(
                    f"Retrieved JWKS for {cache_key} with {len(jwks.get('keys', []))} keys"
                )

            except Exception as e:
                logger.error(f"Failed to retrieve JWKS from {jwks_url}: {e}")
                raise ValueError(f"Cannot retrieve JWKS: {e}")

        return self._jwks_cache[cache_key]

    def validate_jwt_token(
        self, access_token: str, user_pool_id: str, client_id: str, region: str = None
    ) -> dict:
        """
        Validate JWT access token

        Args:
            access_token: The bearer token to validate
            user_pool_id: Cognito User Pool ID
            client_id: Expected client ID
            region: AWS region (uses default if not provided)

        Returns:
            Dict containing token claims if valid

        Raises:
            ValueError: If token is invalid
        """
        if not region:
            region = self.default_region

        try:
            # Decode header to get key ID
            unverified_header = jwt.get_unverified_header(access_token)
            kid = unverified_header.get("kid")

            if not kid:
                raise ValueError("Token missing 'kid' in header")

            # Get JWKS and find matching key
            jwks = self._get_jwks(user_pool_id, region)
            signing_key = None

            for key in jwks.get("keys", []):
                if key.get("kid") == kid:
                    # Handle different versions of PyJWT
                    try:
                        # For newer versions of PyJWT
                        from jwt.algorithms import RSAAlgorithm

                        signing_key = RSAAlgorithm.from_jwk(key)
                    except (ImportError, AttributeError):
                        try:
                            # For older versions of PyJWT
                            from jwt.algorithms import get_default_algorithms

                            algorithms = get_default_algorithms()
                            signing_key = algorithms["RS256"].from_jwk(key)
                        except (ImportError, AttributeError):
                            # For PyJWT 2.0.0+
                            signing_key = PyJWK.from_jwk(json.dumps(key)).key
                    break

            if not signing_key:
                raise ValueError(f"No matching key found for kid: {kid}")

            # Set up issuer for validation
            issuer = f"https://cognito-idp.{region}.amazonaws.com/{user_pool_id}"

            # Validate and decode token
            claims = jwt.decode(
                access_token,
                signing_key,
                algorithms=["RS256"],
                issuer=issuer,
                options={
                    "verify_aud": False,  # M2M tokens might not have audience
                    "verify_exp": True,  # Always check expiration
                    "verify_iat": True,  # Check issued at time
                },
            )

            # Additional validations
            token_use = claims.get("token_use")
            if token_use not in ["access", "id"]:  # Allow both access and id tokens
                raise ValueError(f"Invalid token_use: {token_use}")

            # For M2M tokens, check client_id
            token_client_id = claims.get("client_id")
            if token_client_id and token_client_id != client_id:
                logger.warning("Token issued for different client than expected")
                # Don't fail immediately - could be user token with different structure

            logger.info("Successfully validated JWT token for client/user")
            return claims

        except jwt.ExpiredSignatureError:
            error_msg = "Token has expired"
            logger.warning(error_msg)
            raise ValueError(error_msg)
        except jwt.InvalidTokenError as e:
            error_msg = f"Invalid token: {e}"
            logger.warning(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"JWT validation error: {e}"
            logger.error(error_msg)
            raise ValueError(f"Token validation failed: {e}")

    def validate_with_boto3(self, access_token: str, region: str = None) -> dict:
        """
        Validate token using boto3 GetUser API (works for user tokens)

        Args:
            access_token: The bearer token to validate
            region: AWS region

        Returns:
            Dict containing user information if valid

        Raises:
            ValueError: If token is invalid
        """
        if not region:
            region = self.default_region

        try:
            cognito_client = self._get_cognito_client(region)
            response = cognito_client.get_user(AccessToken=access_token)

            # Extract user attributes
            user_attributes = {}
            for attr in response.get("UserAttributes", []):
                user_attributes[attr["Name"]] = attr["Value"]

            result = {
                "username": response.get("Username"),
                "user_attributes": user_attributes,
                "user_status": response.get("UserStatus"),
                "token_use": "access",  # boto3 method implies access token
                "auth_method": "boto3",
            }

            logger.info(
                f"Successfully validated token via boto3 for user {hash_username(result['username'])}"
            )
            return result

        except ClientError as e:
            error_code = e.response["Error"]["Code"]
            error_message = e.response["Error"]["Message"]

            if error_code == "NotAuthorizedException":
                error_msg = "Invalid or expired access token"
                logger.warning(f"Cognito error {error_code}: {error_message}")
                raise ValueError(error_msg)
            elif error_code == "UserNotFoundException":
                error_msg = "User not found"
                logger.warning(f"Cognito error {error_code}: {error_message}")
                raise ValueError(error_msg)
            else:
                logger.error(f"Cognito error {error_code}: {error_message}")
                raise ValueError(f"Token validation failed: {error_message}")

        except Exception as e:
            logger.error(f"Boto3 validation error: {e}")
            raise ValueError(f"Token validation failed: {e}")

    def validate_self_signed_token(self, access_token: str) -> dict:
        """
        Validate self-signed JWT token generated by this auth server.

        Args:
            access_token: The JWT token to validate

        Returns:
            Dict containing validation results

        Raises:
            ValueError: If token is invalid
        """
        try:
            # Decode and validate JWT using shared SECRET_KEY
            claims = jwt.decode(
                access_token,
                SECRET_KEY,
                algorithms=["HS256"],
                issuer=JWT_ISSUER,
                audience=JWT_AUDIENCE,
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": True,
                },
                leeway=30,  # 30 second leeway for clock skew
            )

            # Validate token_use
            token_use = claims.get("token_use")
            if token_use != "access":
                raise ValueError(f"Invalid token_use: {token_use}")

            # Extract scopes from space-separated string
            scope_string = claims.get("scope", "")
            scopes = scope_string.split() if scope_string else []

            # Extract groups from claims (for OAuth user tokens)
            groups = claims.get("groups", [])
            if isinstance(groups, str):
                groups = [groups]

            logger.info(
                f"Successfully validated self-signed token for user: {claims.get('sub')}, "
                f"groups: {groups}"
            )

            return {
                "valid": True,
                "method": "self_signed",
                "data": claims,
                "client_id": claims.get("client_id", "user-generated"),
                "username": claims.get("sub", ""),
                "expires_at": claims.get("exp"),
                "scopes": scopes,
                "groups": groups,
                "token_type": "user_generated",
            }

        except jwt.ExpiredSignatureError:
            error_msg = "Self-signed token has expired"
            logger.warning(error_msg)
            raise ValueError(error_msg)
        except jwt.InvalidTokenError as e:
            error_msg = f"Invalid self-signed token: {e}"
            logger.warning(error_msg)
            raise ValueError(error_msg)
        except Exception as e:
            error_msg = f"Self-signed token validation error: {e}"
            logger.error(error_msg)
            raise ValueError(f"Self-signed token validation failed: {e}")

    def validate_token(
        self, access_token: str, user_pool_id: str, client_id: str, region: str = None
    ) -> dict:
        """
        Comprehensive token validation with fallback methods.
        Now supports both Cognito tokens and self-signed tokens.

        Args:
            access_token: The bearer token to validate
            user_pool_id: Cognito User Pool ID
            client_id: Expected client ID
            region: AWS region

        Returns:
            Dict containing validation results and token information
        """
        if not region:
            region = self.default_region

        # First try self-signed token validation (faster)
        try:
            # Quick check if it might be our token by attempting to decode without verification
            unverified_claims = jwt.decode(access_token, options={"verify_signature": False})
            if unverified_claims.get("iss") == JWT_ISSUER:
                logger.debug("Token appears to be self-signed, validating...")
                return self.validate_self_signed_token(access_token)
        except Exception:
            # Not our token or malformed, continue to Cognito validation
            pass

        # Try JWT validation with Cognito
        try:
            jwt_claims = self.validate_jwt_token(access_token, user_pool_id, client_id, region)

            # Extract scopes and other info
            scopes = []
            if "scope" in jwt_claims:
                scopes = jwt_claims["scope"].split() if jwt_claims["scope"] else []

            return {
                "valid": True,
                "method": "jwt",
                "data": jwt_claims,
                "client_id": jwt_claims.get("client_id") or "",
                "username": jwt_claims.get("cognito:username") or jwt_claims.get("username") or "",
                "expires_at": jwt_claims.get("exp"),
                "scopes": scopes,
                "groups": jwt_claims.get("cognito:groups", []),
            }

        except ValueError as jwt_error:
            logger.debug(f"JWT validation failed: {jwt_error}, trying boto3")

            # Try boto3 validation as fallback
            try:
                boto3_data = self.validate_with_boto3(access_token, region)

                return {
                    "valid": True,
                    "method": "boto3",
                    "data": boto3_data,
                    "client_id": "",  # boto3 method doesn't provide client_id
                    "username": boto3_data.get("username") or "",
                    "user_attributes": boto3_data.get("user_attributes", {}),
                    "scopes": [],  # boto3 method doesn't provide scopes
                    "groups": [],
                }

            except ValueError as boto3_error:
                logger.debug(f"Boto3 validation failed: {boto3_error}")
                raise ValueError(
                    f"All validation methods failed. JWT: {jwt_error}, Boto3: {boto3_error}"
                )


# Create global validator instance
validator = SimplifiedCognitoValidator()


def _is_registry_api_request(
    original_url: str,
) -> bool:
    """Check if the request is for the Registry API (vs MCP Gateway).

    Registry API requests include:
    - /api/* - Core registry operations
    - /v0.1/* - Anthropic registry API and A2A agent API

    Args:
        original_url: The X-Original-URL header value from nginx.

    Returns:
        True if this is a registry API request, False if MCP gateway request.
    """
    if not original_url:
        return False

    parsed = urlparse(original_url)
    path = parsed.path

    for pattern in REGISTRY_API_PATTERNS:
        if path.startswith(pattern):
            return True

    return False


def _is_federation_api_request(
    original_url: str,
) -> bool:
    """Check if the request is for federation or peer management APIs.

    Args:
        original_url: The X-Original-URL header value from nginx.

    Returns:
        True if this is a federation/peer API request.
    """
    if not original_url:
        return False

    parsed = urlparse(original_url)
    path = parsed.path

    for pattern in FEDERATION_API_PATTERNS:
        if path.startswith(pattern):
            return True

    return False


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {"status": "healthy", "service": "simplified-auth-server"}


@app.get("/validate")
async def validate_request(request: Request):
    """
    Validate a request by extracting configuration from headers and validating the bearer token.

    Expected headers:
    - Authorization: Bearer <token>
    - X-User-Pool-Id: <user_pool_id>
    - X-Client-Id: <client_id>
    - X-Region: <region> (optional, defaults to us-east-1)
    - X-Original-URL: <original_url> (optional, for scope validation)

    Returns:
        HTTP 200 with user info headers if valid, HTTP 401/403 if invalid

    Raises:
        HTTPException: If the token is missing, invalid, or configuration is incomplete
    """

    # Capture start time for MCP audit logging
    import uuid

    start_time = time.perf_counter()
    request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
    mcp_session_id = request.headers.get("Mcp-Session-Id")

    try:
        # Extract headers
        # Check for X-Authorization first (custom header used by this gateway)
        # Only if X-Authorization is not present, check standard Authorization header
        authorization = request.headers.get("X-Authorization")
        if not authorization:
            authorization = request.headers.get("Authorization")
        cookie_header = request.headers.get("Cookie", "")
        user_pool_id = request.headers.get("X-User-Pool-Id")
        client_id = request.headers.get("X-Client-Id")
        region = request.headers.get("X-Region", "us-east-1")
        original_url = request.headers.get("X-Original-URL")
        body = request.headers.get("X-Body")

        # Extract server_name and endpoint from original_url early for logging
        server_name_from_url = None
        endpoint_from_url = None
        if original_url:
            try:
                parsed_url = urlparse(original_url)
                path = parsed_url.path.strip("/")

                # Strip the registry's root path prefix so server_name extraction
                # works correctly when the registry is hosted on a sub-path (e.g. /registry)
                registry_prefix = REGISTRY_ROOT_PATH.strip("/")
                if registry_prefix and path.startswith(registry_prefix):
                    path = path[len(registry_prefix) :].lstrip("/")

                path_parts = path.split("/") if path else []

                # MCP endpoints that should be treated as endpoints, not server names
                mcp_endpoints = {"mcp", "sse", "messages"}

                # For peer/federated registries, path is: peer-name/server-name/endpoint
                # For local servers, path is: server-name/endpoint
                # We need to capture the full server path, excluding the MCP endpoint
                if len(path_parts) >= 2 and path_parts[-1] in mcp_endpoints:
                    # Last part is MCP endpoint, everything before is server path
                    server_name_from_url = "/".join(path_parts[:-1])
                    endpoint_from_url = path_parts[-1]
                elif len(path_parts) >= 1:
                    # No recognized MCP endpoint at end - use entire path as server name
                    # This handles MCP server URLs like /peer-registry-lob-1/cloudflare-docs
                    # BUT exclude /api/ paths - those are Registry API requests, not MCP servers
                    if path_parts[0] != "api":
                        server_name_from_url = "/".join(path_parts)
                        endpoint_from_url = None

                logger.info(
                    f"Extracted server_name '{server_name_from_url}' and endpoint '{endpoint_from_url}' from original_url: {original_url}"
                )
            except Exception as e:
                logger.warning(
                    f"Failed to extract server_name from original_url {original_url}: {e}"
                )

        # Read request body
        request_payload = None
        try:
            if body:
                payload_text = body  # .decode('utf-8')
                logger.info(
                    f"Raw Request Payload ({len(payload_text)} chars): {payload_text[:1000]}..."
                )
                request_payload = json.loads(payload_text)
                logger.info(f"JSON RPC Request Payload: {json.dumps(request_payload, indent=2)}")
            else:
                logger.info("No request body provided, skipping payload parsing")
        except UnicodeDecodeError as e:
            logger.warning(f"Could not decode body as UTF-8: {e}")
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse JSON RPC payload: {e}")
        except Exception as e:
            logger.error(f"Error reading request payload: {type(e).__name__}: {e}")

        # Log request for debugging with anonymized IP
        client_ip = get_client_ip(request)
        logger.info(f"Validation request from {anonymize_ip(client_ip)}")
        logger.info(f"Request Method: {request.method}")

        # Log masked HTTP headers for GDPR/SOX compliance
        all_headers = dict(request.headers)
        masked_headers = mask_headers(all_headers)
        logger.debug(f"HTTP Headers (masked): {json.dumps(masked_headers, indent=2)}")

        # Log specific headers for debugging with masked sensitive data
        logger.info(
            f"Key Headers: Authorization={bool(authorization)}, Cookie={bool(cookie_header)}, "
            f"User-Pool-Id={mask_sensitive_id(user_pool_id) if user_pool_id else 'None'}, "
            f"Client-Id={mask_sensitive_id(client_id) if client_id else 'None'}, "
            f"Region={region}, Original-URL={original_url}"
        )
        logger.info(f"Server Name from URL: {server_name_from_url}")

        # Only activate static token auth when there is no session cookie
        # (UI uses cookies, CLI uses Bearer)
        has_session_cookie = cookie_header and "mcp_gateway_session=" in cookie_header

        # Federation static token auth: scoped access to federation/peer endpoints only
        # Check this BEFORE the full admin static token
        if (
            FEDERATION_STATIC_TOKEN_AUTH_ENABLED
            and _is_federation_api_request(original_url)
            and not has_session_cookie
        ):
            if not authorization:
                logger.warning(
                    "Federation static token: Authorization header missing. "
                    "Hint: Use 'Authorization: Bearer <FEDERATION_STATIC_TOKEN>'."
                )
                return JSONResponse(
                    content={"detail": "Authorization header required"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer", "Connection": "close"},
                )

            if not authorization.startswith("Bearer "):
                logger.warning(
                    "Federation static token: Authorization header must use Bearer scheme"
                )
                return JSONResponse(
                    content={"detail": "Authorization header must use Bearer scheme"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer", "Connection": "close"},
                )

            bearer_token = authorization[len("Bearer ") :].strip()

            # Check federation token first, then fall through to admin token check
            if hmac.compare_digest(bearer_token, FEDERATION_STATIC_TOKEN):
                logger.info(f"Federation static token: Authenticated for {original_url}")

                federation_scopes = [
                    "federation/read",
                    "federation/peers",
                ]
                response_data = {
                    "valid": True,
                    "username": "federation-peer",
                    "client_id": "federation-static",
                    "scopes": federation_scopes,
                    "method": "federation-static",
                    "groups": [],
                    "server_name": None,
                    "tool_name": None,
                }

                response = JSONResponse(content=response_data, status_code=200)
                response.headers["X-User"] = "federation-peer"
                response.headers["X-Username"] = "federation-peer"
                response.headers["X-Client-Id"] = "federation-static"
                response.headers["X-Scopes"] = " ".join(federation_scopes)
                response.headers["X-Auth-Method"] = "federation-static"
                response.headers["X-Server-Name"] = ""
                response.headers["X-Tool-Name"] = ""

                return response

            # If federation token didn't match, DON'T return 403 here.
            # Fall through to the admin static token check below (if enabled).
            # If admin token also doesn't match, that block will return 403.
            # If admin token is NOT enabled, fall through to JWT validation.

        # Static token auth: validate Registry API requests with static API key
        if (
            REGISTRY_STATIC_TOKEN_AUTH_ENABLED
            and _is_registry_api_request(original_url)
            and not has_session_cookie
        ):
            if not authorization:
                logger.warning(
                    "Network-trusted mode enabled but Authorization header missing. "
                    "Hint: Use 'Authorization: Bearer <REGISTRY_API_TOKEN>'."
                )
                return JSONResponse(
                    content={"detail": "Authorization header required"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer", "Connection": "close"},
                )

            # Validate static API key (REGISTRY_API_TOKEN is guaranteed to be set here)
            if not authorization.startswith("Bearer "):
                logger.warning("Static token auth: Authorization header must use Bearer scheme")
                return JSONResponse(
                    content={"detail": "Authorization header must use Bearer scheme"},
                    status_code=401,
                    headers={"WWW-Authenticate": "Bearer", "Connection": "close"},
                )

            bearer_token = authorization[len("Bearer ") :].strip()
            if not hmac.compare_digest(bearer_token, REGISTRY_API_TOKEN):
                logger.warning("Static token auth: Invalid API token provided")
                return JSONResponse(
                    content={"detail": "Invalid API token"},
                    status_code=403,
                    headers={"Connection": "close"},
                )

            logger.info(
                f"Network-trusted mode: Bypassing auth validation for registry API "
                f"request to {original_url}"
            )

            network_trusted_scopes = [
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ]
            response_data = {
                "valid": True,
                "username": "network-user",
                "client_id": "network-trusted",
                "scopes": network_trusted_scopes,
                "method": "network-trusted",
                "groups": ["mcp-registry-admin"],
                "server_name": None,
                "tool_name": None,
            }

            response = JSONResponse(content=response_data, status_code=200)
            response.headers["X-User"] = "network-user"
            response.headers["X-Username"] = "network-user"
            response.headers["X-Client-Id"] = "network-trusted"
            response.headers["X-Scopes"] = " ".join(network_trusted_scopes)
            response.headers["X-Auth-Method"] = "network-trusted"
            response.headers["X-Server-Name"] = ""
            response.headers["X-Tool-Name"] = ""

            return response

        # Initialize validation result
        validation_result = None

        # FIRST: Check for session cookie if present
        if "mcp_gateway_session=" in cookie_header:
            logger.info("Session cookie detected, attempting session validation")
            # Extract cookie value
            cookie_value = None
            for cookie in cookie_header.split(";"):
                if cookie.strip().startswith("mcp_gateway_session="):
                    cookie_value = cookie.strip().split("=", 1)[1]
                    break

            if cookie_value:
                try:
                    validation_result = await validate_session_cookie(cookie_value)
                    # Log validation result without exposing username or tokens
                    safe_result = _mask_sensitive_dict(validation_result)
                    safe_result["username"] = hash_username(validation_result.get("username", ""))
                    logger.info(f"Session cookie validation result: {safe_result}")
                    logger.info(
                        f"Session cookie validation successful for user: {hash_username(validation_result['username'])}"
                    )
                except ValueError as e:
                    logger.warning(f"Session cookie validation failed: {e}")
                    # Fall through to JWT validation

        # SECOND: If no valid session cookie, check for JWT token
        if not validation_result:
            # Validate required headers for JWT
            if not authorization or not authorization.startswith("Bearer "):
                logger.warning(
                    "Missing or invalid Authorization header and no valid session cookie"
                )
                raise HTTPException(
                    status_code=401,
                    detail="Missing or invalid Authorization header. Expected: Bearer <token> or valid session cookie",
                    headers={"WWW-Authenticate": "Bearer", "Connection": "close"},
                )

            # Extract token
            access_token = authorization.split(" ")[1]

            # Get authentication provider based on AUTH_PROVIDER environment variable
            try:
                auth_provider = get_auth_provider()
                logger.info(f"Using authentication provider: {auth_provider.__class__.__name__}")

                # Provider-specific validation
                if hasattr(auth_provider, "validate_token"):
                    # For Keycloak, no additional headers needed
                    validation_result = auth_provider.validate_token(access_token)
                    logger.info(
                        f"Token validation successful using {auth_provider.__class__.__name__}"
                    )
                else:
                    # Fallback to old validation for compatibility
                    if not user_pool_id:
                        logger.warning("Missing X-User-Pool-Id header for Cognito validation")
                        raise HTTPException(
                            status_code=400,
                            detail="Missing X-User-Pool-Id header",
                            headers={"Connection": "close"},
                        )

                    if not client_id:
                        logger.warning("Missing X-Client-Id header for Cognito validation")
                        raise HTTPException(
                            status_code=400,
                            detail="Missing X-Client-Id header",
                            headers={"Connection": "close"},
                        )

                    # Use old validator for backward compatibility
                    validation_result = validator.validate_token(
                        access_token=access_token,
                        user_pool_id=user_pool_id,
                        client_id=client_id,
                        region=region,
                    )

            except Exception as e:
                logger.error(f"Authentication provider error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Authentication provider configuration error",
                    headers={"Connection": "close"},
                )

        logger.info(f"Token validation successful using method: {validation_result['method']}")

        # Parse server and tool information from original URL if available
        server_name = server_name_from_url  # Use the server_name we extracted earlier
        tool_name = None

        if original_url and request_payload:
            # We already extracted server_name above, now just get tool_name from URL parsing
            _, tool_name = parse_server_and_tool_from_url(original_url)
            logger.debug(f"Parsed from original URL: server='{server_name}', tool='{tool_name}'")

            # Try to extract tool name from request payload if not found in URL
            if server_name and not tool_name and request_payload:
                try:
                    # Look for tool name in JSON-RPC 2.0 format and other MCP patterns
                    if isinstance(request_payload, dict):
                        # JSON-RPC 2.0 format: method field contains the tool name
                        tool_name = request_payload.get("method")

                        # If not found in method, check other common patterns
                        if not tool_name:
                            tool_name = request_payload.get("tool") or request_payload.get("name")

                        # Check for nested tool reference in params
                        if not tool_name and "params" in request_payload:
                            params = request_payload["params"]
                            if isinstance(params, dict):
                                tool_name = (
                                    params.get("name") or params.get("tool") or params.get("method")
                                )

                        logger.info(f"Extracted tool name from JSON-RPC payload: '{tool_name}'")
                    else:
                        logger.warning(f"Payload is not a dictionary: {type(request_payload)}")
                except Exception as e:
                    logger.error(f"Error processing request payload for tool extraction: {e}")

        # Validate scope-based access if we have server/tool information
        # For providers that use groups (Keycloak, Entra ID, Cognito), map groups to scopes
        user_groups = validation_result.get("groups", [])
        auth_method = validation_result.get("method", "")
        if user_groups and auth_method in ["keycloak", "entra", "cognito"]:
            # Map IdP groups to scopes using the group mappings (query DocumentDB)
            user_scopes = await map_groups_to_scopes(user_groups)
            logger.info(f"Mapped {auth_method} groups {user_groups} to scopes: {user_scopes}")
        else:
            user_scopes = validation_result.get("scopes", [])
        if server_name:
            # For ANY server access, enforce scope validation (fail closed principle)
            # This includes MCP initialization methods that may not have a specific tool

            # Determine the method to validate:
            # 1. If we have a tool_name from JSON-RPC payload, use that
            # 2. If we have an endpoint from the REST API URL, use that
            # 3. Otherwise default to "initialize"
            method = (
                tool_name
                if tool_name
                else (endpoint_from_url if endpoint_from_url else "initialize")
            )
            logger.info(
                f"Method determined for validation: '{method}' (tool_name={tool_name}, endpoint_from_url={endpoint_from_url})"
            )
            actual_tool_name = None

            # For tools/call, extract the actual tool name from params
            if method == "tools/call" and isinstance(request_payload, dict):
                params = request_payload.get("params", {})
                if isinstance(params, dict):
                    actual_tool_name = params.get("name")
                    logger.info(f"Extracted actual tool name for tools/call: '{actual_tool_name}'")

            # Check if user has any scopes - if not, deny access (fail closed)
            if not user_scopes:
                logger.warning(
                    f"Access denied for user {hash_username(validation_result.get('username', ''))} to {server_name}.{method} (tool: {actual_tool_name}) - no scopes configured"
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied to {server_name}.{method} - user has no scopes configured",
                    headers={"Connection": "close"},
                )

            if not await validate_server_tool_access(
                server_name, method, actual_tool_name, user_scopes
            ):
                logger.warning(
                    f"Access denied for user {hash_username(validation_result.get('username', ''))} to {server_name}.{method} (tool: {actual_tool_name})"
                )
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied to {server_name}.{method}",
                    headers={"Connection": "close"},
                )
            logger.info(
                f"Scope validation passed for {server_name}.{method} (tool: {actual_tool_name})"
            )
        else:
            logger.debug("No server information available, skipping scope validation")

        # Prepare JSON response data
        response_data = {
            "valid": True,
            "username": validation_result.get("username") or "",
            "client_id": validation_result.get("client_id") or "",
            "scopes": user_scopes,
            "method": validation_result.get("method") or "",
            "groups": validation_result.get("groups", []),
            "server_name": server_name,
            "tool_name": tool_name,
        }
        logger.info(
            f"Full validation result: {json.dumps(_mask_sensitive_dict(validation_result), indent=2)}"
        )
        logger.info(f"Response data being sent: {json.dumps(response_data, indent=2)}")

        # Log MCP server access event if this is an MCP request (has server_name)
        if server_name:
            duration_ms = (time.perf_counter() - start_time) * 1000
            mcp_logger = get_mcp_logger()
            if mcp_logger:
                try:
                    # Build identity from validation result
                    identity = Identity(
                        username=validation_result.get("username") or "anonymous",
                        auth_method=validation_result.get("method") or "unknown",
                        provider=validation_result.get("provider"),
                        groups=validation_result.get("groups", []),
                        scopes=user_scopes,
                        is_admin=validation_result.get("is_admin", False),
                        credential_type="bearer_token" if authorization else "session_cookie",
                    )

                    # Build MCP server info
                    mcp_server = MCPServer(
                        name=server_name,
                        path=f"/{server_name}" if server_name else "/",
                        proxy_target=original_url or "",
                    )

                    # Log the MCP access event
                    await mcp_logger.log_mcp_access(
                        request_id=request_id,
                        identity=identity,
                        mcp_server=mcp_server,
                        request_body=body.encode("utf-8") if body else b"",
                        response_status="success",
                        duration_ms=duration_ms,
                        mcp_session_id=mcp_session_id,
                        transport="streamable-http",  # Default, could be extracted from request
                        client_ip=get_client_ip(request),
                        forwarded_for=request.headers.get("X-Forwarded-For"),
                        user_agent=request.headers.get("User-Agent"),
                    )
                    logger.debug(f"MCP access logged for {server_name}")
                except Exception as e:
                    # Don't fail the request if logging fails
                    logger.warning(f"Failed to log MCP access event: {e}")

        # Create JSON response with headers that nginx can use
        response = JSONResponse(content=response_data, status_code=200)

        # Set headers for nginx auth_request_set directives
        response.headers["X-User"] = validation_result.get("username") or ""
        response.headers["X-Username"] = validation_result.get("username") or ""
        response.headers["X-Client-Id"] = validation_result.get("client_id") or ""
        response.headers["X-Scopes"] = " ".join(user_scopes)
        response.headers["X-Auth-Method"] = validation_result.get("method") or ""
        response.headers["X-Server-Name"] = server_name or ""
        response.headers["X-Tool-Name"] = tool_name or ""

        return response

    except ValueError as e:
        logger.warning(f"Token validation failed: {e}")
        # Log failed MCP access attempt
        if server_name_from_url:
            duration_ms = (time.perf_counter() - start_time) * 1000
            mcp_logger = get_mcp_logger()
            if mcp_logger:
                try:
                    identity = Identity(
                        username="anonymous",
                        auth_method="unknown",
                        credential_type="none",
                    )
                    mcp_server = MCPServer(
                        name=server_name_from_url,
                        path=f"/{server_name_from_url}",
                        proxy_target=original_url or "",
                    )
                    await mcp_logger.log_mcp_access(
                        request_id=request_id,
                        identity=identity,
                        mcp_server=mcp_server,
                        request_body=body.encode("utf-8") if body else b"",
                        response_status="error",
                        duration_ms=duration_ms,
                        mcp_session_id=mcp_session_id,
                        error_code=401,
                        error_message=str(e),
                        client_ip=get_client_ip(request),
                        forwarded_for=request.headers.get("X-Forwarded-For"),
                        user_agent=request.headers.get("User-Agent"),
                    )
                except Exception as log_err:
                    logger.warning(f"Failed to log MCP access error: {log_err}")
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer", "Connection": "close"},
        )
    except HTTPException as e:
        # Re-raise client error HTTPExceptions (4xx) as-is
        if 400 <= e.status_code < 500:
            raise
        # For non-client HTTPExceptions, convert to 500
        logger.error(f"HTTP error during validation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal validation error: {str(e)}",
            headers={"Connection": "close"},
        )
    except Exception as e:
        logger.error(f"Unexpected error during validation: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Internal validation error: {str(e)}",
            headers={"Connection": "close"},
        )
    finally:
        pass


@app.get("/config")
async def get_auth_config():
    """Return the authentication configuration info"""
    try:
        auth_provider = get_auth_provider()
        provider_info = auth_provider.get_provider_info()

        if provider_info.get("provider_type") == "keycloak":
            return {
                "auth_type": "keycloak",
                "description": "Keycloak JWT token validation",
                "required_headers": ["Authorization: Bearer <token>"],
                "optional_headers": [],
                "provider_info": provider_info,
            }
        else:
            return {
                "auth_type": "cognito",
                "description": "Header-based Cognito token validation",
                "required_headers": [
                    "Authorization: Bearer <token>",
                    "X-User-Pool-Id: <pool_id>",
                    "X-Client-Id: <client_id>",
                ],
                "optional_headers": ["X-Region: <region> (default: us-east-1)"],
                "provider_info": provider_info,
            }
    except Exception as e:
        logger.error(f"Error getting auth config: {e}")
        return {
            "auth_type": "unknown",
            "description": f"Error getting provider config: {e}",
            "error": str(e),
        }


@app.post("/admin/federation-token")
async def manage_federation_token(request: Request):
    """Revoke or rotate federation static token at runtime.

    Requires the admin static token (REGISTRY_API_TOKEN) for authentication.
    """
    global FEDERATION_STATIC_TOKEN, FEDERATION_STATIC_TOKEN_AUTH_ENABLED

    # Authenticate with admin token
    authorization = request.headers.get("Authorization", "")
    if not authorization.startswith("Bearer "):
        return JSONResponse(
            content={"detail": "Bearer token required"},
            status_code=401,
        )

    bearer_token = authorization[len("Bearer ") :].strip()
    if not REGISTRY_API_TOKEN or not hmac.compare_digest(bearer_token, REGISTRY_API_TOKEN):
        return JSONResponse(
            content={"detail": "Admin token required"},
            status_code=403,
        )

    body = await request.json()
    new_token = body.get("new_token")

    # Validate minimum token length if a new token is provided
    if new_token and len(new_token) < MIN_FEDERATION_TOKEN_LENGTH:
        return JSONResponse(
            content={
                "detail": (
                    f"Token must be at least {MIN_FEDERATION_TOKEN_LENGTH} characters. "
                    'Generate with: python3 -c "import secrets; print(secrets.token_urlsafe(32))"'
                )
            },
            status_code=400,
        )

    if new_token:
        FEDERATION_STATIC_TOKEN = new_token
        FEDERATION_STATIC_TOKEN_AUTH_ENABLED = True
        logger.info("Federation static token rotated via admin API")
        return {
            "action": "rotated",
            "message": (
                "Federation static token rotated. "
                "WARNING: This is an in-memory change only. Update FEDERATION_STATIC_TOKEN "
                "in your .env file or container environment for persistence across restarts."
            ),
        }
    else:
        FEDERATION_STATIC_TOKEN = ""
        FEDERATION_STATIC_TOKEN_AUTH_ENABLED = False
        logger.info("Federation static token revoked via admin API")
        return {
            "action": "revoked",
            "message": (
                "Federation static token revoked. Federation endpoints now require OAuth2 JWT. "
                "WARNING: This is an in-memory change only. Update your .env file or container "
                "environment to set FEDERATION_STATIC_TOKEN_AUTH_ENABLED=false for persistence "
                "across restarts."
            ),
        }


@app.post("/internal/tokens", response_model=GenerateTokenResponse)
async def generate_user_token(request: GenerateTokenRequest):
    """
    Generate or refresh a JWT token for a user.

    This endpoint supports two modes:
    1. If user has stored OAuth tokens (from login), refresh them if needed and return
    2. Otherwise, fall back to generating M2M token using client credentials

    This is an internal API endpoint meant to be called only by the registry service.
    The generated token will have the same or fewer privileges than the user currently has.

    Args:
        request: Token generation request containing user context and requested scopes

    Returns:
        JWT token with expiration info (either refreshed user token or M2M token)

    Raises:
        HTTPException: If request is invalid or user doesn't have required permissions
    """
    try:
        # Extract user context
        user_context = request.user_context
        username = user_context.get("username")
        user_scopes = user_context.get("scopes", [])

        if not username:
            raise HTTPException(
                status_code=400,
                detail="Username is required in user context",
                headers={"Connection": "close"},
            )

        # Check rate limiting
        if not check_rate_limit(username):
            raise HTTPException(
                status_code=429,
                detail=f"Rate limit exceeded. Maximum {MAX_TOKENS_PER_USER_PER_HOUR} tokens per hour.",
                headers={"Connection": "close"},
            )

        # Use user's current scopes if no specific scopes requested
        requested_scopes = request.requested_scopes if request.requested_scopes else user_scopes

        # Validate that requested scopes are subset of user's current scopes
        if not validate_scope_subset(user_scopes, requested_scopes):
            invalid_scopes = set(requested_scopes) - set(user_scopes)
            raise HTTPException(
                status_code=403,
                detail=f"Requested scopes exceed user permissions. Invalid scopes: {list(invalid_scopes)}",
                headers={"Connection": "close"},
            )

        # Check if user has stored OAuth tokens from their login session
        provider = user_context.get("provider")
        auth_method = user_context.get("auth_method")
        user_groups = user_context.get("groups", [])
        user_email = user_context.get("email", "")

        logger.info(
            f"Token request for user '{hash_username(username)}': "
            f"auth_method={auth_method}, provider={provider}, "
            f"groups={user_groups}, scopes={requested_scopes}"
        )

        # For OAuth users, generate a self-signed JWT with their identity and groups
        # This token is issued by our auth server and can be verified using SECRET_KEY
        if auth_method == "oauth2":
            logger.info(
                f"Generating self-signed JWT for OAuth user '{hash_username(username)}' "
                f"with groups: {user_groups}"
            )

            current_time = int(time.time())
            expires_in = DEFAULT_TOKEN_LIFETIME_HOURS * 3600  # 8 hours default

            # Build JWT claims
            jwt_claims = {
                "iss": JWT_ISSUER,
                "aud": JWT_AUDIENCE,
                "sub": username,
                "preferred_username": username,
                "email": user_email,
                "groups": user_groups,
                "scope": " ".join(requested_scopes) if requested_scopes else "",
                "token_use": "access",
                "auth_method": "oauth2",
                "provider": provider,
                "iat": current_time,
                "exp": current_time + expires_in,
                "description": request.description,
            }

            # Sign the JWT with our SECRET_KEY
            access_token = jwt.encode(jwt_claims, SECRET_KEY, algorithm="HS256")

            logger.info(
                f"Generated self-signed JWT for user '{hash_username(username)}', "
                f"expires in {expires_in} seconds"
            )

            return GenerateTokenResponse(
                access_token=access_token,
                refresh_token=None,
                expires_in=expires_in,
                refresh_expires_in=0,
                scope=" ".join(requested_scopes) if requested_scopes else "openid profile email",
                issued_at=current_time,
                description=request.description,
            )

        # Fall back to M2M token using client credentials flow
        try:
            auth_provider = get_auth_provider()
            provider_info = auth_provider.get_provider_info()
            provider_type = provider_info.get("provider_type", "unknown")

            logger.info(
                f"Generating M2M token for user '{hash_username(username)}' using {provider_type}"
            )

            if provider_type == "keycloak":
                # Request token from Keycloak using M2M client credentials
                token_data = auth_provider.get_m2m_token(scope="openid email profile")
            elif provider_type == "entra":
                # Request token from Entra ID using client credentials
                token_data = auth_provider.get_m2m_token()
            else:
                raise HTTPException(
                    status_code=500,
                    detail=f"Token generation not supported for provider: {provider_type}",
                    headers={"Connection": "close"},
                )

            access_token = token_data.get("access_token")
            refresh_token_value = token_data.get("refresh_token")
            expires_in = token_data.get("expires_in", 300)
            refresh_expires_in = token_data.get("refresh_expires_in", 0)
            scope = token_data.get("scope", "openid email profile")

            if not access_token:
                raise ValueError(f"No access token returned from {provider_type}")

            current_time = int(time.time())

            logger.info(
                f"Generated {provider_type} M2M token for user '{hash_username(username)}' "
                f"with scopes: {requested_scopes}, expires in {expires_in} seconds"
            )

            return GenerateTokenResponse(
                access_token=access_token,
                refresh_token=refresh_token_value,
                expires_in=expires_in,
                refresh_expires_in=refresh_expires_in,
                scope=scope,
                issued_at=current_time,
                description=request.description,
            )

        except ValueError as e:
            logger.error(f"Token generation failed: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate token: {e}",
                headers={"Connection": "close"},
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating token: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal error generating token",
            headers={"Connection": "close"},
        )


@app.post("/internal/reload-scopes")
async def reload_scopes(request: Request, authorization: str | None = Header(None)):
    """
    Reload the scopes configuration.

    Accepts internal service authentication via self-signed JWT (Bearer token)
    signed with the shared SECRET_KEY. Also accepts Basic Auth with
    ADMIN_USER/ADMIN_PASSWORD as a deprecated fallback.
    """
    if not authorization:
        logger.warning("No Authorization header found for reload-scopes request")
        raise HTTPException(
            status_code=401,
            detail="Authentication required",
            headers={"WWW-Authenticate": "Bearer"},
        )

    caller_identity = "unknown"

    if authorization.startswith("Bearer "):
        # Validate self-signed JWT using shared SECRET_KEY
        token = authorization.split(" ", 1)[1]
        try:
            claims = jwt.decode(
                token,
                SECRET_KEY,
                algorithms=["HS256"],
                issuer=JWT_ISSUER,
                audience=JWT_AUDIENCE,
                options={
                    "verify_exp": True,
                    "verify_iat": True,
                    "verify_iss": True,
                    "verify_aud": True,
                },
                leeway=30,
            )
            token_use = claims.get("token_use")
            if token_use != "access":
                raise ValueError(f"Invalid token_use: {token_use}")
            caller_identity = claims.get("sub", "service")
            logger.info(f"Reload-scopes authorized via JWT for: {caller_identity}")
        except jwt.ExpiredSignatureError:
            logger.warning("Expired JWT token for reload-scopes request")
            raise HTTPException(status_code=401, detail="Token has expired")
        except (jwt.InvalidTokenError, ValueError) as e:
            logger.warning(f"JWT validation failed for reload-scopes: {e}")
            raise HTTPException(status_code=401, detail="Invalid token")

    elif authorization.startswith("Basic "):
        # Deprecated: Basic Auth with ADMIN_USER/ADMIN_PASSWORD
        import base64

        try:
            encoded_credentials = authorization.split(" ")[1]
            decoded_credentials = base64.b64decode(encoded_credentials).decode("utf-8")
            username, password = decoded_credentials.split(":", 1)
        except (IndexError, ValueError, Exception) as e:
            logger.warning(f"Failed to decode Basic Auth credentials: {e}")
            raise HTTPException(
                status_code=401,
                detail="Invalid authentication format",
                headers={"WWW-Authenticate": "Basic"},
            )

        admin_user = os.environ.get("ADMIN_USER", "admin")
        admin_password = os.environ.get("ADMIN_PASSWORD")

        if not admin_password:
            logger.error("ADMIN_PASSWORD not set and Basic Auth attempted on reload-scopes")
            raise HTTPException(status_code=500, detail="Server configuration error")

        if username != admin_user or password != admin_password:
            logger.warning(f"Failed admin authentication attempt for reload-scopes from {username}")
            raise HTTPException(
                status_code=401,
                detail="Invalid admin credentials",
                headers={"WWW-Authenticate": "Basic"},
            )

        caller_identity = username
        logger.warning(
            "reload-scopes called with deprecated Basic Auth. "
            "Migrate to Bearer token using shared SECRET_KEY."
        )
    else:
        raise HTTPException(status_code=401, detail="Unsupported authentication scheme")

    # Reload the scopes configuration
    global SCOPES_CONFIG
    try:
        SCOPES_CONFIG = await reload_scopes_config()
        logger.info(f"Successfully reloaded scopes configuration by '{caller_identity}'")

        return JSONResponse(
            status_code=200,
            content={
                "message": "Scopes configuration reloaded successfully",
                "timestamp": datetime.utcnow().isoformat(),
                "group_mappings_count": len(SCOPES_CONFIG.get("group_mappings", {})),
            },
        )
    except Exception as e:
        logger.error(f"Failed to reload scopes configuration: {e}")
        raise HTTPException(status_code=500, detail="Failed to reload scopes configuration")


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simplified Auth Server")

    parser.add_argument(
        "--host",
        type=str,
        default="0.0.0.0",
        help="Host for the server to listen on (default: 0.0.0.0)",
    )

    parser.add_argument(
        "--port",
        type=int,
        default=8888,
        help="Port for the server to listen on (default: 8888)",
    )

    parser.add_argument(
        "--region",
        type=str,
        default="us-east-1",
        help="Default AWS region (default: us-east-1)",
    )

    return parser.parse_args()


def main():
    """Run the server"""
    args = parse_arguments()

    # Update global validator with default region
    global validator
    validator = SimplifiedCognitoValidator(region=args.region)

    logger.info(f"Starting simplified auth server on {args.host}:{args.port}")
    logger.info(f"Default region: {args.region}")

    uvicorn.run(app, host=args.host, port=args.port, proxy_headers=True, forwarded_allow_ips="*")


if __name__ == "__main__":
    main()


# Load OAuth2 providers configuration
def load_oauth2_config():
    """Load the OAuth2 providers configuration from oauth2_providers.yml"""
    try:
        oauth2_file = Path(__file__).parent / "oauth2_providers.yml"
        with open(oauth2_file) as f:
            config = yaml.safe_load(f)

        # Substitute environment variables in configuration
        processed_config = substitute_env_vars(config)
        return processed_config
    except Exception as e:
        logger.error(f"Failed to load OAuth2 configuration: {e}")
        return {"providers": {}, "session": {}, "registry": {}}


def auto_derive_cognito_domain(user_pool_id: str) -> str:
    """
    Auto-derive Cognito domain from User Pool ID.

    Example: us-east-1_KmP5A3La3 → us-east-1kmp5a3la3
    """
    if not user_pool_id:
        return ""

    # Remove underscore and convert to lowercase
    domain = user_pool_id.replace("_", "").lower()
    logger.info(f"Auto-derived Cognito domain '{domain}' from user pool ID '{user_pool_id}'")
    return domain


def substitute_env_vars(config):
    """Recursively substitute environment variables in configuration"""
    if isinstance(config, dict):
        return {k: substitute_env_vars(v) for k, v in config.items()}
    elif isinstance(config, list):
        return [substitute_env_vars(item) for item in config]
    elif isinstance(config, str) and "${" in config:
        try:
            # Handle special case for auto-derived Cognito domain
            if "COGNITO_DOMAIN:-auto" in config:
                # Check if COGNITO_DOMAIN is set, if not auto-derive from user pool ID
                cognito_domain = os.environ.get("COGNITO_DOMAIN")
                if not cognito_domain:
                    user_pool_id = os.environ.get("COGNITO_USER_POOL_ID", "")
                    cognito_domain = auto_derive_cognito_domain(user_pool_id)

                # Replace the template with the derived domain
                config = config.replace("${COGNITO_DOMAIN:-auto}", cognito_domain)

            template = Template(config)
            result = template.substitute(os.environ)

            # Convert string booleans to actual booleans
            if result.lower() == "true":
                return True
            elif result.lower() == "false":
                return False

            return result
        except KeyError as e:
            logger.warning(f"Environment variable not found for template {config}: {e}")
            return config
    else:
        return config


# Global OAuth2 configuration
OAUTH2_CONFIG = load_oauth2_config()

# Initialize SECRET_KEY and signer for session management
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    # Generate a secure random key (32 bytes = 256 bits of entropy)
    SECRET_KEY = secrets.token_hex(32)
    logger.warning(
        "No SECRET_KEY environment variable found. Using a randomly generated key. "
        "While this is more secure than a hardcoded default, it will change on restart. "
        "Set a permanent SECRET_KEY environment variable for production."
    )

signer = URLSafeTimedSerializer(SECRET_KEY)

# Initialize MCP audit logger for logging MCP server access events
# This logs all MCP requests that pass through the auth validation
_mcp_audit_logger = None
_mcp_logger = None
_mcp_audit_repository = None


def get_mcp_logger() -> MCPLogger | None:
    """Get or initialize the MCP logger instance."""
    global _mcp_audit_logger, _mcp_logger, _mcp_audit_repository

    if _mcp_logger is None:
        try:
            # Check if MCP audit logging is enabled via settings
            if settings.audit_log_enabled:
                # Initialize MongoDB repository if MongoDB is enabled
                audit_repository = None
                mongodb_enabled = getattr(settings, "audit_log_mongodb_enabled", False)
                if mongodb_enabled:
                    try:
                        from registry.repositories.audit_repository import DocumentDBAuditRepository

                        _mcp_audit_repository = DocumentDBAuditRepository()
                        audit_repository = _mcp_audit_repository
                        logger.info("MCP audit MongoDB repository initialized")
                    except Exception as e:
                        logger.warning(f"Failed to initialize MCP audit MongoDB repository: {e}")
                        mongodb_enabled = False

                _mcp_audit_logger = AuditLogger(
                    log_dir=settings.audit_log_dir,
                    rotation_hours=settings.audit_log_rotation_hours,
                    rotation_max_mb=settings.audit_log_rotation_max_mb,
                    local_retention_hours=settings.audit_log_local_retention_hours,
                    stream_name="mcp-server-access",
                    mongodb_enabled=mongodb_enabled,
                    audit_repository=audit_repository,
                )
                _mcp_logger = MCPLogger(_mcp_audit_logger)
                logger.info(
                    f"MCP audit logger initialized successfully (MongoDB: {mongodb_enabled})"
                )
            else:
                logger.info("MCP audit logging is disabled")
        except Exception as e:
            logger.warning(f"Failed to initialize MCP audit logger: {e}")
            _mcp_logger = None

    return _mcp_logger


def get_enabled_providers():
    """Get list of enabled OAuth2 providers, filtered by AUTH_PROVIDER env var if set"""
    enabled = []

    # Check if AUTH_PROVIDER env var is set to filter to only one provider
    auth_provider_env = os.getenv("AUTH_PROVIDER")

    # First, collect all enabled providers from YAML
    yaml_enabled_providers = []
    for provider_name, config in OAUTH2_CONFIG.get("providers", {}).items():
        if config.get("enabled", False):
            yaml_enabled_providers.append(provider_name)

    if auth_provider_env:
        logger.info(
            f"AUTH_PROVIDER is set to '{auth_provider_env}', filtering providers accordingly"
        )

        # Check if the specified provider exists in the config
        if auth_provider_env not in OAUTH2_CONFIG.get("providers", {}):
            logger.error(
                f"AUTH_PROVIDER '{auth_provider_env}' not found in oauth2_providers.yml configuration"
            )
            return []

        # Check if the specified provider is enabled in YAML
        provider_config = OAUTH2_CONFIG["providers"][auth_provider_env]
        if not provider_config.get("enabled", False):
            logger.warning(
                f"AUTH_PROVIDER '{auth_provider_env}' is set but this provider is disabled in oauth2_providers.yml"
            )
            logger.warning(
                f"To fix this, either set AUTH_PROVIDER to one of the enabled providers: {yaml_enabled_providers} or enable '{auth_provider_env}' in oauth2_providers.yml"
            )
            return []

        # Warn about providers being filtered out
        filtered_providers = [p for p in yaml_enabled_providers if p != auth_provider_env]
        if filtered_providers:
            logger.warning(
                f"AUTH_PROVIDER override: Filtering out enabled providers {filtered_providers} - only showing '{auth_provider_env}'"
            )
            logger.warning(
                "To show all enabled providers, remove the AUTH_PROVIDER environment variable"
            )
    else:
        logger.info("AUTH_PROVIDER not set, returning all enabled providers from config")

    for provider_name, config in OAUTH2_CONFIG.get("providers", {}).items():
        if config.get("enabled", False):
            # If AUTH_PROVIDER is set, only include that specific provider
            if auth_provider_env and provider_name != auth_provider_env:
                logger.debug(f"Skipping provider '{provider_name}' due to AUTH_PROVIDER filter")
                continue

            enabled.append(
                {
                    "name": provider_name,
                    "display_name": config.get("display_name", provider_name.title()),
                }
            )
            logger.debug(f"Enabled provider: {provider_name}")

    logger.info(f"Returning {len(enabled)} enabled providers: {[p['name'] for p in enabled]}")
    return enabled


@app.get("/oauth2/providers")
async def get_oauth2_providers():
    """Get list of enabled OAuth2 providers for the login page"""
    try:
        # Debug: log environment variable for troubleshooting
        auth_provider_env = os.getenv("AUTH_PROVIDER")
        logger.info(f"Debug: AUTH_PROVIDER environment variable = '{auth_provider_env}'")

        providers = get_enabled_providers()
        return {"providers": providers}
    except Exception as e:
        logger.error(f"Error getting OAuth2 providers: {e}")
        return {"providers": [], "error": str(e)}


@app.get("/oauth2/login/{provider}")
async def oauth2_login(provider: str, request: Request, redirect_uri: str = None):
    """Initiate OAuth2 login flow"""
    try:
        if provider not in OAUTH2_CONFIG.get("providers", {}):
            raise HTTPException(status_code=404, detail=f"Provider {provider} not found")

        provider_config = OAUTH2_CONFIG["providers"][provider]
        if not provider_config.get("enabled", False):
            raise HTTPException(status_code=400, detail=f"Provider {provider} is disabled")

        # Generate state parameter for security
        state = secrets.token_urlsafe(32)

        # Determine the OAuth2 callback URI based on the request origin
        # This is critical for dual-mode (CloudFront + custom domain) deployments
        # The callback_uri MUST match exactly between authorization and token exchange
        host = request.headers.get("host", "localhost:8888")
        # Check CloudFront header first, then X-Forwarded-Proto for HTTPS detection
        cloudfront_proto = request.headers.get("x-cloudfront-forwarded-proto", "").lower()
        forwarded_proto = request.headers.get("x-forwarded-proto", "").lower()
        scheme = (
            "https"
            if cloudfront_proto == "https"
            or forwarded_proto == "https"
            or request.url.scheme == "https"
            else "http"
        )
        logger.info(
            f"OAuth2 login - host: {host}, x-cloudfront-forwarded-proto: {cloudfront_proto}, x-forwarded-proto: {forwarded_proto}, scheme: {scheme}"
        )

        # Special case for localhost to include port
        if "localhost" in host and ":" not in host:
            auth_server_url = f"{scheme}://localhost:8888{ROOT_PATH}"
        else:
            auth_server_url = f"{scheme}://{host}{ROOT_PATH}"

        callback_uri = f"{auth_server_url}/oauth2/callback/{provider}"
        logger.info(f"OAuth2 callback URI (from request host): {callback_uri}")

        # Store state, redirect URI, and callback_uri in session for callback validation
        # The callback_uri is stored so token exchange uses the exact same URI
        session_data = {
            "state": state,
            "provider": provider,
            "redirect_uri": redirect_uri
            or OAUTH2_CONFIG.get("registry", {}).get("success_redirect", "/"),
            "callback_uri": callback_uri,  # Store for token exchange
        }

        # Create temporary session for OAuth2 flow
        temp_session = signer.dumps(session_data)

        auth_params = {
            "client_id": provider_config["client_id"],
            "response_type": provider_config["response_type"],
            "scope": " ".join(provider_config["scopes"]),
            "state": state,
            "redirect_uri": callback_uri,
        }

        auth_url = f"{provider_config['auth_url']}?{urllib.parse.urlencode(auth_params)}"

        # Create response with temporary session cookie
        response = RedirectResponse(url=auth_url, status_code=302)
        cookie_secure = scheme == "https"
        response.set_cookie(
            key="oauth2_temp_session",
            value=temp_session,
            max_age=600,  # 10 minutes for OAuth2 flow
            httponly=True,
            secure=cookie_secure,
            samesite="lax",
        )

        logger.info(f"Initiated OAuth2 login for provider {provider}")
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating OAuth2 login for {provider}: {e}")
        error_url = OAUTH2_CONFIG.get("registry", {}).get("error_redirect", "/login")
        return RedirectResponse(url=f"{error_url}?error=oauth2_init_failed", status_code=302)


@app.get("/oauth2/callback/{provider}")
async def oauth2_callback(
    provider: str,
    request: Request,
    code: str = None,
    state: str = None,
    error: str = None,
    oauth2_temp_session: str = Cookie(None),
):
    """Handle OAuth2 callback and create user session"""
    try:
        if error:
            logger.warning(f"OAuth2 error from {provider}: {error}")
            error_url = OAUTH2_CONFIG.get("registry", {}).get("error_redirect", "/login")
            return RedirectResponse(
                url=f"{error_url}?error=oauth2_error&details={error}", status_code=302
            )

        if not code or not state or not oauth2_temp_session:
            raise HTTPException(status_code=400, detail="Missing required OAuth2 parameters")

        # Validate temporary session
        try:
            temp_session_data = signer.loads(oauth2_temp_session, max_age=600)
        except (SignatureExpired, BadSignature):
            raise HTTPException(status_code=400, detail="Invalid or expired OAuth2 session")

        # Validate state parameter
        if state != temp_session_data.get("state"):
            raise HTTPException(status_code=400, detail="Invalid state parameter")

        # Validate provider
        if provider != temp_session_data.get("provider"):
            raise HTTPException(status_code=400, detail="Provider mismatch")

        provider_config = OAUTH2_CONFIG["providers"][provider]

        # Exchange authorization code for access token
        # Use the callback_uri stored in the session (must match what was used in authorization)
        callback_uri = temp_session_data.get("callback_uri")
        if callback_uri:
            # Extract auth_server_url from the stored callback_uri
            # callback_uri format: {auth_server_url}/oauth2/callback/{provider}
            auth_server_url = callback_uri.rsplit(f"/oauth2/callback/{provider}", 1)[0]
            logger.info(f"Using stored callback_uri for token exchange: {callback_uri}")
        else:
            # Fallback for sessions created before this fix
            auth_server_external_url = os.environ.get("AUTH_SERVER_EXTERNAL_URL")
            if auth_server_external_url:
                auth_server_url = auth_server_external_url.rstrip("/")
                logger.info(
                    f"Fallback: Using AUTH_SERVER_EXTERNAL_URL for token exchange: {auth_server_url}"
                )
            else:
                host = request.headers.get("host", "localhost:8888")
                scheme = (
                    "https"
                    if request.headers.get("x-forwarded-proto") == "https"
                    or request.url.scheme == "https"
                    else "http"
                )
                if "localhost" in host and ":" not in host:
                    auth_server_url = f"{scheme}://localhost:8888{ROOT_PATH}"
                else:
                    auth_server_url = f"{scheme}://{host}{ROOT_PATH}"
                logger.warning(f"Fallback: Using dynamic URL for token exchange: {auth_server_url}")

        token_data = await exchange_code_for_token(provider, code, provider_config, auth_server_url)
        logger.info(f"Token data keys: {list(token_data.keys())}")

        # For Cognito and Keycloak, try to extract user info from JWT tokens
        if provider in ["cognito", "keycloak"]:
            try:
                if provider == "cognito":
                    # Extract Cognito configuration from environment
                    user_pool_id = os.environ.get("COGNITO_USER_POOL_ID")
                    client_id = provider_config["client_id"]
                    region = os.environ.get("AWS_REGION", "us-east-1")

                    if user_pool_id and client_id:
                        # Use our existing token validation to get groups from JWT
                        validator = SimplifiedCognitoValidator(region)
                        token_validation = validator.validate_token(
                            token_data["access_token"], user_pool_id, client_id, region
                        )

                        logger.info(f"Token validation result: {token_validation}")

                        # Extract user info from token validation
                        mapped_user = {
                            "username": token_validation.get("username"),
                            "email": token_validation.get(
                                "username"
                            ),  # Cognito username is usually email
                            "name": token_validation.get("username"),
                            "groups": token_validation.get("groups", []),
                        }
                        logger.info(f"User extracted from JWT token: {mapped_user}")
                    else:
                        logger.warning(
                            "Missing Cognito configuration for JWT validation, falling back to userInfo"
                        )
                        raise ValueError("Missing Cognito config")
                elif provider == "keycloak":
                    # For Keycloak, decode the ID token to get user information
                    if "id_token" in token_data:
                        import jwt

                        # Decode without verification for now (we trust the token since we just got it)
                        id_token_claims = jwt.decode(
                            token_data["id_token"], options={"verify_signature": False}
                        )
                        logger.info(f"ID token claims: {id_token_claims}")

                        # Extract user info from ID token claims
                        mapped_user = {
                            "username": id_token_claims.get("preferred_username")
                            or id_token_claims.get("sub"),
                            "email": id_token_claims.get("email"),
                            "name": id_token_claims.get("name")
                            or id_token_claims.get("given_name"),
                            "groups": id_token_claims.get("groups", []),
                        }
                        logger.info(f"User extracted from Keycloak ID token: {mapped_user}")
                    else:
                        logger.warning(
                            "No ID token found in Keycloak response, falling back to userInfo"
                        )
                        raise ValueError("Missing ID token")

            except Exception as e:
                logger.warning(
                    f"JWT token validation failed: {e}, falling back to userInfo endpoint"
                )
                # Fallback to userInfo endpoint
                user_info = await get_user_info(token_data["access_token"], provider_config)
                logger.info(f"Raw user info from {provider}: {user_info}")
                mapped_user = map_user_info(user_info, provider_config)
                logger.info(f"Mapped user info from userInfo: {mapped_user}")
        elif provider == "entra":
            # For Entra ID, prioritize ID token claims over userinfo endpoint
            try:
                if "id_token" in token_data:
                    import jwt

                    # Decode without verification (we trust the token since we just got it from Microsoft)
                    id_token_claims = jwt.decode(
                        token_data["id_token"], options={"verify_signature": False}
                    )
                    logger.info(f"Entra ID token claims: {id_token_claims}")

                    # Extract user info from ID token claims
                    # Entra ID can return groups as either 'groups' or 'roles' depending on configuration
                    groups = id_token_claims.get("groups", [])
                    if not groups:
                        groups = id_token_claims.get("roles", [])

                    mapped_user = {
                        "username": id_token_claims.get("preferred_username")
                        or id_token_claims.get("email")
                        or id_token_claims.get("upn")
                        or id_token_claims.get("sub"),
                        "email": id_token_claims.get("email")
                        or id_token_claims.get("preferred_username"),
                        "name": id_token_claims.get("name") or id_token_claims.get("given_name"),
                        "groups": groups,
                    }
                    logger.info(f"User extracted from Entra ID token: {mapped_user}")
                else:
                    logger.warning("No ID token found in Entra response, falling back to userInfo")
                    raise ValueError("Missing ID token")

            except Exception as e:
                logger.warning(
                    f"Entra ID token parsing failed: {e}, falling back to userInfo endpoint"
                )
                # Fallback to userInfo endpoint
                user_info = await get_user_info(token_data["access_token"], provider_config)
                logger.info(f"Raw user info from {provider}: {user_info}")
                mapped_user = map_user_info(user_info, provider_config)
                logger.info(f"Mapped user info from userInfo: {mapped_user}")
        else:
            # For other providers, use userInfo endpoint
            user_info = await get_user_info(token_data["access_token"], provider_config)
            logger.info(f"Raw user info from {provider}: {user_info}")
            mapped_user = map_user_info(user_info, provider_config)
            logger.info(f"Mapped user info: {mapped_user}")

        # Create session cookie compatible with registry
        # OAuth tokens are conditionally stored based on OAUTH_STORE_TOKENS_IN_SESSION
        # Disable for large tokens (e.g., Entra ID) to avoid cookie size limits (4096 bytes)
        session_data = {
            "username": mapped_user["username"],
            "email": mapped_user.get("email"),
            "name": mapped_user.get("name"),
            "groups": mapped_user.get("groups", []),
            "provider": provider,
            "auth_method": "oauth2",
        }

        if OAUTH_STORE_TOKENS_IN_SESSION:
            session_data.update(
                {
                    "access_token": token_data.get("access_token"),
                    "refresh_token": token_data.get("refresh_token"),
                    "token_expires_in": token_data.get("expires_in"),
                    "token_obtained_at": int(time.time()),
                }
            )

        registry_session = signer.dumps(session_data)

        # Redirect to registry with session cookie
        redirect_url = temp_session_data.get(
            "redirect_uri", OAUTH2_CONFIG.get("registry", {}).get("success_redirect", "/")
        )
        response = RedirectResponse(url=redirect_url, status_code=302)

        # Set registry-compatible session cookie
        # Check if HTTPS is terminated at load balancer/CloudFront
        is_https = is_request_https(request)

        # Only set secure=True if the original request was HTTPS
        cookie_secure_config = OAUTH2_CONFIG.get("session", {}).get("secure", False)
        cookie_secure = cookie_secure_config and is_https
        cookie_samesite = OAUTH2_CONFIG.get("session", {}).get("samesite", "lax")
        cookie_domain = OAUTH2_CONFIG.get("session", {}).get("domain", "")

        # Handle domain configuration - only use explicitly configured values
        # Empty string or placeholder means no domain attribute (exact host only)
        if not cookie_domain or cookie_domain == "${SESSION_COOKIE_DOMAIN}":
            cookie_domain = None
            logger.info("No cookie domain configured - cookie will be set for exact host only")
        else:
            logger.info(f"Using explicitly configured cookie domain: {cookie_domain}")

        logger.info(
            f"Auth server setting session cookie: secure={cookie_secure} (config={cookie_secure_config}, is_https={is_https}), samesite={cookie_samesite}, domain={cookie_domain or 'not set'}, x-forwarded-proto={request.headers.get('x-forwarded-proto', 'not set')}, request_scheme={request.url.scheme}"
        )

        cookie_params = {
            "key": "mcp_gateway_session",  # Same as registry SESSION_COOKIE_NAME
            "value": registry_session,
            "max_age": OAUTH2_CONFIG.get("session", {}).get("max_age_seconds", 28800),
            "httponly": OAUTH2_CONFIG.get("session", {}).get("httponly", True),
            "samesite": cookie_samesite,
            "secure": cookie_secure,
            "path": "/",  # Ensure cookie is sent for all paths
        }

        # Only set domain if configured or inferred (for cross-subdomain cookies)
        if cookie_domain:
            cookie_params["domain"] = cookie_domain

        response.set_cookie(**cookie_params)

        # Clear temporary OAuth2 session
        response.delete_cookie("oauth2_temp_session")

        logger.info(
            f"Successfully authenticated user {hash_username(mapped_user['username'])} via {provider}"
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in OAuth2 callback for {provider}: {e}")
        error_url = OAUTH2_CONFIG.get("registry", {}).get("error_redirect", "/login")
        return RedirectResponse(url=f"{error_url}?error=oauth2_callback_failed", status_code=302)


async def exchange_code_for_token(
    provider: str, code: str, provider_config: dict, auth_server_url: str = None
) -> dict:
    """Exchange authorization code for access token"""
    if auth_server_url is None:
        auth_server_url = (
            os.environ.get("AUTH_SERVER_URL", "http://localhost:8888").rstrip("/") + ROOT_PATH
        )

    async with httpx.AsyncClient() as client:
        token_data = {
            "grant_type": provider_config["grant_type"],
            "client_id": provider_config["client_id"],
            "client_secret": provider_config["client_secret"],
            "code": code,
            "redirect_uri": f"{auth_server_url}/oauth2/callback/{provider}",
        }

        headers = {"Accept": "application/json"}
        if provider == "github":
            headers["Accept"] = "application/json"

        response = await client.post(provider_config["token_url"], data=token_data, headers=headers)
        response.raise_for_status()
        return response.json()


async def get_user_info(access_token: str, provider_config: dict) -> dict:
    """Get user information from OAuth2 provider"""
    async with httpx.AsyncClient() as client:
        headers = {"Authorization": f"Bearer {access_token}"}

        response = await client.get(provider_config["user_info_url"], headers=headers)
        response.raise_for_status()
        return response.json()


def map_user_info(user_info: dict, provider_config: dict) -> dict:
    """Map provider-specific user info to our standard format"""
    mapped = {
        "username": user_info.get(provider_config["username_claim"]),
        "email": user_info.get(provider_config["email_claim"]),
        "name": user_info.get(provider_config["name_claim"]),
        "groups": [],
    }

    # Handle groups if provider supports them
    groups_claim = provider_config.get("groups_claim")
    logger.info(f"Looking for groups using claim: {groups_claim}")
    logger.info(f"Available claims in user_info: {list(user_info.keys())}")

    if groups_claim and groups_claim in user_info:
        groups = user_info[groups_claim]
        if isinstance(groups, list):
            mapped["groups"] = groups
        elif isinstance(groups, str):
            mapped["groups"] = [groups]
        logger.info(f"Found groups via {groups_claim}: {mapped['groups']}")
    else:
        # Try alternative group claims for Cognito
        for possible_group_claim in ["cognito:groups", "groups", "custom:groups"]:
            if possible_group_claim in user_info:
                groups = user_info[possible_group_claim]
                if isinstance(groups, list):
                    mapped["groups"] = groups
                elif isinstance(groups, str):
                    mapped["groups"] = [groups]
                logger.info(
                    f"Found groups via alternative claim {possible_group_claim}: {mapped['groups']}"
                )
                break

        if not mapped["groups"]:
            logger.warning(
                f"No groups found in user_info. Available fields: {list(user_info.keys())}"
            )

    return mapped


@app.get("/oauth2/logout/{provider}")
async def oauth2_logout(provider: str, request: Request, redirect_uri: str = None):
    """Initiate OAuth2 logout flow to clear provider session"""
    try:
        if provider not in OAUTH2_CONFIG.get("providers", {}):
            raise HTTPException(status_code=404, detail=f"Provider {provider} not found")

        provider_config = OAUTH2_CONFIG["providers"][provider]
        logout_url = provider_config.get("logout_url")

        if not logout_url:
            # If provider doesn't support logout URL, just redirect
            redirect_url = redirect_uri or OAUTH2_CONFIG.get("registry", {}).get(
                "success_redirect", "/login"
            )
            return RedirectResponse(url=redirect_url, status_code=302)

        # Build full redirect URI
        full_redirect_uri = redirect_uri or "/login"
        if not full_redirect_uri.startswith("http"):
            # Make it a full URL - extract registry URL from request's referer or use environment
            registry_base = os.environ.get("REGISTRY_URL")
            if not registry_base:
                # Try to derive from the request
                referer = request.headers.get("referer", "")
                if referer:
                    from urllib.parse import urlparse

                    parsed = urlparse(referer)
                    registry_base = f"{parsed.scheme}://{parsed.netloc}"
                else:
                    registry_base = "http://localhost"

            full_redirect_uri = f"{registry_base.rstrip('/')}{full_redirect_uri}"

        # Detect provider type and build appropriate logout URL
        # Keycloak uses post_logout_redirect_uri, Cognito uses logout_uri
        if "keycloak" in provider.lower() or "/realms/" in logout_url:
            # Keycloak logout parameters
            logout_params = {
                "client_id": provider_config["client_id"],
                "post_logout_redirect_uri": full_redirect_uri,
            }
        else:
            # Cognito logout parameters
            logout_params = {
                "client_id": provider_config["client_id"],
                "logout_uri": full_redirect_uri,
            }

        logout_redirect_url = f"{logout_url}?{urllib.parse.urlencode(logout_params)}"

        logger.info(f"Redirecting to {provider} logout: {logout_redirect_url}")
        return RedirectResponse(url=logout_redirect_url, status_code=302)

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error initiating logout for {provider}: {e}")
        # Fallback to local redirect
        redirect_url = redirect_uri or OAUTH2_CONFIG.get("registry", {}).get(
            "success_redirect", "/login"
        )
        return RedirectResponse(url=redirect_url, status_code=302)
