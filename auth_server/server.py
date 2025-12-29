"""
Simplified Authentication server that validates JWT tokens against Amazon Cognito.
Configuration is passed via headers instead of environment variables.
"""

import argparse
import logging
import os
import boto3
import jwt
import requests
import json
import yaml
import time
import uuid
import hashlib
from jwt.api_jwk import PyJWK
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass
import importlib
from typing import Dict, Optional, List, Any
from functools import lru_cache
from botocore.exceptions import ClientError
from fastapi import FastAPI, Header, HTTPException, Request, Cookie
from fastapi.responses import JSONResponse, Response, RedirectResponse
import uvicorn
from pydantic import BaseModel
from pathlib import Path
from itsdangerous import URLSafeTimedSerializer, SignatureExpired, BadSignature
import secrets
import urllib.parse
import httpx
from string import Template
import os as _os
import json as _json

from gateway_session import (
    build_session_cookie_payload,
)
from gateway_csrf import (
    validate_csrf_token,
)
from gateway_session import (
    normalize_session_data,
)
from auth_server.enforceai.db.data_layer import (
    EnforceAIDataLayer,
)
from auth_server.enforceai.stores.sqlite.session_store import (
    SqliteSessionStore,
)
from auth_server.enforceai.stores.sqlite.user_store import (
    SqliteUserStore,
)

# Import metrics middleware (support repo + Docker module layouts)
try:
    from .metrics_middleware import add_auth_metrics_middleware
except ImportError:  # pragma: no cover
    from metrics_middleware import add_auth_metrics_middleware

# Import core route modules (support repo + Docker module layouts)
try:
    from .routes.core_routes import router as core_router
except ImportError:  # pragma: no cover
    from routes.core_routes import router as core_router

# Import OAuth2 config + routes (support repo + Docker module layouts)
try:
    from .routes.oauth2_context import (
        CSRF_TOKEN_MAX_AGE_SECONDS,
        SECRET_KEY,
        SESSION_COOKIE_NAME,
        signer,
    )
    from .routes.oauth2_routes import (
        router as oauth2_router,
    )
except ImportError:  # pragma: no cover
    from routes.oauth2_context import (
        CSRF_TOKEN_MAX_AGE_SECONDS,
        SECRET_KEY,
        SESSION_COOKIE_NAME,
        signer,
    )
    from routes.oauth2_routes import (
        router as oauth2_router,
    )

# Import provider factory (support repo + Docker module layouts)
try:
    from .providers.factory import get_auth_provider
except ImportError:  # pragma: no cover
    from providers.factory import get_auth_provider

try:
    from .providers.cognito_validator import SimplifiedCognitoValidator
except ImportError:  # pragma: no cover
    from providers.cognito_validator import SimplifiedCognitoValidator

try:
    from .constants import (
        JWT_AUDIENCE,
        JWT_ISSUER,
    )
except ImportError:  # pragma: no cover
    from constants import (
        JWT_AUDIENCE,
        JWT_ISSUER,
    )

try:
    from .routes.internal_routes import (
        router as internal_router,
    )
except ImportError:  # pragma: no cover
    from routes.internal_routes import (
        router as internal_router,
    )

try:
    import auth_server.scopes_config as scopes_config
except Exception:  # noqa: BLE001
    import scopes_config  # type: ignore[no-redef]

@dataclass(frozen=True)
class _EnforceAIRuntime:
    DependencyUnavailableError: type[Exception]
    EnforceAIError: type[Exception]
    evaluate_tool_call: object
    resolve_callable_tools_for_server: object
    load_scope_catalog: object
    get_enforceai_settings: object
    get_enforceai_stores: object
    get_identity_resolver: object
    get_upstream_oauth_token_client: object


@lru_cache(maxsize=1)
def _load_enforceai_runtime() -> _EnforceAIRuntime:
    for base in ("auth_server.enforceai", "enforceai"):
        try:
            importlib.import_module(base)
        except ModuleNotFoundError:
            continue

        errors_module = importlib.import_module(f"{base}.errors")
        evaluate_module = importlib.import_module(f"{base}.fgac.evaluate")
        catalog_module = importlib.import_module(f"{base}.fgac.catalog")
        dependency_module = importlib.import_module(f"{base}.auth.dependency")

        return _EnforceAIRuntime(
            DependencyUnavailableError=errors_module.DependencyUnavailableError,
            EnforceAIError=errors_module.EnforceAIError,
            evaluate_tool_call=evaluate_module.evaluate_tool_call,
            resolve_callable_tools_for_server=evaluate_module.resolve_callable_tools_for_server,
            load_scope_catalog=catalog_module.load_scope_catalog,
            get_enforceai_settings=dependency_module.get_enforceai_settings,
            get_enforceai_stores=dependency_module.get_enforceai_stores,
            get_identity_resolver=dependency_module.get_identity_resolver,
            get_upstream_oauth_token_client=dependency_module.get_upstream_oauth_token_client,
        )

    raise RuntimeError("EnforceAI runtime could not be imported")


def get_identity_resolver():
    return _load_enforceai_runtime().get_identity_resolver()


def get_enforceai_settings():
    return _load_enforceai_runtime().get_enforceai_settings()


def get_enforceai_stores():
    return _load_enforceai_runtime().get_enforceai_stores()


def get_upstream_oauth_token_client():
    return _load_enforceai_runtime().get_upstream_oauth_token_client()


def load_scope_catalog(
    *,
    path: Optional[Path] = None,
):
    return _load_enforceai_runtime().load_scope_catalog(path=path)


def evaluate_tool_call(
    *,
    identity,
    catalog,
    server: str,
    tool: str,
    allowed_tools,
):
    return _load_enforceai_runtime().evaluate_tool_call(
        identity=identity,
        catalog=catalog,
        server=server,
        tool=tool,
        allowed_tools=allowed_tools,
    )


def resolve_callable_tools_for_server(
    *,
    identity,
    catalog,
    server: str,
    allowed_tools,
):
    return _load_enforceai_runtime().resolve_callable_tools_for_server(
        identity=identity,
        catalog=catalog,
        server=server,
        allowed_tools=allowed_tools,
    )


def _load_enforceai_management_router():
    for base in ("auth_server.enforceai", "enforceai"):
        try:
            importlib.import_module(base)
        except ModuleNotFoundError:
            continue

        try:
            module = importlib.import_module(f"{base}.api.management_routes")
        except ModuleNotFoundError:
            continue

        router = getattr(module, "router", None)
        if router is not None:
            return router

    return None

# Configure logging
logging.basicConfig(
    level=logging.INFO,  # Set the log level to INFO
    # Define log message format
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)
logger = logging.getLogger(__name__)

# EnforceAI required guard (fail-fast)
def _enforce_enforceai_required() -> None:
    enforceai_required_raw = _os.environ.get("ENFORCEAI_REQUIRED", "false")
    enforceai_required = enforceai_required_raw.strip().lower() in {"1", "true", "yes"}
    if not enforceai_required:
        return

    db_path = (_os.environ.get("ENFORCEAI_DB_PATH") or "").strip()
    if not db_path:
        raise RuntimeError(
            "ENFORCEAI_REQUIRED=true but ENFORCEAI_DB_PATH is not set; refusing to start"
        )

    scopes_catalog_path = (
        (_os.environ.get("ENFORCEAI_SCOPES_CATALOG_PATH") or "").strip()
        or (_os.environ.get("SCOPES_CATALOG_PATH") or "").strip()
    )
    if not scopes_catalog_path:
        raise RuntimeError(
            "ENFORCEAI_REQUIRED=true but ENFORCEAI_SCOPES_CATALOG_PATH is not set; refusing to start"
        )

    try:
        EnforceAIDataLayer(db_path=Path(db_path)).initialize()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            "ENFORCEAI_REQUIRED=true but EnforceAI DB initialization failed; refusing to start"
        ) from exc


_enforce_enforceai_required()

# Utility functions for GDPR/SOX compliance
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
    if not ip_address or ip_address == 'unknown':
        return ip_address
    if '.' in ip_address:  # IPv4
        parts = ip_address.split('.')
        if len(parts) == 4:
            return f"{'.'.join(parts[:3])}.xxx"
    elif ':' in ip_address:  # IPv6
        # Mask last segment
        parts = ip_address.split(':')
        if len(parts) > 1:
            parts[-1] = 'xxxx'
            return ':'.join(parts)
    return ip_address

def mask_token(token: str) -> str:
    """Mask JWT token showing only last 4 characters."""
    if not token:
        return "***EMPTY***"
    if len(token) > 20:
        return f"...{token[-4:]}"
    return "***MASKED***"

def mask_headers(headers: dict) -> dict:
    """Mask sensitive headers for logging compliance."""
    masked = {}
    for key, value in headers.items():
        key_lower = key.lower()
        if key_lower in ['x-authorization', 'authorization', 'cookie']:
            if 'bearer' in str(value).lower():
                # Extract token part and mask it
                parts = str(value).split(' ', 1)
                if len(parts) == 2:
                    masked[key] = f"Bearer {mask_token(parts[1])}"
                else:
                    masked[key] = mask_token(value)
            else:
                masked[key] = "***MASKED***"
        elif key_lower in ['x-user-pool-id', 'x-client-id']:
            masked[key] = mask_sensitive_id(value)
        else:
            masked[key] = value
    return masked


def _resolve_enforceai_scopes_catalog_path() -> Optional[Path]:
    raw = _os.environ.get("ENFORCEAI_SCOPES_CATALOG_PATH") or _os.environ.get(
        "SCOPES_CATALOG_PATH"
    )
    if raw is None:
        return None

    stripped = raw.strip()
    if not stripped:
        return None

    return Path(stripped)


def _emit_enforceai_audit_event(
    *,
    action: str,
    outcome: str,
    user_id: str,
    agent_id: str,
    request_id: Optional[str],
    details: dict[str, Any],
) -> None:
    try:
        print(
            _json.dumps(
                {
                    "event_type": "enforceai_audit",
                    "action": action,
                    "outcome": outcome,
                    "user_id": user_id,
                    "agent_id": agent_id,
                    "request_id": request_id,
                    "details": details,
                },
                separators=(",", ":"),
                sort_keys=True,
                default=str,
            ),
            flush=True,
        )
    except Exception:  # noqa: BLE001 - best-effort
        logger.exception("Failed to emit EnforceAI audit event to stdout")

    try:
        stores = get_enforceai_stores()
        stores.audit_store.append_event(
            occurred_at=datetime.now(timezone.utc).replace(microsecond=0),
            user_id=user_id,
            agent_id=agent_id,
            action=action,
            outcome=outcome,
            request_id=request_id,
            details=details,
        )
    except Exception:  # noqa: BLE001 - best-effort
        logger.exception("Failed to persist EnforceAI audit event")

def map_groups_to_scopes(groups: List[str]) -> List[str]:
    """
    Map identity provider groups to MCP scopes using the group_mappings from scopes.yml configuration.
    
    Args:
        groups: List of group names from identity provider (Cognito, Keycloak, etc.)
        
    Returns:
        List of MCP scopes
    """
    scopes = []
    current_scopes_config = scopes_config.get_scopes_config()
    group_mappings = current_scopes_config.get('group_mappings', {})
    
    for group in groups:
        if group in group_mappings:
            group_scopes = group_mappings[group]
            scopes.extend(group_scopes)
            logger.debug(f"Mapped group '{group}' to scopes: {group_scopes}")
        else:
            logger.debug(f"No scope mapping found for group: {group}")
    
    # Remove duplicates while preserving order
    seen = set()
    unique_scopes = []
    for scope in scopes:
        if scope not in seen:
            seen.add(scope)
            unique_scopes.append(scope)
    
    logger.info(f"Final mapped scopes: {unique_scopes}")
    return unique_scopes

def validate_session_cookie(cookie_value: str) -> Dict[str, any]:
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

        normalized = normalize_session_data(
            data,
            default_provider="local",
            max_age_seconds=28800,
        )

        db_path_raw = _os.environ.get("ENFORCEAI_DB_PATH")
        if db_path_raw:
            try:
                db_path = Path(db_path_raw.strip())
                EnforceAIDataLayer(db_path=db_path).initialize()
                store = SqliteSessionStore(db_path=db_path)
                record = store.get_session_by_id(session_id=normalized.session_id)

                if record is None:
                    logger.info(
                        "No server-side session record found; accepting stateless session cookie"
                    )
                elif record.revoked_at is not None:
                    raise ValueError("Session invalidated")
                else:
                    store.touch_session(
                        session_id=normalized.session_id,
                        now=datetime.now(timezone.utc).replace(microsecond=0),
                    )
            except ValueError:
                raise
            except Exception:
                logger.warning(
                    "Skipping server-side session validation; EnforceAI DB unavailable",
                    exc_info=True,
                )

        # Extract user info
        username = normalized.user_id
        groups = normalized.groups or []
        
        # Map groups to scopes
        scopes = map_groups_to_scopes(groups)
        
        logger.info(f"Session cookie validated for user: {hash_username(username)}")
        
        return {
            'valid': True,
            'username': username,
            'scopes': scopes,
            'method': 'session_cookie',
            'groups': groups,
            'client_id': '',  # Not applicable for session
            'data': data  # Include full data for consistency
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

def parse_server_and_tool_from_url(original_url: str) -> tuple[Optional[str], Optional[str]]:
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
        path = parsed_url.path.strip('/')
        
        # The path should be in format: /server_name/...
        # Extract the first path component as server name
        path_parts = path.split('/') if path else []
        server_name = path_parts[0] if path_parts else None
        
        logger.debug(f"Parsed server name '{server_name}' from URL path: {path}")
        return server_name, None  # Tool name would need to be extracted from request payload
        
    except Exception as e:
        logger.error(f"Failed to parse server/tool from URL {original_url}: {e}")
        return None, None


def _normalize_server_name(name: str) -> str:
    """
    Normalize server name by removing trailing slash for comparison.

    This handles cases where a server is registered with a trailing slash
    but accessed without one (or vice versa).

    Args:
        name: Server name to normalize

    Returns:
        Normalized server name (without trailing slash)
    """
    return name.rstrip('/') if name else name


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
    if normalized_name1 == '*':
        return True
    return normalized_name1 == _normalize_server_name(name2)


def validate_server_tool_access(server_name: str, method: str, tool_name: str, user_scopes: List[str]) -> bool:
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
        logger.info(f"=== VALIDATE_SERVER_TOOL_ACCESS START ===")
        logger.info(f"Requested server: '{server_name}'")
        logger.info(f"Requested method: '{method}'")
        logger.info(f"Requested tool: '{tool_name}'")
        logger.info(f"User scopes: {user_scopes}")
        current_scopes_config = scopes_config.get_scopes_config()
        logger.info(
            "Available scopes config keys: %s",
            list(current_scopes_config.keys()) if current_scopes_config else "None",
        )

        if not current_scopes_config:
            logger.warning("No scopes configuration loaded, allowing access")
            logger.info(f"=== VALIDATE_SERVER_TOOL_ACCESS END: ALLOWED (no config) ===")
            return True
            
        # Check each user scope to see if it grants access
        for scope in user_scopes:
            logger.info(f"--- Checking scope: '{scope}' ---")
            scope_config = current_scopes_config.get(scope, [])
            
            if not scope_config:
                logger.info(f"Scope '{scope}' not found in configuration")
                continue
                
            logger.info(f"Scope '{scope}' config: {scope_config}")
            
            # The scope_config is directly a list of server configurations
            # since the permission type is already encoded in the scope name
            for server_config in scope_config:
                logger.info(f"  Examining server config: {server_config}")
                server_config_name = server_config.get('server')
                logger.info(f"  Server name in config: '{server_config_name}' vs requested: '{server_name}'")

                if _server_names_match(server_config_name, server_name):
                    logger.info(f"  ✓ Server name matches!")
                    
                    # Check methods first
                    allowed_methods = server_config.get('methods', [])
                    logger.info(f"  Allowed methods for server '{server_name}': {allowed_methods}")
                    logger.info(f"  Checking if method '{method}' is in allowed methods...")

                    # Check if all methods are allowed (wildcard support)
                    has_wildcard_methods = 'all' in allowed_methods or '*' in allowed_methods

                    # for all methods except tools/call we are good if the method is allowed
                    # for tools/call we need to do an extra validation to check if the tool
                    # itself is allowed or not
                    if (method in allowed_methods or has_wildcard_methods) and method != 'tools/call':
                        logger.info(f"  ✓ Method '{method}' found in allowed methods!")
                        logger.info(f"Access granted: scope '{scope}' allows access to {server_name}.{method}")
                        logger.info(f"=== VALIDATE_SERVER_TOOL_ACCESS END: GRANTED ===")
                        return True
                    
                    # Check tools if method not found in methods
                    allowed_tools = server_config.get('tools', [])
                    logger.info(f"  Allowed tools for server '{server_name}': {allowed_tools}")

                    # Check if all tools are allowed (wildcard support)
                    has_wildcard_tools = 'all' in allowed_tools or '*' in allowed_tools

                    # For tools/call, check if the specific tool is allowed
                    if method == 'tools/call' and tool_name:
                        logger.info(f"  Checking if tool '{tool_name}' is in allowed tools for tools/call...")
                        if tool_name in allowed_tools or has_wildcard_tools:
                            logger.info(f"  ✓ Tool '{tool_name}' found in allowed tools!")
                            logger.info(f"Access granted: scope '{scope}' allows access to {server_name}.{method} for tool {tool_name}")
                            logger.info(f"=== VALIDATE_SERVER_TOOL_ACCESS END: GRANTED ===")
                            return True
                        else:
                            logger.info(f"  ✗ Tool '{tool_name}' NOT found in allowed tools")
                    else:
                        # For other methods, check if method is in tools list (backward compatibility)
                        logger.info(f"  Checking if method '{method}' is in allowed tools...")
                        if method in allowed_tools or has_wildcard_tools:
                            logger.info(f"  ✓ Method '{method}' found in allowed tools!")
                            logger.info(f"Access granted: scope '{scope}' allows access to {server_name}.{method}")
                            logger.info(f"=== VALIDATE_SERVER_TOOL_ACCESS END: GRANTED ===")
                            return True
                        else:
                            logger.info(f"  ✗ Method '{method}' NOT found in allowed tools")
                else:
                    logger.info(f"  ✗ Server name does not match")
        
        logger.warning(f"Access denied: no scope allows access to {server_name}.{method} (tool: {tool_name}) for user scopes: {user_scopes}")
        logger.info(f"=== VALIDATE_SERVER_TOOL_ACCESS END: DENIED ===")
        return False
        
    except Exception as e:
        logger.error(f"Error validating server/tool access: {e}")
        logger.info(f"=== VALIDATE_SERVER_TOOL_ACCESS END: ERROR ===")
        return False  # Deny access on error

# Create FastAPI app
app = FastAPI(
    title="Simplified Auth Server",
    description="Authentication server for validating JWT tokens against Amazon Cognito with header-based configuration",
    version="0.1.0"
)

# Add metrics collection middleware
add_auth_metrics_middleware(app)
app.include_router(core_router)
app.include_router(oauth2_router)
app.include_router(internal_router)
app.state.session_secret_key = SECRET_KEY
app.state.session_signer = signer

try:
    if _os.environ.get("ENFORCEAI_DB_PATH"):
        enforceai_management_router = _load_enforceai_management_router()
        if enforceai_management_router is not None:
            app.include_router(enforceai_management_router)
    else:
        logger.info("ENFORCEAI_DB_PATH not set; skipping EnforceAI management routes")
except Exception:  # noqa: BLE001 - best-effort; server should still start
    logger.exception("Failed to mount EnforceAI management routes")

try:
    from auth_server.enforceai.errors import (  # type: ignore[import-not-found]
        DependencyUnavailableError,
        EnforceAIError,
    )
except Exception:  # noqa: BLE001
    DependencyUnavailableError = None  # type: ignore[assignment]
    EnforceAIError = None  # type: ignore[assignment]


if EnforceAIError is not None:

    @app.exception_handler(EnforceAIError)  # type: ignore[arg-type]
    async def _handle_enforceai_error(
        request: Request,
        exc: "EnforceAIError",
    ) -> JSONResponse:
        del request
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.public_message},
        )


@app.on_event("startup")
def _seed_enforceai_password_admin_user() -> None:
    db_path_raw = _os.environ.get("ENFORCEAI_DB_PATH")
    if not db_path_raw:
        return

    admin_username = _os.environ.get("ADMIN_USER", "admin").strip() or "admin"
    admin_password = _os.environ.get("ADMIN_PASSWORD")
    if not admin_password or not admin_password.strip():
        logger.warning("ADMIN_PASSWORD missing; skipping EnforceAI admin user seeding")
        return

    admin_email = _os.environ.get("ADMIN_EMAIL")
    if not admin_email or not admin_email.strip():
        admin_email = f"{admin_username}@local"

    try:
        db_path = Path(db_path_raw.strip())
        EnforceAIDataLayer(db_path=db_path).initialize()
        user_store = SqliteUserStore(db_path=db_path)

        existing = user_store.get_user_by_id(user_id=f"local|{admin_username}")
        if existing is not None:
            return

        from auth_server.enforceai.users.passwords import (
            hash_password,
        )

        password_hash = hash_password(admin_password).encoded
        user_store.create_local_user(
            username=admin_username,
            email=admin_email,
            password_hash=password_hash,
            role="admin",
        )
        logger.info(
            "Seeded EnforceAI admin user record for password login",
        )
    except Exception:
        logger.exception("Failed to seed EnforceAI password admin user")

# Create global validator instance
validator = SimplifiedCognitoValidator(
    secret_key=SECRET_KEY,
    jwt_issuer=JWT_ISSUER,
    jwt_audience=JWT_AUDIENCE,
)

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
    
    
    try:
        enforceai_enabled = bool(_os.environ.get("ENFORCEAI_DB_PATH"))
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
        original_path = ""
        if original_url:
            try:
                from urllib.parse import urlparse

                original_path = urlparse(original_url).path or ""
            except Exception:
                original_path = ""
        body = request.headers.get("X-Body")
        
        is_registry_api_request = original_path.startswith("/api/")

        # Extract server_name from original_url early for logging
        server_name_from_url = None
        if original_url:
            try:
                from urllib.parse import urlparse
                parsed_url = urlparse(original_url)
                path = parsed_url.path.strip('/')
                path_parts = path.split('/') if path else []
                server_name_from_url = path_parts[0] if path_parts else None
                logger.info(f"Extracted server_name '{server_name_from_url}' from original_url: {original_url}")
            except Exception as e:
                logger.warning(f"Failed to extract server_name from original_url {original_url}: {e}")

        if is_registry_api_request:
            server_name_from_url = None
        
        # Read request body
        request_payload = None
        try:
            if body:
                payload_text = body #.decode('utf-8')
                logger.info(f"Raw Request Payload ({len(payload_text)} chars): {payload_text[:1000]}...")
                request_payload = json.loads(payload_text)
                logger.info(f"JSON RPC Request Payload: {json.dumps(request_payload, indent=2)}")
            else:
                logger.info(f"No request body provided, skipping payload parsing")
        except UnicodeDecodeError as e:
            logger.warning(f"Could not decode body as UTF-8: {e}")
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse JSON RPC payload: {e}")
        except Exception as e:
            logger.error(f"Error reading request payload: {type(e).__name__}: {e}")

        server_name = server_name_from_url
        tool_name = None
        if request_payload and isinstance(request_payload, dict):
            tool_name = request_payload.get("method") or request_payload.get("tool") or request_payload.get("name")
            if not tool_name and "params" in request_payload and isinstance(
                request_payload.get("params"),
                dict,
            ):
                tool_name = (
                    request_payload["params"].get("method")
                    or request_payload["params"].get("tool")
                    or request_payload["params"].get("name")
                )

        if enforceai_enabled:
            runtime = _load_enforceai_runtime()
            dependency_unavailable_error = runtime.DependencyUnavailableError
            enforceai_error = runtime.EnforceAIError

            has_non_cookie_credentials = any(
                value and value.strip()
                for value in (
                    request.headers.get("authorization"),
                    request.headers.get("x-authorization"),
                    request.headers.get("x-gateway-token"),
                    request.headers.get("x-api-key"),
                )
            )

            allow_cookie_auth = original_path.startswith("/api/")
            cookie_header = request.headers.get("Cookie", "")

            if allow_cookie_auth and not has_non_cookie_credentials:
                if "mcp_gateway_session=" not in cookie_header:
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication required",
                        headers={"Connection": "close"},
                    )

                cookie_value = None
                for cookie in cookie_header.split(";"):
                    if cookie.strip().startswith("mcp_gateway_session="):
                        cookie_value = cookie.strip().split("=", 1)[1]
                        break

                if not cookie_value:
                    raise HTTPException(
                        status_code=401,
                        detail="Authentication required",
                        headers={"Connection": "close"},
                    )

                try:
                    validation_result = validate_session_cookie(cookie_value)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=401,
                        detail=str(exc),
                        headers={"Connection": "close"},
                    ) from exc

                response_data = {
                    "valid": True,
                    "username": validation_result.get("username") or "",
                    "client_id": validation_result.get("client_id") or "",
                    "scopes": validation_result.get("scopes", []),
                    "method": validation_result.get("method") or "",
                    "groups": validation_result.get("groups", []),
                    "server_name": server_name_from_url,
                    "tool_name": tool_name,
                }
                response = JSONResponse(
                    content=response_data,
                    status_code=200,
                )
                response.headers["X-User"] = validation_result.get("username") or ""
                response.headers["X-Username"] = validation_result.get("username") or ""
                response.headers["X-Client-Id"] = validation_result.get("client_id") or ""
                response.headers["X-Scopes"] = " ".join(validation_result.get("scopes", []))
                response.headers["X-Auth-Method"] = validation_result.get("method") or ""
                response.headers["X-Server-Name"] = server_name_from_url or ""
                response.headers["X-Tool-Name"] = tool_name or ""
                return response

            try:
                resolver = get_identity_resolver()
                catalog_path = _resolve_enforceai_scopes_catalog_path()
                if catalog_path is None:
                    catalog = load_scope_catalog()
                else:
                    catalog = load_scope_catalog(path=catalog_path)
                identity = await resolver.resolve_identity(headers=dict(request.headers))
            except dependency_unavailable_error as exc:
                raise HTTPException(
                    status_code=503,
                    detail=exc.public_message,
                    headers={"Connection": "close"},
                ) from exc
            except enforceai_error as exc:
                raise HTTPException(
                    status_code=exc.status_code,
                    detail=exc.public_message,
                    headers={"Connection": "close"},
                ) from exc
            except Exception as exc:  # noqa: BLE001
                logger.exception("Unexpected EnforceAI failure during identity resolution")
                raise HTTPException(
                    status_code=503,
                    detail="Enforcement dependency unavailable",
                    headers={"Connection": "close"},
                ) from exc

            method = tool_name or "initialize"
            actual_tool_name = None
            if method == "tools/call" and isinstance(request_payload, dict):
                params = request_payload.get("params")
                if isinstance(params, dict):
                    actual_tool_name = params.get("name")

            request_id = request.headers.get("X-Request-Id")
            if not request_id and isinstance(request_payload, dict) and "id" in request_payload:
                request_id_value = request_payload.get("id")
                if request_id_value is not None:
                    request_id = str(request_id_value)

            allowed_tools = None
            if isinstance(identity.metadata, dict):
                allowed_tools = identity.metadata.get("agent_allowed_tools")

            allowed_tools_header_value = ""
            if server_name and method == "tools/list":
                tool_policy = resolve_callable_tools_for_server(
                    identity=identity,
                    catalog=catalog,
                    server=server_name,
                    allowed_tools=allowed_tools,
                )
                if tool_policy.all_tools:
                    allowed_tools_header_value = "*"
                else:
                    allowed_tools_header_value = _json.dumps(sorted(tool_policy.tools))

            if server_name and method in {"tools/list", "tools/call"}:
                if method == "tools/list":
                    _emit_enforceai_audit_event(
                        action="tools/list",
                        outcome="allow",
                        user_id=identity.user_id,
                        agent_id=identity.agent_id,
                        request_id=request_id,
                        details={
                            "provider": identity.provider,
                            "server": server_name,
                            "allowed_tools": allowed_tools_header_value,
                        },
                    )

                if method == "tools/call":
                    if not actual_tool_name:
                        _emit_enforceai_audit_event(
                            action="tools/call",
                            outcome="deny",
                            user_id=identity.user_id,
                            agent_id=identity.agent_id,
                            request_id=request_id,
                            details={
                                "provider": identity.provider,
                                "server": server_name,
                                "reason": "missing_tool_name",
                            },
                        )
                        raise HTTPException(
                            status_code=403,
                            detail="Forbidden",
                            headers={"Connection": "close"},
                        )

                    decision = evaluate_tool_call(
                        identity=identity,
                        catalog=catalog,
                        server=server_name,
                        tool=actual_tool_name,
                        allowed_tools=allowed_tools,
                    )
                    _emit_enforceai_audit_event(
                        action="tools/call",
                        outcome="allow" if decision.allowed else "deny",
                        user_id=identity.user_id,
                        agent_id=identity.agent_id,
                        request_id=request_id,
                        details={
                            "provider": identity.provider,
                            "server": server_name,
                            "tool": actual_tool_name,
                            "reason": decision.reason,
                            "matched_scope": decision.matched_scope,
                        },
                    )
                    if not decision.allowed:
                        raise HTTPException(
                            status_code=403,
                            detail="Forbidden",
                            headers={"Connection": "close"},
                        )

            from auth_server.enforceai.models.upstream_auth import (
                UpstreamAuthConfig,
                UpstreamAuthInjection,
            )
            from auth_server.enforceai.upstream.headers import (
                ENFORCEAI_ERROR_CODE_HEADER,
                ENFORCEAI_UPSTREAM_API_KEY_HEADER,
                ENFORCEAI_UPSTREAM_API_KEY_HEADER_NAME_HEADER,
                ENFORCEAI_UPSTREAM_AUTHORIZATION_HEADER,
                ENFORCEAI_UPSTREAM_MODE_HEADER,
                MCP_AUTH_TYPE_HEADER,
                MCP_CLAIMS_HEADER,
                MCP_PRINCIPAL_HEADER,
                MCP_PROVIDER_HEADER,
                MCP_SCOPES_HEADER,
            )
            from auth_server.enforceai.upstream.resolver import (
                UpstreamInjectionError,
                resolve_upstream_injection,
            )

            server_path = request.headers.get("X-EnforceAI-Server-Path")
            if not server_path and server_name:
                server_path = f"/{server_name}"

            upstream_type = (
                request.headers.get("X-EnforceAI-Upstream-Auth-Type") or "none"
            ).strip()
            upstream_binding = (
                request.headers.get("X-EnforceAI-Upstream-Credential-Binding") or "service"
            ).strip()
            upstream_provider = request.headers.get("X-EnforceAI-Upstream-Provider")
            default_upstream_mode = "none" if upstream_type == "none" else "gateway-managed"
            upstream_mode = (
                request.headers.get("X-EnforceAI-Upstream-Mode") or default_upstream_mode
            ).strip()
            upstream_mode = upstream_mode or default_upstream_mode
            upstream_header_name = request.headers.get("X-EnforceAI-Upstream-Header-Name")
            upstream_scheme = request.headers.get("X-EnforceAI-Upstream-Scheme")

            injection = None
            if upstream_type in {"api-key", "jwt", "oauth2", "oidc", "provider-oauth"}:
                header_name = upstream_header_name
                scheme = upstream_scheme
                if not header_name:
                    if upstream_type == "api-key":
                        header_name = "X-API-Key"
                        scheme = None
                    else:
                        header_name = "Authorization"
                        scheme = scheme or "Bearer"
                injection = UpstreamAuthInjection(
                    header_name=header_name,
                    scheme=scheme,
                )

            upstream_auth = UpstreamAuthConfig(
                mode=upstream_mode,
                type=upstream_type,
                provider=upstream_provider,
                credential_binding=upstream_binding,
                injection=injection,
            )

            try:
                oauth_providers = None
                oauth_token_client = None
                oauth_refresh_skew_seconds = 0
                if upstream_auth.type in {"oauth2", "oidc", "provider-oauth"}:
                    settings = get_enforceai_settings()
                    oauth_providers = settings.upstream_oauth_providers
                    oauth_token_client = get_upstream_oauth_token_client()
                    oauth_refresh_skew_seconds = settings.upstream_oauth_refresh_skew_seconds

                allow_missing_upstream = False
                if tool_name == "tools/list" and upstream_auth.type in {
                    "oauth2",
                    "oidc",
                    "provider-oauth",
                }:
                    allow_missing_upstream = True

                injection_result = await resolve_upstream_injection(
                    server_path=server_path,
                    upstream_auth=upstream_auth,
                    identity=identity,
                    stores=get_enforceai_stores(),
                    oauth_providers=oauth_providers,
                    oauth_token_client=oauth_token_client,
                    oauth_refresh_skew_seconds=oauth_refresh_skew_seconds,
                    allow_missing_credential=allow_missing_upstream,
                )
            except UpstreamInjectionError as exc:
                raise HTTPException(
                    status_code=exc.status_code,
                    detail=exc.public_message,
                    headers={
                        "Connection": "close",
                        ENFORCEAI_ERROR_CODE_HEADER: exc.error_code,
                    },
                ) from exc

            response_data = {
                "valid": True,
                "username": identity.user_id,
                "client_id": "",
                "scopes": identity.scopes,
                "method": identity.provider,
                "groups": [],
                "server_name": server_name,
                "tool_name": tool_name,
            }

            response = JSONResponse(
                content=response_data,
                status_code=200,
            )
            response.headers["X-User"] = identity.user_id
            response.headers["X-Username"] = identity.user_id
            response.headers["X-Client-Id"] = ""
            response.headers["X-Scopes"] = " ".join(identity.scopes)
            response.headers["X-Auth-Method"] = identity.provider
            response.headers["X-Server-Name"] = server_name or ""
            response.headers["X-Tool-Name"] = tool_name or ""
            response.headers["X-Agent-Id"] = identity.agent_id
            response.headers["X-Allowed-Tools"] = allowed_tools_header_value
            response.headers[MCP_PRINCIPAL_HEADER] = injection_result.mcp_principal
            response.headers[MCP_AUTH_TYPE_HEADER] = injection_result.mcp_auth_type
            response.headers[MCP_SCOPES_HEADER] = injection_result.mcp_scopes
            response.headers[MCP_PROVIDER_HEADER] = injection_result.mcp_provider
            response.headers[MCP_CLAIMS_HEADER] = injection_result.mcp_claims
            response.headers[ENFORCEAI_UPSTREAM_MODE_HEADER] = injection_result.mode
            response.headers[ENFORCEAI_UPSTREAM_AUTHORIZATION_HEADER] = (
                injection_result.upstream_authorization
            )
            response.headers[ENFORCEAI_UPSTREAM_API_KEY_HEADER] = injection_result.upstream_api_key
            response.headers[ENFORCEAI_UPSTREAM_API_KEY_HEADER_NAME_HEADER] = (
                injection_result.upstream_api_key_header
            )
            return response
        
        # Log request for debugging with anonymized IP
        client_ip = request.client.host if request.client else 'unknown'
        logger.info(f"Validation request from {anonymize_ip(client_ip)}")
        logger.info(f"Request Method: {request.method}")
        
        # Log masked HTTP headers for GDPR/SOX compliance
        all_headers = dict(request.headers)
        masked_headers = mask_headers(all_headers)
        logger.debug(f"HTTP Headers (masked): {json.dumps(masked_headers, indent=2)}")
        
        # Log specific headers for debugging with masked sensitive data
        logger.info(f"Key Headers: Authorization={bool(authorization)}, Cookie={bool(cookie_header)}, "
                    f"User-Pool-Id={mask_sensitive_id(user_pool_id) if user_pool_id else 'None'}, "
                    f"Client-Id={mask_sensitive_id(client_id) if client_id else 'None'}, "
                    f"Region={region}, Original-URL={original_url}")
        logger.info(f"Server Name from URL: {server_name_from_url}")
        
        # Initialize validation result
        validation_result = None
        
        # FIRST: Check for session cookie if present
        if "mcp_gateway_session=" in cookie_header:
            logger.info("Session cookie detected, attempting session validation")
            # Extract cookie value
            cookie_value = None
            for cookie in cookie_header.split(';'):
                if cookie.strip().startswith('mcp_gateway_session='):
                    cookie_value = cookie.strip().split('=', 1)[1]
                    break
            
            if cookie_value:
                try:
                    validation_result = validate_session_cookie(cookie_value)
                    # Log validation result without exposing username
                    safe_result = {k: v for k, v in validation_result.items() if k != 'username'}
                    safe_result['username'] = hash_username(validation_result.get('username', ''))
                    logger.info(f"Session cookie validation result: {safe_result}")
                    logger.info(f"Session cookie validation successful for user: {hash_username(validation_result['username'])}")
                except ValueError as e:
                    logger.warning(f"Session cookie validation failed: {e}")
                    # Fall through to JWT validation
        
        # SECOND: If no valid session cookie, check for JWT token
        if not validation_result:
            # Validate required headers for JWT
            if not authorization or not authorization.startswith("Bearer "):
                logger.warning("Missing or invalid Authorization header and no valid session cookie")
                raise HTTPException(
                    status_code=401,
                    detail="Missing or invalid Authorization header. Expected: Bearer <token> or valid session cookie",
                    headers={"WWW-Authenticate": "Bearer", "Connection": "close"}
                )
            
            # Extract token
            access_token = authorization.split(" ")[1]
            
            # Get authentication provider based on AUTH_PROVIDER environment variable
            try:
                auth_provider = get_auth_provider()
                logger.info(f"Using authentication provider: {auth_provider.__class__.__name__}")
                
                # Provider-specific validation
                if hasattr(auth_provider, 'validate_token'):
                    # For Keycloak, no additional headers needed
                    validation_result = auth_provider.validate_token(access_token)
                    logger.info(f"Token validation successful using {auth_provider.__class__.__name__}")
                else:
                    # Fallback to old validation for compatibility
                    if not user_pool_id:
                        logger.warning("Missing X-User-Pool-Id header for Cognito validation")
                        raise HTTPException(
                            status_code=400,
                            detail="Missing X-User-Pool-Id header",
                            headers={"Connection": "close"}
                        )
                    
                    if not client_id:
                        logger.warning("Missing X-Client-Id header for Cognito validation")
                        raise HTTPException(
                            status_code=400,
                            detail="Missing X-Client-Id header",
                            headers={"Connection": "close"}
                        )
                    
                    # Use old validator for backward compatibility
                    validation_result = validator.validate_token(
                        access_token=access_token,
                        user_pool_id=user_pool_id,
                        client_id=client_id,
                        region=region
                    )
                    
            except Exception as e:
                logger.error(f"Authentication provider error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail=f"Authentication provider configuration error: {str(e)}",
                    headers={"Connection": "close"}
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
                        tool_name = request_payload.get('method')
                        
                        # If not found in method, check other common patterns
                        if not tool_name:
                            tool_name = request_payload.get('tool') or request_payload.get('name')
                            
                        # Check for nested tool reference in params
                        if not tool_name and 'params' in request_payload:
                            params = request_payload['params']
                            if isinstance(params, dict):
                                tool_name = params.get('name') or params.get('tool') or params.get('method')
                        
                        logger.info(f"Extracted tool name from JSON-RPC payload: '{tool_name}'")
                    else:
                        logger.warning(f"Payload is not a dictionary: {type(request_payload)}")
                except Exception as e:
                    logger.error(f"Error processing request payload for tool extraction: {e}")
        
        # Validate scope-based access if we have server/tool information
        # For providers that use groups (Keycloak, Entra ID, Cognito), map groups to scopes
        user_groups = validation_result.get('groups', [])
        auth_method = validation_result.get('method', '')
        if user_groups and auth_method in ['keycloak', 'entra', 'cognito']:
            # Map IdP groups to scopes using the group mappings
            user_scopes = map_groups_to_scopes(user_groups)
            logger.info(f"Mapped {auth_method} groups {user_groups} to scopes: {user_scopes}")
        else:
            user_scopes = validation_result.get('scopes', [])
        if server_name:
            # For ANY server access, enforce scope validation (fail closed principle)
            # This includes MCP initialization methods that may not have a specific tool

            method = tool_name if tool_name else "initialize"  # Default to initialize if no tool specified
            actual_tool_name = None

            # For tools/call, extract the actual tool name from params
            if method == 'tools/call' and isinstance(request_payload, dict):
                params = request_payload.get('params', {})
                if isinstance(params, dict):
                    actual_tool_name = params.get('name')
                    logger.info(f"Extracted actual tool name for tools/call: '{actual_tool_name}'")

            # Check if user has any scopes - if not, deny access (fail closed)
            if not user_scopes:
                logger.warning(f"Access denied for user {hash_username(validation_result.get('username', ''))} to {server_name}.{method} (tool: {actual_tool_name}) - no scopes configured")
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied to {server_name}.{method} - user has no scopes configured",
                    headers={"Connection": "close"}
                )

            if not validate_server_tool_access(server_name, method, actual_tool_name, user_scopes):
                logger.warning(f"Access denied for user {hash_username(validation_result.get('username', ''))} to {server_name}.{method} (tool: {actual_tool_name})")
                raise HTTPException(
                    status_code=403,
                    detail=f"Access denied to {server_name}.{method}",
                    headers={"Connection": "close"}
                )
            logger.info(f"Scope validation passed for {server_name}.{method} (tool: {actual_tool_name})")
        else:
            logger.debug("No server information available, skipping scope validation")
        
        # Prepare JSON response data
        response_data = {
            'valid': True,
            'username': validation_result.get('username') or '',
            'client_id': validation_result.get('client_id') or '',
            'scopes': user_scopes,
            'method': validation_result.get('method') or '',
            'groups': validation_result.get('groups', []),
            'server_name': server_name,
            'tool_name': tool_name
        }
        logger.info(f"Full validation result: {json.dumps(validation_result, indent=2)}")
        logger.info(f"Response data being sent: {json.dumps(response_data, indent=2)}")
        # Create JSON response with headers that nginx can use
        response = JSONResponse(content=response_data, status_code=200)
        
        # Set headers for nginx auth_request_set directives
        response.headers["X-User"] = validation_result.get('username') or ''
        response.headers["X-Username"] = validation_result.get('username') or ''
        response.headers["X-Client-Id"] = validation_result.get('client_id') or ''
        response.headers["X-Scopes"] = ' '.join(user_scopes)
        response.headers["X-Auth-Method"] = validation_result.get('method') or ''
        response.headers["X-Server-Name"] = server_name or ''
        response.headers["X-Tool-Name"] = tool_name or ''
        
        return response
        
    except ValueError as e:
        logger.warning(f"Token validation failed: {e}")
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer", "Connection": "close"},
        )
    except HTTPException as e:
        # Preserve explicit auth/enforcement HTTP status codes
        if e.status_code in {401, 403, 409, 424, 503}:
            raise
        # For other HTTPExceptions, let them fall through to general handler
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
            headers={"Connection": "close"}
        )
    finally:
        pass

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
    validator = SimplifiedCognitoValidator(
        region=args.region,
        secret_key=SECRET_KEY,
        jwt_issuer=JWT_ISSUER,
        jwt_audience=JWT_AUDIENCE,
    )
    
    logger.info(f"Starting simplified auth server on {args.host}:{args.port}")
    logger.info(f"Default region: {args.region}")
    
    uvicorn.run(app, host=args.host, port=args.port)

if __name__ == "__main__":
    main()
SAFE_CSRF_METHODS: set[str] = {"GET", "HEAD", "OPTIONS", "TRACE"}


def _has_non_cookie_credentials_for_csrf(
    request: Request,
) -> bool:
    authorization = request.headers.get("authorization") or ""
    if authorization.strip():
        return True

    for header_name in ("x-api-key", "x-gateway-token", "x-authorization"):
        value = request.headers.get(header_name)
        if value and value.strip():
            return True

    return False


@app.middleware("http")
async def enforce_csrf_middleware(
    request: Request,
    call_next,
):
    if request.method in SAFE_CSRF_METHODS:
        return await call_next(request)

    if not request.url.path.startswith("/enforceai"):
        return await call_next(request)

    if _has_non_cookie_credentials_for_csrf(request):
        return await call_next(request)

    cookie_value = request.cookies.get(SESSION_COOKIE_NAME)
    if cookie_value is None or not cookie_value.strip():
        return await call_next(request)

    try:
        session_payload = signer.loads(cookie_value, max_age=28800)
    except (SignatureExpired, BadSignature):
        return await call_next(request)
    except Exception:
        return await call_next(request)

    normalized = normalize_session_data(
        session_payload,
        default_provider="local",
        max_age_seconds=28800,
    )

    csrf_header = request.headers.get("x-csrf-token") or ""
    error = validate_csrf_token(
        secret_key=SECRET_KEY,
        token=csrf_header,
        session_id=normalized.session_id,
        max_age_seconds=CSRF_TOKEN_MAX_AGE_SECONDS,
    )
    if error is not None:
        return JSONResponse(
            status_code=403,
            content={"detail": error},
        )

    return await call_next(request)

# OAuth2 routes live in `auth_server/routes/oauth2_routes.py` and are mounted on the app above.
