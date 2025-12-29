from __future__ import annotations

import hashlib
import json
import logging
import os
from datetime import (
    datetime,
    timezone,
)
from pathlib import Path
from typing import Any, Optional

from itsdangerous import (
    BadSignature,
    SignatureExpired,
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

try:
    import auth_server.scopes_config as scopes_config
except Exception:  # noqa: BLE001
    import scopes_config  # type: ignore[no-redef]

try:
    from .routes.oauth2_context import (
        signer,
    )
except ImportError:  # pragma: no cover
    from routes.oauth2_context import (
        signer,
    )

logger = logging.getLogger(__name__)


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
        parts = ip_address.split(":")
        if len(parts) > 1:
            parts[-1] = "xxxx"
            return ":".join(parts)
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
    masked: dict[str, Any] = {}
    for key, value in headers.items():
        key_lower = key.lower()
        if key_lower in ["x-authorization", "authorization", "cookie"]:
            if "bearer" in str(value).lower():
                parts = str(value).split(" ", 1)
                if len(parts) == 2:
                    masked[key] = f"Bearer {mask_token(parts[1])}"
                else:
                    masked[key] = mask_token(str(value))
            else:
                masked[key] = "***MASKED***"
        elif key_lower in ["x-user-pool-id", "x-client-id"]:
            masked[key] = mask_sensitive_id(str(value))
        else:
            masked[key] = value
    return masked


def map_groups_to_scopes(groups: list[str]) -> list[str]:
    """Map identity provider groups to MCP scopes using the group mappings config."""
    current_scopes_config = scopes_config.get_scopes_config()
    group_mappings = current_scopes_config.get("group_mappings", {})

    scopes: list[str] = []
    for group in groups:
        if group in group_mappings:
            group_scopes = group_mappings[group]
            scopes.extend(group_scopes)
            logger.debug("Mapped group '%s' to scopes: %s", group, group_scopes)
        else:
            logger.debug("No scope mapping found for group: %s", group)

    seen: set[str] = set()
    unique_scopes: list[str] = []
    for scope in scopes:
        if scope not in seen:
            seen.add(scope)
            unique_scopes.append(scope)

    logger.info("Final mapped scopes: %s", unique_scopes)
    return unique_scopes


def validate_session_cookie(cookie_value: str) -> dict[str, Any]:
    """Validate session cookie using the itsdangerous signer."""
    try:
        data = signer.loads(cookie_value, max_age=28800)

        normalized = normalize_session_data(
            data,
            default_provider="local",
            max_age_seconds=28800,
        )

        db_path_raw = os.environ.get("ENFORCEAI_DB_PATH")
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

        username = normalized.user_id
        groups = normalized.groups or []
        scopes = map_groups_to_scopes(groups)

        logger.info("Session cookie validated for user: %s", hash_username(username))

        return {
            "valid": True,
            "username": username,
            "scopes": scopes,
            "method": "session_cookie",
            "groups": groups,
            "client_id": "",
            "data": data,
        }
    except SignatureExpired as exc:
        logger.warning("Session cookie has expired")
        raise ValueError("Session cookie has expired") from exc
    except BadSignature as exc:
        logger.warning("Invalid session cookie signature")
        raise ValueError("Invalid session cookie") from exc
    except Exception as exc:
        logger.error("Session cookie validation error: %s", exc)
        raise ValueError(f"Session cookie validation failed: {exc}") from exc


def parse_server_and_tool_from_url(
    original_url: str,
) -> tuple[Optional[str], Optional[str]]:
    """Parse server name and tool name from the original URL."""
    try:
        from urllib.parse import urlparse

        parsed_url = urlparse(original_url)
        path = parsed_url.path.strip("/")
        path_parts = path.split("/") if path else []
        server_name = path_parts[0] if path_parts else None
        logger.debug("Parsed server name '%s' from URL path: %s", server_name, path)
        return server_name, None
    except Exception as exc:
        logger.error("Failed to parse server/tool from URL %s: %s", original_url, exc)
        return None, None


def _normalize_server_name(name: str) -> str:
    return name.rstrip("/") if name else name


def _server_names_match(name1: str, name2: str) -> bool:
    normalized_name1 = _normalize_server_name(name1)
    if normalized_name1 == "*":
        return True
    return normalized_name1 == _normalize_server_name(name2)


def validate_server_tool_access(
    server_name: str,
    method: str,
    tool_name: Optional[str],
    user_scopes: list[str],
) -> bool:
    try:
        logger.info("=== VALIDATE_SERVER_TOOL_ACCESS START ===")
        logger.info("Requested server: '%s'", server_name)
        logger.info("Requested method: '%s'", method)
        logger.info("Requested tool: '%s'", tool_name)
        logger.info("User scopes: %s", user_scopes)

        current_scopes_config = scopes_config.get_scopes_config()
        logger.info(
            "Available scopes config keys: %s",
            list(current_scopes_config.keys()) if current_scopes_config else "None",
        )

        if not current_scopes_config:
            logger.warning("No scopes configuration loaded, allowing access")
            logger.info("=== VALIDATE_SERVER_TOOL_ACCESS END: ALLOWED (no config) ===")
            return True

        for scope in user_scopes:
            logger.info("--- Checking scope: '%s' ---", scope)
            scope_config = current_scopes_config.get(scope, [])

            if not scope_config:
                logger.info("Scope '%s' not found in configuration", scope)
                continue

            logger.info("Scope '%s' config: %s", scope, scope_config)

            for server_config in scope_config:
                logger.info("  Examining server config: %s", server_config)
                server_config_name = server_config.get("server")
                logger.info(
                    "  Server name in config: '%s' vs requested: '%s'",
                    server_config_name,
                    server_name,
                )

                if not _server_names_match(server_config_name, server_name):
                    logger.info("  Server name does not match")
                    continue

                logger.info("  Server name matches")

                allowed_methods = server_config.get("methods", [])
                logger.info("  Allowed methods for server '%s': %s", server_name, allowed_methods)
                has_wildcard_methods = "all" in allowed_methods or "*" in allowed_methods

                if (method in allowed_methods or has_wildcard_methods) and method != "tools/call":
                    logger.info("Access granted: scope '%s' allows access to %s.%s", scope, server_name, method)
                    logger.info("=== VALIDATE_SERVER_TOOL_ACCESS END: GRANTED ===")
                    return True

                allowed_tools = server_config.get("tools", [])
                logger.info("  Allowed tools for server '%s': %s", server_name, allowed_tools)
                has_wildcard_tools = "all" in allowed_tools or "*" in allowed_tools

                if method == "tools/call" and tool_name:
                    logger.info("  Checking if tool '%s' is allowed for tools/call", tool_name)
                    if tool_name in allowed_tools or has_wildcard_tools:
                        logger.info(
                            "Access granted: scope '%s' allows access to %s.%s for tool %s",
                            scope,
                            server_name,
                            method,
                            tool_name,
                        )
                        logger.info("=== VALIDATE_SERVER_TOOL_ACCESS END: GRANTED ===")
                        return True
                    logger.info("  Tool '%s' not found in allowed tools", tool_name)
                else:
                    logger.info("  Checking if method '%s' is in allowed tools...", method)
                    if method in allowed_tools or has_wildcard_tools:
                        logger.info("Access granted: scope '%s' allows access to %s.%s", scope, server_name, method)
                        logger.info("=== VALIDATE_SERVER_TOOL_ACCESS END: GRANTED ===")
                        return True
                    logger.info("  Method '%s' not found in allowed tools", method)

        logger.warning(
            "Access denied: no scope allows access to %s.%s (tool: %s) for user scopes: %s",
            server_name,
            method,
            tool_name,
            user_scopes,
        )
        logger.info("=== VALIDATE_SERVER_TOOL_ACCESS END: DENIED ===")
        return False
    except Exception as exc:
        logger.error("Error validating server/tool access: %s", exc)
        logger.info("=== VALIDATE_SERVER_TOOL_ACCESS END: ERROR ===")
        return False

