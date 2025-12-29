from __future__ import annotations

import json
import logging
import os
import sqlite3
from typing import (
    Any,
    Awaitable,
    Callable,
    Optional,
)

from fastapi import (
    HTTPException,
    status,
)
from pydantic import (
    BaseModel,
    Field,
)


def _require_admin_user_context(
    user_context: dict | None,
) -> dict:
    if user_context is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    if not user_context.get("is_admin", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Admin required",
        )

    return user_context


def _require_can_modify_servers(
    user_context: dict | None,
) -> dict:
    if user_context is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required",
        )

    if not user_context.get("can_modify_servers", False):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Insufficient privileges to modify servers",
        )

    return user_context


async def _apply_toggle_side_effects(
    *,
    service_path: str,
    server_info: dict,
    new_state: bool,
    server_service: Any,
    faiss_service: Any,
    health_service: Any,
    nginx_service: Any,
    logger: logging.Logger,
) -> tuple[str, str | None]:
    status_result = "disabled"
    last_checked_iso = None
    if new_state:
        logger.info(
            f"Performing immediate health check for {service_path} upon toggle ON..."
        )
        try:
            status_result, last_checked_dt = (
                await health_service.perform_immediate_health_check(service_path)
            )
            last_checked_iso = (
                last_checked_dt.isoformat() if last_checked_dt else None
            )
            logger.info(
                f"Immediate health check for {service_path} completed. Status: {status_result}"
            )
        except Exception as exc:  # noqa: BLE001 - best effort health check
            logger.error(
                f"ERROR during immediate health check for {service_path}: {exc}"
            )
            status_result = f"error: immediate check failed ({type(exc).__name__})"
    else:
        status_result = "disabled"
        logger.info(f"Service {service_path} toggled OFF. Status set to disabled.")

    await faiss_service.add_or_update_service(service_path, server_info, new_state)

    enabled_servers = {
        path: server_service.get_server_info(path)
        for path in server_service.get_enabled_services()
    }
    await nginx_service.generate_config_async(enabled_servers)

    await health_service.broadcast_health_update(service_path)

    return status_result, last_checked_iso


async def _apply_remove_side_effects(
    *,
    service_path: str,
    server_service: Any,
    faiss_service: Any,
    health_service: Any,
    nginx_service: Any,
    logger: logging.Logger,
    remove_server_scopes: Callable[[str], Awaitable[None]] | None = None,
    scopes_error_log_level: str = "warning",
) -> None:
    await faiss_service.remove_service(service_path)

    enabled_servers = {
        server_path: server_service.get_server_info(server_path)
        for server_path in server_service.get_enabled_services()
    }
    await nginx_service.generate_config_async(enabled_servers)

    await health_service.broadcast_health_update(service_path)

    if remove_server_scopes is None:
        return

    try:
        await remove_server_scopes(service_path)
    except Exception as exc:  # noqa: BLE001 - best effort cleanup
        message = f"Failed to remove server {service_path} from scopes: {exc}"
        if scopes_error_log_level == "error":
            logger.error(message)
        else:
            logger.warning(message)


def _normalize_upstream_auth_payload(
    *,
    upstream_auth: object | None,
    auth_type: str | None,
    auth_provider: str | None,
    headers: object | None,
) -> dict:
    try:
        from auth_server.enforceai.models.upstream_auth import (
            normalize_upstream_auth,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed if EnforceAI is unavailable
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upstream auth normalization unavailable",
        ) from exc

    try:
        normalized = normalize_upstream_auth(
            upstream_auth=upstream_auth,
            auth_type=auth_type,
            auth_provider=auth_provider,
            headers=headers,
        )
    except ValueError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid upstream auth configuration: {exc}",
        ) from exc

    return normalized.model_dump()


def _normalize_server_path(
    *,
    raw_path: str,
) -> str:
    stripped = raw_path.strip()
    if not stripped:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="path is required",
        )
    if not stripped.startswith("/"):
        return f"/{stripped}"
    return stripped


def _parse_tags(
    *,
    tags: str,
) -> list[str]:
    if not tags:
        return []

    return [tag.strip() for tag in tags.split(",") if tag.strip()]


def _parse_supported_transports(
    *,
    supported_transports: str | None,
    logger: logging.Logger | None = None,
) -> list[str]:
    if not supported_transports:
        return ["streamable-http"]

    try:
        if supported_transports.startswith("["):
            transports = json.loads(supported_transports)
        else:
            transports = [t.strip() for t in supported_transports.split(",")]
    except Exception as exc:  # noqa: BLE001 - user-provided form data
        if logger is not None:
            logger.warning(
                f"Failed to parse supported_transports, using default: {exc}"
            )
        return ["streamable-http"]

    if not isinstance(transports, list):
        return ["streamable-http"]

    normalized = [str(t).strip() for t in transports if str(t).strip()]
    return normalized or ["streamable-http"]


def _parse_headers(
    *,
    headers: str | None,
    logger: logging.Logger | None = None,
) -> object:
    if not headers:
        return []

    try:
        return json.loads(headers) if isinstance(headers, str) else headers
    except Exception as exc:  # noqa: BLE001 - user-provided form data
        if logger is not None:
            logger.warning(f"Failed to parse headers: {exc}")
        return []


def _parse_tool_list(
    *,
    tool_list_json: str | None,
    logger: logging.Logger | None = None,
) -> list:
    if not tool_list_json:
        return []

    try:
        return (
            json.loads(tool_list_json)
            if isinstance(tool_list_json, str)
            else tool_list_json
        )
    except Exception as exc:  # noqa: BLE001 - user-provided form data
        if logger is not None:
            logger.warning(f"Failed to parse tool_list_json: {exc}")
        return []


def _build_server_entry_from_form(
    *,
    name: str,
    description: str,
    path: str,
    proxy_pass_url: str,
    tags: str,
    num_tools: int,
    num_stars: int,
    is_python: bool,
    license_str: str,
    auth_provider: str | None,
    auth_type: str | None,
    upstream_auth: str | None,
    supported_transports: str | None,
    headers: str | None,
    tool_list_json: str | None,
    logger: logging.Logger | None = None,
) -> tuple[str, dict]:
    normalized_path = path
    if not normalized_path.startswith("/"):
        normalized_path = f"/{normalized_path}"

    tag_list = _parse_tags(tags=tags)
    transports_list = _parse_supported_transports(
        supported_transports=supported_transports,
        logger=logger,
    )
    parsed_headers = _parse_headers(
        headers=headers,
        logger=logger,
    )
    upstream_auth_payload = _normalize_upstream_auth_payload(
        upstream_auth=upstream_auth,
        auth_type=auth_type,
        auth_provider=auth_provider,
        headers=parsed_headers,
    )
    tool_list = _parse_tool_list(
        tool_list_json=tool_list_json,
        logger=logger,
    )

    server_entry: dict[str, object] = {
        "server_name": name,
        "description": description,
        "path": normalized_path,
        "proxy_pass_url": proxy_pass_url,
        "supported_transports": transports_list,
        "auth_type": auth_type if auth_type else "none",
        "upstream_auth": upstream_auth_payload,
        "tags": tag_list,
        "num_tools": num_tools,
        "num_stars": num_stars,
        "is_python": is_python,
        "license": license_str,
        "tool_list": tool_list,
    }

    if auth_provider:
        server_entry["auth_provider"] = auth_provider
    if parsed_headers:
        server_entry["headers"] = parsed_headers

    return normalized_path, server_entry


class ServerCreateRequest(BaseModel):
    name: str = Field(
        ...,
        min_length=1,
        max_length=100,
    )
    path: str = Field(
        ...,
        min_length=1,
        max_length=50,
    )
    proxy_pass_url: str = Field(
        ...,
        min_length=1,
    )
    description: Optional[str] = None
    tags: Optional[list[str]] = None
    upstream_auth: Optional[dict[str, Any]] = None
    overwrite: bool = Field(
        default=False,
        description="If true, replace an existing server at the same path.",
    )


class ServerUpdateRequest(BaseModel):
    name: Optional[str] = Field(
        default=None,
        min_length=1,
        max_length=100,
    )
    proxy_pass_url: Optional[str] = Field(
        default=None,
        min_length=1,
    )
    description: Optional[str] = Field(
        default=None,
        max_length=500,
    )
    tags: Optional[list[str]] = None
    upstream_auth: Optional[dict[str, Any]] = None
    enabled: Optional[bool] = None


def _enforce_proxy_pass_url_allowlist(
    *,
    proxy_pass_url: str,
) -> None:
    db_path = os.getenv("ENFORCEAI_DB_PATH")
    if db_path is None or not db_path.strip():
        return

    try:
        from pathlib import Path

        from auth_server.enforceai.egress.allowlist import (
            check_proxy_pass_url,
        )
        from auth_server.enforceai.stores.sqlite.egress_allowlist_store import (
            SqliteEgressAllowlistStore,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed if EnforceAI is unavailable
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Egress allowlist enforcement unavailable",
        ) from exc

    store = SqliteEgressAllowlistStore(db_path=Path(db_path))
    try:
        entries = store.list_entries(include_expired=False)
    except Exception as exc:  # noqa: BLE001 - treat as enforcement misconfigured
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Egress allowlist store unavailable",
        ) from exc

    if not entries:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="proxy_pass_url not allowed (egress allowlist is empty)",
        )

    decision = check_proxy_pass_url(
        proxy_pass_url=proxy_pass_url,
        entries=entries,
    )
    if not decision.allowed:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"proxy_pass_url not allowed: {decision.reason}",
        )


def _enforce_upstream_oauth_provider_configured(
    *,
    upstream_auth: dict,
) -> None:
    upstream_auth_type = (upstream_auth.get("type") or "").strip()
    if upstream_auth_type not in {"oauth2", "oidc", "provider-oauth"}:
        return

    provider_id = (upstream_auth.get("provider") or "").strip()
    if not provider_id:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid upstream auth configuration: provider is required for OAuth upstream auth",
        )

    db_path = os.getenv("ENFORCEAI_DB_PATH")
    if db_path is None or not db_path.strip():
        return

    try:
        from pathlib import Path

        from auth_server.enforceai.db.connection import (
            sqlite_connection,
        )
    except Exception as exc:  # noqa: BLE001 - fail closed if EnforceAI is unavailable
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Upstream OAuth provider validation unavailable",
        ) from exc

    try:
        with sqlite_connection(Path(db_path)) as connection:
            row = connection.execute(
                "SELECT 1 FROM upstream_oauth_providers WHERE provider_id = ?",
                (provider_id,),
            ).fetchone()
    except sqlite3.Error as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Upstream OAuth provider registry unavailable",
        ) from exc

    if row is None:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Invalid upstream auth configuration: unknown upstream OAuth provider '{provider_id}'",
        )
