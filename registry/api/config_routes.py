"""Configuration API endpoint for deployment mode awareness."""

import json
import logging
import time
from enum import Enum
from typing import Annotated, Any, Dict, List

from fastapi import APIRouter, Depends, HTTPException, Query, Request
from fastapi.responses import PlainTextResponse

from ..auth.dependencies import enhanced_auth
from ..core.config import settings, DeploymentMode, RegistryMode
from ..core.metrics import CONFIG_VIEW_REQUESTS, CONFIG_EXPORT_REQUESTS

logger = logging.getLogger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Rate limiting state (in-memory sliding window, per-user)
# ---------------------------------------------------------------------------
_rate_limit_cache: Dict[str, List[float]] = {}
RATE_LIMIT_REQUESTS = 10
RATE_LIMIT_WINDOW_SECONDS = 60


# ---------------------------------------------------------------------------
# Configuration group definitions — 11 groups, ordered 1-11
# Each field tuple: (settings_attr_name, display_label, is_sensitive)
# ---------------------------------------------------------------------------
CONFIG_GROUPS: Dict[str, Dict[str, Any]] = {
    "deployment": {
        "title": "Deployment Mode",
        "order": 1,
        "fields": [
            ("deployment_mode", "Deployment Mode", False),
            ("registry_mode", "Registry Mode", False),
            ("nginx_updates_enabled", "Nginx Updates Enabled", False),
        ],
    },
    "storage": {
        "title": "Storage Backend",
        "order": 2,
        "fields": [
            ("storage_backend", "Storage Backend", False),
            ("documentdb_host", "DocumentDB Host", False),
            ("documentdb_port", "DocumentDB Port", False),
            ("documentdb_database", "DocumentDB Database", False),
            ("documentdb_namespace", "DocumentDB Namespace", False),
            ("documentdb_use_tls", "Use TLS", False),
            ("documentdb_use_iam", "Use IAM Auth", False),
            ("documentdb_username", "Username", True),
            ("documentdb_password", "Password", True),
        ],
    },
    "auth": {
        "title": "Authentication",
        "order": 3,
        "fields": [
            ("auth_provider", "Auth Provider", False),
            ("auth_server_url", "Auth Server URL", False),
            ("auth_server_external_url", "Auth Server External URL", False),
            ("session_max_age_seconds", "Session Max Age", False),
            ("session_cookie_secure", "Secure Cookie", False),
            ("session_cookie_domain", "Cookie Domain", False),
            ("oauth_store_tokens_in_session", "Store OAuth IdP Tokens in Session Cookie", False),
            ("registry_static_token_auth_enabled", "Static Token Auth Enabled", False),
            ("registry_api_token", "Registry API Token", True),
            ("max_tokens_per_user_per_hour", "JWT Token Vending Rate Limit (per user/hour)", False),
            ("secret_key", "Secret Key", True),
        ],
    },
    "embeddings": {
        "title": "Embeddings / Vector Search",
        "order": 4,
        "fields": [
            ("embeddings_provider", "Provider", False),
            ("embeddings_model_name", "Model Name", False),
            ("embeddings_model_dimensions", "Dimensions", False),
            ("embeddings_aws_region", "AWS Region", False),
            ("vector_search_ef_search", "Vector Search EF", False),
            ("embeddings_api_key", "API Key", True),
            ("embeddings_secret_key", "Secret Key", True),
        ],
    },
    "health_check": {
        "title": "Health Checks",
        "order": 5,
        "fields": [
            ("health_check_interval_seconds", "Check Interval", False),
            ("health_check_timeout_seconds", "Check Timeout", False),
        ],
    },
    "websocket": {
        "title": "WebSocket Settings",
        "order": 6,
        "fields": [
            ("max_websocket_connections", "Max Connections", False),
            ("websocket_send_timeout_seconds", "Send Timeout", False),
            ("websocket_broadcast_interval_ms", "Broadcast Interval", False),
            ("websocket_max_batch_size", "Max Batch Size", False),
            ("websocket_cache_ttl_seconds", "Cache TTL", False),
        ],
    },
    "security_servers": {
        "title": "Security Scanning (MCP Servers)",
        "order": 7,
        "fields": [
            ("security_scan_enabled", "Scan Enabled", False),
            ("security_scan_on_registration", "Scan on Registration", False),
            ("security_block_unsafe_servers", "Block Unsafe", False),
            ("security_analyzers", "Analyzers", False),
            ("security_scan_timeout", "Scan Timeout", False),
            ("security_add_pending_tag", "Add Pending Tag", False),
            ("mcp_scanner_llm_api_key", "LLM API Key", True),
        ],
    },
    "security_agents": {
        "title": "Security Scanning (Agents)",
        "order": 8,
        "fields": [
            ("agent_security_scan_enabled", "Scan Enabled", False),
            ("agent_security_scan_on_registration", "Scan on Registration", False),
            ("agent_security_block_unsafe_agents", "Block Unsafe", False),
            ("agent_security_analyzers", "Analyzers", False),
            ("agent_security_scan_timeout", "Scan Timeout", False),
            ("agent_security_add_pending_tag", "Add Pending Tag", False),
            ("a2a_scanner_llm_api_key", "LLM API Key", True),
        ],
    },
    "audit": {
        "title": "Audit Logging",
        "order": 9,
        "fields": [
            ("audit_log_enabled", "Enabled", False),
            ("audit_log_dir", "Log Directory", False),
            ("audit_log_rotation_hours", "Rotation Hours", False),
            ("audit_log_rotation_max_mb", "Max Size (MB)", False),
            ("audit_log_local_retention_hours", "Local Retention Hours", False),
            ("audit_log_mongodb_enabled", "MongoDB Enabled", False),
            ("audit_log_mongodb_ttl_days", "MongoDB TTL Days", False),
            ("audit_log_health_checks", "Log Health Checks", False),
            ("audit_log_static_assets", "Log Static Assets", False),
        ],
    },
    "federation": {
        "title": "Federation",
        "order": 10,
        "fields": [
            ("registry_id", "Registry ID", False),
            ("federation_static_token_auth_enabled", "Static Token Auth Enabled", False),
            ("federation_static_token", "Federation Static Token", True),
        ],
    },
    "discovery": {
        "title": "Well-Known Discovery",
        "order": 11,
        "fields": [
            ("enable_wellknown_discovery", "Enabled", False),
            ("wellknown_cache_ttl", "Cache TTL", False),
        ],
    },
}


# ---------------------------------------------------------------------------
# Sensitive field patterns for automatic detection (defense-in-depth)
# ---------------------------------------------------------------------------
SENSITIVE_PATTERNS = (
    "_password",
    "_secret",
    "_api_key",
    "_token",
    "_key",
    "_credential",
)


def _is_sensitive_field(field_name: str) -> bool:
    """Check if a field should be treated as sensitive based on its name."""
    field_lower = field_name.lower()
    return any(pattern in field_lower for pattern in SENSITIVE_PATTERNS)


def _mask_sensitive_value(value: Any) -> str:
    """Mask a sensitive value for display.

    Returns "(not set)" for None/empty, "****" for ≤4 chars,
    first 4 chars + up to 8 asterisks for longer values.
    """
    if value is None or value == "":
        return "(not set)"
    str_value = str(value)
    if len(str_value) <= 4:
        return "****"
    return str_value[:4] + "*" * min(len(str_value) - 4, 8)


def _format_value(
    field_name: str,
    value: Any,
    is_sensitive: bool,
) -> Dict[str, Any]:
    """Format a configuration value for the API response.

    Returns dict with keys: raw, display, is_masked, unit.
    Handles _seconds/_ms/_hours/_days/_mb suffixes and human-readable time.
    """
    if is_sensitive:
        return {
            "raw": None,
            "display": _mask_sensitive_value(value),
            "is_masked": True,
            "unit": None,
        }

    display = str(value)
    unit = None

    if field_name.endswith("_seconds"):
        unit = "seconds"
        if isinstance(value, (int, float)) and value >= 3600:
            hours = value / 3600
            display = f"{value} ({hours:.1f} hours)"
        elif isinstance(value, (int, float)) and value >= 60:
            minutes = value / 60
            display = f"{value} ({minutes:.0f} minutes)"
    elif field_name.endswith("_ms"):
        unit = "ms"
    elif field_name.endswith("_hours"):
        unit = "hours"
    elif field_name.endswith("_days"):
        unit = "days"
    elif field_name.endswith("_mb"):
        unit = "MB"

    return {
        "raw": value,
        "display": display,
        "is_masked": False,
        "unit": unit,
    }


def _get_field_value(field_name: str) -> Any:
    """Read a field value from the global settings instance.

    Extracts .value from Enum instances and handles the computed
    nginx_updates_enabled property.
    """
    value = getattr(settings, field_name, None)

    # Extract primitive from Enum
    if hasattr(value, "value"):
        value = value.value

    return value


# ---------------------------------------------------------------------------
# Rate limiter (in-memory sliding window)
# ---------------------------------------------------------------------------


def _check_rate_limit(user_id: str) -> bool:
    """Return True if the request is within the rate limit, False otherwise.

    Uses a per-user sliding window of RATE_LIMIT_WINDOW_SECONDS with a max
    of RATE_LIMIT_REQUESTS.
    """
    now = time.time()
    window_start = now - RATE_LIMIT_WINDOW_SECONDS

    if user_id not in _rate_limit_cache:
        _rate_limit_cache[user_id] = []

    # Prune timestamps outside the window
    _rate_limit_cache[user_id] = [t for t in _rate_limit_cache[user_id] if t > window_start]

    if len(_rate_limit_cache[user_id]) >= RATE_LIMIT_REQUESTS:
        return False

    _rate_limit_cache[user_id].append(now)
    return True


# ---------------------------------------------------------------------------
# Response cache (60-second TTL)
# ---------------------------------------------------------------------------
_config_cache: Dict[str, Any] = {}
_config_cache_time: float = 0
CONFIG_CACHE_TTL_SECONDS = 60


def _get_cached_config_response() -> Dict[str, Any]:
    """Return cached config response, rebuilding if TTL has expired."""
    global _config_cache, _config_cache_time

    now = time.time()
    if _config_cache and (now - _config_cache_time) < CONFIG_CACHE_TTL_SECONDS:
        return _config_cache

    _config_cache = _build_config_response()
    _config_cache_time = now
    return _config_cache


def _build_config_response() -> Dict[str, Any]:
    """Build the full configuration response with grouped settings."""
    groups = []

    for group_id, group_def in sorted(
        CONFIG_GROUPS.items(),
        key=lambda x: x[1]["order"],
    ):
        fields = []
        for field_name, display_name, is_sensitive in group_def["fields"]:
            value = _get_field_value(field_name)
            actual_sensitive = is_sensitive or _is_sensitive_field(field_name)
            formatted = _format_value(field_name, value, actual_sensitive)

            fields.append(
                {
                    "key": field_name,
                    "label": display_name,
                    "value": formatted["display"],
                    "raw_value": formatted["raw"],
                    "is_masked": formatted["is_masked"],
                    "unit": formatted["unit"],
                }
            )

        groups.append(
            {
                "id": group_id,
                "title": group_def["title"],
                "order": group_def["order"],
                "fields": fields,
            }
        )

    return {
        "groups": groups,
        "total_groups": len(groups),
        "is_local_dev": settings.is_local_dev,
    }


# ---------------------------------------------------------------------------
# GET /api/config/full — admin-only full configuration view
# ---------------------------------------------------------------------------


@router.get(
    "/full",
    summary="Get full registry configuration",
    description="Returns all configuration parameters grouped by category. Admin only.",
)
async def get_full_config(
    request: Request,
    user_context: Annotated[dict, Depends(enhanced_auth)],
) -> Dict[str, Any]:
    """Get full configuration with grouped parameters."""
    if not user_context.get("is_admin", False):
        raise HTTPException(
            status_code=403,
            detail="Admin access required to view full configuration",
        )

    username = user_context.get("username", "unknown")

    if not _check_rate_limit(username):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
        )

    CONFIG_VIEW_REQUESTS.labels(user_type="admin").inc()

    # Audit log
    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        "Config view requested by user=%s ip=%s groups=%s",
        username,
        client_ip,
        list(CONFIG_GROUPS.keys()),
    )

    audit_logger = getattr(request.app.state, "audit_logger", None)
    if audit_logger:
        try:
            from ..audit.models import (
                RegistryApiAccessRecord,
                Identity,
                Request as AuditRequest,
                Response as AuditResponse,
                Action,
            )
            from datetime import datetime, timezone
            import uuid

            record = RegistryApiAccessRecord(
                timestamp=datetime.now(timezone.utc),
                request_id=str(uuid.uuid4()),
                identity=Identity(
                    username=username,
                    auth_method=user_context.get("auth_method", "unknown"),
                    is_admin=True,
                    credential_type="session_cookie",
                ),
                request=AuditRequest(
                    method="GET",
                    path="/api/config/full",
                    client_ip=client_ip,
                ),
                response=AuditResponse(status_code=200, duration_ms=0),
                action=Action(
                    operation="read",
                    resource_type="config",
                    description="Viewed full system configuration",
                ),
            )
            await audit_logger.log_event(record)
        except Exception:
            logger.debug("Could not write structured audit event for config_view", exc_info=True)

    return _get_cached_config_response()


# ---------------------------------------------------------------------------
# Existing endpoint (unchanged)
# ---------------------------------------------------------------------------


@router.get(
    "",
    summary="Get registry configuration",
    description="Returns the current deployment mode, registry mode, and enabled features",
)
async def get_config() -> Dict[str, Any]:
    """Get current registry configuration."""
    return {
        "deployment_mode": settings.deployment_mode.value,
        "registry_mode": settings.registry_mode.value,
        "nginx_updates_enabled": settings.nginx_updates_enabled,
        "features": {
            "mcp_servers": settings.registry_mode
            in (RegistryMode.FULL, RegistryMode.MCP_SERVERS_ONLY),
            "agents": settings.registry_mode in (RegistryMode.FULL, RegistryMode.AGENTS_ONLY),
            "skills": settings.registry_mode in (RegistryMode.FULL, RegistryMode.SKILLS_ONLY),
            "federation": settings.registry_mode == RegistryMode.FULL,
            "gateway_proxy": settings.deployment_mode == DeploymentMode.WITH_GATEWAY,
        },
    }


# ---------------------------------------------------------------------------
# Export format enum and export helpers
# ---------------------------------------------------------------------------


class ExportFormat(str, Enum):
    """Supported configuration export formats."""

    ENV = "env"
    JSON = "json"
    TFVARS = "tfvars"
    YAML = "yaml"


def _export_as_env(include_sensitive: bool = False) -> str:
    """Export configuration as .env file format.

    Uppercased keys, group header comments, commented-out sensitive fields
    when include_sensitive is False, lowercase booleans, commented-out None values.
    """
    lines = [
        "# MCP Gateway Registry Configuration",
        f"# Exported: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "# WARNING: Sensitive values are masked unless explicitly included",
        "",
    ]

    for group_id, group_def in sorted(CONFIG_GROUPS.items(), key=lambda x: x[1]["order"]):
        lines.append(f"# === {group_def['title']} ===")
        for field_name, _display_name, is_sensitive in group_def["fields"]:
            value = _get_field_value(field_name)
            env_key = field_name.upper()
            sensitive = is_sensitive or _is_sensitive_field(field_name)

            if sensitive:
                if include_sensitive:
                    lines.append(f"{env_key}={value}")
                else:
                    lines.append(f"# {env_key}=<SENSITIVE_VALUE_MASKED>")
            elif value is None:
                lines.append(f"# {env_key}=")
            elif isinstance(value, bool):
                lines.append(f"{env_key}={str(value).lower()}")
            else:
                lines.append(f"{env_key}={value}")
        lines.append("")

    return "\n".join(lines)


def _export_as_json(include_sensitive: bool = False) -> str:
    """Export configuration as JSON with _metadata and configuration sections.

    Uses json.dumps with default=str for non-serialisable types.
    """
    config: Dict[str, Dict[str, Any]] = {}
    for group_id, group_def in CONFIG_GROUPS.items():
        group_config: Dict[str, Any] = {}
        for field_name, _display_name, is_sensitive in group_def["fields"]:
            value = _get_field_value(field_name)
            sensitive = is_sensitive or _is_sensitive_field(field_name)

            if sensitive and not include_sensitive:
                group_config[field_name] = "<MASKED>"
            else:
                group_config[field_name] = value
        config[group_id] = group_config

    return json.dumps(
        {
            "_metadata": {
                "exported_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "registry_mode": settings.registry_mode.value,
                "includes_sensitive": include_sensitive,
            },
            "configuration": config,
        },
        indent=2,
        default=str,
    )


def _export_as_tfvars(include_sensitive: bool = False) -> str:
    """Export configuration as Terraform .tfvars format.

    Lowercase keys, quoted strings, unquoted booleans/numbers,
    commented-out sensitive/None values.
    """
    lines = [
        "# MCP Gateway Registry - Terraform Variables",
        f"# Exported: {time.strftime('%Y-%m-%d %H:%M:%S UTC', time.gmtime())}",
        "",
    ]

    for group_id, group_def in sorted(CONFIG_GROUPS.items(), key=lambda x: x[1]["order"]):
        lines.append(f"# {group_def['title']}")
        for field_name, _display_name, is_sensitive in group_def["fields"]:
            value = _get_field_value(field_name)
            tf_key = field_name.lower()
            sensitive = is_sensitive or _is_sensitive_field(field_name)

            if sensitive:
                if include_sensitive:
                    if isinstance(value, str):
                        lines.append(f'{tf_key} = "{value}"')
                    else:
                        lines.append(f"{tf_key} = {value}")
                else:
                    lines.append(f'# {tf_key} = "<SENSITIVE>"')
            elif value is None:
                lines.append(f"# {tf_key} = null")
            elif isinstance(value, bool):
                lines.append(f"{tf_key} = {str(value).lower()}")
            elif isinstance(value, (int, float)):
                lines.append(f"{tf_key} = {value}")
            elif isinstance(value, str):
                lines.append(f'{tf_key} = "{value}"')
            else:
                lines.append(f'{tf_key} = "{value}"')
        lines.append("")

    return "\n".join(lines)


def _export_as_yaml(include_sensitive: bool = False) -> str:
    """Export configuration as YAML with metadata and configuration sections.

    Lowercase booleans, multi-line string handling with block scalar (|).
    """
    lines = [
        "# MCP Gateway Registry Configuration",
        f"# Exported: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        "",
        "metadata:",
        f"  exported_at: {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())}",
        f"  registry_mode: {settings.registry_mode.value}",
        f"  includes_sensitive: {str(include_sensitive).lower()}",
        "",
        "configuration:",
    ]

    for group_id, group_def in sorted(CONFIG_GROUPS.items(), key=lambda x: x[1]["order"]):
        lines.append(f"  # {group_def['title']}")
        lines.append(f"  {group_id}:")
        for field_name, _display_name, is_sensitive in group_def["fields"]:
            value = _get_field_value(field_name)
            sensitive = is_sensitive or _is_sensitive_field(field_name)

            if sensitive:
                if include_sensitive:
                    if isinstance(value, str):
                        lines.append(f'    {field_name}: "{value}"')
                    else:
                        lines.append(f"    {field_name}: {value}")
                else:
                    lines.append(f'    {field_name}: "<MASKED>"')
            elif value is None:
                lines.append(f"    {field_name}: null")
            elif isinstance(value, bool):
                lines.append(f"    {field_name}: {str(value).lower()}")
            elif isinstance(value, str):
                if "\n" in value:
                    lines.append(f"    {field_name}: |")
                    for line in value.split("\n"):
                        lines.append(f"      {line}")
                else:
                    lines.append(f'    {field_name}: "{value}"')
            else:
                lines.append(f"    {field_name}: {value}")
        lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# GET /api/config/export — admin-only configuration export
# ---------------------------------------------------------------------------

_EXPORT_CONTENT_TYPES = {
    ExportFormat.ENV: "text/plain",
    ExportFormat.JSON: "application/json",
    ExportFormat.TFVARS: "text/plain",
    ExportFormat.YAML: "application/x-yaml",
}

_EXPORT_FILENAMES = {
    ExportFormat.ENV: "mcp-registry.env",
    ExportFormat.JSON: "mcp-registry-config.json",
    ExportFormat.TFVARS: "mcp-registry.tfvars",
    ExportFormat.YAML: "mcp-registry-config.yaml",
}

_EXPORT_FUNCTIONS = {
    ExportFormat.ENV: _export_as_env,
    ExportFormat.JSON: _export_as_json,
    ExportFormat.TFVARS: _export_as_tfvars,
    ExportFormat.YAML: _export_as_yaml,
}


@router.get(
    "/export",
    summary="Export registry configuration",
    description="Export configuration in various formats. Admin only.",
)
async def export_config(
    request: Request,
    user_context: Annotated[dict, Depends(enhanced_auth)],
    format: ExportFormat = Query(
        ExportFormat.ENV,
        description="Export format: env, json, tfvars, yaml",
    ),
    include_sensitive: bool = Query(
        False,
        description="Include sensitive values (use with caution)",
    ),
) -> PlainTextResponse:
    """Export configuration in the specified format."""
    if not user_context.get("is_admin", False):
        raise HTTPException(
            status_code=403,
            detail="Admin access required to export configuration",
        )

    username = user_context.get("username", "unknown")

    if not _check_rate_limit(username):
        raise HTTPException(
            status_code=429,
            detail="Rate limit exceeded. Try again later.",
        )

    CONFIG_EXPORT_REQUESTS.labels(
        format=format.value,
        includes_sensitive=str(include_sensitive),
    ).inc()

    # Audit log
    client_ip = request.client.host if request.client else "unknown"
    logger.info(
        "Config export requested by user=%s format=%s include_sensitive=%s ip=%s",
        username,
        format.value,
        include_sensitive,
        client_ip,
    )

    audit_logger = getattr(request.app.state, "audit_logger", None)
    if audit_logger:
        try:
            from ..audit.models import (
                RegistryApiAccessRecord,
                Identity,
                Request as AuditRequest,
                Response as AuditResponse,
                Action,
            )
            from datetime import datetime, timezone
            import uuid

            record = RegistryApiAccessRecord(
                timestamp=datetime.now(timezone.utc),
                request_id=str(uuid.uuid4()),
                identity=Identity(
                    username=username,
                    auth_method=user_context.get("auth_method", "unknown"),
                    is_admin=True,
                    credential_type="session_cookie",
                ),
                request=AuditRequest(
                    method="GET",
                    path="/api/config/export",
                    client_ip=client_ip,
                    query_params={
                        "format": format.value,
                        "include_sensitive": include_sensitive,
                    },
                ),
                response=AuditResponse(status_code=200, duration_ms=0),
                action=Action(
                    operation="read",
                    resource_type="config",
                    description=f"Exported configuration as {format.value}",
                ),
            )
            await audit_logger.log_event(record)
        except Exception:
            logger.debug("Could not write structured audit event for config_export", exc_info=True)

    content = _EXPORT_FUNCTIONS[format](include_sensitive)
    media_type = _EXPORT_CONTENT_TYPES[format]
    filename = _EXPORT_FILENAMES[format]

    return PlainTextResponse(
        content=content,
        media_type=media_type,
        headers={"Content-Disposition": f'attachment; filename="{filename}"'},
    )
