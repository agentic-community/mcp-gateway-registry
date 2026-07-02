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
import re
import secrets

# Import shared scopes loader and repository factory from registry common module
import sys
import time
import urllib.parse
import uuid
from collections.abc import Mapping
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
from fastapi import APIRouter, Cookie, Depends, FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse, RedirectResponse, Response

# Import metrics middleware
from internal_request_token import (
    mint_mcp_proxy_token,
    mint_registry_ui_token,
    verify_mcp_proxy_token,
)
from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer
from jwt.api_jwk import PyJWK
from metrics_middleware import add_auth_metrics_middleware

try:
    from observability.meters import token_mint_total
except ImportError:
    from auth_server.observability.meters import token_mint_total

# Import provider factory
from providers.factory import get_auth_provider
from pydantic import (
    BaseModel,
    Field,
    StrictStr,
    field_validator,
)

sys.path.insert(0, "/app")
# Import MCP audit logging components
from registry.audit.mcp_logger import MCPLogger
from registry.audit.models import Identity, MCPServer, TokenMintAuditRecord
from registry.audit.service import AuditLogger
from registry.audit.sink import emit_audit_event
from registry.common.scopes_loader import reload_scopes_config
from registry.core.config import settings
from registry.repositories.factory import get_scope_repository

# Configure logging using shared module (RotatingFileHandler + optional MongoDB)
from registry.utils.logging_setup import setup_logging as _setup_logging
from registry.utils.request_utils import get_client_ip

# Let setup_logging resolve the file path from settings.log_dir /
# {service_name}.log. Honors APP_LOG_DIR overrides and the new
# /var/log/containers/ai-registry default introduced by issue #987.
_auth_log_file = _setup_logging(service_name="auth-server")
logger = logging.getLogger(__name__)
logger.info(f"Auth-server logging configured. Writing to file: {_auth_log_file}")

# Import JWT constants from shared internal auth module
from registry.auth.internal import (
    _INTERNAL_JWT_AUDIENCE as JWT_AUDIENCE,
)
from registry.auth.internal import (
    _INTERNAL_JWT_ISSUER as JWT_ISSUER,
)
from registry.auth.internal import validate_internal_auth

MAX_TOKEN_LIFETIME_HOURS = 24
DEFAULT_TOKEN_LIFETIME_HOURS = 8

# Rate limiting for token generation (simple in-memory counter)
user_token_generation_counts = {}
MAX_TOKENS_PER_USER_PER_HOUR = int(os.environ.get("MAX_TOKENS_PER_USER_PER_HOUR", "100"))


# MCP tools/list filter configuration (Issue #1026).
#
# Prefer the canonical registry.core.config.settings fields when they
# exist; fall back to env vars so this module stays importable during
# the rollout window in which the Settings class may not yet carry the
# new fields (parallel agent work).
def _endpoint_is_localhost(url: str) -> bool:
    """Return True if the URL host is 127.0.0.1, ::1, or localhost."""
    from urllib.parse import urlsplit

    try:
        host = urlsplit(url).hostname or ""
    except ValueError:
        return False
    return host in {"localhost", "127.0.0.1", "::1"}


def _log_otel_state() -> None:
    """Emit a single startup log line describing OTel SDK + legacy-flag state.

    Issue #1122: lets operators see at a glance whether OTel emission is
    active, which OTLP endpoint is configured, the export interval, and
    whether the legacy HTTP POST path is also enabled. Also warns if the
    OTLP endpoint uses HTTP to a non-localhost host.
    """
    from opentelemetry import metrics

    provider = metrics.get_meter_provider()
    provider_name = type(provider).__name__
    otlp_endpoint = os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT", "")
    legacy = os.getenv("METRICS_LEGACY_HTTP_POST", "false").lower() == "true"
    interval_ms = os.getenv("OTEL_METRIC_EXPORT_INTERVAL_MS", "15000")

    if "NoOp" in provider_name or "Default" in provider_name or "Proxy" in provider_name:
        logger.warning(
            "OTel metrics DISABLED (provider=%s). Set OTEL_EXPORTER_OTLP_ENDPOINT "
            "or OTEL_EXPORTER_PROMETHEUS_HOST to enable. Legacy HTTP POST: %s.",
            provider_name,
            legacy,
        )
        return

    logger.info(
        "OTel metrics enabled (provider=%s, endpoint=%s, interval_ms=%s, legacy_http_post=%s)",
        provider_name,
        otlp_endpoint,
        interval_ms,
        legacy,
    )

    if otlp_endpoint.startswith("http://") and not _endpoint_is_localhost(otlp_endpoint):
        logger.warning(
            "OTEL_EXPORTER_OTLP_ENDPOINT uses http:// to a non-localhost host (%s). "
            "Telemetry will be UNENCRYPTED in transit. Use https:// in production.",
            otlp_endpoint,
        )


def _read_mcp_filter_enabled() -> bool:
    try:
        value = getattr(settings, "mcp_tools_list_filter_enabled", None)
        if value is not None:
            return bool(value)
    except Exception:
        pass
    raw = os.getenv("MCP_TOOLS_LIST_FILTER_ENABLED", "true").lower()
    return raw in ("true", "1", "yes")


# The ONE canonical per-user egress bucket. The browser consent path keys the
# vault on this value (the registry's nginx_proxied_auth enforces the cookie
# session's auth_method == "oauth2"), so vend-read MUST resolve to the SAME value
# for one human identity or the user loops on consent forever.
_EGRESS_CANONICAL_PER_USER_METHOD: str = "oauth2"

# Validator ``method`` values that all denote "a real human authenticated via the
# per-user IdP" -- they are different TOKEN FORMATS / provider names for the same
# kind of principal. A user may consent via a cookie session (reported as
# ``session_cookie`` -> inner ``oauth2``) and later vend with a Keycloak-issued
# bearer obtained via Dynamic Client Registration (reported as ``keycloak``), or
# a gateway-minted JWT (``self_signed``). All must canonicalize to the SAME vault
# bucket. NON-per-user methods (``federation-static``, ``network-trusted``) are
# deliberately absent so they pass through raw and are rejected by the vend's
# is_per_user_auth_method() check.
_PER_USER_IDP_METHODS: frozenset[str] = frozenset(
    {
        "oauth2",
        "session_cookie",
        "self_signed",
        "jwt",
        "boto3",
        "keycloak",
        "entra",
        "cognito",
        "okta",
        "auth0",
        "pingfederate",
    }
)


def _canonical_auth_method(validation_result: dict) -> str:
    """The ONE canonical egress principal method, stamped into both internal tokens.

    The per-user egress vault keys on this value, so consent-write and vend-read
    MUST agree on the same bucket for one human identity -- otherwise the user
    loops on consent forever. Different token *formats* / IdP provider names
    represent the SAME kind of per-user principal and must therefore all
    canonicalize to the single per-user bucket (``oauth2``), NOT the raw method
    string:

    - ``session_cookie``: the browser cookie session; the registry enforces its
      inner ``auth_method == "oauth2"``. This is the CONSENT-WRITE side, so it
      defines the canonical bucket value.
    - ``self_signed``: a JWT this gateway minted (UI "generate token", or the
      egress OAuth-facade ``/token`` mint).
    - ``keycloak``/``entra``/``cognito``/``okta``/``auth0``/``pingfederate``: a
      bearer issued directly by the per-user IdP -- notably what a Dynamic Client
      Registration (DCR) client (Claude Code, Codex) presents. Without folding
      these into ``oauth2`` the DCR vend keys on bucket ``keycloak`` while consent
      wrote to ``oauth2`` -> permanent miss -> the DCR consent loop. (This was the
      live DCR failure.)
    - ``jwt``/``boto3``: other per-user token formats.

    NON-per-user methods (``federation-static``, ``network-trusted``) and unknown
    methods pass through unchanged so the vend's per-user check still rejects them.
    Mirrors ``registry.egress_auth.service.canonical_auth_method``.
    """
    method = validation_result.get("method") or ""
    if method in _PER_USER_IDP_METHODS:
        return _EGRESS_CANONICAL_PER_USER_METHOD
    return method


def _attach_mcp_proxy_token(
    request: "Request",
    response: "JSONResponse",
    subject: str,
    scopes: list[str],
    server_name: str,
    auth_method: str = "",
) -> None:
    """Mint and attach the X-Internal-Token for the /mcp-proxy hop.

    Only mints when nginx forwarded a resolved upstream (``X-Resolved-Upstream``)
    into this /validate subrequest -- i.e. this validation backs an MCP-proxy
    request, not a UI/API location. The token binds the resolved upstream so
    mcp_proxy can ignore the forgeable inbound headers. If minting fails (e.g.
    empty subject), no token is attached: mcp_proxy then rejects (fail-closed)
    rather than trusting unsigned headers.

    ``auth_method`` is the canonical egress principal method; pass
    ``_canonical_auth_method(validation_result)`` at the call sites, NOT the raw
    ``validation_result["method"]``.

    When ``AUTH_SERVER_NGINX_MARKER_SECRET`` is configured, the token is
    minted ONLY if nginx force-set the matching ``X-Validate-Source-Secret`` on
    this subrequest. An empty marker mints unconditionally; this is rejected at
    startup when egress is enabled (see Settings._validate_egress_auth_config),
    so the empty-marker branch only remains reachable when egress is disabled.
    """
    resolved_upstream = request.headers.get("X-Resolved-Upstream", "")
    if not resolved_upstream:
        return

    marker = settings.auth_server_nginx_marker_secret
    if marker and not secrets.compare_digest(
        request.headers.get("X-Validate-Source-Secret", ""), marker
    ):
        logger.warning(
            "/validate: X-Resolved-Upstream present but nginx marker missing/mismatched; "
            "refusing to mint mcp-proxy token (possible direct-:8888 bypass)"
        )
        return
    try:
        response.headers["X-Internal-Token"] = mint_mcp_proxy_token(
            subject=subject,
            scopes=scopes,
            server_name=server_name,
            upstream_url=resolved_upstream,
            auth_method=auth_method,
        )
    except ValueError as exc:
        logger.error(f"/validate: could not mint mcp-proxy token: {exc}")


def _attach_registry_ui_token(
    request: "Request",
    response: "JSONResponse",
    subject: str,
    session_id: str,
    groups: list[str],
    auth_method: str,
    client_id: str,
) -> None:
    """Mint and attach the X-Internal-Token-Registry for the registry /api/ hop.

    Only mints when nginx forwarded the registry-API marker (``X-Registry-Api-Auth``)
    into this /validate subrequest -- i.e. this validation backs a registry /api/
    request, not an MCP-proxy or other location. Carries its own header
    (X-Internal-Token-Registry) distinct from the /mcp-proxy hop's X-Internal-Token,
    so the two never collide on a shared /validate response. The token is a thin
    identity assertion (sub/session_id/groups/auth_method/client_id); the registry
    resolves entitlements server-side. If minting fails (e.g. empty subject), no
    token is attached: the registry then rejects (fail-closed) rather than trusting
    unsigned headers.
    """
    if not request.headers.get("X-Registry-Api-Auth"):
        return
    try:
        response.headers["X-Internal-Token-Registry"] = mint_registry_ui_token(
            subject=subject,
            session_id=session_id,
            groups=groups,
            auth_method=auth_method,
            client_id=client_id,
        )
    except ValueError as exc:
        logger.error(f"/validate: could not mint registry-ui token: {exc}")


def _read_mcp_proxy_max_body_bytes() -> int:
    default_bytes = 2 * 1024 * 1024
    minimum_bytes = 1024
    try:
        value = getattr(settings, "mcp_proxy_max_body_bytes", None)
        if value is not None:
            candidate = int(value)
            return max(candidate, minimum_bytes)
    except (TypeError, ValueError) as e:
        logger.debug(f"settings.mcp_proxy_max_body_bytes parse failed, falling back to env: {e}")
    raw = os.getenv("MCP_PROXY_MAX_BODY_BYTES")
    if not raw:
        return default_bytes
    try:
        candidate = int(raw)
    except ValueError:
        logger.warning(f"Invalid MCP_PROXY_MAX_BODY_BYTES={raw!r}; using default {default_bytes}")
        return default_bytes
    return max(candidate, minimum_bytes)


def _read_mcp_proxy_timeout() -> float:
    """Read the upstream MCP proxy timeout in seconds.

    Resolution order:
    1. ``settings.mcp_proxy_timeout`` (registry Settings field / MCP_PROXY_TIMEOUT)
    2. ``MCP_PROXY_TIMEOUT`` environment variable (fallback when settings unset)
    3. Default: 30.0 seconds

    The minimum is 1 second to prevent accidental zero/negative values.
    """
    default_timeout = 30.0
    minimum_timeout = 1.0
    try:
        value = getattr(settings, "mcp_proxy_timeout", None)
        if value is not None:
            candidate = float(value)
            return max(candidate, minimum_timeout)
    except (TypeError, ValueError) as e:
        logger.debug(f"settings.mcp_proxy_timeout parse failed, falling back to env: {e}")
    raw = os.getenv("MCP_PROXY_TIMEOUT")
    if not raw:
        return default_timeout
    try:
        candidate = float(raw)
    except ValueError:
        logger.warning(f"Invalid MCP_PROXY_TIMEOUT={raw!r}; using default {default_timeout}")
        return default_timeout
    return max(candidate, minimum_timeout)


# Global scopes configuration (will be loaded during FastAPI startup)
SCOPES_CONFIG = {}


def _log_scopes_loaded(scopes_config: dict) -> None:
    """Log the loaded scopes config, loudly when the collection is empty.

    An empty scopes collection means every authenticated user falls back to
    read-only access regardless of group membership. Scopes are not auto-seeded,
    so an empty config is almost always a skipped post-deployment step. Emit a
    WARNING with actionable remediation in that case (issue #1248); otherwise
    keep the existing INFO line unchanged.
    """
    group_mappings = scopes_config.get("group_mappings", {})
    if not group_mappings:
        logger.warning(
            "Loaded scopes configuration on startup with 0 group mappings. "
            "The scopes collection is EMPTY — all users will be read-only. "
            "Seed scopes via the post-deployment init (run-documentdb-init.sh) "
            "or load-scopes.py."
        )
    else:
        logger.info(
            f"Loaded scopes configuration on startup with {len(group_mappings)} group mappings"
        )


# Static token auth: use static API key instead of IdP JWT for Registry API
_registry_static_token_requested: bool = (
    os.environ.get("REGISTRY_STATIC_TOKEN_AUTH_ENABLED", "false").lower() == "true"
)

# Static API key for Registry API (must match Bearer token value when enabled)
REGISTRY_API_TOKEN: str = os.environ.get("REGISTRY_API_TOKEN", "")

# Issue #779: multiple static API keys with per-key groups.
_REGISTRY_API_KEYS_RAW: str = os.environ.get("REGISTRY_API_KEYS", "").strip()

# Issue #1127: user-to-group fallback via DocumentDB.
# When a user JWT validated by one of these providers carries an empty
# groups claim, look the user up in the idp_user_groups collection and use
# the groups stored there. CSV, case-insensitive; default covers
# PingFederate which does not always emit a groups claim.
_IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS_RAW: str = os.environ.get(
    "IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS", "pingfederate"
)
IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS: list[str] = [
    p.strip().lower()
    for p in _IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS_RAW.split(",")
    if p.strip()
]

# Validate configuration: static token auth requires at least one token source
if _registry_static_token_requested and not REGISTRY_API_TOKEN and not _REGISTRY_API_KEYS_RAW:
    logging.error(
        "REGISTRY_STATIC_TOKEN_AUTH_ENABLED=true but neither REGISTRY_API_TOKEN "
        "nor REGISTRY_API_KEYS is set. Static token auth is DISABLED. "
        "Set at least one of these or disable the feature. "
        "Falling back to standard IdP JWT validation."
    )
    REGISTRY_STATIC_TOKEN_AUTH_ENABLED: bool = False
else:
    REGISTRY_STATIC_TOKEN_AUTH_ENABLED: bool = _registry_static_token_requested


# ---------------------------------------------------------------------------
# Multi-key static token config model and parser (Issue #779)
# ---------------------------------------------------------------------------

_KEY_NAME_PATTERN: re.Pattern = re.compile(r"^[a-z0-9][a-z0-9_-]{0,63}$")

_RESERVED_KEY_NAMES: frozenset = frozenset(
    {
        "legacy",
        "network-user",
        "network-trusted",
    }
)

_STATIC_TOKEN_MAP: dict[str, dict] = {}


class _RegistryApiKeyEntry(BaseModel):
    """Config entry parsed from REGISTRY_API_KEYS."""

    name: str = Field(
        ...,
        description="Key name (log-safe identifier)",
    )
    key: str = Field(
        ...,
        min_length=32,
        description=(
            "The Bearer token value. Minimum 32 chars matches the default "
            "output of python3 -c 'import secrets; print(secrets.token_urlsafe(32))'."
        ),
    )
    groups: list[str] = Field(
        ...,
        min_length=1,
        description="Groups this key is mapped to",
    )

    @field_validator("name")
    @classmethod
    def _validate_name(
        cls,
        v: str,
    ) -> str:
        if not _KEY_NAME_PATTERN.match(v):
            raise ValueError(f"Invalid key name '{v}': must match ^[a-z0-9][a-z0-9_-]{{0,63}}$")
        if v in _RESERVED_KEY_NAMES:
            raise ValueError(
                f"Key name '{v}' is reserved (legacy/internal). Pick a different name."
            )
        return v


def _repair_stripped_json(
    raw: str,
) -> str:
    """Re-quote a JSON-like string where docker-compose stripped double quotes.

    Converts e.g. {name:{key:val,groups:[g1]}} back to valid JSON by adding
    double quotes around all bare identifiers and values.
    """
    result = []
    i = 0
    while i < len(raw):
        ch = raw[i]
        if ch in "{}[],:":
            result.append(ch)
            i += 1
        elif ch in " \t\n\r":
            i += 1
        else:
            # Read a bare token (everything until a structural char)
            j = i
            while j < len(raw) and raw[j] not in "{}[],:":
                j += 1
            token = raw[i:j].strip()
            result.append(f'"{token}"')
            i = j
    return "".join(result)


def _parse_registry_api_keys(
    raw: str,
) -> list[_RegistryApiKeyEntry]:
    """Parse REGISTRY_API_KEYS env var into validated entries.

    Returns:
        List of entries. Empty list if raw is empty.

    Raises:
        ValueError: on malformed JSON, duplicate name, duplicate key value,
            reserved name, or validation failure on any entry.
    """
    if not raw:
        return []

    try:
        doc = json.loads(raw)
    except json.JSONDecodeError:
        # Docker Compose strips double quotes from .env values containing JSON.
        # Attempt to recover by re-quoting bare identifiers:
        #   {name:{key:val,...}} -> {"name":{"key":"val",...}}
        repaired = _repair_stripped_json(raw)
        try:
            doc = json.loads(repaired)
            logging.warning(
                "REGISTRY_API_KEYS was not valid JSON (docker-compose may have "
                "stripped quotes). Auto-repaired successfully."
            )
        except json.JSONDecodeError as e2:
            raise ValueError(f"REGISTRY_API_KEYS is not valid JSON: {e2}") from e2

    if not isinstance(doc, dict):
        raise ValueError("REGISTRY_API_KEYS must be a JSON object")

    entries: list[_RegistryApiKeyEntry] = []
    seen_names: set[str] = set()
    seen_keys: set[str] = set()

    for name, value in doc.items():
        if name in seen_names:
            raise ValueError(f"Duplicate key name in REGISTRY_API_KEYS: {name}")

        if not isinstance(value, dict):
            raise ValueError(f"Entry for '{name}' must be an object")

        try:
            entry = _RegistryApiKeyEntry(name=name, **value)
        except Exception as e:
            raise ValueError(f"Invalid entry '{name}': {e}") from e

        if entry.key in seen_keys:
            raise ValueError(f"Duplicate key value across entries (conflicts around name '{name}')")

        seen_names.add(entry.name)
        seen_keys.add(entry.key)
        entries.append(entry)

    return entries


async def _build_static_token_map() -> None:
    """Build _STATIC_TOKEN_MAP from env config. Fail-closed on any error."""
    global REGISTRY_STATIC_TOKEN_AUTH_ENABLED, _STATIC_TOKEN_MAP

    if not REGISTRY_STATIC_TOKEN_AUTH_ENABLED:
        return

    token_map: dict[str, dict] = {}

    try:
        parsed = _parse_registry_api_keys(_REGISTRY_API_KEYS_RAW)
    except ValueError as e:
        logging.error(
            "Failed to parse REGISTRY_API_KEYS: %s. Static-token auth DISABLED.",
            e,
        )
        REGISTRY_STATIC_TOKEN_AUTH_ENABLED = False
        return

    for entry in parsed:
        scopes = await map_groups_to_scopes(entry.groups)
        if not scopes:
            logging.warning(
                "Static key '%s' has no scope mappings for groups %s. "
                "Requests using this key will get 403 on all protected endpoints.",
                entry.name,
                entry.groups,
            )
        token_map[entry.name] = {
            "key_bytes": entry.key.encode("utf-8"),
            "groups": list(entry.groups),
            "scopes": scopes,
        }

    if REGISTRY_API_TOKEN:
        # Legacy entry uses the well-known admin scopes directly to avoid a DB
        # roundtrip. The list must include the UI scope name "mcp-registry-admin"
        # so the registry resolves admin UI permissions through the standard path
        # (the hard-coded admin branch was removed in #779).
        token_map["legacy"] = {
            "key_bytes": REGISTRY_API_TOKEN.encode("utf-8"),
            "groups": ["mcp-registry-admin"],
            "scopes": [
                "mcp-registry-admin",
                "mcp-servers-unrestricted/read",
                "mcp-servers-unrestricted/execute",
            ],
            "username_override": "network-user",
            "client_id_override": "network-trusted",
        }

    _STATIC_TOKEN_MAP = token_map

    if not _STATIC_TOKEN_MAP:
        logging.warning(
            "Static-token auth ENABLED but no keys loaded. "
            "Check REGISTRY_API_TOKEN / REGISTRY_API_KEYS. "
            "All bearer tokens will fall through to JWT validation."
        )
    else:
        logging.info(
            "Static-token auth: loaded %d key(s): %s",
            len(_STATIC_TOKEN_MAP),
            sorted(_STATIC_TOKEN_MAP.keys()),
        )


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


def _is_redirect_within_cookie_domain(
    url: str,
    cookie_domain: str,
    request: Request | None = None,
) -> bool:
    """Check if an absolute redirect URL is safe to redirect to.

    An absolute URL is safe when either:
      - its (scheme, host) matches the inbound request's origin (same-origin
        is always safe because the browser already trusts that host), or
      - its host is within SESSION_COOKIE_DOMAIN, which defines the
        cross-subdomain trust boundary for deployments that share the session
        cookie across hosts (e.g. ".example.com" covers the apex and any
        subdomain).

    SESSION_COOKIE_DOMAIN is optional: .env.example documents the empty default
    for single-domain deployments, so this helper must not reject same-origin
    redirects when it is unset.
    """
    parsed = urlparse(url)
    if parsed.scheme not in ("http", "https"):
        return False
    hostname = (parsed.hostname or "").lower()

    if request is not None:
        forwarded_proto = request.headers.get("x-forwarded-proto", "").split(",")[0].strip()
        forwarded_host = request.headers.get("x-forwarded-host", "").split(",")[0].strip()
        request_scheme = forwarded_proto or request.url.scheme
        request_host = (forwarded_host or request.url.hostname or "").lower()
        if request_host and parsed.scheme == request_scheme and hostname == request_host:
            return True

    if not cookie_domain:
        return False
    apex = cookie_domain.lstrip(".").lower()
    return hostname == apex or hostname.endswith(f".{apex}")


def _is_safe_redirect_url(
    url: str,
    allowed_hosts: set[str] | None = None,
) -> bool:
    """Validate that a redirect URL is safe (relative or same-origin).

    Prevents open redirect attacks by ensuring the URL is either:
    - A relative path (no scheme or netloc)
    - An absolute URL with an allowed hostname and safe scheme (http/https)

    Args:
        url: The URL to validate.
        allowed_hosts: Set of allowed hostnames. If None, only relative URLs are allowed.

    Returns:
        True if the URL is safe to redirect to, False otherwise.
    """
    if not url:
        return False
    parsed = urlparse(url)
    # Allow relative URLs (no scheme and no netloc)
    if not parsed.scheme and not parsed.netloc:
        return True
    # Block non-http(s) schemes (e.g., javascript:, data:, etc.)
    if parsed.scheme not in ("http", "https"):
        return False
    # If allowed_hosts is provided, check hostname
    if allowed_hosts is not None:
        return parsed.hostname in allowed_hosts
    # No allowed_hosts and URL is absolute — reject by default
    return False


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

    # Resolve all groups to scopes in a single query. Issuing one query per
    # group serialized a DB round-trip per group on every authenticated
    # request, which dominated latency for users with many groups on a remote
    # cluster. get_group_mappings_bulk collapses that into one $in query.
    try:
        scope_repo = get_scope_repository()
        scopes = await scope_repo.get_group_mappings_bulk(groups)
        logger.debug(f"Mapped {len(groups)} groups to scopes: {scopes}")
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
        # Decrypt cookie (max_age=28800 for 8 hours). The cookie now carries
        # only an opaque session_id; the full record lives server-side.
        payload = signer.loads(cookie_value, max_age=28800)

        # Reject legacy dict-payload cookies (forces re-login post-rollout).
        if not isinstance(payload, str):
            raise ValueError("Legacy session cookie format; please re-login")

        from session_store import resolve_session

        session_data = await resolve_session(payload)
        if not session_data or not session_data.get("username"):
            raise ValueError("Session not found or expired")

        username = session_data["username"]
        groups = session_data.get("groups", [])
        scopes = await map_groups_to_scopes(groups)

        logger.info(f"Session cookie validated for user: {hash_username(username)}")

        return {
            "valid": True,
            "username": username,
            "scopes": scopes,
            "method": "session_cookie",
            "groups": groups,
            "client_id": "",  # Not applicable for session
            "data": session_data,
        }
    except SignatureExpired:
        logger.warning("Session cookie has expired")
        raise ValueError("Session cookie has expired")
    except BadSignature:
        logger.warning("Invalid session cookie signature")
        raise ValueError("Invalid session cookie")
    except ValueError:
        raise
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


async def filter_tools_list_response(
    server_name: str,
    user_scopes: list[str],
    tools_list: list[dict],
) -> list[dict]:
    """
    Filter an upstream tools/list result array down to the tools the
    caller is allowed to see.

    For each tool, delegates to validate_server_tool_access using method
    "tools/call" so the allowlist source of truth matches what actually
    happens when the tool is invoked. Admin / wildcard handling is
    inherited from validate_server_tool_access (server: "*" or "all", or
    tools: ["*"] / ["all"]).

    Args:
        server_name: Name of the MCP server whose tools/list is being filtered.
        user_scopes: Scopes resolved for the caller.
        tools_list: The raw tools array from the upstream JSON-RPC result.

    Returns:
        A new list containing only the tool dicts the caller is allowed
        to see. Malformed entries (non-dict, missing "name") are dropped
        silently. Never raises.
    """
    before_count = len(tools_list) if isinstance(tools_list, list) else 0
    kept: list[dict] = []

    if not isinstance(tools_list, list):
        logger.info(f"filter_tools_list_response: server={server_name} before=0 after=0")
        return kept

    for tool in tools_list:
        if not isinstance(tool, dict):
            continue
        tool_name = tool.get("name")
        if not tool_name or not isinstance(tool_name, str):
            continue
        try:
            allowed = await validate_server_tool_access(
                server_name,
                "tools/call",
                tool_name,
                user_scopes,
            )
        except Exception as exc:
            # Fail closed on unexpected errors; drop the tool silently.
            logger.error(
                f"filter_tools_list_response: validation error for "
                f"server={server_name} tool={tool_name}: {exc}"
            )
            continue
        if allowed:
            kept.append(tool)

    after_count = len(kept)
    logger.info(
        f"filter_tools_list_response: server={server_name} "
        f"before={before_count} after={after_count}"
    )
    return kept


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
    # Log OTel SDK + metrics emission state (issue #1122)
    _log_otel_state()

    # Startup: Load scopes configuration
    global SCOPES_CONFIG
    try:
        SCOPES_CONFIG = await reload_scopes_config()
        _log_scopes_loaded(SCOPES_CONFIG)
    except Exception as e:
        logger.error(f"Failed to load scopes configuration on startup: {e}", exc_info=True)
        # Fall back to empty config
        SCOPES_CONFIG = {"group_mappings": {}}

    # Build multi-key static token map (Issue #779).
    # Runs after scopes are loaded so map_groups_to_scopes can resolve groups.
    await _build_static_token_map()

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

# Issue #1122: programmatic FastAPI auto-instrumentation (HTTP semantic
# conventions). See registry/main.py for the full rationale. Skipped when
# the opentelemetry-instrument wrapper has already instrumented the app to
# avoid the "already instrumented" warning and any double-instrumentation
# side effects observed on ECS.
try:
    from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

    if getattr(app, "_is_instrumented_by_opentelemetry", False):
        logger.info(
            "FastAPI already instrumented by opentelemetry-instrument; skipping programmatic instrument_app"
        )
    else:
        FastAPIInstrumentor.instrument_app(app)
        logger.info("Programmatic FastAPI auto-instrumentation enabled (issue #1122)")
except ImportError:
    logger.debug("opentelemetry-instrumentation-fastapi not installed; HTTP auto-metrics disabled")
except Exception as exc:
    logger.warning("FastAPI auto-instrumentation failed: %s", exc)


# Router for service-to-service /internal/* endpoints.
#
# Every route registered on this router inherits the
# ``validate_internal_auth`` dependency, so no /internal/* handler can
# accidentally ship without the internal-JWT gate — a new handler just
# needs ``@internal_router.post("/foo")`` and the signed-Bearer check
# is already in place before the handler body runs.
#
# Individual handlers additionally declare ``caller: str = Depends(
# validate_internal_auth)`` so (a) the caller identity is available for
# audit logging and (b) the Authorization header shows up in OpenAPI.
# The dependency is cached per-request by FastAPI, so declaring it
# twice (router-level gate + handler-level injection) does not
# re-validate the JWT.
internal_router = APIRouter(
    prefix="/internal",
    dependencies=[Depends(validate_internal_auth)],
)


@app.on_event("startup")
async def startup_event():
    """Load scopes configuration on startup."""
    global SCOPES_CONFIG
    try:
        SCOPES_CONFIG = await reload_scopes_config()
        _log_scopes_loaded(SCOPES_CONFIG)
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


# Resource-binding enforcement helpers. Single source of truth
# lives in ``registry.auth.resource_binding``; both the registry's pre-mint
# check and the auth-server's /validate guard call the same functions so
# they can never disagree on what is blocked or how a URL classifies.
from registry.auth.resource_binding import (
    RESOURCE_ID_CLAIM,
    RESOURCE_TYPE_CLAIM,
    RESOURCE_TYPES,
    TOKEN_KIND_CLAIM,
    ResourceType,
    TokenKind,
    check_resource_token_allowed,
    classify_request_url,
    is_resource_token_introspection_path,
    normalize_resource_id,
)

# String identifier used in ``validation_result['method']`` for tokens
# minted and verified by this server (as opposed to external IdP tokens
# like Cognito/Keycloak). Used by the resource-binding enforcement block
# to scope the legacy-token deprecation warning to our own tokens only.
# Centralized so a future rename of the value cannot silently break the
# warning gating.
AUTH_METHOD_SELF_SIGNED: str = "self_signed"  # nosec B105 - identifier, not a credential


class ResourceBinding(BaseModel):
    """Identifies a single resource that a JWT should be bound to.

    A resource-bound token is scoped to exactly one (type, id) pair. The auth
    server refuses to mint a binding for a resource the user cannot access,
    and the edge guards refuse requests where the token's binding does not
    match the resource being accessed.
    """

    type: ResourceType = Field(
        ..., description="Resource type. One of: " + ", ".join(RESOURCE_TYPES)
    )
    # StrictStr refuses non-string input (int, bool, list) rather than
    # Pydantic v2's default lax coercion, which would silently turn
    # ``{"id": 123}`` into ``"123"``. Resource ids are compared
    # byte-for-byte against URL paths at the edge; a coerced numeric id
    # would silently mint a token nobody can actually use.
    id: StrictStr = Field(..., min_length=1, description="Resource identifier (path or slug)")

    @field_validator("id")
    @classmethod
    def _normalize_id(cls, v: str) -> str:
        # Reject path traversal, URL-encoded payloads, and control
        # characters (null byte, CR, LF, tabs, etc.) before any
        # normalization. Control characters in particular can cause
        # truncation in C-backed URL parsers downstream and should never
        # appear in a legitimate resource id.
        if ".." in v or "%" in v:
            raise ValueError("Resource id must not contain '..' or percent-encoded characters")
        if any(ord(c) < 0x20 or ord(c) == 0x7F for c in v):
            raise ValueError("Resource id must not contain control characters")
        stripped = v.strip()
        if not stripped:
            raise ValueError("Resource id cannot be empty")
        # Delegate to the shared normalizer so mint-time and compare-time
        # canonicalize identically.
        return normalize_resource_id(stripped)


class GenerateTokenRequest(BaseModel):
    """Request model for token generation"""

    user_context: dict[str, Any]
    requested_scopes: list[str] = []
    expires_in_hours: int = DEFAULT_TOKEN_LIFETIME_HOURS
    description: str | None = None
    resource: ResourceBinding | None = None
    correlation_id: str | None = None


class GenerateTokenResponse(BaseModel):
    """Response model for token generation"""

    access_token: str
    refresh_token: str | None = None
    token_type: str = "Bearer"  # nosec B105 - OAuth2 standard token type per RFC 6750
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
            if token_use != "access":  # nosec B105 - OAuth2 token type validation per RFC 6749, not a password
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
                "method": AUTH_METHOD_SELF_SIGNED,
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
        except Exception as e:
            # Not our token or malformed, continue to Cognito validation
            logger.debug(f"Token is not self-signed or malformed, falling back to Cognito: {e}")

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


def _check_registry_static_token(
    bearer_token: str,
) -> dict | None:
    """Return the identity payload if the bearer matches a configured static
    key, else None.

    Each pair-wise comparison uses hmac.compare_digest so individual
    comparisons are constant-time. We iterate all configured entries without
    early return as belt-and-braces so total comparison time is independent
    of which entry (if any) matched. With small N this matters less than the
    per-comparison guarantee, but costs almost nothing.

    For the legacy REGISTRY_API_TOKEN entry (map key "legacy"), the returned
    username and client_id are overridden to "network-user" /
    "network-trusted" to preserve back-compat with pre-#779 audit log
    consumers.

    See issue #779.
    """
    bearer_bytes = bearer_token.encode("utf-8")
    matched_entry: dict | None = None
    matched_name: str | None = None

    for name, entry in _STATIC_TOKEN_MAP.items():
        if hmac.compare_digest(bearer_bytes, entry["key_bytes"]):
            if matched_entry is None:
                matched_entry = entry
                matched_name = name

    if matched_entry is None:
        return None

    username = matched_entry.get("username_override", matched_name)
    client_id = matched_entry.get("client_id_override", matched_name)

    return {
        "username": username,
        "client_id": client_id,
        "groups": list(matched_entry["groups"]),
        "scopes": list(matched_entry["scopes"]),
    }


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

                logger.debug(
                    "Extracted server_name '%s' and endpoint '%s' from original_url: %s",
                    server_name_from_url,
                    endpoint_from_url,
                    original_url,
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
                logger.debug(
                    "Raw Request Payload (%d chars): %s...",
                    len(payload_text),
                    payload_text[:1000],
                )
                request_payload = json.loads(payload_text)
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug("JSON RPC Request Payload: %s", json.dumps(request_payload))
            else:
                logger.debug("No request body provided, skipping payload parsing")
        except UnicodeDecodeError as e:
            logger.warning(f"Could not decode body as UTF-8: {e}")
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse JSON RPC payload: {e}")
        except Exception as e:
            logger.error(f"Error reading request payload: {type(e).__name__}: {e}")

        # Log request for debugging with anonymized IP
        client_ip = get_client_ip(request)
        logger.debug("Validation request from %s", anonymize_ip(client_ip))
        logger.debug("Request Method: %s", request.method)

        # Log masked HTTP headers for GDPR/SOX compliance
        all_headers = dict(request.headers)
        masked_headers = mask_headers(all_headers)
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug("HTTP Headers (masked): %s", json.dumps(masked_headers))

        # Log specific headers for debugging with masked sensitive data
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Key Headers: Authorization=%s, Cookie=%s, "
                "User-Pool-Id=%s, Client-Id=%s, Region=%s, Original-URL=%s",
                bool(authorization),
                bool(cookie_header),
                mask_sensitive_id(user_pool_id) if user_pool_id else "None",
                mask_sensitive_id(client_id) if client_id else "None",
                region,
                original_url,
            )

        logger.debug("Server Name from URL: %s", server_name_from_url)

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

                _attach_mcp_proxy_token(
                    request,
                    response,
                    subject="federation-peer",
                    scopes=federation_scopes,
                    server_name="",
                    auth_method="federation-static",
                )
                # Federation peers have no session row; the registry resolves
                # nothing server-side (no groups), and _derive_user_context
                # short-circuits auth_method == "federation-static" to a
                # no-access context. Minting here keeps /api/federation reachable.
                _attach_registry_ui_token(
                    request,
                    response,
                    subject="federation-peer",
                    session_id="",
                    groups=[],
                    auth_method="federation-static",
                    client_id="federation-static",
                )

                return response

            # If federation token didn't match, DON'T return 403 here.
            # Fall through to the admin static token check below (if enabled).
            # If admin token also doesn't match, that block will return 403.
            # If admin token is NOT enabled, fall through to JWT validation.

        # Static token auth: accept REGISTRY_API_TOKEN as an ADDITIONAL accepted
        # credential on Registry API paths. A missing or mismatched bearer falls
        # through to JWT/session validation so Okta tokens and UI-issued self-
        # signed JWTs remain accepted. See issue #871.
        #
        # Extension point for #779 (multi-key static tokens) is the helper
        # _check_registry_static_token; the control flow here does not change.
        if (
            REGISTRY_STATIC_TOKEN_AUTH_ENABLED
            and _is_registry_api_request(original_url)
            and not has_session_cookie
        ):
            if authorization and authorization.startswith("Bearer "):
                bearer_token = authorization[len("Bearer ") :].strip()
                identity = _check_registry_static_token(bearer_token)
                if identity is not None:
                    logger.info(
                        "Network-trusted mode: key='%s' for %s",
                        identity["username"],
                        original_url,
                    )

                    response_data = {
                        "valid": True,
                        "username": identity["username"],
                        "client_id": identity["client_id"],
                        "scopes": identity["scopes"],
                        "method": "network-trusted",
                        "groups": identity["groups"],
                        "server_name": None,
                        "tool_name": None,
                    }

                    response = JSONResponse(content=response_data, status_code=200)
                    response.headers["X-User"] = identity["username"]
                    response.headers["X-Username"] = identity["username"]
                    response.headers["X-Client-Id"] = identity["client_id"]
                    response.headers["X-Scopes"] = " ".join(identity["scopes"])
                    response.headers["X-Auth-Method"] = "network-trusted"
                    response.headers["X-Server-Name"] = ""
                    response.headers["X-Tool-Name"] = ""

                    _attach_mcp_proxy_token(
                        request,
                        response,
                        subject=identity["username"],
                        scopes=identity["scopes"],
                        server_name="",
                        auth_method="network-trusted",
                    )
                    # Network-trusted static-token callers have no session row;
                    # the registry uses the claim's groups directly. Minting here
                    # keeps REGISTRY_API_KEYS access to /api/* working.
                    _attach_registry_ui_token(
                        request,
                        response,
                        subject=identity["username"],
                        session_id="",
                        groups=identity["groups"],
                        auth_method="network-trusted",
                        client_id=identity["client_id"],
                    )

                    return response

                # Bearer present but does not match any static token. Fall
                # through to JWT validation below (Okta RS256 / self-signed
                # HS256). Intentionally does NOT log any portion of the bearer.
                logger.debug("Static token mismatch; falling through to JWT validation")
            else:
                # No Authorization header or non-Bearer scheme. Fall through to
                # session/JWT validation, which returns 401 if nothing matches.
                logger.debug(
                    "Registry API request without Bearer credential; "
                    "falling through to session/JWT validation"
                )

        # Initialize validation result
        validation_result = None

        # Log which cookie names are present on the inbound /validate request.
        # Distinguishes "browser didn't send mcp_gateway_session" (cookie scope
        # / jar collision) from "nginx stripped it" (sub-request config drift)
        # when diagnosing spurious 401s.
        if cookie_header:
            cookie_names = sorted(
                {
                    c.split("=", 1)[0].strip()
                    for c in cookie_header.split(";")
                    if c.strip() and "=" in c
                }
            )
            mgs_count = cookie_header.count("mcp_gateway_session=")
            logger.info(
                f"Cookie names present: {cookie_names} "
                f"(mcp_gateway_session occurrences: {mgs_count})"
            )

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
                # Try self-signed token first (tokens minted by this auth server).
                # This must run before provider-specific validation because the
                # Connect button generates locally-signed JWTs with iss=mcp-auth-server
                # regardless of the configured auth provider (Entra, Okta, etc.).
                try:
                    unverified = jwt.decode(access_token, options={"verify_signature": False})
                    if unverified.get("iss") == JWT_ISSUER:
                        validation_result = validator.validate_self_signed_token(access_token)
                        logger.info("Token validated as self-signed (iss=mcp-auth-server)")
                except Exception as e:
                    logger.debug(f"Self-signed check failed, continuing to provider: {e}")

                if not validation_result:
                    auth_provider = get_auth_provider()
                    logger.info(
                        f"Using authentication provider: {auth_provider.__class__.__name__}"
                    )

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

            except ValueError as e:
                # ValueError from a provider's validate_token() indicates the
                # token itself is bad: missing kid, signature mismatch, expired,
                # wrong audience, etc. Per RFC 9728 §5.1 / MCP 2025-06-18, the
                # client must see a 401 with WWW-Authenticate so its discovery
                # flow can kick in. Returning 500 here was a pre-existing bug
                # that prevented Claude Code / Cursor from re-triggering the
                # OAuth dance after a stale token (issue #989).
                logger.warning(f"Token validation failed: {e}")
                raise HTTPException(
                    status_code=401,
                    detail=f"Token validation failed: {e}",
                    headers={"WWW-Authenticate": "Bearer", "Connection": "close"},
                )
            except Exception as e:
                # Unexpected non-validation errors (network failure reaching
                # IdP, provider misconfiguration, etc.) remain 500.
                logger.error(f"Authentication provider error: {e}")
                raise HTTPException(
                    status_code=500,
                    detail="Authentication provider configuration error",
                    headers={"Connection": "close"},
                )

        logger.debug("Token validation successful using method: %s", validation_result["method"])

        # Enrich groups from MongoDB if empty (for M2M clients)
        try:
            from mongodb_groups_enrichment import (
                enrich_groups_from_mongodb,
                should_enrich_groups,
            )

            client_id = validation_result.get("client_id")
            current_groups = validation_result.get("groups", [])
            should_enrich = should_enrich_groups(validation_result)
            logger.info(
                f"Enrichment check: client_id={client_id}, "
                f"groups={current_groups}, should_enrich={should_enrich}"
            )

            if should_enrich:
                enriched_groups = await enrich_groups_from_mongodb(client_id, current_groups)

                if enriched_groups != current_groups:
                    validation_result["groups"] = enriched_groups
                    logger.info(
                        f"Groups enriched from MongoDB for client {client_id}: {enriched_groups}"
                    )
        except Exception as e:
            logger.warning(f"Failed to enrich groups from MongoDB: {e}")
            # Don't fail validation if enrichment fails

        # Issue #1127: enrich user groups from idp_user_groups collection if
        # the validated user token has no groups claim (e.g., PingFederate
        # without the custom ATM groups attribute). Gated per-provider via
        # IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS so unrelated IdPs are not
        # affected.
        try:
            from mongodb_groups_enrichment import (
                enrich_user_groups_from_mongodb,
                should_enrich_user_groups,
            )

            # Read provider and username from the inner `data` dict first.
            # For session_cookie / oauth2 paths the outer `method` is the
            # transport ("session_cookie"), while `data.provider` is the
            # actual IdP ("pingfederate", "keycloak", ...) — that's what we
            # need to match against the enabled-providers allowlist. Same
            # for the username: outer `username` may be a hashed session
            # subject (e.g. "user_8c6976e5"), while `data.username` is the
            # IdP-side login id (e.g. "admin"). Fall back to the outer
            # values for direct-token paths where there's no `data`.
            data = validation_result.get("data") or {}
            user_provider = data.get("provider") or validation_result.get("method")
            username_for_lookup = data.get("username") or validation_result.get("username", "")
            current_user_groups = validation_result.get("groups", [])

            if should_enrich_user_groups(
                username_for_lookup,
                current_user_groups,
                user_provider,
                IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS,
            ):
                enriched_user_groups = await enrich_user_groups_from_mongodb(
                    username_for_lookup,
                    current_user_groups,
                    user_provider,
                )

                if enriched_user_groups != current_user_groups:
                    validation_result["groups"] = enriched_user_groups
                    logger.info(
                        "Enriched user '%s' (provider=%s) from idp_user_groups: %s",
                        username_for_lookup,
                        user_provider,
                        enriched_user_groups,
                    )
        except Exception as e:
            logger.warning(f"Failed to enrich user groups from MongoDB: {e}")
            # Don't fail validation if enrichment fails

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
        # For providers that use groups (Keycloak, Entra ID, Cognito, Okta, Auth0), map groups to scopes
        user_groups = validation_result.get("groups", [])
        auth_method = validation_result.get("method", "")
        existing_scopes = validation_result.get("scopes", []) or []
        if user_groups and auth_method in ["keycloak", "entra", "cognito", "okta", "auth0"]:
            # Map IdP groups to scopes using the group mappings (query DocumentDB)
            user_scopes = await map_groups_to_scopes(user_groups)
            logger.debug("Mapped %s groups %s to scopes: %s", auth_method, user_groups, user_scopes)
        elif (
            user_groups
            and not existing_scopes
            and (validation_result.get("data") or {}).get("provider") == "pingfederate"
        ):
            # Issue #1127: PingFederate user-group fallback enrichment runs
            # AFTER initial validation, so session_cookie scopes were
            # computed from empty groups. Re-map now using the enriched
            # groups. Gated on (a) groups present from enrichment, (b)
            # initial scope mapping returned empty (so we don't re-run for
            # already-resolved sessions), (c) inner provider is exactly
            # "pingfederate" — keeps Keycloak/Okta/etc. completely
            # unchanged.
            user_scopes = await map_groups_to_scopes(user_groups)
            logger.debug(
                "Re-mapped pingfederate groups %s to scopes: %s "
                "after fallback enrichment (transport=%s)",
                user_groups,
                user_scopes,
                auth_method,
            )
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
            logger.debug(
                "Method determined for validation: '%s' (tool_name=%s, endpoint_from_url=%s)",
                method,
                tool_name,
                endpoint_from_url,
            )
            actual_tool_name = None

            # For tools/call, extract the actual tool name from params
            if method == "tools/call" and isinstance(request_payload, dict):
                params = request_payload.get("params", {})
                if isinstance(params, dict):
                    actual_tool_name = params.get("name")
                    logger.debug(
                        "Extracted actual tool name for tools/call: '%s'", actual_tool_name
                    )

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
            logger.debug(
                "Scope validation passed for %s.%s (tool: %s)",
                server_name,
                method,
                actual_tool_name,
            )
        else:
            logger.debug("No server information available, skipping scope validation")

        # --- Resource-bound token enforcement ---
        # Resource-bound tokens carry the claims `token_kind: "resource"`,
        # `resource_type`, and `resource_id`. At the edge we:
        #   1. Refuse to let a resource-bound token reach endpoints that
        #      could escalate or bypass the binding
        #      (/api/tokens/generate, /api/admin/*, /api/search/*).
        #   2. Require the claims to match the (resource_type, resource_id)
        #      parsed from the request URL. Mismatch = 403.
        #
        # Every self-signed token this server mints since carries
        # a ``token_kind`` claim. Self-signed tokens WITHOUT the claim are
        # either artifacts that outlived their deployment or forged
        # attempts — neither is acceptable, so we reject hard.
        #
        # External IdP tokens (Cognito, Keycloak, Entra, Okta, Auth0) and
        # session-cookie authentication do NOT produce a self-signed JWT;
        # their ``validation_result["data"]`` does not contain
        # ``token_kind`` by design, and they flow through the
        # ``token_kind is None`` branch as unrestricted user tokens. The
        # rejection therefore gates on ``validation_method`` being the
        # self-signed method specifically — an external-IdP request without
        # a ``token_kind`` claim is the normal, permanent behavior, not a
        # legacy state.
        token_claims = validation_result.get("data") or {}
        token_kind = token_claims.get(TOKEN_KIND_CLAIM)
        validation_method = validation_result.get("method")
        if token_kind is None:
            if validation_method == AUTH_METHOD_SELF_SIGNED:
                logger.warning(
                    "Self-signed token without token_kind claim rejected "
                    f"for user {hash_username(validation_result.get('username') or '')} "
                    "(token_kind is required on every self-signed token)"
                )
                raise HTTPException(
                    status_code=403,
                    detail=(
                        "Token is missing the required token_kind claim. "
                        "Re-issue the token from the current gateway."
                    ),
                    headers={"Connection": "close"},
                )
            # External IdP / session / static / federation: no token_kind
            # and never has been — treat as user token, no warning.
        elif token_kind == TokenKind.USER:
            # Explicit no-op branch for clarity: user-kind tokens have no
            # resource binding and reach the same enforcement path as a
            # legacy/external-IdP token.
            pass
        elif token_kind == TokenKind.RESOURCE:
            claim_type = token_claims.get(RESOURCE_TYPE_CLAIM)
            claim_id = token_claims.get(RESOURCE_ID_CLAIM)
            # Strip whitespace before the truthy check so a whitespace-only
            # claim (e.g. ``resource_id: "   "``) is reported as "missing
            # required claims" rather than giving a misleading
            # "does not permit this request" error downstream.
            if (
                not isinstance(claim_type, str)
                or not claim_type.strip()
                or not isinstance(claim_id, str)
                or not claim_id.strip()
            ):
                logger.warning(
                    f"Resource token for {hash_username(validation_result.get('username') or '')} "
                    "missing resource_type or resource_id claim"
                )
                raise HTTPException(
                    status_code=403,
                    detail="Resource-bound token is missing required claims",
                    headers={"Connection": "close"},
                )

            # Fail-closed if the edge did not forward the original URL.
            # Without it, we cannot determine which resource was requested,
            # so a resource-bound token must not be accepted blindly.
            if not original_url:
                logger.warning(
                    f"Resource token for {hash_username(validation_result.get('username') or '')} "
                    "rejected: X-Original-URL header missing from subrequest"
                )
                raise HTTPException(
                    status_code=403,
                    detail="Resource-bound token cannot be validated: request URL unavailable",
                    headers={"Connection": "close"},
                )

            request_path = urlparse(original_url).path
            if not check_resource_token_allowed(request_path, root_path=REGISTRY_ROOT_PATH):
                logger.warning(
                    f"Resource token for {hash_username(validation_result.get('username') or '')} "
                    f"rejected at blocked endpoint: {original_url}"
                )
                raise HTTPException(
                    status_code=403,
                    detail="Resource-bound tokens cannot access this endpoint",
                    headers={"Connection": "close"},
                )

            # Introspection endpoints on the allow-list (e.g. /api/auth/me)
            # are reachable by every token but do not classify to any
            # (type, id) — they are utility endpoints, not resources.
            # Accept here, before classification, so resource-bound tokens
            # can verify themselves without tripping the "unclassifiable"
            # check below.
            if is_resource_token_introspection_path(request_path, root_path=REGISTRY_ROOT_PATH):
                logger.info(f"Resource-bound token on introspection endpoint: {request_path}")
            else:
                classified = classify_request_url(request_path, root_path=REGISTRY_ROOT_PATH)
                if classified is None:
                    logger.warning(
                        f"Resource token for "
                        f"{hash_username(validation_result.get('username') or '')} "
                        f"could not classify request URL: {original_url}"
                    )
                    raise HTTPException(
                        status_code=403,
                        detail="Resource-bound token cannot be used on this endpoint",
                        headers={"Connection": "close"},
                    )

                req_type, req_id = classified
                # Compare normalized ids so the claim "foo" matches URL
                # "/foo". ResourceType inherits from str so req_type ==
                # claim_type compares string values.
                # Use the same normalizer mint-side uses, so the claim is
                # canonicalized identically on both ends (strips leading
                # AND trailing slashes). Mint-side normalization guarantees
                # the claim is already canonical, but re-normalizing at
                # the edge keeps the two paths symmetric and prevents a
                # future claim that slips through without canonicalization
                # from silently failing comparison.
                normalized_claim_id = normalize_resource_id(claim_id or "")
                if req_type.value != claim_type or req_id != normalized_claim_id:
                    # Keep the specific binding details in the server log
                    # for operational debugging, but return a generic
                    # client-facing error so a leaked token does not
                    # disclose the exact resource it was issued for.
                    logger.warning(
                        "Resource token binding mismatch for user "
                        f"{hash_username(validation_result.get('username') or '')}: "
                        f"claim={claim_type}:{normalized_claim_id}, "
                        f"request={req_type}:{req_id}"
                    )
                    raise HTTPException(
                        status_code=403,
                        detail="Resource-bound token does not permit this request",
                        headers={"Connection": "close"},
                    )
                logger.info(f"Resource-bound token match: {claim_type}:{normalized_claim_id}")
        else:
            # Unknown ``token_kind`` value. Only this code is supposed to
            # mint self-signed tokens (and it emits "user" or "resource").
            # Anything else came from a forged claim (the attacker would
            # need SECRET_KEY) or a future feature that has not yet been
            # taught how to enforce. Fail-closed so the failure mode of
            # an unexpected claim is "denied", not "silently admin".
            logger.warning(
                f"Unknown token_kind {token_kind!r} rejected for user "
                f"{hash_username(validation_result.get('username') or '')}"
            )
            raise HTTPException(
                status_code=403,
                detail="Token has an unrecognized token_kind claim",
                headers={"Connection": "close"},
            )

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
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(
                "Full validation result: %s",
                json.dumps(_mask_sensitive_dict(validation_result)),
            )
            logger.debug("Response data being sent: %s", json.dumps(response_data))

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
        response.headers["X-Groups"] = " ".join(validation_result.get("groups", []))

        # Canonical egress principal method: cookie callers resolve to
        # "oauth2" (the session record's value), not the literal "session_cookie".
        # Both internal tokens stamp THIS so consent-write and vend-read agree.
        _canon_auth_method = _canonical_auth_method(validation_result)

        _attach_mcp_proxy_token(
            request,
            response,
            subject=validation_result.get("username") or "",
            scopes=user_scopes,
            server_name=server_name or "",
            auth_method=_canon_auth_method,
        )

        # Registry /api/ hop token. Discriminate cookie vs JWT-bearer: the cookie
        # sub-path has an opaque server-side session_id (the registry resolves live
        # groups from the session store); the JWT-bearer sub-path has no session
        # row, so session_id is empty and the registry uses the groups claim.
        _registry_session_id = ""
        if validation_result.get("method") == "session_cookie":
            _registry_session_id = (validation_result.get("data") or {}).get("session_id") or ""
        _attach_registry_ui_token(
            request,
            response,
            subject=validation_result.get("username") or "",
            session_id=_registry_session_id,
            groups=validation_result.get("groups", []),
            # Canonical: was validation_result["method"] (== "session_cookie"
            # for cookie users) while the registry overrides to "oauth2" -- the two
            # disagreed. Stamp the canonical value so they match.
            auth_method=_canon_auth_method,
            client_id=validation_result.get("client_id") or "",
        )

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
            detail="Internal validation error",
            headers={"Connection": "close"},
        )
    except Exception as e:
        logger.exception("Unexpected error during validation")
        raise HTTPException(
            status_code=500,
            detail="Internal validation error",
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
        logger.exception("Error getting auth config")
        return {
            "auth_type": "unknown",
            "description": "Error getting provider config",
            "error": "Internal server error",
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
        FEDERATION_STATIC_TOKEN = ""  # nosec B105 - Intentional token revocation, clearing the variable
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


async def _emit_token_mint_audit(
    request_id: str,
    correlation_id: str | None,
    username: str,
    auth_method: str,
    provider: str | None,
    internal_caller: str,
    token_kind: str,
    resource_type: str | None,
    resource_id: str | None,
    token_path: str,
    requested_scopes: list[str],
    expires_in_seconds: int | None,
    outcome: str,
    failure_reason: str | None = None,
) -> None:
    """Emit a token-mint audit record and increment the mint metric.

    Best-effort: any failure here is logged and swallowed so token minting is
    never broken by observability.
    """
    try:
        token_mint_total.add(
            1,
            {
                "token_kind": token_kind,
                "resource_type": resource_type or "none",
                "token_path": token_path,
                "outcome": outcome,
            },
        )
    except Exception:
        logger.debug("token_mint metric increment failed", exc_info=True)

    try:
        record = TokenMintAuditRecord(
            request_id=request_id,
            correlation_id=correlation_id,
            username_hash=hash_username(username),
            auth_method=auth_method,
            provider=provider,
            internal_caller=internal_caller,
            token_kind=token_kind,
            resource_type=resource_type,
            resource_id=resource_id,
            token_path=token_path,
            requested_scopes=requested_scopes,
            expires_in_seconds=expires_in_seconds,
            outcome=outcome,
            failure_reason=failure_reason,
        )
        emit_audit_event(record)

        audit_logger = get_audit_logger()
        if audit_logger is not None:
            await audit_logger.log_event(record)
    except Exception:
        logger.warning("Failed to emit token-mint audit record", exc_info=True)


@internal_router.post("/tokens", response_model=GenerateTokenResponse)
async def generate_user_token(
    body: GenerateTokenRequest,
    caller: str = Depends(validate_internal_auth),
):
    """
    Generate or refresh a JWT token for a user.

    This endpoint supports two modes:
    1. If user has stored OAuth tokens (from login), refresh them if needed and return
    2. Otherwise, fall back to generating M2M token using client credentials

    This is an internal API endpoint meant to be called only by the registry service.
    The generated token will have the same or fewer privileges than the user currently has.

    Authentication is enforced at the router level: every route on
    ``internal_router`` requires a Bearer JWT signed with the shared
    ``SECRET_KEY`` (see ``registry.auth.internal.generate_internal_token``).
    The ``caller`` parameter re-declares the same dependency so the
    identity is available for audit logging and the Authorization
    header appears in OpenAPI. FastAPI caches per-request dependencies,
    so the JWT is validated exactly once.

    Args:
        body: Token generation request containing user context and requested scopes
        caller: Identity of the trusted internal service that signed the request

    Returns:
        JWT token with expiration info (either refreshed user token or M2M token)

    Raises:
        HTTPException: 401 if internal auth is missing/invalid; 400/403/429 for
            request validation / permission / rate-limit failures.
    """
    logger.info(f"/internal/tokens call from '{caller}'")

    request = body  # keep the existing variable name used throughout the body
    mint_request_id = str(uuid.uuid4())
    correlation_id = request.correlation_id

    # Initialize audit context up front so the unexpected-error handler can
    # reference these directly instead of introspecting locals(). They are
    # overwritten with the real values once the user context is parsed below.
    username = "unknown"
    auth_method = "unknown"
    provider = None

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
            await _emit_token_mint_audit(
                request_id=mint_request_id,
                correlation_id=correlation_id,
                username=username,
                auth_method=user_context.get("auth_method", "unknown"),
                provider=user_context.get("provider"),
                internal_caller=caller,
                token_kind=(TokenKind.RESOURCE.value if request.resource else TokenKind.USER.value),
                resource_type=(request.resource.type.value if request.resource else None),
                resource_id=(request.resource.id if request.resource else None),
                token_path="unknown",
                requested_scopes=request.requested_scopes,
                expires_in_seconds=None,
                outcome="failure",
                failure_reason="rate_limited",
            )
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

        # For OAuth and network-trusted users, generate a self-signed JWT with their identity and groups
        # This token is issued by our auth server and can be verified using SECRET_KEY
        if auth_method in ("oauth2", "network-trusted"):
            logger.info(
                f"Generating self-signed JWT for {auth_method} user '{hash_username(username)}' "
                f"with groups: {user_groups}"
            )

            current_time = int(time.time())
            # Honour the caller's requested lifetime, clamped to the
            # server-wide maximum (#889).  Values <= 0 or above the cap
            # are silently clamped; omitted values fall back to the
            # default (8 h).
            effective_hours = min(
                max(request.expires_in_hours, 1),
                MAX_TOKEN_LIFETIME_HOURS,
            )
            expires_in = effective_hours * 3600
            if request.expires_in_hours != DEFAULT_TOKEN_LIFETIME_HOURS:
                logger.info(
                    f"Token lifetime: requested={request.expires_in_hours}h, "
                    f"effective={effective_hours}h (max={MAX_TOKEN_LIFETIME_HOURS}h)"
                )

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
                "auth_method": auth_method,
                "provider": provider,
                "iat": current_time,
                "exp": current_time + expires_in,
                "description": request.description,
                TOKEN_KIND_CLAIM: (
                    TokenKind.RESOURCE.value if request.resource else TokenKind.USER.value
                ),
            }

            # For resource-bound tokens, add resource_type and resource_id
            # claims. Authorization that the user can reach this resource is
            # expected to have already been performed by the caller (registry)
            # before this endpoint was invoked (see
            # registry/api/server_routes.py::/tokens/generate).
            if request.resource:
                jwt_claims[RESOURCE_TYPE_CLAIM] = request.resource.type.value
                jwt_claims[RESOURCE_ID_CLAIM] = request.resource.id
                logger.info(
                    f"Minting resource-bound token for user '{hash_username(username)}': "
                    f"type={request.resource.type.value}, id={request.resource.id}"
                )

            # Sign the JWT with our SECRET_KEY
            access_token = jwt.encode(jwt_claims, SECRET_KEY, algorithm="HS256")

            logger.info(
                f"Generated self-signed JWT for user '{hash_username(username)}', "
                f"expires in {expires_in} seconds"
            )

            await _emit_token_mint_audit(
                request_id=mint_request_id,
                correlation_id=correlation_id,
                username=username,
                auth_method=auth_method,
                provider=provider,
                internal_caller=caller,
                token_kind=(TokenKind.RESOURCE.value if request.resource else TokenKind.USER.value),
                resource_type=(request.resource.type.value if request.resource else None),
                resource_id=(request.resource.id if request.resource else None),
                token_path="self_signed",
                requested_scopes=requested_scopes,
                expires_in_seconds=expires_in,
                outcome="success",
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

            await _emit_token_mint_audit(
                request_id=mint_request_id,
                correlation_id=correlation_id,
                username=username,
                auth_method=auth_method or "m2m",
                provider=provider,
                internal_caller=caller,
                token_kind="user",
                resource_type=None,
                resource_id=None,
                token_path="m2m",
                requested_scopes=requested_scopes,
                expires_in_seconds=expires_in,
                outcome="success",
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
            await _emit_token_mint_audit(
                request_id=mint_request_id,
                correlation_id=correlation_id,
                username=username,
                auth_method=auth_method or "m2m",
                provider=provider,
                internal_caller=caller,
                token_kind="user",
                resource_type=None,
                resource_id=None,
                token_path="m2m",
                requested_scopes=requested_scopes,
                expires_in_seconds=None,
                outcome="failure",
                failure_reason="provider_error",
            )
            raise HTTPException(
                status_code=500,
                detail=f"Failed to generate token: {e}",
                headers={"Connection": "close"},
            )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Unexpected error generating token: {e}")
        await _emit_token_mint_audit(
            request_id=mint_request_id,
            correlation_id=correlation_id,
            username=username,
            auth_method=auth_method,
            provider=provider,
            internal_caller=caller,
            token_kind="unknown",
            resource_type=None,
            resource_id=None,
            token_path="unknown",
            requested_scopes=[],
            expires_in_seconds=None,
            outcome="failure",
            failure_reason="unexpected_error",
        )
        raise HTTPException(
            status_code=500,
            detail="Internal error generating token",
            headers={"Connection": "close"},
        )


@internal_router.post("/reload-scopes")
async def reload_scopes(caller_identity: str = Depends(validate_internal_auth)):
    """
    Reload the scopes configuration.

    Authentication is enforced at the router level (see
    ``internal_router``): the caller must present a Bearer JWT signed
    with the shared ``SECRET_KEY``. Re-declaring the dependency here
    surfaces the caller identity for audit logging without re-validating
    the JWT (FastAPI caches per-request dependencies).
    """
    logger.info(f"Reload-scopes authorized via JWT for: {caller_identity}")

    # Reload the scopes configuration
    global SCOPES_CONFIG
    try:
        SCOPES_CONFIG = await reload_scopes_config()
        logger.info(f"Successfully reloaded scopes configuration by '{caller_identity}'")

        # Rebuild static token map so per-key scopes pick up any
        # group-to-scope mapping changes that triggered this reload.
        await _build_static_token_map()

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


# Mount the /internal/* router. All routes registered on
# ``internal_router`` inherit the signed-Bearer authentication
# requirement via its router-level dependency; mounting it here keeps
# the gate co-located with the handlers it protects.
app.include_router(internal_router)


def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Simplified Auth Server")

    parser.add_argument(
        "--host",
        type=str,
        default=os.getenv("AUTH_SERVER_HOST", "127.0.0.1"),  # nosec B104
        help="Host for the server to listen on (default: 127.0.0.1, override with AUTH_SERVER_HOST env var)",
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

# Initialize SECRET_KEY and signer for session management.
# Fail loud: a per-replica random key would silently break sessions across replicas
# (auth_server signs with key A, registry verifies with key B → BadSignature on every request).
SECRET_KEY = os.environ.get("SECRET_KEY")
if not SECRET_KEY:
    raise RuntimeError(
        "SECRET_KEY environment variable is required. "
        "Set it to a value at least 32 bytes long, identical across all auth_server "
        "and registry replicas (see chart values.yaml: global.secretKey)."
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


def get_audit_logger() -> AuditLogger | None:
    """Return the shared AuditLogger, initializing it if needed.

    Reuses the same DocumentDB-backed AuditLogger that MCP access logging uses.
    Returns None when audit logging is disabled or initialization failed.
    """
    get_mcp_logger()
    return _mcp_audit_logger


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
        logger.exception("Error getting OAuth2 providers")
        return {"providers": [], "error": "Internal server error"}


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
        auth_server_external_url = os.environ.get("AUTH_SERVER_EXTERNAL_URL", "").rstrip("/")
        if auth_server_external_url:
            # AUTH_SERVER_EXTERNAL_URL is the complete public base URL and
            # already includes any path prefix (e.g. "/auth-server" in path
            # routing mode); ROOT_PATH is only used in the host-fallback branch.
            auth_server_url = auth_server_external_url
            scheme = "https" if auth_server_external_url.startswith("https") else "http"
            logger.info(f"OAuth2 login - using AUTH_SERVER_EXTERNAL_URL: {auth_server_url}")
        else:
            host = request.headers.get("host", "localhost:8888")
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

        # Validate the OAuth provider auth URL has a safe scheme before redirecting
        parsed_auth_url = urlparse(auth_url)
        if parsed_auth_url.scheme not in ("http", "https"):
            logger.error(
                f"Unsafe OAuth2 auth URL scheme '{parsed_auth_url.scheme}' for provider {provider}"
            )
            raise HTTPException(
                status_code=400,
                detail="Invalid OAuth2 provider configuration",
            )

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
        if not _is_safe_redirect_url(error_url):
            error_url = "/login"
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
            # Validate error_url is a safe redirect target and URL-encode user-supplied error details
            if not _is_safe_redirect_url(error_url):
                error_url = "/login"
            safe_details = urllib.parse.quote(str(error), safe="")
            return RedirectResponse(
                url=f"{error_url}?error=oauth2_error&details={safe_details}", status_code=302
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

                    # Group overage: when a user is a member of more groups
                    # than will fit in the token (Entra caps inline groups
                    # around 150-200), Entra omits them and signals via
                    # `hasgroups` or `_claim_names.groups`. Fall back to
                    # Microsoft Graph /me/memberOf so the user gets their
                    # real group set instead of an empty session (#929).
                    from providers.entra import EntraIdProvider

                    if EntraIdProvider.has_group_overage(id_token_claims):
                        logger.info("Entra ID token signals group overage; resolving via Graph")
                        graph_groups = await EntraIdProvider.fetch_groups_via_graph(
                            token_data["access_token"]
                        )
                        if graph_groups:
                            groups = graph_groups

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
        elif provider == "okta":
            # For Okta, decode the ID token to get groups (userinfo doesn't include groups)
            try:
                if "id_token" in token_data:
                    import jwt

                    id_token_claims = jwt.decode(
                        token_data["id_token"], options={"verify_signature": False}
                    )
                    logger.info(f"Okta ID token claims: {id_token_claims}")

                    mapped_user = {
                        "username": id_token_claims.get("preferred_username")
                        or id_token_claims.get("email")
                        or id_token_claims.get("sub"),
                        "email": id_token_claims.get("email"),
                        "name": id_token_claims.get("name") or id_token_claims.get("given_name"),
                        "groups": id_token_claims.get("groups", []),
                    }
                    logger.info(f"User extracted from Okta ID token: {mapped_user}")
                else:
                    logger.warning("No ID token found in Okta response, falling back to userInfo")
                    raise ValueError("Missing ID token")

            except Exception as e:
                logger.warning(
                    f"Okta ID token parsing failed: {e}, falling back to userInfo endpoint"
                )
                user_info = await get_user_info(token_data["access_token"], provider_config)
                logger.info(f"Raw user info from {provider}: {user_info}")
                mapped_user = map_user_info(user_info, provider_config)
                logger.info(f"Mapped user info from userInfo: {mapped_user}")
        elif provider == "auth0":
            # For Auth0, delegate ID token parsing to the Auth0Provider
            # which validates issuer/audience claims and extracts groups
            # from a custom namespaced claim configured via Auth0 Actions/Rules
            try:
                auth0_provider = get_auth_provider("auth0")
                mapped_user = auth0_provider.extract_user_from_tokens(token_data)
                logger.info(f"User extracted from Auth0 ID token: {mapped_user}")

            except Exception as e:
                logger.warning(
                    f"Auth0 ID token parsing failed: {e}, falling back to userInfo endpoint"
                )
                # Fallback to userInfo endpoint
                user_info = await get_user_info(token_data["access_token"], provider_config)
                logger.info(f"Raw user info from {provider}: {user_info}")
                mapped_user = map_user_info(user_info, provider_config)
                logger.info(f"Mapped user info from userInfo: {mapped_user}")
        elif provider == "pingfederate":
            # For PingFederate, decode the ID token to get groups
            try:
                if "id_token" in token_data:
                    import jwt

                    id_token_claims = jwt.decode(
                        token_data["id_token"], options={"verify_signature": False}
                    )
                    logger.info(f"PingFederate ID token claims: {id_token_claims}")

                    groups_claim_name = os.getenv("PINGFEDERATE_GROUPS_CLAIM", "groups")
                    mapped_user = {
                        "username": id_token_claims.get("preferred_username")
                        or id_token_claims.get("email")
                        or id_token_claims.get("sub"),
                        "email": id_token_claims.get("email"),
                        "name": id_token_claims.get("name") or id_token_claims.get("given_name"),
                        "groups": id_token_claims.get(groups_claim_name, []),
                    }
                    logger.info(f"User extracted from PingFederate ID token: {mapped_user}")
                else:
                    logger.warning(
                        "No ID token found in PingFederate response, falling back to userInfo"
                    )
                    raise ValueError("Missing ID token")

            except Exception as e:
                logger.warning(
                    f"PingFederate ID token parsing failed: {e}, falling back to userInfo endpoint"
                )
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

        # Issue #1127: PingFederate (and any IdP in the fallback allow-list)
        # may return an empty groups claim because group memberships are not
        # configured in the IdP's user store. Enrich groups from the
        # idp_user_groups MongoDB collection BEFORE we persist the session
        # so downstream consumers (`enhanced_auth`, the audit page, etc.)
        # that read groups directly from the session see the right value.
        # The /validate endpoint also runs this enrichment, so existing
        # sessions remain consistent.
        session_groups = mapped_user.get("groups", []) or []
        try:
            from mongodb_groups_enrichment import (
                enrich_user_groups_from_mongodb,
                should_enrich_user_groups,
            )

            if should_enrich_user_groups(
                mapped_user["username"],
                session_groups,
                provider,
                IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS,
            ):
                enriched = await enrich_user_groups_from_mongodb(
                    mapped_user["username"],
                    session_groups,
                    provider,
                )
                if enriched != session_groups:
                    logger.info(
                        "Session groups enriched at OAuth2 callback for user "
                        "'%s' (provider=%s): %s -> %s",
                        mapped_user["username"],
                        provider,
                        session_groups,
                        enriched,
                    )
                    session_groups = enriched
        except Exception as e:
            logger.warning(
                f"Session-time group enrichment failed for "
                f"{mapped_user['username']} (provider={provider}): {e}"
            )

        # Filter the (possibly enriched) group list down to the scope-relevant
        # subset BEFORE persisting it. IdPs such as Entra ID can return hundreds
        # or thousands of groups; storing them all bloats the X-Groups header
        # (nginx buffer overflow -> 500s) and makes the per-request groups->scopes
        # lookup do one DB query per group. Filtering here is lossless for
        # authorization because unmapped groups never produce scopes. Runs after
        # enrichment so the final set is filtered.
        from group_filter import filter_session_groups

        session_groups = await filter_session_groups(
            session_groups,
            username_hash=hash_username(mapped_user["username"]),
        )

        # Persist the full session record server-side and put only an opaque
        # session_id in the browser cookie. This prevents cookie-size failures
        # for IdPs that return large groups claims (e.g. Entra ID with many
        # group memberships) and keeps id_token off the client entirely.
        session_max_age = OAUTH2_CONFIG.get("session", {}).get("max_age_seconds", 28800)
        # id_token is encrypted at rest server-side and required for OIDC SSO
        # logout (id_token_hint).
        from session_store import create_session

        session_id = await create_session(
            username=mapped_user["username"],
            email=mapped_user.get("email"),
            name=mapped_user.get("name"),
            groups=session_groups,
            provider=provider,
            auth_method="oauth2",
            max_age_seconds=session_max_age,
            id_token=token_data.get("id_token"),
        )
        registry_session = signer.dumps(session_id)

        # Redirect to registry with session cookie
        redirect_url = temp_session_data.get(
            "redirect_uri", OAUTH2_CONFIG.get("registry", {}).get("success_redirect", "/")
        )
        # Validate redirect_url to prevent open redirect attacks. Relative URLs
        # are always safe; absolute URLs must be same-origin with the inbound
        # request or within SESSION_COOKIE_DOMAIN when that is configured.
        cookie_domain = os.environ.get("SESSION_COOKIE_DOMAIN", "").strip()
        redirect_parsed = urlparse(redirect_url)
        redirect_is_safe = (
            not redirect_parsed.scheme and not redirect_parsed.netloc
        ) or _is_redirect_within_cookie_domain(redirect_url, cookie_domain, request)
        if not redirect_is_safe:
            logger.warning(f"Blocked unsafe redirect URL: {redirect_url}, falling back to /")
            redirect_url = "/"
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
            logger.info("Using explicitly configured cookie domain")

        logger.info(
            f"Auth server setting session cookie: is_https={is_https}, domain={'configured' if cookie_domain else 'not set'}, x-forwarded-proto={request.headers.get('x-forwarded-proto', 'not set')}, request_scheme={request.url.scheme}"
        )

        cookie_params = {
            "key": "mcp_gateway_session",  # Same as registry SESSION_COOKIE_NAME
            "value": registry_session,
            "max_age": session_max_age,
            "httponly": OAUTH2_CONFIG.get("session", {}).get("httponly", True),
            "samesite": cookie_samesite,
            "secure": cookie_secure,
            "path": "/",  # Ensure cookie is sent for all paths
        }

        # Only set domain if configured or inferred (for cross-subdomain cookies)
        if cookie_domain:
            cookie_params["domain"] = cookie_domain

        response.set_cookie(**cookie_params)

        # Clear temporary OAuth2 session. The cookie was set without a
        # domain attribute, so delete must also omit domain to match.
        response.delete_cookie("oauth2_temp_session", path="/")

        logger.info(
            f"Successfully authenticated user {hash_username(mapped_user['username'])} via {provider}"
        )
        return response

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in OAuth2 callback for {provider}: {e}")
        error_url = OAUTH2_CONFIG.get("registry", {}).get("error_redirect", "/login")
        if not _is_safe_redirect_url(error_url):
            error_url = "/login"
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
    logger.info(f"Looking for groups claim (configured={'yes' if groups_claim else 'no'})")
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
async def oauth2_logout(
    provider: str,
    request: Request,
    redirect_uri: str = None,
    id_token_hint: str | None = None,
):
    """Initiate OAuth2 logout flow to clear provider session"""
    try:
        if provider not in OAUTH2_CONFIG.get("providers", {}):
            raise HTTPException(status_code=404, detail=f"Provider {provider} not found")

        # Reject absolute redirect_uri that escapes the
        # deployment's cookie domain before forwarding it to the IdP. The IdP's
        # post_logout_redirect_uri allow-list is the authoritative check, but
        # this guards against misconfigured IdP clients and makes the intent
        # explicit at our boundary.
        if redirect_uri:
            parsed = urlparse(redirect_uri)
            if parsed.scheme or parsed.netloc:
                cookie_domain = os.environ.get("SESSION_COOKIE_DOMAIN", "").strip()
                if not _is_redirect_within_cookie_domain(redirect_uri, cookie_domain, request):
                    logger.warning(
                        f"Blocked unsafe logout redirect_uri for {provider}: {redirect_uri}"
                    )
                    redirect_uri = None

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
                    parsed = urlparse(referer)
                    registry_base = f"{parsed.scheme}://{parsed.netloc}"
                else:
                    registry_base = "http://localhost"

            full_redirect_uri = f"{registry_base.rstrip('/')}{full_redirect_uri}"

        # Detect provider type and build appropriate logout URL
        # Keycloak uses post_logout_redirect_uri, Cognito uses logout_uri
        parsed_logout_url = urlparse(logout_url)
        logout_hostname = parsed_logout_url.hostname or ""
        logout_path = parsed_logout_url.path or ""

        if "keycloak" in provider.lower() or "/realms/" in logout_path:
            # Keycloak logout parameters
            logout_params = {
                "client_id": provider_config["client_id"],
                "post_logout_redirect_uri": full_redirect_uri,
            }
            if id_token_hint:
                logout_params["id_token_hint"] = id_token_hint
            logger.debug(f"Keycloak logout params built: has_id_token_hint={bool(id_token_hint)}")
        elif logout_hostname == "login.microsoftonline.com" or "entra" in provider.lower():
            # Entra ID logout parameters
            logout_params = {
                "post_logout_redirect_uri": full_redirect_uri,
            }
            if id_token_hint:
                logout_params["id_token_hint"] = id_token_hint
            logger.debug(f"Entra ID logout params built: has_id_token_hint={bool(id_token_hint)}")
        elif "okta" in provider.lower() or (
            logout_hostname and logout_hostname.endswith(".okta.com")
        ):
            # Okta logout parameters
            logout_params = {
                "post_logout_redirect_uri": full_redirect_uri,
            }
            if id_token_hint:
                logout_params["id_token_hint"] = id_token_hint
            logger.debug(f"Okta logout params built: has_id_token_hint={bool(id_token_hint)}")
        else:
            # Cognito logout parameters (no id_token_hint support)
            logout_params = {
                "client_id": provider_config["client_id"],
                "logout_uri": full_redirect_uri,
            }
            logger.debug("Cognito logout params built (no id_token_hint)")

        logout_redirect_url = f"{logout_url}?{urllib.parse.urlencode(logout_params)}"

        logger.info(f"Redirecting to {provider} logout")
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


# ---------------------------------------------------------------------------
# MCP tools/list filter proxy hop (Issue #1026)
# ---------------------------------------------------------------------------
#
# nginx sends MCP POSTs that need tools/list filtering to this endpoint
# instead of routing them directly to the upstream MCP server. The nginx
# location block is expected to set two headers before forwarding:
#
#   X-Upstream-Url:  Full URL of the upstream MCP server to forward to.
#   X-Scopes:        Space-separated user scopes (already populated by
#                    the /validate auth_request hop). The proxy trusts
#                    this header because it comes from the same nginx
#                    pass that ran auth_request and can only have been
#                    written after /validate accepted the caller.
#
# All headers from the incoming request (except hop-by-hop headers) are
# forwarded to the upstream. The upstream response headers are likewise
# forwarded back to the client (except framing/encoding headers that
# Starlette must recompute) so streamable-http session state such as the
# ``Mcp-Session-Id`` header that the upstream emits during ``initialize``
# propagates end-to-end. Without that header the MCP client cannot
# establish a session and follow-up calls fail with "Missing session ID".
# For any JSON-RPC method other than "tools/list", the upstream response
# body is returned unchanged ("uniform proxy"). For tools/list, the
# response body is buffered (up to MCP_PROXY_MAX_BODY_BYTES), filtered
# via filter_tools_list_response, and returned with an updated
# result.tools array.


_HOP_BY_HOP_HEADERS: frozenset[str] = frozenset(
    {
        "connection",
        "keep-alive",
        "proxy-authenticate",
        "proxy-authorization",
        "te",
        "trailers",
        "transfer-encoding",
        "upgrade",
        "host",
        "content-length",
    }
)


# Allowlist of upstream response headers the proxy is permitted to forward
# back to the MCP client. The auth-server sits on a trust boundary in front
# of arbitrary upstream MCP servers, so the default posture is to drop
# everything and explicitly opt-in the small set of headers MCP clients
# actually act on. This prevents an upstream from dictating response-header
# policy that browsers / MCP clients honor (Set-Cookie, Strict-Transport-
# Security, Content-Security-Policy, Access-Control-Allow-*, Location,
# Server, etc.) and from overriding framing headers Starlette must set
# itself (Content-Length, Content-Encoding, Transfer-Encoding, Connection).
#
# Entries are stored lowercase; matching is case-insensitive (HTTP header
# names are case-insensitive per RFC 9110 section 5.1). Adding to this set
# is a security-relevant change -- every new entry should be reviewed for
# whether an upstream-supplied value could be abused before it lands.
#
# Why each entry is here:
#   mcp-session-id     streamable-http MCP servers emit this on initialize
#                      and clients must echo it on follow-up requests to
#                      identify the session (MCP spec, "Session
#                      management").
#   x-mcp-session-id   legacy alias some MCP servers / clients still emit;
#                      kept for compatibility with the same flow.
#   www-authenticate   required by the PRM / OAuth resource-metadata flow
#                      added in PR #1115 so 401 responses point clients at
#                      the authorization server.
#   retry-after        lets clients honor upstream 429 / 503 backoff hints
#                      rather than retry-storm the upstream.
_FORWARDED_RESPONSE_HEADERS: frozenset[str] = frozenset(
    {
        "mcp-session-id",
        "x-mcp-session-id",
        "www-authenticate",
        "retry-after",
    }
)


async def _read_bounded(
    response: httpx.Response,
    max_bytes: int,
) -> bytes:
    """Read upstream body in chunks, enforcing an upper size bound.

    Raises HTTPException(413) if total bytes exceed max_bytes.
    """
    size = 0
    chunks: list[bytes] = []
    async for chunk in response.aiter_bytes(chunk_size=64 * 1024):
        size += len(chunk)
        if size > max_bytes:
            raise HTTPException(
                status_code=413,
                detail=(
                    f"Upstream tools/list response exceeded {max_bytes} bytes; refusing to buffer."
                ),
            )
        chunks.append(chunk)
    return b"".join(chunks)


def _forward_headers(
    incoming: dict[str, str],
) -> dict[str, str]:
    """Copy incoming request headers, stripping hop-by-hop and proxy-hint
    headers so httpx can set them correctly for the upstream connection.
    """
    forwarded: dict[str, str] = {}
    for key, value in incoming.items():
        lower = key.lower()
        if lower in _HOP_BY_HOP_HEADERS:
            continue
        if lower in ("x-upstream-url",):
            # Never leak this internal routing header to the upstream.
            continue
        forwarded[key] = value
    return forwarded


# Headers that MUST be stripped before injecting a vaulted egress token:
# the user's gateway IdP JWT / session cookie / X-Authorization are full gateway
# credentials and must never reach a third-party SaaS upstream; the X-User*/
# X-Internal-Token/X-Scopes family is gateway-internal identity/routing. Only
# applied on the oauth_user egress path (other servers keep existing behavior).
_EGRESS_STRIP_HEADERS: frozenset[str] = frozenset(
    {
        "authorization",
        "x-authorization",
        "proxy-authorization",
        "cookie",
        "x-user",
        "x-username",
        "x-client-id",
        "x-scopes",
        "x-auth-method",
        "x-server-name",
        "x-tool-name",
        "x-groups",
        "x-internal-token",
        "x-user-pool-id",
        "x-region",
        "x-original-url",
    }
)


async def _vend_egress_token(
    internal_proxy_token: str,
    server_first_segment: str,
) -> dict | None:
    """Call the registry's internal egress-token vend endpoint.

    Forwards the verified X-Internal-Token; the registry re-verifies it,
    re-derives sub/auth_method from the signed claims, runs the allowlist
    and upstream cross-check, and vends. Returns the JSON response dict, or
    None on transport failure (treated as a clean miss -> consent).
    """
    from registry.auth.internal import generate_internal_token

    base = settings.egress_registry_internal_url.rstrip("/")
    try:
        service_token = generate_internal_token(subject="auth-server", purpose="egress-token-vend")
    except ValueError as exc:
        logger.error(f"egress vend: cannot mint internal service token: {exc}")
        return None

    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            resp = await client.post(
                f"{base}/_egress_internal/egress-token",
                json={"server_path": server_first_segment},
                headers={
                    "Authorization": f"Bearer {service_token}",
                    "X-Internal-Token": internal_proxy_token,
                    "Content-Type": "application/json",
                },
            )
    except httpx.HTTPError as exc:
        logger.error(f"egress vend: registry unreachable: {exc}")
        return None

    if resp.status_code != 200:
        logger.warning(f"egress vend: registry returned {resp.status_code}")
        return None
    try:
        return resp.json()
    except ValueError:
        return None


# JSON-RPC client requests on which MCP permits an InputRequiredResult (MRTR
# spec: only tools/call, prompts/get, resources/read). Emitting it on any other
# method (e.g. initialize, tools/list) would be a protocol violation, so those
# get a plain JSON-RPC error instead.
_ELICITATION_PERMITTED_METHODS: frozenset[str] = frozenset(
    {"tools/call", "prompts/get", "resources/read"}
)


# Protocol version the gateway advertises when it answers `initialize` locally
# (the fallback used if the client did not state one). The handshake is a
# capability negotiation between the client and THIS server (the gateway); per
# the MCP lifecycle spec it does not require contacting the upstream, so the
# gateway answers it itself for an egress server whose token is not yet vaulted.
_DEFAULT_PROTOCOL_VERSION: str = "2025-11-25"


def _local_initialize_response(
    req_id: object,
    incoming_payload: object,
    connect_url: str = "",
    provider: str = "the provider",
):
    """Answer an MCP ``initialize`` locally, without proxying to the upstream.

    For an egress-configured server whose per-user token is NOT yet vaulted, the
    upstream (e.g. GitHub) is itself an OAuth resource server that 401s every
    call -- including ``initialize``. But ``initialize`` is capability
    negotiation between the client and the gateway; the MCP lifecycle spec does
    not require it to reach the upstream. Answering it here lets the (legacy,
    handshake-based) client complete the handshake so it can proceed to the
    token-requiring methods, where the egress consent elicitation is surfaced.

    The protocol version echoes the client's requested version when present so
    the client does not see a version it did not ask for.

    When a ``connect_url`` is supplied, the synthetic result also carries an
    ``instructions`` string naming the provider and the connect URL. Per the MCP
    lifecycle spec ``instructions`` is optional guidance the client MAY surface
    to the user/model, so this gives a best-effort, connect-time hint of the
    consent step (clients that ignore it still get consent on the first
    tools/call). This is the gateway's own handshake response -- the upstream
    cannot be reached pre-consent (it 401s) -- so adding the field rewrites
    nothing of the provider's.
    """
    requested_version = _DEFAULT_PROTOCOL_VERSION
    if isinstance(incoming_payload, dict):
        params = incoming_payload.get("params")
        if isinstance(params, dict) and params.get("protocolVersion"):
            requested_version = params["protocolVersion"]
    result = {
        "protocolVersion": requested_version,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "mcp-gateway-registry", "version": "1.0.0"},
    }
    if connect_url:
        result["instructions"] = (
            f"This server requires connecting your {provider} account before its "
            f"tools can be used. Open this URL in a browser, approve access, then "
            f"use the server's tools:\n{connect_url}"
        )
    return JSONResponse(
        status_code=200,
        content={"jsonrpc": "2.0", "id": req_id, "result": result},
    )


def _egress_consent_response(
    server_name: str,
    incoming_method: str | None,
    req_id: object,
    vend: dict,
):
    """Build the consent-required response for an egress server with no token.

    Implements the ``2025-11-25`` URL-mode elicitation OAuth pattern: on a
    ``tools/call`` (etc.) that needs a third-party token the user has not yet
    granted, the server returns a ``URLElicitationRequiredError`` (JSON-RPC error
    code ``-32042``) whose ``data.elicitations[]`` carries a ``mode: "url"``
    elicitation with a unique ``elicitationId`` and the gateway connect URL. The
    client gets user consent, opens the URL (third-party OAuth happens out of
    band, token vaulted by the gateway), then retries the original ``tools/call``.

    Spec: https://modelcontextprotocol.io/specification/2025-11-25/client/elicitation
    (URL Elicitation Required Error). ``-32042`` is the documented signal and is a
    JSON-RPC *error*, so a client that does not understand it still does not
    mistake it for success.

    NOTE: the ``2026-07-28``/draft MRTR ``InputRequiredResult`` (a *result* with
    ``resultType: "input_required"``) is a DIFFERENT, later mechanism that
    replaced server-initiated requests; current clients negotiate ``2025-11-25``
    and do not understand it, so we emit ``-32042`` here.
    """
    connect_url = vend.get("connect_url") or vend.get("authorize_url") or ""
    provider = vend.get("provider") or "the provider"
    message = f"Connect your {provider} account to use this server."
    # A short, unique correlation handle for this elicitation. It is only an
    # identifier the connect route can echo back via
    # notifications/elicitation/complete -- it carries NO state and needs no
    # integrity (the real principal/TTL binding lives in the session-verified
    # connect route + the vend's request_state). Must stay short: it is appended
    # to the connect URL, and using the ~700-char AEAD request_state blob here
    # blew the elicitation URL past client length limits (kiro rejected it).
    elicitation_id = secrets.token_urlsafe(12)

    # Thread the elicitationId into the connect URL so the connect route can
    # correlate completion (and per spec, the connect URL is what enforces the
    # same-user anti-phishing check, not the third-party endpoint).
    sep = "&" if "?" in connect_url else "?"
    url_with_id = (
        f"{connect_url}{sep}{urllib.parse.urlencode({'elicitationId': elicitation_id})}"
        if connect_url
        else ""
    )

    # Baseline (LLD-mandated, default): on a tools/call (etc.) return a SUCCESSFUL
    # JSON-RPC result with isError=true whose text carries the connect URL. This
    # works on EVERY MCP client (no -32042 support needed); the human sees the URL
    # in the tool output, connects, and re-runs. Elicitation below is the opt-in
    # enhancement for clients that understand url-mode.
    if (
        not settings.egress_consent_use_elicitation
        and incoming_method in _ELICITATION_PERMITTED_METHODS
        and connect_url
    ):
        logger.info(
            "mcp_proxy: egress consent required for server=%s method=%s; "
            "returning isError=true tool result with connect URL (baseline)",
            server_name,
            incoming_method,
        )
        return JSONResponse(
            status_code=200,
            content={
                "jsonrpc": "2.0",
                "id": req_id,
                "result": {
                    "content": [
                        {
                            "type": "text",
                            "text": (
                                f"{message}\n\nTo connect, open this URL in your "
                                f"browser, approve access, then run this tool "
                                f"again:\n{url_with_id}"
                            ),
                        }
                    ],
                    "isError": True,
                },
            },
        )

    if incoming_method in _ELICITATION_PERMITTED_METHODS and connect_url:
        logger.info(
            "mcp_proxy: egress consent required for server=%s method=%s; "
            "returning URLElicitationRequiredError (-32042, url-mode)",
            server_name,
            incoming_method,
        )
        return JSONResponse(
            status_code=200,
            content={
                "jsonrpc": "2.0",
                "id": req_id,
                "error": {
                    "code": -32042,
                    "message": message,
                    "data": {
                        "elicitations": [
                            {
                                "mode": "url",
                                "elicitationId": elicitation_id,
                                "url": url_with_id,
                                "message": message,
                            }
                        ]
                    },
                },
            },
        )

    # Method is not one the URLElicitationRequiredError pattern applies to, or we
    # have no connect URL: return a generic JSON-RPC error that still carries the
    # connect URL so a human can self-serve. (In practice the dispatch routes
    # tools/list and notifications elsewhere, so this is a defensive fallback.)
    logger.info(
        "mcp_proxy: egress consent required for server=%s method=%s; "
        "returning generic JSON-RPC error (no url-elicitation for this method)",
        server_name,
        incoming_method,
    )
    return JSONResponse(
        status_code=200,
        content={
            "jsonrpc": "2.0",
            "id": req_id,
            "error": {
                "code": -32001,
                "message": (f"{message} Visit: {connect_url}" if connect_url else message),
                "data": {
                    "connect_url": connect_url,
                    "reason": "egress_consent_required",
                },
            },
        },
    )


def _select_forwarded_response_headers(
    upstream_headers: Mapping[str, str],
) -> dict[str, str]:
    """Select upstream MCP response headers safe to forward back to the
    client, applying the ``_FORWARDED_RESPONSE_HEADERS`` allowlist with
    case-insensitive matching.

    Anything outside the allowlist is silently dropped. This is the
    enforcement point for the trust boundary between the auth-server and
    arbitrary upstream MCP servers -- an upstream cannot dictate
    ``Set-Cookie``, ``Location``, ``Strict-Transport-Security`` etc. on
    responses the gateway issues.

    The original header casing from the upstream is preserved on the
    returned dict so the bytes on the wire match what the MCP server sent
    (e.g. ``Mcp-Session-Id`` rather than ``mcp-session-id``). HTTP header
    lookups remain case-insensitive on Starlette's ``Response.headers``.
    """
    selected: dict[str, str] = {}
    for key, value in upstream_headers.items():
        if key.lower() in _FORWARDED_RESPONSE_HEADERS:
            selected[key] = value
    return selected


@app.post("/mcp-proxy/{server_name:path}", dependencies=[Depends(verify_mcp_proxy_token)])
async def mcp_proxy(
    server_name: str,
    request: Request,
):
    """Forward an MCP JSON-RPC POST to the upstream and optionally filter
    the tools/list result by the caller's tool allowlist.

    nginx routes here after the /validate auth_request succeeds. The
    verify_mcp_proxy_token dependency has already verified the /validate-minted
    X-Internal-Token and stashed its claims on request.state.mcp_proxy_claims;
    identity/scopes/upstream are read from those verified claims, NOT from the
    inbound X-User/X-Scopes/X-Upstream-Url headers.

    For methods other than "tools/list" or when filtering is disabled,
    the upstream response is returned verbatim. For "tools/list", the
    result.tools array is filtered using filter_tools_list_response.
    """
    # verify_mcp_proxy_token (route dependency) has already verified the token
    # and stashed the claims; it raises 401 before reaching here on any failure,
    # so claims is always present. Identity/scopes/destination come from the
    # verified token -- the inbound X-User/X-Scopes/X-Upstream-Url are ignored.
    claims = request.state.mcp_proxy_claims
    upstream_url = claims["upstream_url"]
    user_scopes: list[str] = list(claims.get("scopes") or [])

    # Append the MCP sub-path from the request. server_name captures the full
    # path after /mcp-proxy/ (e.g. "airegistry-tools/mcp"). The first segment
    # is the registered server name; everything after is the sub-path that must
    # be appended to the upstream URL so the backend receives the correct route.
    # Skip if the upstream URL already ends with the sub-path (e.g. proxy_pass_url
    # is https://docs.mcp.cloudflare.com/mcp and sub_path is also /mcp).
    #
    # SECURITY: do NOT move this sub-path append into nginx. The X-Internal-Token
    # binds the PRE-append upstream_url (the $backend_url /validate saw). Keeping
    # the append here means the bound claim equals the upstream BASE and the
    # outbound URL is base + sub_path on that same bound host -- the destination
    # host is cryptographically pinned and the sub-path is confined to it. Moving
    # the append to nginx would diverge the signed base from what /validate saw
    # and 401 every request.
    if "/" in server_name:
        sub_path = server_name.split("/", 1)[1].lstrip("/")
        if sub_path and not upstream_url.rstrip("/").endswith("/" + sub_path):
            upstream_url = upstream_url.rstrip("/") + "/" + sub_path

    # Read the incoming body once; we forward it to the upstream.
    try:
        request_body = await request.body()
    except Exception as exc:
        logger.error(f"mcp_proxy: failed to read request body: {exc}")
        raise HTTPException(status_code=400, detail="Invalid request body") from exc

    # Determine the JSON-RPC method (best-effort; non-JSON bodies pass
    # through as-is).
    incoming_method: str | None = None
    incoming_payload: object = None
    try:
        if request_body:
            incoming_payload = json.loads(request_body.decode("utf-8"))
            if isinstance(incoming_payload, dict):
                incoming_method = incoming_payload.get("method")
    except (UnicodeDecodeError, json.JSONDecodeError):
        incoming_method = None
    except Exception as exc:
        logger.debug(f"mcp_proxy: could not parse incoming body as JSON-RPC: {exc}")
        incoming_method = None

    filter_enabled = _read_mcp_filter_enabled()
    max_body_bytes = _read_mcp_proxy_max_body_bytes()
    proxy_timeout = _read_mcp_proxy_timeout()
    forward_headers = _forward_headers(dict(request.headers))

    # True once we inject a vaulted egress token below. An egress upstream is
    # itself an OAuth resource server: if it rejects our injected token it 401s
    # with its OWN WWW-Authenticate (resource_metadata pointing at the upstream's
    # PRM, e.g. https://mcp.slack.com/.well-known/oauth-protected-resource). That
    # header is on the forward allowlist, so without intervention the gateway
    # would relay the upstream's resource identifier to the MCP client, which
    # rejects it as not matching the gateway resource it connected to (the
    # cross-resource "Protected resource ... does not match expected ..." error).
    # We drop the foreign header on this path (see the 401 handling below).
    egress_token_injected = False

    # Per-user egress credential vault. When the feature is on, ask the
    # registry to vend this user's third-party token for the resolved server. The
    # registry re-verifies the signed proxy token, enforces per-user/upstream
    # authz, and returns consent_required for non-oauth_user servers.
    # On a real vend we strip the user's own gateway credentials/identity
    # before injecting the vaulted token. On a consent-required miss for an
    # egress-configured server we DO NOT forward unauthenticated -- we ask the
    # user to connect via MCP URL-mode elicitation (see _egress_consent_response).
    if settings.egress_auth_enabled:
        internal_proxy_token = request.headers.get("X-Internal-Token", "")
        if internal_proxy_token:
            server_first_segment = (server_name or "").split("/", 1)[0]
            vend = await _vend_egress_token(internal_proxy_token, server_first_segment)
            if vend and vend.get("access_token"):
                # Token is vaulted (consent done): strip the user's gateway
                # credentials/identity and inject the vaulted upstream token.
                forward_headers = {
                    k: v
                    for k, v in forward_headers.items()
                    if k.lower() not in _EGRESS_STRIP_HEADERS
                }
                forward_headers["Authorization"] = f"Bearer {vend['access_token']}"
                egress_token_injected = True
            elif vend and (vend.get("connect_url") or vend.get("authorize_url")):
                # Egress is configured for this server but the user has no usable
                # token, and the upstream is itself an OAuth resource server that
                # 401s every call (including initialize). Break the handshake
                # deadlock by handling the non-upstream methods at the gateway:
                #
                #   - initialize: answered LOCALLY (capability negotiation with the
                #     client; the lifecycle spec does not require reaching the
                #     upstream). Lets a legacy handshake-based client complete the
                #     handshake instead of seeing the upstream's 401.
                #   - notifications/*: acked locally (no response body expected).
                #   - tools/list: answered LOCALLY with a single synthetic
                #     "connect" tool. The real upstream list needs the token, and
                #     erroring here dead-ends clients; the tools spec lets
                #     tools/list return an auth-dependent (here: connect-only) set.
                #   - tools/call (and prompts/get, resources/read): need the
                #     third-party token, so we ask the user to connect via MCP
                #     URL-mode elicitation. The gateway is the MCP server's OAuth
                #     client to the provider; the token never transits the MCP
                #     client, which performs no OAuth itself (it opens the connect
                #     URL and retries). Spec:
                #     https://modelcontextprotocol.io/specification/draft/client/elicitation
                req_id = incoming_payload.get("id") if isinstance(incoming_payload, dict) else None
                if incoming_method == "initialize":
                    logger.info(
                        "mcp_proxy: egress server=%s has no token; answering "
                        "initialize locally to complete the handshake",
                        server_name,
                    )
                    return _local_initialize_response(
                        req_id,
                        incoming_payload,
                        connect_url=vend.get("connect_url") or vend.get("authorize_url") or "",
                        provider=vend.get("provider") or "the provider",
                    )
                if incoming_method and incoming_method.startswith("notifications/"):
                    # Notifications have no result; ack with 202 and no body.
                    return Response(status_code=202)
                if incoming_method == "tools/list":
                    # No vaulted token yet: the upstream tool list needs the
                    # token, and tools/list MUST NOT error (that dead-ends
                    # clients). Return an EMPTY list. The user connects the
                    # account out of band via the Registry "Connected Accounts"
                    # page (the initialize `instructions` nudge points there);
                    # once vaulted, the vend HITs and the real upstream tools are
                    # proxied. A tools/call before connecting still gets the
                    # consent nudge via _egress_consent_response below.
                    logger.info(
                        "mcp_proxy: egress server=%s has no token; returning EMPTY "
                        "tools/list (connect via the Connected Accounts page)",
                        server_name,
                    )
                    return JSONResponse(
                        status_code=200,
                        content={"jsonrpc": "2.0", "id": req_id, "result": {"tools": []}},
                    )
                return _egress_consent_response(
                    server_name=server_name,
                    incoming_method=incoming_method,
                    req_id=req_id,
                    vend=vend,
                )

    logger.info(
        f"mcp_proxy: server={server_name} method={incoming_method} filter_enabled={filter_enabled} timeout={proxy_timeout}"
    )

    try:
        async with httpx.AsyncClient(timeout=proxy_timeout) as client:
            async with client.stream(
                "POST",
                upstream_url,
                content=request_body,
                headers=forward_headers,
                params=dict(request.query_params),
            ) as upstream_response:
                # Buffer with the same cap whether or not we filter, so a
                # pathological upstream cannot DoS the proxy.
                body_bytes = await _read_bounded(upstream_response, max_body_bytes)
                status_code = upstream_response.status_code
                content_type = upstream_response.headers.get("content-type", "application/json")
                # Snapshot upstream headers BEFORE leaving the stream
                # context. httpx releases response.headers when the
                # async-with block exits; reading them afterwards returns
                # an empty mapping and Mcp-Session-Id is silently lost.
                # Capture MCP session headers before the response stream closes
                upstream_headers = dict(upstream_response.headers)
    except HTTPException:
        raise
    except httpx.TimeoutException as exc:
        logger.error(f"mcp_proxy: upstream timeout for {upstream_url}: {exc}")
        raise HTTPException(
            status_code=504,
            detail="Upstream MCP server timed out",
        ) from exc
    except httpx.HTTPError as exc:
        logger.error(f"mcp_proxy: upstream error for {upstream_url}: {exc}")
        raise HTTPException(
            status_code=502,
            detail="Upstream MCP server error",
        ) from exc

    # Only filter successful tools/list JSON responses.
    should_filter = (
        filter_enabled
        and incoming_method == "tools/list"
        and 200 <= status_code < 300
        and "application/json" in content_type.lower()
    )

    # Apply the response-header allowlist. Anything outside
    # _FORWARDED_RESPONSE_HEADERS (Set-Cookie, Location, HSTS, CSP,
    # framing headers, etc.) is dropped here so an upstream MCP server
    # cannot dictate response-header policy on the gateway. The
    # allowlist itself is the auditable enforcement point -- adding to
    # it requires a security review (see comment on the constant).
    response_headers = _select_forwarded_response_headers(upstream_headers)

    # Egress trust boundary: when we injected a vaulted egress token and the
    # upstream still 401s, the token is bad/insufficient (e.g. a Slack bot token
    # where mcp.slack.com requires a user token). The upstream's WWW-Authenticate
    # advertises the UPSTREAM's resource_metadata; relaying it makes the MCP
    # client chase the upstream's PRM and fail the cross-resource check against
    # the gateway URL it connected to. Drop it so the client does not see a
    # foreign resource identifier. (Re-consent is surfaced on the token-requiring
    # methods via the URL-mode elicitation, not via this passthrough 401.)
    if egress_token_injected and status_code == 401:
        for key in [k for k in response_headers if k.lower() == "www-authenticate"]:
            del response_headers[key]
        logger.warning(
            "mcp_proxy: egress server=%s upstream 401 with injected token; "
            "dropped upstream WWW-Authenticate to avoid cross-resource PRM mismatch",
            server_name,
        )

    if not should_filter:
        # Forward the upstream body and content_type unchanged. Many MCP
        # servers reply with text/event-stream (SSE) rather than plain JSON;
        # wrapping that in {"raw": "..."} via JSONResponse breaks every
        # MCP client that expects either JSON-RPC or SSE.
        return Response(
            content=body_bytes,
            status_code=status_code,
            media_type=content_type,
            headers=response_headers,
        )

    try:
        parsed = json.loads(body_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        logger.warning(
            f"mcp_proxy: tools/list upstream returned non-JSON body for server={server_name}: {exc}"
        )
        return JSONResponse(
            content=_safe_parse_body(body_bytes),
            status_code=status_code,
            headers=response_headers,
        )

    result = parsed.get("result") if isinstance(parsed, dict) else None
    if isinstance(result, dict) and isinstance(result.get("tools"), list):
        filtered = await filter_tools_list_response(
            server_name,
            user_scopes,
            result["tools"],
        )
        result["tools"] = filtered
        parsed["result"] = result

    return JSONResponse(content=parsed, status_code=status_code, headers=response_headers)


def _safe_parse_body(
    body_bytes: bytes,
) -> Any:
    """Best-effort JSON parse for passthrough bodies.

    Returns the decoded JSON object if possible, otherwise a wrapper
    dict exposing the raw text. Keeps the proxy behavior lenient with
    non-standard upstream responses.
    """
    try:
        return json.loads(body_bytes.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {"raw": body_bytes.decode("utf-8", errors="replace")}


# ---------------------------------------------------------------------------
# Startup: legacy-scope audit (Issue #1026, LLD Step 9)
# ---------------------------------------------------------------------------


async def _audit_legacy_scopes_on_startup() -> int:
    """Emit one WARN per legacy scope row with a missing or invalid tools
    allowlist.

    Prefers the canonical helper from registry.auth.access_resolver when
    it is importable; falls back to a local shim that uses the same
    scope_repo auth_server already talks to. Never raises; returns the
    number of warnings emitted so tests / startup logs can assert.
    """
    try:
        from registry.auth.access_resolver import (
            audit_legacy_scopes_on_startup as _canonical_audit,
        )
    except Exception:
        _canonical_audit = None

    if _canonical_audit is not None:
        try:
            return await _canonical_audit()
        except Exception as exc:
            logger.error(
                f"Legacy scope audit (canonical) failed: {exc}",
                exc_info=True,
            )
            return 0

    # Local shim: same shape as the LLD helper, using list_groups to
    # enumerate scope names.
    warnings_emitted = 0
    try:
        scope_repo = get_scope_repository()
    except Exception as exc:
        logger.error(f"Legacy scope audit: cannot obtain scope repo: {exc}")
        return 0

    try:
        groups = await scope_repo.list_groups()
    except Exception as exc:
        logger.error(f"Legacy scope audit: list_groups failed: {exc}")
        return 0

    for scope_name in groups.keys():
        try:
            rules = await scope_repo.get_server_scopes(scope_name)
        except Exception as exc:
            logger.warning(f"Legacy scope audit: get_server_scopes({scope_name}) failed: {exc}")
            continue
        for rule in rules or []:
            if not isinstance(rule, dict) or "server" not in rule:
                continue
            server_name = rule.get("server")
            tools = rule.get("tools")
            methods = rule.get("methods") or []
            if tools is None:
                logger.warning(
                    f"legacy_scope_missing_tools scope={scope_name} "
                    f"server={server_name} methods={methods} "
                    "(post-upgrade this will deny all tools/list and "
                    "tools/call; migrate to tools: ['all'] if wildcard "
                    "was intended)"
                )
                warnings_emitted += 1
            elif (
                isinstance(tools, list)
                and not tools
                and ("tools/call" in methods or "all" in methods or "*" in methods)
            ):
                logger.warning(
                    f"empty_tools_list_with_call_method scope={scope_name} server={server_name}"
                )
                warnings_emitted += 1

    if warnings_emitted:
        logger.warning(
            f"Legacy scope audit: {warnings_emitted} warnings. See log lines "
            "above for specific scope/server pairs. Update the mcp-scopes "
            "collection before upgrading."
        )
    else:
        logger.info("Legacy scope audit: no issues found.")
    return warnings_emitted


@app.on_event("startup")
async def _run_legacy_scope_audit_on_startup() -> None:
    """Run the legacy-scope audit once during auth_server boot."""
    try:
        await _audit_legacy_scopes_on_startup()
    except Exception as exc:
        logger.error(
            f"Legacy scope audit errored during startup: {exc}",
            exc_info=True,
        )
