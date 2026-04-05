"""Pydantic models and helper functions for AgentCore auto-registration.

Contains data models for discovered resources and sync results,
plus utility functions for URL construction, slugification, auth
scheme mapping, and token loading.
"""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any
from urllib.parse import quote

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Constants
DEFAULT_REGISTRY_URL = "http://localhost"
DEFAULT_TOKEN_FILE = ".token"
DEFAULT_REGION = "us-east-1"
DEFAULT_TIMEOUT = 30
DEFAULT_MANIFEST_PATH = "token_refresh_manifest.json"
READY_STATUS = "READY"


# ---------------------------------------------------------------------------
# Pydantic Models
# ---------------------------------------------------------------------------


class TargetInfo(BaseModel):
    """Discovered Gateway Target information."""

    target_id: str = Field(..., description="Target ID")
    name: str = Field(..., description="Target name")
    description: str | None = Field(None, description="Target description")
    status: str = Field(..., description="Target status")
    target_type: str = Field(
        ..., description="mcpServer, lambda, apiGateway, etc."
    )
    endpoint: str | None = Field(
        None, description="MCP server endpoint (for mcpServer type)"
    )


class GatewayInfo(BaseModel):
    """Discovered AgentCore Gateway information."""

    gateway_id: str = Field(..., description="Gateway ID")
    gateway_arn: str = Field(..., description="Gateway ARN")
    gateway_url: str = Field(..., description="Gateway MCP endpoint URL")
    name: str = Field(..., description="Gateway name")
    description: str | None = Field(None, description="Gateway description")
    status: str = Field(..., description="Gateway status")
    authorizer_type: str = Field(..., description="CUSTOM_JWT, AWS_IAM, or NONE")
    authorizer_config: dict[str, Any] | None = Field(
        None, description="Authorizer configuration"
    )
    targets: list[TargetInfo] = Field(
        default_factory=list, description="Gateway targets"
    )


class RuntimeInfo(BaseModel):
    """Discovered AgentCore Runtime information."""

    runtime_id: str = Field(..., description="Runtime ID")
    runtime_arn: str = Field(..., description="Runtime ARN")
    runtime_name: str = Field(..., description="Runtime name")
    description: str | None = Field(None, description="Runtime description")
    status: str = Field(..., description="Runtime status")
    server_protocol: str = Field(..., description="MCP, HTTP, or A2A")
    authorizer_config: dict[str, Any] | None = Field(
        None, description="Authorizer configuration"
    )
    invocation_url: str = Field(..., description="Constructed invocation URL")


class SyncResult(BaseModel):
    """Result of a sync operation."""

    resource_type: str = Field(..., description="gateway, runtime, or target")
    resource_name: str = Field(..., description="Resource name")
    resource_arn: str = Field(..., description="Resource ARN")
    registration_type: str = Field(..., description="mcp_server or agent")
    path: str = Field(..., description="Registry path")
    status: str = Field(..., description="registered, skipped, failed, dry_run")
    message: str | None = Field(None, description="Status message or error")


class SyncSummary(BaseModel):
    """Summary of sync operation."""

    total_gateways: int = Field(0, description="Total gateways found")
    total_runtimes: int = Field(0, description="Total runtimes found")
    total_targets: int = Field(0, description="Total mcpServer targets found")
    registered: int = Field(0, description="Successfully registered")
    skipped: int = Field(0, description="Skipped (already exists)")
    failed: int = Field(0, description="Failed to register")
    credentials_saved: int = Field(0, description="Credentials persisted to .env")
    tokens_generated: int = Field(0, description="Egress tokens generated")
    dry_run: bool = Field(False, description="Whether this was a dry run")
    results: list[SyncResult] = Field(
        default_factory=list, description="Individual results"
    )


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------


def _slugify(name: str) -> str:
    """Convert name to URL-safe slug.

    Lowercase, replace spaces/underscores with hyphens, remove
    non-alphanumeric characters, collapse consecutive hyphens,
    strip leading/trailing hyphens. Idempotent.
    """
    slug = name.lower().replace(" ", "-").replace("_", "-")
    slug = re.sub(r"[^a-z0-9-]", "", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug


_UPPERCASE_WORDS: set[str] = {
    "mcp", "a2a", "sre", "api", "http", "https", "aws", "iam",
    "jwt", "oidc", "sso", "idp", "llm", "ai", "ml",
}


def _display_name(name: str) -> str:
    """Convert a slug or underscore-separated name to a human-readable title.

    Preserves common acronyms in uppercase (MCP, A2A, SRE, API, etc.).

    Examples:
        geo-mcp -> Geo MCP
        weather_time_observability_gateway -> Weather Time Observability Gateway
        my-custom-sre-agent -> My Custom SRE Agent
    """
    words = name.replace("-", " ").replace("_", " ").split()
    result = []
    for word in words:
        if word.lower() in _UPPERCASE_WORDS:
            result.append(word.upper())
        else:
            result.append(word.capitalize())
    return " ".join(result)


def _validate_https_url(url: str, resource_name: str) -> bool:
    """Validate that URL uses HTTPS protocol.

    Args:
        url: URL to validate.
        resource_name: Name of resource for logging.

    Returns:
        True if valid HTTPS URL, False otherwise.
    """
    if not url:
        logger.warning(f"Empty URL for resource: {resource_name}")
        return False

    if not url.startswith("https://"):
        logger.warning(
            f"Insecure URL for {resource_name}: {url} - "
            f"Expected HTTPS, skipping registration"
        )
        return False

    return True


def _build_invocation_url(region: str, runtime_arn: str) -> str:
    """Build the invocation URL for an AgentCore Runtime.

    Format: https://bedrock-agentcore.{region}.amazonaws.com/runtimes/{encoded-ARN}/invocations
    """
    encoded_arn = quote(runtime_arn, safe="")
    return (
        f"https://bedrock-agentcore.{region}.amazonaws.com"
        f"/runtimes/{encoded_arn}/invocations"
    )


def _get_auth_scheme(authorizer_type: str) -> str:
    """Map AgentCore authorizer type to registry auth scheme.

    CUSTOM_JWT -> bearer, AWS_IAM -> bearer, NONE -> none.
    Unknown types default to none.
    """
    mapping = {
        "CUSTOM_JWT": "bearer",
        "AWS_IAM": "bearer",
        "NONE": "none",
    }
    return mapping.get(authorizer_type, "none")


def _load_token(token_file: str) -> str:
    """Load JWT token from a JSON file.

    Supports two formats:
    - Flat: ``{"access_token": "..."}`` or ``{"token": "..."}``
    - Nested: ``{"tokens": {"access_token": "..."}}``

    Raises FileNotFoundError, ValueError on missing file, bad JSON,
    or missing token field.
    """
    abs_path = os.path.abspath(token_file)
    try:
        with open(abs_path) as f:
            data = json.load(f)
            # Try top-level first, then nested under "tokens"
            token = data.get("access_token") or data.get("token")
            if not token:
                tokens_obj = data.get("tokens", {})
                if isinstance(tokens_obj, dict):
                    token = tokens_obj.get("access_token") or tokens_obj.get("token")
            if not token:
                raise ValueError(
                    f"No access_token or token field in token file: {abs_path}"
                )
            return token
    except FileNotFoundError:
        raise FileNotFoundError(f"Token file not found: {abs_path}")
    except json.JSONDecodeError as e:
        raise ValueError(f"Invalid JSON in token file {abs_path}: {e}")
