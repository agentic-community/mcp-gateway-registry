"""Transform upstream MCP Registry server.json into internal registration format.

Detects the upstream schema via the $schema field and maps it to the dict
format expected by InternalServiceRegistration in api/registry_client.py.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any

from .mcp_registry_schema import McpRegistryServerJson

logger = logging.getLogger(__name__)

MCP_REGISTRY_SCHEMA_MARKER = "modelcontextprotocol/registry"

VALID_RUNTIME_HINTS: set[str] = {"npx", "uvx", "docker", "command"}


def _slugify(name: str) -> str:
    """Convert a namespaced name to a URL-safe path slug.

    Examples:
        io.example/calculator-mcp -> io-example-calculator-mcp
        My Cool Server -> my-cool-server
    """
    slug = name.lower()
    slug = re.sub(r"[^a-z0-9\-]", "-", slug)
    slug = re.sub(r"-+", "-", slug)
    slug = slug.strip("-")
    return slug


def _derive_proxy_pass_url(parsed: McpRegistryServerJson) -> str | None:
    """Derive proxy_pass_url from explicit field or first remote."""
    if parsed.proxy_pass_url:
        return parsed.proxy_pass_url

    if parsed.remotes:
        return parsed.remotes[0].url

    return None


def _derive_deployment(parsed: McpRegistryServerJson) -> str:
    """Derive deployment type from explicit field or schema contents."""
    if parsed.deployment:
        return parsed.deployment

    if parsed.remotes:
        return "remote"

    if parsed.packages:
        first_transport = parsed.packages[0].transport
        if first_transport and first_transport.type == "stdio":
            return "local"

    return "remote"


def _derive_supported_transports(parsed: McpRegistryServerJson) -> list[str]:
    """Derive supported transports from explicit field or remotes."""
    if parsed.supported_transports:
        return parsed.supported_transports

    if parsed.remotes:
        return [r.type for r in parsed.remotes]

    return ["streamable-http"]


def _derive_transport(parsed: McpRegistryServerJson) -> str:
    """Derive preferred transport from explicit field or first remote."""
    if parsed.transport:
        return parsed.transport

    if parsed.remotes:
        return parsed.remotes[0].type

    return "auto"


def _derive_auth_scheme(parsed: McpRegistryServerJson) -> str:
    """Derive auth scheme from explicit field or remote headers."""
    if parsed.auth_scheme:
        return parsed.auth_scheme

    if parsed.remotes:
        for header in parsed.remotes[0].headers:
            if header.name.lower() == "authorization":
                if "bearer" in header.value.lower():
                    return "bearer"

    return "none"


def _derive_local_runtime(parsed: McpRegistryServerJson) -> dict[str, Any] | None:
    """Build local_runtime dict from packages when deployment is local."""
    if not parsed.packages:
        return None

    pkg = parsed.packages[0]
    runtime_type = (pkg.runtime_hint or "command").lower()
    if runtime_type not in VALID_RUNTIME_HINTS:
        runtime_type = "command"

    env: dict[str, str] = {}
    required_env: list[str] = []

    for env_var in pkg.environment_variables:
        if env_var.is_required:
            required_env.append(env_var.name)
        if env_var.default:
            env[env_var.name] = env_var.default

    runtime: dict[str, Any] = {
        "type": runtime_type,
        "package": pkg.identifier,
        "version": pkg.version,
    }

    if env:
        runtime["env"] = env
    if required_env:
        runtime["required_env"] = required_env

    return runtime


def _build_metadata(
    parsed: McpRegistryServerJson,
    original_data: dict[str, Any],
) -> dict[str, Any]:
    """Build metadata dict preserving the original upstream spec fields."""
    metadata = dict(parsed.metadata) if parsed.metadata else {}

    spec_data: dict[str, Any] = {}

    if parsed.repository:
        spec_data["repository"] = parsed.repository.model_dump()

    if parsed.packages:
        spec_data["packages"] = [p.model_dump(by_alias=True) for p in parsed.packages]

    if parsed.remotes:
        spec_data["remotes"] = [r.model_dump(by_alias=True) for r in parsed.remotes]

    if parsed.meta:
        spec_data["_meta"] = parsed.meta

    if parsed.version:
        spec_data["version"] = parsed.version

    if parsed.schema_url:
        spec_data["$schema"] = parsed.schema_url

    spec_data["original_name"] = parsed.name

    metadata["mcp_registry_spec"] = spec_data
    return metadata


def is_mcp_registry_schema(data: dict[str, Any]) -> bool:
    """Check if a JSON config dict uses the upstream MCP Registry schema.

    Detection is based on the $schema field containing the marker string.
    """
    schema_value = data.get("$schema", "")
    if not isinstance(schema_value, str):
        return False

    return MCP_REGISTRY_SCHEMA_MARKER in schema_value


def transform_mcp_registry_to_internal(data: dict[str, Any]) -> dict[str, Any]:
    """Transform an upstream MCP Registry server.json into internal format.

    Returns a dict with keys matching what InternalServiceRegistration expects.
    The original upstream-specific fields are preserved in metadata["mcp_registry_spec"].
    """
    parsed = McpRegistryServerJson.model_validate(data)

    path = parsed.path if parsed.path else f"/{_slugify(parsed.name)}"
    if not path.startswith("/"):
        path = f"/{path}"

    server_name = parsed.title if parsed.title else parsed.name
    deployment = _derive_deployment(parsed)

    tags = list(parsed.tags) if parsed.tags else []
    if "mcp-registry-spec" not in tags:
        tags.append("mcp-registry-spec")

    tool_list = parsed.tool_list or []

    result: dict[str, Any] = {
        "path": path,
        "server_name": server_name,
        "name": server_name,
        "description": parsed.description,
        "proxy_pass_url": _derive_proxy_pass_url(parsed),
        "deployment": deployment,
        "transport": _derive_transport(parsed),
        "supported_transports": _derive_supported_transports(parsed),
        "auth_scheme": _derive_auth_scheme(parsed),
        "tags": tags,
        "num_tools": parsed.num_tools or len(tool_list),
        "tool_list": tool_list,
        "tool_list_json": json.dumps(tool_list) if tool_list else None,
        "metadata": _build_metadata(parsed, data),
        "status": parsed.status or "active",
        "version": parsed.version,
    }

    if parsed.visibility:
        result["visibility"] = parsed.visibility

    if parsed.allowed_groups:
        result["allowed_groups"] = parsed.allowed_groups

    if deployment == "local":
        local_runtime = _derive_local_runtime(parsed)
        if local_runtime:
            result["local_runtime"] = local_runtime
        result["proxy_pass_url"] = None

    logger.info(
        f"Transformed MCP Registry server.json: "
        f"name={parsed.name} -> path={path}, deployment={deployment}"
    )

    return result
