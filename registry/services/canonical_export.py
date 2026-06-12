"""Project a stored (bespoke) server document into a canonical MCP Registry server.json.

Read-only, pure functions. See issue #1187 and the LLD for the field mapping.
Private helpers (prefixed _) first, public functions after.
"""

import copy
import logging
import re
from functools import lru_cache
from typing import Any
from urllib.parse import urlparse

from ..core.config import settings

logger = logging.getLogger(__name__)

DEFAULT_SCHEMA_URL: str = (
    "https://raw.githubusercontent.com/modelcontextprotocol/registry/"
    "main/docs/reference/server-json/draft/server.schema.json"
)
MAX_CANONICAL_DESCRIPTION: int = 100

# Map the registry's local_runtime.type to a canonical packages[].registryType.
# registryType is a FREE-FORM string in the upstream schema (examples only, no enum),
# so any value validates. There is no package-registry analogue for a raw "command",
# so it maps to "mcpb" (the upstream example value for a packaged/bundled server).
RUNTIME_TO_REGISTRY_TYPE: dict[str, str] = {
    "npx": "npm",
    "uvx": "pypi",
    "docker": "oci",
    "command": "mcpb",
}

INTERNAL_FIELDS: list[str] = [
    "id",
    "server_name",
    "tags",
    "num_tools",
    "license",
    "deployment",
    "registered_by",
    "proxy_pass_url",
    "auth_scheme",
    "auth_provider",
    "path",
    "is_active",
    "is_enabled",
    "registered_at",
    "updated_at",
    "tool_list",
    "visibility",
    "allowed_groups",
    "status",
    "provider_organization",
    "provider_url",
    "source_created_at",
    "source_updated_at",
    "mcp_server_version",
    "health_status",
    "last_checked_iso",
]

LOCAL_EXTRA_FIELDS: list[str] = ["image_digest", "platforms"]


@lru_cache(maxsize=1)
def _reverse_dns_base(registry_url: str) -> str:
    """Convert a registry URL host into a reverse-DNS base label.

    https://mcpgateway.mycorp.com -> com.mycorp.mcpgateway
    http://localhost:8000        -> localhost
    """
    host = urlparse(registry_url).hostname or "localhost"
    labels = [label for label in host.split(".") if label]
    reversed_host = ".".join(reversed(labels)) if len(labels) > 1 else host
    # Namespace segment may contain only [a-zA-Z0-9.-]; map illegal chars to '-'.
    return re.sub(r"[^a-zA-Z0-9.-]", "-", reversed_host)


def _meta_namespace() -> str:
    """Reverse-DNS namespace for the registry's own _meta block, derived from REGISTRY_URL."""
    return f"{_reverse_dns_base(settings.registry_url)}/internal"


def _name_vendor() -> str:
    """Vendor prefix for synthesized canonical names, derived from REGISTRY_URL."""
    return _reverse_dns_base(settings.registry_url)


def _derive_canonical_name(stored: dict) -> str:
    """Synthesize a reverse-DNS name when the original wasn't preserved."""
    name = stored.get("server_name", "") or ""
    if "/" in name and "." in name.split("/", 1)[0]:
        return name
    slug = (stored.get("path") or "").lstrip("/")
    return f"{_name_vendor()}/{slug or 'unknown'}"


def _truncate_description(text: str) -> tuple[str, bool]:
    """Return (possibly-truncated description, was_truncated)."""
    if len(text) <= MAX_CANONICAL_DESCRIPTION:
        return text, False
    return text[: MAX_CANONICAL_DESCRIPTION - 1] + "…", True


def _build_remotes(
    stored: dict,
    spec: dict,
) -> list[dict] | None:
    """Prefer preserved remotes; else synthesize from proxy_pass_url."""
    if spec.get("remotes"):
        return spec["remotes"]
    if stored.get("proxy_pass_url"):
        transport = (stored.get("supported_transports") or ["streamable-http"])[0]
        return [{"type": transport, "url": stored["proxy_pass_url"]}]
    return None


def _local_runtime_to_package(lr: dict) -> dict:
    """Inverse of _derive_local_runtime in mcp_registry_transform.py."""
    env_vars: list[dict[str, Any]] = []
    seen: dict[str, int] = {}

    for name in lr.get("required_env", []) or []:
        seen[name] = len(env_vars)
        env_vars.append({"name": name, "isRequired": True})

    for key, value in (lr.get("env") or {}).items():
        if key in seen:
            env_vars[seen[key]]["default"] = value
        else:
            env_vars.append({"name": key, "default": value})

    pkg: dict[str, Any] = {
        # registryType is REQUIRED on a package by the upstream schema, so we always
        # emit it (never omit). It is free-form, so "mcpb" for command-type is valid.
        "registryType": RUNTIME_TO_REGISTRY_TYPE.get(lr.get("type", ""), "mcpb"),
        "identifier": lr.get("package", ""),
        "version": lr.get("version") or "1.0.0",
        "transport": {"type": "stdio"},
        "runtimeHint": lr.get("type", "command"),
    }
    if env_vars:
        pkg["environmentVariables"] = env_vars
    return pkg


def _build_packages(
    stored: dict,
    spec: dict,
) -> list[dict] | None:
    """Prefer preserved packages; else synthesize from local_runtime."""
    if spec.get("packages"):
        return spec["packages"]
    lr = stored.get("local_runtime")
    if not lr:
        return None
    return [_local_runtime_to_package(lr)]


def _build_internal_meta(
    stored: dict,
    truncated_full: str | None,
) -> dict:
    """Assemble the registry's own _meta block."""
    internal: dict[str, Any] = {k: stored[k] for k in INTERNAL_FIELDS if k in stored}

    lr = stored.get("local_runtime") or {}
    for extra_key in LOCAL_EXTRA_FIELDS:
        if extra_key in lr and lr[extra_key] is not None:
            internal[extra_key] = lr[extra_key]

    extra_meta = {
        k: v for k, v in (stored.get("metadata") or {}).items() if k != "mcp_registry_spec"
    }
    if extra_meta:
        internal["metadata"] = extra_meta

    if truncated_full is not None:
        internal["description_full"] = truncated_full

    return internal


def to_canonical(stored: dict) -> tuple[dict, bool]:
    """Project a stored server dict into a canonical server.json.

    Returns (canonical_doc, description_was_truncated).
    """
    spec = (stored.get("metadata") or {}).get("mcp_registry_spec") or {}

    raw_description = stored.get("description", "") or ""
    description, truncated = _truncate_description(raw_description)

    used_spec = bool(spec)
    logger.debug(
        "canonical_export: path=%s used_spec=%s",
        stored.get("path"),
        used_spec,
    )

    out: dict[str, Any] = {
        "$schema": spec.get("$schema", DEFAULT_SCHEMA_URL),
        "name": spec.get("original_name") or _derive_canonical_name(stored),
        "description": description,
        "version": stored.get("version") or spec.get("version") or "0.0.0",
    }

    remotes = _build_remotes(stored, spec)
    if remotes:
        out["remotes"] = remotes

    packages = _build_packages(stored, spec)
    if packages:
        out["packages"] = packages

    if spec.get("repository"):
        out["repository"] = spec["repository"]

    preserved_meta = spec.get("_meta") or {}
    internal = _build_internal_meta(
        stored,
        raw_description if truncated else None,
    )
    out["_meta"] = {**preserved_meta, _meta_namespace(): internal}

    return out, truncated


def _strip_remote_urls(remotes: Any) -> None:
    """Strip the backend `url` from each entry in a `remotes` list, in-place."""
    if not isinstance(remotes, list):
        return
    for remote in remotes:
        if isinstance(remote, dict):
            remote.pop("url", None)


def _redact_node(node: Any) -> None:
    """Recursively strip backend URLs from a canonical-doc subtree, in-place."""
    if isinstance(node, dict):
        # proxy_pass_url is the registry's internal backend URL. It lives in our
        # own _meta namespace, but a preserved upstream namespace could echo it
        # too, so drop it wherever it appears.
        node.pop("proxy_pass_url", None)
        # `url` is a backend URL only inside a `remotes` entry. Scope the removal
        # so non-backend URLs (repository.url, provider_url, ...) are preserved.
        if "remotes" in node:
            _strip_remote_urls(node["remotes"])
        for value in node.values():
            _redact_node(value)
    elif isinstance(node, list):
        for item in node:
            _redact_node(item)


def redact_backend_urls(canonical: dict) -> dict:
    """Return a redacted COPY of a canonical doc. Does NOT mutate the input."""
    redacted = copy.deepcopy(canonical)
    _redact_node(redacted)
    return redacted
