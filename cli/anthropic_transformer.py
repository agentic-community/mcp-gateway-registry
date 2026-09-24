#!/usr/bin/env python3
"""Transform Anthropic MCP Registry server format to Gateway Registry format.

This module provides utilities to convert server definitions from the
Anthropic MCP Registry API format into the format expected by the
MCP Gateway Registry.
"""

import argparse
import json
import logging
import os
import sys
from typing import (
    Any,
)

# Configure logging with basicConfig
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s,p%(process)s,{%(filename)s:%(lineno)d},%(levelname)s,%(message)s",
)

logger = logging.getLogger(__name__)


# Constants
DEFAULT_BASE_PORT: int = 8100
DEFAULT_TRANSPORT: str = "stdio"
DEFAULT_DESCRIPTION: str = "MCP server imported from Anthropic Registry"
DEFAULT_LICENSE: str = "MIT"
DEFAULT_AUTH_PROVIDER: str = "keycloak"


def _extract_remote_info(
    remotes: list[dict[str, Any]],
) -> tuple[str | None, str, str, list[dict[str, str]]]:
    """Extract the remote URL and transport type from the ``remotes`` field.

    Remote-declared authentication headers are deliberately NOT wired in. The
    header placeholder name (e.g. ``{smithery_api_key}``) is fully controlled by
    the (unauthenticated, public-registry) definition, so preserving it as
    ``${ENV_VAR}`` would let an imported server select an arbitrary local secret
    — ``${SECRET_KEY}``, ``${AWS_SECRET_ACCESS_KEY}`` — to be resolved by the
    gateway and sent to the registrant-controlled backend once the server is
    enabled. There is no safe way to honor an attacker-chosen secret reference,
    so auth is dropped here (``scheme=none``, no headers); an operator configures
    auth explicitly for servers they choose to deploy.

    Args:
        remotes: List of remote server configurations.

    Returns:
        Tuple of ``(remote_url, transport_type, "none", [])``.
    """
    remote_url = None
    transport_type = DEFAULT_TRANSPORT

    if remotes:
        remote = remotes[0]
        remote_url = remote.get("url")
        transport_type = remote.get("type", "streamable-http")

    return remote_url, transport_type, "none", []


def _generate_tags(name: str) -> list[str]:
    """Generate tags from server name.

    Args:
        name: Server name (may contain slashes)

    Returns:
        List of tags including name parts and 'anthropic-registry'
    """
    name_parts = name.replace("/", "-").split("-")
    tags = name_parts + ["anthropic-registry"]
    return tags


def _reject_control_chars(field: str, value: str) -> None:
    """Raise if a string field carries an ASCII control character (incl. newline).

    These fields are copied verbatim from an unauthenticated remote record into
    the generated config that downstream shell tooling consumes; a control
    character is never legitimate and would corrupt the config contract.
    """
    if any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value):
        raise ValueError(f"{field} must not contain control characters")


def _reject_ssrf_target(url: str) -> None:
    """Reject an attacker-supplied backend URL that targets a non-public address.

    Reuses the repository's single hardened URL guard (``FEDERATION_PROFILE`` —
    public-only, empty allowlist) for a static, no-DNS SSRF category check:
    loopback, private, link-local, reserved, and cloud-metadata IP literals
    (including obfuscated spellings) and nginx metacharacters are denied. The
    rebinding-safe connect-time defense remains the guarded transport at the
    fetch sink; this blocks the direct IP-literal payloads at the source. Fails
    closed (raises) if the guard cannot be loaded or the URL is rejected.

    Args:
        url: The remote-supplied backend URL.

    Raises:
        ValueError: If the guard is unavailable or the URL is not a public
            http(s) target.
    """
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    if repo_root not in sys.path:
        sys.path.insert(0, repo_root)
    try:
        from registry.exceptions import UrlValidationError
        from registry.utils.url_guard import FEDERATION_PROFILE, validate_url
    except ImportError as exc:  # cannot verify -> deny
        raise ValueError(f"cannot load URL guard to validate proxy_pass_url: {exc}") from exc

    try:
        validate_url(
            url,
            profile=FEDERATION_PROFILE,
            resolve=False,
            reject_nginx_metacharacters=True,
        )
    except UrlValidationError as exc:
        raise ValueError(f"proxy_pass_url failed SSRF validation: {exc}") from exc


def _validate_remote_fields(
    name: str, description: str, proxy_url: str, remote_url: str | None
) -> None:
    """Validate remote-derived fields before they are emitted into a config.

    Fails closed (raises ``ValueError``) on anything malformed so a malicious or
    malformed public-registry definition is rejected rather than propagated.

    Args:
        name: Server name from the remote record.
        description: Server description from the remote record.
        proxy_url: Backend URL that becomes ``proxy_pass_url``.
        remote_url: The raw attacker-supplied remote URL (``None`` when the
            record has no remote and a local placeholder is used instead). Only a
            real remote URL is SSRF-checked; the local placeholder is trusted.

    Raises:
        ValueError: If a field is the wrong type, empty where required, carries a
            control character, is not an http(s) URL, or targets a non-public
            address.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("server name must be a non-empty string")
    _reject_control_chars("server name", name)

    if not isinstance(description, str):
        raise ValueError("description must be a string")
    _reject_control_chars("description", description)

    if not isinstance(proxy_url, str) or not proxy_url.strip():
        raise ValueError("proxy_pass_url must be a non-empty string")
    _reject_control_chars("proxy_pass_url", proxy_url)
    if any(ch.isspace() for ch in proxy_url):
        raise ValueError("proxy_pass_url must not contain whitespace")
    if not (proxy_url.startswith("http://") or proxy_url.startswith("https://")):
        raise ValueError("proxy_pass_url must be an http:// or https:// URL")

    # An attacker-supplied remote URL must be a public target (no SSRF to
    # loopback/private/metadata). The tool-generated local placeholder is exempt.
    if remote_url:
        _reject_ssrf_target(remote_url)


def transform_anthropic_to_gateway(
    anthropic_response: dict[str, Any], base_port: int = DEFAULT_BASE_PORT
) -> dict[str, Any]:
    """Transform Anthropic ServerResponse to Gateway Registry Config format.

    Args:
        anthropic_response: Server data from Anthropic Registry API
        base_port: Base port number for local proxy URLs

    Returns:
        Dictionary in Gateway Registry configuration format

    Example:
        >>> response = {"server": {"name": "brave-search", ...}}
        >>> config = transform_anthropic_to_gateway(response)
        >>> print(config["server_name"])
        brave-search
    """
    server = anthropic_response.get("server", anthropic_response)
    name = server["name"]

    tags = _generate_tags(name)

    remotes = server.get("remotes", [])
    remote_url, transport_type, auth_scheme, auth_headers = _extract_remote_info(remotes)

    # NOTE: auth-header values are kept as ``${ENV_VAR}`` placeholders and are
    # NEVER resolved to real secret values here. Resolving an attacker-named
    # placeholder at import time would bake a live secret into the on-disk
    # config, argv, and stdout; the gateway resolves placeholders at proxy time
    # for servers the operator has vetted and enabled.
    safe_path = name.replace("/", "-")

    proxy_url = remote_url if remote_url else f"http://localhost:{base_port}/"

    description = server.get("description", DEFAULT_DESCRIPTION)

    # Reject a malformed / unsafe remote definition rather than emitting it.
    _validate_remote_fields(name, description, proxy_url, remote_url)

    return {
        "server_name": name,
        "description": description,
        "path": f"/{safe_path}",
        "proxy_pass_url": proxy_url,
        "auth_provider": DEFAULT_AUTH_PROVIDER if auth_scheme != "none" else None,
        "auth_scheme": auth_scheme,
        "supported_transports": [transport_type],
        "tags": tags,
        "headers": auth_headers if auth_headers else [],
        "num_tools": 0,
        "license": DEFAULT_LICENSE,
        "remote_url": remote_url,
        "tool_list": [],
    }


# Fields the user-facing register_service tool does not accept.
_UNSUPPORTED_FIELDS: tuple[str, ...] = (
    "repository_url",
    "website_url",
    "package_npm",
    "remote_url",
)


def main(argv: list[str] | None = None) -> int:
    """Transform a fetched Anthropic record file into a Gateway Registry config.

    Reads the remote record as data from ``input_file``, transforms and validates
    it, and writes the resulting config to ``output_file``. Fails closed
    (non-zero exit) on any invalid or unsafe input.

    Args:
        argv: Optional argument list (defaults to ``sys.argv[1:]``).

    Returns:
        Process exit code (0 on success, 1 on failure).
    """
    parser = argparse.ArgumentParser(
        description="Transform an Anthropic MCP Registry record into a Gateway Registry config."
    )
    parser.add_argument("input_file", help="path to the fetched Anthropic server JSON")
    parser.add_argument("output_file", help="path to write the Gateway Registry config JSON")
    parser.add_argument(
        "--base-port", type=int, default=DEFAULT_BASE_PORT, help="base port for placeholder URLs"
    )
    parser.add_argument("--path", default=None, help="override the generated service path")
    args = parser.parse_args(argv)

    try:
        with open(args.input_file, encoding="utf-8") as handle:
            data = json.load(handle)
    except (OSError, json.JSONDecodeError) as exc:
        print(f"ERROR: could not read remote record: {exc}", file=sys.stderr)
        return 1

    try:
        result = transform_anthropic_to_gateway(data, args.base_port)
    except (ValueError, KeyError, TypeError) as exc:
        print(f"ERROR: invalid or unsafe remote definition: {exc}", file=sys.stderr)
        return 1

    if args.path is not None:
        result["path"] = args.path

    for field in _UNSUPPORTED_FIELDS:
        result.pop(field, None)

    with open(args.output_file, "w", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2)

    return 0


if __name__ == "__main__":
    sys.exit(main())
