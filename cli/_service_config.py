#!/usr/bin/env python3
"""Data-only helpers for ``service_mgmt.sh``.

Every command here reads the *untrusted* payload (server configuration or
health-check output) from **stdin**, and takes only script-controlled selectors
(a field name, a filter string) on ``argv``. The payload therefore never reaches
an interpreter or shell code position.

``service_mgmt.sh`` previously spliced registrant-controlled JSON straight into
inline ``python3 -c`` source (``config = json.loads('''$config_json''')``) and
into ``eval``-ed command strings. A value containing ``'''``, ``$(...)``,
backticks, or a shell metacharacter escaped the literal and executed as Python
or shell — and that JSON is fetched unauthenticated from a public registry by
``import_from_anthropic_registry.sh``. Passing the payload as data on stdin (and
never through ``eval``) closes that entire class of injection.

All commands fail closed: malformed input exits non-zero with an ``ERROR:``
message rather than falling back to a permissive default.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from typing import Any

# Fields accepted by the register_service tool spec. Anything else is rejected
# so a malicious import cannot smuggle extra keys into the registry.
ALLOWED_FIELDS: frozenset[str] = frozenset(
    {
        "server_name",
        "path",
        "proxy_pass_url",
        "description",
        "tags",
        "num_tools",
        "license",
        "auth_provider",
        "auth_scheme",
        "supported_transports",
        "headers",
        "tool_list",
        "repository_url",
        "website_url",
        "package_npm",
    }
)

REQUIRED_FIELDS: tuple[str, ...] = ("server_name", "path", "proxy_pass_url")

# String fields that flow onward into shell variables, the two-line validate
# contract, and generated config. A control character in any of them is never
# legitimate and would corrupt the downstream contract, so reject it.
_CONTROLLED_STRING_FIELDS: tuple[str, ...] = (
    "server_name",
    "path",
    "proxy_pass_url",
    "description",
    "license",
)


def _fail(message: str) -> None:
    """Print a fail-closed error to stdout and exit non-zero."""
    print(message)
    sys.exit(1)


def _has_control_chars(value: str) -> bool:
    """True if the string contains an ASCII control character (incl. newline)."""
    return any(ord(ch) < 0x20 or ord(ch) == 0x7F for ch in value)


def _read_stdin_json() -> dict[str, Any]:
    """Parse the config object from stdin, failing closed on anything invalid."""
    raw = sys.stdin.read()
    try:
        config = json.loads(raw)
    except json.JSONDecodeError as exc:
        _fail(f"ERROR: Invalid JSON in config: {exc}")
    if not isinstance(config, dict):
        _fail("ERROR: Config must be a JSON object")
    return config


def _validate(config: dict[str, Any]) -> None:
    """Validate and normalize a server config; emit config JSON + service name.

    Prints the (possibly normalized) config as JSON on the first line and the
    derived service name on the second. Exits non-zero with ``ERROR:`` details
    on any validation failure.
    """
    missing_fields = [f for f in REQUIRED_FIELDS if f not in config or not config[f]]
    if missing_fields:
        _fail(f"ERROR: Missing required fields in config: {missing_fields}")

    # Reject control characters in fields that flow into shell/contract lines.
    for field in _CONTROLLED_STRING_FIELDS:
        value = config.get(field)
        if isinstance(value, str) and _has_control_chars(value):
            _fail(f"ERROR: {field} must not contain control characters")

    # Handle bedrock-agentcore specific URL formatting.
    auth_provider = config.get("auth_provider", "")
    if auth_provider == "bedrock-agentcore":
        path = config["path"]
        if not path.startswith("/"):
            path = "/" + path
        if not path.endswith("/"):
            path = path + "/"
        config["path"] = path

        proxy_url = config["proxy_pass_url"]
        if proxy_url.endswith("/mcp/"):
            proxy_url = proxy_url[:-5]
        elif proxy_url.endswith("/mcp"):
            proxy_url = proxy_url[:-4]
        if not proxy_url.endswith("/"):
            proxy_url = proxy_url + "/"
        config["proxy_pass_url"] = proxy_url

    errors: list[str] = []

    if not isinstance(config["server_name"], str) or not config["server_name"].strip():
        errors.append("server_name must be a non-empty string")

    if not isinstance(config["path"], str):
        errors.append("path must be a string")
    elif not config["path"].startswith("/"):
        errors.append('path must start with "/"')
    elif len(config["path"]) < 2:
        errors.append('path must be more than just "/"')

    if not isinstance(config["proxy_pass_url"], str):
        errors.append("proxy_pass_url must be a string")
    elif not (
        config["proxy_pass_url"].startswith("http://")
        or config["proxy_pass_url"].startswith("https://")
    ):
        errors.append("proxy_pass_url must start with http:// or https://")

    unknown_fields = set(config.keys()) - ALLOWED_FIELDS
    if unknown_fields:
        errors.append(
            f"Unknown fields not allowed by register_service tool spec: {sorted(unknown_fields)}"
        )

    if "description" in config and config["description"] is not None:
        if not isinstance(config["description"], str):
            errors.append("description must be a string")

    if "tags" in config and config["tags"] is not None:
        if not isinstance(config["tags"], list):
            errors.append("tags must be a list")
        elif not all(isinstance(tag, str) for tag in config["tags"]):
            errors.append("all tags must be strings")

    if "num_tools" in config and config["num_tools"] is not None:
        if not isinstance(config["num_tools"], int) or config["num_tools"] < 0:
            errors.append("num_tools must be a non-negative integer")

    if "license" in config and config["license"] is not None:
        if not isinstance(config["license"], str):
            errors.append("license must be a string")

    if errors:
        print("ERROR: Config validation failed:")
        for error in errors:
            print(f"  - {error}")
        sys.exit(1)

    service_name = config["path"].lstrip("/").rstrip("/")

    print(json.dumps(config))
    print(service_name)


def _get(config: dict[str, Any], field: str, omit_falsy: bool) -> None:
    """Print a single config field as a shell-consumable string.

    Container values are emitted as JSON. With ``omit_falsy`` an empty container
    prints nothing (used for optional headers); otherwise an empty container
    still prints its JSON form (e.g. ``[]`` for tags).
    """
    value = config.get(field)
    if isinstance(value, dict | list):
        print("" if (omit_falsy and not value) else json.dumps(value))
    elif value is None:
        print("")
    else:
        print(value)


def _format_health(output: str, service_filter: str) -> None:
    """Render the healthcheck JSON found in ``output`` as human-readable text."""
    json_start = output.find("{")
    if json_start == -1:
        print("No JSON found in output")
        sys.exit(1)

    brace_count = 0
    json_end = json_start
    for i, char in enumerate(output[json_start:], json_start):
        if char == "{":
            brace_count += 1
        elif char == "}":
            brace_count -= 1
            if brace_count == 0:
                json_end = i + 1
                break

    json_text = output[json_start:json_end]
    try:
        data = json.loads(json_text)
    except json.JSONDecodeError as exc:
        print(f"Error parsing JSON: {exc}")
        print("Raw output:")
        print(output)
        sys.exit(1)

    health_data = data["structuredContent"] if "structuredContent" in data else data

    current_time = datetime.now(UTC)

    print("Health Check Results:")
    print("=" * 50)

    for service_path, info in health_data.items():
        if service_filter and service_filter not in service_path:
            continue

        status = info.get("status", "unknown")
        last_checked = info.get("last_checked_iso", "")
        num_tools = info.get("num_tools", 0)

        if last_checked:
            try:
                check_time = datetime.fromisoformat(last_checked.replace("Z", "+00:00"))
                time_diff = current_time - check_time
                seconds_ago = int(time_diff.total_seconds())
                time_str = f"{seconds_ago} seconds ago"
            except ValueError:
                time_str = "unknown time"
        else:
            time_str = "never checked"

        if status == "healthy":
            status_display = "✓ healthy"
        elif status == "unhealthy":
            status_display = "✗ unhealthy"
        elif "auth-expired" in status:
            status_display = "⚠ healthy-auth-expired"
        else:
            status_display = f"? {status}"

        print(f"Service: {service_path}")
        print(f"  Status: {status_display}")
        print(f"  Last checked: {time_str}")
        print(f"  Tools available: {num_tools}")
        print()


def main(argv: list[str]) -> None:
    """Dispatch a data-only subcommand. See module docstring for the contract."""
    if len(argv) < 2:
        _fail("ERROR: usage: _service_config.py {validate|get|format-health} [args]")

    command = argv[1]

    if command == "validate":
        _validate(_read_stdin_json())
    elif command == "get":
        rest = argv[2:]
        omit_falsy = False
        if rest and rest[0] == "--omit-falsy":
            omit_falsy = True
            rest = rest[1:]
        if not rest:
            _fail("ERROR: get requires a field name")
        _get(_read_stdin_json(), rest[0], omit_falsy)
    elif command == "format-health":
        service_filter = argv[2] if len(argv) > 2 else ""
        _format_health(sys.stdin.read(), service_filter)
    else:
        _fail(f"ERROR: unknown command: {command}")


if __name__ == "__main__":
    main(sys.argv)
