"""Matching rules for one server_access entry, shared by the gateway and the registry.

The auth server's ``validate_server_tool_access`` decides every MCP call with
these functions, and the tool access listing (issue #2102) uses the same ones
to report which groups can call a tool. Keeping one copy means the listing
cannot claim access the gateway refuses, or the reverse.

This module must stay free of ``mcp`` imports: the auth server image does not
ship the MCP SDK.
"""

from __future__ import annotations

from typing import Any

# A tools or methods list containing either value means "every tool" or
# "every MCP method" on the rule's server.
WILDCARD_VALUES: frozenset[str] = frozenset(("all", "*"))

# Only "*" is a cross-server wildcard at the gateway. The registry's own
# resolver also accepts "all"; the two differ on purpose (see access_resolver).
SERVER_WILDCARD: str = "*"


def normalize_server_name(
    name: str | None,
) -> str | None:
    """Strip leading and trailing slashes so '/x/' and 'x' compare equal.

    Args:
        name: A server name or path from a rule or a request.

    Returns:
        The name without surrounding slashes, or the input when it is empty.
    """
    return name.strip("/") if name else name


def server_names_match(
    rule_server: str | None,
    server_name: str | None,
) -> bool:
    """Return True when a rule's server names the requested server.

    Args:
        rule_server: The ``server`` value of a server_access rule.
        server_name: The server being accessed, as a path or bare name.

    Returns:
        True for an exact match after slash normalization, or when the rule
        names the cross-server wildcard "*".
    """
    normalized = normalize_server_name(rule_server)
    if normalized == SERVER_WILDCARD:
        return True
    return normalized == normalize_server_name(server_name)


def normalize_name_list(
    value: Any,
) -> list[str]:
    """Return a rule's tools or methods value as a list of strings.

    A bare string is read as a one-item list, the way the registry resolver
    already reads it, so a string is compared by equality and never by
    substring. A missing or malformed value grants nothing.

    Args:
        value: The raw ``tools`` or ``methods`` value from a rule.

    Returns:
        A list of names. Empty for None or an unsupported type.
    """
    if isinstance(value, str):
        return [value]
    if isinstance(value, list):
        return [item for item in value if isinstance(item, str)]
    return []


def has_wildcard(
    names: list[str],
) -> bool:
    """Return True when a tools or methods list holds "all" or "*"."""
    return any(name in WILDCARD_VALUES for name in names)


def tool_call_allowed(
    allowed_tools: list[str],
    tool_name: str,
) -> bool:
    """Return True when a rule's tools list allows tools/call for tool_name.

    The gateway does not consult ``methods`` for tools/call: a rule allows a
    tool when it names the tool or holds a tools wildcard.

    Args:
        allowed_tools: The rule's tools, already passed through normalize_name_list.
        tool_name: The tool being called.

    Returns:
        True if the tool is allowed.
    """
    return tool_name in allowed_tools or has_wildcard(allowed_tools)


def rule_grants_tool_call(
    rule: dict[str, Any],
    server_name: str,
    tool_name: str,
) -> bool:
    """Return True when one server_access rule lets a caller call this tool.

    Args:
        rule: A flattened server_access rule with ``server`` and ``tools``.
        server_name: The server key the gateway checks (path without slashes).
        tool_name: The tool name the client sends in tools/call.

    Returns:
        True if this rule alone allows the call.
    """
    if not server_names_match(rule.get("server"), server_name):
        return False
    return tool_call_allowed(normalize_name_list(rule.get("tools")), tool_name)


def rule_grants_mcp_method(
    rule: dict[str, Any],
    server_name: str,
    method: str,
) -> bool:
    """Return True when one rule allows an MCP method other than tools/call.

    Mirrors the gateway's non-HTTP branch: the method is in ``methods``, or
    ``methods`` holds a wildcard, or (for backward compatibility) the method is
    named in ``tools``. A tools wildcard never grants a method.

    Args:
        rule: A flattened server_access rule.
        server_name: The server key the gateway checks.
        method: An MCP method such as "initialize".

    Returns:
        True if this rule alone allows the method.
    """
    if not server_names_match(rule.get("server"), server_name):
        return False
    methods = normalize_name_list(rule.get("methods"))
    if method in methods or has_wildcard(methods):
        return True
    return method in normalize_name_list(rule.get("tools"))
