"""The Lua router's MCP 2026-07-28 constants match registry/common/mcp_protocol.py.

nginx Lua cannot import Python, so docker/lua/virtual_router.lua keeps its own
copy of these values. This test fails when the two copies drift apart.
"""

import re
from pathlib import Path

from registry.common import mcp_protocol

ROUTER: Path = Path(__file__).resolve().parents[2] / "docker" / "lua" / "virtual_router.lua"


def _lua_constant(
    name: str,
) -> str:
    match = re.search(rf"^local {name} = (.+)$", ROUTER.read_text(), re.M)
    assert match, f"{name} is not defined in {ROUTER.name}"
    return match.group(1).strip().strip('"')


def test_version_and_meta_keys_match():
    assert _lua_constant("MODERN_PROTOCOL_VERSION") == mcp_protocol.MODERN_PROTOCOL_VERSION
    assert _lua_constant("META_PROTOCOL_VERSION_KEY") == mcp_protocol.META_PROTOCOL_VERSION_KEY
    assert _lua_constant("META_CLIENT_INFO_KEY") == mcp_protocol.META_CLIENT_INFO_KEY
    assert (
        _lua_constant("META_CLIENT_CAPABILITIES_KEY") == mcp_protocol.META_CLIENT_CAPABILITIES_KEY
    )


def test_error_codes_match():
    assert int(_lua_constant("JSONRPC_HEADER_MISMATCH")) == mcp_protocol.HEADER_MISMATCH
    assert (
        int(_lua_constant("JSONRPC_UNSUPPORTED_PROTOCOL_VERSION"))
        == mcp_protocol.UNSUPPORTED_PROTOCOL_VERSION
    )


def test_cacheable_methods_and_no_reuse_ttl_match():
    text = ROUTER.read_text()
    block = re.search(r"local CACHEABLE_RESULT_METHODS = \{(.*?)\n\}", text, re.S)
    assert block, "CACHEABLE_RESULT_METHODS is not defined in the router"
    lua_methods = set(re.findall(r'\["([^"]+)"\] = true', block.group(1)))

    assert lua_methods == set(mcp_protocol.CACHEABLE_RESULT_METHODS)
    assert int(_lua_constant("NO_REUSE_TTL_MS")) == mcp_protocol.NO_REUSE_TTL_MS
