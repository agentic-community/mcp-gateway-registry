-- Tests for the MCP 2026-07-28 header check in capture_body.lua.
--
-- The auth server authorizes the request body, while a 2026-07-28 backend may
-- route on Mcp-Method and Mcp-Name, so capture_body.lua must refuse a request
-- whose headers name a different operation than its body. Legacy requests,
-- which carry none of these headers, must pass unchanged.
--
-- Run from the repo root with OpenResty's luajit (provides lua-cjson):
--   docker run --rm -v "$PWD":/repo -w /repo \
--     -e LUA_CPATH="/usr/local/openresty/lualib/?.so;;" \
--     openresty/openresty:alpine /usr/local/openresty/luajit/bin/luajit \
--     tests/lua/test_capture_body.lua

_G._CB_TEST = true

local cjson = require("cjson")

local failures = 0
local function check(cond, msg)
    if cond then
        print("  ok   - " .. msg)
    else
        failures = failures + 1
        print("  FAIL - " .. msg)
    end
end

-- Only the base64 values the tests use; plain luajit has no ngx.decode_base64.
local BASE64_VALUES = { ["c2VhcmNo"] = "search" }
_G.ngx = {
    decode_base64 = function(value) return BASE64_VALUES[value] end,
}

local cb = dofile("docker/lua/capture_body.lua")

local MODERN = "2026-07-28"
local META = { ["io.modelcontextprotocol/protocolVersion"] = MODERN }


local function _body(method, params, id)
    return cjson.encode({ jsonrpc = "2.0", id = id or 7, method = method, params = params })
end


local function _call_body(tool_name)
    return _body("tools/call", { name = tool_name, arguments = {}, _meta = META })
end


local function _modern_headers(method, name)
    return { version = MODERN, method = method, name = name }
end


print("legacy requests pass unchanged")
do
    local body = _body("tools/call", { name = "search", arguments = {} })
    check(cb._header_mismatch(body, {}) == nil, "no MCP headers at all")
    check(cb._header_mismatch(body, { version = "2025-11-25" }) == nil,
        "legacy MCP-Protocol-Version header")
    check(cb._header_mismatch("not json", { version = "2025-11-25" }) == nil,
        "legacy header with a body that is not JSON")
    check(cb._header_mismatch('[{"jsonrpc":"2.0","id":1,"method":"ping"}]', {}) == nil,
        "legacy batch without MCP headers")
    check(cb._header_mismatch('{"jsonrpc":"2.0","id":3,"result":{}}', {}) == nil,
        "legacy client response body")
end

print("matching 2026-07-28 requests pass")
do
    check(cb._header_mismatch(_call_body("search"), _modern_headers("tools/call", "search")) == nil,
        "tools/call with matching Mcp-Name")
    check(cb._header_mismatch(_call_body("search"),
        _modern_headers("tools/call", "=?base64?c2VhcmNo?=")) == nil,
        "base64-encoded Mcp-Name is decoded before comparing")
    check(cb._header_mismatch(_body("tools/list", { _meta = META }),
        _modern_headers("tools/list", nil)) == nil, "tools/list needs no Mcp-Name")
    local read = _body("resources/read", { uri = "file:///a", _meta = META })
    check(cb._header_mismatch(read, _modern_headers("resources/read", "file:///a")) == nil,
        "resources/read compares Mcp-Name with params.uri")
end

print("mismatched 2026-07-28 requests are rejected")
do
    local message, id = cb._header_mismatch(_call_body("delete_all"),
        _modern_headers("tools/call", "search"))
    check(message == "Mcp-Name does not match params.name", "Mcp-Name names another tool")
    check(id == 7, "the error carries the request id")

    check(cb._header_mismatch(_call_body("search"), _modern_headers("tools/list", "search"))
        == "Mcp-Method does not match the body method", "Mcp-Method names another method")
    check(cb._header_mismatch(_call_body("search"), _modern_headers(nil, "search"))
        == "Mcp-Method does not match the body method", "modern request without Mcp-Method")
    check(cb._header_mismatch(_call_body("search"), _modern_headers("tools/call", nil))
        == "Mcp-Name does not match params.name", "modern tools/call without Mcp-Name")
    check(cb._header_mismatch(_body("tools/list", { _meta = META }),
        _modern_headers("tools/list", "search"))
        == "Mcp-Name is not allowed for method tools/list", "Mcp-Name on a method without a name")
end

print("routing headers are checked even without a 2026 version header")
do
    local legacy_call = _body("tools/call", { name = "delete_all", arguments = {} })
    check(cb._header_mismatch(legacy_call, { method = "tools/call", name = "search" })
        == "Mcp-Name does not match params.name", "Mcp-Name sent with a legacy body")
    check(cb._header_mismatch(legacy_call, { version = "2025-11-25", method = "tools/list" })
        == "Mcp-Method does not match the body method", "Mcp-Method sent with a legacy header")
end

print("protocol version must agree between header and body")
do
    local legacy_meta = { ["io.modelcontextprotocol/protocolVersion"] = "2025-11-25" }
    local body = _body("tools/list", { _meta = legacy_meta })
    check(cb._header_mismatch(body, _modern_headers("tools/list", nil))
        == "MCP-Protocol-Version does not match params._meta", "header 2026, body 2025")
    check(cb._header_mismatch(_body("tools/list", {}), _modern_headers("tools/list", nil))
        == "MCP-Protocol-Version does not match params._meta", "header 2026, body without _meta")
    check(cb._header_mismatch(_body("tools/list", { _meta = META }), { method = "tools/list" })
        == "MCP-Protocol-Version does not match params._meta", "body 2026, header missing")
end

print("modern headers need a single JSON-RPC request body")
do
    local message, id = cb._header_mismatch('[{"jsonrpc":"2.0","id":1,"method":"tools/list"}]',
        _modern_headers("tools/list", nil))
    check(message == "MCP routing headers need a single JSON-RPC request body", "batch body")
    check(id == nil, "no request id for a body that is not a request")
    check(cb._header_mismatch("{not json", { method = "tools/call" }) ~= nil,
        "Mcp-Method with a body that is not JSON")
end

if failures > 0 then
    print(failures .. " test(s) failed")
    os.exit(1)
end
print("all capture_body tests passed")
