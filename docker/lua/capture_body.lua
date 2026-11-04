-- capture_body.lua: Read request body and encode it in X-Body header for auth_request
--
-- It also rejects an MCP 2026-07-28 request whose routing headers disagree
-- with its body. The auth server authorizes the body, while a 2026-07-28
-- backend may route on the Mcp-Method and Mcp-Name headers, so the two must
-- name the same operation before the request goes any further.
local cjson = require "cjson"

local META_PROTOCOL_VERSION_KEY = "io.modelcontextprotocol/protocolVersion"
local JSONRPC_HEADER_MISMATCH = -32020
local BASE64_SENTINEL_PREFIX = "=?base64?"
local BASE64_SENTINEL_SUFFIX = "?="

-- The params field that Mcp-Name mirrors, per method (MCP 2026-07-28).
local MCP_NAME_SOURCE = {
    ["tools/call"] = "name",
    ["resources/read"] = "uri",
    ["prompts/get"] = "name",
}


-- Undo the =?base64?...?= sentinel encoding of an Mcp-Name header value.
local function _decode_mcp_header_value(value)
    if type(value) ~= "string" then
        return value
    end
    local prefix_len, suffix_len = #BASE64_SENTINEL_PREFIX, #BASE64_SENTINEL_SUFFIX
    if #value <= prefix_len + suffix_len
        or value:sub(1, prefix_len) ~= BASE64_SENTINEL_PREFIX
        or value:sub(-suffix_len) ~= BASE64_SENTINEL_SUFFIX then
        return value
    end
    return ngx.decode_base64(value:sub(prefix_len + 1, -suffix_len - 1)) or value
end


-- True when a protocol version string names the 2026 era or later.
local function _is_modern_version(version)
    return type(version) == "string" and version:sub(1, 5) >= "2026-"
end


-- Compare MCP-Protocol-Version with params._meta. Returns a message or nil.
local function _version_mismatch(header_version, params)
    local meta = type(params) == "table" and params._meta or nil
    local body_version = type(meta) == "table" and meta[META_PROTOCOL_VERSION_KEY] or nil
    if type(body_version) ~= "string" then
        body_version = nil
    end
    if not _is_modern_version(header_version) and not _is_modern_version(body_version) then
        return nil
    end
    if header_version ~= body_version then
        return "MCP-Protocol-Version does not match params._meta"
    end
    return nil
end


-- Compare Mcp-Method and Mcp-Name with the body. Returns a message or nil.
local function _routing_mismatch(headers, method, params, modern)
    if headers.method ~= method and (headers.method ~= nil or modern) then
        return "Mcp-Method does not match the body method"
    end
    local name_field = MCP_NAME_SOURCE[method]
    if not name_field then
        if headers.name ~= nil and headers.name ~= "" then
            return "Mcp-Name is not allowed for method " .. method
        end
        return nil
    end
    if headers.name == nil and not modern then
        return nil
    end
    local body_name = type(params) == "table" and params[name_field] or nil
    if _decode_mcp_header_value(headers.name) ~= body_name then
        return "Mcp-Name does not match params." .. name_field
    end
    return nil
end


-- Check the 2026-07-28 headers against the body.
-- headers: { version = ..., method = ..., name = ... } (nil when absent).
-- Returns nil when they agree (or none apply), else (message, request_id).
local function _header_mismatch(body_data, headers)
    local modern = _is_modern_version(headers.version)
    local ok, body = pcall(cjson.decode, body_data)
    local is_request = ok and type(body) == "table" and type(body.method) == "string"
    if not is_request then
        if modern or headers.method ~= nil or headers.name ~= nil then
            return "MCP routing headers need a single JSON-RPC request body", nil
        end
        return nil
    end
    local message = _version_mismatch(headers.version, body.params)
    if not message then
        local meta = type(body.params) == "table" and body.params._meta or nil
        local body_version = type(meta) == "table" and meta[META_PROTOCOL_VERSION_KEY] or nil
        local body_modern = _is_modern_version(body_version)
        message = _routing_mismatch(headers, body.method, body.params, modern or body_modern)
    end
    if message then
        return message, body.id
    end
    return nil
end


-- Answer 400 with a JSON-RPC HeaderMismatch error and stop the request.
local function _reject_header_mismatch(message, request_id)
    ngx.log(ngx.WARN, "Rejecting MCP request: header mismatch: " .. message)
    ngx.status = ngx.HTTP_BAD_REQUEST
    ngx.header["Content-Type"] = "application/json"
    ngx.say(cjson.encode({
        jsonrpc = "2.0",
        id = request_id or cjson.null,
        error = { code = JSONRPC_HEADER_MISMATCH, message = "Header mismatch: " .. message },
    }))
    return ngx.exit(ngx.HTTP_OK)
end


if _G._CB_TEST then
    return {
        _decode_mcp_header_value = _decode_mcp_header_value,
        _header_mismatch = _header_mismatch,
    }
end

-- Strip any client-supplied copies of the headers this script owns so a
-- caller cannot forge the scope-decision inputs the auth server trusts.
ngx.req.clear_header("X-Body")
ngx.req.clear_header("X-Body-Uninspectable")

-- Read the request body
ngx.req.read_body()
local body_data = ngx.req.get_body_data()

if body_data then
    local mismatch, request_id = _header_mismatch(body_data, {
        version = ngx.var.http_mcp_protocol_version,
        method = ngx.var.http_mcp_method,
        name = ngx.var.http_mcp_name,
    })
    if mismatch then
        return _reject_header_mismatch(mismatch, request_id)
    end

    -- Strip newlines to prevent breaking HTTP header format
    -- (JSON whitespace is insignificant per RFC 8259, so this is safe)
    local clean_body = body_data:gsub("[\r\n]+", " ")
    -- Set the X-Body header with the cleaned body data
    ngx.req.set_header("X-Body", clean_body)
    ngx.log(ngx.INFO, "Captured request body (" .. string.len(body_data) .. " bytes) for auth validation")
else
    -- get_body_data() returns nil in two cases:
    --   1. There is genuinely no request body (e.g. an empty POST).
    --   2. The body was larger than client_body_buffer_size and nginx spilled
    --      it to a temp file instead of keeping it in memory.
    --
    -- Case 2 is a scope-check bypass risk: the auth server would see no X-Body,
    -- default the method to the unprivileged "initialize", and authorize on
    -- that -- while the full (potentially privileged) body is still forwarded
    -- upstream. Detect the spill-to-file case and flag it so /validate can fail
    -- closed rather than authorizing an uninspectable body. (The auth-server
    -- mcp-proxy hop also re-authorizes the exact forwarded body; this header is
    -- defense-in-depth at the edge.) /validate refuses such a body, so the
    -- header check above does not need to run on it.
    local body_file = ngx.req.get_body_file()
    if body_file then
        ngx.req.set_header("X-Body-Uninspectable", "1")
        ngx.log(ngx.WARN,
            "Request body spilled to temp file (" .. tostring(body_file) ..
            "); marking uninspectable for fail-closed scope validation")
    else
        ngx.log(ngx.INFO, "No request body found")
    end
end
