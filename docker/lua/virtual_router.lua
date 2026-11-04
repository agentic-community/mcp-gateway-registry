-- virtual_router.lua: JSON-RPC router for Virtual MCP Servers
-- Routes tools/list, tools/call, resources/list, resources/read,
-- prompts/list, prompts/get, ping, and initialize requests to the correct backend.
-- Implements per-client session management with two-tier cache:
--   L1: ngx.shared.virtual_server_map (30s TTL, per-worker fast path)
--   L2: MongoDB via /_internal/sessions/ FastAPI endpoints
local cjson = require "cjson"

-- Ensure empty Lua tables serialize as JSON arrays [] not objects {}
local empty_array_mt = cjson.empty_array_mt

-- Extract JSON from an SSE-formatted response body.
-- SSE format: "event: message\ndata: {json}\n\n"
-- If the body is already raw JSON, return it as-is.
local function _parse_sse_body(body)
    if not body or body == "" then
        return nil
    end
    -- If it starts with '{' or '[', it's already raw JSON
    local first_char = string.sub(body, 1, 1)
    if first_char == "{" or first_char == "[" then
        return body
    end
    -- Extract the last "data: " line (SSE format)
    local json_data = nil
    for line in string.gmatch(body, "[^\r\n]+") do
        local data = string.match(line, "^data:%s*(.+)")
        if data then
            json_data = data
        end
    end
    return json_data
end


-- Force a table to serialize as a JSON array (handles empty tables -> [] not {})
local function _as_json_array(t)
    if type(t) ~= "table" then
        if cjson.empty_array then return cjson.empty_array end
        return setmetatable({}, empty_array_mt)
    end
    if next(t) == nil then
        if cjson.empty_array then return cjson.empty_array end
    end
    return setmetatable(t, empty_array_mt)
end

local _M = {}

-- Shared dict for L1 session cache and mapping cache
local session_cache = ngx.shared.virtual_server_map

-- Cache TTL constants
local MAPPING_CACHE_TTL = 10
local SESSION_CACHE_TTL = 30
local ENRICHED_CACHE_TTL = 60
local STATELESS_CACHE_VALUE = "1"


-- Supported MCP protocol versions (newest first for negotiation)
local SUPPORTED_PROTOCOL_VERSIONS = {
    ["2025-11-25"] = true,
    ["2025-06-18"] = true,
    ["2025-03-26"] = true,
    ["2024-11-05"] = true,
}
local LATEST_PROTOCOL_VERSION = "2025-11-25"


-- Resolve the authenticated user identity for the current request.
-- Used both when creating sessions (_handle_initialize) and when enforcing
-- session ownership, so the values compared are always the same.
-- Returns nil when no real identity is present: nginx auth_request_set yields
-- an EMPTY STRING (not nil) for an absent upstream X-User header, and in Lua
-- "" is truthy, so a plain `a or b or "anonymous"` chain would lock onto the
-- empty string and never fall through. We normalize "" to nil at each step so
-- callers can fail closed instead of treating an unauthenticated/identity-less
-- request as a shared "anonymous" owner whose sessions everyone could hijack.
local function _auth_user_id()
    local u = ngx.var.auth_user
    if u and u ~= "" then
        return u
    end
    u = ngx.var.auth_username
    if u and u ~= "" then
        return u
    end
    return nil
end


-- Forward the validated caller identity to backend subrequests, fail-closed.
--
-- auth_request_set variables ($auth_user, $auth_username) are scoped to the
-- parent request and do NOT propagate into ngx.location.capture subrequests
-- via proxy_set_header, so we copy them onto request headers here; the
-- _vs_backend_* locations then forward them to the upstream MCP server via
-- $http_x_user/$http_x_username, enabling backends to enforce write-gates and
-- record audit attribution.
--
-- SECURITY: $http_x_user is a CLIENT-controllable request header. The backend
-- value must be the gateway-validated identity or nothing -- never a value the
-- caller supplied. When auth_user is empty (e.g. an M2M / client-credentials
-- token that authenticates with a client_id but no user), we must CLEAR any
-- inbound X-User rather than leave it intact, otherwise a caller could spoof
-- X-User to the backend. This mirrors the direct-server path, which forwards
-- `proxy_set_header X-User $auth_user;` unconditionally so a client header can
-- never survive. Always overwrite or clear; never pass through.
local function _forward_identity_headers()
    local auth_user = ngx.var.auth_user
    local auth_username = ngx.var.auth_username
    if auth_user and auth_user ~= "" then
        ngx.req.set_header("X-User", auth_user)
    else
        ngx.req.clear_header("X-User")
    end
    if auth_username and auth_username ~= "" then
        ngx.req.set_header("X-Username", auth_username)
    else
        ngx.req.clear_header("X-Username")
    end
end


-- JSON Schema keywords whose value must serialize as an array.
local SCHEMA_ARRAY_KEYS = {
    required = true, enum = true, allOf = true, anyOf = true,
    oneOf = true, examples = true, prefixItems = true,
}

-- Keywords holding a map of caller-chosen names to subschemas. A member named
-- "required" there is a property name, not the keyword, so descend into the
-- members without matching their names against SCHEMA_ARRAY_KEYS.
local SCHEMA_MAP_KEYS = {
    properties = true, patternProperties = true,
    definitions = true, ["$defs"] = true,
}

-- cjson decodes an empty JSON array to a bare table, which re-encodes as {}.
-- Tag those tables so the keywords above survive the round trip as [].
local function _tag_empty_schema_arrays(node, depth)
    if type(node) ~= "table" or depth > 16 then
        return
    end
    for key, value in pairs(node) do
        if type(value) == "table" then
            if SCHEMA_MAP_KEYS[key] then
                for _, subschema in pairs(value) do
                    _tag_empty_schema_arrays(subschema, depth + 1)
                end
            elseif SCHEMA_ARRAY_KEYS[key] and next(value) == nil then
                setmetatable(value, empty_array_mt)
            else
                _tag_empty_schema_arrays(value, depth + 1)
            end
        end
    end
end


-- Ensure inputSchema has "type": "object" as required by MCP spec
local function _ensure_mcp_schema(schema)
    if not schema or type(schema) ~= "table" then
        return { type = "object", properties = {} }
    end
    _tag_empty_schema_arrays(schema, 0)
    if schema.type == "object" then
        return schema
    end
    if not schema.type then
        schema.type = "object"
        return schema
    end
    -- Non-object type: wrap it
    return { type = "object", properties = { value = schema } }
end


-- Read and cache virtual server mapping from JSON file. The cache is keyed by
-- the rendered config's generation ($vs_config_generation): lua_shared_dict
-- survives an nginx reload, and a mapping cached under the previous config can
-- name internal locations the new config no longer defines.
local function _get_mapping(server_id)
    local cache_key = "mapping:" .. (ngx.var.vs_config_generation or "") .. ":" .. server_id
    local cached = session_cache:get(cache_key)
    if cached then
        local ok, mapping = pcall(cjson.decode, cached)
        if ok then
            return mapping
        end
        ngx.log(ngx.WARN, "Failed to decode cached mapping for server_id=", server_id)
    end

    -- Read from file
    local path = "/etc/nginx/lua/virtual_mappings/" .. server_id .. ".json"
    local f, err = io.open(path, "r")
    if not f then
        ngx.log(ngx.ERR, "Could not open mapping file: ", path, " error: ", tostring(err))
        return nil
    end

    local content = f:read("*a")
    f:close()

    local ok, mapping = pcall(cjson.decode, content)
    if not ok then
        ngx.log(ngx.ERR, "Failed to parse mapping JSON for server_id=", server_id)
        return nil
    end

    -- Cache in shared dict (TTL 10 seconds to reduce stale data after reload)
    session_cache:set(cache_key, content, MAPPING_CACHE_TTL)

    return mapping
end


-- Build a JSON-RPC error response
local function _jsonrpc_error(id, code, message)
    return cjson.encode({
        jsonrpc = "2.0",
        id = id,
        error = {
            code = code,
            message = message,
        },
    })
end


-- Build a JSON-RPC success response
local function _jsonrpc_result(id, result)
    return cjson.encode({
        jsonrpc = "2.0",
        id = id,
        result = result,
    })
end


-- Check if user scopes satisfy required scopes
local function _has_scopes(user_scopes_str, required_scopes)
    if not required_scopes or #required_scopes == 0 then
        return true
    end
    if not user_scopes_str or user_scopes_str == "" then
        return false
    end

    -- Parse space-separated scopes into a set
    local user_scopes = {}
    for scope in string.gmatch(user_scopes_str, "%S+") do
        user_scopes[scope] = true
    end

    -- Check all required scopes are present
    for _, required in ipairs(required_scopes) do
        if not user_scopes[required] then
            return false
        end
    end
    return true
end


-- Read a capture response header regardless of the upstream's header casing.
local function _response_header(res, name)
    if not res or not res.header then
        return nil
    end
    return res.header[name] or res.header[string.lower(name)]
end


-- Outcomes of the backend-bound authorization subrequest.
-- MCP 2026-07-28 protocol constants, shared by the client-facing path (route())
-- and the backend-facing path (_call_backend).
local MODERN_PROTOCOL_VERSION = "2026-07-28"
local META_PROTOCOL_VERSION_KEY = "io.modelcontextprotocol/protocolVersion"
local META_CLIENT_INFO_KEY = "io.modelcontextprotocol/clientInfo"
local META_CLIENT_CAPABILITIES_KEY = "io.modelcontextprotocol/clientCapabilities"
local META_SERVER_INFO_KEY = "io.modelcontextprotocol/serverInfo"
local JSONRPC_HEADER_MISMATCH = -32020
local JSONRPC_UNSUPPORTED_PROTOCOL_VERSION = -32022
local BASE64_SENTINEL_PREFIX = "=?base64?"
local BASE64_SENTINEL_SUFFIX = "?="
-- Methods whose target name (params.name or params.uri) is mirrored in Mcp-Name.
local MCP_NAME_SOURCE = {
    ["tools/call"] = "name",
    ["prompts/get"] = "name",
    ["resources/read"] = "uri",
}

local AUTH_OK = "ok"
local AUTH_DENIED = "denied"   -- the gateway refused the backing grant (401/403)
local AUTH_FAILED = "failed"   -- the check itself failed: 429/5xx/timeout/no token
-- The grant check passed but the backend's session could not be established.
local SESSION_FAILED = "session_failed"
-- A token minted before a cold initialize is reused only if younger than this.
-- It stays under auth-server's INTERNAL_TOKEN_TTL_SECONDS floor (5s), so the
-- reused token is live whatever TTL is configured; a slower initialize (vend,
-- refresh, upstream handshake) re-authorizes instead of sending an expired one.
local TOKEN_REUSE_SECONDS = 4

-- The backend subrequest bypasses nginx auth_request, so explicitly validate
-- the exact rewritten JSON-RPC request at a dedicated backend-bound location.
-- The validated internal token is required even for plain backends; their
-- proxy location must clear it before forwarding upstream.
--
-- Only a 401/403 is a grant decision. A rate limit, an auth-server error or
-- timeout, or a 200 without a token is a failure of that one backend's check:
-- reporting it as "access denied" would blank healthy siblings and tell the
-- client to stop retrying.
local function _authorize_backend(backend_location, request_body)
    local auth_location, substitutions = backend_location:gsub("^/_vs_backend_", "/_vs_auth_", 1)
    if substitutions ~= 1 then
        return AUTH_FAILED
    end
    ngx.req.set_header("X-Body", request_body)
    ngx.req.clear_header("X-Body-Uninspectable")
    ngx.req.clear_header("X-Internal-Token")
    local res = ngx.location.capture(auth_location, {
        method = ngx.HTTP_GET,
    })
    local status = res and res.status or "nil"
    if res and (res.status == 401 or res.status == 403) then
        ngx.log(ngx.WARN, "Backend authorization denied for ", backend_location,
            " status=", status)
        return AUTH_DENIED
    end
    local token = _response_header(res, "X-Internal-Token")
    if not res or res.status ~= 200 or type(token) ~= "string" or token == "" then
        ngx.log(ngx.ERR, "Backend authorization check failed for ", backend_location,
            " status=", status)
        return AUTH_FAILED
    end
    ngx.req.set_header("X-Internal-Token", token)
    return AUTH_OK
end

-- Initialize a backend and return its session ID (nil for stateless) and success.
-- Only a configured egress backend may produce a pre-consent handshake.
local function _initialize_backend(backend_location, allow_pre_consent)
    local init_body = cjson.encode({
        jsonrpc = "2.0",
        id = "init-" .. (ngx.var.request_id or "0"),
        method = "initialize",
        params = {
            protocolVersion = LATEST_PROTOCOL_VERSION,
            capabilities = {},
            clientInfo = {
                name = "mcp-gateway-virtual-router",
                version = "1.0.0",
            },
        },
    })

    ngx.req.set_header("Mcp-Session-Id", "")
    ngx.req.set_header("Accept", "application/json, text/event-stream")

    local auth = _authorize_backend(backend_location, init_body)
    if auth ~= AUTH_OK then
        return nil, false, auth == AUTH_DENIED
    end
    local res = ngx.location.capture(backend_location, {
        method = ngx.HTTP_POST,
        body = init_body,
    })

    -- Only the gateway's own grant check (_authorize_backend above) is an
    -- authorization decision. A 401/403 from the backend hop comes from the
    -- backend (or its revoked upstream credential) and is a failure of that one
    -- backend, never a denial that blocks every sibling.
    if not res or res.status ~= 200 or res.truncated then
        ngx.log(ngx.ERR, "Backend initialize failed for ", backend_location,
            " status=", res and res.status or "nil")
        return nil, false
    end

    -- HTTP 200 can still carry a JSON-RPC error (including an OAuth consent
    -- failure); only a genuine initialize result establishes stateless state.
    local json_body = _parse_sse_body(res.body)
    local ok, data = pcall(cjson.decode, json_body or "")
    if not ok or type(data) ~= "table" or type(data.result) ~= "table" then
        ngx.log(ngx.ERR, "Backend initialize returned no result for ", backend_location)
        return nil, false
    end

    -- The gateway can answer initialize locally while the caller has not yet
    -- consented to egress credentials. It has not established a backend session.
    -- Only the egress broker may say so; the marker from a plain backend is a
    -- failed initialize.
    local initialized = _response_header(res, "X-MCP-Backend-Initialized")
    if initialized == "0" then
        if allow_pre_consent then
            return nil, false, false, true
        end
        return nil, false
    end

    local session_id = res.header and
        (res.header["Mcp-Session-Id"] or res.header["mcp-session-id"])
    if session_id == "" then
        session_id = nil
    end
    return session_id, true
end


-- Resolve backend state from L1, L2, or initialize. Returns session ID, success,
-- authorization denial, pre-consent initialization, and whether an initialize
-- ran (its own authorization replaced the caller's backend token). Successful
-- stateless initialize returns (nil, true); pre-consent must not be persisted as
-- stateless. The client session was already validated against the owner.
local function _get_backend_session(client_session_id, backend_location, server_id, backend_version,
                                    allow_pre_consent)
    local version = type(backend_version) == "string" and backend_version or ""
    local session_key = client_session_id .. ":" .. backend_location
        .. (version ~= "" and (":" .. #version .. ":" .. version) or "")
    local owner = _auth_user_id()
    if not owner then
        return nil, false
    end
    local owner_key = ":" .. #owner .. ":" .. owner
    local cache_key = "bsess:" .. session_key .. owner_key
    local stateless_key = "bsess_stateless:" .. session_key .. owner_key
    local session_id = session_cache:get(cache_key)
    if session_id then
        return session_id, true
    end
    if session_cache:get(stateless_key) == STATELESS_CACHE_VALUE then
        return nil, true
    end
    local backend_path = ngx.escape_uri(client_session_id) .. ":" .. backend_location
        .. (version ~= "" and (":" .. #version .. ":" .. ngx.escape_uri(version)) or "")
    local res = ngx.location.capture(
        "/_internal/sessions/backend/" .. backend_path .. "?user_id=" .. ngx.escape_uri(owner), {
            method = ngx.HTTP_GET,
        })
    if res and res.status == 200 then
        local ok, data = pcall(cjson.decode, res.body)
        if ok and type(data) == "table" then
            if data.stateless == true and data.backend_session_id == cjson.null then
                session_cache:set(stateless_key, STATELESS_CACHE_VALUE, SESSION_CACHE_TTL)
                return nil, true
            end
            if data.stateless == false and type(data.backend_session_id) == "string"
                and data.backend_session_id ~= "" then
                session_cache:set(cache_key, data.backend_session_id, SESSION_CACHE_TTL)
                return data.backend_session_id, true
            end
        end
        return nil, false
    end
    if not res or res.status ~= 404 then
        return nil, false
    end

    ngx.log(ngx.INFO, "Initializing backend session for ", session_key)
    local initialized, denied, pre_consent
    session_id, initialized, denied, pre_consent = _initialize_backend(
        backend_location, allow_pre_consent)
    if not initialized then
        return nil, false, denied, pre_consent, true
    end

    local store_body = cjson.encode({
        backend_session_id = session_id or cjson.null,
        stateless = session_id == nil,
        client_session_id = client_session_id,
        user_id = owner,
        virtual_server_path = "/virtual/" .. server_id,
    })
    local stored = ngx.location.capture("/_internal/sessions/backend/" .. backend_path, {
        method = ngx.HTTP_PUT,
        body = store_body,
    })
    if not stored or stored.status ~= 200 then
        ngx.log(ngx.ERR, "Failed to persist backend session for ", backend_location)
        return nil, false
    end
    if session_id then
        session_cache:set(cache_key, session_id, SESSION_CACHE_TTL)
    else
        session_cache:set(stateless_key, STATELESS_CACHE_VALUE, SESSION_CACHE_TTL)
    end
    return session_id, true, false, false, true
end


-- Invalidate a backend session from both L1 and L2 caches
local function _invalidate_backend_session(client_session_id, backend_location, backend_version)
    local version = type(backend_version) == "string" and backend_version or ""
    local session_key = client_session_id .. ":" .. backend_location
        .. (version ~= "" and (":" .. #version .. ":" .. version) or "")
    local owner = _auth_user_id()
    if not owner then
        return
    end
    local owner_key = ":" .. #owner .. ":" .. owner
    session_cache:delete("bsess:" .. session_key .. owner_key)
    session_cache:delete("bsess_stateless:" .. session_key .. owner_key)

    -- Remove from L2, scoped to the authenticated owner (symmetric with the GET
    -- lookup). Escape the session-id segment in the subrequest URI for the same
    -- defense-in-depth reason as the GET/PUT paths (a no-op for valid vs-<hex>
    -- ids, which is all that reaches here past the route() allowlist).
    local backend_path = ngx.escape_uri(client_session_id) .. ":" .. backend_location
        .. (version ~= "" and (":" .. #version .. ":" .. ngx.escape_uri(version)) or "")
    local owner_qs = "?user_id=" .. ngx.escape_uri(owner)
    ngx.location.capture("/_internal/sessions/backend/" .. backend_path .. owner_qs, {
        method = ngx.HTTP_DELETE,
    })
end


-- Optional fields of a backend tool that the virtual tools/list passes through
-- unchanged. inputSchema and the name/description are handled separately.
local PASSTHROUGH_TOOL_FIELDS = { "title", "annotations", "outputSchema", "icons", "execution", "_meta" }


-- Copy the pass-through fields from one tool table to another. outputSchema
-- gets the same empty-array tagging as inputSchema, so a [] keeps its type
-- through cjson (including the per-user tools/list cache).
local function _copy_tool_fields(from, to)
    for _, field in ipairs(PASSTHROUGH_TOOL_FIELDS) do
        local value = from[field]
        if value ~= nil and value ~= cjson.null then
            if field == "outputSchema" and type(value) == "table" then
                _tag_empty_schema_arrays(value, 0)
            end
            to[field] = value
        end
    end
    return to
end


-- A mapped tool without a valid backend identity cannot be safely displayed.
-- The decision helper is defined later, before the router executes.
local _blocked_backend_tools


-- Collect unique backend locations from a mapping's tools array
local function _collect_backend_locations(mapping)
    local locations = {}
    local seen = {}

    if mapping.tools then
        for _, tool in ipairs(mapping.tools) do
            local loc = tool.backend_location
            if loc and not seen[loc] then
                seen[loc] = true
                locations[#locations + 1] = loc
            end
        end
    end

    return locations
end


-- Append mapping-file metadata for one backend when live discovery fails.
local function _append_mapping_tools_for_backend(enriched_tools, mapping, backend_location)
    if not mapping.tools then
        return
    end

    for _, tool in ipairs(mapping.tools) do
        if tool.backend_location == backend_location then
            enriched_tools[#enriched_tools + 1] = {
                name = tool.name,
                original_name = tool.original_name or tool.name,
                backend_server_path = tool.backend_server_path,
                block_server_path = tool.block_server_path,
                description = tool.description or "",
                inputSchema = _ensure_mcp_schema(tool.inputSchema),
                required_scopes = tool.required_scopes,
            }
        end
    end
end


-- Outcomes of one backend's discovery-list fetch. Every aggregator classifies
-- backends with the SAME function so tools, resources and prompts cannot
-- disagree about what an empty, unsupported, pending-consent or failed backend
-- means:
--   ok          -- a list result (possibly empty)
--   consent     -- the gateway answered for a backend whose caller has not
--                  connected an egress credential yet (empty list, marked by
--                  the broker's X-Egress-Consent-Required header)
--   unsupported -- the backend answered "method not found"
--   failed      -- transport/protocol failure, or a 401/403/error the backend
--                  itself returned; contributes nothing this time
--   unverified  -- the gateway's grant check could not run (429/5xx/timeout);
--                  contributes nothing and never falls back to mapping metadata
--   denied      -- the gateway's own grant refusal for the rewritten request
--                  (_authorize_backend); blocks the whole aggregate
local LIST_OK = "ok"
local LIST_CONSENT = "consent"
local LIST_UNSUPPORTED = "unsupported"
local LIST_FAILED = "failed"
local LIST_UNVERIFIED = "unverified"
local LIST_DENIED = "denied"
local JSONRPC_METHOD_NOT_FOUND = -32601


-- Classify a 200 JSON-RPC list response body. A JSON-RPC error is the
-- backend's answer, not the gateway's grant decision, so it never denies the
-- aggregate. The consent marker is honoured only from the egress broker
-- (``from_broker``): a plain backend proxies straight to its registrant.
local function _classify_list_body(res, result_key, from_broker)
    local json_body = _parse_sse_body(res.body)
    local ok, data = pcall(cjson.decode, json_body or "")
    if not ok or type(data) ~= "table" then
        return LIST_FAILED
    end
    if type(data.error) == "table" then
        if data.error.code == JSONRPC_METHOD_NOT_FOUND then
            return LIST_UNSUPPORTED
        end
        return LIST_FAILED
    end
    if type(data.result) ~= "table" or type(data.result[result_key]) ~= "table" then
        return LIST_FAILED
    end
    if from_broker and _response_header(res, "X-Egress-Consent-Required") then
        return LIST_CONSENT
    end
    local next_cursor = data.result.nextCursor
    if type(next_cursor) ~= "string" or next_cursor == "" then
        next_cursor = nil
    end
    return LIST_OK, data.result[result_key], next_cursor
end


-- ---------------------------------------------------------------------------
-- MCP 2026-07-28 backends
--
-- One server/discover probe per backend (cached) decides how the router talks
-- to it. A backend that lists 2026-07-28 gets stateless 2026-07-28 requests:
-- no initialize, no backend session, the protocol version in params._meta, and
-- MCP-Protocol-Version / Mcp-Method / Mcp-Name (and Mcp-Param-*) set from the
-- body the router actually sends. Every other backend keeps the initialize
-- handshake. As on the legacy path, each request body is authorized at the
-- backend-bound /validate before it is sent.
-- ---------------------------------------------------------------------------
local ERA_MODERN = "modern"
local ERA_LEGACY = "legacy"
-- How long a backend's era is trusted before it is probed again.
local BACKEND_ERA_CACHE_TTL = 300
-- How long a tool's x-mcp-header map, learned from tools/list, is reused.
local X_MCP_HEADER_CACHE_TTL = 300
local MAX_X_MCP_HEADER_DEPTH = 8
local ROUTER_CLIENT_INFO = { name = "mcp-gateway-virtual-router", version = "1.0.0" }


-- Build a 2026-07-28 _meta. A 2026-07-28 client's own _meta (its clientInfo,
-- clientCapabilities, progressToken) is kept, so a backend can address it, for
-- example with a multi round-trip input request the client declared support for.
local function _modern_meta(existing)
    local meta = {}
    if type(existing) == "table" then
        for key, value in pairs(existing) do
            meta[key] = value
        end
    end
    meta[META_PROTOCOL_VERSION_KEY] = MODERN_PROTOCOL_VERSION
    if meta[META_CLIENT_INFO_KEY] == nil then
        meta[META_CLIENT_INFO_KEY] = ROUTER_CLIENT_INFO
    end
    if meta[META_CLIENT_CAPABILITIES_KEY] == nil then
        -- An empty table encodes as {}: no client capabilities declared.
        meta[META_CLIENT_CAPABILITIES_KEY] = {}
    end
    return meta
end


-- Set (or, for a false value, clear) request headers that backend subrequests
-- inherit. Returns the previous values for _restore_request_headers.
local function _set_request_headers(values)
    local current = ngx.req.get_headers()
    local previous = {}
    for name, value in pairs(values) do
        previous[name] = current[name] or false
        if value then
            ngx.req.set_header(name, value)
        else
            ngx.req.clear_header(name)
        end
    end
    return previous
end


local function _restore_request_headers(previous)
    for name, value in pairs(previous) do
        if value then
            ngx.req.set_header(name, value)
        else
            ngx.req.clear_header(name)
        end
    end
end


-- Encode a value for Mcp-Name or Mcp-Param-*: strings as is, integers in
-- decimal, booleans as true/false. A value that is not plain visible ASCII, has
-- leading or trailing whitespace, or looks like the sentinel itself is sent as
-- =?base64?...?=. Returns nil for a value that cannot be mirrored.
local function _encode_mcp_header_value(value)
    local text
    if type(value) == "boolean" then
        text = value and "true" or "false"
    elseif type(value) == "number" then
        if value ~= math.floor(value) then
            return nil
        end
        text = string.format("%d", value)
    elseif type(value) == "string" then
        text = value
    else
        return nil
    end
    local looks_like_sentinel = text:sub(1, #BASE64_SENTINEL_PREFIX) == BASE64_SENTINEL_PREFIX
        and text:sub(-#BASE64_SENTINEL_SUFFIX) == BASE64_SENTINEL_SUFFIX
    if text:find("[^\32-\126]") or text:find("^%s") or text:find("%s$") or looks_like_sentinel then
        return BASE64_SENTINEL_PREFIX .. ngx.encode_base64(text) .. BASE64_SENTINEL_SUFFIX
    end
    return text
end


-- Decide whether a backend speaks 2026-07-28, probing it once with
-- server/discover. Only an answer that lists 2026-07-28 makes it modern. A
-- probe the gateway could not authorize or deliver is not cached, so the next
-- request decides again; until then the backend keeps the handshake path.
local function _backend_era(backend_loc, backend_version)
    -- backend_version may be cjson.null (a JSON null in the mapping file).
    local version = type(backend_version) == "string" and backend_version or ""
    local cache_key = "bera:" .. backend_loc .. ":" .. version
    local cached = session_cache:get(cache_key)
    if cached then
        return cached
    end
    local probe = cjson.encode({
        jsonrpc = "2.0",
        id = "disc-" .. (ngx.var.request_id or "0"),
        method = "server/discover",
        params = { _meta = _modern_meta(nil) },
    })
    local previous = _set_request_headers({
        ["MCP-Protocol-Version"] = MODERN_PROTOCOL_VERSION,
        ["Mcp-Method"] = "server/discover",
        ["Mcp-Name"] = false,
        ["Mcp-Session-Id"] = false,
    })
    local auth = _authorize_backend(backend_loc, probe)
    local res = nil
    if auth == AUTH_OK then
        res = ngx.location.capture(backend_loc, { method = ngx.HTTP_POST, body = probe })
    end
    _restore_request_headers(previous)
    if auth ~= AUTH_OK or not res then
        return ERA_LEGACY
    end

    local era = ERA_LEGACY
    local ok, data = pcall(cjson.decode, _parse_sse_body(res.body) or "")
    if res.status == 200 and ok and type(data) == "table" and type(data.result) == "table"
        and type(data.result.supportedVersions) == "table" then
        for _, version in ipairs(data.result.supportedVersions) do
            if version == MODERN_PROTOCOL_VERSION then
                era = ERA_MODERN
            end
        end
    end
    session_cache:set(cache_key, era, BACKEND_ERA_CACHE_TTL)
    ngx.log(ngx.INFO, "Backend ", backend_loc, " protocol era: ", era)
    return era
end


-- Send one request to a 2026-07-28 backend: add the 2026-07-28 _meta, set the
-- mirrored headers from that exact body, authorize it, send it. No session.
-- param_headers maps Mcp-Param-* header names to encoded values (tools/call).
-- Returns (res, outcome, pre_consent) like _call_backend.
local function _call_modern_backend(backend_loc, body, param_headers)
    local ok, request = pcall(cjson.decode, body)
    if not ok or type(request) ~= "table" then
        return nil, AUTH_FAILED, false
    end
    local params = type(request.params) == "table" and request.params or {}
    params._meta = _modern_meta(params._meta)
    request.params = params
    local modern_body = cjson.encode(request)

    local headers = {
        ["MCP-Protocol-Version"] = MODERN_PROTOCOL_VERSION,
        ["Mcp-Method"] = request.method,
        ["Mcp-Name"] = false,
        ["Mcp-Session-Id"] = false,
    }
    local name_field = MCP_NAME_SOURCE[request.method]
    if name_field and params[name_field] ~= nil then
        headers["Mcp-Name"] = _encode_mcp_header_value(params[name_field]) or false
    end
    for name, value in pairs(param_headers or {}) do
        headers[name] = value
    end

    local previous = _set_request_headers(headers)
    local auth = _authorize_backend(backend_loc, modern_body)
    local res = nil
    if auth == AUTH_OK then
        res = ngx.location.capture(backend_loc, { method = ngx.HTTP_POST, body = modern_body })
    end
    _restore_request_headers(previous)
    for name, _ in pairs(param_headers or {}) do
        ngx.req.clear_header(name)
    end
    if auth ~= AUTH_OK then
        return nil, auth, false
    end
    return res, AUTH_OK, false
end


-- Find the x-mcp-header annotations in a tool's inputSchema. Only properties
-- reachable through a chain of "properties" keys count. Returns a list of
-- { header = "Mcp-Param-<name>", path = { key, ... } }, or (nil, reason) when an
-- annotation breaks a 2026-07-28 rule, in which case the tool must be dropped.
local function _x_mcp_header_entries(schema)
    local entries, seen = {}, {}
    local function walk(node, path, depth)
        if depth > MAX_X_MCP_HEADER_DEPTH or type(node) ~= "table"
            or type(node.properties) ~= "table" then
            return true
        end
        for key, child in pairs(node.properties) do
            if type(child) == "table" then
                local child_path = { unpack(path) }
                child_path[#child_path + 1] = key
                local name = child["x-mcp-header"]
                if name ~= nil then
                    if type(name) ~= "string" or not name:match("^[%w!#%$%%&'%*%+%-%.%^_`|~]+$") then
                        return false, "x-mcp-header is not a valid header token"
                    end
                    if child.type ~= "string" and child.type ~= "integer" and child.type ~= "boolean" then
                        return false, "x-mcp-header on a non-primitive or number parameter"
                    end
                    if seen[name:lower()] then
                        return false, "duplicate x-mcp-header " .. name
                    end
                    seen[name:lower()] = true
                    entries[#entries + 1] = { header = "Mcp-Param-" .. name, path = child_path }
                end
                local ok, reason = walk(child, child_path, depth + 1)
                if not ok then
                    return false, reason
                end
            end
        end
        return true
    end
    local ok, reason = walk(schema, {}, 1)
    if not ok then
        return nil, reason
    end
    return entries
end


-- Mcp-Param-* headers for one call: the value at each annotated path, encoded.
-- A missing or null argument sends no header, as the spec requires.
local function _param_headers(entries, arguments)
    local headers = {}
    for _, entry in ipairs(entries or {}) do
        local value = arguments
        for _, key in ipairs(entry.path) do
            value = type(value) == "table" and value[key] or nil
        end
        if value ~= nil and value ~= cjson.null then
            local encoded = _encode_mcp_header_value(value)
            if encoded then
                headers[entry.header] = encoded
            end
        end
    end
    return headers
end


local function _x_mcp_cache_key(backend_loc, tool_name)
    return "xmcp:" .. backend_loc .. ":" .. #tool_name .. ":" .. tool_name
end


-- Drop tools whose x-mcp-header annotations are invalid (the spec says a client
-- MUST exclude them) and remember the rest so tools/call can mirror arguments.
local function _absorb_modern_tool_list(backend_loc, tools)
    local kept = {}
    for _, tool in ipairs(tools) do
        local name = type(tool) == "table" and tool.name
        if type(name) == "string" then
            local entries, reason = _x_mcp_header_entries(tool.inputSchema)
            if entries then
                session_cache:set(_x_mcp_cache_key(backend_loc, name), cjson.encode(entries),
                    X_MCP_HEADER_CACHE_TTL)
                kept[#kept + 1] = tool
            else
                ngx.log(ngx.WARN, "Dropping tool ", name, " from ", backend_loc, ": ", reason)
            end
        end
    end
    session_cache:set("xmcp_listed:" .. backend_loc, "1", X_MCP_HEADER_CACHE_TTL)
    return kept
end


-- Send one JSON-RPC request to a backend: authorize the exact body, resolve
-- the session state, dispatch, and retry once with fresh state when the
-- backend rejects the session (HTTP 400/404/410).
--
-- The body is authorized before any initialize (a caller without the grant
-- never initializes the backend). An initialize authorizes its own body and
-- replaces the token, so the request's body and token are restored afterwards
-- while that token is younger than TOKEN_REUSE_SECONDS, and authorized again
-- otherwise. Each backend-bound /validate is an audit record and a unit of the
-- backing server's rate limit, so a warm session costs exactly one.
-- The retry covers a cached stateless marker too: a backend that has since
-- become stateful rejects session-less requests and must be re-initialized.
--
-- Returns (res, outcome, pre_consent): outcome is AUTH_OK with the response,
-- AUTH_DENIED (grant refused), AUTH_FAILED (the grant could not be checked), or
-- SESSION_FAILED (the grant passed, the backend session could not be set up).
local function _call_backend(backend_loc, body, client_session_id, server_id, backend_version,
                             allow_pre_consent, param_headers)
    -- A backend that speaks 2026-07-28 gets a stateless request, no session.
    if _backend_era(backend_loc, backend_version) == ERA_MODERN then
        return _call_modern_backend(backend_loc, body, param_headers)
    end
    -- Returns session_id, outcome, pre_consent, and whether an initialize ran
    -- (it authorized its own body, replacing the request's token).
    local function resolve()
        if not client_session_id then
            return nil, AUTH_OK, false, false
        end
        local session_id, initialized, denied, pre_consent, fresh = _get_backend_session(
            client_session_id, backend_loc, server_id, backend_version, allow_pre_consent)
        if denied then
            return nil, AUTH_DENIED, false, false
        end
        if not initialized and not pre_consent then
            return nil, SESSION_FAILED, false, false
        end
        return session_id, AUTH_OK, pre_consent == true, fresh == true
    end

    local request_token, minted_at
    local function authorize()
        local auth = _authorize_backend(backend_loc, body)
        if auth == AUTH_OK then
            request_token = ngx.req.get_headers()["X-Internal-Token"]
            minted_at = ngx.now()
        end
        return auth
    end

    local function send(session_id, initialized)
        -- Pre-consent is not a backend session: never send a synthetic ID.
        ngx.req.set_header("Mcp-Session-Id", session_id or "")
        if initialized then
            -- initialize authorized (and sent) its own body; restore this
            -- request's body and token, or authorize it again once stale.
            if ngx.now() - minted_at < TOKEN_REUSE_SECONDS then
                ngx.req.set_header("X-Body", body)
                ngx.req.set_header("X-Internal-Token", request_token)
            else
                local auth = authorize()
                if auth ~= AUTH_OK then
                    return nil, auth
                end
            end
        end
        return ngx.location.capture(backend_loc, { method = ngx.HTTP_POST, body = body }), AUTH_OK
    end

    -- The exact request is checked before any initialize, so a caller without
    -- the grant for it never initializes the backend.
    local outcome = authorize()
    if outcome ~= AUTH_OK then
        return nil, outcome, false
    end
    local session_id, pre_consent, fresh
    session_id, outcome, pre_consent, fresh = resolve()
    if outcome ~= AUTH_OK then
        return nil, outcome, false
    end
    local res
    res, outcome = send(session_id, fresh)
    if outcome ~= AUTH_OK then
        return nil, outcome, pre_consent
    end

    -- Only session-related statuses trigger one fresh-state retry; auth errors
    -- never cause a second initialize or credential vend.
    if res and client_session_id and not pre_consent
        and (res.status == 400 or res.status == 404 or res.status == 410) then
        ngx.log(ngx.WARN, "Backend session state rejected (", res.status, ") for ",
            backend_loc, " -- retrying with fresh state")
        _invalidate_backend_session(client_session_id, backend_loc, backend_version)
        session_id, outcome, pre_consent, fresh = resolve()
        if outcome ~= AUTH_OK then
            return nil, outcome, false
        end
        res, outcome = send(session_id, fresh)
        if outcome ~= AUTH_OK then
            return nil, outcome, pre_consent
        end
    end
    return res, AUTH_OK, pre_consent
end


-- Most pages of one backend's list the router reads. A backend that keeps
-- returning nextCursor beyond this is cut off, so it cannot hold discovery open.
local MAX_LIST_PAGES = 50


-- Fetch one page of a backend's discovery list. Returns (outcome, items, cursor).
local function _fetch_backend_list_page(backend_loc, method_name, result_key, client_session_id,
                                        server_id, allow_pre_consent, cursor)
    local req_body = cjson.encode({
        jsonrpc = "2.0",
        id = "dl-" .. (ngx.var.request_id or "0"),
        method = method_name,
        params = cursor and { cursor = cursor } or {},
    })
    ngx.req.clear_header("X-MCP-Server-Version")
    local res, auth = _call_backend(backend_loc, req_body, client_session_id, server_id, nil,
        allow_pre_consent)
    if auth == AUTH_DENIED then
        return LIST_DENIED
    end
    if auth == AUTH_FAILED then
        return LIST_UNVERIFIED
    end

    -- A 2026-07-28 backend answers a method it does not implement with HTTP 404
    -- and -32601. That is "unsupported", the same as the 200 form legacy
    -- servers use, not a backend failure.
    if auth == AUTH_OK and res and res.status == 404 and not res.truncated then
        local ok, data = pcall(cjson.decode, _parse_sse_body(res.body) or "")
        if ok and type(data) == "table" and type(data.error) == "table"
            and data.error.code == JSONRPC_METHOD_NOT_FOUND then
            return LIST_UNSUPPORTED
        end
    end

    -- Only the gateway's grant check denies. A 401/403 here is the backend's own
    -- answer, e.g. a revoked upstream credential.
    if auth ~= AUTH_OK or not res or res.status ~= 200 or res.truncated then
        ngx.log(ngx.ERR, "Failed to fetch ", method_name, " from ", backend_loc,
            " status=", res and res.status or "nil")
        return LIST_FAILED
    end
    local outcome, items, next_cursor = _classify_list_body(res, result_key, allow_pre_consent)
    if outcome == LIST_FAILED then
        ngx.log(ngx.ERR, "Unusable ", method_name, " response from ", backend_loc)
    end
    return outcome, items, next_cursor
end


-- Fetch one backend's discovery list (tools/list, resources/list, prompts/list),
-- following nextCursor across pages. Returns an outcome (see above) and, for
-- LIST_OK, every item. A page that fails makes the whole fetch fail, so a
-- partial list is never presented or cached as complete.
local function _fetch_backend_list(backend_loc, method_name, result_key, client_session_id,
                                   server_id, allow_pre_consent)
    local all_items = {}
    local cursor = nil
    for page = 1, MAX_LIST_PAGES do
        local outcome, items, next_cursor = _fetch_backend_list_page(backend_loc, method_name,
            result_key, client_session_id, server_id, allow_pre_consent, cursor)
        if outcome ~= LIST_OK then
            if page > 1 and outcome ~= LIST_DENIED then
                ngx.log(ngx.ERR, method_name, " page ", page, " failed for ", backend_loc)
                return LIST_FAILED
            end
            return outcome, items
        end
        for _, item in ipairs(items) do
            all_items[#all_items + 1] = item
        end
        if not next_cursor then
            break
        end
        if page == MAX_LIST_PAGES then
            ngx.log(ngx.WARN, method_name, " from ", backend_loc, " still had a cursor after ",
                MAX_LIST_PAGES, " pages; keeping ", #all_items, " items")
        end
        cursor = next_cursor
    end
    if method_name == "tools/list" and _backend_era(backend_loc, nil) == ERA_MODERN then
        all_items = _absorb_modern_tool_list(backend_loc, all_items)
    end
    return LIST_OK, all_items
end


-- Backend locations whose mapped tools are brokered through egress credentials.
local function _egress_locations(mapping)
    local locations = {}
    for _, tool in ipairs(mapping.tools or {}) do
        if tool.egress_auth_mode and tool.egress_auth_mode ~= "none" then
            locations[tool.backend_location] = true
        end
    end
    return locations
end


-- Fetch every backend's list and classify it. Returns per-backend results in
-- mapping order. A backend whose grant the caller lacks is reported LIST_DENIED
-- and omitted by the aggregators; it never blanks siblings the caller may use.
local function _discover_backends(method_name, result_key, mapping, client_session_id, server_id)
    local egress_locations = _egress_locations(mapping)
    local results = {}
    for _, backend_loc in ipairs(_collect_backend_locations(mapping)) do
        local outcome, items = _fetch_backend_list(backend_loc, method_name, result_key,
            client_session_id, server_id, egress_locations[backend_loc])
        results[#results + 1] = {
            location = backend_loc,
            outcome = outcome,
            items = items,
            egress = egress_locations[backend_loc] == true,
        }
    end
    return results
end


-- Whether any backend answered (ok, consent pending, or unsupported), and
-- whether any backend refused the caller's grant. When nothing answered the
-- aggregate is an error -- a denial if every non-answer included a refusal.
local function _summarize_backends(results)
    local answered, any_denied = false, false
    for _, result in ipairs(results) do
        if result.outcome == LIST_DENIED then
            any_denied = true
        elseif result.outcome ~= LIST_FAILED and result.outcome ~= LIST_UNVERIFIED then
            answered = true
        end
    end
    return answered, any_denied
end


-- Handle tools/list method - proxy to backends for full metadata, with cache
-- The result object for a list the router builds. For a 2026-07-28 client it
-- also carries resultType and the cache hints: the list is filtered per user
-- (scopes, egress consent, blocked tools), so no shared cache may store it.
local function _list_result(result_key, items, modern)
    local result = { [result_key] = _as_json_array(items) }
    if modern then
        result.resultType = "complete"
        result.cacheScope = "private"
        result.ttlMs = ENRICHED_CACHE_TTL * 1000
    end
    return result
end


local function _handle_tools_list(request_id, mapping, user_scopes_str, client_session_id, server_id,
                                  modern)
    -- Egress discovery depends on individual credentials and must be fresh.
    local has_egress_backend = next(_egress_locations(mapping)) ~= nil

    if not _has_scopes(user_scopes_str, mapping.required_scopes) then
        return _jsonrpc_error(request_id, -32603, "Access denied: missing required server scopes")
    end

    -- Each upstream can export a tool with the same original name. Only its
    -- own mapping may contribute to this virtual server's discovery results.
    local allowed_tools = {}
    if mapping.tools then
        for _, tool in ipairs(mapping.tools) do
            local loc = tool.backend_location
            if loc then
                allowed_tools[loc] = allowed_tools[loc] or {}
                allowed_tools[loc][tool.original_name or tool.name] = tool
            end
        end
    end

    -- Backend discovery can depend on user-specific egress credentials. Only
    -- a validated principal may read/write this cache; scope filtering remains
    -- request-local, and every cache hit reauthorizes each backing server.
    local user_id = _auth_user_id()
    if not user_id then
        return _jsonrpc_error(request_id, -32603, "Authenticated user identity required")
    end
    -- Partitioned by rendered config generation, like the mapping cache.
    local enriched_cache_key = "tools_enriched:" .. (ngx.var.vs_config_generation or "") .. ":"
        .. (server_id or "unknown") .. ":" .. #user_id .. ":" .. user_id
    local enriched_tools = nil
    local cached_enriched = not has_egress_backend and session_cache:get(enriched_cache_key)
    if cached_enriched then
        -- Every hit re-checks every backing grant. A cached list was built with
        -- each backend's tools, so if any grant no longer passes (or the check
        -- cannot run) rebuild instead of serving tools the caller may have lost.
        local req_body = cjson.encode({
            jsonrpc = "2.0", id = "tl-" .. (ngx.var.request_id or "0"),
            method = "tools/list", params = {},
        })
        ngx.req.clear_header("X-MCP-Server-Version")
        local all_granted = true
        for _, backend_loc in ipairs(_collect_backend_locations(mapping)) do
            if _authorize_backend(backend_loc, req_body) ~= AUTH_OK then
                all_granted = false
                break
            end
        end
        local ok, cached = pcall(cjson.decode, cached_enriched)
        if all_granted and ok and type(cached) == "table" then
            enriched_tools = cached
        end
    end

    if not enriched_tools then
        local results = _discover_backends("tools/list", "tools", mapping,
            client_session_id, server_id)

        enriched_tools = {}
        local cacheable = true
        local answered = false
        local any_denied = false
        for _, result in ipairs(results) do
            if result.outcome == LIST_OK then
                answered = true
                -- Consent may change within the TTL; never freeze an empty list.
                if #result.items == 0 then
                    cacheable = false
                end
                for _, bt in ipairs(result.items) do
                    local mapping_entry = allowed_tools[result.location]
                        and allowed_tools[result.location][bt.name]
                    if mapping_entry then
                        -- Use the mapping's display name (alias) instead of original name
                        local display_name = mapping_entry.name
                        -- Use mapping's description if non-empty (override), else backend's
                        local desc = mapping_entry.description
                        if not desc or desc == "" then
                            desc = bt.description or ""
                        end
                        enriched_tools[#enriched_tools + 1] = _copy_tool_fields(bt, {
                            name = display_name,
                            original_name = mapping_entry.original_name or mapping_entry.name,
                            backend_server_path = mapping_entry.backend_server_path,
                            block_server_path = mapping_entry.block_server_path,
                            description = desc,
                            inputSchema = _ensure_mcp_schema(bt.inputSchema or bt.input_schema),
                            required_scopes = mapping_entry.required_scopes,
                        })
                    end
                end
            elseif result.outcome == LIST_CONSENT then
                -- Both grants already passed (_authorize_backend), but the caller has
                -- not connected the backend's credential. Publish the mapped tools so
                -- the client can call one and receive the connect/PAT instructions;
                -- never cache this per-credential state.
                answered = true
                cacheable = false
                _append_mapping_tools_for_backend(enriched_tools, mapping, result.location)
            elseif result.outcome == LIST_UNSUPPORTED then
                answered = true
            elseif result.outcome == LIST_DENIED then
                -- The caller lacks this backing grant: its tools are omitted (a call
                -- to one is still refused); siblings stay listed.
                any_denied = true
                cacheable = false
            else
                cacheable = false
                if result.egress or result.outcome == LIST_UNVERIFIED then
                    -- A credentialed backend's mapping is not proof its tools are
                    -- reachable for this user, and an unverified grant is not a
                    -- grant: hide them, keep the healthy siblings.
                    ngx.log(ngx.WARN, "Backend tools/list unavailable for ", result.location,
                        " (", result.outcome, ") -- omitting its tools from this response")
                else
                    answered = true
                    ngx.log(ngx.WARN, "Backend tools/list fetch failed for ", result.location,
                        " -- falling back to mapping file metadata for this backend")
                    _append_mapping_tools_for_backend(enriched_tools, mapping, result.location)
                end
            end
        end

        if #results > 0 and not answered then
            if any_denied then
                return _jsonrpc_error(request_id, -32603, "Backend access denied")
            end
            return _jsonrpc_error(request_id, -32603, "Backend discovery unavailable")
        end

        -- Cache only fully discovered results. A fallback response remains complete,
        -- but the next request should retry failed backends instead of serving it for 60s.
        if cacheable and not has_egress_backend then
            local ok_enc, encoded = pcall(cjson.encode, enriched_tools)
            if ok_enc then
                session_cache:set(enriched_cache_key, encoded, ENRICHED_CACHE_TTL)
            end
        end
    end
    -- Discovery metadata may be cached, but each backing server's block state
    -- is fetched once per response. An unreadable state is an error, not an
    -- apparently successful empty catalog.
    local blocked_by_path = {}
    local tools = setmetatable({}, empty_array_mt)
    for _, tool in ipairs(enriched_tools) do
        if _has_scopes(user_scopes_str, tool.required_scopes) then
            local path = tool.block_server_path or
                (type(tool.backend_version) ~= "string" and tool.backend_server_path)
            if not path then
                return _jsonrpc_error(request_id, -32603, "Backend tool block state unavailable")
            end
            if not blocked_by_path[path] then
                blocked_by_path[path] = _blocked_backend_tools(path)
                if not blocked_by_path[path] then
                    return _jsonrpc_error(request_id, -32603, "Backend tool block state unavailable")
                end
            end
            if not blocked_by_path[path][tool.original_name] then
                tools[#tools + 1] = _copy_tool_fields(tool, {
                    name = tool.name,
                    description = tool.description or "",
                    inputSchema = _ensure_mcp_schema(tool.inputSchema),
                })
            end
        end
    end

    return _jsonrpc_result(request_id, _list_result("tools", tools, modern))
end


-- Per-user cache key for a resource/prompt item -> backend lookup, or nil when
-- the caller has no validated identity (never cache across principals). Keyed
-- by the rendered config generation like the mapping cache: a lookup recorded
-- under the previous config can name a backend this server no longer maps.
local function _lookup_cache_key(result_key, server_id)
    local user_id = _auth_user_id()
    if not user_id then
        return nil
    end
    return result_key .. "_lookup:" .. (ngx.var.vs_config_generation or "") .. ":"
        .. (server_id or "unknown") .. ":" .. #user_id .. ":" .. user_id
end


-- Aggregate resource/prompt lists without a cross-user cache: backend content
-- and access can depend on the current user's egress credentials. Returns the
-- items, an item -> backend lookup, and a denial flag; items are nil only when
-- no backend answered (denied when that included a grant refusal). A backend
-- whose grant the caller lacks is omitted.
--
-- A complete answer (every backend ok or unsupported, none credentialed) also
-- stores the caller's item -> backend lookup for _resolve_item_backend.
local function _proxy_list_to_backends(method_name, result_key, mapping, client_session_id, server_id)
    local results = _discover_backends(method_name, result_key, mapping,
        client_session_id, server_id)
    local answered, any_denied = _summarize_backends(results)
    if #results > 0 and not answered then
        return nil, nil, any_denied
    end

    local aggregated = setmetatable({}, empty_array_mt)
    local lookup = {}
    local complete = true
    for _, result in ipairs(results) do
        if result.egress or (result.outcome ~= LIST_OK and result.outcome ~= LIST_UNSUPPORTED) then
            complete = false
        end
        if result.outcome == LIST_OK then
            for _, item in ipairs(result.items) do
                aggregated[#aggregated + 1] = item
                local lookup_key = result_key == "resources" and item.uri or item.name
                if lookup_key then
                    lookup[lookup_key] = result.location
                end
            end
        end
    end

    local cache_key = _lookup_cache_key(result_key, server_id)
    if complete and cache_key then
        local ok_enc, encoded = pcall(cjson.encode, lookup)
        if ok_enc then
            session_cache:set(cache_key, encoded, ENRICHED_CACHE_TTL)
        end
    end
    return aggregated, lookup
end


-- The backend that owns one resource (by uri) or prompt (by name). Served from
-- the caller's own lookup cache when a recent complete discovery recorded the
-- item; otherwise rediscovered. The cache only routes: the read itself still
-- authorizes the owning backend, so a revoked grant is refused there.
-- Returns the backend location, or nil plus "denied"/"failed"/"missing".
local function _resolve_item_backend(method_name, result_key, item_key, mapping,
                                     client_session_id, server_id)
    local cache_key = _lookup_cache_key(result_key, server_id)
    local cached = cache_key and session_cache:get(cache_key)
    if cached then
        local ok, lookup = pcall(cjson.decode, cached)
        local location = ok and type(lookup) == "table" and lookup[item_key]
        -- Only a backend the CURRENT mapping still routes for this server.
        if type(location) == "string" then
            for _, mapped in ipairs(_collect_backend_locations(mapping)) do
                if mapped == location then
                    return location
                end
            end
        end
    end
    local _, lookup, denied = _proxy_list_to_backends(method_name, result_key, mapping,
        client_session_id, server_id)
    if not lookup then
        return nil, denied and "denied" or "failed"
    end
    if not lookup[item_key] then
        return nil, "missing"
    end
    return lookup[item_key]
end


-- Proxy a single request to a specific backend with session management and stale retry.
-- Returns the response body directly. Used for tools/call, resources/read, prompts/get.
local function _proxy_to_backend(request_id, method_name, proxied_params,
                                  backend_location, client_session_id, server_id,
                                  backend_version, label, allow_pre_consent, param_headers)
    local proxied_body = cjson.encode({
        jsonrpc = "2.0",
        id = request_id,
        method = method_name,
        params = proxied_params,
    })

    -- Select the mapped backend version before both initialize and call.
    if type(backend_version) == "string" and backend_version ~= "" then
        ngx.req.set_header("X-MCP-Server-Version", backend_version)
    else
        ngx.req.clear_header("X-MCP-Server-Version")
    end

    local res, outcome = _call_backend(backend_location, proxied_body, client_session_id,
        server_id, backend_version, allow_pre_consent, param_headers)
    if outcome == AUTH_DENIED then
        ngx.status = 403
        ngx.say(_jsonrpc_error(request_id, -32603, "Backend access denied"))
        return
    end
    if outcome ~= AUTH_OK then
        -- The backing grant was not refused; the backend (or its check) is
        -- unavailable right now. Retryable, unlike a denial.
        ngx.status = 502
        ngx.say(_jsonrpc_error(request_id, -32603,
            "Backend unavailable for " .. (label or method_name)))
        return
    end
    if not res then
        ngx.status = 200
        ngx.say(_jsonrpc_error(request_id, -32603,
            "Backend request failed for " .. (label or method_name)))
        return
    end

    -- Forward backend response
    ngx.status = res.status
    if res.header and res.header["Content-Type"] then
        ngx.header["Content-Type"] = res.header["Content-Type"]
    else
        ngx.header["Content-Type"] = "application/json"
    end
    ngx.print(res.body)
end


-- Validate a client session ID against MongoDB (L2), bound to the caller's
-- authenticated identity AND the virtual server the session was minted for.
-- A session that exists but belongs to a different user is rejected (returns
-- false), preventing session hijacking via a guessed or stolen Mcp-Session-Id
-- header; a session minted for a different virtual server is likewise rejected,
-- so it cannot be replayed across virtual servers.
-- Uses L1 cache to avoid repeated DB lookups; the cache key includes both the
-- user and the virtual server path so a cached "valid" result can never
-- authorize a different user or a different virtual server.
-- Returns (ok, err): ok is true if valid AND owned by the caller for this
-- server. Otherwise ok is false and err is one of:
--   "not_found": the registry answered 404. No live session has this ID for
--       this user and virtual server (never minted, expired after the idle
--       TTL, owned by another user, or minted for another virtual server).
--       Also returned for an empty ID, which route() already rejects with 400.
--   "unavailable": the check itself failed (no response, or any status other
--       than 200 or 404), so the state of the session is unknown.
local function _validate_client_session(client_session_id, user_id, virtual_server_path)
    if not client_session_id or client_session_id == "" then
        return false, "not_found"
    end

    -- L1: fast path check (cache valid sessions for SESSION_CACHE_TTL).
    -- Key on user_id and virtual_server_path so a session validated for one
    -- user/server is never reused to authorize a different one from the shared
    -- per-worker cache.
    local cache_key = "csess_valid:" .. user_id .. ":"
        .. (virtual_server_path or "") .. ":" .. client_session_id
    local cached = session_cache:get(cache_key)
    if cached == "1" then
        return true
    end

    -- L2: validate via internal FastAPI endpoint, passing the authenticated
    -- user and virtual server path so the registry enforces both bindings in
    -- the DB query.
    -- Escape the session ID into the path segment as defense in depth -- the
    -- caller (route()) already allowlists it to ^vs-[0-9a-f]+$, but escaping
    -- here means this function is safe even if a future caller forgets to.
    local query = "?user_id=" .. ngx.escape_uri(user_id)
    if virtual_server_path then
        query = query .. "&virtual_server_path=" .. ngx.escape_uri(virtual_server_path)
    end
    local res = ngx.location.capture(
        "/_internal/sessions/client/" .. ngx.escape_uri(client_session_id) .. query,
        { method = ngx.HTTP_GET }
    )

    if res and res.status == 200 then
        session_cache:set(cache_key, "1", SESSION_CACHE_TTL)
        return true
    end
    if res and res.status == 404 then
        return false, "not_found"
    end

    ngx.log(ngx.ERR, "Client session validation failed for path=", virtual_server_path,
        " status=", res and res.status or "nil")
    return false, "unavailable"
end


-- Negotiate protocol version: if client's version is supported, echo it back;
-- otherwise respond with our latest supported version.
local function _negotiate_protocol_version(client_version)
    if client_version and SUPPORTED_PROTOCOL_VERSIONS[client_version] then
        return client_version
    end
    return LATEST_PROTOCOL_VERSION
end


-- Handle initialize method - create client session, return MCP capabilities.
-- Returns (response_string, err) where err is nil on success, or one of:
--   "unauthenticated" -- no authenticated identity; route() emits 401 (a
--       session must be owned by a concrete user, so we refuse to mint one).
--   "create_failed"   -- the client session could not be created; route()
--       emits an error instead of a misleading 200 with no Mcp-Session-Id
--       (which would make the client 400 on its very next request).
-- Create a client session owned by user_id for this virtual server, via the
-- registry's internal API. Returns the server-minted vs-<hex> ID, or nil.
local function _create_client_session(server_id, user_id)
    local body = cjson.encode({
        user_id = user_id,
        virtual_server_path = "/virtual/" .. server_id,
    })
    local res = ngx.location.capture("/_internal/sessions/client", {
        method = ngx.HTTP_POST,
        body = body,
    })

    local client_session_id = nil
    if res and res.status == 201 then
        local ok, data = pcall(cjson.decode, res.body)
        if ok and type(data) == "table" and type(data.client_session_id) == "string" then
            client_session_id = data.client_session_id
        end
    end
    if not client_session_id then
        ngx.log(ngx.ERR, "Failed to create client session for server=", server_id,
            " status=", res and res.status or "nil")
    end
    return client_session_id
end


local function _handle_initialize(request_id, server_id, params)
    local user_id = _auth_user_id()
    if not user_id then
        ngx.log(ngx.WARN, "initialize rejected: no authenticated user for server=", server_id)
        return nil, "unauthenticated"
    end

    -- Fail loudly if the session could not be created: returning a successful
    -- initialize with no Mcp-Session-Id just defers the failure to the client's
    -- next request (which 400s on the missing session).
    local client_session_id = _create_client_session(server_id, user_id)
    if not client_session_id then
        return nil, "create_failed"
    end

    -- Set Mcp-Session-Id response header so client includes it in future requests
    ngx.header["Mcp-Session-Id"] = client_session_id
    ngx.log(ngx.INFO, "Created client session ", client_session_id,
        " for user=", user_id, " server=", server_id)

    -- Negotiate protocol version with client
    local client_version = params and params.protocolVersion
    local negotiated_version = _negotiate_protocol_version(client_version)

    local result = {
        protocolVersion = negotiated_version,
        capabilities = {
            tools = {
                listChanged = false,
            },
        },
        serverInfo = {
            name = "mcp-gateway-virtual-server",
            version = "1.0.0",
        },
    }
    return _jsonrpc_result(request_id, result)
end
_blocked_backend_tools = function(server_path)
    -- The registry validates this exact active server on every request. A
    -- version-pinned mapping keeps its versioned dispatch path separately.
    if type(server_path) ~= "string" or server_path:sub(1, 1) ~= "/" then
        return nil
    end
    local uri = "/_internal/sessions/tool-block?server_path=" .. ngx.escape_uri(server_path)
    local res = ngx.location.capture(uri, { method = ngx.HTTP_GET })
    if not res or res.status ~= 200 then
        return nil
    end
    local ok, body = pcall(cjson.decode, res.body or "")
    if not ok or type(body) ~= "table" or type(body.blocked_tools) ~= "table" then
        return nil
    end
    local blocked = {}
    for _, name in ipairs(body.blocked_tools) do
        if type(name) ~= "string" then
            return nil
        end
        blocked[name] = true
    end
    return blocked
end


-- Handle tools/call method - proxy to the correct backend with session management
-- Mcp-Param-* headers for a tools/call to a 2026-07-28 backend, from the
-- x-mcp-header annotations in that backend's tools/list. The list is fetched
-- once when it has not been seen recently. Returns nil for any other backend.
local function _modern_param_headers(backend_loc, tool_name, arguments, client_session_id,
                                     server_id, allow_pre_consent)
    if _backend_era(backend_loc, nil) ~= ERA_MODERN then
        return nil
    end
    local cached = session_cache:get(_x_mcp_cache_key(backend_loc, tool_name))
    if not cached and session_cache:get("xmcp_listed:" .. backend_loc) ~= "1" then
        local version = ngx.req.get_headers()["X-MCP-Server-Version"]
        _fetch_backend_list(backend_loc, "tools/list", "tools", client_session_id, server_id,
            allow_pre_consent)
        if version then
            ngx.req.set_header("X-MCP-Server-Version", version)
        end
        cached = session_cache:get(_x_mcp_cache_key(backend_loc, tool_name))
    end
    local ok, entries = pcall(cjson.decode, cached or "[]")
    if not ok or type(entries) ~= "table" then
        return nil
    end
    return _param_headers(entries, arguments)
end


local function _handle_tools_call(request_id, mapping, params, user_scopes_str, client_session_id, server_id)
    -- Enforce server-level required_scopes before processing
    if not _has_scopes(user_scopes_str, mapping.required_scopes) then
        ngx.status = 200
        ngx.say(_jsonrpc_error(request_id, -32603, "Access denied: missing required server scopes"))
        return
    end

    local tool_name = params and params.name
    if not tool_name then
        ngx.status = 200
        ngx.say(_jsonrpc_error(request_id, -32602, "Missing tool name in params"))
        return
    end

    -- Look up tool in backend map
    local tool_info = mapping.tool_backend_map and mapping.tool_backend_map[tool_name]
    if not tool_info then
        ngx.status = 200
        ngx.say(_jsonrpc_error(request_id, -32601, "Tool not found: " .. tool_name))
        return
    end

    -- Enforce per-tool scopes
    -- A backend map entry is not a scope grant: require its matching virtual
    -- tool entry, then check the alias's own required scopes.
    local mapped_tool = nil
    for _, tool_entry in ipairs(mapping.tools or {}) do
        if tool_entry.name == tool_name then
            mapped_tool = tool_entry
            break
        end
    end
    if not mapped_tool or not _has_scopes(user_scopes_str, mapped_tool.required_scopes) then
        ngx.status = 200
        ngx.say(_jsonrpc_error(request_id, -32603,
            "Access denied: missing required scopes for tool: " .. tool_name))
        return
    end

    -- Rewrite tool name to original if aliased
    local original_name = tool_info.original_name or tool_name
    local backend_location = tool_info.backend_location

    if not backend_location then
        ngx.status = 200
        ngx.say(_jsonrpc_error(request_id, -32603, "No backend location for tool: " .. tool_name))
        return
    end
    -- The alias's scope entry and backend dispatch entry must describe the same
    -- selected backend. Never authorize one location/path then dispatch another.
    local mapped_version = type(mapped_tool.backend_version) == "string"
        and mapped_tool.backend_version or ""
    local dispatched_version = type(tool_info.backend_version) == "string"
        and tool_info.backend_version or ""
    if mapped_tool.backend_location ~= backend_location
        or mapped_tool.backend_server_path ~= tool_info.backend_server_path
        or mapped_tool.block_server_path ~= tool_info.block_server_path
        or (mapped_tool.original_name or tool_name) ~= original_name
        or mapped_version ~= dispatched_version then
        ngx.status = 200
        ngx.say(_jsonrpc_error(request_id, -32603, "Inconsistent backend mapping for tool"))
        return
    end
    local block_path = tool_info.block_server_path
        or (type(tool_info.backend_version) ~= "string" and tool_info.backend_server_path)
    local blocked = block_path and _blocked_backend_tools(block_path)
    if not blocked or blocked[original_name] then
        ngx.status = 200
        ngx.say(_jsonrpc_error(request_id, -32603, "Access denied: backend tool blocked or unavailable"))
        return
    end

    -- Build the proxied params with original tool name
    local proxied_params = {}
    if params then
        for k, v in pairs(params) do
            proxied_params[k] = v
        end
    end
    proxied_params.name = original_name

    local is_egress = mapped_tool.egress_auth_mode and mapped_tool.egress_auth_mode ~= "none"
    local param_headers = _modern_param_headers(backend_location, original_name,
        proxied_params.arguments, client_session_id, server_id, is_egress)

    -- Proxy to backend with session management
    _proxy_to_backend(
        request_id, "tools/call", proxied_params,
        backend_location, client_session_id, server_id,
        tool_info.backend_version, "tool:" .. tool_name,
        is_egress, param_headers
    )
end


-- Handle resources/read - proxy to the backend that owns the resource
local function _handle_resources_read(request_id, params, mapping, client_session_id, server_id)
    local uri = params and params.uri
    if not uri then
        ngx.status = 200
        ngx.say(_jsonrpc_error(request_id, -32602, "Missing resource uri in params"))
        return
    end

    -- Resolve resource ownership from the current user's backend discovery.
    local backend_loc, problem = _resolve_item_backend("resources/list", "resources", uri,
        mapping, client_session_id, server_id)
    if problem == "missing" then
        ngx.status = 200
        ngx.say(_jsonrpc_error(request_id, -32601, "Resource not found: " .. uri))
        return
    end
    if not backend_loc then
        ngx.status = problem == "denied" and 403 or 502
        ngx.say(_jsonrpc_error(request_id, -32603,
            problem == "denied" and "Backend access denied" or "Backend discovery failed"))
        return
    end

    _proxy_to_backend(
        request_id, "resources/read", params,
        backend_loc, client_session_id, server_id,
        nil, "resource:" .. uri
    )
end


-- Handle prompts/get - proxy to the backend that owns the prompt
local function _handle_prompts_get(request_id, params, mapping, client_session_id, server_id)
    local name = params and params.name
    if not name then
        ngx.status = 200
        ngx.say(_jsonrpc_error(request_id, -32602, "Missing prompt name in params"))
        return
    end

    -- Resolve prompt ownership from the current user's backend discovery.
    local backend_loc, problem = _resolve_item_backend("prompts/list", "prompts", name,
        mapping, client_session_id, server_id)
    if problem == "missing" then
        ngx.status = 200
        ngx.say(_jsonrpc_error(request_id, -32601, "Prompt not found: " .. name))
        return
    end
    if not backend_loc then
        ngx.status = problem == "denied" and 403 or 502
        ngx.say(_jsonrpc_error(request_id, -32603,
            problem == "denied" and "Backend access denied" or "Backend discovery failed"))
        return
    end

    _proxy_to_backend(
        request_id, "prompts/get", params,
        backend_loc, client_session_id, server_id,
        nil, "prompt:" .. name
    )
end


-- Main entry point
-- ---------------------------------------------------------------------------
-- MCP 2026-07-28 clients
--
-- A 2026-07-28 client sends no initialize and no Mcp-Session-Id. Every request
-- carries its protocol version in params._meta, mirrored in the
-- MCP-Protocol-Version header, and Mcp-Method / Mcp-Name mirror the body.
-- The router validates those headers (a server that processes the body MUST),
-- answers server/discover itself, and serves the rest through a client session
-- it creates on the caller's behalf. That session is owned by the authenticated
-- user and scoped to this virtual server, exactly like one minted by
-- initialize, so backend sessions, grants and egress consent work unchanged.
-- Requests without the 2026-07-28 markers keep the handshake-era path.
-- Spec: https://modelcontextprotocol.io/specification/2026-07-28/basic/transports/streamable-http
-- ---------------------------------------------------------------------------
-- The registry expires an idle client session after 1 hour; reuse the cached
-- one for half of that and re-validate it before every use anyway.
local MODERN_SESSION_CACHE_TTL = 1800
-- Versions advertised by server/discover and in -32022 errors, newest first.
local ADVERTISED_PROTOCOL_VERSIONS = {
    "2026-07-28", "2025-11-25", "2025-06-18", "2025-03-26", "2024-11-05",
}


-- A request is 2026-07-28 when its _meta or its header names that era.
local function _is_modern_request(params)
    local meta = type(params) == "table" and params._meta
    if type(meta) == "table" and type(meta[META_PROTOCOL_VERSION_KEY]) == "string" then
        return true
    end
    local header = ngx.var.http_mcp_protocol_version
    return type(header) == "string" and header:sub(1, 5) == "2026-"
end


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


-- Write a 2026-07-28 protocol error with an HTTP status and optional data.
local function _modern_error(status, request_id, code, message, data)
    ngx.status = status
    ngx.header["Content-Type"] = "application/json"
    local err = { code = code, message = message }
    if data then
        err.data = data
    end
    ngx.say(cjson.encode({ jsonrpc = "2.0", id = request_id or cjson.null, error = err }))
end


-- Validate the version and the mirrored headers of a 2026-07-28 request.
-- Writes the error response and returns false when the request must stop.
local function _validate_modern_request(request_id, method, params)
    local header_version = ngx.var.http_mcp_protocol_version
    local meta = type(params) == "table" and params._meta
    local body_version = type(meta) == "table" and meta[META_PROTOCOL_VERSION_KEY] or nil

    if method == "initialize" then
        _modern_error(400, request_id, JSONRPC_HEADER_MISMATCH,
            "Header mismatch: initialize (the legacy handshake) was sent with 2026-07-28 metadata")
        return false
    end
    if type(header_version) ~= "string" or header_version == "" then
        _modern_error(400, request_id, JSONRPC_HEADER_MISMATCH,
            "Header mismatch: missing MCP-Protocol-Version header")
        return false
    end
    if type(body_version) ~= "string" then
        _modern_error(400, request_id, JSONRPC_HEADER_MISMATCH,
            "Header mismatch: params._meta lacks " .. META_PROTOCOL_VERSION_KEY)
        return false
    end
    if header_version ~= body_version then
        _modern_error(400, request_id, JSONRPC_HEADER_MISMATCH,
            "Header mismatch: MCP-Protocol-Version does not match params._meta")
        return false
    end
    if body_version ~= MODERN_PROTOCOL_VERSION then
        _modern_error(400, request_id, JSONRPC_UNSUPPORTED_PROTOCOL_VERSION,
            "Unsupported protocol version: " .. body_version,
            { supported = ADVERTISED_PROTOCOL_VERSIONS, requested = body_version })
        return false
    end
    if ngx.var.http_mcp_method ~= method then
        _modern_error(400, request_id, JSONRPC_HEADER_MISMATCH,
            "Header mismatch: Mcp-Method does not match the body method")
        return false
    end
    local name_field = MCP_NAME_SOURCE[method]
    if name_field then
        local body_name = type(params) == "table" and params[name_field] or nil
        if _decode_mcp_header_value(ngx.var.http_mcp_name) ~= body_name then
            _modern_error(400, request_id, JSONRPC_HEADER_MISMATCH,
                "Header mismatch: Mcp-Name does not match params." .. name_field)
            return false
        end
    end
    return true
end


-- The 2026-07-28 headers describe the client's request to the router. A
-- backend subrequest carries a different body (the router's own initialize,
-- an alias rewritten to the original tool name, a list with empty params), so
-- the client's copies must not reach it.
local function _clear_client_mcp_headers()
    ngx.req.clear_header("MCP-Protocol-Version")
    ngx.req.clear_header("Mcp-Method")
    ngx.req.clear_header("Mcp-Name")
    for name, _ in pairs(ngx.req.get_headers()) do
        if type(name) == "string" and name:lower():sub(1, 10) == "mcp-param-" then
            ngx.req.clear_header(name)
        end
    end
end


-- Answer server/discover for this virtual server.
local function _handle_discover(request_id)
    return _jsonrpc_result(request_id, {
        resultType = "complete",
        supportedVersions = ADVERTISED_PROTOCOL_VERSIONS,
        -- An empty Lua table encodes as a JSON object ({}), which is what an
        -- advertised capability with no sub-options is.
        capabilities = {
            tools = { listChanged = false },
            resources = {},
            prompts = {},
        },
        _meta = {
            [META_SERVER_INFO_KEY] = { name = "mcp-gateway-virtual-server", version = "1.0.0" },
        },
    })
end


-- Return the client session the router uses for a 2026-07-28 caller, creating
-- one when none is cached or the cached one has expired. Returns (id, err) with
-- err "unavailable" when the registry cannot be reached.
local function _modern_client_session(server_id, user_id)
    local virtual_path = "/virtual/" .. server_id
    local cache_key = "modern_csess:" .. #user_id .. ":" .. user_id .. ":" .. virtual_path
    local cached = session_cache:get(cache_key)
    if cached then
        local ok, err = _validate_client_session(cached, user_id, virtual_path)
        if ok then
            return cached
        end
        if err == "unavailable" then
            return nil, "unavailable"
        end
        session_cache:delete(cache_key)
    end
    local client_session_id = _create_client_session(server_id, user_id)
    if not client_session_id then
        return nil, "unavailable"
    end
    session_cache:set(cache_key, client_session_id, MODERN_SESSION_CACHE_TTL)
    return client_session_id
end


function _M.route()
    -- Per MCP Streamable HTTP 2025-11-25 spec, the client MUST include Accept header
    -- listing both application/json and text/event-stream. Set this on the request so
    -- all ngx.location.capture subrequests to backends inherit it.
    ngx.req.set_header("Accept", "application/json, text/event-stream")
    -- A client-supplied internal token must never enter the backend hop; only
    -- the backend-bound /validate response may mint one.
    ngx.req.clear_header("X-Internal-Token")

    -- Forward the validated caller identity to backend subrequests (fail-closed:
    -- always the gateway-validated value or cleared, never a client-supplied one).
    _forward_identity_headers()

    local request_method = ngx.var.request_method

    -- Handle HTTP GET: per MCP Streamable HTTP 2025-11-25 spec section 3.3,
    -- the server MUST either return Content-Type: text/event-stream or HTTP 405.
    -- We do not support server-initiated SSE streams.
    if request_method == "GET" then
        ngx.status = 405
        ngx.header["Allow"] = "POST"
        return
    end

    -- Handle HTTP DELETE: session termination per MCP spec.
    -- Return 405 Method Not Allowed to indicate we don't support client-initiated termination.
    if request_method == "DELETE" then
        ngx.status = 405
        ngx.header["Content-Type"] = "application/json"
        ngx.header["Allow"] = "POST, GET"
        return
    end

    -- Only POST is accepted for JSON-RPC messages
    if request_method ~= "POST" then
        ngx.status = 405
        ngx.header["Content-Type"] = "application/json"
        ngx.header["Allow"] = "POST, GET, DELETE"
        return
    end

    -- Read request body
    ngx.req.read_body()
    local body = ngx.req.get_body_data()

    if not body then
        ngx.status = 400
        ngx.header["Content-Type"] = "application/json"
        ngx.say(_jsonrpc_error(nil, -32700, "Empty request body"))
        return
    end

    -- Parse JSON-RPC message
    local ok, request = pcall(cjson.decode, body)
    if not ok then
        ngx.status = 400
        ngx.header["Content-Type"] = "application/json"
        ngx.say(_jsonrpc_error(nil, -32700, "Parse error"))
        return
    end

    local request_id = request.id
    local method = request.method
    local params = request.params

    -- Get virtual server ID from nginx variable
    local server_id = ngx.var.virtual_server_id
    if not server_id or server_id == "" then
        ngx.status = 500
        ngx.header["Content-Type"] = "application/json"
        ngx.say(_jsonrpc_error(request_id, -32603, "Virtual server ID not configured"))
        return
    end

    -- Detect JSON-RPC notifications (no "id" field) vs requests (have "id" field).
    -- Per MCP Streamable HTTP spec, notifications and responses MUST get HTTP 202 Accepted
    -- with no body. Only JSON-RPC requests get a JSON-RPC response.
    local is_notification = (request_id == nil) and (method ~= nil)

    -- Handle notifications: return 202 Accepted with no body per MCP spec
    if is_notification then
        if method == "notifications/initialized" then
            ngx.log(ngx.INFO, "Received initialized notification for server=", server_id)
        elseif method == "notifications/cancelled" then
            ngx.log(ngx.INFO, "Received cancelled notification for server=", server_id)
        else
            ngx.log(ngx.INFO, "Received notification method=", method, " for server=", server_id)
        end
        ngx.status = 202
        return
    end

    -- MCP 2026-07-28: validate the mirrored headers, then drop the client's
    -- copies so no backend subrequest inherits them (see _clear_client_mcp_headers).
    local modern = _is_modern_request(params)
    if modern then
        if not _validate_modern_request(request_id, method, params) then
            return
        end
        _clear_client_mcp_headers()
        -- 2026-07-28 has no sessions: ignore any Mcp-Session-Id the client sent.
        ngx.req.clear_header("Mcp-Session-Id")
        if method == "server/discover" then
            ngx.status = 200
            ngx.header["Content-Type"] = "application/json"
            ngx.say(_handle_discover(request_id))
            return
        end
    end

    -- Handle initialize: generate a client session and return capabilities.
    -- A session must be owned by a concrete user; refuse to mint one for a
    -- request with no authenticated identity, and surface a session-creation
    -- failure as an error rather than a misleading 200 with no Mcp-Session-Id.
    if method == "initialize" then
        local init_response, init_err = _handle_initialize(request_id, server_id, params)
        ngx.header["Content-Type"] = "application/json"
        if init_err == "unauthenticated" then
            ngx.status = 401
            ngx.say(_jsonrpc_error(request_id, -32600,
                "Authenticated user identity required."))
            return
        elseif init_err or not init_response then
            ngx.status = 503
            ngx.say(_jsonrpc_error(request_id, -32603,
                "Failed to create session. Please retry."))
            return
        end
        ngx.status = 200
        ngx.say(init_response)
        return
    end

    -- Handle ping: simple echo (no mapping needed). 2026-07-28 removed ping, so a
    -- modern request falls through to "method not found" below.
    if method == "ping" and not modern then
        ngx.status = 200
        ngx.header["Content-Type"] = "application/json"
        ngx.say(_jsonrpc_result(request_id, {}))
        return
    end

    -- Get client session ID from request header (set during initialize)
    local client_session_id = ngx.var.http_mcp_session_id

    -- Resolve the authenticated identity. Fail closed if absent: every session
    -- is owned by a concrete user, so an identity-less request can never own
    -- one. This also avoids treating a missing identity as a shared
    -- "anonymous" owner whose sessions any other identity-less caller could
    -- reach (auth_request normally blocks unauthenticated callers upstream,
    -- but we do not depend on that single layer for session ownership).
    local auth_user = _auth_user_id()
    if not auth_user then
        ngx.status = 401
        ngx.header["Content-Type"] = "application/json"
        ngx.say(_jsonrpc_error(request_id, -32600,
            "Authenticated user identity required."))
        return
    end

    -- A 2026-07-28 client has no session. Use the one the router keeps for this
    -- user on this virtual server; it passes the same owner and server checks.
    if modern then
        local session_err
        client_session_id, session_err = _modern_client_session(server_id, auth_user)
        if not client_session_id then
            ngx.log(ngx.ERR, "No client session for 2026-07-28 caller on server=", server_id,
                " err=", session_err)
            _modern_error(503, request_id, -32603, "Session store unavailable. Please retry.")
            return
        end
    end

    -- Reject any session ID that is not in the server-minted format before it
    -- reaches an internal subrequest. The ID is attacker-controlled (it comes
    -- straight from the Mcp-Session-Id header) and is interpolated into the
    -- /_internal/sessions/ subrequest URI; without this allowlist a crafted
    -- value like "vs-x?user_id=victim&" could inject an earlier user_id query
    -- param and shadow the owner check that the ownership binding relies on.
    -- Server-minted IDs are always "vs-" + uuid4 hex (see create_client_session),
    -- so this strict pattern rejects every injection vector with no escaping
    -- reasoning required.
    if not client_session_id or not string.match(client_session_id, "^vs%-[0-9a-f]+$") then
        ngx.status = 400
        ngx.header["Content-Type"] = "application/json"
        ngx.say(_jsonrpc_error(request_id, -32600,
            "Missing or invalid Mcp-Session-Id. Send an initialize request first."))
        return
    end

    -- Validate client session. Per MCP Streamable HTTP 2025-03-26 to 2025-11-25
    -- (Session Management), 400 is for a request without an Mcp-Session-Id
    -- (handled above) and a terminated session MUST get 404 Not Found, which
    -- tells the client to start a new session. The registry expires idle client
    -- sessions, so an unknown well-formed ID gets 404.
    -- Initialize and ping are exempt; notifications already handled above with 202.
    -- The session must also belong to the authenticated caller AND have been
    -- minted for this virtual server -- presenting another user's session ID,
    -- or replaying a session from a different virtual server, is rejected here
    -- before any session context is loaded or routed, with the same 404 so the
    -- reply does not reveal that the session exists for someone else.
    -- A failed check gets 503 so clients retry instead of re-initializing.
    local session_ok, session_err = _validate_client_session(client_session_id, auth_user,
        "/virtual/" .. server_id)
    if not session_ok then
        ngx.header["Content-Type"] = "application/json"
        if session_err == "unavailable" then
            ngx.status = 503
            ngx.say(_jsonrpc_error(request_id, -32603,
                "Session validation unavailable. Please retry."))
            return
        end
        ngx.status = 404
        ngx.say(_jsonrpc_error(request_id, -32600,
            "Session not found or expired. Send a new initialize request."))
        return
    end

    -- Load mapping for all other methods
    local mapping = _get_mapping(server_id)
    if not mapping then
        ngx.status = 500
        ngx.header["Content-Type"] = "application/json"
        ngx.say(_jsonrpc_error(request_id, -32603, "Virtual server mapping not found"))
        return
    end

    -- Get user scopes from auth
    local user_scopes_str = ngx.var.auth_scopes or ""

    -- Route based on method
    ngx.header["Content-Type"] = "application/json"

    if method == "tools/list" then
        ngx.status = 200
        ngx.say(_handle_tools_list(request_id, mapping, user_scopes_str, client_session_id, server_id,
            modern))

    elseif method == "tools/call" then
        _handle_tools_call(request_id, mapping, params, user_scopes_str, client_session_id, server_id)

    elseif method == "resources/list" then
        -- Enforce server-level required_scopes
        if not _has_scopes(user_scopes_str, mapping.required_scopes) then
            ngx.status = 200
            ngx.say(_jsonrpc_error(request_id, -32603, "Access denied: missing required server scopes"))
            return
        end
        local resources, _, denied = _proxy_list_to_backends("resources/list", "resources",
            mapping, client_session_id, server_id)
        if not resources then
            ngx.status = denied and 403 or 502
            ngx.say(_jsonrpc_error(request_id, -32603,
                denied and "Backend access denied" or "Backend discovery failed"))
            return
        end
        ngx.status = 200
        ngx.say(_jsonrpc_result(request_id, _list_result("resources", resources, modern)))

    elseif method == "resources/templates/list" then
        -- Enforce server-level required_scopes
        if not _has_scopes(user_scopes_str, mapping.required_scopes) then
            ngx.status = 200
            ngx.say(_jsonrpc_error(request_id, -32603, "Access denied: missing required server scopes"))
            return
        end
        local templates, _, denied = _proxy_list_to_backends("resources/templates/list",
            "resourceTemplates", mapping, client_session_id, server_id)
        if not templates then
            ngx.status = denied and 403 or 502
            ngx.say(_jsonrpc_error(request_id, -32603,
                denied and "Backend access denied" or "Backend discovery failed"))
            return
        end
        ngx.status = 200
        ngx.say(_jsonrpc_result(request_id, _list_result("resourceTemplates", templates, modern)))

    elseif method == "resources/read" then
        -- Enforce server-level required_scopes
        if not _has_scopes(user_scopes_str, mapping.required_scopes) then
            ngx.status = 200
            ngx.say(_jsonrpc_error(request_id, -32603, "Access denied: missing required server scopes"))
            return
        end
        _handle_resources_read(request_id, params, mapping, client_session_id, server_id)

    elseif method == "prompts/list" then
        -- Enforce server-level required_scopes
        if not _has_scopes(user_scopes_str, mapping.required_scopes) then
            ngx.status = 200
            ngx.say(_jsonrpc_error(request_id, -32603, "Access denied: missing required server scopes"))
            return
        end
        local prompts, _, denied = _proxy_list_to_backends("prompts/list", "prompts",
            mapping, client_session_id, server_id)
        if not prompts then
            ngx.status = denied and 403 or 502
            ngx.say(_jsonrpc_error(request_id, -32603,
                denied and "Backend access denied" or "Backend discovery failed"))
            return
        end
        ngx.status = 200
        ngx.say(_jsonrpc_result(request_id, _list_result("prompts", prompts, modern)))

    elseif method == "prompts/get" then
        -- Enforce server-level required_scopes
        if not _has_scopes(user_scopes_str, mapping.required_scopes) then
            ngx.status = 200
            ngx.say(_jsonrpc_error(request_id, -32603, "Access denied: missing required server scopes"))
            return
        end
        _handle_prompts_get(request_id, params, mapping, client_session_id, server_id)

    else
        -- 2026-07-28 answers an unknown method with HTTP 404, which a dual-era
        -- client reads as "modern server, method missing", not "legacy server".
        ngx.status = modern and 404 or 200
        ngx.say(_jsonrpc_error(request_id, -32601, "Method not found: " .. tostring(method)))
    end
end

-- Test hook: when loaded by the Lua unit test (which sets _G._VR_TEST), expose
-- internal helpers and return the module WITHOUT executing the router. In
-- production _G._VR_TEST is nil, so this branch is skipped and route() runs.
if _G._VR_TEST then
    _M._fetch_backend_list = _fetch_backend_list
    _M._proxy_list_to_backends = _proxy_list_to_backends
    _M._get_mapping = _get_mapping
    _M._handle_resources_read = _handle_resources_read
    _M._append_mapping_tools_for_backend = _append_mapping_tools_for_backend
    _M._handle_tools_list = _handle_tools_list
    _M._handle_tools_call = _handle_tools_call
    _M._get_backend_session = _get_backend_session
    _M._proxy_to_backend = _proxy_to_backend
    _M._forward_identity_headers = _forward_identity_headers
    _M._is_modern_request = _is_modern_request
    _M._decode_mcp_header_value = _decode_mcp_header_value
    _M._validate_modern_request = _validate_modern_request
    _M._clear_client_mcp_headers = _clear_client_mcp_headers
    _M._modern_client_session = _modern_client_session
    _M._backend_era = _backend_era
    _M._encode_mcp_header_value = _encode_mcp_header_value
    _M._x_mcp_header_entries = _x_mcp_header_entries
    _M._param_headers = _param_headers
    _M._call_backend = _call_backend
    return _M
end

-- Execute routing
_M.route()
