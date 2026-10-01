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


-- Read and cache virtual server mapping from JSON file
local function _get_mapping(server_id)
    local cache_key = "mapping:" .. server_id
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


-- The backend subrequest bypasses nginx auth_request, so explicitly validate
-- the exact rewritten JSON-RPC request at a dedicated backend-bound location.
-- The validated internal token is required even for plain backends; their
-- proxy location must clear it before forwarding upstream.
local function _authorize_backend(backend_location, request_body)
    local auth_location, substitutions = backend_location:gsub("^/_vs_backend_", "/_vs_auth_", 1)
    if substitutions ~= 1 then
        return false
    end
    ngx.req.set_header("X-Body", request_body)
    ngx.req.clear_header("X-Body-Uninspectable")
    ngx.req.clear_header("X-Internal-Token")
    local res = ngx.location.capture(auth_location, {
        method = ngx.HTTP_GET,
    })
    local token = res and res.header and
        (res.header["X-Internal-Token"] or res.header["x-internal-token"])
    if not res or res.status ~= 200 or type(token) ~= "string" or token == "" then
        ngx.log(ngx.WARN, "Backend authorization denied for ", backend_location,
            " status=", res and res.status or "nil")
        return false
    end
    ngx.req.set_header("X-Internal-Token", token)
    return true
end

-- Initialize a backend and return its session ID (nil for stateless) and success.
local function _initialize_backend(backend_location)
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

    if not _authorize_backend(backend_location, init_body) then
        return nil, false, true
    end
    local res = ngx.location.capture(backend_location, {
        method = ngx.HTTP_POST,
        body = init_body,
    })

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

    local session_id = res.header and
        (res.header["Mcp-Session-Id"] or res.header["mcp-session-id"])
    if session_id == "" then
        session_id = nil
    end
    return session_id, true
end


-- Resolve backend state from L1, L2, or initialize. Returns (session ID, success);
-- a successful stateless initialize returns (nil, true), not a fake session ID.
-- The client session was already validated against the authenticated owner.
local function _get_backend_session(client_session_id, backend_location, server_id, backend_version)
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
    local initialized, denied
    session_id, initialized, denied = _initialize_backend(backend_location)
    if not initialized then
        return nil, false, denied
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
    return session_id, true
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
                description = tool.description or "",
                inputSchema = _ensure_mcp_schema(tool.inputSchema),
                required_scopes = tool.required_scopes,
            }
        end
    end
end


-- Fetch tools/list from a backend, distinguishing authorization denial from
-- transient failure. A successful empty list is valid but not cacheable.
local function _fetch_backend_tools_list(backend_location, client_session_id, server_id)
    local req_body = cjson.encode({
        jsonrpc = "2.0",
        id = "tl-" .. (ngx.var.request_id or "0"),
        method = "tools/list",
        params = {},
    })
    ngx.req.clear_header("X-MCP-Server-Version")
    if not _authorize_backend(backend_location, req_body) then
        return {}, false, true
    end

    local backend_session_id = nil
    if client_session_id then
        local initialized, denied
        backend_session_id, initialized, denied = _get_backend_session(
            client_session_id, backend_location, server_id)
        if not initialized then
            return {}, false, denied
        end
    end

    if backend_session_id then
        ngx.req.set_header("Mcp-Session-Id", backend_session_id)
    else
        ngx.req.set_header("Mcp-Session-Id", "")
    end

    if not _authorize_backend(backend_location, req_body) then
        return {}, false, true
    end

    local res = ngx.location.capture(backend_location, {
        method = ngx.HTTP_POST,
        body = req_body,
    })

    -- Stale session retry
    if res and (res.status == 400 or res.status == 404 or res.status == 410)
        and client_session_id and backend_session_id then
        ngx.log(ngx.WARN, "Backend tools/list returned ", res.status,
            " for ", backend_location, " -- retrying with fresh session")
        _invalidate_backend_session(client_session_id, backend_location)
        local new_session_id, initialized, denied = _get_backend_session(
            client_session_id, backend_location, server_id)
        if not initialized then
            return {}, false, denied
        end
        ngx.req.set_header("Mcp-Session-Id", new_session_id or "")
        if not _authorize_backend(backend_location, req_body) then
            return {}, false, true
        end
        res = ngx.location.capture(backend_location, {
            method = ngx.HTTP_POST,
            body = req_body,
        })
    end

    if not res or res.status ~= 200 then
        ngx.log(ngx.ERR, "Failed to fetch tools/list from ", backend_location,
            " status=", res and res.status or "nil")
        return {}, false, res and (res.status == 401 or res.status == 403)
    end

    if res.truncated then
        ngx.log(ngx.ERR, "Truncated tools/list response from ", backend_location)
        return {}, false
    end

    -- Backend may respond with SSE format (text/event-stream) or raw JSON
    local json_body = _parse_sse_body(res.body)
    if not json_body then
        ngx.log(ngx.ERR, "Empty or unparseable tools/list response from ", backend_location)
        return {}, false
    end

    local ok, data = pcall(cjson.decode, json_body)
    if not ok then
        ngx.log(ngx.ERR, "Failed to parse tools/list response from ", backend_location)
        return {}, false
    end

    if type(data) == "table" and type(data.result) == "table"
        and type(data.result.tools) == "table" and not data.error then
        return data.result.tools, true
    end

    ngx.log(ngx.ERR, "Missing tools array in tools/list response from ", backend_location)
    return {}, false, type(data) == "table" and data.error ~= nil
end


-- Handle tools/list method - proxy to backends for full metadata, with cache
local function _handle_tools_list(request_id, mapping, user_scopes_str, client_session_id, server_id)
    -- Egress discovery depends on individual credentials and must be fresh.
    local egress_locations = {}
    for _, tool in ipairs(mapping.tools or {}) do
        if tool.egress_auth_mode and tool.egress_auth_mode ~= "none" then
            egress_locations[tool.backend_location] = true
        end
    end
    local has_egress_backend = next(egress_locations) ~= nil

    if not _has_scopes(user_scopes_str, mapping.required_scopes) then
        return _jsonrpc_error(request_id, -32603, "Access denied: missing required server scopes")
    end

    -- Build a set of allowed tools from the mapping (display_name -> mapping entry)
    local allowed_tools = {}
    if mapping.tools then
        for _, tool in ipairs(mapping.tools) do
            allowed_tools[tool.original_name or tool.name] = tool
        end
    end

    -- Backend discovery can depend on user-specific egress credentials. Only
    -- a validated principal may read/write this cache; scope filtering remains
    -- request-local, and every cache hit reauthorizes each backing server.
    local user_id = _auth_user_id()
    if not user_id then
        return _jsonrpc_error(request_id, -32603, "Authenticated user identity required")
    end
    local backend_locations = _collect_backend_locations(mapping)
    local enriched_cache_key = "tools_enriched:" .. (server_id or "unknown")
        .. ":" .. #user_id .. ":" .. user_id
    local enriched_tools = nil
    local cached_enriched = not has_egress_backend and session_cache:get(enriched_cache_key)
    if cached_enriched then
        local req_body = cjson.encode({
            jsonrpc = "2.0", id = "tl-" .. (ngx.var.request_id or "0"),
            method = "tools/list", params = {},
        })
        ngx.req.clear_header("X-MCP-Server-Version")
        for _, backend_loc in ipairs(backend_locations) do
            if not _authorize_backend(backend_loc, req_body) then
                return _jsonrpc_error(request_id, -32603, "Backend access denied")
            end
        end
        local ok, cached = pcall(cjson.decode, cached_enriched)
        if ok and type(cached) == "table" then
            enriched_tools = cached
        end
    end

    if not enriched_tools then
        enriched_tools = {}
        local all_fetches_ok = true

        for _, backend_loc in ipairs(backend_locations) do
            local backend_tools, backend_ok, access_denied = _fetch_backend_tools_list(
                backend_loc, client_session_id, server_id)
            if access_denied then
                return _jsonrpc_error(request_id, -32603, "Backend access denied")
            end
            if backend_ok and #backend_tools == 0 then
                -- Consent may change within the TTL; never freeze an empty list.
                all_fetches_ok = false
            end
            if backend_ok then
                for _, bt in ipairs(backend_tools) do
                    local mapping_entry = allowed_tools[bt.name]
                    if mapping_entry then
                        -- Use the mapping's display name (alias) instead of original name
                        local display_name = mapping_entry.name
                        -- Use mapping's description if non-empty (override), else backend's
                        local desc = mapping_entry.description
                        if not desc or desc == "" then
                            desc = bt.description or ""
                        end
                        enriched_tools[#enriched_tools + 1] = {
                            name = display_name,
                            description = desc,
                            inputSchema = _ensure_mcp_schema(bt.inputSchema or bt.input_schema),
                            required_scopes = mapping_entry.required_scopes,
                        }
                    end
                end
            else
                all_fetches_ok = false
                if egress_locations[backend_loc] then
                    return _jsonrpc_error(request_id, -32603, "Backend discovery unavailable")
                end
                ngx.log(ngx.WARN, "Backend tools/list fetch failed for ", backend_loc,
                    " -- falling back to mapping file metadata for this backend")
                _append_mapping_tools_for_backend(enriched_tools, mapping, backend_loc)
            end
        end

        -- Cache only fully discovered results. A fallback response remains complete,
        -- but the next request should retry failed backends instead of serving it for 60s.
        if all_fetches_ok and not has_egress_backend then
            local ok_enc, encoded = pcall(cjson.encode, enriched_tools)
            if ok_enc then
                session_cache:set(enriched_cache_key, encoded, ENRICHED_CACHE_TTL)
            end
        end
    end

    -- Scope filter: filter cached tools by user's scopes at request time
    local tools = setmetatable({}, empty_array_mt)
    for _, tool in ipairs(enriched_tools) do
        if _has_scopes(user_scopes_str, tool.required_scopes) then
            tools[#tools + 1] = {
                name = tool.name,
                description = tool.description or "",
                inputSchema = _ensure_mcp_schema(tool.inputSchema),
            }
        end
    end

    return _jsonrpc_result(request_id, { tools = _as_json_array(tools) })
end


-- Aggregate resource/prompt lists without a cross-user cache: backend content
-- and access can depend on the current user's egress credentials.
-- Returns nil on authorization/initialize failure.
local function _proxy_list_to_backends(method_name, result_key, mapping, client_session_id, server_id)
    local aggregated = setmetatable({}, empty_array_mt)
    local lookup = {}
    local backend_locations = _collect_backend_locations(mapping)


    ngx.req.clear_header("X-MCP-Server-Version")
    for _, backend_loc in ipairs(backend_locations) do
        local req_body = cjson.encode({
            jsonrpc = "2.0",
            id = "pl-" .. (ngx.var.request_id or "0"),
            method = method_name,
            params = {},
        })
        if not _authorize_backend(backend_loc, req_body) then
            return nil
        end

        local backend_session_id = nil
        if client_session_id then
            local initialized
            backend_session_id, initialized = _get_backend_session(
                client_session_id, backend_loc, server_id)
            if not initialized then
                return nil
            end
        end

        if backend_session_id then
            ngx.req.set_header("Mcp-Session-Id", backend_session_id)
        else
            ngx.req.set_header("Mcp-Session-Id", "")
        end

        if not _authorize_backend(backend_loc, req_body) then
            return nil
        end

        local res = ngx.location.capture(backend_loc, {
            method = ngx.HTTP_POST,
            body = req_body,
        })

        -- Stale session retry
        if res and (res.status == 400 or res.status == 404 or res.status == 410)
            and client_session_id and backend_session_id then
            ngx.log(ngx.WARN, "Backend ", method_name, " returned ", res.status,
                " for ", backend_loc, " -- retrying with fresh session")
            _invalidate_backend_session(client_session_id, backend_loc)
            local new_session_id, initialized = _get_backend_session(
                client_session_id, backend_loc, server_id)
            if not initialized then
                return nil
            end
            ngx.req.set_header("Mcp-Session-Id", new_session_id or "")
            if not _authorize_backend(backend_loc, req_body) then
                return nil
            end
            res = ngx.location.capture(backend_loc, {
                method = ngx.HTTP_POST,
                body = req_body,
            })
        end

        if not res or res.status == 401 or res.status == 403 then
            return nil
        end
        if res.status == 200 and not res.truncated then
            local json_body = _parse_sse_body(res.body)
            local ok, data = pcall(cjson.decode, json_body or "")
            if ok and type(data) == "table" and type(data.result) == "table"
                and type(data.result[result_key]) == "table" and not data.error then
                for _, item in ipairs(data.result[result_key]) do
                    aggregated[#aggregated + 1] = item
                    local lookup_key = result_key == "resources" and item.uri or item.name
                    if lookup_key then
                        lookup[lookup_key] = backend_loc
                    end
                end
            end
        end
    end

    return aggregated, lookup
end


-- Proxy a single request to a specific backend with session management and stale retry.
-- Returns the response body directly. Used for tools/call, resources/read, prompts/get.
local function _proxy_to_backend(request_id, method_name, proxied_params,
                                  backend_location, client_session_id, server_id,
                                  backend_version, label)
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
    if not _authorize_backend(backend_location, proxied_body) then
        ngx.status = 403
        ngx.say(_jsonrpc_error(request_id, -32603, "Backend access denied"))
        return
    end

    local backend_session_id = nil
    if client_session_id then
        local initialized, denied
        backend_session_id, initialized, denied = _get_backend_session(
            client_session_id, backend_location, server_id, backend_version)
        if not initialized then
            ngx.status = denied and 403 or 502
            ngx.say(_jsonrpc_error(request_id, -32603,
                denied and "Backend access denied" or "Backend initialize failed"))
            return
        end
    end

    -- Set the backend session header for the subrequest proxy
    if backend_session_id then
        ngx.req.set_header("Mcp-Session-Id", backend_session_id)
    else
        ngx.req.set_header("Mcp-Session-Id", "")
    end

    if not _authorize_backend(backend_location, proxied_body) then
        ngx.status = 403
        ngx.say(_jsonrpc_error(request_id, -32603, "Backend access denied"))
        return
    end

    local res = ngx.location.capture(backend_location, {
        method = ngx.HTTP_POST,
        body = proxied_body,
    })

    if not res then
        ngx.status = 200
        ngx.say(_jsonrpc_error(request_id, -32603,
            "Backend request failed for " .. (label or method_name)))
        return
    end

    -- Only session-related statuses trigger a stale stateful session retry;
    -- auth errors must never cause a second initialize or credential vend.
    if (res.status == 400 or res.status == 404 or res.status == 410)
        and client_session_id and backend_session_id then
        ngx.log(ngx.WARN, "Backend session rejected for ", label or method_name,
            " -- retrying with fresh session")

        -- Invalidate stale session
        _invalidate_backend_session(client_session_id, backend_location, backend_version)

        -- Get a fresh session (will re-initialize the backend)
        local new_session_id, initialized, denied = _get_backend_session(
            client_session_id, backend_location, server_id, backend_version)
        if not initialized then
            ngx.status = denied and 403 or 502
            ngx.say(_jsonrpc_error(request_id, -32603,
                denied and "Backend access denied" or "Backend initialize failed"))
            return
        end
        ngx.req.set_header("Mcp-Session-Id", new_session_id or "")
        if not _authorize_backend(backend_location, proxied_body) then
            ngx.status = 403
            ngx.say(_jsonrpc_error(request_id, -32603, "Backend access denied"))
            return
        end

        -- Retry the request
        res = ngx.location.capture(backend_location, {
            method = ngx.HTTP_POST,
            body = proxied_body,
        })

        if not res then
            ngx.status = 200
            ngx.say(_jsonrpc_error(request_id, -32603,
                "Backend request failed after retry for " .. (label or method_name)))
            return
        end
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
-- Returns true if valid AND owned by the caller for this server, false otherwise.
local function _validate_client_session(client_session_id, user_id, virtual_server_path)
    if not client_session_id or client_session_id == "" then
        return false
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

    return false
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
local function _handle_initialize(request_id, server_id, params)
    local user_id = _auth_user_id()
    if not user_id then
        ngx.log(ngx.WARN, "initialize rejected: no authenticated user for server=", server_id)
        return nil, "unauthenticated"
    end
    local virtual_path = "/virtual/" .. server_id

    -- Create client session in MongoDB via internal API
    local body = cjson.encode({
        user_id = user_id,
        virtual_server_path = virtual_path,
    })
    local res = ngx.location.capture("/_internal/sessions/client", {
        method = ngx.HTTP_POST,
        body = body,
    })

    local client_session_id = nil
    if res and res.status == 201 then
        local ok, data = pcall(cjson.decode, res.body)
        if ok then
            client_session_id = data.client_session_id
        end
    end

    -- Fail loudly if the session could not be created: returning a successful
    -- initialize with no Mcp-Session-Id just defers the failure to the client's
    -- next request (which 400s on the missing session).
    if not client_session_id then
        ngx.log(ngx.ERR, "Failed to create client session for server=", server_id,
            " status=", res and res.status or "nil")
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


-- Handle tools/call method - proxy to the correct backend with session management
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

    -- Build the proxied params with original tool name
    local proxied_params = {}
    if params then
        for k, v in pairs(params) do
            proxied_params[k] = v
        end
    end
    proxied_params.name = original_name

    -- Proxy to backend with session management
    _proxy_to_backend(
        request_id, "tools/call", proxied_params,
        backend_location, client_session_id, server_id,
        tool_info.backend_version, "tool:" .. tool_name
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
    local _, lookup = _proxy_list_to_backends("resources/list", "resources",
        mapping, client_session_id, server_id)
    if not lookup then
        ngx.status = 403
        ngx.say(_jsonrpc_error(request_id, -32603, "Backend access denied"))
        return
    end

    local backend_loc = lookup and lookup[uri]
    if not backend_loc then
        ngx.status = 200
        ngx.say(_jsonrpc_error(request_id, -32601, "Resource not found: " .. uri))
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
    local _, lookup = _proxy_list_to_backends("prompts/list", "prompts",
        mapping, client_session_id, server_id)
    if not lookup then
        ngx.status = 403
        ngx.say(_jsonrpc_error(request_id, -32603, "Backend access denied"))
        return
    end

    local backend_loc = lookup and lookup[name]
    if not backend_loc then
        ngx.status = 200
        ngx.say(_jsonrpc_error(request_id, -32601, "Prompt not found: " .. name))
        return
    end

    _proxy_to_backend(
        request_id, "prompts/get", params,
        backend_loc, client_session_id, server_id,
        nil, "prompt:" .. name
    )
end


-- Main entry point
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

    -- Handle ping: simple echo (no mapping needed)
    if method == "ping" then
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

    -- Validate client session: per MCP spec, servers that require a session ID
    -- SHOULD respond with 400 Bad Request to requests without a valid Mcp-Session-Id.
    -- Initialize and ping are exempt; notifications already handled above with 202.
    -- The session must also belong to the authenticated caller AND have been
    -- minted for this virtual server -- presenting another user's session ID,
    -- or replaying a session from a different virtual server, is rejected here
    -- before any session context is loaded or routed.
    if not _validate_client_session(client_session_id, auth_user, "/virtual/" .. server_id) then
        ngx.status = 400
        ngx.header["Content-Type"] = "application/json"
        ngx.say(_jsonrpc_error(request_id, -32600,
            "Missing or invalid Mcp-Session-Id. Send an initialize request first."))
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
        ngx.say(_handle_tools_list(request_id, mapping, user_scopes_str, client_session_id, server_id))

    elseif method == "tools/call" then
        _handle_tools_call(request_id, mapping, params, user_scopes_str, client_session_id, server_id)

    elseif method == "resources/list" then
        -- Enforce server-level required_scopes
        if not _has_scopes(user_scopes_str, mapping.required_scopes) then
            ngx.status = 200
            ngx.say(_jsonrpc_error(request_id, -32603, "Access denied: missing required server scopes"))
            return
        end
        local resources = _proxy_list_to_backends("resources/list", "resources",
            mapping, client_session_id, server_id)
        if not resources then
            ngx.status = 403
            ngx.say(_jsonrpc_error(request_id, -32603, "Backend access denied"))
            return
        end
        ngx.status = 200
        ngx.say(_jsonrpc_result(request_id, { resources = _as_json_array(resources) }))

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
        local prompts = _proxy_list_to_backends("prompts/list", "prompts",
            mapping, client_session_id, server_id)
        if not prompts then
            ngx.status = 403
            ngx.say(_jsonrpc_error(request_id, -32603, "Backend access denied"))
            return
        end
        ngx.status = 200
        ngx.say(_jsonrpc_result(request_id, { prompts = _as_json_array(prompts) }))

    elseif method == "prompts/get" then
        -- Enforce server-level required_scopes
        if not _has_scopes(user_scopes_str, mapping.required_scopes) then
            ngx.status = 200
            ngx.say(_jsonrpc_error(request_id, -32603, "Access denied: missing required server scopes"))
            return
        end
        _handle_prompts_get(request_id, params, mapping, client_session_id, server_id)

    else
        ngx.status = 200
        ngx.say(_jsonrpc_error(request_id, -32601, "Method not found: " .. tostring(method)))
    end
end

-- Test hook: when loaded by the Lua unit test (which sets _G._VR_TEST), expose
-- internal helpers and return the module WITHOUT executing the router. In
-- production _G._VR_TEST is nil, so this branch is skipped and route() runs.
if _G._VR_TEST then
    _M._fetch_backend_tools_list = _fetch_backend_tools_list
    _M._append_mapping_tools_for_backend = _append_mapping_tools_for_backend
    _M._handle_tools_list = _handle_tools_list
    _M._handle_tools_call = _handle_tools_call
    _M._get_backend_session = _get_backend_session
    _M._proxy_to_backend = _proxy_to_backend
    _M._forward_identity_headers = _forward_identity_headers
    return _M
end

-- Execute routing
_M.route()
