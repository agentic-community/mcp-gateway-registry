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


-- Ensure inputSchema has "type": "object" as required by MCP spec
local function _ensure_mcp_schema(schema)
    if not schema or type(schema) ~= "table" then
        return { type = "object", properties = {} }
    end
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


-- Initialize a backend MCP server and extract its session ID
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

    -- Clear the client's Mcp-Session-Id so the backend sees a fresh
    -- initialize request instead of trying to resume a vs-* session.
    ngx.req.set_header("Mcp-Session-Id", "")
    -- MCP spec requires Accept header listing both content types
    ngx.req.set_header("Accept", "application/json, text/event-stream")

    local res = ngx.location.capture(backend_location, {
        method = ngx.HTTP_POST,
        body = init_body,
    })

    if not res or res.status ~= 200 then
        ngx.log(ngx.ERR, "Backend initialize failed for ", backend_location,
            " status=", res and res.status or "nil")
        return nil
    end

    -- Extract Mcp-Session-Id from backend response headers
    local backend_session_id = nil
    if res.header then
        backend_session_id = res.header["Mcp-Session-Id"] or res.header["mcp-session-id"]
    end

    return backend_session_id
end


-- Get or create a backend session for a given client session + backend location.
-- Two-tier cache: L1 shared dict (30s) -> L2 MongoDB -> initialize backend
-- client_session_id is always a gate-validated ^vs-[0-9a-f]+$ id here (route()
-- allowlists it before any method that reaches this function), and
-- backend_location comes from the trusted mapping file -- so session_key has no
-- attacker-injectable query/path metacharacters.
local function _get_backend_session(client_session_id, backend_location, server_id)
    local session_key = client_session_id .. ":" .. backend_location
    local cache_key = "bsess:" .. session_key

    -- L1: shared dict fast path
    local session_id = session_cache:get(cache_key)
    if session_id then
        return session_id
    end

    -- L2: MongoDB via internal FastAPI API, bound to the authenticated owner
    -- so a backend session belonging to a different user is never returned
    -- (defense in depth behind the client-session gate). Escape the session-id
    -- segment as well -- a no-op for valid vs-<hex> ids, but it keeps the
    -- subrequest URI safe regardless of how the key was built.
    local owner = _auth_user_id()
    local owner_qs = owner and ("?user_id=" .. ngx.escape_uri(owner)) or ""
    local backend_path = ngx.escape_uri(client_session_id) .. ":" .. backend_location
    local res = ngx.location.capture(
        "/_internal/sessions/backend/" .. backend_path .. owner_qs, {
            method = ngx.HTTP_GET,
        })
    if res and res.status == 200 then
        local ok, data = pcall(cjson.decode, res.body)
        if ok and data.backend_session_id then
            -- Populate L1 cache
            session_cache:set(cache_key, data.backend_session_id, SESSION_CACHE_TTL)
            return data.backend_session_id
        end
    end

    -- L2 miss: initialize the backend to get a session
    ngx.log(ngx.INFO, "Initializing backend session for ", session_key)
    session_id = _initialize_backend(backend_location)

    if session_id then
        -- Store in L2 (MongoDB). Store the resolved owner ("anonymous" only as
        -- a last-resort audit label; the gate rejects identity-less requests
        -- before any backend session is created).
        local user_id = owner or "anonymous"
        local store_body = cjson.encode({
            backend_session_id = session_id,
            client_session_id = client_session_id,
            user_id = user_id,
            virtual_server_path = "/virtual/" .. server_id,
        })
        ngx.location.capture("/_internal/sessions/backend/" .. backend_path, {
            method = ngx.HTTP_PUT,
            body = store_body,
        })
        -- Populate L1 cache
        session_cache:set(cache_key, session_id, SESSION_CACHE_TTL)
    end

    return session_id
end


-- Invalidate a backend session from both L1 and L2 caches
local function _invalidate_backend_session(client_session_id, backend_location)
    local session_key = client_session_id .. ":" .. backend_location
    local cache_key = "bsess:" .. session_key

    -- Remove from L1 (uses the raw key, matching how _get_backend_session
    -- populated the shared dict)
    session_cache:delete(cache_key)

    -- Remove from L2, scoped to the authenticated owner (symmetric with the GET
    -- lookup). Escape the session-id segment in the subrequest URI for the same
    -- defense-in-depth reason as the GET/PUT paths (a no-op for valid vs-<hex>
    -- ids, which is all that reaches here past the route() allowlist).
    local backend_path = ngx.escape_uri(client_session_id) .. ":" .. backend_location
    local owner = _auth_user_id()
    local owner_qs = owner and ("?user_id=" .. ngx.escape_uri(owner)) or ""
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


-- Fetch tools/list from a single backend via ngx.location.capture.
-- Returns the tools array from the backend, or empty table on failure.
-- On stale session error (status >= 400), invalidates and retries once.
local function _fetch_backend_tools_list(backend_location, client_session_id, server_id)
    local req_body = cjson.encode({
        jsonrpc = "2.0",
        id = "tl-" .. (ngx.var.request_id or "0"),
        method = "tools/list",
        params = {},
    })

    -- Get backend session
    local backend_session_id = nil
    if client_session_id then
        backend_session_id = _get_backend_session(client_session_id, backend_location, server_id)
    end

    if backend_session_id then
        ngx.req.set_header("Mcp-Session-Id", backend_session_id)
    else
        ngx.req.set_header("Mcp-Session-Id", "")
    end

    local res = ngx.location.capture(backend_location, {
        method = ngx.HTTP_POST,
        body = req_body,
    })

    -- Stale session retry
    if res and res.status >= 400 and client_session_id and backend_session_id then
        ngx.log(ngx.WARN, "Backend tools/list returned ", res.status,
            " for ", backend_location, " -- retrying with fresh session")
        _invalidate_backend_session(client_session_id, backend_location)
        local new_session_id = _get_backend_session(client_session_id, backend_location, server_id)
        if new_session_id then
            ngx.req.set_header("Mcp-Session-Id", new_session_id)
        else
            ngx.req.set_header("Mcp-Session-Id", "")
        end
        res = ngx.location.capture(backend_location, {
            method = ngx.HTTP_POST,
            body = req_body,
        })
    end

    if not res or res.status ~= 200 then
        ngx.log(ngx.ERR, "Failed to fetch tools/list from ", backend_location,
            " status=", res and res.status or "nil")
        return {}
    end

    -- Backend may respond with SSE format (text/event-stream) or raw JSON
    local json_body = _parse_sse_body(res.body)
    if not json_body then
        ngx.log(ngx.ERR, "Empty or unparseable tools/list response from ", backend_location)
        return {}
    end

    local ok, data = pcall(cjson.decode, json_body)
    if not ok then
        ngx.log(ngx.ERR, "Failed to parse tools/list response from ", backend_location)
        return {}
    end

    if data.result and data.result.tools then
        return data.result.tools
    end

    return {}
end


-- Handle tools/list method - proxy to backends for full metadata, with cache
local function _handle_tools_list(request_id, mapping, user_scopes_str, client_session_id, server_id)
    -- Enforce server-level required_scopes before processing
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

    -- L1 cache check: enriched tools for this server
    local enriched_tools = nil
    local enriched_cache_key = "tools_enriched:" .. (server_id or "unknown")
    local cached_enriched = session_cache:get(enriched_cache_key)
    if cached_enriched then
        local ok, cached = pcall(cjson.decode, cached_enriched)
        if ok then
            enriched_tools = cached
        end
    end

    -- On cache miss, fetch from backends
    if not enriched_tools then
        enriched_tools = {}
        local backend_locations = _collect_backend_locations(mapping)
        local fetch_ok = false

        for _, backend_loc in ipairs(backend_locations) do
            local backend_tools = _fetch_backend_tools_list(backend_loc, client_session_id, server_id)
            if #backend_tools > 0 then
                fetch_ok = true
            end

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
        end

        -- Fallback: if all backend fetches failed, use mapping file metadata
        if not fetch_ok then
            ngx.log(ngx.WARN, "All backend tools/list fetches failed for server=", server_id,
                " -- falling back to mapping file metadata")
            enriched_tools = {}
            if mapping.tools then
                for _, tool in ipairs(mapping.tools) do
                    enriched_tools[#enriched_tools + 1] = {
                        name = tool.name,
                        description = tool.description or "",
                        inputSchema = _ensure_mcp_schema(tool.inputSchema),
                        required_scopes = tool.required_scopes,
                    }
                end
            end
        end

        -- Cache enriched tools (pre-scope-filtered, 60s TTL)
        local ok_enc, encoded = pcall(cjson.encode, enriched_tools)
        if ok_enc then
            session_cache:set(enriched_cache_key, encoded, ENRICHED_CACHE_TTL)
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


-- Generic helper to proxy list methods (resources/list, prompts/list) to all backends.
-- Aggregates results from all backends into a single array.
-- Caches with key "{method}:{server_id}", 60s TTL.
-- Returns the aggregated array and a lookup map (item_key_value -> backend_location).
local function _proxy_list_to_backends(method_name, result_key, mapping, client_session_id, server_id)
    -- Cache check
    local cache_key = method_name .. ":" .. (server_id or "unknown")
    local cached = session_cache:get(cache_key)
    if cached then
        local ok, data = pcall(cjson.decode, cached)
        if ok then
            -- Ensure decoded items is always a JSON array (empty table from cache loses metatable)
            if data.items and #data.items == 0 then
                data.items = setmetatable({}, empty_array_mt)
            end
            return data.items, data.lookup
        end
    end

    local aggregated = setmetatable({}, empty_array_mt)
    local lookup = {}
    local backend_locations = _collect_backend_locations(mapping)

    for _, backend_loc in ipairs(backend_locations) do
        local req_body = cjson.encode({
            jsonrpc = "2.0",
            id = "pl-" .. (ngx.var.request_id or "0"),
            method = method_name,
            params = {},
        })

        -- Get backend session
        local backend_session_id = nil
        if client_session_id then
            backend_session_id = _get_backend_session(client_session_id, backend_loc, server_id)
        end

        if backend_session_id then
            ngx.req.set_header("Mcp-Session-Id", backend_session_id)
        else
            ngx.req.set_header("Mcp-Session-Id", "")
        end

        local res = ngx.location.capture(backend_loc, {
            method = ngx.HTTP_POST,
            body = req_body,
        })

        -- Stale session retry
        if res and res.status >= 400 and client_session_id and backend_session_id then
            ngx.log(ngx.WARN, "Backend ", method_name, " returned ", res.status,
                " for ", backend_loc, " -- retrying with fresh session")
            _invalidate_backend_session(client_session_id, backend_loc)
            local new_session_id = _get_backend_session(client_session_id, backend_loc, server_id)
            if new_session_id then
                ngx.req.set_header("Mcp-Session-Id", new_session_id)
            else
                ngx.req.set_header("Mcp-Session-Id", "")
            end
            res = ngx.location.capture(backend_loc, {
                method = ngx.HTTP_POST,
                body = req_body,
            })
        end

        if res and res.status == 200 then
            local json_body = _parse_sse_body(res.body)
            local ok, data = pcall(cjson.decode, json_body or "")
            if ok and data.result and data.result[result_key] then
                for _, item in ipairs(data.result[result_key]) do
                    aggregated[#aggregated + 1] = item
                    -- Build lookup: for resources, key on "uri"; for prompts, key on "name"
                    local lookup_key = nil
                    if result_key == "resources" and item.uri then
                        lookup_key = item.uri
                    elseif result_key == "prompts" and item.name then
                        lookup_key = item.name
                    end
                    if lookup_key then
                        lookup[lookup_key] = backend_loc
                    end
                end
            end
        else
            ngx.log(ngx.WARN, "Backend ", method_name, " failed for ", backend_loc,
                " status=", res and res.status or "nil")
        end
    end

    -- Cache aggregated results and lookup map
    local cache_data = { items = aggregated, lookup = lookup }
    local ok_enc, encoded = pcall(cjson.encode, cache_data)
    if ok_enc then
        session_cache:set(cache_key, encoded, ENRICHED_CACHE_TTL)
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

    -- Get or create backend session
    local backend_session_id = nil
    if client_session_id then
        backend_session_id = _get_backend_session(client_session_id, backend_location, server_id)
    end

    -- Set version header if pinned
    if backend_version then
        ngx.req.set_header("X-MCP-Server-Version", backend_version)
    end

    -- Set the backend session header for the subrequest proxy
    if backend_session_id then
        ngx.req.set_header("Mcp-Session-Id", backend_session_id)
    else
        ngx.req.set_header("Mcp-Session-Id", "")
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

    -- Stale session retry: if backend returns an error that looks like a session issue,
    -- invalidate the session and retry once
    if res.status >= 400 and client_session_id and backend_session_id then
        ngx.log(ngx.WARN, "Backend returned ", res.status, " for ", label or method_name,
            " session=", backend_session_id, " -- retrying with fresh session")

        -- Invalidate stale session
        _invalidate_backend_session(client_session_id, backend_location)

        -- Get a fresh session (will re-initialize the backend)
        local new_session_id = _get_backend_session(client_session_id, backend_location, server_id)
        if new_session_id then
            ngx.req.set_header("Mcp-Session-Id", new_session_id)
        else
            ngx.req.set_header("Mcp-Session-Id", "")
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
    if mapping.tools then
        for _, tool_entry in ipairs(mapping.tools) do
            if tool_entry.name == tool_name then
                if not _has_scopes(user_scopes_str, tool_entry.required_scopes) then
                    ngx.status = 200
                    ngx.say(_jsonrpc_error(request_id, -32603,
                        "Access denied: missing required scopes for tool: " .. tool_name))
                    return
                end
                break
            end
        end
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

    -- Look up which backend owns this resource from cached resources/list
    local _, lookup = _proxy_list_to_backends("resources/list", "resources",
        mapping, client_session_id, server_id)

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

    -- Look up which backend owns this prompt from cached prompts/list
    local _, lookup = _proxy_list_to_backends("prompts/list", "prompts",
        mapping, client_session_id, server_id)

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

-- Execute routing
_M.route()
