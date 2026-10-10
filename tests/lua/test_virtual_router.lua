-- Regression tests for virtual MCP backend aggregation and session management.
-- Covers discovery fallback, empty/truncated responses, per-user egress tools,
-- stateless/stateful initialization, and backend authorization before dispatch.
--
-- Run from the repo root with OpenResty's resty CLI (provides lua-cjson):
--   resty tests/lua/test_virtual_router.lua
-- Or via Docker without a local OpenResty install:
--   docker run --rm -v "$PWD":/app -w /app openresty/openresty:alpine \
--     resty tests/lua/test_virtual_router.lua
--
-- The router file is a content_by_lua script; it exposes its internal helpers
-- and returns the module (instead of executing) when _G._VR_TEST is set.

_G._VR_TEST = true

local cjson = require("cjson")

-- Environment guard (issue #1532): the empty-array fix relies on
-- cjson.empty_array_mt, an OpenResty lua-cjson extension. If this suite is ever
-- run against a cjson without it (e.g. Debian lua-cjson 2.1.0, which is what the
-- registry image used to ship), the schema-array assertions below would give a
-- false sense of safety. Fail loudly instead of passing on the wrong runtime.
assert(cjson.empty_array_mt,
    "cjson.empty_array_mt is missing -- these tests require OpenResty's lua-cjson")

local failures = 0
local function check(cond, msg)
    if cond then
        print("  ok   - " .. msg)
    else
        failures = failures + 1
        print("  FAIL - " .. msg)
    end
end

-- Minimal ngx.shared dict mock.
local function _new_dict()
    local store = {}
    return {
        get = function(_, k) return store[k] end,
        set = function(_, k, v) store[k] = v; return true end,
        delete = function(_, k) store[k] = nil end,
        _store = store,
    }
end

local dict = _new_dict()

-- Programmable ngx.location.capture responses, keyed by backend location.
local capture_responses = {}
local capture_handler = nil


_G.ngx = {
    shared = { virtual_server_map = dict },
    location = {
        capture = function(loc, opts)
            if capture_handler then return capture_handler(loc, opts) end
            if loc:sub(1, 9) == "/_vs_auth" then
                return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
            end
            return capture_responses[loc]
        end,
    },
    -- Stateful request-header mock so tests can seed a client-supplied header
    -- and observe whether set_header overwrote it or clear_header removed it.
    req = {
        _headers = {},
        set_header = function(k, v) _G.ngx.req._headers[k] = v end,
        clear_header = function(k) _G.ngx.req._headers[k] = nil end,
        get_headers = function() return _G.ngx.req._headers end,
        read_body = function() end,
        get_body_data = function() return _G.ngx.req._body end,
    },
    log = function() end,
    ERR = 4,
    WARN = 5,
    HTTP_POST = 8,
    HTTP_GET = 2,
    HTTP_PUT = 16,
    HTTP_DELETE = 32,
    escape_uri = function(v) return v end,
    var = { request_id = "0", auth_user = "alice" },
    status = 200,
    say = function(body) _G.ngx._last_body = body end,
    exit = function() end,
    print = function(body) _G.ngx._last_body = body end,
    header = {},
    -- Controllable clock: tests advance _now to simulate a slow initialize.
    _now = 0,
    now = function() return _G.ngx._now end,

}

-- Load the router with the test hook active; it returns the module table.
local M = assert(loadfile("docker/lua/virtual_router.lua"))()

-- ---------------------------------------------------------------------------
print("test: _append_mapping_tools_for_backend appends only the given backend")
do
    local mapping = { tools = {
        { name = "a", backend_location = "/b1", inputSchema = { type = "object" } },
        { name = "b", backend_location = "/b2" },
    } }
    local enriched = {}
    M._append_mapping_tools_for_backend(enriched, mapping, "/b1")
    check(#enriched == 1, "only one tool appended for /b1")
    check(enriched[1] and enriched[1].name == "a", "the appended tool is 'a'")
end

-- ---------------------------------------------------------------------------
print("test: _fetch_backend_list classifies every backend outcome the same way")
do
    local function fetch(response, method, key)
        capture_responses["/_vs_backend_loc"] = response
        return M._fetch_backend_list("/_vs_backend_loc", method or "tools/list", key or "tools",
            nil, "srv")
    end

    local outcome, items = fetch({ status = 200,
        body = cjson.encode({ result = { tools = { { name = "x" } } } }) })
    check(outcome == "ok" and #items == 1, "200 + tools -> ok with items")

    outcome, items = fetch({ status = 200, body = cjson.encode({ result = { tools = {} } }) })
    check(outcome == "ok" and #items == 0, "200 + empty tools -> ok, empty")

    capture_responses["/_vs_backend_loc"] = { status = 200,
        header = { ["x-egress-consent-required"] = "pat" },
        body = cjson.encode({ result = { resources = {} } }) }
    outcome = M._fetch_backend_list("/_vs_backend_loc", "resources/list", "resources",
        nil, "srv", true)
    check(outcome == "consent", "broker consent marker (any header casing) -> consent")

    outcome = fetch({ status = 200, body = cjson.encode({
        error = { code = -32601, message = "Method not found" } }) }, "prompts/list", "prompts")
    check(outcome == "unsupported", "-32601 -> unsupported")

    -- Only the gateway's grant check denies; the backend's own refusal is a
    -- failure of that backend (a registrant must not be able to blank siblings).
    outcome = fetch({ status = 200, body = cjson.encode({
        error = { code = -32603, message = "Access denied" } }) })
    check(outcome == "failed", "backend JSON-RPC access error -> failed")

    outcome = fetch({ status = 403, body = "" })
    check(outcome == "failed", "backend 403 -> failed")

    capture_handler = function(loc)
        if loc == "/_vs_auth_loc" then return { status = 403 } end
    end
    outcome = fetch(nil)
    check(outcome == "denied", "gateway grant refusal -> denied")
    capture_handler = nil

    outcome = fetch({ status = 200, header = { ["X-Egress-Consent-Required"] = "pat" },
        body = cjson.encode({ result = { tools = { { name = "x" } } } }) })
    check(outcome == "ok", "consent marker from a plain backend is ignored")

    outcome = fetch({ status = 500, body = "" })
    check(outcome == "failed", "500 -> failed")

    outcome = fetch({ status = 200, truncated = true,
        body = cjson.encode({ result = { tools = { { name = "x" } } } }) })
    check(outcome == "failed", "truncated -> failed")

    outcome = fetch({ status = 200, body = cjson.encode({ result = {} }) })
    check(outcome == "failed", "missing list key -> failed")
end

-- ---------------------------------------------------------------------------
print("test: _handle_tools_list falls back per-backend and does NOT cache partial")
do
    dict._store["tools_enriched::srv1:5:alice"] = nil
    local mapping = { required_scopes = nil, tools = {
        { name = "live_tool", original_name = "live_tool", backend_location = "/_vs_backend_ok",
          inputSchema = { type = "object" } },
        { name = "down_tool", original_name = "down_tool", backend_location = "/_vs_backend_down",
          inputSchema = { type = "object" } },
    } }
    capture_responses = {}
    capture_responses["/_vs_backend_ok"] = { status = 200,
        body = cjson.encode({ result = { tools = { { name = "live_tool", description = "live" } } } }) }
    capture_responses["/_vs_backend_down"] = { status = 500, body = "" }

    local resp = M._handle_tools_list("1", mapping, "", nil, "srv1")
    local decoded = cjson.decode(resp)
    local names = {}
    for _, t in ipairs(decoded.result.tools) do names[t.name] = true end
    check(names["live_tool"] == true, "live backend tool present")
    check(names["down_tool"] == true, "failed backend tool present via fallback")
    check(dict._store["tools_enriched::srv1:5:alice"] == nil, "partial/fallback result is NOT cached")
end

-- ---------------------------------------------------------------------------
print("test: egress discovery failure never presents mapping-only tools as available")
do
    ngx.var.auth_user = "alice"
    local mapping = { tools = { { name = "alias", original_name = "underlying",
        backend_location = "/_vs_backend_no_consent", egress_auth_mode = "oauth" } } }
    capture_handler = function(loc)
        if loc == "/_vs_auth_no_consent" then
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc == "/_vs_backend_no_consent" then
            return { status = 401, body = "" }
        end
    end
    local response = cjson.decode(M._handle_tools_list(1, mapping, "", nil, "no_consent"))
    check(response.error ~= nil, "OAuth consent failure does not masquerade as discoverable tool")
    check(dict:get("tools_enriched::no_consent:5:alice") == nil,
        "OAuth consent failure is never cached")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------

print("test: _handle_tools_list caches when every backend succeeds")
do
    dict._store["tools_enriched::srv2:5:alice"] = nil
    local mapping = { required_scopes = nil, tools = {
        { name = "live_tool", original_name = "live_tool", backend_location = "/_vs_backend_ok",
          inputSchema = { type = "object" } },
        { name = "down_tool", original_name = "down_tool", backend_location = "/_vs_backend_down",
          inputSchema = { type = "object" } },
    } }
    capture_responses = {}
    capture_responses["/_vs_backend_ok"] = { status = 200,
        body = cjson.encode({ result = { tools = { { name = "live_tool" } } } }) }
    capture_responses["/_vs_backend_down"] = { status = 200,
        body = cjson.encode({ result = { tools = { { name = "down_tool" } } } }) }

    M._handle_tools_list("1", mapping, "", nil, "srv2")
    check(dict._store["tools_enriched::srv2:5:alice"] ~= nil, "fully-discovered result IS cached")
end

-- ---------------------------------------------------------------------------
print("test: _forward_identity_headers overwrites a spoofed client X-User")
do
    -- Client tries to spoof identity; a validated user is present.
    ngx.req._headers = { ["X-User"] = "attacker", ["X-Username"] = "attacker" }
    ngx.var.auth_user = "alice"
    ngx.var.auth_username = "alice@corp"
    M._forward_identity_headers()
    check(ngx.req._headers["X-User"] == "alice",
        "validated auth_user overwrites the client-supplied X-User")
    check(ngx.req._headers["X-Username"] == "alice@corp",
        "validated auth_username overwrites the client-supplied X-Username")
end

-- ---------------------------------------------------------------------------
print("test: _forward_identity_headers CLEARS a spoofed X-User when auth_user is empty")
do
    -- M2M / client-credentials token: authenticates but carries no user, so
    -- nginx auth_request_set yields "" for $auth_user. A client-supplied X-User
    -- must NOT survive to the backend (this is the fail-open bug from #1627).
    ngx.req._headers = { ["X-User"] = "attacker", ["X-Username"] = "attacker" }
    ngx.var.auth_user = ""
    ngx.var.auth_username = ""
    M._forward_identity_headers()
    check(ngx.req._headers["X-User"] == nil,
        "empty auth_user clears the client-supplied X-User (no spoof passthrough)")
    check(ngx.req._headers["X-Username"] == nil,
        "empty auth_username clears the client-supplied X-Username")
end

-- ---------------------------------------------------------------------------
print("test: _forward_identity_headers CLEARS a spoofed X-User when auth_user is nil")
do
    ngx.req._headers = { ["X-User"] = "attacker", ["X-Username"] = "attacker" }
    ngx.var.auth_user = nil
    ngx.var.auth_username = nil
    M._forward_identity_headers()
    check(ngx.req._headers["X-User"] == nil,
        "nil auth_user clears the client-supplied X-User")
    check(ngx.req._headers["X-Username"] == nil,
        "nil auth_username clears the client-supplied X-Username")
end

ngx.var.auth_user = "alice"


-- ---------------------------------------------------------------------------
print("test: _handle_tools_list keeps empty schema arrays as [] (issue #1532)")
do
    dict._store["tools_enriched::srv3:5:alice"] = nil
    local mapping = { required_scopes = nil, tools = {
        { name = "no_arg", original_name = "no_arg", backend_location = "/_vs_backend_ok" },
        { name = "one_arg", original_name = "one_arg", backend_location = "/_vs_backend_ok" },
    } }
    capture_responses = {}
    capture_responses["/_vs_backend_ok"] = { status = 200, body = cjson.encode({ result = { tools = {
        { name = "no_arg", inputSchema = {
            type = "object", properties = {},
            required = setmetatable({}, cjson.empty_array_mt) } },
        { name = "one_arg", inputSchema = {
            type = "object",
            properties = { q = { type = "string",
                                 enum = setmetatable({}, cjson.empty_array_mt) } },
            required = { "q" } } },
    } } }) }

    local body = M._handle_tools_list("1", mapping, "", nil, "srv3")
    check(body:find('"required":[]', 1, true) ~= nil,
        "empty required serializes as [] not {}")
    check(body:find('"required":["q"]', 1, true) ~= nil,
        "non-empty required is still an array")
    check(body:find('"properties":{}', 1, true) ~= nil,
        "empty properties stays an object")
    check(body:find('"enum":[]', 1, true) ~= nil,
        "empty enum nested under properties serializes as []")

    -- The cached path decodes and re-encodes, so it must hold the same shape.
    local cached_body = M._handle_tools_list("1", mapping, "", nil, "srv3")
    check(dict._store["tools_enriched::srv3:5:alice"] ~= nil, "result was cached")
    check(cached_body:find('"required":[]', 1, true) ~= nil,
        "empty required is still [] when served from cache")
end

-- ---------------------------------------------------------------------------
print("test: a property named like a schema keyword stays an object")
do
    dict._store["tools_enriched::srv4:5:alice"] = nil
    local mapping = { required_scopes = nil, tools = {
        { name = "odd", original_name = "odd", backend_location = "/_vs_backend_ok" },
    } }
    capture_responses = {}
    capture_responses["/_vs_backend_ok"] = { status = 200, body = cjson.encode({ result = { tools = {
        { name = "odd", inputSchema = {
            type = "object", properties = { required = { type = "object" } } } },
    } } }) }

    local body = M._handle_tools_list("1", mapping, "", nil, "srv4")
    check(body:find('"required":[]', 1, true) == nil,
        "a property called 'required' is not coerced into an array")
end

-- ---------------------------------------------------------------------------
print("test: successful stateless initialize persists through L1 and L2")
do
    local initialized, lookups, puts, calls = 0, 0, 0, 0
    local saved = nil
    ngx.var.auth_user = "alice"
    capture_handler = function(loc, opts)
        if loc:find("/_internal/sessions/backend/", 1, true) == 1 then
            if opts.method == ngx.HTTP_GET then
                lookups = lookups + 1
                return saved and { status = 200, body = saved } or { status = 404 }
            end
            if opts.method == ngx.HTTP_PUT then
                puts = puts + 1
                saved = opts.body
                return { status = 200 }
            end
        end
        if loc == "/_vs_auth_stateless" then
            local method = cjson.decode(ngx.req._headers["X-Body"]).method
            check(method == "initialize" or method == "tools/call",
                "backend grant receives the rewritten RPC method")
            check(opts.method == ngx.HTTP_GET, "backend grant uses GET /validate")
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc == "/_vs_backend_stateless" then
            local method = cjson.decode(opts.body).method
            if method == "initialize" then
                initialized = initialized + 1
                return { status = 200, body = '{"jsonrpc":"2.0","result":{"capabilities":{}}}' }
            end
            calls = calls + 1
            check(ngx.req._headers["Mcp-Session-Id"] == "", "stateless call has no session header")
            return { status = 200, body = '{"jsonrpc":"2.0","result":{}}' }
        end
    end
    M._proxy_to_backend(1, "tools/call", { name = "original" }, "/_vs_backend_stateless", "vs-a", "srv")
    M._proxy_to_backend(2, "tools/call", { name = "original" }, "/_vs_backend_stateless", "vs-a", "srv")
    check(initialized == 1 and calls == 2 and lookups == 1 and puts == 1,
        "repeated calls reuse successful stateless initialization")
    dict:delete("bsess_stateless:vs-a:/_vs_backend_stateless:5:alice")
    M._proxy_to_backend(3, "tools/call", { name = "original" }, "/_vs_backend_stateless", "vs-a", "srv")
    check(initialized == 1 and calls == 3 and lookups == 2,
        "L2 stateless marker survives L1 expiration")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: failed initialize is retried, not cached as stateless")
do
    local initialized, calls = 0, 0
    capture_handler = function(loc, opts)
        if loc:find("/_internal/sessions/backend/", 1, true) == 1 then
            return { status = 404 }
        end
        if loc == "/_vs_auth_failure" then
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc == "/_vs_backend_failure" then
            if cjson.decode(opts.body).method == "initialize" then
                initialized = initialized + 1
                return { status = 200, body = '{"jsonrpc":"2.0","error":{"code":-32000}}' }
            end
            calls = calls + 1
        end
    end
    M._proxy_to_backend(1, "tools/call", {}, "/_vs_backend_failure", "vs-b", "srv")
    M._proxy_to_backend(2, "tools/call", {}, "/_vs_backend_failure", "vs-b", "srv")
    check(initialized == 2 and calls == 0, "failed initialize retries without calling tool")
    check(dict:get("bsess_stateless:vs-b:/_vs_backend_failure:5:alice") == nil,
        "failure has no stateless marker")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: stateful backend session reuses ID and refreshes stale session")
do
    local initialized, calls, deletes, ids = 0, 0, 0, {}
    capture_handler = function(loc, opts)
        if loc:find("/_internal/sessions/backend/", 1, true) == 1 then
            if opts.method == ngx.HTTP_GET then return { status = 404 } end
            if opts.method == ngx.HTTP_DELETE then deletes = deletes + 1 end
            return { status = 200 }
        end
        if loc == "/_vs_auth_stateful" then
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc == "/_vs_backend_stateful" then
            if cjson.decode(opts.body).method == "initialize" then
                initialized = initialized + 1
                return { status = 200, header = { ["Mcp-Session-Id"] = "sid-" .. initialized },
                    body = '{"jsonrpc":"2.0","result":{"capabilities":{}}}' }
            end
            calls = calls + 1
            ids[calls] = ngx.req._headers["Mcp-Session-Id"]
            if calls == 2 then return { status = 404, body = "" } end
            return { status = 200, body = '{"jsonrpc":"2.0","result":{}}' }
        end
    end
    M._proxy_to_backend(1, "tools/call", {}, "/_vs_backend_stateful", "vs-c", "srv")
    M._proxy_to_backend(2, "tools/call", {}, "/_vs_backend_stateful", "vs-c", "srv")
    check(initialized == 2 and calls == 3 and deletes == 1, "stale ID reinitializes once")
    check(ids[1] == "sid-1" and ids[2] == "sid-1" and ids[3] == "sid-2",
        "stateful calls use own backend session, then refreshed ID")
    capture_handler = nil
end

print("test: null version also survives stateful session invalidation")
do
    local initialized, calls, deletes = 0, 0, 0
    capture_handler = function(loc, opts)
        if loc == "/_vs_auth_null_stale" then
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc:find("/_internal/sessions/backend/", 1, true) == 1 then
            if opts.method == ngx.HTTP_GET then return { status = 404 } end
            if opts.method == ngx.HTTP_DELETE then deletes = deletes + 1 end
            return { status = 200 }
        end
        if loc == "/_vs_backend_null_stale" then
            if cjson.decode(opts.body).method == "initialize" then
                initialized = initialized + 1
                return { status = 200, header = { ["Mcp-Session-Id"] = "sid-" .. initialized },
                    body = '{"jsonrpc":"2.0","result":{}}' }
            end
            calls = calls + 1
            if calls == 1 then return { status = 404, body = "" } end
            return { status = 200, body = '{"jsonrpc":"2.0","result":{}}' }
        end
    end
    M._proxy_to_backend(1, "tools/call", { name = "get_me" },
        "/_vs_backend_null_stale", "vs-null-stale", "srv", cjson.null)
    check(initialized == 2 and calls == 2 and deletes == 1,
        "null version reinitializes a rejected stateful backend session")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: backend sessions are isolated by selected version")
do
    local reads = 0
    local selected_versions = {}
    ngx.var.auth_user = "alice"
    capture_handler = function(loc, opts)
        if loc:find("/_internal/sessions/backend/", 1, true) == 1 then
            if opts.method == ngx.HTTP_GET then
                reads = reads + 1
                return { status = 404 }
            end
            return { status = 200 }
        end
        if loc == "/_vs_auth_versioned" then
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc == "/_vs_backend_versioned" then
            if cjson.decode(opts.body).method == "initialize" then
                selected_versions[#selected_versions + 1] = ngx.req._headers["X-MCP-Server-Version"]
                return { status = 200,
                    header = { ["Mcp-Session-Id"] = "sid-" .. selected_versions[#selected_versions] },
                    body = '{"jsonrpc":"2.0","result":{}}' }
            end
            return { status = 200, body = '{"jsonrpc":"2.0","result":{}}' }
        end
    end
    M._proxy_to_backend(1, "tools/call", {}, "/_vs_backend_versioned", "vs-d", "srv", "v1")
    M._proxy_to_backend(2, "tools/call", {}, "/_vs_backend_versioned", "vs-d", "srv", "v2")
    check(reads == 2 and selected_versions[1] == "v1" and selected_versions[2] == "v2",
        "each pinned version has its own backend initialize and session")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
-- Session caches stay owner-scoped even if a valid-looking client session ID is
-- supplied across identities. The owner-bound L2 query still protects a miss.
do
    local reads = 0
    ngx.var.auth_user = "mallory"
    capture_handler = function(loc, opts)
        if loc:find("/_internal/sessions/backend/", 1, true) == 1 then
            reads = reads + 1
            check(loc:find("user_id=mallory", 1, true) ~= nil,
                "backend L2 lookup is owner-scoped")
            return { status = 404 }
        end
        if loc == "/_vs_auth_stateful" then return { status = 403 } end
    end
    local session_id, valid = M._get_backend_session("vs-c", "/_vs_backend_stateful", "srv")
    check(session_id == nil and not valid and reads == 1,
        "another user cannot use the cached stateful backend session")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: virtual alias dispatches original name to backing grant and upstream")
do
    local seen_auth, seen_backend = nil, nil
    ngx.var.auth_user = "alice"
    local mapping = { tool_backend_map = { alias = {
        backend_location = "/_vs_backend_alias", original_name = "underlying",
    } }, tools = { { name = "alias", backend_location = "/_vs_backend_alias" } } }
    capture_handler = function(loc, opts)
        if loc == "/_vs_auth_alias" then
            seen_auth = cjson.decode(ngx.req._headers["X-Body"]).params.name
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc == "/_vs_backend_alias" then
            seen_backend = cjson.decode(opts.body).params.name
            return { status = 200, body = '{"jsonrpc":"2.0","result":{}}' }
        end
    end
    ngx.req._headers["X-MCP-Server-Version"] = "client-selected"
    M._handle_tools_call(1, mapping, { name = "alias" }, "", nil, "srv")
    check(seen_auth == "underlying" and seen_backend == "underlying",
        "backing grant and upstream see underlying tool, not virtual alias")
    check(ngx.req._headers["X-MCP-Server-Version"] == nil,
        "client-supplied backend version is removed when mapping is unpinned")
    capture_handler = nil
end

-- A mapping file written by json.dump uses null for an unpinned version.
-- cjson decodes null as userdata, not Lua nil; it must not become a header
-- or be used with the length operator when choosing the backend session key.
print("test: unpinned JSON mapping call handles cjson.null version")
do
    local initialized, called = 0, 0
    local mapping = cjson.decode('{"tool_backend_map":{"get_me":{"backend_location":"/_vs_backend_github","original_name":"get_me","backend_version":null}},"tools":[{"name":"get_me","backend_location":"/_vs_backend_github"}]}')
    capture_handler = function(loc, opts)
        if loc == "/_vs_auth_github" then
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc:find("/_internal/sessions/backend/", 1, true) == 1 then
            if opts.method == ngx.HTTP_GET then return { status = 404 } end
            return { status = 200 }
        end
        if loc == "/_vs_backend_github" then
            if cjson.decode(opts.body).method == "initialize" then
                initialized = initialized + 1
            else
                called = called + 1
            end
            return { status = 200, body = '{"jsonrpc":"2.0","result":{}}' }
        end
    end
    ngx.req._headers["X-MCP-Server-Version"] = "caller-chosen"
    M._handle_tools_call(1, mapping, { name = "get_me" }, "", "vs-null", "srv")
    check(initialized == 1 and called == 1,
        "null mapping version initializes and calls the unpinned backend")
    check(ngx.req._headers["X-MCP-Server-Version"] == nil,
        "null mapping version clears caller-selected version header")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: user-specific tools/list cache never crosses users or freezes consent")
do
    local requests, auths = 0, 0
    local mapping = { tools = { { name = "alias", original_name = "underlying",
        backend_location = "/_vs_backend_personal", egress_auth_mode = "pat" } } }
    capture_handler = function(loc, opts)
        if loc == "/_vs_auth_personal" then
            auths = auths + 1
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc == "/_vs_backend_personal" then
            requests = requests + 1
            local name = ngx.var.auth_user == "alice" and "underlying" or "not-allowed"
            return { status = 200, body = cjson.encode({ result = { tools = {
                { name = name, description = ngx.var.auth_user },
            } } }) }
        end
    end
    ngx.var.auth_user = "alice"
    local a = cjson.decode(M._handle_tools_list(1, mapping, "", nil, "personal"))
    ngx.var.auth_user = "bob"
    local b = cjson.decode(M._handle_tools_list(2, mapping, "", nil, "personal"))
    check(a.result.tools[1].name == "alias" and a.result.tools[1].description == "alice",
        "alias and original mapping preserved for first user")
    check(#b.result.tools == 0 and requests == 2, "second user cannot see first user's tools")
    ngx.var.auth_user = "alice"
    M._handle_tools_list(3, mapping, "", nil, "personal")
    check(requests == 3 and auths == 3,
        "egress discovery never reuses cached metadata or backing grant")
    ngx.var.auth_user = "bob"
    M._handle_tools_list(4, mapping, "", nil, "personal")
    check(requests == 4, "empty consent-like discovery is not cached")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: plain backend cache rechecks authorization and isolates principals")
do
    local calls, auths = 0, 0
    local mapping = { tools = { { name = "alias", original_name = "underlying",
        backend_location = "/_vs_backend_plain", egress_auth_mode = "none" } } }
    capture_handler = function(loc, opts)
        if loc == "/_vs_auth_plain" then
            auths = auths + 1
            check(cjson.decode(ngx.req._headers["X-Body"]).method == "tools/list",
                "plain cached discovery authorizes tools/list")
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc == "/_vs_backend_plain" then
            calls = calls + 1
            return { status = 200, body = cjson.encode({ result = { tools = {
                { name = "underlying", description = ngx.var.auth_user },
            } } }) }
        end
    end
    ngx.var.auth_user = "alice"
    M._handle_tools_list(1, mapping, "", nil, "plain")
    M._handle_tools_list(2, mapping, "", nil, "plain")
    ngx.var.auth_user = "bob"
    local response = cjson.decode(M._handle_tools_list(3, mapping, "", nil, "plain"))
    check(calls == 2 and auths == 3,
        "same user caches after reauthorization, new user fetches")
    check(response.result.tools[1].description == "bob", "plain discovery never crosses users")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: denied backing grant never initializes or calls backend")
do
    local backend_calls = 0
    ngx.var.auth_user = "alice"
    capture_handler = function(loc)
        if loc == "/_vs_auth_denied" then
            check(cjson.decode(ngx.req._headers["X-Body"]).params.name == "underlying",
                "backing grant checks original tool before initialize")
            return { status = 403 }
        end
        if loc == "/_vs_backend_denied" then backend_calls = backend_calls + 1 end
        if loc:find("/_internal/sessions/backend/", 1, true) == 1 then
            return { status = 404 }
        end
    end
    ngx.status = 200
    M._proxy_to_backend(1, "tools/call", { name = "underlying" },
        "/_vs_backend_denied", "vs-denied", "srv")
    check(ngx.status == 403 and backend_calls == 0,
        "denied backing grant prevents both initialize and tool call")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: a cold initialize reuses the request token only while it is fresh")
do
    local function call_after_initialize(client_session_id, initialize_seconds)
        local auths, sent_token = 0, nil
        ngx.var.auth_user = "alice"
        capture_handler = function(loc, opts)
            if loc == "/_vs_auth_slow" then
                auths = auths + 1
                return { status = 200, header = { ["X-Internal-Token"] = "tok-" .. auths } }
            end
            if loc:find("/_internal/sessions/backend/", 1, true) == 1 then
                if opts and opts.method == ngx.HTTP_PUT then return { status = 200 } end
                return { status = 404 }
            end
            if loc == "/_vs_backend_slow" then
                local method = cjson.decode(opts.body).method
                if method == "initialize" then
                    ngx._now = ngx._now + initialize_seconds
                    return { status = 200, header = { ["Mcp-Session-Id"] = "be-1" },
                        body = cjson.encode({ jsonrpc = "2.0", id = 1, result = {} }) }
                end
                if method == "tools/call" then
                    sent_token = ngx.req._headers["X-Internal-Token"]
                end
                return { status = 200, body = cjson.encode({ jsonrpc = "2.0", id = 1,
                    result = { content = {} } }) }
            end
        end
        ngx.status = 200
        M._proxy_to_backend(1, "tools/call", { name = "underlying" },
            "/_vs_backend_slow", client_session_id, "srv")
        capture_handler = nil
        return auths, sent_token
    end

    local auths, token = call_after_initialize("vs-fast", 0)
    check(auths == 2 and token == "tok-1",
        "fast initialize restores the pre-checked token (one extra /validate for initialize)")
    auths, token = call_after_initialize("vs-slow", 10)
    check(auths == 3 and token == "tok-3",
        "slow initialize re-authorizes instead of sending a possibly expired token")
end

-- ---------------------------------------------------------------------------
print("test: backend authorization denial blocks upstream and cached discovery")
do
    local upstream = 0
    ngx.var.auth_user = "alice"
    local mapping = { tools = { { name = "alias", original_name = "underlying",
        backend_location = "/_vs_backend_blocked" } } }
    dict:set("tools_enriched::blocked:5:alice", cjson.encode({ { name = "alias" } }))
    ngx.req._headers["X-Internal-Token"] = "client-spoofed"
    ngx.req._headers["X-Body-Uninspectable"] = "1"
    capture_handler = function(loc)
        if loc == "/_vs_auth_blocked" then
            check(ngx.req._headers["X-Internal-Token"] == nil
                and ngx.req._headers["X-Body-Uninspectable"] == nil,
                "client auth metadata cleared before backend authorization")
            return { status = 403 }
        end
        upstream = upstream + 1
    end
    local response = cjson.decode(M._handle_tools_list(1, mapping, "", nil, "blocked"))
    check(response.error ~= nil and upstream == 0,
        "cached tools are denied when backing grant is revoked")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: synthetic initialize cannot poison PAT backend session or mixed discovery")
do
    ngx.var.auth_user = "alice"
    local consent = false
    local initializes, puts, pat_puts, lists, calls = 0, 0, 0, 0, 0
    local mapping = { tools = {
        { name = "personal", original_name = "pat_tool",
          backend_location = "/_vs_backend_pat", egress_auth_mode = "pat" },
        { name = "healthy", original_name = "public_tool",
          backend_location = "/_vs_backend_healthy" },
    } }
    capture_handler = function(loc, opts)
        if loc:find("/_internal/sessions/backend/", 1, true) == 1 then
            if opts.method == ngx.HTTP_GET then return { status = 404 } end
            if opts.method == ngx.HTTP_PUT then
                puts = puts + 1
                if loc:find(":/_vs_backend_pat", 1, true) then
                    pat_puts = pat_puts + 1
                    check(cjson.decode(opts.body).backend_session_id == "real-session",
                        "only a genuine PAT stateful initialize is persisted")
                else
                    check(cjson.decode(opts.body).stateless == true,
                        "healthy backend persists its genuine stateless initialize")
                end
                return { status = 200 }
            end
        end
        if loc == "/_vs_auth_pat" or loc == "/_vs_auth_healthy" then
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc == "/_vs_backend_pat" then
            local method = cjson.decode(opts.body).method
            if method == "initialize" then
                initializes = initializes + 1
                if not consent then
                    return { status = 200,
                        header = { ["X-MCP-Backend-Initialized"] = "0",
                            ["Mcp-Session-Id"] = "synthetic-not-backend" },
                        body = '{"jsonrpc":"2.0","result":{"capabilities":{}}}' }
                end
                return { status = 200, header = { ["Mcp-Session-Id"] = "real-session" },
                    body = '{"jsonrpc":"2.0","result":{"capabilities":{}}}' }
            end
            if method == "tools/list" then
                lists = lists + 1
                check(ngx.req._headers["Mcp-Session-Id"] == (consent and "real-session" or ""),
                    "PAT list uses only a real backend session")
                if not consent then
                    -- The broker's pre-consent answer: an empty, marked list.
                    return { status = 200, header = { ["x-egress-consent-required"] = "pat" },
                        body = '{"jsonrpc":"2.0","result":{"tools":[]}}' }
                end
                return { status = 200,
                    body = '{"jsonrpc":"2.0","result":{"tools":[{"name":"pat_tool"}]}}' }
            end
            calls = calls + 1
            check(ngx.req._headers["X-Internal-Token"] == "signed"
                and ngx.req._headers["Mcp-Session-Id"] == (consent and "real-session" or ""),
                "PAT call carries fresh backing authorization and never synthetic session")
            return { status = 200, body = consent
                and '{"jsonrpc":"2.0","result":{"content":[{"type":"text","text":"live"}]}}'
                or '{"jsonrpc":"2.0","result":{"isError":true,"content":[{"type":"text","text":"Add a PAT in Connected Accounts"}]}}' }
        end
        if loc == "/_vs_backend_healthy" then
            if cjson.decode(opts.body).method == "initialize" then
                return { status = 200, body = '{"jsonrpc":"2.0","result":{}}' }
            end
            return { status = 200, body = '{"jsonrpc":"2.0","result":{"tools":[{"name":"public_tool"}]}}' }
        end
    end
    for i = 1, 2 do
        local response = cjson.decode(M._handle_tools_list(i, mapping, "", "vs-consent", "consent"))
        local listed = {}
        for _, tool in ipairs(response.result and response.result.tools or {}) do
            listed[tool.name] = true
        end
        check(listed.healthy and listed.personal and #response.result.tools == 2,
            "pre-consent PAT publishes its mapped tool beside healthy siblings")
    end
    check(initializes == 2 and lists == 2 and puts == 1 and pat_puts == 0
        and dict:get("bsess_stateless:vs-consent:/_vs_backend_pat:5:alice") == nil,
        "pre-consent PAT initializes again and never persists stateless state")
    check(dict:get("bsess:vs-consent:/_vs_backend_pat:5:alice") == nil,
        "synthetic initialize never caches a backend session ID")
    check(dict:get("tools_enriched::consent:5:alice") == nil,
        "mixed discovery with pre-consent empty tools is not cached")
    local pat_only = { tools = { mapping.tools[1] } }
    local pending = cjson.decode(M._handle_tools_list(3, pat_only, "", "vs-consent", "pat_only"))
    check(pending.result and #pending.result.tools == 1
        and pending.result.tools[1].name == "personal",
        "PAT-only pre-consent tools/list advertises the tool that elicits the PAT")
    check(dict:get("tools_enriched::pat_only:5:alice") == nil,
        "empty pre-consent PAT discovery is never cached")
    ngx.status = 200
    M._handle_tools_call(3, { tools = mapping.tools, tool_backend_map = { personal = {
        backend_location = "/_vs_backend_pat", original_name = "pat_tool" } } },
        { name = "personal" }, "", "vs-consent", "consent")
    local pat_message = cjson.decode(ngx._last_body)
    check(ngx.status == 200 and calls == 1 and pat_message.result.isError == true
        and pat_message.result.content[1].text:find("Add a PAT", 1, true),
        "PAT tool call before consent returns actionable mcp-proxy message")
    consent = true
    local response = cjson.decode(M._handle_tools_list(4, mapping, "", "vs-consent", "consent"))
    local names = {}
    for _, tool in ipairs(response.result.tools) do names[tool.name] = true end
    check(names.personal and names.healthy and puts == 2 and pat_puts == 1,
        "consent permits stateful initialize and both tools appear immediately")
    M._proxy_to_backend(5, "tools/call", { name = "pat_tool" },
        "/_vs_backend_pat", "vs-consent", "consent")
    check(calls == 2 and initializes == 5 and lists == 4
        and cjson.decode(ngx._last_body).result.content[1].text == "live",
        "call after consent reuses genuine backend session and response")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: a plain backend's own errors and 403 fall back to mapping metadata")
do
    ngx.var.auth_user = "alice"
    local mapping = { tools = { { name = "mapped", original_name = "actual",
        backend_location = "/_vs_backend_rpc_error", inputSchema = { type = "object" } } } }
    capture_responses["/_vs_backend_rpc_error"] = { status = 200,
        body = '{"jsonrpc":"2.0","error":{"code":-32000,"message":"offline"}}' }
    local response = cjson.decode(M._handle_tools_list(1, mapping, "", nil, "rpc_error"))
    check(response.result and response.result.tools[1].name == "mapped",
        "plain backend JSON-RPC failure retains mapping metadata")
    check(dict:get("tools_enriched::rpc_error:5:alice") == nil,
        "JSON-RPC error fallback is never cached")
    capture_responses["/_vs_backend_rpc_error"] = { status = 403, body = "" }
    response = cjson.decode(M._handle_tools_list(2, mapping, "", nil, "rpc_error"))
    check(response.result and response.result.tools[1].name == "mapped"
        and dict:get("tools_enriched::rpc_error:5:alice") == nil,
        "a plain backend's own 403 is a backend failure: mapping fallback, never cached")
end

-- ---------------------------------------------------------------------------
print("test: OAuth pre-consent returns connect URL; plain marker never reaches backend")
do
    ngx.var.auth_user = "alice"
    local backend_calls, auths, writes = 0, 0, 0
    local mapping = { tools = { { name = "alias", original_name = "upstream",
        backend_location = "/_vs_backend_oauth", egress_auth_mode = "oauth_user" } },
        tool_backend_map = { alias = { backend_location = "/_vs_backend_oauth",
            original_name = "upstream" } } }
    capture_handler = function(loc, opts)
        if loc == "/_vs_auth_oauth" or loc == "/_vs_auth_plain_marker" then
            auths = auths + 1
            local rpc = cjson.decode(ngx.req._headers["X-Body"])
            check(rpc.method == "initialize" or rpc.method == "tools/call",
                "both initialize and original tool are checked by backend grant")
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc:find("/_internal/sessions/backend/", 1, true) == 1 then
            if opts.method == ngx.HTTP_GET then return { status = 404 } end
            if opts.method == ngx.HTTP_PUT then writes = writes + 1 end
            return { status = 200 }
        end
        if loc == "/_vs_backend_oauth" or loc == "/_vs_backend_plain_marker" then
            if cjson.decode(opts.body).method == "initialize" then
                return { status = 200, header = { ["X-MCP-Backend-Initialized"] = "0" },
                    body = '{"jsonrpc":"2.0","result":{"capabilities":{}}}' }
            end
            backend_calls = backend_calls + 1
            return { status = 200, body = '{"jsonrpc":"2.0","error":{"code":-32042,'
                .. '"data":{"elicitations":[{"mode":"url",'
                .. '"url":"https://gateway.example/connect/oauth"}]}}}' }
        end
    end
    ngx.status = 200
    M._handle_tools_call(17, mapping, { name = "alias" }, "", "vs-oauth", "oauth")
    local response = cjson.decode(ngx._last_body)
    check(ngx.status == 200 and response.error.code == -32042
        and response.error.data.elicitations[1].url == "https://gateway.example/connect/oauth"
        and backend_calls == 1 and auths == 2 and writes == 0,
        "OAuth tool returns proxy connect URL without caching synthetic session")
    M._proxy_to_backend(18, "tools/call", { name = "upstream" },
        "/_vs_backend_plain_marker", "vs-plain-marker", "plain")
    check(ngx.status == 502 and backend_calls == 1 and writes == 0,
        "unexpected pre-consent marker on plain backend fails closed")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: resource and prompt discovery separates backend outage from grant denial")
do
    local mapping = { tools = { { name = "mapped", backend_location = "/_vs_backend_discovery" } } }
    local session_id = "vs-cafe"
    local session_key = "csess_valid:alice:/virtual/discovery:" .. session_id
    local backend_key = "bsess:" .. session_id .. ":/_vs_backend_discovery:5:alice"
    dict:set("mapping::discovery", cjson.encode(mapping))
    dict:set(session_key, "1")
    ngx.var.virtual_server_id = "discovery"
    ngx.var.http_mcp_session_id = session_id
    ngx.var.request_method = "POST"
    ngx.var.auth_user = "alice"
    for _, entry in ipairs({
        { list = "resources/list", get = "resources/read", params = { uri = "file://doc" } },
        { list = "prompts/list", get = "prompts/get", params = { name = "draft" } },
    }) do
        for _, method in ipairs({ entry.list, entry.get }) do
            ngx.req._body = cjson.encode({ jsonrpc = "2.0", id = 1, method = method,
                params = entry.params })
            -- The L2 backend session store is unavailable, but no grant was denied.
            dict:delete(backend_key)
            capture_handler = function(loc)
                if loc == "/_vs_auth_discovery" then
                    return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
                end
                if loc:find("/_internal/sessions/backend/", 1, true) == 1 then
                    return { status = 503 }
                end
            end
            M.route()
            check(ngx.status == 502 and cjson.decode(ngx._last_body).error.message
                == "Backend discovery failed", method .. " returns backend error for session outage")

            -- Once L2 reports a miss, initialize can fail independently.
            capture_handler = function(loc, opts)
                if loc == "/_vs_auth_discovery" then
                    return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
                end
                if loc:find("/_internal/sessions/backend/", 1, true) == 1
                    and opts.method == ngx.HTTP_GET then return { status = 404 } end
                if loc == "/_vs_backend_discovery" then return { status = 503 } end
            end
            M.route()
            check(ngx.status == 502, method .. " reports backend initialize outage")
            capture_handler = function(loc, opts)
                if loc == "/_vs_auth_discovery" then
                    return { status = cjson.decode(ngx.req._headers["X-Body"]).method
                        == "initialize" and 403 or 200,
                        header = { ["X-Internal-Token"] = "signed" } }
                end
                if loc:find("/_internal/sessions/backend/", 1, true) == 1
                    and opts.method == ngx.HTTP_GET then return { status = 404 } end
            end
            M.route()
            check(ngx.status == 403, method .. " preserves denied backend initialize grant")

            capture_handler = function(loc)
                if loc == "/_vs_auth_discovery" then return { status = 403 } end
            end
            M.route()
            check(ngx.status == 403 and cjson.decode(ngx._last_body).error.message
                == "Backend access denied", method .. " keeps genuine grant denial")

            -- A healthy session can still reach a broken or denying backend.
            dict:set(backend_key, "active-session")
            capture_handler = function(loc)
                if loc == "/_vs_auth_discovery" then
                    return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
                end
                if loc == "/_vs_backend_discovery" then return { status = 503 } end
            end
            M.route()
            check(ngx.status == 502, method .. " treats backend HTTP 503 as outage")
            capture_handler = function(loc)
                if loc == "/_vs_auth_discovery" then
                    return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
                end
                if loc == "/_vs_backend_discovery" then return { status = 403 } end
            end
            M.route()
            check(ngx.status == 502,
                method .. " treats the backend's own HTTP 403 as an outage, not a grant denial")

            -- A rejected stateful session retries initialize; classify its result.
            for _, rejected in ipairs({
                { status = 503, expected = 502 },
                { status = 403, expected = 502 },
            }) do
                dict:set(backend_key, "stale-session")
                local init_attempts, list_attempts = 0, 0
                capture_handler = function(loc, opts)
                    if loc == "/_vs_auth_discovery" then
                        return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
                    end
                    if loc:find("/_internal/sessions/backend/", 1, true) == 1 then
                        if opts.method == ngx.HTTP_GET then return { status = 404 } end
                        return { status = 200 }
                    end
                    if loc == "/_vs_backend_discovery" then
                        if cjson.decode(opts.body).method == "initialize" then
                            init_attempts = init_attempts + 1
                            return { status = rejected.status }
                        end
                        list_attempts = list_attempts + 1
                        return { status = 404 }
                    end
                end
                M.route()
                check(ngx.status == rejected.expected and init_attempts == 1
                    and list_attempts == 1,
                    method .. " classifies stale-session reinitialize " .. rejected.status)
            end
        end
    end
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: mixed resource/prompt discovery keeps live siblings before and after PAT")
do
    local connected, deny = false, false
    local calls = {}
    local mapping = { tools = {
        { name = "pat", backend_location = "/_vs_backend_mix_pat", egress_auth_mode = "pat" },
        { name = "unsupported", backend_location = "/_vs_backend_mix_unsupported" },
        { name = "unavailable", backend_location = "/_vs_backend_mix_unavailable" },
        { name = "healthy", backend_location = "/_vs_backend_mix_healthy" },
    } }
    local session_id = "vs-feed"
    dict:set("mapping::mixed_resources", cjson.encode(mapping))
    dict:set("csess_valid:alice:/virtual/mixed_resources:" .. session_id, "1")
    ngx.var.virtual_server_id = "mixed_resources"
    ngx.var.http_mcp_session_id = session_id
    ngx.var.request_method = "POST"
    ngx.var.auth_user = "alice"
    capture_handler = function(loc, opts)
        if loc:find("/_vs_auth_mix_", 1, true) == 1 then
            if deny and loc == "/_vs_auth_mix_unsupported" then return { status = 403 } end
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc:find("/_internal/sessions/backend/", 1, true) == 1 then
            if opts.method == ngx.HTTP_GET then return { status = 404 } end
            if opts.method == ngx.HTTP_PUT then
                check(not loc:find("mix_pat", 1, true) or connected,
                    "pre-consent PAT does not persist a synthetic session")
            end
            return { status = 200 }
        end
        local method = cjson.decode(opts.body).method
        calls[loc .. ":" .. method] = (calls[loc .. ":" .. method] or 0) + 1
        if method == "initialize" then
            if loc == "/_vs_backend_mix_pat" and not connected then
                return { status = 200, header = { ["X-MCP-Backend-Initialized"] = "0" },
                    body = '{"jsonrpc":"2.0","result":{"capabilities":{}}}' }
            end
            if loc == "/_vs_backend_mix_unavailable" then return { status = 503 } end
            return { status = 200, body = '{"jsonrpc":"2.0","result":{"capabilities":{}}}' }
        end
        if loc == "/_vs_backend_mix_pat" and not connected then
            -- The broker answers a pre-consent list itself: empty, marked.
            local key = method == "resources/list" and "resources" or "prompts"
            return { status = 200, header = { ["x-egress-consent-required"] = "pat" },
                body = cjson.encode({ result = { [key] = {} } }) }
        end
        if loc == "/_vs_backend_mix_unsupported" then
            return { status = 200,
                body = '{"jsonrpc":"2.0","error":{"code":-32601,"message":"Method not found"}}' }
        end
        if loc == "/_vs_backend_mix_healthy" or loc == "/_vs_backend_mix_pat" then
            local owner = loc == "/_vs_backend_mix_pat" and "pat" or "healthy"
            if method == "resources/list" then
                return { status = 200, body = cjson.encode({ result = { resources = {
                    { uri = "file://" .. owner } } } }) }
            end
            if method == "prompts/list" then
                return { status = 200, body = cjson.encode({ result = { prompts = {
                    { name = owner } } } }) }
            end
            return { status = 200, body = cjson.encode({ result = { owner = owner } }) }
        end
    end
    for _, connected_now in ipairs({ false, true }) do
        connected = connected_now
        for _, entry in ipairs({
            { list = "resources/list", get = "resources/read", key = "resources",
                params = { uri = "file://healthy" } },
            { list = "prompts/list", get = "prompts/get", key = "prompts",
                params = { name = "healthy" } },
        }) do
            ngx.status = 200
            ngx.req._body = cjson.encode({ jsonrpc = "2.0", id = 7, method = entry.list })
            M.route()
            local response = cjson.decode(ngx._last_body)
            local items = response.result and response.result[entry.key] or {}
            check(ngx.status == 200 and #items == (connected and 2 or 1),
                entry.list .. " discovers only connected, healthy backends")
            ngx.status = 200
            ngx.req._body = cjson.encode({ jsonrpc = "2.0", id = 8,
                method = entry.get, params = entry.params })
            M.route()
            response = cjson.decode(ngx._last_body)
            check(ngx.status == 200 and response.result and response.result.owner == "healthy",
                entry.get .. " resolves healthy backend despite failed siblings")
            if connected then
                ngx.status = 200
                ngx.req._body = cjson.encode({ jsonrpc = "2.0", id = 9,
                    method = entry.get, params = entry.key == "resources"
                        and { uri = "file://pat" } or { name = "pat" } })
                M.route()
                response = cjson.decode(ngx._last_body)
                check(ngx.status == 200 and response.result and response.result.owner == "pat",
                    entry.get .. " reaches connected PAT backend")
            end
        end
    end
    check((calls["/_vs_backend_mix_unavailable:resources/list"] or 0) == 0
        and dict:get("bsess_stateless:vs-feed:/_vs_backend_mix_pat:5:alice") ~= nil,
        "unavailable backend is never listed; connected PAT gains real session")
    -- The caller loses the grant for one backing server: that backend is
    -- omitted, every sibling the caller can still use stays listed and readable.
    deny = true
    for _, method in ipairs({ "resources/list", "resources/read", "prompts/list", "prompts/get" }) do
        ngx.status = 200
        ngx.req._body = cjson.encode({ jsonrpc = "2.0", id = 10,
            method = method, params = { uri = "file://healthy", name = "healthy" } })
        M.route()
        local response = cjson.decode(ngx._last_body)
        local listed = response.result and (response.result.resources or response.result.prompts)
        check(ngx.status == 200 and (listed and #listed == 2
            or response.result and response.result.owner == "healthy"),
            method .. " omits the denied backend and keeps healthy siblings")
    end
    -- Without any grant at all, the aggregate is a denial.
    capture_handler = function(loc)
        if loc:find("/_vs_auth_mix_", 1, true) == 1 then return { status = 403 } end
    end
    for _, method in ipairs({ "resources/list", "prompts/list" }) do
        ngx.status = 200
        ngx.req._body = cjson.encode({ jsonrpc = "2.0", id = 11, method = method })
        M.route()
        check(ngx.status == 403 and cjson.decode(ngx._last_body).error.message
            == "Backend access denied", method .. " denies when no backing grant remains")
    end
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: backends that answered with nothing yield an empty list, not 502")
do
    ngx.var.auth_user = "alice"
    local mapping = { tools = {
        { name = "pat", backend_location = "/_vs_backend_none_pat", egress_auth_mode = "pat" },
        { name = "unsupported", backend_location = "/_vs_backend_none_unsupported" },
    } }
    capture_handler = function(loc, opts)
        if loc:find("/_vs_auth_none_", 1, true) == 1 then
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc == "/_vs_backend_none_pat" then
            return { status = 200, header = { ["x-egress-consent-required"] = "pat" },
                body = cjson.encode({ result = { resources = {} } }) }
        end
        return { status = 200,
            body = '{"jsonrpc":"2.0","error":{"code":-32601,"message":"Method not found"}}' }
    end
    local items, lookup, denied = M._proxy_list_to_backends("resources/list", "resources",
        mapping, nil, "none")
    check(items ~= nil and #items == 0 and lookup ~= nil and not denied,
        "pre-consent plus unsupported backends aggregate to an empty list")
    capture_handler = function(loc)
        if loc:find("/_vs_auth_none_", 1, true) == 1 then
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        return { status = 503 }
    end
    items, lookup, denied = M._proxy_list_to_backends("resources/list", "resources",
        mapping, nil, "none")
    check(items == nil and not denied, "every backend failing is still a discovery failure")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: one credentialed backend outage keeps healthy sibling tools")
do
    local outage_status = 503
    ngx.var.auth_user = "alice"
    local mapping = { tools = {
        { name = "brokered", original_name = "brokered", egress_auth_mode = "oauth_user",
          backend_location = "/_vs_backend_outage_egress", inputSchema = { type = "object" } },
        { name = "healthy", original_name = "healthy",
          backend_location = "/_vs_backend_outage_plain", inputSchema = { type = "object" } },
    } }
    capture_handler = function(loc)
        if loc:find("/_vs_auth_outage_", 1, true) == 1 then
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc == "/_vs_backend_outage_egress" then return { status = outage_status } end
        return { status = 200, body = cjson.encode({ result = { tools = { { name = "healthy" } } } }) }
    end
    local response
    for _, status in ipairs({ 401, 403, 503 }) do
        outage_status = status
        response = cjson.decode(M._handle_tools_list(1, mapping, "", nil, "outage"))
        check(response.result and #response.result.tools == 1
            and response.result.tools[1].name == "healthy",
            "credentialed backend " .. status .. ": healthy sibling stays listed, its tool hidden")
    end
    check(dict:get("tools_enriched::outage:5:alice") == nil, "partial result is not cached")
    capture_handler = function(loc)
        if loc:find("/_vs_auth_outage_", 1, true) == 1 then
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc == "/_vs_backend_outage_egress" then return { status = 503 } end
    end
    local only_egress = { tools = { mapping.tools[1] } }
    response = cjson.decode(M._handle_tools_list(2, only_egress, "", nil, "outage_only"))
    check(response.error ~= nil, "a lone failed credentialed backend is still an error")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: mapping cache is partitioned by rendered config generation")
do
    dict:set("mapping:gen-old:gen_vs", cjson.encode({ tools = { { name = "stale" } } }))
    ngx.var.vs_config_generation = "gen-new"
    local mapping = M._get_mapping("gen_vs")
    check(mapping == nil, "a mapping cached under the previous config is not served")
    ngx.var.vs_config_generation = "gen-old"
    mapping = M._get_mapping("gen_vs")
    check(mapping and mapping.tools[1].name == "stale", "same generation still hits the cache")
    ngx.var.vs_config_generation = nil
end

-- ---------------------------------------------------------------------------
print("test: an auth-check failure is a backend failure, not a grant denial")
do
    ngx.var.auth_user = "alice"
    for _, auth_response in ipairs({
        { status = 429 }, { status = 504 }, { status = 200, header = {} },
    }) do
        capture_handler = function(loc)
            if loc == "/_vs_auth_loc" then return auth_response end
        end
        local outcome = M._fetch_backend_list("/_vs_backend_loc", "tools/list", "tools",
            nil, "srv")
        check(outcome == "unverified", "auth check " .. auth_response.status
            .. (auth_response.header and " without token" or "") .. " -> unverified")
    end
    local mapping = { tools = {
        { name = "flaky", original_name = "flaky", backend_location = "/_vs_backend_flaky" },
        { name = "denied", original_name = "denied", backend_location = "/_vs_backend_nogrant" },
        { name = "healthy", original_name = "healthy", backend_location = "/_vs_backend_fine" },
    } }
    capture_handler = function(loc)
        if loc == "/_vs_auth_flaky" then return { status = 503 } end
        if loc == "/_vs_auth_nogrant" then return { status = 403 } end
        if loc == "/_vs_auth_fine" then
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        return { status = 200, body = cjson.encode({ result = { tools = { { name = "healthy" } } } }) }
    end
    local response = cjson.decode(M._handle_tools_list(1, mapping, "", nil, "mixed_auth"))
    local names = {}
    for _, tool in ipairs(response.result and response.result.tools or {}) do
        names[tool.name] = true
    end
    check(names.healthy and not names.denied and not names.flaky,
        "denied and unverified backends omitted (no mapping fallback), healthy sibling listed")
    check(dict:get("tools_enriched::mixed_auth:5:alice") == nil, "partial result not cached")
    ngx.status = 200
    M._proxy_to_backend(2, "tools/call", { name = "flaky" }, "/_vs_backend_flaky", nil, "srv")
    check(ngx.status == 502, "tools/call with a failing auth check is a retryable 502")
    M._proxy_to_backend(3, "tools/call", { name = "denied" }, "/_vs_backend_nogrant", nil, "srv")
    check(ngx.status == 403, "tools/call without the backing grant is still 403")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: a cached stateless marker re-initializes when the backend rejects it")
do
    ngx.var.auth_user = "alice"
    dict:set("bsess_stateless:vs-flip:/_vs_backend_flip:5:alice", "1")
    local inits, lists = 0, 0
    capture_handler = function(loc, opts)
        if loc == "/_vs_auth_flip" then
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        if loc:find("/_internal/sessions/backend/", 1, true) == 1 then
            if opts.method == ngx.HTTP_GET then return { status = 404 } end
            return { status = 200 }
        end
        if cjson.decode(opts.body).method == "initialize" then
            inits = inits + 1
            return { status = 200, header = { ["Mcp-Session-Id"] = "now-stateful" },
                body = '{"jsonrpc":"2.0","result":{}}' }
        end
        lists = lists + 1
        if ngx.req._headers["Mcp-Session-Id"] ~= "now-stateful" then
            return { status = 400, body = "" }
        end
        return { status = 200, body = cjson.encode({ result = { tools = { { name = "t" } } } }) }
    end
    local outcome, items = M._fetch_backend_list("/_vs_backend_flip", "tools/list", "tools",
        "vs-flip", "srv")
    check(outcome == "ok" and #items == 1 and inits == 1 and lists == 2,
        "stateless marker rejected with 400 -> one re-initialize, then success")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
print("test: resource reads reuse the caller's complete discovery, never another user's")
do
    local lists, reads = 0, 0
    local mapping = { tools = {
        { name = "a", backend_location = "/_vs_backend_docs" },
    } }
    capture_handler = function(loc, opts)
        if loc == "/_vs_auth_docs" then
            return { status = 200, header = { ["X-Internal-Token"] = "signed" } }
        end
        local method = cjson.decode(opts.body).method
        if method == "resources/list" then
            lists = lists + 1
            return { status = 200, body = cjson.encode({ result = { resources = {
                { uri = "file://" .. ngx.var.auth_user } } } }) }
        end
        reads = reads + 1
        return { status = 200, body = cjson.encode({ result = { contents = {} } }) }
    end
    ngx.var.auth_user = "alice"
    for i = 1, 2 do
        ngx.status = 200
        M._handle_resources_read(i, { uri = "file://alice" }, mapping, nil, "docs")
    end
    check(lists == 1 and reads == 2, "second read routes from the cached lookup")
    ngx.var.auth_user = "bob"
    ngx.status = 200
    M._handle_resources_read(3, { uri = "file://alice" }, mapping, nil, "docs")
    check(lists == 2 and reads == 2 and cjson.decode(ngx._last_body).error.code == -32601,
        "another user rediscovers and cannot reach the first user's item")
    -- The backend leaves the virtual server: a cached lookup must not route to it.
    ngx.var.auth_user = "alice"
    ngx.status = 200
    M._handle_resources_read(4, { uri = "file://alice" }, { tools = {} }, nil, "docs")
    check(reads == 2, "cached lookup naming an unmapped backend is never used")
    capture_handler = nil
end

-- ---------------------------------------------------------------------------
if failures > 0 then
    print(string.format("\n%d check(s) FAILED", failures))
    os.exit(1)
end
print("\nAll checks passed")
