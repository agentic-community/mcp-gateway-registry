-- Regression tests for the virtual MCP server router's tools/list aggregation
-- (PR #1634 / originally #1526). Covers the fixed behaviors:
--   * a failed backend falls back to its mapping-file metadata (complete list),
--   * a genuine empty tools array is a success, not a failure,
--   * truncated / malformed backend responses are failures,
--   * partial (fallback) results are NOT cached; fully-discovered results ARE.
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

_G.ngx = {
    shared = { virtual_server_map = dict },
    location = {
        capture = function(loc, _opts) return capture_responses[loc] end,
    },
    req = { set_header = function() end },
    log = function() end,
    ERR = 4,
    WARN = 5,
    HTTP_POST = 8,
    var = { request_id = "0" },
    status = 200,
    say = function() end,
    exit = function() end,
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
print("test: _fetch_backend_tools_list classifies responses correctly")
do
    -- success with tools
    capture_responses["/loc"] = { status = 200,
        body = cjson.encode({ result = { tools = { { name = "x" } } } }) }
    local tools, ok = M._fetch_backend_tools_list("/loc", nil, "srv")
    check(ok == true and #tools == 1, "200 + tools -> (tools, true)")

    -- genuine empty list is success
    capture_responses["/loc"] = { status = 200,
        body = cjson.encode({ result = { tools = {} } }) }
    tools, ok = M._fetch_backend_tools_list("/loc", nil, "srv")
    check(ok == true and #tools == 0, "200 + empty tools -> ([], true)")

    -- http error is failure
    capture_responses["/loc"] = { status = 500, body = "" }
    tools, ok = M._fetch_backend_tools_list("/loc", nil, "srv")
    check(ok == false, "500 -> (_, false)")

    -- truncated is failure
    capture_responses["/loc"] = { status = 200, truncated = true,
        body = cjson.encode({ result = { tools = { { name = "x" } } } }) }
    tools, ok = M._fetch_backend_tools_list("/loc", nil, "srv")
    check(ok == false, "truncated -> (_, false)")

    -- missing tools array is failure
    capture_responses["/loc"] = { status = 200, body = cjson.encode({ result = {} }) }
    tools, ok = M._fetch_backend_tools_list("/loc", nil, "srv")
    check(ok == false, "missing tools array -> (_, false)")
end

-- ---------------------------------------------------------------------------
print("test: _handle_tools_list falls back per-backend and does NOT cache partial")
do
    dict._store["tools_enriched:srv1"] = nil
    local mapping = { required_scopes = nil, tools = {
        { name = "live_tool", original_name = "live_tool", backend_location = "/ok",
          inputSchema = { type = "object" } },
        { name = "down_tool", original_name = "down_tool", backend_location = "/down",
          inputSchema = { type = "object" } },
    } }
    capture_responses = {}
    capture_responses["/ok"] = { status = 200,
        body = cjson.encode({ result = { tools = { { name = "live_tool", description = "live" } } } }) }
    capture_responses["/down"] = { status = 500, body = "" }

    local resp = M._handle_tools_list("1", mapping, "", nil, "srv1")
    local decoded = cjson.decode(resp)
    local names = {}
    for _, t in ipairs(decoded.result.tools) do names[t.name] = true end
    check(names["live_tool"] == true, "live backend tool present")
    check(names["down_tool"] == true, "failed backend tool present via fallback")
    check(dict._store["tools_enriched:srv1"] == nil, "partial/fallback result is NOT cached")
end

-- ---------------------------------------------------------------------------
print("test: _handle_tools_list caches when every backend succeeds")
do
    dict._store["tools_enriched:srv2"] = nil
    local mapping = { required_scopes = nil, tools = {
        { name = "live_tool", original_name = "live_tool", backend_location = "/ok",
          inputSchema = { type = "object" } },
        { name = "down_tool", original_name = "down_tool", backend_location = "/down",
          inputSchema = { type = "object" } },
    } }
    capture_responses = {}
    capture_responses["/ok"] = { status = 200,
        body = cjson.encode({ result = { tools = { { name = "live_tool" } } } }) }
    capture_responses["/down"] = { status = 200,
        body = cjson.encode({ result = { tools = { { name = "down_tool" } } } }) }

    M._handle_tools_list("1", mapping, "", nil, "srv2")
    check(dict._store["tools_enriched:srv2"] ~= nil, "fully-discovered result IS cached")
end

-- ---------------------------------------------------------------------------
if failures > 0 then
    print(string.format("\n%d check(s) FAILED", failures))
    os.exit(1)
end
print("\nAll checks passed")
