-- Execute in the registry nginx Lua runtime, which ships OpenResty's cjson.
local cjson = require("cjson")
assert(cjson.new().decode_array_with_array_mt and cjson.array_mt,
    "the registry's OpenResty cjson array metatable is required")
local configured = cjson.new()
configured.decode_array_with_array_mt(true)
local probe = configured.decode('{"emptyArray":[],"emptyObject":{}}')
assert(getmetatable(probe.emptyArray) == configured.array_mt)
assert(getmetatable(probe.emptyObject) ~= configured.array_mt)
local encoded_probe = configured.encode(probe)
assert(encoded_probe:find('"emptyArray":[]', 1, true))
assert(encoded_probe:find('"emptyObject":{}', 1, true))

local file = "docker/lua/agent_card_rewrite.lua"
local gateway = "https://gateway.example:8443/agent/travel/"
local backend = "https://Backend.Example:443/a2a"

local function rewrite(body, registered, upstream_status)
    local output, status, upstream
    upstream = { status = upstream_status or 200, body = body }
    _G.ngx = {
        HTTP_OK = 200, HTTP_BAD_GATEWAY = 502,
        var = {
            agent_backend_url = registered or backend,
            agent_card_upstream = "/agent/travel/_agent_card_upstream",
            uri = "/agent/travel/.well-known/agent-card.json",
            scheme = "http",
            http_host = "gateway.example:8443",
            http_x_cloudfront_forwarded_proto = "https",
            http_x_forwarded_proto = "http",
        },
        header = {},
        location = { capture = function(path)
            assert(path == "/agent/travel/_agent_card_upstream")
            return upstream
        end },
        say = function(text) output = text end,
        print = function(text) output = text end,
    }
    dofile(file)
    status = ngx.status or 200
    return status, output
end

local function check(status, output, expected)
    assert(status == expected, "unexpected HTTP status: " .. tostring(status))
    assert(output, "missing response body")
    return cjson.decode(output)
end

local raw = [[{"protocolVersion":"1.0","name":"travel","url":"https://Backend.Example:443/a2a","supportedInterfaces":[{"url":"https://Backend.Example:443/a2a/","protocolBinding":"JSONRPC","protocolVersion":"1.0"},{"url":"https://backend.example/a2a","protocolBinding":"HTTP+JSON","protocolVersion":"1.1"},{"url":"https://backend.example/a2a","protocolBinding":"GRPC","protocolVersion":"1.0"},{"url":"https://backend.example/other","protocolBinding":"JSONRPC","protocolVersion":"1.0"},{"url":"wss://backend.example/a2a","protocolBinding":"WEBSOCKET","protocolVersion":"1.0"}],"skills":[{"tags":[],"examples":[{"input":{},"output":[]}]}],"other":{"empty":{},"array":[]}}]]
local code, result = rewrite(raw)
local card = check(code, result, 200)
assert(card.url == gateway)
assert(#card.supportedInterfaces == 2)
assert(card.supportedInterfaces[1].protocolBinding == "JSONRPC")
assert(card.supportedInterfaces[1].protocolVersion == "1.0")
assert(card.supportedInterfaces[2].protocolBinding == "HTTP+JSON")
assert(card.supportedInterfaces[2].protocolVersion == "1.1")
for _, iface in ipairs(card.supportedInterfaces) do assert(iface.url == gateway) end
assert(not result:find("backend.example", 1, true))
assert(result:find('"tags":[]', 1, true))
assert(result:find('"input":{}', 1, true))
assert(result:find('"output":[]', 1, true))
assert(result:find('"array":[]', 1, true))
assert(result:find('"empty":{}', 1, true))

code, result = rewrite([[{"protocolVersion":"1.0","preferredTransport":"HTTP+JSON","supportedInterfaces":[{"url":"https://backend.example/a2a","protocolBinding":"JSONRPC","protocolVersion":"1.0"},{"url":"https://backend.example/a2a","protocolBinding":"HTTP+JSON","protocolVersion":"1.0"}]}]])
card = check(code, result, 200)
assert(card.preferredTransport == "HTTP+JSON", "keep a routable preferred binding")

code, result = rewrite([[{"protocolVersion":"1.0","preferredTransport":"GRPC","supportedInterfaces":[{"url":"https://backend.example/a2a","protocolBinding":"JSONRPC","protocolVersion":"1.0"},{"url":"https://backend.example/grpc","protocolBinding":"GRPC","protocolVersion":"1.0"}]}]])
card = check(code, result, 200)
assert(card.preferredTransport == "JSONRPC", "replace an unservable preferred binding")

code, result = rewrite([[{"protocolVersion":"1.0","supportedInterfaces":[{"url":"https://backend.example/a2a","protocolBinding":"JSONRPC","protocolVersion":"1.0","tenant":"team-1","note":"https://backend.example/secret"}]}]])
card = check(code, result, 200)
assert(card.supportedInterfaces[1].tenant == "team-1")
assert(card.supportedInterfaces[1].note == nil)
assert(not result:find("https://backend.example/secret", 1, true))

-- Registry-internal backend pointers are not part of a remote A2A card.
-- Ordinary metadata URLs retain their meaning and must not become RPC URLs.
code, result = rewrite([[{"protocolVersion":"1.0","url":"https://backend.example/a2a","proxyPassUrl":"https://backend.example/a2a","proxy_pass_url":"https://backend.example/a2a","documentationUrl":"https://backend.example/a2a","iconUrl":"https://backend.example/icon.png","supportedInterfaces":[{"url":"https://backend.example/a2a","protocolBinding":"JSONRPC","protocolVersion":"1.0"}]}]])
card = check(code, result, 200)
assert(card.url == gateway and card.supportedInterfaces[1].url == gateway)
assert(card.proxyPassUrl == nil and card.proxy_pass_url == nil)
assert(card.documentationUrl == "https://backend.example/a2a")
assert(card.iconUrl == "https://backend.example/icon.png")

-- Decode escaped slash and Unicode sequences before comparing addresses.
code, result = rewrite([[{"protocolVersion":"1.0","supportedInterfaces":[{"url":"https:\/\/backend.example\/a2\u0061","protocolBinding":"JSONRPC","protocolVersion":"1.0"}],"skills":[]}]])
card = check(code, result, 200)
assert(card.supportedInterfaces[1].url == gateway)
assert(result:find('"skills":[]', 1, true))

for _, bad in ipairs({
    [[{"protocolVersion":"1.0","url":"https://backend.example/a2a","supportedInterfaces":[{"url":"https://backend.example/a2a","protocolBinding":"GRPC","protocolVersion":"1.0"}]}]],
    [[{"protocolVersion":"1.0","url":"https://backend.example/a2a","supportedInterfaces":[]}]],
    [[{"protocolVersion":"2.0","url":"https://backend.example/a2a","supportedInterfaces":[{"url":"https://backend.example/other","protocolBinding":"JSONRPC","protocolVersion":"2.0"}]}]],
    [[{"protocolVersion":"1.0","url":"https://backend.example/a2a"}]],
    [[{"protocolVersion":"1.0","supportedInterfaces":[{"url":"https://backend.example/a2a/other","protocolBinding":"JSONRPC","protocolVersion":"1.0"}]}]],
    [[{"protocolVersion":"1.0","supportedInterfaces":[{"url":"https://backend.example/a2a?mode=rpc","protocolBinding":"JSONRPC","protocolVersion":"1.0"}]}]],
    "not JSON",
}) do
    code, result = rewrite(bad)
    check(code, result, 502)
    assert(not result:find("backend.example", 1, true))
end
code, result = rewrite(raw, "wss://backend.example/a2a")
check(code, result, 502)
code, result = rewrite(raw, backend, 404)
check(code, result, 502)
assert(not result:find("backend.example", 1, true))

code, result = rewrite([[{"protocolVersion":"1.0.0","supportedInterfaces":[{"url":"https://backend.example/a2a","protocolBinding":"JSONRPC","protocolVersion":"1.0"}]}]])
card = check(code, result, 200)
assert(card.supportedInterfaces[1].url == gateway)


-- Legacy top-level preferred binding must be routable; an additional HTTP
-- binding at this exact endpoint can replace an unusable preferred binding.
code, result = rewrite([[{"protocolVersion":"0.3.0","url":"grpc://backend.example/a2a","preferredTransport":"GRPC","additionalInterfaces":[{"url":"https://backend.example/a2a","transport":"JSONRPC"},{"url":"https://backend.example/grpc","transport":"GRPC"}],"capabilities":{"extensions":[]}}]])
card = check(code, result, 200)
assert(card.url == gateway and card.preferredTransport == "JSONRPC")
assert(#card.additionalInterfaces == 1 and card.additionalInterfaces[1].url == gateway)
assert(not result:find("backend.example", 1, true))
assert(result:find('"extensions":[]', 1, true))

code, result = rewrite([[{"protocolVersion":"0.3.0","url":"https://backend.example/a2a","preferredTransport":"JSONRPC","additionalInterfaces":[{"url":"https://backend.example/a2a","transport":"JSONRPC","note":"https://backend.example/private"}]}]])
card = check(code, result, 200)
assert(card.additionalInterfaces[1].note == nil)
assert(not result:find("https://backend.example/private", 1, true))

-- A 0.x card may include the 1.x field when a shared card model emits its
-- empty default. Its working top-level JSONRPC URL still defines the route.
code, result = rewrite([[{"protocolVersion":"0.3.0","url":"https://backend.example/a2a","preferredTransport":"JSONRPC","supportedInterfaces":[],"proxyPassUrl":"https://backend.example/a2a"}]])
card = check(code, result, 200)
assert(card.url == gateway and card.supportedInterfaces == nil and card.proxyPassUrl == nil)
assert(not result:find("backend.example", 1, true))

-- A mixed 0.x card must not publish unrelated 1.x bindings either.
code, result = rewrite([[{"protocolVersion":"0.3.0","url":"https://backend.example/a2a","supportedInterfaces":[{"url":"https://backend.example/other","protocolBinding":"JSONRPC","protocolVersion":"1.0"}]}]])
card = check(code, result, 200)
assert(card.url == gateway and card.supportedInterfaces == nil)
assert(not result:find("backend.example", 1, true))

code, result = rewrite([[{"protocolVersion":"0.3.0","url":"https://backend.example/a2a","preferredTransport":"JSONRPC","additionalInterfaces":[{"url":"https://backend.example/grpc","transport":"GRPC"}]}]])
card = check(code, result, 200)
assert(card.url == gateway and #card.additionalInterfaces == 0)
assert(result:find('"additionalInterfaces":[]', 1, true))
code, result = rewrite([[{"protocolVersion":"0.3.0","url":"https://backend.example/a2a/","preferredTransport":"HTTP+JSON","skills":[{"tags":[]}]}]])
card = check(code, result, 200)
assert(card.url == gateway and card.preferredTransport == "HTTP+JSON")
assert(result:find('"tags":[]', 1, true))
code, result = rewrite([[{"protocolVersion":"0.2.9","url":"https://backend.example/a2a","skills":[]}]])
card = check(code, result, 200)
assert(card.url == gateway and card.preferredTransport == nil)
assert(result:find('"skills":[]', 1, true))



code, result = rewrite([[{"protocolVersion":"0.3.0","url":"https://backend.example/a2a","preferredTransport":"GRPC"}]])
check(code, result, 502)
print("gateway card rewrite: OK")
