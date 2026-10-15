-- Runs as content_by_lua after the gateway's auth_request, before response
-- headers are sent. A body filter cannot turn an unroutable card into a 502.
local cjson = require("cjson").new()
-- The registry image ships OpenResty's lua-cjson, whose decoder can tag every
-- array (including nested empty arrays) so re-encoding preserves JSON shape.
cjson.decode_array_with_array_mt(true)

local function reject()
    ngx.status = ngx.HTTP_BAD_GATEWAY
    ngx.header.content_type = "application/json"
    ngx.say('{"error":"Agent card has no gateway-routable interface"}')
end

-- The registered proxy_pass_url is provided by nginx, never by the remote card.
-- Compare the entire backend endpoint, not just its origin: the JSON-RPC proxy
-- routes to this URL and cannot serve another path on the same host.
local function endpoint_parts(url)
    if type(url) ~= "string" then return nil end
    local scheme, authority, path = url:match("^([%a]+)://([^/?#]+)([^?#]*)$")
    if not scheme then return nil end
    local host, port = authority:match("^([%w.%-]+):?(%d*)$")
    if not host then return nil end
    scheme = scheme:lower()
    if scheme ~= "http" and scheme ~= "https" then return nil end
    if port == "" then port = scheme == "https" and "443" or "80" end
    port = tonumber(port)
    if not port or port < 1 or port > 65535 then return nil end
    return scheme, host:lower(), port, (path == "" or path == "/") and "" or path:gsub("/$", "")
end

local backend_scheme, backend_host, backend_port, backend_path =
    endpoint_parts(ngx.var.agent_backend_url)
if not backend_scheme then
    reject()
    return
end

local function matches_backend(url)
    local scheme, host, port, path = endpoint_parts(url)
    return scheme == backend_scheme and host == backend_host
        and port == backend_port and path == backend_path
end

local function routable(binding, url)
    return (binding == "JSONRPC" or binding == "HTTP+JSON")
        and matches_backend(url)
end

-- This is an `internal` location; only the already-authorized outer location
-- can fetch it. On any upstream error, never echo the backend's response body.
local response = ngx.location.capture(ngx.var.agent_card_upstream)
if not response or response.status ~= ngx.HTTP_OK then
    reject()
    return
end
local ok, card = pcall(cjson.decode, response.body or "")
if not ok or type(card) ~= "table" or getmetatable(card) == cjson.array_mt then
    reject()
    return
end

-- Preserve the forwarded viewer scheme behind CloudFront/ALB, and the Host
-- port for local/non-default-port installations.
local function valid_proto(value)
    return (value == "http" or value == "https") and value or nil
end
local scheme = valid_proto(ngx.var.http_x_cloudfront_forwarded_proto)
    or valid_proto(ngx.var.http_x_forwarded_proto) or ngx.var.scheme
local base = ngx.var.uri:gsub("/%.well%-known/agent%-card%.json$", "")
local gateway_url = scheme .. "://" .. ngx.var.http_host .. base .. "/"

-- A2A 1.x declares its bindings in supportedInterfaces; a top-level URL is
-- not an independent 1.x binding. Keep the original entries (including
-- protocolVersion and extensions) but drop bindings that cannot be proxied.
if card.supportedInterfaces ~= nil or
    (type(card.protocolVersion) == "string" and card.protocolVersion:match("^1%.")) then
    if type(card.supportedInterfaces) ~= "table"
        or getmetatable(card.supportedInterfaces) ~= cjson.array_mt then
        reject()
        return
    end
    local interfaces = setmetatable({}, cjson.array_mt)
    for _, entry in ipairs(card.supportedInterfaces) do
        if type(entry) == "table" and type(entry.protocolVersion) == "string"
            and entry.protocolVersion ~= "" and routable(entry.protocolBinding, entry.url) then
            interfaces[#interfaces + 1] = {
                url = gateway_url,
                protocolBinding = entry.protocolBinding,
                protocolVersion = entry.protocolVersion,
                tenant = entry.tenant,
            }
        end
    end
    if #interfaces == 0 then
        reject()
        return
    end
    card.supportedInterfaces = interfaces
    -- These legacy fields are not A2A 1.0 declarations. Do not return stale
    -- backend endpoints from a mixed-version card.
    card.additionalInterfaces = nil
    card.additional_interfaces = nil
    card.supported_interfaces = nil
    if card.url ~= nil then card.url = gateway_url end
    if card.preferredTransport ~= nil then
        local preferred_survives = false
        for _, entry in ipairs(interfaces) do
            if entry.protocolBinding == card.preferredTransport then
                preferred_survives = true
                break
            end
        end
        if not preferred_survives then
            card.preferredTransport = interfaces[1].protocolBinding
        end
    end
else
    -- A2A 0.x defaults the top-level URL to JSONRPC when preferredTransport
    -- is absent. Additional interfaces can supply a usable fallback when the
    -- preferred binding uses GRPC or another backend path.
    local preferred_ok = routable(card.preferredTransport or "JSONRPC", card.url)
    local interfaces = setmetatable({}, cjson.array_mt)
    if type(card.additionalInterfaces) == "table"
        and getmetatable(card.additionalInterfaces) == cjson.array_mt then
        for _, entry in ipairs(card.additionalInterfaces) do
            if type(entry) == "table" and routable(entry.transport, entry.url) then
                interfaces[#interfaces + 1] = {
                    url = gateway_url,
                    transport = entry.transport,
                }
            end
        end
    end
    if not preferred_ok and #interfaces == 0 then
        reject()
        return
    end
    if not preferred_ok then
        card.preferredTransport = interfaces[1].transport
    end
    card.url = gateway_url
    if card.additionalInterfaces ~= nil then card.additionalInterfaces = interfaces end
    -- A mixed-version card must not leak an unfiltered binding list.
    card.supportedInterfaces = nil
    card.supported_interfaces = nil
    card.additional_interfaces = nil
end

local encoded, result = pcall(cjson.encode, card)
if not encoded then
    reject()
    return
end
ngx.header.content_type = "application/json"
ngx.print(result)
