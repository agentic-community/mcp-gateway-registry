-- emit_metrics.lua: Capture MCP request metrics in log_by_lua phase (no network I/O)
local ok, cjson = pcall(require, "cjson")
if not ok then return end

local metrics = ngx.shared.metrics_buffer
if not metrics then return end

-- Attacker-influenced values (tool_name from JSON-RPC params.name, client_name
-- from the X-Client-Name header) become Prometheus labels downstream. Bound
-- them at ingest -- replace anything outside a safe charset and truncate to a
-- fixed length -- so randomized values cannot inflate the label alphabet. This
-- is defense-in-depth: the metrics-service processor also enforces a distinct-
-- value (cardinality) cap, which is the reliable per-process enforcement point.
-- The shared dict itself caps at 1024 keys, bounding the buffer, not labels.
local MAX_LABEL_LENGTH = 64

local function normalize_label(value)
    -- Coerce non-strings (a caller may send params.name as a number, table,
    -- etc.); a table has no safe string form, so bucket it as empty.
    if type(value) == "table" then
        return ""
    end
    if value == nil or value == "" then
        return ""
    end
    -- Replace any char outside [A-Za-z0-9-_.:/] with underscore. Slash is kept
    -- because legitimate JSON-RPC methods and tool names use it (tools/call,
    -- namespace/tool) and it is safe inside a Prometheus label value.
    local cleaned = tostring(value):gsub("[^%w%-_%.:/]", "_")
    if #cleaned > MAX_LABEL_LENGTH then
        cleaned = cleaned:sub(1, MAX_LABEL_LENGTH)
    end
    return cleaned
end

-- Skip buffering when no collector is configured (avoids pointless writes that TTL-expire)
local metrics_url = os.getenv("METRICS_SERVICE_URL") or ""
if metrics_url == "" then return end

-- Server/agent name for attribution. A location block may set
-- $metrics_server_name explicitly (e.g. A2A agent blocks, whose URI first
-- segment is the shared "agent" prefix and would otherwise bucket every agent
-- together); otherwise derive it from the first URI path segment: /<server>/...
local server_name = ngx.var.metrics_server_name
if server_name == nil or server_name == "" then
    server_name = ngx.var.uri:match("^/([^/]+)/")
end
if not server_name or server_name == "" then return end

-- Parse JSON-RPC body from X-Body header (set by capture_body.lua in rewrite phase)
local method = "unknown"
local tool_name = ""
local body = ngx.req.get_headers()["X-Body"]
if body then
    local dok, parsed = pcall(cjson.decode, body)
    if dok and parsed.method then
        -- parsed.method is copied verbatim from the JSON-RPC body, so a client
        -- can send an arbitrary method string (recorded even when the upstream
        -- rejects the call). Normalize it like the other attacker-supplied
        -- labels; the processor also caps its distinct-value count. Compare on
        -- the raw value so the tools/call branch is unaffected by normalization.
        local raw_method = parsed.method
        method = normalize_label(raw_method)
        if raw_method == "tools/call" and parsed.params and parsed.params.name then
            tool_name = normalize_label(parsed.params.name)
        end
    end
end

local entry = cjson.encode({
    m = method,
    s = server_name,
    t = tool_name,
    c = normalize_label(ngx.req.get_headers()["X-Client-Name"] or "unknown"),
    ok = ngx.status < 400,
    d = (tonumber(ngx.var.upstream_header_time) or tonumber(ngx.var.request_time) or 0) * 1000,
})

local key = "m:" .. ngx.now() .. ":" .. ngx.worker.pid() .. ":" .. math.random(1, 999999)
local set_ok, set_err = metrics:set(key, entry, 300)
if not set_ok then
    ngx.log(ngx.ERR, "metrics emit: shared dict full, dropping metric: ", set_err)
end
