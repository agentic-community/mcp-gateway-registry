-- flush_metrics.lua: Background timer flushes shared dict buffer to collector endpoint
local ok, cjson = pcall(require, "cjson")
if not ok then return end

local api_key = os.getenv("METRICS_API_KEY") or ""
local metrics_url = os.getenv("METRICS_SERVICE_URL") or ""

if metrics_url == "" then
    ngx.log(ngx.WARN, "metrics flush: DISABLED (METRICS_SERVICE_URL not set)")
    return
end

-- Only http:// is supported (raw TCP cosocket, no TLS)
if metrics_url:sub(1, 8) == "https://" then
    ngx.log(ngx.ERR, "metrics flush: DISABLED -- METRICS_SERVICE_URL uses https:// which is not supported (use http:// for internal service-to-service)")
    return
end

if api_key == "" then
    ngx.log(ngx.WARN, "metrics flush: METRICS_API_KEY not set, requests may be rejected by metrics-service")
end

local host, port = metrics_url:match("http://([^:/]+):?(%d*)")
port = tonumber(port) or 80

local function flush()
    local buf = ngx.shared.metrics_buffer
    if not buf then return end

    local keys = buf:get_keys(1024)
    if #keys == 0 then return end
    if #keys == 1024 then
        ngx.log(ngx.WARN, "metrics flush: buffer at capacity (1024 keys), some metrics may be lost")
    end

    local batch = {}
    local to_delete = {}
    for _, key in ipairs(keys) do
        if key:sub(1, 2) == "m:" then
            local val = buf:get(key)
            if val then
                local dok, e = pcall(cjson.decode, val)
                if dok then
                    batch[#batch + 1] = {
                        type = "tool_execution",
                        value = 1.0,
                        duration_ms = e.d,
                        dimensions = {
                            method = e.m,
                            server_name = e.s,
                            tool_name = e.t,
                            client_name = e.c,
                            success = tostring(e.ok),
                        },
                        metadata = {},
                    }
                    to_delete[#to_delete + 1] = key
                end
            end
        end
    end

    if #batch == 0 then return end

    local payload = cjson.encode({
        service = "nginx",
        version = "1.0.0",
        metrics = batch,
    })

    local sock = ngx.socket.tcp()
    sock:settimeout(5000)
    local conn_ok, err = sock:connect(host, port)
    if not conn_ok then
        ngx.log(ngx.ERR, "metrics flush: connect failed: ", err)
        return
    end

    local req = "POST /metrics HTTP/1.1\r\n"
        .. "Host: " .. host .. "\r\n"
        .. "Content-Type: application/json\r\n"
        .. "X-API-Key: " .. api_key .. "\r\n"
        .. "Content-Length: " .. #payload .. "\r\n"
        .. "Connection: close\r\n\r\n"
        .. payload

    local send_ok, err = sock:send(req)
    if not send_ok then
        ngx.log(ngx.ERR, "metrics flush: send failed: ", err)
        sock:close()
        return
    end

    local line = sock:receive("*l")
    sock:close()

    if line and line:match("200") then
        for _, key in ipairs(to_delete) do
            buf:delete(key)
        end
        if #batch > 1 then
            ngx.log(ngx.INFO, "metrics flush: sent ", #batch, " metrics")
        end
    else
        ngx.log(ngx.ERR, "metrics flush: bad response: ", line or "nil")
    end
end

local function schedule()
    local ok, err = ngx.timer.every(5, function(premature)
        if premature then return end
        local pok, perr = pcall(flush)
        if not pok then
            ngx.log(ngx.ERR, "metrics flush error: ", perr)
        end
    end)
    if not ok then
        ngx.log(ngx.ERR, "metrics flush: failed to create timer: ", err)
    end
end

if ngx.worker.id() == 0 then
    ngx.log(ngx.WARN, "metrics flush: starting on worker 0, host=", host, " port=", port, " api_key_len=", #api_key)
    schedule()
end
