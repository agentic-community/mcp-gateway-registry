-- capture_body.lua: Read request body and encode it in X-Body header for auth_request
local cjson = require "cjson"

-- Strip any client-supplied copies of the headers this script owns so a
-- caller cannot forge the scope-decision inputs the auth server trusts.
ngx.req.clear_header("X-Body")
ngx.req.clear_header("X-Body-Uninspectable")

-- Read the request body
ngx.req.read_body()
local body_data = ngx.req.get_body_data()

if body_data then
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
    -- defense-in-depth at the edge.)
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
