# Cookie Security Design

> **See also:** [session-flow-cookie-based.md](session-flow-cookie-based.md)
> for the end-to-end browser login flow under the post-#1042 server-side
> session store. This document focuses on cross-subdomain cookie domain
> mechanics; the linked doc covers what's *in* the cookie and how the
> session record is laid out server-side.

## Overview

This document explains the design decisions behind the session cookie security implementation in the MCP Gateway Registry, particularly regarding the use of domain cookies for cross-subdomain authentication.

## Background

The MCP Gateway Registry supports authentication through both traditional username/password and OAuth2 providers. In deployments where the auth server and registry are on different subdomains (e.g., `auth.example.com` and `registry.example.com`), session cookies must be shared across these subdomains for seamless authentication.

## Design Decision: Single-Tenant Architecture

This implementation is designed for **single-tenant deployments** where:
- All subdomains are owned and controlled by a single organization
- Users authenticate once and access multiple services on different subdomains

Sharing the session cookie across subdomains is a feature these deployments may want, but it has a cost. Every host under the shared domain receives the cookie, and any of them can set a cookie with its name, so one compromised or user-content host can sign users out or, with a matching Domain and Path, put a browser in a session of its choosing. See [Duplicate session cookies](#duplicate-session-cookies).

## Cookie Security Configuration

### Environment Variables

Three key environment variables control cookie security behavior:

1. **`SESSION_COOKIE_SECURE`** (default: `false`)
   - Set to `true` in production deployments with HTTPS
   - When `true`, cookies are only transmitted over HTTPS connections
   - Prevents man-in-the-middle (MITM) attacks and session hijacking
   - **Production Requirement:** MUST be set to `true` when deployed with HTTPS

2. **`SESSION_COOKIE_DOMAIN`** (default: `None` or empty string)
   - **MUST be explicitly configured** - no automatic domain inference
   - When set (e.g., `.example.com`), enables cross-subdomain cookie sharing
   - Must start with a dot (`.`) to match all subdomains
   - When `None` or empty, cookies are scoped to the exact host that sets them
   - **Format:** `.example.com` (note the leading dot)
   - **Important:** Set to empty string `""` for single-domain deployments
   - **Examples:**
     - Single domain (`mcpgateway.ddns.net`): Leave unset or set to `""`
     - Cross-subdomain (`auth.example.com`, `registry.example.com`): Set to `.example.com`
     - Multi-level domains (`registry.region-1.corp.company.internal`): Set to `.corp.company.internal` if cross-subdomain sharing needed

3. **`SESSION_COOKIE_NAME`** (default: `mcp_gateway_session`)
   - The cookie name both services use: the auth server sets it at login, and the registry and `/validate` read it. Both services refuse to start when it is empty, is not an RFC 6265 token, or starts with `__Host-` or `__Secure-` in any case.

### HTTPS Termination Detection

**Critical Implementation Detail**: The auth server intelligently handles HTTPS termination at load balancers (ALB, nginx, etc.):

- **Backend sees HTTP** but **load balancer terminates HTTPS** → Common in AWS ALB, nginx reverse proxy
- **Solution**: Auth server checks `X-Forwarded-Proto` header to detect original protocol
- **Behavior**: Cookie `secure` flag is set based on **original request protocol**, not backend protocol

**Code Logic** (`auth_server/server.py:1797-1803`):
```python
x_forwarded_proto = request.headers.get("x-forwarded-proto", "")
is_https = x_forwarded_proto == "https" or request.url.scheme == "https"

# Only set secure=True if the original request was HTTPS
cookie_secure_config = OAUTH2_CONFIG.get("session", {}).get("secure", False)
cookie_secure = cookie_secure_config and is_https
```

**Important**:
- If `SESSION_COOKIE_SECURE=true` but `is_https=False`, the secure flag will NOT be set
- This prevents login failures when HTTPS termination is misconfigured
- Check server logs for `is_https=True` in production to verify HTTPS detection is working

### Cookie Security Flags

The implementation sets the following security flags on all session cookies:

| Flag | Value | Purpose |
|------|-------|---------|
| `httponly` | `True` | Prevents JavaScript access, mitigating XSS attacks |
| `samesite` | `"lax"` | Provides CSRF protection while allowing cross-site navigation |
| `secure` | Configurable | Ensures HTTPS-only transmission in production |
| `path` | `"/"` | Explicitly scopes cookie to entire domain |
| `domain` | Configurable | Enables cross-subdomain sharing when needed |

<a id="duplicate-session-cookies"></a>
### Duplicate session cookies

When the `Cookie` header carries the session cookie name more than once, the auth server's `/validate` and every registry read of the session cookie treat the request as having no session cookie. They refuse two copies with equal values too, and log one warning with the count (never a value). `/validate` then tries the request's other credentials, such as a bearer token, exactly as it would with no cookie. Both services read the header with one function, `single_session_cookie` in `registry/common/session_cookie.py`, which matches the exact name, so a cookie such as `old_mcp_gateway_session` is not the session.

Neither copy is safe to pick. A browser orders cookies of one name by longer `Path` first and then by older creation time, so whoever sets the second cookie can make theirs come first or last. The two services could also pick different copies, because Starlette's `request.cookies` keeps the last one and a parser that stops at the first match keeps the first.

Both services read every `Cookie` header field, not only the first. HTTP/2 lets a client send each cookie in its own field, and a proxy that does not join them passes the fields on separately, so a reader of the first field alone would miss a session cookie, or a duplicate of it, in a later field. Both join the fields in arrival order with `join_cookie_fields` from the same module before parsing. Starlette's `request.cookies` also reads every field, but keeps the last value for a name.

In the registry a read that bypasses the shared reader fails closed. `SessionCookieIsolationMiddleware` in `registry/auth/session_cookie.py` is the outermost of the registry's own middleware, so it runs before the audit middleware, every other registry middleware and every route. It reads the session cookie once through the shared reader, caches the result in the request scope, and removes every session cookie pair from the `Cookie` header fields it passes on, for HTTP requests and WebSocket handshakes alike. Every other cookie, the egress consent cookies included, keeps its bytes. `request_session_cookie` answers from the cache, while `request.cookies.get(settings.session_cookie_name)` or a FastAPI `Cookie(alias=...)` parameter gets None. OpenTelemetry's ASGI middleware, which `FastAPIInstrumentor.instrument_app` or the `opentelemetry-instrument` wrapper adds, wraps the whole app and runs first, so it sees the raw `Cookie` header. It records request headers only when `OTEL_INSTRUMENTATION_HTTP_CAPTURE_HEADERS_SERVER_REQUEST` lists them. Never list `Cookie` there, in either service: every span would carry the session cookie. No compose file, Helm chart, Terraform module or `.env.example` in this repository sets that variable.

A browser that sends a duplicated session cookie is signed out. `SessionCookieIsolationMiddleware` adds the expiry of the host-only cookie and the one on `SESSION_COOKIE_DOMAIN`, both at `Path=/`, to every response the registry gives such a request (a 401 from a session check, a 403 from CSRF, the dashboard's redirect to `/login`), unless the response already sets or expires the session cookie, as logout does on every sign-out.

CSRF never skips a request that carries the session cookie. `verify_csrf_token_flexible` and `verify_csrf_token_header_only` in `registry/auth/csrf.py` skip the check only when the request has no session cookie at all, as a bearer-only client does. A request with the cookie, sent once or more, must resolve to a session and present that session's token, or it gets a 403. Skipping CSRF because the registry could not resolve a cookie would rest on `/validate` agreeing that the cookie is no session, and a transient session store error in the registry breaks that agreement.

Logout with a duplicated session cookie also ends the server-side sessions behind it. It checks each copy's signature with the session signer and deletes the session of every copy that verifies and names a live session; an unsigned, forged or expired copy is ignored. Neither copy counts as the caller's session, so this logout starts no sign-out at the identity provider. The browser's logout link is a `GET`, which needs no authentication or CSRF token, so the registry reads the session store once per distinct copy and for at most 8 distinct copies, in the order they arrived. When a request carries more, the registry logs how many it ignored. The log records how many copies came and how many sessions were deleted, never a value. A logout `POST` with a duplicated cookie fails CSRF, gets the expiries, and ends no session.

Changing `SESSION_COOKIE_DOMAIN` can leave a second cookie in a browser. A move from host-only to a domain fixes itself: the registry expires both copies on its next response, and the user signs in again. A move from one domain to another, or from a domain back to host-only, leaves the old domain's cookie in place, because the gateway no longer knows that domain and cannot expire it. Under nginx, `/validate` refuses most such requests before they reach the registry, so the browser's cookies change at the next sign-in or sign-out.

A cookie another host tosses on a parent domain the gateway does not name, beside the gateway's host-only cookie, is worse. The gateway cannot expire a cookie on a domain it does not name, and every new login sets the gateway's cookie beside it again, so the user is signed out on every request until they clear the site's cookies in the browser or the tossed cookie expires.

| Symptom | Cause and fix |
|---------|---------------|
| Users are signed out again and again after `SESSION_COOKIE_DOMAIN` changed; logs show `Refused the session cookie: the Cookie header carries it 2 times` | The browser still holds the session cookie for the old domain beside the new one, and the gateway cannot expire a domain it no longer names. Each user deletes the site's cookies for the old domain in the browser, then signs in again. If the warning appears without a domain change, another host under the parent domain is setting the cookie. |

## Security Considerations

### ✅ Safe Deployment Scenarios

This design is **SAFE** for:

1. **Single-Tenant Production Deployments**
   - Example: `auth.company.com` and `registry.company.com`
   - All subdomains owned by the same organization
   - Configuration:
     ```bash
     SESSION_COOKIE_SECURE=true
     SESSION_COOKIE_DOMAIN=.company.com
     ```

2. **Local Development (localhost)**
   - Local development on `localhost` via HTTP
   - Configuration:
     ```bash
     SESSION_COOKIE_SECURE=false  # MUST be false for HTTP
     SESSION_COOKIE_DOMAIN=       # Leave unset/empty
     ```
   - **Important:** Setting `SESSION_COOKIE_SECURE=true` on localhost will cause login to fail because cookies with `secure=true` are only sent over HTTPS, and localhost typically runs over HTTP.

### ⚠️ Unsafe Deployment Scenarios

This design is **NOT SAFE** for:

1. **Multi-Tenant SaaS Deployments**
   - Example: `customer1.saas-platform.com` and `customer2.saas-platform.com`
   - **Risk:** Setting `SESSION_COOKIE_DOMAIN=.saas-platform.com` would allow:
     - Customer A to access Customer B's sessions
     - Cross-tenant authentication bypass
     - Serious data breach potential

2. **Shared Hosting Environments**
   - Multiple organizations sharing the same root domain
   - **Risk:** Similar to multi-tenant scenario

### Alternative Solutions for Multi-Tenant

If you need multi-tenant deployment, consider these alternatives:

1. **Token-Based Authentication**
   - Use JWT tokens passed via headers instead of cookies
   - Tokens explicitly scoped to each tenant
   - No domain-sharing concerns

2. **Separate Auth Domains per Tenant**
   - `customer1-auth.platform.com` and `customer1-app.platform.com`
   - Different root domains prevent cookie sharing between tenants

3. **Reverse Proxy with Path-Based Routing**
   - Single domain with path-based service routing
   - Example: `platform.com/auth` and `platform.com/registry`
   - No cross-subdomain cookie requirements

4. **Centralized OAuth Flow**
   - OAuth server on separate domain
   - Token exchange instead of session cookies
   - Better tenant isolation

## Attack Scenarios Mitigated

### 1. Session Hijacking (MITM)
- **Threat:** Attacker intercepts session cookies over unencrypted HTTP
- **Mitigation:** `secure=True` flag in production
- **Status:** ✅ Mitigated when `SESSION_COOKIE_SECURE=true`

### 2. Cross-Site Scripting (XSS)
- **Threat:** Malicious JavaScript reads session cookies
- **Mitigation:** `httponly=True` flag
- **Status:** ✅ Always mitigated

### 3. Cross-Site Request Forgery (CSRF)
- **Threat:** Malicious site triggers authenticated requests
- **Mitigation:** `samesite="lax"` flag
- **Status:** ✅ Always mitigated

### 4. Subdomain Cookie Theft (Single-Tenant)
- **Threat:** Attacker controls a subdomain and steals cookies
- **Mitigation:** Only valid in trusted single-tenant environments
- **Status:** ⚠️ Acceptable risk for single-tenant deployments

## Production Deployment Checklist

Before deploying to production:

- [ ] Set `SESSION_COOKIE_SECURE=true` in environment (REQUIRED for HTTPS)
- [ ] Verify HTTPS is properly configured and enforced
- [ ] **IMPORTANT**: If using load balancer with HTTPS termination, ensure `X-Forwarded-Proto` header is set
- [ ] Set `SESSION_COOKIE_DOMAIN` appropriately:
  - **Empty string or unset** for single-domain deployments (RECOMMENDED - safest)
  - **`.example.com`** only if you need cross-subdomain authentication
- [ ] Confirm you are deploying in a single-tenant architecture (NOT multi-tenant SaaS)
- [ ] Test cross-subdomain authentication between auth and registry services (if using domain cookies)
- [ ] Verify cookies are NOT transmitted over HTTP in production
- [ ] Review server logs for cookie configuration at startup:
  - Check for `Auth server setting session cookie: secure=True`
  - Verify `domain` setting matches your configuration
  - Confirm `is_https=True` in production

### Example Production Configurations

**Single-Domain Deployment (RECOMMENDED - Most Secure):**
```bash
# .env for production - single domain (e.g., mcpgateway.example.com)
SESSION_COOKIE_SECURE=true  # REQUIRED for HTTPS
SESSION_COOKIE_DOMAIN=      # Empty = exact host only (safest)
SESSION_COOKIE_NAME=mcp_gateway_session
SESSION_MAX_AGE_SECONDS=28800  # 8 hours
AUTH_SERVER_URL=http://auth-server:8888  # Internal URL
AUTH_SERVER_EXTERNAL_URL=https://mcpgateway.example.com  # External URL
```

**Cross-Subdomain Deployment:**
```bash
# .env for production - cross-subdomain (e.g., auth.example.com + registry.example.com)
SESSION_COOKIE_SECURE=true  # REQUIRED for HTTPS
SESSION_COOKIE_DOMAIN=.example.com  # Note the leading dot
SESSION_COOKIE_NAME=mcp_gateway_session
SESSION_MAX_AGE_SECONDS=28800  # 8 hours
AUTH_SERVER_URL=http://auth-server:8888  # Internal URL
AUTH_SERVER_EXTERNAL_URL=https://auth.example.com  # External URL
```

## Code Implementation

The cookie security implementation is found in:

- **Configuration:** `registry/core/config.py`
  - `session_cookie_secure`: Controls HTTPS-only flag
  - `session_cookie_domain`: Controls cross-subdomain sharing

- **Auth Server Cookie Setting:** `auth_server/server.py` (lines 1800-1831)
  - X-Forwarded-Proto detection for HTTPS termination at load balancer
  - Explicit configuration only - no automatic domain inference
  - Conditional secure flag based on both config AND actual protocol
  - All security flags properly set

- **Registry Cookie Setting:** `registry/auth/routes.py` (lines 139-158)
  - Comprehensive security comments explaining single-tenant model
  - Conditional domain attribute application
  - All security flags properly set

- **Shared reader:** `registry/common/session_cookie.py`
  - `single_session_cookie` and `join_cookie_fields`: the one reader of the `Cookie` header

- **Registry read and expiry:** `registry/auth/session_cookie.py`
  - `request_session_cookie`, `clear_session_cookie` and `SessionCookieIsolationMiddleware`

## Monitoring and Validation

### Runtime Validation

The auth server logs detailed cookie configuration for debugging:

```python
logger.info(f"Auth server setting session cookie: secure={cookie_secure} (config={cookie_secure_config}, is_https={is_https}), samesite={cookie_samesite}, domain={cookie_domain or 'not set'}, x-forwarded-proto={x_forwarded_proto}, request_scheme={request.url.scheme}")
```

Key logging details:
- **secure**: Final secure flag value (after protocol detection)
- **config**: Configured SESSION_COOKIE_SECURE value
- **is_https**: Whether the original request was HTTPS (based on X-Forwarded-Proto or request scheme)
- **domain**: Configured domain or "not set"
- **x-forwarded-proto**: Load balancer protocol header
- **request_scheme**: Direct request protocol

The registry logs successful login events:

```python
logger.info(f"User '{username}' logged in successfully.")
```

### Security Auditing

Periodically review:
1. Cookie flags are properly set in browser developer tools
2. Cookies are NOT transmitted over HTTP in production
3. `secure` flag is enabled in production environments
4. Domain scope matches your deployment architecture

### Browser Developer Tools Verification

In your browser's developer tools (Application/Storage → Cookies), verify:

| Property | Expected Value | Notes |
|----------|---------------|-------|
| `Secure` | ✓ (checked) | Production only |
| `HttpOnly` | ✓ (checked) | Always |
| `SameSite` | `Lax` | Always |
| `Domain` | `.example.com` | If configured |
| `Path` | `/` | Always |

## References

- [OWASP Session Management Cheat Sheet](https://cheatsheetsecurity.org/cheatsheets/session-management-cheat-sheet)
- [MDN: Set-Cookie HTTP Header](https://developer.mozilla.org/en-US/docs/Web/HTTP/Headers/Set-Cookie)
- [RFC 6265: HTTP State Management Mechanism](https://datatracker.ietf.org/doc/html/rfc6265)

## Contact

For questions or security concerns regarding this implementation, please:
- Open an issue in the GitHub repository
- Tag the issue with `security` label
- Provide details about your deployment scenario
