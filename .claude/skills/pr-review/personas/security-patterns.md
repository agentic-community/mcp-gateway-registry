# Security Patterns and Anti-Patterns

This is the project's catalog of security defects that have actually shipped and been fixed, distilled into reusable rules.
It exists so that new code and code reviews do not reintroduce the same classes of vulnerability.

Each pattern is grounded in a defect that actually shipped and was fixed here; the PR numbers under each "How we enforce it" trace back to the fix, cross-checked against the Opus 4.8 source audit and the AWS Security Agent STRIDE threat model.

**How to use this document:**

- **Writing a feature:** read the patterns that touch your surface (new API route? read #2 and #5. Outbound fetch? read #1. New secret/env var? read #3). Apply the rule, not just the specific past fix.
- **Reviewing a PR:** the [Review Checklist](#review-checklist) at the bottom maps each pattern to a yes/no question. The `pr-review` Security Engineer persona and the `new-feature-design` expert review both reference this file.

Each pattern is written as: **the mistake** → **the rule** → **how we enforce it in this codebase** → **what to check**.

---

## 1. Server-side fetch of a user/registry-controlled URL (SSRF)

**The mistake.** Taking a URL that a registrant, peer, or user supplied (a `proxy_pass_url`, federation peer `endpoint`, custom OAuth `token_url`, agent-card URL, skill source) and fetching it with a plain `httpx`/`requests` client. An attacker points it at `169.254.169.254` (cloud metadata), a loopback admin port, or an internal host and either exfiltrates the secret being sent (OAuth `client_secret`, refresh token, federation token) or reaches an internal service. DNS rebinding defeats a naive "resolve then check" because the hostname re-resolves to a private IP between the check and the connect.

**The rule.** Every server-side fetch of a non-first-party URL goes through the shared SSRF guard, fails closed, and is validated on every redirect hop:

- Only `http`/`https` schemes.
- Host must resolve **exclusively** to public IPs (block private, loopback, link-local, reserved, multicast, unspecified, and IPv4-mapped IPv6).
- The cloud metadata address is **never** allowlistable.
- Pin the connection to the validated IP so there is no rebinding window; re-validate redirects.
- Any resolution failure or ambiguity → reject, never fall through to a permissive path.

**How we enforce it.** Use [`registry/utils/url_guard.py`](../../../../registry/utils/url_guard.py) — the single source of truth. Pick the profile: `PROXY_PROFILE` for server/agent targets (operator opt-in via `ssrf_allowed_hosts`/`ssrf_allowed_cidrs`), `SKILL_PROFILE` for skill fetches. For outbound requests use `guarded_async_client(profile)` rather than a bare `httpx.AsyncClient`. Fixed in #1363 (registration), #1396 (egress OAuth token calls), #1398 (federation peer create/update/sync), #1391 (A2A agent-card).

**What to check.** Does any new code build an `httpx`/`aiohttp`/`requests` call from a stored or request-supplied URL without going through `url_guard`? Is a secret being POSTed to that URL (making a guard rejection mandatory *before* the send)? Are redirects followed without re-validation (`follow_redirects=True` on an unguarded client)?

---

## 2. Broken access control and info disclosure on API endpoints

**The mistake.** An external `/api/...` endpoint (or a `.well-known` endpoint) that returns more than the caller is entitled to: backend `proxy_pass_url`/`mcp_endpoint` in a list/search/versions response, all enabled servers to an anonymous caller, or a mutation keyed on the URL path instead of the stored resource identity. The recurring root cause is that a new API endpoint skips the authorization check its older UI-route counterpart enforced.

**The rule.**

- Every read endpoint that can surface backend URLs must gate disclosure behind a fail-closed visibility check; non-admins get the public gateway URL only.
- Every endpoint resolves the resource, returns 404 if absent, and runs the per-user access check (403) **before** returning data or mutating.
- Authorization keys on the **stored** resource identity (`server_info["server_name"]`), never on the attacker-controlled URL path.
- New external API endpoints must mirror the authz of their legacy UI-route counterpart — do not assume network placement protects them.

**How we enforce it.** `should_redact_backend_urls()` in [`registry/services/visibility.py`](../../../../registry/services/visibility.py) (fail-closed for non-admins); `user_can_access_server_path` on the versions/detail endpoints. Fixed in #1388 (versions endpoint), #1397 (read-only info restriction across list/search), #1389 (federation token write-only), #1374 (removed anonymous `/.well-known/mcp-servers`), #1365 (authz/ownership across management routes).

**What to check.** New GET returning server/agent records — does it strip backend URLs for non-admins? New mutation — does it 404-then-403 on the resolved resource? Does the check use the stored name or the path param? Is there a UI route doing this authz that the new API route forgot?

---

## 3. Weak, committed, or default-permissive secrets and config

**The mistake.** Shipping a usable default: `DOCUMENTDB_PASSWORD=admin` in `.env.example`, a hardcoded PingFederate `2FederateM0re`, a Keycloak `changeme`, an unauthenticated MongoDB, dev-mode ports bound to `0.0.0.0`, a `SECRET_KEY` with no length floor, `sslRequired=none`. These make a fresh deploy insecure by default and often get copied into production.

**The rule.**

- Secrets have **no working default**. Required secrets fail closed (Compose `${VAR:?}`, Helm `{{ required }}`, Pydantic validation) rather than falling back to a placeholder.
- Ship weak-value **denylists** and reject known-bad values (`admin`, `dev-root-token`, shipped placeholders) case-insensitively.
- Enforce minimum entropy/length on signing keys (≥32 bytes for `SECRET_KEY`, `AUTH_SERVER_NGINX_MARKER_SECRET`, `METRICS_KEY_PEPPER`).
- Bind non-front-door ports to loopback (`127.0.0.1`) by default; publicly expose only the intended front door. Require an explicit opt-in (`HOST_BIND_IP=0.0.0.0`) for LAN.
- Data-store auth on by default; TLS `external`, never `none`.

**How we enforce it.** The `build_and_run.sh` preflight + `scripts/validate-extra-env.sh` reject weak/reserved values; `_PF_ADMIN_PASS_DENYLIST`; length checks in [`registry/core/config.py`](../../../../registry/core/config.py). Fixed in #1386 (Mongo auth + strong password), #1381 (Keycloak creds), #1367 (SECRET_KEY length), #1376 (loopback bind), #1400/#1368 (denylist + placeholder removal), #1380/#1384 (TLS external).

**What to check.** Any new secret/env var — does it have a working default that would ship insecure? Is it added to the reserved-name lists and the weak-secret denylist? New service port — is it loopback-bound by default? Any `verify=False`, `sslRequired=none`, or `0.0.0.0` bind?

---

## 4. Token trust boundaries

**The mistake.** Treating all tokens as interchangeable: reusing a user's token as an internal service token, trusting a client-supplied session id without binding it to the authenticated user, accepting an id-token without verifying its signature, or relaying the caller's inbound `Authorization`/cookie headers onward to an untrusted upstream (leaking the caller's credential to a third-party MCP/A2A server).

**The rule.**

- Internal service-to-service tokens are minted separately from user tokens and are not accepted where a user token is expected (and vice versa).
- Bind session identity to the authenticated principal; never trust a client-supplied session id as authorization.
- Verify id-token signatures (JWKS: signature, issuer, expiry, audience) before trusting claims.
- Treat inbound client auth headers as **ingress-only**: strip them on egress. Never forward a caller's credential to an upstream the caller does not control.

**How we enforce it.** Separate internal-token path (#1359); session-user binding in `virtual_router.lua` (#1357); id-token signature verification (#1366); egress header stripping (#1369, #1391). See also the LLM Agent Tool-Execution Safety and token-boundary rules in the root `CLAUDE.md`.

**What to check.** Does new code forward `request.headers["Authorization"]` or the session cookie to an outbound call? Is a session id or user id read from the request body/params and used for authz without re-checking the authenticated principal? Is a JWT decoded with `verify_signature=False` or without audience/issuer checks?

---

## 5. Missing CSRF on state-changing endpoints

**The mistake.** A new router with mutating (POST/PUT/PATCH/DELETE) endpoints that does not apply the CSRF dependency. There is no global CSRF middleware, so a router that forgets it leaves high-privilege mutations (IAM CRUD, M2M account creation, federation add/modify/delete) reachable cross-origin using an authenticated operator's session cookie.

**The rule.** Every mutating endpoint reachable with a session cookie applies the CSRF dependency. Non-browser (Bearer-token) clients and read-only GETs are unaffected. Missing/invalid token → 403.

**How we enforce it.** Add `verify_csrf_token_flexible` (from [`registry/auth/csrf.py`](../../../../registry/auth/csrf.py)) as a dependency on every mutating route, matching the pattern already used across the other routers. Fixed in #1390 (IAM management + federation routers were the last two missing it).

**What to check.** Does every new POST/PUT/PATCH/DELETE carry the CSRF dependency? When a new router is added, does it match the CSRF pattern of the existing routers?

---

## 6. Injection through unescaped interpolation

**The mistake.** Building a downstream string (nginx config directive, MongoDB `$regex`, an HTML page, an href in the frontend) by interpolating untrusted input without escaping. Concretely: a `proxy_pass_url` with nginx metacharacters breaking out of a directive; `{"$regex": f"^{path}:"}` letting a crafted path inject regex; a `javascript:`/`data:` URL rendered as a link; server path interpolated into the OAuth callback HTML.

**The rule.**

- Never interpolate untrusted input into a config/query/markup context without a context-appropriate escape.
- nginx: reject metacharacters (`\r`, `\n`, `;`, spaces, quotes, braces) in any value that lands in a directive.
- MongoDB `$regex`: `re.escape()` the interpolated fragment (still open as SA-18a — do not add new ones).
- Frontend: render untrusted hrefs only through the scheme allowlist (`safeUrl.ts`/`SafeLink.tsx`) that blocks `javascript:`/`data:`.
- HTML responses: escape interpolated path/provider values.

**How we enforce it.** `_NGINX_METACHARACTERS` rejection in [`registry/utils/url_guard.py`](../../../../registry/utils/url_guard.py); `safeUrl.ts`/`SafeLink.tsx` (#1394); callback HTML escaping (#1396). Parameterized queries and allowlist-validated identifiers per the SQL guidance in `CLAUDE.md`.

**What to check.** Any f-string or `+` that puts request/registry data into a query, config, shell, or markup string? Any new href/redirect that renders a stored URL without the scheme allowlist? Any `$regex` built from input without `re.escape`?

---

## 7. Secret and PII leakage into logs and responses

**The mistake.** Logging raw request headers, the full user-context dict, decoded OIDC claims, or a token-bearing payload; returning a write-only secret (federation token) in a read/list response; writing a freshly minted credential to stdout or a world-readable file.

**The rule.**

- Route header dumps, body/user-context dicts, and OIDC claims through the redaction layer before logging. Log claim **names** and masked identifiers, never claim values or tokens.
- Secrets are **write-only in the API**: accept on create/update, never echo on read/list (use a response schema that excludes them).
- CLI-minted credentials go to owner-only (`0600`) files, never stdout/logs.

**How we enforce it.** [`registry/common/log_redaction.py`](../../../../registry/common/log_redaction.py) (`redact_headers`, `redact_mapping`, `safe_identity_summary`); write-only federation schema (#1389); `0600` credential files (#1405). Fixed in #1405, #1389, #1397.

**What to check.** Does new logging print headers, a full request/user dict, a token, or OIDC claim values? Does a response model include a secret field that should be write-only? Does a CLI command print a minted secret?

---

## 8. Dependency CVE exposure

**The mistake.** A dependency floor low enough to permit a version with a known CVE (`aiohttp>=3.8.0` permits several fixed only in 3.11.12), or carrying an unused dependency that drags in CVE-bearing transitive deps (`python-jose` → `ecdsa`/`rsa`/`pyasn1`), relying on the lockfile alone to mitigate (off-lock/manifest installs stay exposed).

**The rule.**

- Raise manifest floors above the fixed version, not just the lockfile — off-lock installs must be safe too.
- Remove dependencies that are declared but never imported.
- Apply the floor consistently across the root and every sub-project manifest/lock.

**How we enforce it.** #1387 (dropped unused `python-jose`), #1401 (raised `aiohttp` floor, extended the python-jose guard to all sub-projects, refreshed all `uv.lock`). A guard test asserts `python-jose` stays out of every manifest/lock.

**What to check.** Does a new dependency's floor sit above known CVE fixes? Is a newly added dependency actually imported? Was the change applied to all relevant manifests, not one?

---

## 9. LLM agent and A2A execution safety

**The mistake.** An LLM tool loop that executes any `tool_use` the model emits without a human gate (the model output is untrusted and steerable by prompt injection), and an A2A/agent HTTP endpoint that accepts messages based on network reachability alone (any process that can reach port 9000 drives the agent).

**The rule** (also in root `CLAUDE.md` under "LLM Agent Tool-Execution Safety"):

- Gate every mutating/destructive tool call behind mandatory human confirmation; classify read vs. mutate at execution; fail closed (deny) when no confirmation channel exists.
- Deny-by-default allowlist for any shell/exec tool; reject unknown executables and shell metacharacters; scrub the environment.
- Authenticate the agent endpoint with an inbound bearer JWT (signature/issuer/expiry/audience) on every request before the message reaches the model; bind loopback by default; auth fails closed if JWKS is unconfigured.

**How we enforce it.** `cli/src/agent/toolPolicy.ts` (confirmation gate + allowlist) and per-agent `auth_middleware.py` binding `AGENT_HOST=127.0.0.1` (#1392); egress credential stripping (#1391). System-prompt `<security>` text is **not** an enforcement control — enforce at the execution boundary.

**What to check.** Does a new agent tool execute a mutation without a confirmation gate? Does a shell/exec tool run arbitrary executables? Does a new agent HTTP endpoint validate a JWT before invoking the model, or rely on being "internal"?

---

## 10. Authenticating with a shared or guessable key; timing oracles

**The mistake.** Comparing an API key or signature with `==` (timing oracle), hashing it without a per-deployment secret (rainbow-tableable), or signing telemetry with a key that is a public constant in the source (anyone with the repo forges valid payloads). Also: a rate-limit response that leaks whether a key was valid.

**The rule.**

- Hash authentication keys with a **required per-deployment pepper**; compare in constant time.
- Do not authenticate on a secret that ships in the source tree.
- Rate-limit responses must not distinguish valid from invalid credentials.

**How we enforce it.** Metrics API key: peppered HMAC-SHA256 with constant-time compare, `METRICS_KEY_PEPPER` required (min-32, fails closed), rate-limit oracle closed (#1399). Note: telemetry signing (TM-20) still uses a public constant key — treat as partial, do not model new auth on it.

**What to check.** Any `==` on a secret/token/signature (use `hmac.compare_digest`)? Any key hashed without a per-deployment secret? Any auth based on a constant baked into the repo? Does an error/rate-limit path reveal credential validity?

---

## 11. MCP proxy body integrity

**The mistake.** Authorizing a request based on metadata (path/headers) but forwarding a body that was not the one inspected, or forwarding a body that could not be inspected — letting a caller smuggle an unauthorized JSON-RPC method past the authz check.

**The rule.** Re-authorize the **exact** forwarded body; fail closed if the body cannot be inspected. Strip internal capture headers (`X-Body-*`) before forwarding upstream.

**How we enforce it.** MCP proxy body-integrity re-authorization (#1393); internal `X-Body` capture headers stripped before upstream forward (part of #1391 line).

**What to check.** Does the proxy path authorize the same bytes it forwards? Does it fail closed on an uninspectable/oversized body? Are internal capture headers stripped on egress?

---

## Review Checklist

Fast pass for a PR touching the relevant surface. Any "no" is a blocker until justified.

**Outbound requests**
- [ ] Every fetch of a stored/request-supplied URL goes through `url_guard` with the right profile (#1)
- [ ] Secrets are never POSTed to an unvalidated URL; redirects are re-validated (#1)

**API endpoints**
- [ ] New GET strips backend URLs for non-admins via the visibility check (#2)
- [ ] Mutations 404-then-403 on the resolved resource; authz keys on stored identity, not the path (#2)
- [ ] New external API route mirrors its legacy UI-route authz (#2)
- [ ] Every mutating endpoint carries the CSRF dependency (#5)

**Secrets and config**
- [ ] No new secret ships with a working default; required secrets fail closed (#3)
- [ ] New secret/env var added to reserved-name lists + weak-secret denylist (#3)
- [ ] New ports loopback-bound by default; no `0.0.0.0`, `verify=False`, or `sslRequired=none` (#3)
- [ ] Signing keys enforce a minimum length (#3)

**Tokens and identity**
- [ ] Inbound auth headers/cookies are stripped on egress, never forwarded upstream (#4)
- [ ] JWTs verified (signature/issuer/expiry/audience); no client-supplied session id trusted for authz (#4)

**Injection**
- [ ] No untrusted input interpolated into nginx/query/HTML/href without escaping (#6)
- [ ] `$regex` fragments use `re.escape`; frontend hrefs use the scheme allowlist (#6)

**Logging and responses**
- [ ] No headers/user-context/tokens/OIDC claim values logged; use the redaction layer (#7)
- [ ] Secret fields are write-only in response schemas; CLI secrets go to `0600` files (#7)

**Dependencies**
- [ ] New dependency floors sit above known CVE fixes across all manifests; unused deps removed (#8)

**Agents / proxy / key auth**
- [ ] Mutating agent tools gated behind confirmation; agent HTTP endpoints validate a JWT (#9)
- [ ] Secrets compared with `hmac.compare_digest`; keys peppered per-deployment; no source-constant auth (#10)
- [ ] MCP proxy re-authorizes the exact forwarded body and fails closed on an uninspectable one (#11)

---

*Maintained alongside `CLAUDE.md` (subprocess/SQL/LLM-agent security sections) and the security scratchpad in `.scratchpad/sec-findings/`. When a new class of security fix lands, add the pattern here so reviews catch the next instance.*
