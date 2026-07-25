# Security Guidelines

Security invariants and lessons for this codebase, distilled from remediating
security-review findings. **Read this before touching authentication,
authorization, outbound requests, credential handling, config generation, or
logging.** CLAUDE.md carries a compact always-loaded checklist; this doc is the
detailed, categorized reference behind it.

## How to use / extend this doc

- These are enforceable rules, not suggestions. Code review should reject changes
  that violate them.
- **When you find or fix a NEW category of security issue, add a lesson here**
  (one entry per pattern, not per individual finding): the mistake, why it
  happened, and the rule that prevents recurrence. Keep the always-loaded
  checklist in CLAUDE.md in sync when a new category is added.
- Prefer a single shared, hardened utility per concern (URL validation, secret
  validation, privileged-scope checks) over per-call-site reimplementations —
  the drift between copies is where the gaps live.

## Fail-closed principle

On any error, missing config, or ambiguous state, DENY. Never fall back to an
allow, a permissive default, or a "temporary" bypass. A check that can be
silently skipped (optional param, truthiness on a value that can be empty,
sanitizer that isn't called) is equivalent to no check.

## Secrets & signing keys

- **Validate a shared signing secret for MISSING *and* WEAK, at every entrypoint
  that signs.** Presence checks are insufficient (a long shipped placeholder
  passes a length check). Reject unset/empty/whitespace-only; reject `< 32` chars
  on the *stripped* value; reject a denylist of known-weak literals (e.g. the
  `.env.example` default, `development-secret-key`, `changeme`). Run the
  weak-value check BEFORE the length check so a known placeholder produces the
  right error. Normalize (strip) and reuse the same value everywhere so all
  services derive an identical key (avoids cross-replica signature mismatches).
  Wire the validator into EVERY signing entrypoint — grep for all of them.
- **Every credential that grants privilege must go through the ONE canonical
  signing-secret validator at its startup entrypoint — the legacy/backward-compat
  path is where this check gets skipped.** When a newer keyed scheme enforces a
  minimum length/weak-value bar but an older single-value credential is read raw
  (`os.environ.get("TOKEN", "")`) and then promoted to a privileged (often admin,
  unrestricted-scope) entry, that legacy credential is asymmetrically weaker than
  both the keyed entries and the app signing secret. Run it through the SAME
  `validate_signing_secret` helper (missing AND weak: unset/empty/whitespace,
  `< 32` stripped chars, known-weak literals, weak-check before length) at the
  point it is read/built, and fail closed. Presence may be optional (unset simply
  means the legacy entry is not created — that is the safe outcome), but a value
  that IS present must be strong; never silently accept a weak privilege-granting
  credential just because it predates the current key scheme. This applies to ALL
  sibling credential paths, not only the reported one — a federation/static bypass
  token, each per-key entry in a keyed scheme, and any other value that bypasses
  IdP validation must each run through the SAME validator. A `min_length`-only
  Pydantic constraint or a `logging.warning(...)` on a short value is NOT fail
  closed: a weak privilege-granting token that is merely warned about is still
  armed. Reject it — raise where a raise is safe, or (for an optional feature that
  degrades gracefully) disable the feature so the weak token is never armed —
  never merely warn.
- **Match placeholder markers as substrings ANYWHERE, not just as a prefix.** An
  operator rarely leaves the `.env.example` value verbatim — they prepend a
  prefix or edit the middle (`internal-CHANGE-ME-...`, `prod-generate-with-openssl-...`).
  A `startswith`-only check lets those through. Also test `marker in normalized`
  for a small set of narrow markers (`change-me`, `replace-me`,
  `generate-with-openssl`, …). Keep the markers hyphen/underscore-bearing or
  otherwise outside `[0-9a-f]`/base64 so a genuine `openssl rand -hex 32` secret
  can never collide with one (no false-positive on real high-entropy keys).
- **Never ship a vendor/default credential fallback in code.**
  `os.environ.get("PASS", "<default>")` is a silent foot-gun. Use one chokepoint
  that raises when unset, and denylist the known-weak value so an env var
  explicitly set to it is also rejected (fail closed even if compose passes
  `${PASS:-weak}`). A dev container's own provisioned password is deployment
  config; the application code must never supply it as a default.
- **Example credentials in help text/docs must be unmistakably fake** (`YOUR_*`,
  `EXAMPLE_*`). Do not alter functional lookup keys that merely resemble IDs.
- **Never log secrets, tokens, PII, or full credential/claim payloads.** Redact
  before logging; log identifiers/counts, not values. Watch: request-header
  dumps, `updates`/body dicts that contain tokens, decoded id_token claims. This
  includes setup/debug scripts in `VERBOSE`/`DEBUG` modes — never echo a password
  or bearer token (not even a prefix); those logs land in CI/CloudWatch/shell
  history. More sinks that recur: (a) a **JSON-RPC / request body dump** — a
  `tools/call` carries user-supplied tool arguments that may be credentials/PII;
  log the size + method/id, never the body. (b) **OAuth token-endpoint error
  bodies** (`response.text` / the full error JSON, logged OR raised) — they echo
  the `client_id` and partial credential context in `error_description`; log only
  the standard `error` code + status. (c) **IdP group-name lists** — group names
  are organizational PII (org units, teams, cost centers); log the count, not the
  names. A "masked" token prefix is still a leak — a base64 JWT header/signature
  fragment is reconstructable; log `<token len=N>`, not `token[:10]`. (d) **A
  registrant/backend URL** (`proxy_pass_url`, an MCP/SSE endpoint, an agent
  `base_url`, an inbound `request.url`) can carry credentials in userinfo
  (`user:pass@host`) or a secret in the query (`?access_token=...`); route every
  such URL through `registry/common/log_redaction.py` `redact_url` (keeps
  `scheme://host[:port]/path`, drops userinfo/query/fragment, fails closed) at
  EVERY log site, not just the reported one. (e) **The caller's own authorization
  policy** (`user_scopes`, `scope_config`, `accessible_services`, `ui_permissions`)
  is sensitive and must never sit at INFO — especially on a hot per-request path
  (the `/validate` subrequest, list-servers): keep the verbose policy trace at
  DEBUG and emit exactly ONE INFO allow/deny audit line carrying only
  server/method/tool identifiers, at every return including the fail-closed
  exception branch. Watch for developer traces mislabeled by LEVEL, not just by
  content — a `logger.info(f"DEBUG: ...")` or `[TRACE]` line is still emitted at
  INFO in production; the `# TODO: replace with debug` marker means it was never
  meant to ship at that level. After fixing the reported site, grep the whole
  package for the same variable/pattern — these leaks travel in packs across
  health-check, connection, and registration code paths.
- **Never reflect an exception/stack trace into a response the caller sees**
  (CWE-209, CodeQL `py/stack-trace-exposure`). `HTTPException(detail=str(e))`,
  `return {"error": str(e)}` from a route, and `HTMLResponse(f"...{exc}...")` all
  ship internal detail (issuer/audience/IdP config, decryption/`SECRET_KEY` hints,
  resolved internal hostnames, DB/stack context) to any client — worst on PUBLIC
  endpoints (`.well-known/*`). Log the specifics server-side (`logger.exception`)
  and return a **generic** message/marker. Classify on the raw message if you must
  (status-code mapping), but do not echo it. A user-supplied value already in the
  request (e.g. the URL they asked you to fetch) is fine to reflect; the wrapped
  exception is not.
- **A hardcoded/shipped hashing or HMAC key is equivalent to an unsalted hash.**
  If a stored-credential hash (e.g. an API-key `key_hash`) is HMAC'd with a
  compile-time constant, anyone with the DB or a hash can brute-force it offline,
  and it's identical across deployments. Read the pepper from a REQUIRED
  per-deployment secret, validated fail-closed (missing/weak/short/placeholder).
  A deployment-wide pepper keeps hashes deterministic for a `UNIQUE`-column lookup
  while defeating cross-deployment/offline attack. Changing it invalidates old
  hashes — document key re-issue.
- **Write-only credentials need a separate token-free response schema.** A
  bidirectional secret stored on a config object (e.g. a `federation_token`) must
  never serialize on a GET/list response. Use a dedicated response model (or
  `SecretStr`) that omits it and exposes only `has_<secret>: bool` — not `exclude`
  bolted onto the shared read/write model, which is easy to regress.
- **A short TTL limits but does not eliminate replay of internal tokens.** Mint
  every internal service token with a unique `jti` and enforce single-use via an
  atomic insert into a shared, replica-visible store (unique index + TTL index),
  fail closed on missing-jti / replay / store-error. BEFORE making a token
  single-use, confirm it isn't legitimately verified more than once per flow —
  e.g. a forwarded proxy token checked at two hops; that class must NOT be
  single-use (bind a nonce to the known hops or use mTLS instead).

## SSRF & outbound requests from user/registry-controlled input

- **One hardened URL guard, used by every outbound fetch.** Block RFC-1918,
  loopback, link-local, reserved, multicast, unspecified, and cloud-metadata
  (`169.254.169.254` — never allowlistable); unwrap IPv4-mapped IPv6; require
  http/https. Fail closed. Do not create per-call-site `_is_safe_url` variants.
- **Don't rely on the language runtime's `is_private` to cover reserved ranges —
  block them explicitly and pin the range in a test.** Ranges like CGNAT / shared
  address space (`100.64.0.0/10`, RFC 6598) are only classified private on newer
  runtimes; depending on interpreter semantics means a downgrade or a stdlib
  change silently re-opens an SSRF pivot. Add the range to the guard's explicit
  block list (treated like the other private ranges — operator-CIDR-allowlistable,
  never for metadata) and add a unit test asserting both a sample IP is blocked
  AND the exact network is pinned, so a semantics change fails loudly instead of
  quietly. Apply to every guard implementation, including any legacy one still in
  use.
- **Validate at registration (structural) AND pin the resolved IP at fetch
  time.** A pre-fetch check followed by a separate client call re-resolves DNS =
  TOCTOU / DNS-rebinding. Pin the validated public IP into the transport
  (preserve Host header + TLS SNI) and re-validate on every redirect hop.
- **Re-validate the destination through the SSRF guard at the moment a stored
  credential is attached.** Registration-time validation is not enough when the
  destination is mutable — a `proxy_pass_url`/endpoint changed after registration
  (or a registration-time bypass) could point a stored credential at a
  private/metadata address. Every site that decrypts a stored credential and
  hands it to an outbound request (including an external scanner or SDK client
  that opens its own connection, which the pinned guarded transport cannot
  protect) must re-run `validate_url` with the proxy profile first and fail
  closed: no valid destination, no credential. Grep every `decrypt_credential` /
  credential-attach site, not just the reported one; a single guarded helper
  they all call is ideal. Make the header/credential builder SELF-GUARD on the
  destination it is handed rather than trusting each caller to have validated
  first — a builder that decrypts a secret only because the caller "should have"
  checked leaks that secret the first time a caller forgets. Pass the destination
  into the builder and have it withhold the decrypted secret (return only
  non-secret headers) when the destination is missing or fails validation.
- **Never build or attach credentials for a target that fails validation.**
  Validate the URL before decrypting/attaching stored credentials, so a
  malicious registered URL cannot exfiltrate them.
- **Every fetch path uses the guarded client** — grep for raw `httpx`/SDK clients
  and third-party SDKs that own their own client. Internal targets are opt-in via
  an explicit allowlist (`SSRF_ALLOWED_HOSTS`/`SSRF_ALLOWED_CIDRS`), default deny.
- **A feature merged in parallel with a hardening PR is the classic gap.** When a
  hardening PR routes "every outbound URL" through the guard, its diff only covers
  files that existed on its branch. A feature developed concurrently (its own new
  handlers/clients) is invisible to that PR and ships an unguarded sink. After a
  hardening wave, re-audit any feature that landed alongside it: grep the
  feature's own files for raw clients/URL fields, don't trust "the SSRF PR handled
  it." (This is exactly how the egress OAuth `custom_token_url` — carrying the
  client_secret and user refresh_token — reached a bare `httpx` POST after the
  SSRF PR shipped.)
- **An OAuth token endpoint IS an outbound sink carrying secrets.** A per-provider
  `token_url` (especially a registrant-supplied "custom" one) receives the client
  secret and, on refresh, the user's refresh_token — treat it exactly like a
  proxy_pass/agent URL: validate at registration (`require_https=True`) and pin at
  fetch time. The authorize URL is a browser redirect — bound it too, but the
  token URL is where credential exfiltration happens.
- **A stricter security context must not inherit a looser guard's allowlist
  bypass — and the fix belongs on the transport, not in a pre-check.** A URL
  guard shared with a lower-risk path may have a trusted-domain allowlist that
  skips the IP check; on a credentialed egress path (e.g. federation sync
  attaching a bearer token) that bypass is a hole. Give the sensitive path its
  own **dedicated profile with an empty allowlist** and build its client from the
  **pinned guarded transport** with that profile. A pre-fetch "resolves only to
  public IPs" check layered on top of a plain client is NOT enough: the plain
  client re-resolves at connect time, so a host that passed the check can still
  rebind to a private/metadata address before the credential-bearing socket
  opens. Write-time and fetch-time must share one allowlist (both empty for the
  credential path) so a validated-then-rebound endpoint cannot slip through.
- **Keep TLS verification ON by default for privileged outbound calls.** Never
  ship `verify=False` on an admin-API client. Trust a private/self-signed cert
  via an explicit CA bundle env var (fail closed if the configured bundle is
  missing); any insecure escape hatch must be explicit opt-in, logged, default
  off.
- **Every alternate/override URL field is a bypass — validate the whole family,
  not just the primary field.** A server record often carries a main backend URL
  (`proxy_pass_url`) plus optional override endpoints (`mcp_endpoint`,
  `sse_endpoint`) that are fetched and interpolated into config exactly like the
  main URL. If only the primary field goes through the canonical guard, the
  override field is an unguarded SSRF / config-injection sink. At EVERY write path
  (register, edit, internal-register, version-add), run each present override
  field through the SAME `validate_proxy_pass_url()` as the primary; empty/unset
  is fine, present-but-invalid is rejected. Then RE-validate the resolved URL at
  fetch time (`resolve=True`) on each fetch path (health check, tool discovery,
  and especially before handing the URL to an external subprocess the pinned
  guarded client cannot protect — e.g. a scanner CLI) so a value that was rebound
  after registration, or that reached the connect through the override field, is
  still blocked before any credential is attached or process spawned.

## Injection (nginx config generation, NoSQL/regex)

- **Sanitize at EVERY interpolation site AND validate at the source.** A
  sanitizer that exists but isn't called is worthless. Apply the nginx-value
  sanitizer to every user/registry value entering a generated directive
  (`proxy_pass_url`, backend, host, path), and reject metacharacters + non-http(s)
  schemes at registration.
- **Escape user input used in `$regex`/regex-match queries** (`re.escape`), and
  never fall back to a raw user string when tokenization yields nothing. Escape
  at the SINK (the repository method that builds the query), not at the caller —
  a caller-escape contract silently breaks the moment a new caller forgets it.
- **A user-supplied value interpolated into an outbound URL PATH segment must be
  validated against a strict identifier allowlist — mirror the resource's own
  canonical name rule (e.g. `^[a-z0-9]+(-[a-z0-9]+)*\Z`) rather than a looser
  superset — BEFORE building the URL.** Otherwise a value like
  `../../api/management/iam/users` normalizes (via the HTTP client) to a
  DIFFERENT endpoint after the request is issued, and if the client carries a
  privileged credential (M2M / API token) the traversed request inherits that
  privilege. A non-empty `.strip()` check is not sufficient. Anchor the pattern
  with `\Z`, not `$`: Python's `$` also matches just before a trailing newline,
  so `^...+$` would accept `"validname\n"`. Validate at the interpolation site —
  do not trust the downstream service to reject the traversed path. Fail closed:
  return the handler's error shape, do not raise. Query parameters are lower risk
  (they don't traverse the path), but still reject absolute (`/`-leading) and
  `..`-containing values as defense-in-depth.

## Authorization & ownership

- **Deny by default; never treat a broad or execute scope as admin.** A helper
  meant for "can edit my own resource" must not gate admin-only operations.
  Require explicit `is_admin` or a named admin group/scope from centralized
  privileged constants.
- **Enforce ownership server-side before EVERY mutation — across the whole
  endpoint family, not just the reported one.** If register-overwrite needs an
  ownership check, so do the version, rename, auth-credential, and delete
  siblings. Add CSRF (or non-cookie auth) to all state-changing endpoints. The
  member that gets forgotten is usually delete/remove — a family whose update
  handlers require permission-AND-ownership but whose delete handler checks only
  permission lets a user with a delete grant destroy someone else's resource.
  Combine permission and ownership (defense in depth) uniformly, and fail closed
  when ownership cannot be established (a missing `registered_by` denies a
  non-admin).
- **`getattr(a_dict, "key", None)` always returns None** (dicts don't expose keys
  as attributes) — the guard becomes dead code that never denies. Use
  `dict.get("key")`; watch for dict-vs-Pydantic-model confusion.
- **No substring matching for privilege decisions** (`"unrestricted" in scope`
  accepted access scopes as admin). Match exact, centralized constants.
- **Reserve the wildcard/sentinel names at every write that turns user input
  into an authorization key.** If a resolver treats a magic value (`all`, `*`)
  as a cross-cutting wildcard, then any name derived from user input — a server
  registration `path` normalized (`lstrip("/")`) into a scope `server` value —
  that equals that sentinel silently escalates to "all resources". Reject the
  reserved names at the registration/validation chokepoint (fail closed, exact
  set — do not over-strip so adjacent names like `all-tools` stay valid) AND at
  the deepest write sink as defense-in-depth. Enumerate ALL write sinks: a
  direct-write path that skips the canonical `add_*` helper (e.g. a bulk
  `import_group` that persists the rule list verbatim via `replace_one`) bypasses
  the sink guard and needs its own check. Make the scan flatten/normalize/coerce
  IDENTICALLY to the resolver's read path (share the flatten helper; `str()`-
  coerce exactly as the resolver does) so the guard can never be blind to a shape
  or type the resolver would still honor as a wildcard. Keep the read-side
  sentinel set and the write-side reject set cross-referenced so they stay in
  lockstep. Existing rows written before the fix are a separate data-cleanup
  concern (audit `{"server": {"$in": ["all","*"]}}`), not covered by a code guard.
- **Canonicalize a path before any deny-list / classifier decision.** A path
  that drives an authorization decision (a resource-token deny-list, a route
  classifier) arrives from an attacker-controlled raw request URI, typically
  percent-encoded (nginx forwards `$request_uri`). Percent-decode ONCE and
  resolve `.`/`..` segments to a canonical absolute path BEFORE matching — an
  encoded traversal like `/api/agents/%2e%2e/tokens/generate` must classify
  identically to its canonical `/api/tokens/generate`, or it bypasses the
  byte-exact deny-list. Decode exactly once (a doubly-encoded `%252e` stays
  literal and fails closed), clamp traversal at root, and put the normalization
  inside the single shared entrypoint so every consumer benefits.
- **Attach a shared/global credential only on explicit opt-in.** Make the
  privileged code path default to not attaching it and gate its use behind an
  admin check.
- **A static, long-lived, non-expiring token must be least-privilege.** Never
  bundle a management/write scope onto a token whose purpose is read/data-sync.
  A token meant for federation (or any) data sync should grant only the read
  scope (`.../read`); create/update/delete of peers or config is a management
  operation that must stay behind a real admin credential, not the shared static
  token. The blast radius of a leaked never-expiring token is exactly the union
  of the scopes it carries, so keep that union minimal and audit any grant that
  couples a read scope with a management scope on the same static token.
- **Never forward the caller's inbound credential to a proxied/untrusted
  destination.** When the gateway proxies to a registrant-controlled upstream (or
  an agent calls a discovered remote agent), strip `Authorization`/`Cookie` from
  the forwarded request — the upstream is authenticated by the gateway's own
  mechanism, not by relaying the caller's registry token. nginx subrequests
  inherit the parent request's headers, so `Cookie`/`Authorization` must be
  explicitly cleared (`proxy_set_header ... "";`) on any location that proxies
  directly to a registrant-controlled backend. For outbound service-to-service
  calls, mint an audience-restricted, short-lived delegation token rather than
  re-sending the inbound one.
- **Never accept a credential as a URL query parameter.** Query strings land in
  access logs, the audit trail, and browser history. Take secrets via a request
  header or POST body; a credential that arrives in the query string should be
  ignored, not honored. Keep audit masking as defense-in-depth (substring match
  on token/secret/credential/auth/key/password, not an exact-name allowlist).
- **LLM-emitted tool calls need a fail-closed enforcement boundary, not prompt
  guidance.** Any autonomous agent loop that can run shell commands or mutating
  actions must gate them at the execution point: a mandatory human-confirmation
  step for destructive/mutating calls plus a deny-by-default executable
  allowlist, with a scrubbed environment. System-prompt "be careful" text is not
  enforcement — the LLM can be steered past it. Agent/A2A endpoints that drive
  such a loop must require authentication (validate the bearer JWT against the
  IdP JWKS) and must not bind `0.0.0.0` by default.
- **Verify externally-supplied JWTs** (signature/issuer/audience/expiry) against
  the IdP JWKS before trusting any claim. Never `verify_signature=False` on a
  token whose claims drive identity or authorization.
- **Never derive a verification decision from an unverified claim.** Decoding a
  token unverified to read `aud`/`cid` and then setting `verify_aud=False` (or
  picking the issuer/algorithm) from that value defeats the check — any
  genuinely-signed token from the same issuer, even one minted for a different
  resource in the tenant, is then accepted (confused deputy). Enforce audience
  against a config-driven allowlist with `verify_aud=True`; fail closed (reject)
  when the allowlist is unconfigured rather than accepting any audience.
- **Never auto-grant groups/roles/admin from a code-shipped mapping.** A
  hardcoded `client_id → [groups]` (or default-admin) table in an M2M/SSO sync
  path silently confers privilege. Drive the mapping from config, fail closed to
  no groups when unset/malformed, and grep every provider's sync sibling (okta,
  auth0, …) for the same pattern.
- **Never mint privilege (groups/scopes/roles) from an unverified request
  body.** A token-mint endpoint that stamps the groups/scopes carried in its
  POST body into the issued token trusts the caller to be honest about identity.
  Reconcile against an authoritative source — resolve the session id from the
  session store and mint only the intersection of requested and session-held
  groups, rejecting any privileged group the session does not hold. Where no
  session-backed source exists (pure service-to-service), keep the subset check
  but make the trust boundary explicit and fail closed on a missing/malformed
  context (a `groups: "admin"` string must be rejected, not iterated
  character-by-character). Note the residual: a caller holding the internal
  signing key can bypass the endpoint entirely, so this is defense in depth
  pending asymmetric signing.
- **Group enrichment from a mutable store must be strictly gated and audited.**
  When empty-group tokens are enriched from a DB collection, gate it to exactly
  the token class it is designed for (machine/M2M) — a check of "has some
  client_id" is not "is an M2M client"; a self-signed user token can carry a
  client_id and would otherwise be escalated from the M2M table. Gate by an
  explicit token-type marker AND the sentinel value, fail closed if either is
  missing, honor the record's disabled flag in the query AND on the returned doc,
  and emit a WARNING-level audit line whenever enrichment adds a privileged
  group so a write to the collection is attributable.
- **Protect the admin population from lockout and self-harm.** Admin user-mgmt
  must refuse self-deletion, refuse removing/demoting the LAST admin (count
  remaining admins first, fail closed if the population can't be enumerated), and
  emit a distinct audit event on any admin-tier grant. Derive "who is admin" from
  the SAME privileged-scope rule the request-time check uses, not a separate
  notion.
- **A config/secrets export must be deny-by-default for sensitive values.** An
  admin "export configuration" endpoint that can emit secrets (SECRET_KEY, IdP
  client secrets, DB passwords, API keys) concentrates the whole system's
  credentials into one response — a single over-broad export is a full compromise.
  One `include_sensitive` flag is not enough: it is easily set by habit, a copied
  script, or a CSRF/confused-deputy request. Gate the sensitive payload behind a
  SEPARATE explicit acknowledgement (e.g. `confirm_sensitive_export`) in addition
  to `include_sensitive`, reject (fail closed) when the acknowledgement is absent,
  and redact the sensitive values otherwise. Audit every sensitive export.
- **Don't disclose deployment topology / feature surface to anonymous callers —
  it's reconnaissance.** A config/topology read (deployment mode, registry mode,
  active auth provider, enabled feature flags, proxy-update state) tells an
  attacker how the system is wired before they authenticate. Gate it behind an
  authenticated session and fail closed (401) for anonymous callers. Serve the
  genuinely pre-login needs (app title, available OAuth providers, auth-server
  URL) from dedicated MINIMAL anonymous endpoints, not a broad config dump — so
  gating the config endpoint doesn't break the login page. Then sweep the OTHER
  anonymous endpoints for the same fields: a load-balancer `/health` probe, a
  `/status`, or an OpenAPI/schema dump commonly re-leaks the exact topology you
  just gated — trim them to a liveness signal (probes rely on the HTTP status,
  not the body). RFC-mandated anonymous surfaces (OAuth `.well-known` discovery)
  are the deliberate exception.
- **Honor the disabled/inactive flag everywhere access is derived, on EVERY
  request.** A user/group/client marked disabled must contribute no
  groups/scopes and be denied — enforce it at the group→scope enrichment / session
  resolve / validate point (which runs per request), not only at login, and add
  it to the DB query filter AND re-check the returned doc (defense in depth).
  Fail closed if the flag can't be read. Grep every group-source sibling (user
  groups, M2M-client groups) — the sync path may actively write `enabled: false`.
  **Treat only an explicit "active" value as active; anything else is disabled.**
  A schemaless store (MongoDB/DocumentDB) does not enforce field types, so an
  `enabled` field can hold `False`, `null`, `0`, `""`, or the string `"false"`.
  A truthiness/identity re-check like `enabled is not False` passes every one of
  those non-`False` values (`0 is not False` is `True`) — the wrong direction.
  The re-check must be `enabled is True` (with a missing field treated as active
  only for documented backward-compat), and the query filter (`{"$ne": False}`)
  is a pre-filter, not the authority. Prefer widening the deny set: anything that
  is not provably active is disabled.
- **Trust forwarded request metadata only from the proxy hop, never the client.**
  For audit client-IP, take `X-Real-IP` or the rightmost/trusted `X-Forwarded-For`
  entry (configurable proxy-hop count), and fall back to the direct peer when the
  chain is shorter than expected — never the spoofable leftmost entry. Validate
  the inbound `Host` against an allowlist (or a configured external URL) before
  using it to build an OAuth `redirect_uri` or any external URL; fail closed to
  the configured host on an unexpected `Host`.
- **Validate an OAuth login/logout `redirect_uri` against an EXACT-MATCH
  allowlist of registered callbacks, not a domain-suffix / subdomain match.** A
  "same-origin OR within `SESSION_COOKIE_DOMAIN`" check admits any host inside
  the cookie domain — including an attacker-controlled subdomain
  (`evil.example.com` under `.example.com`) — which is an open redirect. Read a
  configured allowlist of full absolute URIs (normalize scheme+host+port+path)
  and, when it is set, permit an absolute redirect ONLY if it exactly matches an
  entry; check it BEFORE any suffix fallback. Relative-path redirects (no
  scheme/host) are same-origin by construction and stay allowed. Keep the
  suffix behavior only as an unset-allowlist fallback for backward compat, and
  document the allowlist as the hardened path. Apply the same check to BOTH the
  login-success redirect and the logout `redirect_uri`. Fail closed: an absolute
  URI that is neither relative nor allowlisted nor (fallback) within-domain is
  rejected to a safe default (`/login` or `/`).
- **The session cookie's `SameSite` must stay `Lax` for OAuth login to work.**
  The cookie is set on the OAuth callback response, which is reached via a
  top-level cross-site navigation from the IdP; a `Strict` cookie is not sent on
  that navigation, so the browser drops it and login silently fails. `Strict` is
  NOT a valid "hardening" here — CSRF is defended separately by the CSRF token,
  not by `SameSite=Strict`. Leave a comment at the `set_cookie` site so a future
  reader does not "tighten" it and break login.
- **Bind the OAuth2/OIDC authorization-code flow to the specific login.** A valid
  signature is necessary but not sufficient — a correctly-signed id_token minted
  for a different login can be replayed/injected. Send a per-login `nonce` on the
  authorization request, persist it with the flow's transient state (server-side
  store or a signed integrity-protected cookie), and after signature verification
  assert `claims["nonce"] == stored_nonce`. Add PKCE (`code_challenge=S256`;
  `code_verifier` on token exchange) and fail closed if the verifier is missing on
  callback. Route every provider branch through one verification chokepoint —
  never a userInfo fallback that skips nonce binding. **Validate the CSRF `state`
  at the actual token-exchange point, not a post-hoc check** — if the callback
  handler exchanges the code inline, a `state` comparison that runs later in the
  calling flow is dead code; gate the exchange itself and fail closed when
  `state` is missing/absent/mismatched or no authorization request is in flight.
- **Cross-resource permission confusion: one entity type's access grant must
  never gate a different entity type.** A skill-listing fast-path keyed on
  `accessible_agents` (an agent grant) let a broad agent grant expose private
  skills. Filter each resource by ITS OWN access check (skill access for skills,
  server access for servers); the only universal bypass is real admin. Fail
  closed (omit) when access is unclear.
- **A dynamically-minted permission name must not collide with a
  privilege-derivation rule.** Admin here is derived from holding any
  `ui_permission` whose action starts with a mutating prefix
  (`create_/modify_/delete_/...`) granted for `["all"]`. Minting a per-type scope
  named `create_<type>_entity` would therefore SILENTLY PROMOTE anyone granted it
  to full admin. When you add a new family of dynamic scopes that reuse the
  system's `{action}_{resource}` convention: (a) put the admin-derivation rule
  behind ONE shared predicate (`is_admin_conferring_action`) in the dependency-free
  constants module, and make EVERY consumer (`_user_is_admin`, the privileged-write
  `_grants_admin`) call it — grep for stray `.startswith(PREFIXES)` checks that
  bypass it; (b) exclude the new dynamic family from admin-derivation there
  (exact regex, e.g. `^(create|modify|delete)_.+_entity$`), and keep the minted
  action set EXACTLY equal to the excluded set (add a runtime `assert not
  is_admin_conferring_action(x)` invariant so they can't drift); (c) constrain the
  `<type>`/resource name charset at the source (`^[a-z0-9_-]+$`) so the minted key
  is safe to use as a Mongo `$set` dotted field path (no `.`/`$` injection) and no
  crafted name can slip a real admin action past the exclusion. NOTE the drift
  trap: a *separate* privileged-WRITE guard that flags "any `["all"]` grant" (not
  routed through the shared predicate) will disagree with the admin-derivation
  rule about the excluded family — that disagreement is safe only if it fails
  CLOSED (over-restrictive), and any admin-population counter that consumes it will
  over-count; audit both directions.
- **Bring a new first-class asset to authorization parity across its WHOLE
  surface, not just the reported gap.** A custom/extensible entity type that has
  per-record ownership but no per-type permission layer is missing create-gating
  AND type-scoped discovery. Fixing only the ungated CREATE leaves enumeration
  open; gate list/get/create/modify/delete/rate uniformly (view gate hides
  existence with 404, mutate gate denies with 403), layer the per-type scope ON
  TOP of the existing per-record owner/visibility checks (defense-in-depth, don't
  replace them), and wire the SAME type-level gate into every discovery projection
  (search, catalog) BEFORE per-record filtering so a caller with no list scope
  sees zero records — including public ones — exactly as the equivalent
  server/agent list scope behaves.
- **Resolve every per-asset permission through one canonical `(family, action)`
  map — never pass a raw scope string to a gate.** Each asset family's gate must
  enforce its OWN family's scope; a hand-typed scope-name argument lets a sibling
  family's scope be substituted by accident, and admins mask it (their `["all"]`
  grants + is_admin bypass satisfy any name), so the bleed sits latent. Keep the
  scope-name convention in ONE typed table keyed by `(family, action)`
  (`registry/auth/asset_permissions.py`: `asset_scope_name` /
  `user_has_asset_permission`), and have call sites pass a logical ACTION
  ("modify", "toggle") — not a scope string — so the family is fixed by the call
  and the map picks the name. Preserve the EXACT on-disk names in the map
  (persisted `ui_permissions` keys — do not "normalize" `list_service` vs
  `list_agents` pluralization; renaming a persisted key is a breaking data
  migration, not a refactor). After adding a family, grep for gates that bypass
  the map (a raw `user_has_ui_permission_for_service("..._<otherfamily>"`, an
  inline `ui_permissions.get("<scope>")`) and confirm each enforces its own
  family. When a gate starts enforcing a scope that was previously unenforced,
  provision that scope on EVERY seed surface at once (seed JSONs, the Helm
  `mongodb-configure` configmap, the IAM UI permission keys) and ship a dry-run
  backfill — otherwise existing non-admins lose the access on upgrade.
- **Authorize the exact bytes you act on — never a separately-captured copy.** If
  one component captures a request body for the authz decision and another
  forwards a different copy to the sink, they can diverge (size-triggered
  spill-to-file where the capture is empty, whitespace/newline normalization,
  content-length games) and the sink executes what the authorizer never saw.
  Re-run the scope/tool check inline on the forwarded bytes immediately before the
  outbound call; when the scope-relevant body can't be captured/parsed, FAIL
  CLOSED — do not default to an unprivileged method.
- **Redaction and access checks on reads must be uniform across the whole entity
  family.** The same internal fields (backend URLs, authz config) leak through the
  read sibling that forgot the guard — `/versions`, bulk `/all`, discovery
  projections that re-project the field under a new name, search-result shaping,
  admin-config reads. Use one shared redaction-decision helper + field-stripper;
  gate authz-model reads behind the same admin check as their writes; for an
  unauthenticated public surface fail closed to the derived URL and never emit a
  stored internal override. **This applies to per-caller LIST PRUNING, not just
  field redaction:** a nested collection whose visibility is scoped per user
  (e.g. a server's `tool_list`, filtered by the caller's tool allowlist) must be
  pruned by the SAME shared filter on every endpoint that returns it — the
  listing, the single-item GET, the detail/`server.json`/catalog/search
  projections. The single-item GET is the one most often missed after the list
  endpoint is fixed (this was the `get_server` / `get_server_details` tool-name
  leak that survived the catalog fix). Keep any derived count (`num_tools`)
  consistent with the pruned list so the count can't become an enumeration
  oracle. Fail closed (empty) on a missing/empty allowlist; admin/wildcard pass
  through.
- **Redact the DERIVED field, not just the raw source field.** Nulling
  `proxy_pass_url`/`mcp_endpoint` on a response is not enough if a *computed*
  field (a "connect URL", `endpoint_url`, `transport.url`) is built from that
  same internal value and returned alongside — the URL-builder often echoes an
  explicit endpoint override verbatim as its top-priority branch. Make the
  builder itself redaction-aware (pass the redaction decision in; when set,
  return only the public/gateway form or None), and assert in tests that the
  internal host is absent from the derived field, not only from the raw one.
  (This is exactly the leak that survived the first pass on semantic search's
  `endpoint_url` while the raw fields were already nulled.)

## Frontend (React) URL / href handling

- **One shared URL-scheme guard, applied to every dynamic href / `window.open` /
  markdown link.** React does not block `javascript:` (or `data:`/`vbscript:`) in
  `href`. Any dynamic URL rendered from server-registration or federation data is
  a stored-XSS vector that runs in the viewer's authenticated session on click.
  Allowlist `http`/`https`/`mailto`; strip control chars + whitespace before
  scheme extraction and lowercase it (defeats `Java\tScript:`, leading-space, NUL,
  mixed case); render unsafe schemes as inert text. Enforce with the
  `react/jsx-no-script-url` lint rule so regressions are caught. (Server-side
  template `innerHTML`/inline-`onclick` is a separate sink — guard it too.)

## Deployment surface & insecure-by-default config

- **Publish sensitive ports on loopback, not `0.0.0.0`.** In compose/dev stacks,
  bind datastores, vaults, IdP admin ports, and backend services to `127.0.0.1:`;
  only the intended front door (nginx 80/443) listens on all interfaces.
- **Never ship a working default for a required secret.** Use `${VAR:?message}`
  so the stack fails fast, and reject known-weak literals in a preflight
  validator. An active `PASSWORD=admin` in `.env.example` (vs a blank/commented
  placeholder) seeds real deployments with the weak value.
- **Harden a shared secret on EVERY reference, not just the container that owns
  it.** A vault's root token and every client that authenticates with the same
  token must both be `${VAR:?}` — if the owning service fails fast but a client
  env still says `${VAR:-dev-root-token}`, the weak literal is back (and the
  client silently can't reach the strong-token service). Grep every occurrence of
  the variable across all compose files when you harden one.
- **A fail-fast `${VAR:?}` needs a matching generator, or the standard entrypoint
  dead-ends.** If `build_and_run.sh` (or equivalent) doesn't auto-generate the
  now-required secret into `.env`, a fresh run aborts at compose-up. Add a
  generator block (idempotent: only when missing/empty; never clobber an
  operator value) for every secret you make required.
- **A non-empty `.env.example` placeholder passes `${VAR:?}` presence but is still
  a known credential.** Either ship the placeholder BLANK (so `:?` catches the
  copy-verbatim deploy) or add the exact placeholder string to the preflight
  denylist. Match the denylist CASE-INSENSITIVELY — an exact-match check is
  trivially dodged by a case-mutated copy (`Change-Me-...`).
- **No secrets as Docker build `ARG`s** — they persist in image history. Inject at
  runtime.
- **Don't bind-mount broad credential dirs** (e.g. `~/.aws`) into a
  network-facing container by default; mount the single needed file or use a
  scoped role.
- **Pin image tags; never `:latest`.** And raise dependency floors above known
  CVEs even when a lockfile currently mitigates — off-lock installs are exposed;
  prefer consolidating on one library over carrying a redundant vulnerable one.
- **Sweep EVERY manifest for a floor, not just the one in the finding.** The same
  dependency is often declared in the root project, each sub-project, and the
  Lambda/ops `requirements.txt` — a floor patched in one can stay vulnerable in
  another (e.g. `aiohttp>=3.14.0` in auth_server while the root still says
  `>=3.8.0`). Grep the dependency name across all manifests and raise every
  occurrence. Floor to the *first patched release* that clears the CVE, not the
  newest version — a floor bump is a security minimum, not a version chase.
- **A removed vulnerable dependency needs a regression guard that covers every
  manifest AND lockfile.** A test that only asserts absence in one sub-project
  lets the library reappear via the root or another sub-project uncaught.
  Parametrize the guard over all `pyproject.toml`/`requirements.txt` and all
  `uv.lock` files, and scan every source tree for the import.
- **Regenerate lockfiles with the project's own tooling, not a bare `uv lock`.**
  This repo uses `make uv-update-locks`, which stamps the `[options] exclude-newer`
  quarantine header on every lock; a raw `uv lock` drops that header (and bumps
  the revision format), producing an inconsistent, non-reproducible lockfile. Do
  not run `uv run` between relock and commit — it re-resolves the root lock against
  the older committed cutoff and silently strips the freshly-written header.
- **A dangerous operational toggle** (disabling TLS, wiping data) must require an
  explicit acknowledgement flag AND a fail-closed environment guard (e.g.
  localhost-only). Setup scripts must force credential rotation (`temporary:
  true`) and never grant privileged scopes to anonymous dynamic client
  registration.
- **IAM policies: least privilege, no wildcard Action or Resource.** Scope
  `Action` to the exact operations the code actually calls (grep the client) and
  `Resource` to specific ARNs (or an account/region-pinned pattern), never `*`.
  Gate cross-account `sts:AssumeRole` behind an explicit configured role-ARN list
  that defaults empty (fail closed — no trust when unset). Keep IaC surfaces
  (Terraform + CDK) in parity.

## Availability (DoS) & audit

- **Rate-limit at the inbound edge, not the shared internal dependency.** When
  many authenticated locations fan out to one internal subrequest (e.g. nginx
  `auth_request /validate`), bound the request rate on the inbound edge locations
  — never on the internal target (that throttles legitimate auth). Cover exact-
  match locations that don't fall through to a prefix, and any dynamically-
  generated location blocks. Apply it at the layer common to ALL deployment modes
  (nginx `limit_req`/`limit_conn`), since infra WAFs are topology-specific and
  often default-off. Rate-limit-classifier maps key off the normalized `$uri`
  with a fail-safe empty default (a miss only broadens coverage). Choose generous
  defaults + burst so normal bursty clients aren't broken.
- **Audit must be durable by default and attributable.** If audit is enabled but
  no durable sink is available, fail closed at startup (loud opt-out for dev), and
  emit a distinct CRITICAL log if a record is dropped at runtime (don't fail the
  request — that's a self-DoS). Internal-service actions must be attributable to a
  specific actor (per-instance/per-purpose `sub`), not a shared service identity.
  Tamper-evidence (append-only / HMAC chain) is best served by an immutable
  external store — infra, deferrable. **Run the durable-sink guard in EVERY
  process that initializes an audit logger, not just the main app's startup.** A
  multi-process deployment (registry + auth-server) can have a second process
  that builds its own audit sink lazily (e.g. the auth-server token-mint logger)
  — if that path swallows the sink-init exception and degrades to a
  drop-everything logger, the most forensically critical records (token issuance)
  vanish silently while the main process looks healthy. Call the same
  fail-closed guard where each logger is constructed, re-raise past any generic
  `except`, and prime it at that process's startup so it fails to boot rather
  than lazily on first use.
- **Client IP for audit/attribution: derive from a non-spoofable source, and
  scope the ASGI proxy-header trust to the real peer.** The left-most
  `X-Forwarded-For` entry is fully client-controlled — never use it. Resolve via
  the shared `get_client_ip()` (`registry/utils/request_utils.py`): prefer nginx's
  `X-Real-IP`, else the `TRUSTED_PROXY_HOPS`-th entry counted from the RIGHT (the
  hop a trusted proxy appended), else the direct socket peer; validate every
  candidate as a well-formed IP and fail toward the peer. Separately, the ASGI
  server's own forwarded-header trust must be scoped to the actual upstream:
  launch uvicorn with `--forwarded-allow-ips` set to the real peer (loopback
  `127.0.0.1,::1` when nginx sits in the same pod/container), **never `*`**. With
  `*`, uvicorn's `ProxyHeadersMiddleware` overwrites `request.client` with the
  left-most (spoofable) XFF entry, poisoning the direct-peer fallback in
  `get_client_ip()` and any other code that reads `request.client`. This is
  defense-in-depth for the resolver above (which already ignores the left-most
  entry) — but the two must agree, or a code path that falls through to
  `request.client` silently trusts a forged value. Scoping to loopback is safe:
  the real client IP still arrives via `X-Real-IP`, and scheme/HTTPS detection
  reads the `X-Forwarded-Proto` header directly (not uvicorn's scheme rewrite), so
  OAuth redirect construction is unaffected.
- **An nginx `auth_request` subrequest does NOT inherit the parent location's
  `proxy_set_header` directives.** Headers the outer location sets for its own
  `proxy_pass` (e.g. `X-Real-IP $remote_addr`, a sanitized `X-Forwarded-For`) are
  absent on the `/validate`-style subrequest unless set again INSIDE the
  subrequest's own `location` block. A subrequest with `proxy_pass_request_headers
  on` therefore forwards the raw, client-supplied headers verbatim to the auth
  backend — so the backend can see a spoofed `X-Forwarded-For` even when the outer
  location "sanitized" it. When an auth/validate backend derives a client IP or
  any trusted value from headers, set (and overwrite) those headers explicitly in
  the subrequest's `location` block; do not assume outer-block sanitization
  reaches it. Corollary for reviewers: a `1.2.3.4:0` line in the ASGI *access log*
  proves `request.client` was set from a forged header, but says NOTHING about the
  *audit* value — those come from two different resolvers; check the durable audit
  record before concluding a spoof succeeded.

## Canonical helpers — use these, never reinvent or copy-paste

There is exactly ONE blessed implementation per concern. Before writing a new
outbound HTTP call, auth gate, redaction, or secret check, use the helper below.
A hand-rolled or copy-pasted variant is how these vulnerabilities get reopened —
the drift between copies is the hole.

- **Outbound HTTP from any user/registry/federation-controlled URL →
  `registry/utils/url_guard.py`.** Never construct a bare `httpx.Client` /
  `httpx.AsyncClient` (or a third-party SDK client) for such fetches. Use
  `guarded_client(profile=...)` / `guarded_async_client(profile=...)` — they
  validate + pin the resolved public IP inside the transport (rebinding-safe,
  re-validated per redirect). Pick the profile: `SKILL_PROFILE` (skill/doc
  fetches) or `PROXY_PROFILE` (server/agent proxy targets). Validate at
  registration with `validate_proxy_pass_url()` / `validate_agent_url()` /
  `validate_server_path()`; reject nginx metacharacters with
  `contains_nginx_metacharacters()`. Internal targets are opt-in via
  `SSRF_ALLOWED_HOSTS` / `SSRF_ALLOWED_CIDRS`, default deny. (Legacy
  `ard_net_guard.py` predates this — prefer `url_guard`; do not add new callers
  to ad-hoc `_is_safe_url` variants.)
- **State-changing endpoint CSRF → `registry/auth/csrf.py`.** Add
  `Depends(verify_csrf_token_flexible)` (or `verify_csrf_token_header_only`) to
  every mutating route. Don't invent per-router CSRF logic; match the dependency
  every other router uses. **Apply it UNIFORMLY across the ENTIRE session-auth
  mutation family, not just the reported route** — a per-route dependency is
  trivial to forget on a new endpoint, so grep the whole family (every
  `@router.(post|put|patch|delete)` handler that depends on the session-cookie
  auth dependency — `enhanced_auth` AND `nginx_proxied_auth`, which falls back to
  the cookie — and mutates state) and close every gap. `verify_csrf_token_flexible`
  correctly NO-OPS for non-cookie (Bearer/CLI) callers, so it is safe to add to a
  route even if some callers are non-browser. It reads the token from the
  `X-CSRF-Token` header OR a `csrf_token` form field, so a JSON-body route works
  fine via the header. Skip only routes authed purely by `validate_internal_auth`
  / internal-token (not browser/cookie surfaces).
- **Internal service-to-service auth → `registry/auth/internal.py`.**
  `generate_internal_token()` to mint, `validate_internal_auth` /
  `validate_internal_session_secret` as the route dependency. Internal tokens are
  a distinct trust class from user tokens — do not hand-roll JWT checks that
  accept both.
- **Signing-secret validation → `registry/common/secret_key.py`**
  (`validate_secret_key` / `validate_signing_secret`). Any new secret that signs
  or seals must go through it (fail-closed: missing/weak/short). *(Arrives with
  the secret-key-hardening change; until merged, that module lives on that
  branch.)*
- **Backend-URL / authz-config redaction on reads →
  `registry/services/visibility.py`.** Use the shared access check
  (`user_can_access_*_from_doc`) and the shared redaction decision + field
  stripper for every read that serializes a server/agent entity — not a
  per-endpoint reimplementation. *(The redaction helpers arrive with the
  readonly-info-disclosure change; the access helpers are already on main.)*
- **Frontend dynamic URLs → the shared `isSafeUrl` / `SafeLink`
  (`frontend/src/utils/safeUrl.ts`, `frontend/src/components/SafeLink.tsx`).**
  Never bind a server/federation-supplied value straight into `href` /
  `window.open` / a markdown link. *(Arrives with the frontend-xss-hrefs change.)*
- **Privileged outbound TLS →** trust private certs via an explicit CA-bundle env
  var, never `verify=False`. `guarded_client`/`guarded_async_client` take
  `verify=` and default it to `True`.
- **Log redaction → `registry/common/log_redaction.py`** (`redact_headers`,
  `redact_mapping`, `redact_url` for a single URL string) and, for OIDC identity,
  `safe_identity_summary()` in
  `auth_server/server.py`. NEVER log a raw header dict, a request/response body
  dict, a `user_context`, `updates`, or decoded id_token claims — route it
  through the redactor (masks any `*token*`/`*secret*`/`*credential*`/auth/cookie
  key) or log identifiers/counts only. This is the exact pattern that gets
  re-introduced by a casual `logger.info(request.headers)` — reuse the helper.
  *(Arrives with the sensitive-logging change; the header redactor also has a
  sibling in `registry/utils/request_utils.py::redact_sensitive_headers`.)*
- **Writing a credential to a file → create it owner-only and atomically:**
  `os.open(path, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)` then `fdopen`.
  Never `open()`-then-`chmod` (a readable window), never the process umask,
  never print a minted secret (even a prefix) to stdout — write it to a 0600
  file and print the path. *(Applied in the cli-secret-hygiene change; candidate
  for a shared `write_secret_file()` helper.)*
- **Never pass a secret as a subprocess command-line argument.** argv is
  world-readable via `ps` / `/proc/<pid>/cmdline` to any local user for the life
  of the child. Hand it to the child via `env=` (a name-only reference like
  `MCP_SCANNER_LLM_API_KEY`) or stdin. When an unavoidable third-party tool only
  accepts the secret on argv, keep it off every hop you control and record the
  external constraint as a scoped residual (see FOLLOWUPS F-19/F-20).

When you add a NEW canonical helper, list it here so the next agent reaches for
it instead of rebuilding it.

## Cross-cutting habit

**When you fix a finding, grep for the same pattern repo-wide before closing.**
Almost every finding has siblings the report didn't list — extra fetch sinks,
extra unowned endpoints, extra log sites. The root-cause fix plus a repo-wide
sweep beats patching the single reported line.
