// Command go-validate is a fast-path sidecar for the auth-server's /validate
// endpoint. It verifies the RS256 bearer tokens that carry the bulk of gateway
// traffic and reverse-proxies everything else to the unchanged Python auth-server.
// Design: .scratchpad/ant-hackathon-aug-2026/final/lld.md
package main

import (
	"crypto/hmac"
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"os"
	"strings"
	"sync/atomic"
)

// identity is the resolved caller identity written into the response headers.
type identity struct {
	Sub      string
	Username string
	ClientID string
}

// counters is a tiny lock-free metrics surface exposed at /metrics.
type counters struct {
	fastOK   atomic.Int64
	unauth   atomic.Int64
	fallback atomic.Int64
}

type server struct {
	cfg      Config
	ks       *keysetCache
	scopes   *scopeResolver
	fallback http.Handler
	stats    counters
}

// perUserIdPMethods fold to the single canonical egress bucket "oauth2" in the
// internal tokens (mirrors _canonical_auth_method / egress_auth.canonical_auth_method).
// The per-user egress vault keys on this, so consent-write and vend-read must agree.
var perUserIdPMethods = map[string]bool{
	"session_cookie": true, "self_signed": true, "keycloak": true, "entra": true,
	"cognito": true, "okta": true, "auth0": true, "pingfederate": true,
	"jwt": true, "boto3": true,
}

// missingReason lists which fast-path prerequisites are unset, so a FALLBACK-ONLY
// startup log tells the operator exactly what to fix instead of just "not ready".
func missingReason(cfg Config) string {
	missing := []string{}
	if cfg.SecretKey == "" {
		missing = append(missing, "SECRET_KEY")
	}
	if cfg.JWKSURL == "" {
		missing = append(missing, "JWKS_URL (or KEYCLOAK_URL to derive it)")
	}
	if len(cfg.Issuers) == 0 {
		missing = append(missing, "VALIDATE_ISSUER (or KEYCLOAK_URL/KEYCLOAK_EXTERNAL_URL to derive it)")
	}
	if len(cfg.Audiences) == 0 {
		missing = append(missing, "VALIDATE_AUDIENCE (or KEYCLOAK_CLIENT_ID/KEYCLOAK_M2M_CLIENT_ID to derive it)")
	}
	if len(missing) == 0 {
		return "no missing config"
	}
	return "missing " + strings.Join(missing, ", ")
}

// canonicalAuthMethod returns the egress-principal bucket stamped into the internal
// tokens. Per-user IdP methods canonicalize to "oauth2"; others pass through.
func canonicalAuthMethod(method string) string {
	if perUserIdPMethods[method] {
		return "oauth2"
	}
	return method
}

// serverNameFromOriginalURL extracts the first path segment of X-Original-URL
// (the MCP server name / traversal-guard segment). Empty for /api/ and root.
func serverNameFromOriginalURL(original string) string {
	u, err := url.Parse(original)
	if err != nil {
		return ""
	}
	p := strings.Trim(u.Path, "/")
	if p == "" || strings.HasPrefix(p, "api/") || p == "api" {
		return ""
	}
	if i := strings.IndexByte(p, '/'); i >= 0 {
		return p[:i]
	}
	return p
}

// extractBearer returns the bearer token, honoring X-Authorization over
// Authorization. The A2A rule: an /agent/... path must NOT fall back from
// X-Authorization to Authorization (that header carries the target agent's
// credential and is forwarded end-to-end).
func extractBearer(r *http.Request) (string, bool) {
	isAgentPath := strings.Contains(r.Header.Get("X-Original-URL"), "/agent/")
	if xa := r.Header.Get("X-Authorization"); xa != "" {
		return parseBearer(xa)
	}
	if isAgentPath {
		return "", false
	}
	if a := r.Header.Get("Authorization"); a != "" {
		return parseBearer(a)
	}
	return "", false
}

func parseBearer(v string) (string, bool) {
	const p = "Bearer "
	if len(v) > len(p) && strings.EqualFold(v[:len(p)], p) {
		return strings.TrimSpace(v[len(p):]), true
	}
	return "", false
}

// mapClaims turns verified claims into a caller identity. Keycloak claim shape
// today; additional IdPs add a case here (config + claim map, not a rewrite).
func mapClaims(c *Claims) identity {
	clientID := c.Azp
	if clientID == "" {
		clientID = c.ClientID
	}
	username := c.Username
	if username == "" {
		username = c.Sub
	}
	return identity{
		Sub:      c.Sub,
		Username: username,
		ClientID: clientID,
	}
}

// handleValidate is the hot path: verify the RS256 bearer and either answer 200
// with identity headers, 401 for a recognized-invalid token, or fall back to
// Python for anything unrecognized (cookies, other IdPs, opaque tokens).
func (s *server) handleValidate(w http.ResponseWriter, r *http.Request) {
	// nginx's auth_request subrequest never carries a usable body, but for a
	// POST/PUT/PATCH origin request nginx forwards the original Content-Length
	// with NO body. httputil.ReverseProxy would then block copying that phantom
	// body to the auth-server, hanging the subrequest until nginx times out
	// (504 -> auth_request collapses it to 500). Every mutating request (e.g.
	// POST /api/tokens/generate) hit this; GETs did not. Neutralize the body so
	// both the fast path and the fallback operate on headers only. /validate
	// reads identity from headers/cookies, never from the body.
	if r.Body != nil {
		_ = r.Body.Close()
	}
	r.Body = http.NoBody
	r.ContentLength = 0
	r.Header.Del("Content-Length")

	// NOTE: do NOT strip request headers before falling back. nginx sets
	// legitimate inputs on the /validate subrequest (e.g. X-Client-Id from
	// $http_x_client_id, X-Original-URL, X-Registry-Api-Auth). The fallback must
	// be byte-identical to a direct nginx->Python /validate call, and Python is
	// authoritative for identity there (exactly as today, sidecar or not).
	if !s.cfg.FastPathReady {
		s.stats.fallback.Add(1)
		s.fallback.ServeHTTP(w, r)
		return
	}

	tok, ok := extractBearer(r)
	if !ok {
		s.stats.fallback.Add(1)
		s.fallback.ServeHTTP(w, r) // cookie / no-bearer -> Python
		return
	}

	claims, err := verifyRS256(tok, s.ks, s.cfg.Issuers, s.cfg.Audiences)
	switch err {
	case nil:
		// verified below
	case errInvalidToken:
		s.stats.unauth.Add(1)
		w.Header().Set("WWW-Authenticate", "Bearer")
		w.WriteHeader(http.StatusUnauthorized) // recognized but invalid -> 401 (fail closed)
		return
	default:
		// errNotJWT / errUnknownKey / errWrongAlg -> other IdP / opaque / unknown kid
		s.stats.fallback.Add(1)
		s.fallback.ServeHTTP(w, r)
		return
	}

	// Fast path. Resolve scopes exactly as Python does (group->scope mapping,
	// M2M enrichment). If we cannot resolve them safely (no DB snapshot, or a
	// user token that would need idp_user_groups enrichment), fall back.
	ident := mapClaims(claims)
	if s.scopes == nil {
		s.stats.fallback.Add(1)
		s.fallback.ServeHTTP(w, r)
		return
	}
	scopes, ok := s.scopes.resolve(claims.Groups, ident.ClientID)
	if !ok {
		s.stats.fallback.Add(1)
		s.fallback.ServeHTTP(w, r)
		return
	}

	// Identity in the RESPONSE is fully controlled below (Set overwrites), so a
	// client cannot inject identity even though we never mutate the request.
	serverName := serverNameFromOriginalURL(r.Header.Get("X-Original-URL"))
	egressUser := ident.Sub                              // canonical egress vault id = OIDC sub (bearer callers)
	canonMethod := canonicalAuthMethod(s.cfg.AuthMethod) // internal-token auth_method claim

	h := w.Header()
	h.Set("X-User", ident.Username)
	h.Set("X-Username", ident.Username)
	h.Set("X-Client-Id", ident.ClientID)
	h.Set("X-Scopes", scopesToHeader(scopes))
	h.Set("X-Auth-Method", s.cfg.AuthMethod)
	h.Set("X-Server-Name", serverName)
	h.Set("X-Tool-Name", "")
	h.Set("X-Groups", strings.Join(claims.Groups, " "))

	// Registry /api/ hop: thin identity token, minted only when nginx set the marker.
	if r.Header.Get("X-Registry-Api-Auth") != "" {
		if tok, err := mintRegistryUIToken(
			s.cfg.SecretKey, ident.Username, "", claims.Groups,
			canonMethod, ident.ClientID, egressUser,
		); err == nil {
			h.Set("X-Internal-Token-Registry", tok)
		} else {
			log.Printf("could not mint registry-ui token: %v", err)
		}
	}

	// /mcp-proxy hop: scope+upstream-bound token, minted only when nginx forwarded
	// the resolved upstream AND (if configured) the matching source-secret marker.
	if up := r.Header.Get("X-Resolved-Upstream"); up != "" {
		if s.cfg.MarkerSecret == "" ||
			hmac.Equal([]byte(r.Header.Get("X-Validate-Source-Secret")), []byte(s.cfg.MarkerSecret)) {
			if tok, err := mintMCPProxyToken(
				s.cfg.SecretKey, ident.Username, scopes, serverName, up,
				canonMethod, egressUser,
			); err == nil {
				h.Set("X-Internal-Token", tok)
			} else {
				log.Printf("could not mint mcp-proxy token: %v", err)
			}
		} else {
			log.Printf("X-Resolved-Upstream present but source-secret marker mismatch; not minting")
		}
	}

	s.stats.fastOK.Add(1)
	w.WriteHeader(http.StatusOK)
}

// handleHealth reports readiness (B5). Degraded when the fast path is enabled but
// the JWKS keyset is not currently healthy.
func (s *server) handleHealth(w http.ResponseWriter, _ *http.Request) {
	if s.cfg.FastPathReady && !s.ks.healthy.Load() {
		w.WriteHeader(http.StatusServiceUnavailable)
		_, _ = w.Write([]byte("degraded: jwks unhealthy\n"))
		return
	}
	w.WriteHeader(http.StatusOK)
	_, _ = w.Write([]byte("ok\n"))
}

// handleMetrics exposes counters plus two health gauges so a "fast path silently
// not working" state is visible on a dashboard, not just in logs: govalidate_
// fastpath_ready (is it configured to accelerate at all) and govalidate_jwks_
// healthy (can it currently verify tokens). When ready=1 but jwks_healthy=0, the
// fast path is degraded and everything is falling back to Python — alert on that.
func (s *server) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	ready := int64(0)
	if s.cfg.FastPathReady {
		ready = 1
	}
	jwksHealthy := int64(0)
	refreshFails := int64(0)
	if s.ks != nil {
		if s.ks.healthy.Load() {
			jwksHealthy = 1
		}
		refreshFails = s.ks.refreshFails.Load()
	}
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	_, _ = w.Write([]byte(
		"# HELP govalidate_fastpath_ready 1 if the fast path is configured (else all requests proxy to Python)\n" +
			"# TYPE govalidate_fastpath_ready gauge\n" +
			"govalidate_fastpath_ready " + itoa(ready) + "\n" +
			"# HELP govalidate_jwks_healthy 1 if the JWKS keyset is currently loaded (else fast path degraded)\n" +
			"# TYPE govalidate_jwks_healthy gauge\n" +
			"govalidate_jwks_healthy " + itoa(jwksHealthy) + "\n" +
			"# TYPE govalidate_jwks_refresh_failures_total counter\n" +
			"govalidate_jwks_refresh_failures_total " + itoa(refreshFails) + "\n" +
			"# TYPE govalidate_fastpath_ok counter\n" +
			"govalidate_fastpath_ok " + itoa(s.stats.fastOK.Load()) + "\n" +
			"# TYPE govalidate_unauthorized counter\n" +
			"govalidate_unauthorized " + itoa(s.stats.unauth.Load()) + "\n" +
			"# TYPE govalidate_fallback counter\n" +
			"govalidate_fallback " + itoa(s.stats.fallback.Load()) + "\n"))
}

func itoa(n int64) string {
	return strconvFormat(n)
}

// runHealthcheck is invoked via `go-validate -healthcheck` (used by the container
// healthcheck, since the distroless image has no shell/curl). It GETs the local
// /health endpoint and exits 0 on 200, 1 otherwise.
func runHealthcheck(listen string) {
	addr := listen
	if strings.HasPrefix(addr, ":") {
		addr = "127.0.0.1" + addr
	}
	resp, err := http.Get("http://" + addr + "/health")
	if err != nil {
		log.Fatalf("healthcheck failed: %v", err)
	}
	defer resp.Body.Close()
	if resp.StatusCode != http.StatusOK {
		log.Fatalf("healthcheck: status %d", resp.StatusCode)
	}
}

func main() {
	if len(os.Args) > 1 && os.Args[1] == "-healthcheck" {
		runHealthcheck(getenv("GOVALIDATE_LISTEN", ":8899"))
		return
	}
	cfg := loadConfig()

	target, err := url.Parse(cfg.FallbackURL)
	if err != nil {
		log.Fatalf("invalid AUTH_FALLBACK_URL %q: %v", cfg.FallbackURL, err)
	}
	s := &server{
		cfg:      cfg,
		fallback: httputil.NewSingleHostReverseProxy(target),
	}
	if cfg.JWKSURL != "" {
		s.ks = newKeysetCache(cfg.JWKSURL, cfg.JWKSRefreshSec)
	}
	// Scope resolver loads mcp_scopes + idp_m2m_clients snapshots for group->scope
	// parity with Python. Nil when DB is unconfigured -> handler falls back.
	s.scopes = newScopeResolver(cfg.ScopeTTLSec)

	mux := http.NewServeMux()
	mux.HandleFunc("/validate", s.handleValidate)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/metrics", s.handleMetrics)

	if cfg.FastPathReady {
		log.Printf("go-validate listening on %s | mode=fast-path | fallback=%s", cfg.Listen, cfg.FallbackURL)
		log.Printf("accepted issuers=%v | accepted audiences=%v", cfg.Issuers, cfg.Audiences)
	} else {
		// Loud: an operator who deployed the sidecar expecting acceleration must
		// see WHY it is only proxying, not discover it from a flat latency graph.
		log.Printf("WARN go-validate listening on %s in FALLBACK-ONLY mode: %s. Every /validate request is proxied to Python (correct, NOT accelerated). fallback=%s",
			cfg.Listen, missingReason(cfg), cfg.FallbackURL)
	}
	log.Fatal(http.ListenAndServe(cfg.Listen, mux))
}
