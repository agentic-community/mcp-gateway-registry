// Command go-validate is a fast-path sidecar for the auth-server's /validate
// endpoint. It verifies the RS256 bearer tokens that carry the bulk of gateway
// traffic and reverse-proxies everything else to the unchanged Python auth-server.
// Design: .scratchpad/ant-hackathon-aug-2026/final/lld.md
package main

import (
	"log"
	"net/http"
	"net/http/httputil"
	"net/url"
	"strings"
	"sync/atomic"
)

// identity is the resolved caller identity written into the response headers.
type identity struct {
	Sub      string
	Username string
	ClientID string
	Scopes   string
	Groups   string
	Method   string
}

// trustHeaders are identity/trust headers a client must never be able to inject
// (B2). They are stripped from every inbound request before verification.
var trustHeaders = []string{
	"X-User", "X-Username", "X-Client-Id", "X-Scopes",
	"X-Groups", "X-Auth-Method", "X-Internal-Token-Registry",
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
	fallback http.Handler
	stats    counters
}

// stripClientTrustHeaders removes any client-supplied identity headers (B2).
func stripClientTrustHeaders(r *http.Request) {
	for _, h := range trustHeaders {
		r.Header.Del(h)
	}
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

// mapClaims turns verified claims into an identity. Keycloak claim shape today;
// additional IdPs add a case here (config + claim map, not a rewrite).
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
		Scopes:   c.Scope,
		Groups:   strings.Join(c.Groups, " "),
		Method:   "go-fastpath",
	}
}

// writeIdentityHeaders sets the headers nginx consumes via auth_request_set.
func writeIdentityHeaders(w http.ResponseWriter, ident identity, internal string) {
	h := w.Header()
	h.Set("X-User", ident.Username)
	h.Set("X-Username", ident.Username)
	h.Set("X-Client-Id", ident.ClientID)
	h.Set("X-Scopes", ident.Scopes)
	h.Set("X-Groups", ident.Groups)
	h.Set("X-Auth-Method", ident.Method)
	h.Set("X-Internal-Token-Registry", internal)
}

// handleValidate is the hot path: strip trust headers, verify the RS256 bearer,
// and either answer 200 with identity headers, 401 for a recognized-invalid
// token, or fall back to Python for anything unrecognized.
func (s *server) handleValidate(w http.ResponseWriter, r *http.Request) {
	stripClientTrustHeaders(r) // B2: never trust client-supplied identity headers

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

	claims, err := verifyRS256(tok, s.ks, s.cfg.Issuer, s.cfg.Audience)
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

	ident := mapClaims(claims)
	internal, err := mintInternalToken(ident, s.cfg.Audience, s.cfg.SecretKey)
	if err != nil {
		// Minting should never fail; if it does, defer to Python rather than 500.
		s.stats.fallback.Add(1)
		s.fallback.ServeHTTP(w, r)
		return
	}
	writeIdentityHeaders(w, ident, internal)
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

func (s *server) handleMetrics(w http.ResponseWriter, _ *http.Request) {
	w.Header().Set("Content-Type", "text/plain; version=0.0.4")
	_, _ = w.Write([]byte(
		"govalidate_fastpath_ok " + itoa(s.stats.fastOK.Load()) + "\n" +
			"govalidate_unauthorized " + itoa(s.stats.unauth.Load()) + "\n" +
			"govalidate_fallback " + itoa(s.stats.fallback.Load()) + "\n"))
}

func itoa(n int64) string {
	return strconvFormat(n)
}

func main() {
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

	mux := http.NewServeMux()
	mux.HandleFunc("/validate", s.handleValidate)
	mux.HandleFunc("/health", s.handleHealth)
	mux.HandleFunc("/metrics", s.handleMetrics)

	mode := "fast-path"
	if !cfg.FastPathReady {
		mode = "FALLBACK-ONLY (SECRET_KEY/JWKS_URL/VALIDATE_ISSUER/VALIDATE_AUDIENCE not all set)"
	}
	log.Printf("go-validate listening on %s | mode=%s | fallback=%s", cfg.Listen, mode, cfg.FallbackURL)
	log.Fatal(http.ListenAndServe(cfg.Listen, mux))
}
