package main

import (
	"fmt"
	"log"
	"os"
	"strings"
)

// Config holds the sidecar's runtime settings, all sourced from the environment.
// The fast path is enabled only when SecretKey, JWKSURL, Issuer and Audience are
// all present; otherwise the sidecar runs in fallback-only mode (every request is
// reverse-proxied to the Python auth-server). This fails closed: when we cannot
// safely verify a token ourselves, Python remains authoritative.
type Config struct {
	Listen    string
	SecretKey string
	// Provider selects the verifier: "keycloak" (default) or "cognito". Other IdPs
	// are not fast-pathed yet and stay in fallback-only mode (deferred to Python).
	Provider string
	JWKSURL  string
	// Issuers/Audiences are lists: a Keycloak token's iss is whatever host the
	// client reached Keycloak through (external URL for browser logins, internal
	// or localhost for service/M2M callers), and its aud names the specific
	// client. Matching ANY member mirrors the Python Keycloak provider, which
	// accepts three issuer URLs and a set of gateway-identifying audiences. For
	// Cognito, Issuers holds the single user-pool issuer.
	Issuers   []string
	Audiences []string
	// Cognito access tokens are not audience-bound: the client binding is the
	// "client_id" claim checked against AcceptedClientIDs (web + IDE + M2M ids).
	// M2MAcceptAny is the "*" wildcard, honored for machine tokens only.
	AcceptedClientIDs []string
	M2MAcceptAny      bool
	FallbackURL       string
	JWKSRefreshSec    int
	ScopeTTLSec       int
	AuthMethod        string
	MarkerSecret      string
	FastPathReady     bool
}

// knownWeakSecrets are literals that must never be accepted as a signing key.
var knownWeakSecrets = map[string]bool{
	"secret":          true,
	"changeme":        true,
	"change-me":       true,
	"password":        true,
	"your-secret-key": true,
	"your_secret_key": true,
	"test":            true,
	"dev":             true,
	"mcp-secret-key":  true,
	"default":         true,
}

// validateSecretKey enforces the signing-secret invariant: reject missing AND weak
// keys (weak-check before length). Returns an error string when invalid.
func validateSecretKey(key string) string {
	stripped := strings.TrimSpace(key)
	if stripped == "" {
		return "SECRET_KEY is empty or whitespace"
	}
	if knownWeakSecrets[strings.ToLower(stripped)] {
		return "SECRET_KEY is a known-weak literal"
	}
	if len(stripped) < 32 {
		return "SECRET_KEY is too short (need >= 32 stripped chars)"
	}
	return ""
}

// getenv returns the env value or a default.
func getenv(key, def string) string {
	if v := os.Getenv(key); v != "" {
		return v
	}
	return def
}

// parseList splits a comma/whitespace-separated env value into a trimmed,
// de-duplicated, non-empty slice (order preserved). Lets VALIDATE_ISSUER /
// VALIDATE_AUDIENCE carry more than one value.
func parseList(s string) []string {
	fields := strings.FieldsFunc(s, func(r rune) bool {
		return r == ',' || r == ' ' || r == '\t' || r == '\n' || r == '\r'
	})
	seen := map[string]bool{}
	out := []string{}
	for _, f := range fields {
		if f == "" || seen[f] {
			continue
		}
		seen[f] = true
		out = append(out, f)
	}
	return out
}

// deriveIssuers builds the same three issuer URLs the Python Keycloak provider
// accepts (external, internal, localhost), so a token minted against any of
// those hosts fast-paths instead of falling back. Empty bases are skipped.
func deriveIssuers(realm string) []string {
	out := []string{}
	add := func(base string) {
		base = strings.TrimRight(base, "/")
		if base == "" {
			return
		}
		iss := fmt.Sprintf("%s/realms/%s", base, realm)
		for _, e := range out {
			if e == iss {
				return
			}
		}
		out = append(out, iss)
	}
	add(os.Getenv("KEYCLOAK_EXTERNAL_URL")) // browser logins
	add(os.Getenv("KEYCLOAK_URL"))          // internal service-to-service
	add("http://localhost:8080")            // local dev / host-minted tokens
	return out
}

// deriveAudiences builds the set of gateway-identifying audiences Python accepts
// (web client id, M2M client id, "mcp-gateway"). "account" is deliberately
// excluded: it rides on EVERY realm token, so accepting it would let a token
// minted for a different client in the same realm be replayed against the
// gateway (a same-realm cross-client confused-deputy).
func deriveAudiences() []string {
	out := []string{}
	add := func(a string) {
		a = strings.TrimSpace(a)
		if a == "" || a == "account" {
			return
		}
		for _, e := range out {
			if e == a {
				return
			}
		}
		out = append(out, a)
	}
	add(os.Getenv("KEYCLOAK_CLIENT_ID"))
	add(os.Getenv("KEYCLOAK_M2M_CLIENT_ID"))
	add("mcp-gateway")
	return out
}

// dropAccount removes "account" from an operator-supplied audience list and logs
// why, so copying the old VALIDATE_AUDIENCE=account value cannot silently reopen
// the cross-client confused-deputy that Python fails closed on.
func dropAccount(in []string) []string {
	out := []string{}
	for _, a := range in {
		if a == "account" {
			log.Printf("ignoring VALIDATE_AUDIENCE entry \"account\": it rides on every realm token; accepting it is a cross-client confused-deputy (matches Python, which rejects it)")
			continue
		}
		out = append(out, a)
	}
	return out
}

// detectProvider resolves which verifier to run. An explicit AUTH_PROVIDER wins
// (the auth-server already sets it); otherwise infer from which IdP env is present.
// Only "cognito" and "keycloak" are fast-pathed; anything else stays fallback-only.
func detectProvider() string {
	p := strings.ToLower(strings.TrimSpace(os.Getenv("AUTH_PROVIDER")))
	if p == "cognito" || p == "keycloak" {
		return p
	}
	if os.Getenv("COGNITO_USER_POOL_ID") != "" {
		return "cognito"
	}
	if os.Getenv("KEYCLOAK_URL") != "" {
		return "keycloak"
	}
	return p // e.g. "entra"/"okta"/"default"/"" -> keycloak-style derivation yields nothing -> fallback-only
}

// deriveCognito fills the Cognito issuer, JWKS URL and accepted client-id
// allowlist from the COGNITO_* / AWS_REGION env the auth-server already has,
// mirroring auth_server/providers/cognito.py. Explicit VALIDATE_ISSUER / JWKS_URL
// still win. Returns nothing derivable when COGNITO_USER_POOL_ID is unset.
func deriveCognito(cfg *Config) {
	region := getenv("AWS_REGION", "us-east-1")
	pool := strings.TrimSpace(os.Getenv("COGNITO_USER_POOL_ID"))
	if pool != "" {
		issuer := fmt.Sprintf("https://cognito-idp.%s.amazonaws.com/%s", region, pool)
		if len(cfg.Issuers) == 0 {
			cfg.Issuers = []string{issuer}
		}
		if cfg.JWKSURL == "" {
			cfg.JWKSURL = issuer + "/.well-known/jwks.json"
		}
	}

	// accepted_client_ids = web client + IDE client + explicit M2M ids; "*" = M2M
	// wildcard (machine tokens only).
	ids := []string{}
	add := func(v string) {
		v = strings.TrimSpace(v)
		if v == "" {
			return
		}
		for _, e := range ids {
			if e == v {
				return
			}
		}
		ids = append(ids, v)
	}
	add(os.Getenv("COGNITO_CLIENT_ID"))
	add(os.Getenv("IDE_OAUTH_CLIENT_ID"))
	for _, m := range parseList(os.Getenv("COGNITO_M2M_CLIENT_IDS")) {
		if m == "*" {
			cfg.M2MAcceptAny = true
			continue
		}
		add(m)
	}
	cfg.AcceptedClientIDs = ids
}

// loadConfig reads configuration from the environment and validates the signing key.
// It exits the process (fail closed) when SECRET_KEY is present but weak/invalid.
func loadConfig() Config {
	cfg := Config{
		Listen:         getenv("GOVALIDATE_LISTEN", ":8899"),
		SecretKey:      os.Getenv("SECRET_KEY"),
		JWKSURL:        os.Getenv("JWKS_URL"),
		FallbackURL:    getenv("AUTH_FALLBACK_URL", "http://auth-server:8888"),
		JWKSRefreshSec: atoiDefault(os.Getenv("JWKS_REFRESH_SECONDS"), 300),
		ScopeTTLSec:    atoiDefault(os.Getenv("SCOPE_SNAPSHOT_TTL_SECONDS"), 60),
		AuthMethod:     getenv("VALIDATE_AUTH_METHOD", "keycloak"),
		MarkerSecret:   os.Getenv("AUTH_SERVER_NGINX_MARKER_SECRET"),
	}

	cfg.Provider = detectProvider()

	// An explicit VALIDATE_ISSUER / JWKS_URL always wins for either provider.
	cfg.Issuers = parseList(os.Getenv("VALIDATE_ISSUER"))

	if cfg.Provider == "cognito" {
		// Cognito: derive issuer + JWKS + accepted client-id allowlist from the
		// COGNITO_* env. Access tokens are client_id-bound (no aud), so there is
		// no audience list; scopes come from cognito:groups or the scope claim.
		deriveCognito(&cfg)
	} else {
		// Keycloak (default): auto-derive JWKS_URL and the accepted issuer list
		// (external/internal/localhost realm URLs) + audience list from KEYCLOAK_*,
		// so operators only opt in. Explicit values win. A token matches when its
		// iss equals ANY issuer and its aud contains ANY audience.
		realm := getenv("KEYCLOAK_REALM", "mcp-gateway")
		if cfg.JWKSURL == "" {
			if kc := strings.TrimRight(os.Getenv("KEYCLOAK_URL"), "/"); kc != "" {
				cfg.JWKSURL = fmt.Sprintf("%s/realms/%s/protocol/openid-connect/certs", kc, realm)
			}
		}
		if len(cfg.Issuers) == 0 {
			cfg.Issuers = deriveIssuers(realm)
		}
		// "account" is stripped (fail closed) even if supplied explicitly.
		cfg.Audiences = dropAccount(parseList(os.Getenv("VALIDATE_AUDIENCE")))
		if len(cfg.Audiences) == 0 {
			cfg.Audiences = deriveAudiences()
		}
	}

	// B3: validate the signing secret. If a secret is provided at all it must be
	// strong; a weak secret is a hard failure (never fall open to a bad key).
	if cfg.SecretKey != "" {
		if msg := validateSecretKey(cfg.SecretKey); msg != "" {
			log.Fatalf("startup refused: %s", msg)
		}
	}

	// Fast path requires everything needed to verify AND mint safely. The
	// per-provider audience check differs: Keycloak needs an audience allowlist;
	// Cognito needs the client-id allowlist (or the M2M wildcard).
	base := cfg.SecretKey != "" && cfg.JWKSURL != "" && len(cfg.Issuers) > 0
	if cfg.Provider == "cognito" {
		cfg.FastPathReady = base && (len(cfg.AcceptedClientIDs) > 0 || cfg.M2MAcceptAny)
	} else {
		cfg.FastPathReady = base && len(cfg.Audiences) > 0
	}

	return cfg
}
