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
	Listen         string
	SecretKey      string
	JWKSURL        string
	Issuer         string
	Audience       string
	FallbackURL    string
	JWKSRefreshSec int
	ScopeTTLSec    int
	AuthMethod     string
	MarkerSecret   string
	FastPathReady  bool
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

// loadConfig reads configuration from the environment and validates the signing key.
// It exits the process (fail closed) when SECRET_KEY is present but weak/invalid.
func loadConfig() Config {
	cfg := Config{
		Listen:         getenv("GOVALIDATE_LISTEN", ":8899"),
		SecretKey:      os.Getenv("SECRET_KEY"),
		JWKSURL:        os.Getenv("JWKS_URL"),
		Issuer:         os.Getenv("VALIDATE_ISSUER"),
		Audience:       os.Getenv("VALIDATE_AUDIENCE"),
		FallbackURL:    getenv("AUTH_FALLBACK_URL", "http://auth-server:8888"),
		JWKSRefreshSec: atoiDefault(os.Getenv("JWKS_REFRESH_SECONDS"), 300),
		ScopeTTLSec:    atoiDefault(os.Getenv("SCOPE_SNAPSHOT_TTL_SECONDS"), 60),
		AuthMethod:     getenv("VALIDATE_AUTH_METHOD", "keycloak"),
		MarkerSecret:   os.Getenv("AUTH_SERVER_NGINX_MARKER_SECRET"),
	}

	// Auto-derive JWKS_URL and VALIDATE_ISSUER from the KEYCLOAK_* env vars that
	// every deployment already provides, so operators only opt in (+ set the
	// audience) instead of hand-computing these. Explicit values always win.
	// The audience claim is NOT derivable (Keycloak varies it per client), so it
	// stays operator-supplied; an unset/mismatched audience fails safe (fallback).
	realm := getenv("KEYCLOAK_REALM", "mcp-gateway")
	if cfg.JWKSURL == "" {
		if kc := strings.TrimRight(os.Getenv("KEYCLOAK_URL"), "/"); kc != "" {
			cfg.JWKSURL = fmt.Sprintf("%s/realms/%s/protocol/openid-connect/certs", kc, realm)
		}
	}
	if cfg.Issuer == "" {
		// Tokens carry the issuer of the URL the client used, usually the external
		// Keycloak URL; fall back to the internal URL when no external is set.
		iss := strings.TrimRight(os.Getenv("KEYCLOAK_EXTERNAL_URL"), "/")
		if iss == "" {
			iss = strings.TrimRight(os.Getenv("KEYCLOAK_URL"), "/")
		}
		if iss != "" {
			cfg.Issuer = fmt.Sprintf("%s/realms/%s", iss, realm)
		}
	}

	// B3: validate the signing secret. If a secret is provided at all it must be
	// strong; a weak secret is a hard failure (never fall open to a bad key).
	if cfg.SecretKey != "" {
		if msg := validateSecretKey(cfg.SecretKey); msg != "" {
			log.Fatalf("startup refused: %s", msg)
		}
	}

	// Fast path requires everything needed to verify AND mint safely.
	cfg.FastPathReady = cfg.SecretKey != "" &&
		cfg.JWKSURL != "" &&
		cfg.Issuer != "" &&
		cfg.Audience != ""

	return cfg
}
