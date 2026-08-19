package main

import "testing"

func TestLoadConfig_DerivesJWKSandIssuersFromKeycloak(t *testing.T) {
	t.Setenv("SECRET_KEY", "unit-test-secret-32-bytes-xxxxxxxxxx")
	t.Setenv("KEYCLOAK_URL", "http://keycloak:8080/")
	t.Setenv("KEYCLOAK_EXTERNAL_URL", "https://kc.example.com")
	t.Setenv("KEYCLOAK_REALM", "mcp-gateway")
	t.Setenv("KEYCLOAK_CLIENT_ID", "mcp-gateway-web")
	t.Setenv("KEYCLOAK_M2M_CLIENT_ID", "mcp-gateway-m2m")
	// JWKS_URL / VALIDATE_ISSUER / VALIDATE_AUDIENCE unset -> must all be derived
	t.Setenv("JWKS_URL", "")
	t.Setenv("VALIDATE_ISSUER", "")
	t.Setenv("VALIDATE_AUDIENCE", "")
	cfg := loadConfig()
	if cfg.JWKSURL != "http://keycloak:8080/realms/mcp-gateway/protocol/openid-connect/certs" {
		t.Fatalf("JWKS not derived: %q", cfg.JWKSURL)
	}
	// All three Python-parity issuers must be present (external, internal, localhost).
	for _, want := range []string{
		"https://kc.example.com/realms/mcp-gateway",
		"http://keycloak:8080/realms/mcp-gateway",
		"http://localhost:8080/realms/mcp-gateway",
	} {
		if !containsStr(cfg.Issuers, want) {
			t.Fatalf("derived issuers %v missing %q", cfg.Issuers, want)
		}
	}
	// Audiences derived from the client ids + "mcp-gateway"; never "account".
	for _, want := range []string{"mcp-gateway-web", "mcp-gateway-m2m", "mcp-gateway"} {
		if !containsStr(cfg.Audiences, want) {
			t.Fatalf("derived audiences %v missing %q", cfg.Audiences, want)
		}
	}
	if containsStr(cfg.Audiences, "account") {
		t.Fatal("derived audiences must never include \"account\" (cross-client confused-deputy)")
	}
	if !cfg.FastPathReady {
		t.Fatal("fast path should be ready after derivation")
	}
}

func TestLoadConfig_ExplicitListsWin(t *testing.T) {
	t.Setenv("SECRET_KEY", "unit-test-secret-32-bytes-xxxxxxxxxx")
	t.Setenv("KEYCLOAK_URL", "http://keycloak:8080")
	t.Setenv("JWKS_URL", "https://explicit/certs")
	// Comma/space-separated lists with an interspersed "account" that must be dropped.
	t.Setenv("VALIDATE_ISSUER", "https://a/iss, https://b/iss")
	t.Setenv("VALIDATE_AUDIENCE", "account, my-aud")
	cfg := loadConfig()
	if cfg.JWKSURL != "https://explicit/certs" {
		t.Fatalf("explicit JWKS must win: %q", cfg.JWKSURL)
	}
	if len(cfg.Issuers) != 2 || cfg.Issuers[0] != "https://a/iss" || cfg.Issuers[1] != "https://b/iss" {
		t.Fatalf("explicit issuer list must win: %v", cfg.Issuers)
	}
	// "account" stripped even when set explicitly; only "my-aud" survives.
	if len(cfg.Audiences) != 1 || cfg.Audiences[0] != "my-aud" {
		t.Fatalf("account must be dropped from explicit audiences: %v", cfg.Audiences)
	}
}

func TestLoadConfig_NoKeycloak_FallbackOnly(t *testing.T) {
	t.Setenv("SECRET_KEY", "")
	t.Setenv("KEYCLOAK_URL", "")
	t.Setenv("KEYCLOAK_EXTERNAL_URL", "")
	t.Setenv("JWKS_URL", "")
	t.Setenv("VALIDATE_ISSUER", "")
	t.Setenv("VALIDATE_AUDIENCE", "")
	cfg := loadConfig()
	if cfg.FastPathReady {
		t.Fatal("no config -> fallback-only (FastPathReady must be false)")
	}
}
