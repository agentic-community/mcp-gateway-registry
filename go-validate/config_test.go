package main

import "testing"

func TestLoadConfig_DerivesJWKSandIssuerFromKeycloak(t *testing.T) {
	t.Setenv("SECRET_KEY", "unit-test-secret-32-bytes-xxxxxxxxxx")
	t.Setenv("KEYCLOAK_URL", "http://keycloak:8080/")
	t.Setenv("KEYCLOAK_EXTERNAL_URL", "https://kc.example.com")
	t.Setenv("KEYCLOAK_REALM", "mcp-gateway")
	t.Setenv("VALIDATE_AUDIENCE", "account")
	// JWKS_URL / VALIDATE_ISSUER intentionally unset -> must be derived
	t.Setenv("JWKS_URL", "")
	t.Setenv("VALIDATE_ISSUER", "")
	cfg := loadConfig()
	if cfg.JWKSURL != "http://keycloak:8080/realms/mcp-gateway/protocol/openid-connect/certs" {
		t.Fatalf("JWKS not derived: %q", cfg.JWKSURL)
	}
	if cfg.Issuer != "https://kc.example.com/realms/mcp-gateway" {
		t.Fatalf("issuer not derived from external url: %q", cfg.Issuer)
	}
	if !cfg.FastPathReady {
		t.Fatal("fast path should be ready after derivation + audience")
	}
}

func TestLoadConfig_ExplicitValuesWin(t *testing.T) {
	t.Setenv("SECRET_KEY", "unit-test-secret-32-bytes-xxxxxxxxxx")
	t.Setenv("KEYCLOAK_URL", "http://keycloak:8080")
	t.Setenv("JWKS_URL", "https://explicit/certs")
	t.Setenv("VALIDATE_ISSUER", "https://explicit/iss")
	t.Setenv("VALIDATE_AUDIENCE", "account")
	cfg := loadConfig()
	if cfg.JWKSURL != "https://explicit/certs" || cfg.Issuer != "https://explicit/iss" {
		t.Fatalf("explicit values must win: %q %q", cfg.JWKSURL, cfg.Issuer)
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
