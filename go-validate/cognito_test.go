package main

import (
	"crypto/rand"
	"crypto/rsa"
	"testing"
	"time"
)

const cIss = "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_pool"

// cognitoAccessClaims builds a minimal Cognito ACCESS-token claim set.
func cognitoAccessClaims() map[string]any {
	return map[string]any{
		"iss":       cIss,
		"token_use": "access",
		"client_id": "web-client",
		"username":  "alice",
		"sub":       "sub-123",
		"scope":     "mcp-servers-unrestricted/read",
		"exp":       time.Now().Add(time.Hour).Unix(),
	}
}

func TestVerifyCognito(t *testing.T) {
	priv, _ := rsa.GenerateKey(rand.Reader, 2048)
	ks := testKeyset(&priv.PublicKey, "kid1")
	accepted := []string{"web-client", "ide-client"}

	t.Run("valid access token, client_id allowed", func(t *testing.T) {
		tok := mintRS256(t, priv, "kid1", cognitoAccessClaims())
		c, err := verifyCognito(tok, ks, cIss, accepted, false)
		if err != nil || c == nil || c.CognitoUsername != "alice" || c.ClientID != "web-client" {
			t.Fatalf("valid token failed: err=%v claims=%+v", err, c)
		}
	})
	t.Run("id token -> fallback (not access)", func(t *testing.T) {
		cl := cognitoAccessClaims()
		cl["token_use"] = "id"
		tok := mintRS256(t, priv, "kid1", cl)
		if _, err := verifyCognito(tok, ks, cIss, accepted, false); err != errNotJWT {
			t.Fatalf("id token must fall back (errNotJWT), got %v", err)
		}
	})
	t.Run("wrong issuer -> fallback", func(t *testing.T) {
		tok := mintRS256(t, priv, "kid1", cognitoAccessClaims())
		if _, err := verifyCognito(tok, ks, "https://other/pool", accepted, false); err != errUnknownKey {
			t.Fatalf("wrong issuer must fall back (errUnknownKey), got %v", err)
		}
	})
	t.Run("bad signature -> 401", func(t *testing.T) {
		tok := mintRS256(t, priv, "kid1", cognitoAccessClaims())
		if _, err := verifyCognito(tok[:len(tok)-2]+"xx", ks, cIss, accepted, false); err != errInvalidToken {
			t.Fatalf("bad sig must be errInvalidToken, got %v", err)
		}
	})
	t.Run("client_id not in allowlist -> fallback", func(t *testing.T) {
		cl := cognitoAccessClaims()
		cl["client_id"] = "rogue-client"
		tok := mintRS256(t, priv, "kid1", cl)
		if _, err := verifyCognito(tok, ks, cIss, accepted, false); err != errUnknownKey {
			t.Fatalf("unlisted client_id must fall back, got %v", err)
		}
	})
	t.Run("m2m wildcard accepts a MACHINE token with any client_id", func(t *testing.T) {
		cl := cognitoAccessClaims()
		cl["client_id"] = "some-agent-client"
		delete(cl, "username") // machine token: no username
		tok := mintRS256(t, priv, "kid1", cl)
		if _, err := verifyCognito(tok, ks, cIss, accepted, true); err != nil {
			t.Fatalf("m2m wildcard should accept a machine token: %v", err)
		}
	})
	t.Run("m2m wildcard does NOT widen USER tokens", func(t *testing.T) {
		cl := cognitoAccessClaims()
		cl["client_id"] = "some-agent-client"
		// username present -> a user token -> wildcard must not apply.
		tok := mintRS256(t, priv, "kid1", cl)
		if _, err := verifyCognito(tok, ks, cIss, accepted, true); err != errUnknownKey {
			t.Fatalf("wildcard must not accept a USER token with an unlisted client_id, got %v", err)
		}
	})
}

// TestResolveCognito_ScopeSources verifies the two scope paths: cognito:groups ->
// group->scope mapping, and no-group token -> the token's own scope claim.
func TestResolveCognito_ScopeSources(t *testing.T) {
	priv, _ := rsa.GenerateKey(rand.Reader, 2048)
	ks := testKeyset(&priv.PublicKey, "kid1")
	res := &scopeResolver{}
	res.snap.Store(&scopeSnapshot{
		scopes:    []scopeDoc{{ID: "mcp-servers-unrestricted/read", GroupMappings: []string{"admins"}}},
		m2mGroups: map[string][]string{},
	})
	s := &server{
		cfg: Config{
			Provider: "cognito", FastPathReady: true,
			Issuers: []string{cIss}, AcceptedClientIDs: []string{"web-client"},
		},
		ks:     ks,
		scopes: res,
	}

	t.Run("user token with cognito:groups -> group mapping", func(t *testing.T) {
		cl := cognitoAccessClaims()
		cl["cognito:groups"] = []string{"admins"}
		cl["scope"] = "ignored-when-groups-present"
		tok := mintRS256(t, priv, "kid1", cl)
		ident, groups, scopes, method, verdict := s.resolveCognito(tok)
		if verdict != vOK || method != "cognito" || ident.Username != "alice" {
			t.Fatalf("verdict=%d method=%q ident=%+v", verdict, method, ident)
		}
		if len(groups) != 1 || groups[0] != "admins" {
			t.Fatalf("groups wrong: %v", groups)
		}
		if len(scopes) != 1 || scopes[0] != "mcp-servers-unrestricted/read" {
			t.Fatalf("group->scope mapping wrong: %v", scopes)
		}
	})
	t.Run("machine token (no groups) -> token scope claim", func(t *testing.T) {
		cl := cognitoAccessClaims()
		delete(cl, "username")
		cl["scope"] = "mcp-servers-unrestricted/read mcp-servers-unrestricted/execute"
		tok := mintRS256(t, priv, "kid1", cl)
		ident, groups, scopes, _, verdict := s.resolveCognito(tok)
		if verdict != vOK {
			t.Fatalf("machine token should resolve OK, verdict=%d", verdict)
		}
		if len(groups) != 0 {
			t.Fatalf("machine token should carry no groups, got %v", groups)
		}
		if len(scopes) != 2 {
			t.Fatalf("scopes should come from the token claim: %v", scopes)
		}
		if ident.Username != "sub-123" { // no username -> falls back to sub
			t.Fatalf("machine identity should be sub, got %q", ident.Username)
		}
	})
}
