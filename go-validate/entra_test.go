package main

import (
	"crypto/rand"
	"crypto/rsa"
	"testing"
	"time"
)

const (
	eIssV2 = "https://login.microsoftonline.com/tenant-123/v2.0"
	eIssV1 = "https://sts.windows.net/tenant-123/"
	eAud   = "client-abc"
)

var eIssuers = []string{eIssV2, eIssV1}
var eAuds = []string{eAud, "api://client-abc"}

func entraClaims() map[string]any {
	return map[string]any{
		"iss": eIssV2, "aud": eAud, "exp": time.Now().Add(time.Hour).Unix(),
		"sub": "u-1", "preferred_username": "alice@corp", "azp": "client-abc",
		"groups": []string{"admins"}, "scope": "User.Read",
	}
}

func TestVerifyEntra(t *testing.T) {
	priv, _ := rsa.GenerateKey(rand.Reader, 2048)
	ks := testKeyset(&priv.PublicKey, "kid1")

	t.Run("valid v2 access token", func(t *testing.T) {
		tok := mintRS256(t, priv, "kid1", entraClaims())
		c, err := verifyEntra(tok, ks, eIssuers, eAuds)
		if err != nil || c.Username != "alice@corp" {
			t.Fatalf("valid entra token failed: err=%v", err)
		}
	})
	t.Run("v1 issuer accepted", func(t *testing.T) {
		cl := entraClaims()
		cl["iss"] = eIssV1
		tok := mintRS256(t, priv, "kid1", cl)
		if _, err := verifyEntra(tok, ks, eIssuers, eAuds); err != nil {
			t.Fatalf("v1 issuer should be accepted: %v", err)
		}
	})
	t.Run("api:// audience accepted", func(t *testing.T) {
		cl := entraClaims()
		cl["aud"] = "api://client-abc"
		tok := mintRS256(t, priv, "kid1", cl)
		if _, err := verifyEntra(tok, ks, eIssuers, eAuds); err != nil {
			t.Fatalf("api:// audience should be accepted: %v", err)
		}
	})
	t.Run("id_token (nonce) -> fallback", func(t *testing.T) {
		cl := entraClaims()
		cl["nonce"] = "abc123" // id_token-only claim
		tok := mintRS256(t, priv, "kid1", cl)
		if _, err := verifyEntra(tok, ks, eIssuers, eAuds); err != errUnknownKey {
			t.Fatalf("id_token must be deferred to Python (errUnknownKey), got %v", err)
		}
	})
	t.Run("wrong audience -> fallback", func(t *testing.T) {
		cl := entraClaims()
		cl["aud"] = "some-other-app"
		tok := mintRS256(t, priv, "kid1", cl)
		if _, err := verifyEntra(tok, ks, eIssuers, eAuds); err != errUnknownKey {
			t.Fatalf("unlisted audience must fall back, got %v", err)
		}
	})
}

func TestEntraGroups_RolesFallbackForM2M(t *testing.T) {
	// User token: groups claim wins.
	u := &Claims{Groups: []string{"g1"}, Roles: []string{"r1"}}
	if got := entraGroups(u); len(got) != 1 || got[0] != "g1" {
		t.Fatalf("user token should use groups, got %v", got)
	}
	// M2M token: no groups -> roles are the membership.
	m := &Claims{Roles: []string{"App.Admin"}}
	if got := entraGroups(m); len(got) != 1 || got[0] != "App.Admin" {
		t.Fatalf("M2M token should use roles, got %v", got)
	}
}

func TestMapEntraClaims(t *testing.T) {
	c := &Claims{Sub: "s-1", Username: "bob@corp", Azp: "cli-x"}
	id := mapEntraClaims(c, "fallback-client")
	if id.Username != "bob@corp" || id.ClientID != "cli-x" {
		t.Fatalf("map wrong: %+v", id)
	}
	// azp absent -> fall back to configured client id.
	c2 := &Claims{Sub: "s-2"}
	id2 := mapEntraClaims(c2, "fallback-client")
	if id2.Username != "s-2" || id2.ClientID != "fallback-client" {
		t.Fatalf("fallback map wrong: %+v", id2)
	}
}
