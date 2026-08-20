package main

import (
	"crypto/rand"
	"crypto/rsa"
	"encoding/json"
	"testing"
	"time"
)

const oIss = "https://dev-123.okta.com/oauth2/aus1"
const oAud = "okta-client-1"

func oktaClaims() map[string]any {
	return map[string]any{
		"iss": oIss, "aud": oAud, "exp": time.Now().Add(time.Hour).Unix(),
		"sub": "okta-user-1", "cid": "okta-client-1",
		"groups": []string{"admins"}, "scp": []string{"registry.read"},
	}
}

func TestVerifyOkta(t *testing.T) {
	priv, _ := rsa.GenerateKey(rand.Reader, 2048)
	ks := testKeyset(&priv.PublicKey, "kid1")
	issuers := []string{oIss}
	auds := []string{oAud}

	t.Run("valid access token", func(t *testing.T) {
		tok := mintRS256(t, priv, "kid1", oktaClaims())
		c, err := verifyRS256(tok, ks, issuers, auds)
		if err != nil || c.Cid != "okta-client-1" {
			t.Fatalf("valid okta token failed: err=%v", err)
		}
	})
	t.Run("wrong audience -> fallback", func(t *testing.T) {
		cl := oktaClaims()
		cl["aud"] = "other"
		tok := mintRS256(t, priv, "kid1", cl)
		if _, err := verifyRS256(tok, ks, issuers, auds); err != errUnknownKey {
			t.Fatalf("unlisted audience must fall back, got %v", err)
		}
	})
}

func TestMapOktaClaims(t *testing.T) {
	c := &Claims{Sub: "okta-user-1", Cid: "cli-1"}
	id := mapOktaClaims(c, "fallback")
	if id.Username != "okta-user-1" || id.ClientID != "cli-1" {
		t.Fatalf("okta map wrong: %+v", id)
	}
	// cid absent -> fall back to configured client id.
	c2 := &Claims{Sub: "u2"}
	if id2 := mapOktaClaims(c2, "fallback"); id2.ClientID != "fallback" {
		t.Fatalf("okta fallback client id wrong: %+v", id2)
	}
}

func TestScpOrScope(t *testing.T) {
	// scp as array
	c := &Claims{Scp: json.RawMessage(`["a","b"]`)}
	if got := c.scpOrScope(); len(got) != 2 || got[0] != "a" {
		t.Fatalf("scp array parse wrong: %v", got)
	}
	// scp as string
	c2 := &Claims{Scp: json.RawMessage(`"x y"`)}
	if got := c2.scpOrScope(); len(got) != 2 || got[1] != "y" {
		t.Fatalf("scp string parse wrong: %v", got)
	}
	// fall back to scope
	c3 := &Claims{Scope: "s1 s2 s3"}
	if got := c3.scpOrScope(); len(got) != 3 {
		t.Fatalf("scope fallback wrong: %v", got)
	}
}
