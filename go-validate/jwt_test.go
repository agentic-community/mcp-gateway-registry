package main

import (
	"crypto"
	"crypto/hmac"
	"crypto/rand"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/binary"
	"encoding/json"
	"math/big"
	"testing"
	"time"
)

// --- helpers -------------------------------------------------------------

// mintRS256 signs a test RS256 JWT with priv, header kid, and the given claims.
func mintRS256(t *testing.T, priv *rsa.PrivateKey, kid string, claims map[string]any) string {
	t.Helper()
	hdr := map[string]string{"alg": "RS256", "typ": "JWT", "kid": kid}
	hb, _ := json.Marshal(hdr)
	cb, _ := json.Marshal(claims)
	signingInput := b64urlEncode(hb) + "." + b64urlEncode(cb)
	digest := sha256.Sum256([]byte(signingInput))
	sig, err := rsa.SignPKCS1v15(rand.Reader, priv, crypto.SHA256, digest[:])
	if err != nil {
		t.Fatalf("sign: %v", err)
	}
	return signingInput + "." + b64urlEncode(sig)
}

// testKeyset builds a keysetCache holding one public key under kid.
func testKeyset(pub *rsa.PublicKey, kid string) *keysetCache {
	ks := &keysetCache{}
	m := map[string]*rsa.PublicKey{kid: pub}
	ks.keys.Store(&m)
	return ks
}

const (
	tIss = "https://kc/realms/mcp-gateway"
	tAud = "account"
)

func baseClaims() map[string]any {
	return map[string]any{
		"iss": tIss, "aud": tAud, "exp": time.Now().Add(time.Hour).Unix(),
		"sub": "svc-sub", "preferred_username": "svc", "azp": "svc-client",
		"groups": []string{"admins"}, "scope": "profile email",
	}
}

// --- verifyRS256 --------------------------------------------------------

func TestVerifyRS256(t *testing.T) {
	priv, _ := rsa.GenerateKey(rand.Reader, 2048)
	ks := testKeyset(&priv.PublicKey, "kid1")

	t.Run("valid", func(t *testing.T) {
		tok := mintRS256(t, priv, "kid1", baseClaims())
		c, err := verifyRS256(tok, ks, []string{tIss}, []string{tAud})
		if err != nil || c == nil || c.Username != "svc" || c.Azp != "svc-client" {
			t.Fatalf("valid token failed: err=%v claims=%+v", err, c)
		}
	})
	t.Run("bad signature -> 401 (errInvalidToken)", func(t *testing.T) {
		tok := mintRS256(t, priv, "kid1", baseClaims())
		if _, err := verifyRS256(tok[:len(tok)-2]+"xx", ks, []string{tIss}, []string{tAud}); err != errInvalidToken {
			t.Fatalf("want errInvalidToken, got %v", err)
		}
	})
	t.Run("expired -> 401", func(t *testing.T) {
		c := baseClaims()
		c["exp"] = time.Now().Add(-time.Hour).Unix()
		tok := mintRS256(t, priv, "kid1", c)
		if _, err := verifyRS256(tok, ks, []string{tIss}, []string{tAud}); err != errInvalidToken {
			t.Fatalf("want errInvalidToken (expired), got %v", err)
		}
	})
	t.Run("unknown kid -> fallback (errUnknownKey)", func(t *testing.T) {
		tok := mintRS256(t, priv, "otherkid", baseClaims())
		if _, err := verifyRS256(tok, ks, []string{tIss}, []string{tAud}); err != errUnknownKey {
			t.Fatalf("want errUnknownKey, got %v", err)
		}
	})
	t.Run("wrong issuer -> fallback", func(t *testing.T) {
		tok := mintRS256(t, priv, "kid1", baseClaims())
		if _, err := verifyRS256(tok, ks, []string{"https://other"}, []string{tAud}); err != errUnknownKey {
			t.Fatalf("want errUnknownKey (iss), got %v", err)
		}
	})
	t.Run("wrong audience -> fallback (never 401)", func(t *testing.T) {
		tok := mintRS256(t, priv, "kid1", baseClaims())
		if _, err := verifyRS256(tok, ks, []string{tIss}, []string{"different-aud"}); err != errUnknownKey {
			t.Fatalf("want errUnknownKey (aud), got %v", err)
		}
	})
	t.Run("not a JWT -> fallback", func(t *testing.T) {
		if _, err := verifyRS256("not.a", ks, []string{tIss}, []string{tAud}); err != errNotJWT {
			t.Fatalf("want errNotJWT, got %v", err)
		}
	})
	t.Run("iss matches ANY issuer in the list", func(t *testing.T) {
		// token minted with tIss; tIss is the SECOND accepted issuer.
		tok := mintRS256(t, priv, "kid1", baseClaims())
		issuers := []string{"https://external/realms/mcp-gateway", tIss}
		if _, err := verifyRS256(tok, ks, issuers, []string{tAud}); err != nil {
			t.Fatalf("token iss should match a non-first list member: %v", err)
		}
	})
	t.Run("aud matches ANY audience in the list", func(t *testing.T) {
		c := baseClaims()
		c["aud"] = []string{"mcp-gateway", "account"} // realm token shape
		tok := mintRS256(t, priv, "kid1", c)
		// Accept mcp-gateway (parity), NOT account -> must still verify on mcp-gateway.
		if _, err := verifyRS256(tok, ks, []string{tIss}, []string{"mcp-gateway"}); err != nil {
			t.Fatalf("aud list member should match: %v", err)
		}
		// A token carrying ONLY account must NOT verify when account is not accepted.
		c2 := baseClaims()
		c2["aud"] = "account"
		tok2 := mintRS256(t, priv, "kid1", c2)
		if _, err := verifyRS256(tok2, ks, []string{tIss}, []string{"mcp-gateway"}); err != errUnknownKey {
			t.Fatalf("account-only token must fall back when account not accepted, got %v", err)
		}
	})
}

// --- audContains --------------------------------------------------------

func TestAudContains(t *testing.T) {
	str := &Claims{Aud: json.RawMessage(`"account"`)}
	if !str.audContains("account") || str.audContains("nope") {
		t.Error("string aud match failed")
	}
	arr := &Claims{Aud: json.RawMessage(`["mcp-gateway","account"]`)}
	if !arr.audContains("account") || !arr.audContains("mcp-gateway") || arr.audContains("nope") {
		t.Error("array aud match failed")
	}
}

// --- mint round-trips ---------------------------------------------------

// decodeHS256 verifies the HMAC and returns the claims.
func decodeHS256(t *testing.T, tok, secret string) map[string]any {
	t.Helper()
	var h, p, sig string
	parts := 0
	last := 0
	for i := 0; i <= len(tok); i++ {
		if i == len(tok) || tok[i] == '.' {
			seg := tok[last:i]
			switch parts {
			case 0:
				h = seg
			case 1:
				p = seg
			case 2:
				sig = seg
			}
			parts++
			last = i + 1
		}
	}
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(h + "." + p))
	want := b64urlEncode(mac.Sum(nil))
	if !hmac.Equal([]byte(want), []byte(sig)) {
		t.Fatalf("HS256 signature mismatch")
	}
	pb, _ := base64.RawURLEncoding.DecodeString(p)
	var claims map[string]any
	if err := json.Unmarshal(pb, &claims); err != nil {
		t.Fatalf("decode claims: %v", err)
	}
	return claims
}

func TestMintRegistryUIToken(t *testing.T) {
	secret := "unit-test-secret-32-bytes-xxxxxxxxxx"
	tok, err := mintRegistryUIToken(secret, "svc-sub", "", []string{"admins"}, "oauth2", "svc-client", "svc-sub")
	if err != nil {
		t.Fatal(err)
	}
	c := decodeHS256(t, tok, secret)
	if c["iss"] != "mcp-auth-server" || c["aud"] != "mcp-registry-ui" || c["sub"] != "svc-sub" ||
		c["token_use"] != "mcp-registry-ui" || c["auth_method"] != "oauth2" || c["client_id"] != "svc-client" {
		t.Fatalf("registry-ui claims wrong: %+v", c)
	}
	if s, ok := c["scopes"].([]any); !ok || len(s) != 0 {
		t.Fatalf("registry-ui token must carry empty scopes, got %v", c["scopes"])
	}
}

func TestMintMCPProxyToken(t *testing.T) {
	secret := "unit-test-secret-32-bytes-xxxxxxxxxx"
	tok, err := mintMCPProxyToken(secret, "svc-sub", []string{"s1", "s2"}, "myserver/mcp", "http://up:9/mcp", "oauth2", "svc-sub")
	if err != nil {
		t.Fatal(err)
	}
	c := decodeHS256(t, tok, secret)
	if c["aud"] != "mcp-proxy" || c["token_use"] != "mcp-proxy" || c["server"] != "myserver" ||
		c["upstream_url"] != "http://up:9/mcp" {
		t.Fatalf("mcp-proxy claims wrong: %+v", c)
	}
	if s, ok := c["scopes"].([]any); !ok || len(s) != 2 {
		t.Fatalf("mcp-proxy token must carry scopes, got %v", c["scopes"])
	}
}

func TestMintInternal_EmptySubjectFailsClosed(t *testing.T) {
	if _, err := mintInternal("secret", "aud", "", nil, nil); err == nil {
		t.Fatal("empty subject must fail closed")
	}
}

// --- parseKey -----------------------------------------------------------

func TestParseKey(t *testing.T) {
	priv, _ := rsa.GenerateKey(rand.Reader, 2048)
	n := base64.RawURLEncoding.EncodeToString(priv.PublicKey.N.Bytes())
	eb := make([]byte, 4)
	binary.BigEndian.PutUint32(eb, uint32(priv.PublicKey.E))
	// trim leading zero bytes
	i := 0
	for i < len(eb)-1 && eb[i] == 0 {
		i++
	}
	e := base64.RawURLEncoding.EncodeToString(eb[i:])
	pub, err := parseKey(jwk{Kty: "RSA", Kid: "k", N: n, E: e})
	if err != nil {
		t.Fatal(err)
	}
	if pub.N.Cmp(priv.PublicKey.N) != 0 || pub.E != priv.PublicKey.E {
		t.Fatalf("parsed key mismatch: E got %d want %d", pub.E, priv.PublicKey.E)
	}
	_ = big.NewInt(0)
}
