package main

import (
	"crypto"
	"crypto/hmac"
	"crypto/rsa"
	"crypto/sha256"
	"encoding/base64"
	"encoding/json"
	"errors"
	"strconv"
	"strings"
	"time"
)

// Sentinel errors let the handler decide: fall back to Python (unrecognized) vs
// return 401 (recognized but invalid). This encodes the fail-closed boundary.
var (
	errNotJWT         = errors.New("not a JWT")             // -> fallback
	errUnknownKey     = errors.New("unknown kid/issuer")    // -> fallback
	errInvalidToken   = errors.New("invalid token")         // -> 401
	errWrongAlg       = errors.New("unexpected alg")        // -> fallback (could be HS/none from elsewhere)
)

const clockLeewaySeconds = 30

// atoiDefault parses an int, returning def on failure/empty.
func atoiDefault(s string, def int) int {
	if s == "" {
		return def
	}
	n, err := strconv.Atoi(s)
	if err != nil {
		return def
	}
	return n
}

// b64urlDecode decodes a base64url segment (no padding).
func b64urlDecode(seg string) ([]byte, error) {
	return base64.RawURLEncoding.DecodeString(seg)
}

// b64urlEncode encodes to base64url (no padding).
func b64urlEncode(b []byte) string {
	return base64.RawURLEncoding.EncodeToString(b)
}

type jwtHeader struct {
	Alg string `json:"alg"`
	Kid string `json:"kid"`
	Typ string `json:"typ"`
}

// Claims is the subset of RS256 claims the fast path reads. Groups and scope are
// decoded permissively because IdPs vary in shape.
type Claims struct {
	Iss      string          `json:"iss"`
	Aud      json.RawMessage `json:"aud"`
	Exp      int64           `json:"exp"`
	Sub      string          `json:"sub"`
	Username string          `json:"preferred_username"`
	Azp      string          `json:"azp"`
	ClientID string          `json:"client_id"`
	Scope    string          `json:"scope"`
	Groups   []string        `json:"groups"`
	raw      map[string]any
}

// audContains reports whether the token audience matches want (aud may be a string
// or an array of strings per RFC 7519).
func (c *Claims) audContains(want string) bool {
	var single string
	if err := json.Unmarshal(c.Aud, &single); err == nil {
		return single == want
	}
	var many []string
	if err := json.Unmarshal(c.Aud, &many); err == nil {
		for _, a := range many {
			if a == want {
				return true
			}
		}
	}
	return false
}

// verifyRS256 verifies an RS256 JWT against the cached keyset and enforces
// iss/aud/exp from config (never from the token). It returns the parsed claims on
// success, or a sentinel error telling the caller whether to fall back or 401.
func verifyRS256(token string, ks *keysetCache, issuer, audience string) (*Claims, error) {
	parts := strings.Split(token, ".")
	if len(parts) != 3 {
		return nil, errNotJWT
	}
	headerBytes, err := b64urlDecode(parts[0])
	if err != nil {
		return nil, errNotJWT
	}
	var h jwtHeader
	if err := json.Unmarshal(headerBytes, &h); err != nil {
		return nil, errNotJWT
	}
	if h.Alg != "RS256" {
		return nil, errWrongAlg
	}
	pub := ks.key(h.Kid)
	if pub == nil {
		return nil, errUnknownKey
	}

	// Verify signature over "header.payload".
	signingInput := parts[0] + "." + parts[1]
	sig, err := b64urlDecode(parts[2])
	if err != nil {
		return nil, errInvalidToken
	}
	digest := sha256.Sum256([]byte(signingInput))
	if err := rsa.VerifyPKCS1v15(pub, crypto.SHA256, digest[:], sig); err != nil {
		return nil, errInvalidToken
	}

	// Decode claims.
	payloadBytes, err := b64urlDecode(parts[1])
	if err != nil {
		return nil, errInvalidToken
	}
	var c Claims
	if err := json.Unmarshal(payloadBytes, &c); err != nil {
		return nil, errInvalidToken
	}
	_ = json.Unmarshal(payloadBytes, &c.raw)

	// Enforce iss/aud/exp (config-driven, fail closed).
	if c.Iss != issuer {
		return nil, errUnknownKey // different issuer -> let Python handle it
	}
	if !c.audContains(audience) {
		return nil, errInvalidToken
	}
	now := time.Now().Unix()
	if c.Exp != 0 && now > c.Exp+clockLeewaySeconds {
		return nil, errInvalidToken
	}
	return &c, nil
}

// mintInternalToken produces the HS256 X-Internal-Token-Registry that downstream
// services expect, signed with the shared SECRET_KEY. TTL is 30s to match Python.
func mintInternalToken(ident identity, audience, secret string) (string, error) {
	header := map[string]string{"alg": "HS256", "typ": "JWT"}
	now := time.Now().Unix()
	claims := map[string]any{
		"iss":       "mcp-auth-server",
		"aud":       audience,
		"sub":       ident.Sub,
		"username":  ident.Username,
		"client_id": ident.ClientID,
		"scopes":    ident.Scopes,
		"token_use": "internal",
		"iat":       now,
		"exp":       now + 30,
	}
	hb, err := json.Marshal(header)
	if err != nil {
		return "", err
	}
	cb, err := json.Marshal(claims)
	if err != nil {
		return "", err
	}
	signingInput := b64urlEncode(hb) + "." + b64urlEncode(cb)
	mac := hmac.New(sha256.New, []byte(secret))
	mac.Write([]byte(signingInput))
	sig := b64urlEncode(mac.Sum(nil))
	return signingInput + "." + sig, nil
}
