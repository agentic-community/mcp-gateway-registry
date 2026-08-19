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
	errNotJWT       = errors.New("not a JWT")          // -> fallback
	errUnknownKey   = errors.New("unknown kid/issuer") // -> fallback
	errInvalidToken = errors.New("invalid token")      // -> 401
	errWrongAlg     = errors.New("unexpected alg")     // -> fallback (could be HS/none from elsewhere)
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
	// Cognito-specific claim shapes (unused by other providers, decoded permissively).
	CognitoGroups   []string `json:"cognito:groups"`
	TokenUse        string   `json:"token_use"`
	CognitoUsername string   `json:"username"`
	raw             map[string]any
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

// audMatchesAny reports whether the token audience contains ANY accepted value.
func (c *Claims) audMatchesAny(accepted []string) bool {
	for _, a := range accepted {
		if c.audContains(a) {
			return true
		}
	}
	return false
}

// containsStr reports whether want is in list.
func containsStr(list []string, want string) bool {
	for _, s := range list {
		if s == want {
			return true
		}
	}
	return false
}

// verifyRS256 verifies an RS256 JWT against the cached keyset and enforces
// iss/aud/exp from config (never from the token). It returns the parsed claims on
// success, or a sentinel error telling the caller whether to fall back or 401.
// parseVerifyDecode does the IdP-agnostic half of RS256 verification: structural
// parse, alg=RS256, key lookup by kid, signature check, claim decode, and expiry.
// Provider-specific iss/aud/client_id policy is layered on by the callers
// (verifyRS256 for Keycloak, verifyCognito for Cognito). The sentinel errors keep
// the fail-closed boundary: fallback vs 401.
func parseVerifyDecode(token string, ks *keysetCache) (*Claims, error) {
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

	now := time.Now().Unix()
	if c.Exp != 0 && now > c.Exp+clockLeewaySeconds {
		return nil, errInvalidToken
	}
	return &c, nil
}

func verifyRS256(token string, ks *keysetCache, issuers, audiences []string) (*Claims, error) {
	c, err := parseVerifyDecode(token, ks)
	if err != nil {
		return nil, err
	}
	// Enforce iss/aud (config-driven, fail closed). iss must match ANY accepted
	// issuer; aud must contain ANY accepted audience (mirrors the Python Keycloak
	// provider's valid_issuers / accepted_audiences lists).
	if !containsStr(issuers, c.Iss) {
		return nil, errUnknownKey // different issuer -> let Python handle it
	}
	if !c.audMatchesAny(audiences) {
		// Audience is a policy decision, not proof of forgery. Defer to Python
		// (authoritative) rather than 401 so we never reject a token the full
		// handler would have accepted. Only a bad signature or expiry -> 401.
		return nil, errUnknownKey
	}
	return c, nil
}

// Internal-token issuer + audiences (mirror auth_server/internal_request_token.py).
const (
	internalIssuer         = "mcp-auth-server"
	mcpProxyAudience       = "mcp-proxy"
	mcpProxyTokenUse       = "mcp-proxy"
	mcpRegistryUIAudience  = "mcp-registry-ui"
	mcpRegistryUITokenUse  = "mcp-registry-ui"
	internalTokenTTLSecond = 30
)

// mintInternal signs an HS256 internal JWT with the shared SECRET_KEY. It refuses
// an empty subject (fail closed), exactly like _mint_internal_token: an anonymous
// but valid token must never be issued.
func mintInternal(
	secret, audience, subject string,
	scopes []string,
	extra map[string]any,
) (string, error) {
	if subject == "" {
		return "", errors.New("cannot mint internal token with empty subject")
	}
	if scopes == nil {
		scopes = []string{}
	}
	now := time.Now().Unix()
	claims := map[string]any{
		"iss":    internalIssuer,
		"aud":    audience,
		"sub":    subject,
		"scopes": scopes,
		"iat":    now,
		"exp":    now + internalTokenTTLSecond,
	}
	for k, v := range extra {
		claims[k] = v
	}
	header := map[string]string{"alg": "HS256", "typ": "JWT"}
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
	return signingInput + "." + b64urlEncode(mac.Sum(nil)), nil
}

// mintRegistryUIToken mirrors mint_registry_ui_token: a thin identity assertion
// for the registry /api/ hop (no scopes encoded; registry derives them).
func mintRegistryUIToken(
	secret, subject, sessionID string,
	groups []string,
	authMethod, clientID, egressUser string,
) (string, error) {
	if groups == nil {
		groups = []string{}
	}
	return mintInternal(secret, mcpRegistryUIAudience, subject, []string{}, map[string]any{
		"session_id":  sessionID,
		"groups":      groups,
		"auth_method": authMethod,
		"client_id":   clientID,
		"egress_user": egressUser,
		"token_use":   mcpRegistryUITokenUse,
	})
}

// mintMCPProxyToken mirrors mint_mcp_proxy_token: binds scopes + resolved upstream
// for the /mcp-proxy hop. server is the first path segment (traversal guard).
func mintMCPProxyToken(
	secret, subject string,
	scopes []string,
	serverName, upstreamURL, authMethod, egressUser string,
) (string, error) {
	server := serverName
	if i := indexByte(server, '/'); i >= 0 {
		server = server[:i]
	}
	return mintInternal(secret, mcpProxyAudience, subject, scopes, map[string]any{
		"server":       server,
		"upstream_url": upstreamURL,
		"auth_method":  authMethod,
		"egress_user":  egressUser,
		"token_use":    mcpProxyTokenUse,
	})
}

// indexByte returns the index of the first b in s, or -1.
func indexByte(s string, b byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == b {
			return i
		}
	}
	return -1
}
