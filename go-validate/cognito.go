package main

// Cognito fast path. Mirrors auth_server/providers/cognito.py exactly:
//
//   - Issuer is https://cognito-idp.<region>.amazonaws.com/<user_pool_id>; JWKS
//     lives at <issuer>/.well-known/jwks.json.
//   - Cognito issues ACCESS tokens (token_use="access", NO aud claim, client in
//     the "client_id" claim) and ID tokens (token_use="id", aud=client_id). MCP
//     clients send the ACCESS token to the resource server, so the fast path
//     verifies access tokens and defers id/login tokens to Python.
//   - Access tokens are not audience-bound, so the client binding is the
//     "client_id" claim checked against an allowlist (web client + IDE client +
//     M2M client ids). "*" is an M2M-only wildcard: accept any client_id, but
//     ONLY for machine tokens (no "username" claim), so it can never widen which
//     clients may mint a USER token.
//   - Scopes: a USER access token carries cognito:groups -> group->scope mapping
//     (same DocumentDB path Keycloak uses). A machine / no-group token carries no
//     groups; its authorization is the token's own "scope" claim (Cognito
//     resource-server scopes = registry scope names). This matches server.py's
//     validate scope finalization.

// verifyCognito verifies a Cognito RS256 access token and applies the client_id
// allowlist policy. Sentinel errors preserve the fail-closed boundary: a
// recognized-invalid token (bad sig / expiry) -> 401; anything else (id token,
// unknown kid, wrong issuer, client_id not allowed) -> fall back to Python.
func verifyCognito(
	token string,
	ks *keysetCache,
	issuer string,
	acceptedClientIDs []string,
	m2mAcceptAny bool,
) (*Claims, error) {
	c, err := parseVerifyDecode(token, ks)
	if err != nil {
		return nil, err
	}
	if c.Iss != issuer {
		return nil, errUnknownKey // different issuer -> let Python handle it
	}
	// Only access tokens flow through the resource server; id/login tokens are
	// audience-bound and handled by Python. Treat non-access as "not ours".
	if c.TokenUse != "access" {
		return nil, errNotJWT
	}
	// A machine (client_credentials) token has no end-user "username" claim. The
	// "*" wildcard accepts any client_id but ONLY for such machine tokens.
	isMachine := c.CognitoUsername == ""
	if !containsStr(acceptedClientIDs, c.ClientID) && !(m2mAcceptAny && isMachine) {
		// client_id not in the allowlist is a policy decision, not proof of
		// forgery -> defer to Python (authoritative), never a 401.
		return nil, errUnknownKey
	}
	return c, nil
}

// mapCognitoClaims turns verified Cognito claims into a caller identity. Cognito
// carries the end-user handle in "username" (not preferred_username); fall back
// to sub for machine tokens.
func mapCognitoClaims(c *Claims) identity {
	username := c.CognitoUsername
	if username == "" {
		username = c.Sub
	}
	return identity{
		Sub:      c.Sub,
		Username: username,
		ClientID: c.ClientID,
	}
}
