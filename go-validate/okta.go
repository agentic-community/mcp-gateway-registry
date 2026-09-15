package main

// Okta fast path. Mirrors auth_server/providers/okta.py:
//
//   - Org vs custom-authorization-server URLs: with OKTA_AUTH_SERVER_ID set the
//     issuer is https://<domain>/oauth2/<id> and JWKS <issuer>/v1/keys; without
//     it the org issuer is https://<domain> and JWKS <domain>/oauth2/v1/keys.
//   - Accepted audiences: the web client id + M2M client id + any operator-
//     configured M2M audiences (custom auth servers mint M2M tokens whose aud is
//     an API identifier, not a client id). Closed allowlist, verified.
//   - Groups: the "groups" claim. Client id: the "cid" claim. Username: "sub".
//     Scopes: "scp" (array/string) or "scope".
//
// Okta access tokens are standard RS256 with a single issuer, so verification is
// plain verifyRS256; only the claim mapping differs from Keycloak.

// mapOktaClaims turns verified Okta claims into a caller identity. Okta uses
// "sub" as the principal and "cid" as the client id.
func mapOktaClaims(
	c *Claims,
	fallbackClientID string,
) identity {
	username := c.Sub
	if username == "" {
		username = c.Username
	}
	clientID := c.Cid
	if clientID == "" {
		clientID = fallbackClientID
	}
	return identity{
		Sub:      c.Sub,
		Username: username,
		ClientID: clientID,
	}
}
