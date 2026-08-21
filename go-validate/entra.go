package main

// Microsoft Entra ID fast path. Mirrors auth_server/providers/entra.py:
//
//   - JWKS at <login_base>/<tenant>/discovery/v2.0/keys.
//   - TWO accepted issuers: v2 <login_base>/<tenant>/v2.0 and v1
//     https://sts.windows.net/<tenant>/ (M2M/v1 tokens use the v1 issuer).
//   - Accepted audiences: the client id, api://<client-id>, and the operator's
//     Application ID URI. Verified as a closed allowlist (never a wildcard).
//   - id_token replay guard: an Entra id_token shares the JWKS/issuer and has
//     aud == client_id, so signature+issuer+audience do NOT distinguish it from
//     an access token. Reject on id_token-only claims (nonce/at_hash/c_hash) --
//     here we defer such tokens to Python (which rejects them), never accept.
//   - Groups: the "groups" claim for user tokens; for M2M tokens Entra puts
//     membership in "roles", so fall back to roles when groups is empty.

// verifyEntra verifies an Entra access token: standard RS256 + issuer-list +
// audience-list, then the id_token-only-claim guard. A token carrying an
// id_token-only claim is deferred to Python (errUnknownKey -> fallback), matching
// the authoritative path which rejects it, so the fast path never accepts a token
// Python would reject.
func verifyEntra(
	token string,
	ks *keysetCache,
	issuers []string,
	audiences []string,
) (*Claims, error) {
	c, err := verifyRS256(token, ks, issuers, audiences)
	if err != nil {
		return nil, err
	}
	for _, idTokenOnly := range []string{"nonce", "at_hash", "c_hash"} {
		if c.hasClaim(idTokenOnly) {
			return nil, errUnknownKey // id_token presented as access token -> defer to Python
		}
	}
	return c, nil
}

// entraGroups returns the group memberships: the "groups" claim for user tokens,
// or the "roles" claim for M2M tokens (which carry membership there).
func entraGroups(c *Claims) []string {
	if len(c.Groups) > 0 {
		return c.Groups
	}
	return c.Roles
}

// mapEntraClaims turns verified Entra claims into a caller identity. Username is
// preferred_username (falling back to sub); the client is in azp.
func mapEntraClaims(
	c *Claims,
	fallbackClientID string,
) identity {
	username := c.Username
	if username == "" {
		username = c.Sub
	}
	clientID := c.Azp
	if clientID == "" {
		clientID = fallbackClientID
	}
	return identity{
		Sub:      c.Sub,
		Username: username,
		ClientID: clientID,
	}
}
