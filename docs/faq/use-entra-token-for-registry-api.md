# Can I use an Entra ID token to call the registry API instead of the UI-generated token?

Yes -- you can use Entra ID-based tokens directly for API authorization instead of the tokens from the registry UI. The recommended approach is to create an M2M (Machine-to-Machine) identity in Entra ID and assign it to a registry group to control its access.

## Setup Steps

1. **Register an App Registration** in Entra ID with client credentials (client ID + client secret)
2. In the registry UI, go to **Settings > IAM > M2M Accounts** and create an M2M account linked to this Entra ID app
3. **Assign the M2M account to a group** -- this restricts its access to only the servers/tools that group allows (see [How do I restrict server visibility by Entra group?](restrict-server-visibility-by-entra-group.md))
4. **Request tokens** directly from Entra ID using the standard OAuth2 client credentials flow:

```bash
curl -X POST "https://login.microsoftonline.com/{TENANT_ID}/oauth2/v2.0/token" \
  -H "Content-Type: application/x-www-form-urlencoded" \
  -d "client_id={M2M_CLIENT_ID}" \
  -d "client_secret={M2M_CLIENT_SECRET}" \
  -d "scope=api://{APP_CLIENT_ID}/.default" \
  -d "grant_type=client_credentials"
```

Where:
- `{TENANT_ID}` is your Azure AD Tenant ID
- `{M2M_CLIENT_ID}` is the M2M service account Client ID
- `{M2M_CLIENT_SECRET}` is the M2M service account Client Secret
- `{APP_CLIENT_ID}` is the Application (client) ID of your MCP Gateway app registration in Entra ID
- `.default` requests all scopes that admin consent has been granted for

5. **Use the resulting token** in API calls:

```bash
curl -H "Authorization: Bearer {ACCESS_TOKEN}" \
  https://your-registry-url/api/servers
```

## How Token Validation Works

The registry validates Entra ID tokens (RS256) by:
1. Fetching the JWKS from your Entra ID tenant
2. Verifying the token signature, issuer, and audience claims
3. Extracting group claims from the token
4. Mapping group claims to registry scopes

The M2M identity will only see the servers and tools that its assigned group allows.

## Related Documentation

- [Entra ID Setup - M2M Token Generation](https://github.com/agentic-community/mcp-gateway-registry/blob/main/docs/entra-id-setup.md#generating-jwt-tokens-for-m2m-accounts) -- covers direct token requests, credentials provider scripts, and token usage
- [Authentication Overview](https://github.com/agentic-community/mcp-gateway-registry/blob/main/docs/auth.md) -- covers all three identity types (Human, Programmatic, M2M) and how group-to-scope mapping works for each
- [Auth Management](https://github.com/agentic-community/mcp-gateway-registry/blob/main/docs/auth-mgmt.md) -- M2M account creation and token usage examples
