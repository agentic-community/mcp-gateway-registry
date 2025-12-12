# Identity Model — EnforceAI Gateway

## IdentityContext Definition
```
IdentityContext {
    user_id: string,
    agent_id: string,
    provider: "oidc" | "gateway-token" | "api-key",
    scopes: string[],
    user_roles?: string[],
    metadata?: Record<string, any>
}
```

## Rules
- Constructed once per request
- Agent identity NEVER comes from IdP
- Authorization uses agent scopes only

## Supported Credential Sources
- OIDC JWT
- Gateway Token
- API Key
