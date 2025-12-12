# Gateway Tokens — EnforceAI Gateway

## Token Format
```
{
  "iss": "<gateway>",
  "sub": "<user_id>",
  "agent_id": "<agent_id>",
  "scopes": [...],
  "iat": ts,
  "exp": ts
}
```

## Rules
- Only gateway issues tokens
- Tokens MUST embed user_id + agent_id + scopes
- No agent may have implicit permissions
- Signature required (HS256 or ES256)
