# Agent Registry — EnforceAI Gateway

## Agent Model
```
Agent {
    agent_id: string,
    user_id: string,
    scopes: string[],
    allowed_tools?: string[],
    metadata?: Record<string, any>,
    created_at: timestamp,
    revoked: boolean
}
```

## Rules
- Every agent belongs to exactly one user
- Scopes must be assigned explicitly
- Revoked agents cannot authenticate

## Required CRUD APIs
- POST /agents
- GET /agents/:user
- PATCH /agents/:agent_id
- DELETE /agents/:agent_id
- POST /agents/:agent_id/revoke
