# FGAC Model — EnforceAI Gateway

## Enforcement Logic
1. Extract IdentityContext
2. Determine tool + action
3. Validate:
   allow = action in scopes AND tool in allowed_tools
4. Default-deny for unknown actions
5. Log all decisions

## Audit Fields
- user_id
- agent_id
- action
- tool
- decision
- reason
- timestamp
