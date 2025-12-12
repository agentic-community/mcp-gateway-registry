# ARCHITECTURE LOCK — EnforceAI Gateway
# Status: ACTIVE
# Purpose: Defines non-negotiable architectural rules for all agents.

## 1. Identity Model is Stable
IdentityContext:
- user_id
- agent_id
- scopes
- provider
- metadata

## 2. Agent Identity is External to IdP
- Users come from IdP
- Agents come from gateway
- Do not model agents inside IdP

## 3. Authentication Modes are Fixed
Allowed:
- OIDC JWT
- Gateway Tokens
- API Keys

## 4. Authorization is Agent-Scoped
- Scopes decide authority
- User roles do not override agents

## 5. No LLMs in Runtime Path
- No dynamic auth decisions via LLM

## 6. Gateway Architecture is Stable
Components:
- Stateless Gateway
- Stateful Enforcement Point
- Optional Registry

## 7. No Permission Elevation
- Agent scopes cannot exceed registry-defined scopes

## 8. Backward Compatibility
- OIDC must be generic

## 9. Critical Terms are Locked
- agent_id, user_id, scopes

## 10. Only explicit user requests may change this file.
