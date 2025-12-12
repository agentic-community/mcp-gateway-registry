# Decision 0016 — Management Surface (CLI-First, Self-Service)
*Status: Accepted*  
*Date: 2025-12-12*

## Context
EnforceAI requires agent and credential lifecycle management (agents, gateway tokens, API keys). Phase 1 focuses on stabilizing core identity/enforcement behavior with minimal UI scope.
The enforcement path must not depend on UI availability.

## Decision
- Phase 1 management is CLI-first.
- Management is self-service per user:
  - Users can create, list, update, and revoke their own agents and credentials.
  - Cross-user administrative operations are deferred to an explicit admin feature later.

## Consequences
- Faster iteration and lower risk while core enforcement is developed.
- UI can be added later without changing the enforcement architecture.
