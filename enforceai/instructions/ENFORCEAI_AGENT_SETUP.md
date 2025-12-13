# EnforceAI Agent Setup Guide

## Purpose
These instructions tell AI agents exactly how to initialize themselves,
load context, and work safely on the EnforceAI Gateway project.

## Initialization Checklist
1. Load `enforceai/architecture/architecture_lock.md`
2. Load `enforceai/instructions/ENFORCEAI_CONTEXT.md`
3. Load `enforceai/session_state/latest.md`
4. Load `enforceai/architecture/identity_model.md`
5. Load `enforceai/architecture/agent_registry.md`
6. Load `enforceai/architecture/fgac_model.md`
7. Load `enforceai/architecture/gateway_tokens.md`
8. Load `enforceai/decisions/` (most recent ADRs relevant to the task)
9. Load `enforceai/plans/` (execute the current phase plan)
10. Load `enforceai/roadmap/` (if present)
11. Ask for missing context instead of guessing
12. Confirm role (Architect, Engineer, Reviewer, Tester)

## Anchors
- Gateway uses IdentityContext(user_id, agent_id, scopes)
- Agent identity is external to IdP
- FGAC is agent-scoped
- No LLMs in runtime path
- Authentication modes: OIDC, Gateway Token, API Key

## Rules
- Never modify locked architecture unless asked
- Never redesign authentication model
- Never merge agent & user identity
- Never override scopes from IdP roles

## End-of-Session Requirements
- Update session_state/latest.md
- Summarize progress + next steps
- List tests executed and their results
