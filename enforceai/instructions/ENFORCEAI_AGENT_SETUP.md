# EnforceAI Agent Setup Guide

## Purpose
These instructions tell AI agents exactly how to initialize themselves,
load context, and work safely on the EnforceAI Gateway project.

## Initialization Checklist
1. Load architecture_lock.md
2. Load session_state/latest.md
3. Load identity_model.md
4. Load agent_registry.md
5. Load roadmap
6. Ask for missing context instead of guessing
7. Confirm role (Architect, Engineer, Reviewer, Tester)

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
