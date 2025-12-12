#!/usr/bin/env bash

set -e

ROOT_DIR=$(pwd)
ENF_DIR="$ROOT_DIR/enforceai"

FORCE=0
if [[ "$1" == "--force" ]]; then
  FORCE=1
  echo "[WARN] Force mode enabled: existing files will be overwritten."
fi

create_file() {
  local path="$1"
  local content="$2"

  if [[ -f "$path" && $FORCE -eq 0 ]]; then
    echo "[SKIP] $path already exists (use --force to overwrite)"
    return
  fi

  mkdir -p "$(dirname "$path")"
  echo "$content" > "$path"
  echo "[OK] Created: $path"
}

echo "------------------------------------------------------"
echo " EnforceAI Gateway Repository Initialization"
echo "------------------------------------------------------"
echo ""
echo "Project root: $ROOT_DIR"
echo "Creating folder structure..."
echo ""

mkdir -p "$ENF_DIR/architecture"
mkdir -p "$ENF_DIR/roadmap"
mkdir -p "$ENF_DIR/testing"
mkdir -p "$ENF_DIR/decisions"
mkdir -p "$ENF_DIR/session_state"
mkdir -p "$ENF_DIR/instructions"

echo "[OK] Base directories created."
echo ""

# ---------------------------------------------------------------------------
# architecture_lock.md
# ---------------------------------------------------------------------------

create_file "$ENF_DIR/architecture/architecture_lock.md" \
"# ARCHITECTURE LOCK — EnforceAI Gateway
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

## 10. Only explicit user requests may change this file."

# ---------------------------------------------------------------------------
# identity_model.md
# ---------------------------------------------------------------------------

create_file "$ENF_DIR/architecture/identity_model.md" \
"# Identity Model — EnforceAI Gateway

## IdentityContext Definition
\`\`\`
IdentityContext {
    user_id: string,
    agent_id: string,
    provider: \"oidc\" | \"gateway-token\" | \"api-key\",
    scopes: string[],
    user_roles?: string[],
    metadata?: Record<string, any>
}
\`\`\`

## Rules
- Constructed once per request
- Agent identity NEVER comes from IdP
- Authorization uses agent scopes only

## Supported Credential Sources
- OIDC JWT
- Gateway Token
- API Key"

# ---------------------------------------------------------------------------
# agent_registry.md
# ---------------------------------------------------------------------------

create_file "$ENF_DIR/architecture/agent_registry.md" \
"# Agent Registry — EnforceAI Gateway

## Agent Model
\`\`\`
Agent {
    agent_id: string,
    user_id: string,
    scopes: string[],
    allowed_tools?: string[],
    metadata?: Record<string, any>,
    created_at: timestamp,
    revoked: boolean
}
\`\`\`

## Rules
- Every agent belongs to exactly one user
- Scopes must be assigned explicitly
- Revoked agents cannot authenticate

## Required CRUD APIs
- POST /agents
- GET /agents/:user
- PATCH /agents/:agent_id
- DELETE /agents/:agent_id
- POST /agents/:agent_id/revoke"

# ---------------------------------------------------------------------------
# gateway_tokens.md
# ---------------------------------------------------------------------------

create_file "$ENF_DIR/architecture/gateway_tokens.md" \
"# Gateway Tokens — EnforceAI Gateway

## Token Format
\`\`\`
{
  \"iss\": \"<gateway>\",
  \"sub\": \"<user_id>\",
  \"agent_id\": \"<agent_id>\",
  \"scopes\": [...],
  \"iat\": ts,
  \"exp\": ts
}
\`\`\`

## Rules
- Only gateway issues tokens
- Tokens MUST embed user_id + agent_id + scopes
- No agent may have implicit permissions
- Signature required (HS256 or ES256)"

# ---------------------------------------------------------------------------
# fgac_model.md
# ---------------------------------------------------------------------------

create_file "$ENF_DIR/architecture/fgac_model.md" \
"# FGAC Model — EnforceAI Gateway

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
- timestamp"

# ---------------------------------------------------------------------------
# session_state/latest.md
# ---------------------------------------------------------------------------

create_file "$ENF_DIR/session_state/latest.md" \
"# Session State — Latest

## Last Completed Work
- Repo initialized
- Base architecture files created

## Current Task
- Begin implementing the roadmap
- Start OIDC validator design
- Draft database schema for agents

## Next Steps
1. IdentityResolver scaffold
2. Agent registry models
3. Gateway token signing system

## Outstanding Questions
- DB backend choice
- Token signing algorithm"

# ---------------------------------------------------------------------------
# instructions/ENFORCEAI_AGENT_SETUP.md
# ---------------------------------------------------------------------------

create_file "$ENF_DIR/instructions/ENFORCEAI_AGENT_SETUP.md" \
"# EnforceAI Agent Setup Guide

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
- Summarize progress + next steps"

echo ""
echo "------------------------------------------------------"
echo " EnforceAI Gateway initialization complete!"
echo "------------------------------------------------------"
echo "Your project is now ready for multi-agent, multi-session AI development."