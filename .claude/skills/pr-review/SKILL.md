---
name: pr-review
description: Review a GitHub pull request using multiple expert personas. Takes a PR URL as input, analyzes the changes, and generates comprehensive review feedback from different perspectives (Merge Specialist, Frontend, Backend, Security, DevOps, AI/Agent, SRE, Chief Architect).
license: Apache-2.0
metadata:
  author: mcp-gateway-registry
  version: "1.1"
---

# PR Review Skill

Use this skill to review GitHub pull requests comprehensively using multiple expert personas. Each persona brings specialized knowledge to identify issues from different perspectives.

## Input

The skill takes a GitHub PR URL as input:
- Format: `https://github.com/{owner}/{repo}/pull/{number}`
- Example: `https://github.com/agentic-community/mcp-gateway-registry/pull/123`

## Output

Creates review documentation in `.scratchpad/pr-{pr-number}/` containing:
- `review.md` - Comprehensive review from all personas

## Workflow

### Step 1: Parse PR URL and Fetch PR Details

1. Extract the PR number from the URL
2. Use `gh pr view {number}` to get PR details
3. Use `gh pr diff {number}` to get the changes
4. Identify which files are changed and their types (frontend, backend, etc.)

### Step 2: Determine Relevant Personas

Based on the files changed, determine which personas should review:

| Changed Files | Personas to Engage |
|---------------|-------------------|
| `/frontend/**` | Merge Specialist, Frontend Developer, Chief Architect |
| `/registry/**` | Merge Specialist, Backend Developer, Security Engineer, SRE, Chief Architect |
| `/registry/core/config.py`, `/registry/api/config_routes.py` | Merge Specialist, Backend Developer, **DevOps Engineer**, Security Engineer, Chief Architect |
| `/auth_server/**` | Merge Specialist, Backend Developer, Security Engineer, Chief Architect |
| `/terraform/**`, `/charts/**`, `/docker/**` | Merge Specialist, DevOps Engineer, SRE, Chief Architect |
| `/agents/**`, `/servers/**` | Merge Specialist, AI/Agent Developer, Backend Developer, Chief Architect |
| `/metrics-service/**` | Merge Specialist, SRE Engineer, Backend Developer, Chief Architect |
| `*.md`, `docs/**` | Merge Specialist, Chief Architect |
| `pyproject.toml`, `requirements*.txt` | Merge Specialist, DevOps Engineer, Security Engineer, Chief Architect |
| `tests/**` | Merge Specialist, Backend Developer, Chief Architect |
| `.env.example` | Merge Specialist, **DevOps Engineer**, Chief Architect |

**Note:** Merge Specialist and Chief Architect always participate in every review.

### Step 2.5: Detect New or Modified Configuration Parameters (CRITICAL)

Before running any reviews, determine whether this PR introduces or modifies any configuration parameters across the three deployment surfaces. If it does, the unified parameter reference must be updated in the same PR — missing updates are a **blocker**.

**Detection command:**

```bash
# Any diff that touches one of the canonical parameter-carrying files triggers this check
gh pr diff {pr-number} --name-only | grep -E \
  -e '^\.env\.example$' \
  -e '^docker-compose(\.|$)' \
  -e '^terraform/aws-ecs/terraform\.tfvars\.example$' \
  -e '^terraform/aws-ecs/variables\.tf$' \
  -e '^terraform/aws-ecs/modules/.+/(variables|ecs-services)\.tf$' \
  -e '^charts/.+/values\.yaml$' \
  -e '^charts/.+/templates/(deployment|secret)\.yaml$' \
  -e '^registry/core/config\.py$' \
  -e '^registry/api/config_routes\.py$'
```

**If any files match**, every new or renamed parameter must be reflected in `docs/unified-parameter-reference.md`. Verify with:

```bash
# For each new parameter name, confirm the reference file mentions it
for PARAM in $(gh pr diff {pr-number} | grep -E '^\+[A-Z_]{3,}=' | sed 's/^+//;s/=.*//' | sort -u); do
  if ! grep -q "$PARAM" docs/unified-parameter-reference.md; then
    echo "MISSING from unified-parameter-reference.md: $PARAM"
  fi
done
```

**Merge-blocking checks (add to the Merge Specialist review section):**

- [ ] `docs/unified-parameter-reference.md` is included in the diff whenever any parameter-carrying file is touched.
- [ ] Every new `.env` variable has a row, with Docker / Terraform / Helm columns filled (or explicitly blank with a justification in the PR description).
- [ ] Every new Terraform variable appears in the reference.
- [ ] Every new Helm value appears in the reference.
- [ ] Secrets are flagged with **(secret)**.
- [ ] New rows live in an existing logical group, or the PR adds a new group with a clear rationale.
- [ ] Renamed parameters have the old row updated in place (not a duplicate).
- [ ] Deleted parameters have their row removed (not left stale).
- [ ] `registry/api/config_routes.py` `CONFIG_GROUPS` is updated so the parameter surfaces in `GET /api/config/full` and the System Config UI.

If any of the above is missing, the review verdict is **REQUEST CHANGES** with a blocker titled "Unified parameter reference not updated".

### Step 3: Run Tests and Quality Checks

Before reviewing, run the test suite to verify the PR doesn't break anything:

```bash
# Checkout the PR
gh pr checkout {pr-number}

# Run tests
uv run pytest tests/ -n 8 --tb=short

# Run linting
uv run ruff check . && uv run ruff format --check .

# Run security scan (if applicable)
uv run bandit -r registry/ auth_server/ -q

# Return to main branch when done
git checkout main
```

### Step 4: Create Review Folder

Create the folder structure:

```
.scratchpad/pr-{pr-number}/
└── review.md
```

### Step 5: Conduct Multi-Persona Review

For each relevant persona, adopt that perspective and review the changes. Reference the persona definition files:

- [Merge Specialist](personas/merge-specialist.md) - Always included
- [Frontend Developer](personas/frontend-developer.md) - For frontend changes
- [Backend Developer](personas/backend-developer.md) - For backend/API changes
- [Security Engineer](personas/security-engineer.md) - For auth/security changes
- [DevOps Engineer](personas/devops-engineer.md) - For infrastructure changes
- [AI/Agent Developer](personas/ai-agent-developer.md) - For agent/MCP changes
- [SRE Engineer](personas/sre-engineer.md) - For observability/metrics changes
- [Chief Architect](personas/chief-architect.md) - Always included (final synthesis)

### Step 6: Write Comprehensive Review (review.md)

Generate the review document using this structure:

```markdown
# PR Review: #{pr-number} - {pr-title}

*Review Date: {date}*
*PR URL: {pr-url}*
*Author: {author}*

## PR Summary

{Brief description of what the PR does based on PR description and changes}

### Files Changed

| File | Type | Lines Added | Lines Removed |
|------|------|-------------|---------------|
| {file} | {type} | +{n} | -{n} |

### Test Results

| Check | Status | Details |
|-------|--------|---------|
| Unit Tests | {PASS/FAIL} | {summary} |
| Integration Tests | {PASS/FAIL} | {summary} |
| Linting | {PASS/FAIL} | {summary} |
| Security Scan | {PASS/FAIL} | {summary} |

### Configuration Parameter Surface Check

*Only required when the PR touches any parameter-carrying file (see Step 2.5). Mark "Not Applicable" if the detection command returned no matches.*

| Check | Status | Details |
|-------|--------|---------|
| Unified parameter reference updated (`docs/unified-parameter-reference.md`) | {PASS/FAIL/N/A} | {list of new/renamed/removed parameter names and which rows were added} |
| Docker column populated (`.env.example`, `docker-compose*.yml`) | {PASS/FAIL/N/A} | — |
| Terraform column populated (`variables.tf`, `terraform.tfvars.example`, module wiring) | {PASS/FAIL/N/A} | — |
| Helm column populated (`charts/.../values.yaml`, stack values, templates) | {PASS/FAIL/N/A} | — |
| `registry/api/config_routes.py` `CONFIG_GROUPS` updated | {PASS/FAIL/N/A} | — |
| Secrets flagged with **(secret)** and wired through Secrets Manager / `secretKeyRef` | {PASS/FAIL/N/A} | — |

---

## Review Panel

| Role | Reviewer | Verdict |
|------|----------|---------|
| Merge Specialist | Gatekeeper | {verdict} |
| {Role} | {Name} | {verdict} |
| Chief Architect | Atlas | {verdict} |

---

{Include each relevant persona's review section using the format from their persona file}

---

## Review Summary

| Reviewer | Verdict | Blockers | Key Concerns |
|----------|---------|----------|--------------|
| {Reviewer} | {verdict} | {count} | {summary} |

### Blockers (Must Fix)

1. {Blocker description}
   - Raised by: {persona}
   - File: `{file:line}`
   - Fix: {suggested fix}

### Should Fix (Important)

1. {Issue description}
   - Raised by: {persona}
   - File: `{file:line}`
   - Recommendation: {suggestion}

### Consider (Nice to Have)

1. {Suggestion}
   - Raised by: {persona}

---

## Final Recommendation

**Overall Verdict: {APPROVE / APPROVE WITH CHANGES / REQUEST CHANGES}**

### Required Actions Before Merge

- [ ] {Action 1}
- [ ] {Action 2}
- [ ] (If config params changed) `docs/unified-parameter-reference.md` updated and all three surface columns consistent with the diff

### Post-Merge Actions

- [ ] {Action 1}
```

### Step 7: Present Review Summary

After creating the review document, present a summary to the user:

1. Display the overall verdict
2. List any blockers that must be addressed
3. Provide the path to the full review document
4. Offer to explain any specific findings in detail

## Review Principles

### From CLAUDE.md

- **Simplicity**: Code should be maintainable by entry-level developers
- **No Over-engineering**: Only make changes that are directly requested
- **Security First**: Check for OWASP vulnerabilities, proper input validation
- **Test Coverage**: Verify tests exist for new functionality
- **Documentation**: Ensure docstrings and comments are appropriate

### Severity Levels

- **Blocker**: Must be fixed before merge (security vulnerabilities, failing tests, breaking changes)
- **Major**: Should be fixed before merge (code quality issues, missing tests)
- **Minor**: Nice to fix (style issues, documentation improvements)

### Verdict Criteria

**APPROVE**:
- All tests pass
- No security vulnerabilities
- Code quality meets standards
- No breaking changes (or justified)

**APPROVE WITH CHANGES**:
- Minor issues that should be addressed
- No blockers
- Author can address and merge

**REQUEST CHANGES**:
- Failing tests
- Security vulnerabilities
- Breaking changes without justification
- Significant code quality issues

## Example Usage

User: "/pr-review https://github.com/agentic-community/mcp-gateway-registry/pull/456"

1. Parse URL: PR #456
2. Fetch PR details and diff
3. Identify changed files: `registry/routes/auth.py`, `tests/unit/test_auth.py`
4. Determine personas: Merge Specialist, Backend Developer, Security Engineer, Chief Architect
5. Run tests: All pass
6. Create `.scratchpad/pr-456/review.md`
7. Conduct reviews from each persona
8. Present summary with verdict

## Notes

- Always run tests before reviewing to ensure baseline quality
- Focus review effort on areas most relevant to the changed files
- Be constructive and specific - provide file/line references
- Acknowledge good practices, not just problems
- Consider the author's experience level when phrasing feedback
