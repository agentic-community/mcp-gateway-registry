---
name: new-feature-design
description: "Design and document new features with GitHub issue, low-level design (LLD), expert review, and testing plan. Creates structured documentation in .scratchpad/ with issue spec, technical design with diagrams and pseudo-code, multi-persona expert review, and a testing plan covering functional (curl and registry_management.py), backwards-compatibility, UX, ECS/terraform, and E2E API tests. Supports starting from a user description OR an existing GitHub issue URL. Folder naming: issue-{number}/ for existing issues, {feature-name}/ for new features."
license: Apache-2.0
metadata:
  author: mcp-gateway-registry
  version: "1.5"
---

# New Feature Design Skill

Use this skill when the user wants to design a new feature for the MCP Gateway Registry. This skill creates comprehensive design documentation that enables entry-level developers to implement the feature.

## Input Modes

This skill supports two input modes:

1. **User Description Mode** - User describes the feature they want to design
2. **GitHub Issue URL Mode** - User provides a URL to an existing GitHub issue

### Detecting Input Mode

When the user invokes this skill:
- If they provide a GitHub issue URL (e.g., `https://github.com/owner/repo/issues/123`), use **GitHub Issue URL Mode**
- Otherwise, use **User Description Mode**

## Workflow

### User Description Mode Workflow
1. **Gather Requirements** - Ask clarifying questions about the feature
2. **Quick Codebase Review** - Explore the codebase to understand structure
3. **Create Design Folder** - Create `.scratchpad/{feature-name}/` directory
4. **Write GitHub Issue** - Create `github-issue.md` for issue creation
5. **Deep Codebase Analysis** - Thoroughly explore relevant code
6. **Write Low-Level Design** - Create `lld.md` with technical details
7. **Expert Review** - Create `review.md` with multi-persona feedback
8. **Write Testing Plan** - Create `testing.md` with functional, backwards-compat, UX, ECS, and E2E tests
9. **Present Summary & Seek Guidance** - Present findings and ask for direction

### GitHub Issue URL Mode Workflow
1. **Fetch GitHub Issue** - Retrieve issue content using `gh` CLI
2. **Analyze Issue Content** - Extract requirements, design elements, and context
3. **Quick Codebase Review** - Explore the codebase to understand structure
4. **Create Design Folder** - Create `.scratchpad/issue-{number}/` directory (e.g., `issue-500` for GitHub issue #500)
5. **Summarize Existing Issue** - Create `github-issue.md` summarizing the existing issue
6. **Clarify Requirements** - Ask user about any gaps or ambiguities found
7. **Deep Codebase Analysis** - Thoroughly explore relevant code
8. **Write Low-Level Design** - Create `lld.md` with technical details
9. **Expert Review** - Create `review.md` with multi-persona feedback
10. **Write Testing Plan** - Create `testing.md` with functional, backwards-compat, UX, ECS, and E2E tests
11. **Present Summary & Seek Guidance** - Present findings and ask for direction

---

## GitHub Issue URL Mode

### Step 1: Fetch GitHub Issue

When a user provides a GitHub issue URL, fetch the issue content using the `gh` CLI:

```bash
# Extract owner, repo, and issue number from URL
# URL format: https://github.com/{owner}/{repo}/issues/{number}

# Fetch issue details
gh issue view {number} --repo {owner}/{repo} --json title,body,labels,state,assignees,comments

# For more context, also fetch related PRs if any
gh issue view {number} --repo {owner}/{repo} --json linkedPullRequests
```

### Step 2: Analyze Issue Content

After fetching the issue, analyze it for:

1. **Existing Requirements**
   - Problem statement
   - User stories or use cases
   - Acceptance criteria
   - Scope definition

2. **Existing Design Elements**
   - Technical proposals in the issue body
   - Architecture suggestions
   - API designs mentioned
   - Data model proposals
   - Diagrams or pseudocode

3. **Discussion Context**
   - Comments with additional requirements
   - Decisions made in discussion
   - Concerns raised by team members
   - Alternative approaches discussed

4. **Metadata**
   - Labels (indicates feature type, priority)
   - Assignees (who might have context)
   - Linked PRs (prior implementation attempts)

### Step 3: Create Issue Summary

Create a summary document that captures what was found:

```markdown
# GitHub Issue Analysis: #{issue_number}

*Source: {issue_url}*
*Fetched: {date}*

## Issue Overview

| Field | Value |
|-------|-------|
| Title | {title} |
| State | {state} |
| Labels | {labels} |
| Assignees | {assignees} |

## Extracted Requirements

### Problem Statement
{extracted or "Not explicitly stated - needs clarification"}

### User Stories
{extracted user stories or "None found"}

### Acceptance Criteria
{extracted criteria or "Not defined - will need to establish"}

### Scope
{inferred scope or "Not specified"}

## Existing Design Elements

### Technical Proposals
{any technical details from issue body}

### Architecture Suggestions
{any architecture mentioned}

### API Design
{any API designs proposed}

### Open Questions in Issue
{questions raised but not answered}

## Discussion Summary

### Key Decisions Made
{decisions from comments}

### Concerns Raised
{concerns from team members}

### Alternative Approaches Mentioned
{alternatives discussed}

## Gaps Identified

The following information is missing and needs clarification:
1. {gap 1}
2. {gap 2}
```

### Step 4: Clarify Requirements

Based on the analysis, ask the user only about gaps not covered in the issue:

1. **Feature name** (derive from issue title if possible, use kebab-case)
2. **Missing requirements** - Only ask about gaps identified
3. **Design decisions** - If alternatives were discussed but not decided
4. **Scope confirmation** - Verify the scope interpretation is correct

Example prompt:
```
I've analyzed GitHub issue #{number}: "{title}"

The issue provides:
- {what's available}

However, I need clarification on:
1. {gap question 1}
2. {gap question 2}

Also, the issue discusses two approaches for {topic}:
- Option A: {description}
- Option B: {description}

Which approach should we pursue in the design?
```

---

## User Description Mode

## Step 1: Gather Requirements

Before creating any files, ask the user:

1. What is the feature name? (will be used for folder name, use kebab-case)
2. What problem does this feature solve?
3. Who are the users/consumers of this feature?
4. Are there any constraints or requirements?
5. What is the expected scope (small/medium/large)?

## Step 2: Quick Codebase Review

Before creating any design documents, perform a quick exploration of the codebase to understand:

1. **Project Structure** - Use Glob to understand the overall directory layout
   - Identify main source directories (`registry/`, `src/`, etc.)
   - Locate existing models, routes, services, and utilities
   - Find configuration files and constants

2. **Related Components** - Search for existing features similar to the one being designed
   - Use Grep to find relevant keywords and patterns
   - Identify existing patterns and conventions used in the codebase

3. **Entry Points** - Understand how the application is structured
   - Find the main FastAPI application and router setup
   - Identify middleware, dependencies, and shared utilities

This quick review should take 5-10 minutes and helps you ask better clarifying questions and avoid proposing designs that conflict with existing architecture.

## Step 3: Create Design Folder

### Folder Naming Convention

**IMPORTANT:** Use different naming conventions based on the input mode:

| Input Mode | Folder Name | Example |
|------------|-------------|---------|
| **GitHub Issue URL Mode** | `issue-{number}/` | `issue-500/` for GitHub issue #500 |
| **User Description Mode** | `{feature-name}/` | `rate-limiting/` for a rate limiting feature |

This convention makes it easy to:
- Trace design documents back to their source GitHub issue
- Avoid duplicate work on the same issue
- Organize designs consistently

### Folder Structure

Create the folder structure:

```
.scratchpad/issue-{number}/     # For GitHub Issue URL Mode
# OR
.scratchpad/{feature-name}/     # For User Description Mode

├── github-issue.md    # GitHub issue specification or summary
├── lld.md             # Low-level design document
├── review.md          # Expert review document
└── testing.md         # Testing plan (functional, backwards-compat, UX, ECS, E2E)
```

## Step 4: Write GitHub Issue (github-issue.md)

**For User Description Mode:** Create a comprehensive GitHub issue specification.

**For GitHub Issue URL Mode:** Create a summary document that consolidates the existing issue content with any clarifications gathered. Use the "Issue Summary from URL" template below instead.

### Template: New Issue Specification (User Description Mode)

```markdown
# GitHub Issue: {Feature Title}

## Title
{concise title for the issue}

## Labels
- {appropriate labels from: enhancement, feature-request, api, frontend, backend, etc.}

## Description

### Problem Statement
{What problem does this solve? Why is it needed?}

### Proposed Solution
{High-level description of the solution}

### User Stories
- As a {user type}, I want to {action} so that {benefit}
- ...

### Acceptance Criteria
- [ ] {Criterion 1}
- [ ] {Criterion 2}
- ...

### Out of Scope
- {What is explicitly NOT included}

### Dependencies
- {Any dependent issues or external dependencies}

### Related Issues
- #{issue numbers if any}
```

### Template: Issue Summary from URL (GitHub Issue URL Mode)

When working from an existing GitHub issue, create a summary document:

```markdown
# Issue Summary: {Issue Title}

*Source Issue: [{owner}/{repo}#{number}]({issue_url})*
*Fetched: {date}*
*Status: {open/closed}*

## Issue Metadata

| Field | Value |
|-------|-------|
| Labels | {labels} |
| Assignees | {assignees} |
| Created | {created_date} |
| Last Updated | {updated_date} |

## Original Problem Statement

{Copy or summarize the problem statement from the issue body}

## Requirements Extracted

### From Issue Body
{Requirements found in the original issue}

### From Discussion (Comments)
{Additional requirements or clarifications from comments}

### Clarified with User
{Any requirements clarified during our conversation}

## Existing Design Elements

{Any technical proposals, architecture suggestions, or design decisions already in the issue}

### Technical Approach
{If specified in the issue}

### API Design
{If specified in the issue}

### Data Models
{If specified in the issue}

## Acceptance Criteria

### From Issue
{Criteria from the original issue}

### Additional (Established)
{Any additional criteria we established}

## Scope

{Scope as defined or inferred from the issue}

## Out of Scope

{What is explicitly excluded}

## Key Decisions from Discussion

| Decision | Context | Decided By |
|----------|---------|------------|
| {decision} | {why} | {who/when} |

## Open Questions Resolved

| Question | Resolution |
|----------|------------|
| {question from issue} | {answer we determined} |

## Dependencies

{Any dependencies mentioned in the issue}

## Notes

{Any additional context or notes relevant to the design}
```

## Step 5: Deep Codebase Analysis

**CRITICAL:** Before writing the LLD, you MUST thoroughly understand all relevant code in the repository. This is not optional - a design that doesn't account for existing code patterns will fail during implementation.

### What to Analyze

1. **Existing Models and Data Structures**
   - Read ALL relevant Pydantic models in `registry/models/`
   - Understand field types, validators, and relationships
   - Identify any models that need to be extended or referenced

2. **Service Layer Patterns**
   - Read existing services in `registry/services/`
   - Understand how business logic is organized
   - Identify common patterns (error handling, logging, caching)
   - Note any base classes or utility functions used

3. **Route/API Patterns**
   - Read existing routes in `registry/routes/`
   - Understand request/response patterns
   - Identify how authentication, validation, and error responses are handled
   - Note middleware and dependencies used

4. **Storage Layer**
   - Read storage implementations in `registry/storage/`
   - Understand how data is persisted
   - Identify any abstraction layers or interfaces

5. **Configuration and Constants**
   - Read configuration files and constants
   - Understand environment variable patterns
   - Identify feature flags or configuration options

6. **Existing Tests**
   - Read relevant test files in `tests/`
   - Understand testing patterns and fixtures used
   - Identify how mocking is done

### How to Analyze

Use the Task tool with subagent_type=Explore for thorough investigation:

```
Task tool with prompt: "Thoroughly analyze the service layer patterns in registry/services/.
Read all service files and document: 1) Common patterns used, 2) Error handling approaches,
3) Logging conventions, 4) Any base classes or utilities, 5) How services interact with storage."
```

For each area, you should:
- Read the actual code, not just file names
- Understand the "why" behind design decisions
- Note any TODOs or known issues
- Identify code that your feature will need to integrate with

### Document Your Findings

Create a brief section in your LLD documenting:
- Key files reviewed
- Patterns identified
- Integration points for the new feature
- Any constraints or limitations discovered

## Step 6: Write Low-Level Design (lld.md)

Create a detailed technical design document. This is the most critical document - it should contain enough detail for an entry-level developer to implement the feature.

```markdown
# Low-Level Design: {Feature Name}

*Created: {date}*
*Author: Claude*
*Status: Draft*

## Table of Contents
1. [Overview](#overview)
2. [Codebase Analysis](#codebase-analysis)
3. [Architecture](#architecture)
4. [Data Models](#data-models)
5. [API Design](#api-design)
6. [Configuration Parameters](#configuration-parameters)
7. [New Dependencies](#new-dependencies)
8. [Implementation Details](#implementation-details)
9. [Observability](#observability)
10. [Scaling Considerations](#scaling-considerations)
11. [File Changes](#file-changes)
12. [Testing Strategy](#testing-strategy)
13. [Alternatives Considered](#alternatives-considered)
14. [Rollout Plan](#rollout-plan)

## Overview

### Problem Statement
{Detailed problem description}

### Goals
- {Goal 1}
- {Goal 2}

### Non-Goals
- {What this design explicitly does NOT address}

## Codebase Analysis

*Summary of the deep codebase analysis performed before writing this LLD.*

### Key Files Reviewed

| File/Directory | Purpose | Relevance to This Feature |
|----------------|---------|---------------------------|
| `registry/models/{model}.py` | {Description} | {How it relates} |
| `registry/services/{service}.py` | {Description} | {How it relates} |
| `registry/routes/{route}.py` | {Description} | {How it relates} |

### Existing Patterns Identified

1. **Pattern Name**: {Description of the pattern and where it's used}
   - Files: `{file1.py}`, `{file2.py}`
   - How we'll follow this: {How the new feature will use this pattern}

2. **Pattern Name**: {Description}
   - Files: `{files}`
   - How we'll follow this: {How we'll apply it}

### Integration Points

| Component | Integration Type | Details |
|-----------|------------------|---------|
| {Existing component} | {Extends/Uses/Depends on} | {Specific integration details} |

### Constraints and Limitations Discovered

- {Constraint 1}: {How it affects the design}
- {Constraint 2}: {How it affects the design}

### Code Snippets Reference

{Include relevant code snippets from existing codebase that the new feature will integrate with or follow patterns from}

```python
# From registry/services/example_service.py:45-60
# This shows the pattern we'll follow for...
def existing_pattern_example():
    pass
```

## Architecture

### System Context Diagram
{ASCII diagram showing how this feature fits into the overall system}

```
┌─────────────────┐     ┌─────────────────┐     ┌─────────────────┐
│                 │     │                 │     │                 │
│   Component A   │────▶│   New Feature   │────▶│   Component B   │
│                 │     │                 │     │                 │
└─────────────────┘     └─────────────────┘     └─────────────────┘
```

### Sequence Diagram
{Show the flow of requests/data}

```
User          Frontend       Backend        Database
  │               │              │              │
  │──── Request ──▶│              │              │
  │               │── API Call ──▶│              │
  │               │              │── Query ────▶│
  │               │              │◀── Result ───│
  │               │◀── Response ─│              │
  │◀── Display ───│              │              │
```

### Component Diagram
{Show internal components and their relationships}

## Data Models

### New Models
{Define any new Pydantic models with full field definitions}

```python
class NewModel(BaseModel):
    """Description of the model."""

    field_name: str = Field(
        ...,
        description="What this field represents",
        min_length=1,
        max_length=100
    )
    optional_field: Optional[int] = Field(
        default=None,
        description="Optional field description"
    )
```

### Model Changes
{Changes to existing models}

### Database Schema Changes
{If applicable, show collection/table changes}

## API Design

### New Endpoints

#### POST /api/v1/{endpoint}

**Description:** {What this endpoint does}

**Request:**
```json
{
    "field": "value"
}
```

**Response (200 OK):**
```json
{
    "id": "123",
    "status": "success"
}
```

**Error Responses:**
- 400 Bad Request: {when}
- 401 Unauthorized: {when}
- 404 Not Found: {when}

### API Changes
{Changes to existing endpoints}

## Configuration Parameters

**IMPORTANT:** Clearly document all new configuration parameters this feature requires.

### New Environment Variables

| Variable Name | Type | Default | Required | Description |
|---------------|------|---------|----------|-------------|
| `FEATURE_ENABLED` | bool | `true` | No | Enable/disable the feature |
| `FEATURE_INTERVAL_SECONDS` | int | `30` | No | Polling interval in seconds |

### Settings Class Updates

Add to `registry/core/config.py`:

```python
    # Feature-specific settings
    feature_enabled: bool = Field(
        default=True,
        description="Enable/disable feature X"
    )
    feature_interval_seconds: int = Field(
        default=30,
        description="Interval for feature X operations"
    )
```

### Configuration Validation

{Any validation rules for configuration values}

## New Dependencies

**IMPORTANT:** List all new Python packages or libraries required by this feature.

### Python Packages

| Package | Version | Purpose | Added By |
|---------|---------|---------|----------|
| `package-name` | `latest` | {Why needed} | Backend |
| `another-package` | `latest` | {Why needed} | Frontend |

### Adding Dependencies

```bash
# Add to pyproject.toml
uv add package-name
```

### No New Dependencies

If no new dependencies are required, explicitly state:
- "This feature uses only existing dependencies from the project."

## Implementation Details

### Step-by-Step Implementation

#### Step 1: {First Step}

**File:** `path/to/file.py`
**Lines:** {approximate line numbers or "new file"}

```python
# Pseudo-code or actual code
def new_function(
    param1: str,
    param2: int
) -> dict:
    """
    Description of what this function does.

    Args:
        param1: Description
        param2: Description

    Returns:
        Description of return value
    """
    # Step 1: Validate input
    if not param1:
        raise ValueError("param1 is required")

    # Step 2: Process data
    result = process_data(param1, param2)

    # Step 3: Return response
    return {"status": "success", "data": result}
```

#### Step 2: {Second Step}

{Continue with detailed steps...}

### Error Handling

{How errors should be handled}

### Logging

{What should be logged and at what level}

## Observability

### Tracing

{Define trace spans for this feature}

| Span Name | Attributes | Parent Span |
|-----------|------------|-------------|
| `feature.operation_name` | `param1`, `param2` | `http.request` |

### Metrics

{Define metrics to track for this feature}

| Metric Name | Type | Labels | Description |
|-------------|------|--------|-------------|
| `feature_requests_total` | Counter | `status`, `endpoint` | Total requests |
| `feature_duration_seconds` | Histogram | `operation` | Operation duration |

### Logging Points

{Key operations that should emit logs}

| Log Level | Event | Data to Include |
|-----------|-------|-----------------|
| INFO | Operation started | request_id, user_id |
| ERROR | Operation failed | error_type, context |

## Scaling Considerations

### Current Load Assumptions

{Expected request volume and data size}

### Horizontal Scaling

{How this feature scales across multiple instances}

### Bottlenecks

{Potential bottlenecks and mitigation strategies}

### Caching Strategy

{If applicable, what can be cached and for how long}

## File Changes

### New Files

| File Path | Description |
|-----------|-------------|
| `registry/routes/new_feature.py` | API routes for new feature |
| `registry/services/new_feature_service.py` | Business logic |
| `tests/unit/test_new_feature.py` | Unit tests |

### Modified Files

| File Path | Lines | Change Description |
|-----------|-------|-------------------|
| `registry/main.py` | ~50 | Add router import and include |
| `registry/models/domain.py` | ~100-150 | Add new model |

### API Client Updates (IMPORTANT)

**When new API endpoints are added or modified, the following files MUST also be updated:**

| File Path | Update Required |
|-----------|-----------------|
| `api/registry_client.py` | Add Pydantic response models and client methods for new endpoints |
| `api/registry_management.py` | Add CLI commands (argparse parsers) and handler functions |
| `api/openapi.json` | Regenerate OpenAPI spec if using auto-generation |

**Example additions for a new endpoint `GET /api/feature/{path}/data`:**

```python
# In api/registry_client.py:
class FeatureDataResponse(BaseModel):
    """Response model for feature data endpoint."""
    path: str = Field(..., description="Feature path")
    data: dict = Field(..., description="Feature data")

def get_feature_data(self, path: str) -> FeatureDataResponse:
    """Get feature data from the registry."""
    response = self._make_request("GET", f"/api/feature/{path}/data")
    return FeatureDataResponse(**response)

# In api/registry_management.py:
def cmd_feature_data(args: argparse.Namespace) -> None:
    """Get feature data command handler."""
    client = _get_client(args)
    result = client.get_feature_data(path=args.path)
    # Display result...
```

This ensures the registry management CLI stays in sync with backend API capabilities.

### Estimated Lines of Code

| Category | Lines |
|----------|-------|
| New Python code | ~{X} |
| New test code | ~{X} |
| Modified code | ~{X} |
| **Total** | **~{X}** |

## Testing Strategy

**IMPORTANT:** Explicitly list all new test files and test cases required for this feature.

### New Test Files

| Test File | Type | Description |
|-----------|------|-------------|
| `tests/unit/services/test_feature_service.py` | Unit | Service layer unit tests |
| `tests/unit/repositories/test_feature_repository.py` | Unit | Repository unit tests |
| `tests/integration/test_feature_integration.py` | Integration | End-to-end integration tests |
| `tests/unit/api/test_feature_routes.py` | Unit | API route unit tests |

### Unit Tests Required

List specific unit test cases:

| Test Case | File | What It Tests |
|-----------|------|---------------|
| `test_happy_path` | `test_feature_service.py` | Normal operation with valid inputs |
| `test_error_handling` | `test_feature_service.py` | Error cases and edge conditions |
| `test_validation` | `test_feature_service.py` | Input validation logic |

Example unit test structure:

```python
class TestNewFeature:
    """Tests for new feature."""

    def test_happy_path(self):
        """Test normal operation."""
        # Arrange
        input_data = {...}

        # Act
        result = function_under_test(input_data)

        # Assert
        assert result["status"] == "success"

    def test_error_case(self):
        """Test error handling."""
        with pytest.raises(ValueError):
            function_under_test(None)
```

### Integration Tests Required

List specific integration test cases:

| Test Case | File | What It Tests |
|-----------|------|---------------|
| `test_end_to_end_flow` | `test_feature_integration.py` | Complete feature workflow |
| `test_database_persistence` | `test_feature_integration.py` | Data is correctly persisted |
| `test_concurrent_operations` | `test_feature_integration.py` | Concurrency handling |

### Test Coverage Requirements

- Minimum coverage for new code: 80%
- All public methods must have tests
- Error paths must be tested

### Manual Testing

{Steps for manual verification}

## Alternatives Considered

**IMPORTANT:** Document alternative approaches that were considered and why they were rejected.

### Alternative 1: {Alternative Approach Name}

**Description:** {Brief description of the alternative}

**Pros:**
- {Advantage 1}
- {Advantage 2}

**Cons:**
- {Disadvantage 1}
- {Disadvantage 2}

**Why Rejected:** {Clear explanation of why this wasn't chosen}

### Alternative 2: {Another Alternative}

**Description:** {Brief description}

**Pros:**
- {Advantage}

**Cons:**
- {Disadvantage}

**Why Rejected:** {Explanation}

### Comparison Matrix

| Criteria | Chosen Approach | Alternative 1 | Alternative 2 |
|----------|-----------------|---------------|---------------|
| Complexity | Low | Medium | High |
| Performance | Good | Better | Best |
| Maintainability | High | Medium | Low |
| Implementation Time | 1 week | 2 weeks | 3 weeks |

## Rollout Plan

### Phase 1: Development
- [ ] Implement core functionality
- [ ] Write unit tests
- [ ] Code review

### Phase 2: Testing
- [ ] Integration testing
- [ ] Security review
- [ ] Performance testing (if applicable)

### Phase 3: Deployment
- [ ] Deploy to staging
- [ ] Verify in staging
- [ ] Deploy to production
- [ ] Monitor for issues

## Open Questions

- {Any unresolved questions that need answers}

## References

- {Links to relevant documentation}
- {Links to similar implementations}
```

## Step 7: Expert Review (review.md)

Create a review document with feedback from multiple expert personas:

```markdown
# Expert Review: {Feature Name}

*Review Date: {date}*
*LLD Version: 1.0*

## Review Panel

| Role | Reviewer | Status |
|------|----------|--------|
| Frontend Engineer | Pixel | Pending |
| Backend Engineer | Byte | Pending |
| SRE/DevOps Engineer | Circuit | Pending |
| Security Engineer | Cipher | Pending |
| SMTS (Overall) | Sage | Pending |

---

## Frontend Engineer Review

**Reviewer:** Pixel
**Focus Areas:** UI/UX, React components, state management, API integration

### Assessment

#### Strengths
- {Positive aspects of the design from frontend perspective}

#### Concerns
- {Issues or risks identified}

#### New Libraries Required

| Library | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| `library-name` | `latest` | {What it does} | {Why it's needed vs alternatives} |
| None | - | - | No new frontend dependencies required |

#### Better Alternatives Considered

{Are there better frontend approaches? Discuss alternatives like different state management, component libraries, etc.}

#### Recommendations
1. {Specific recommendation}
2. {Specific recommendation}

#### Questions for Author
- {Questions that need clarification}

### Verdict: {APPROVED / APPROVED WITH CHANGES / NEEDS REVISION}

---

## Backend Engineer Review

**Reviewer:** Byte
**Focus Areas:** API design, data models, business logic, database queries, performance

### Assessment

#### Strengths
- {Positive aspects from backend perspective}

#### Concerns
- {Issues or risks identified}

#### New Libraries Required

| Library | Version | Purpose | Justification |
|---------|---------|---------|---------------|
| `library-name` | `latest` | {What it does} | {Why it's needed vs alternatives} |
| None | - | - | No new backend dependencies required |

#### Better Alternatives Considered

{Are there better backend approaches? Discuss alternatives like different algorithms, data structures, caching strategies, etc.}

#### Recommendations
1. {Specific recommendation}
2. {Specific recommendation}

#### Questions for Author
- {Questions that need clarification}

### Verdict: {APPROVED / APPROVED WITH CHANGES / NEEDS REVISION}

---

## SRE/DevOps Engineer Review

**Reviewer:** Circuit
**Focus Areas:** Deployment, monitoring, scaling, infrastructure, reliability

### Assessment

#### Strengths
- {Positive aspects from SRE perspective}

#### Concerns
- {Issues or risks identified}

#### Infrastructure Dependencies

| Resource | Type | Purpose | Cost Impact |
|----------|------|---------|-------------|
| `resource-name` | AWS Service / Tool | {What it does} | {Estimated cost} |
| None | - | - | No new infrastructure required |

#### Better Alternatives Considered

{Are there better infrastructure approaches? Discuss alternatives like different AWS services, deployment strategies, monitoring tools, etc.}

#### Recommendations
1. {Specific recommendation}
2. {Specific recommendation}

#### Operational Checklist
- [ ] Monitoring/alerting considered
- [ ] Logging sufficient for debugging
- [ ] Graceful degradation planned
- [ ] Rollback strategy defined
- [ ] Resource requirements estimated

### Verdict: {APPROVED / APPROVED WITH CHANGES / NEEDS REVISION}

---

## Security Engineer Review

**Reviewer:** Cipher
**Focus Areas:** Authentication, authorization, input validation, data protection, OWASP

### Assessment

#### Strengths
- {Positive aspects from security perspective}

#### Concerns
- {Issues or risks identified}

#### Security Checklist
- [ ] Input validation adequate
- [ ] Authentication/authorization correct
- [ ] No sensitive data exposure
- [ ] No injection vulnerabilities
- [ ] Rate limiting considered
- [ ] Audit logging included

#### Recommendations
1. {Specific recommendation}
2. {Specific recommendation}

### Verdict: {APPROVED / APPROVED WITH CHANGES / NEEDS REVISION}

---

## SMTS Overall Review

**Reviewer:** Sage
**Focus Areas:** Architecture, code quality, maintainability, alignment with project goals

### Executive Summary

{2-3 sentence summary of the design and overall assessment}

### Architecture Assessment

- **Alignment with existing patterns:** {Good/Needs Work}
- **Maintainability:** {Good/Needs Work}
- **Scalability:** {Good/Needs Work}
- **Testability:** {Good/Needs Work}

### Cross-Cutting Concerns

{Issues that span multiple review areas}

### Final Recommendations

1. **Must Fix (Blockers):**
   - {Critical issues that must be addressed}

2. **Should Fix (Important):**
   - {Important issues to address before implementation}

3. **Consider (Nice to Have):**
   - {Suggestions for improvement}

### Overall Verdict: {APPROVED / APPROVED WITH CHANGES / NEEDS REVISION}

---

## Review Summary

| Reviewer | Verdict | Blockers | Key Concerns |
|----------|---------|----------|--------------|
| Frontend | {verdict} | {count} | {summary} |
| Backend | {verdict} | {count} | {summary} |
| SRE/DevOps | {verdict} | {count} | {summary} |
| Security | {verdict} | {count} | {summary} |
| SMTS | {verdict} | {count} | {summary} |

### Next Steps

1. {Action item}
2. {Action item}
3. {Action item}

### Sign-Off

- [ ] All blockers resolved
- [ ] Design updated based on feedback
- [ ] Ready for implementation
```

## Important Guidelines

### Design Principles
- Favor simple designs over unnecessary complexity
- Prefer straightforward code over clever solutions
- Design for maintainability by entry-level developers
- Add observability from the start, not as an afterthought

### Documentation Quality
1. **Be Thorough**: The LLD should be detailed enough that someone unfamiliar with the codebase can implement it
2. **Use Diagrams**: ASCII diagrams help visualize the design
3. **Include Code**: Show actual or pseudo-code for key functions
4. **Specify Files**: Always mention which files to create/modify and approximate line numbers
5. **Consider All Aspects**: Think about error handling, logging, testing, and deployment
6. **Expert Reviews**: Make the reviews realistic - identify actual issues, not just praise

## Example Usage

### Example 1: User Description Mode

User: "Design a new feature for rate limiting on tool calls"

1. Ask clarifying questions about rate limiting requirements
2. Quick codebase review to understand existing architecture
3. Create `.scratchpad/rate-limiting/`
4. Write `github-issue.md` with rate limiting requirements
5. Deep codebase analysis of relevant services
6. Write `lld.md` with:
   - Architecture showing rate limiter component
   - Sequence diagram for rate-limited requests
   - Redis/in-memory counter data structures
   - API changes for rate limit headers
   - Middleware implementation details
   - Configuration options
7. Write `review.md` with expert feedback on:
   - Frontend: How to display rate limit info to users
   - Backend: Algorithm choice, storage considerations
   - SRE: Redis availability, monitoring needs
   - Security: Rate limit bypass prevention
   - SMTS: Overall architecture fit
8. Write `testing.md` with:
   - Functional curl tests for the rate limit headers endpoint
   - `registry_management.py` tests for any new CLI flags
   - Backwards-compat tests confirming existing endpoints still respond identically when under limit
   - UX tests for surfacing rate limit feedback in the UI
   - ECS/terraform tests for any new env vars (e.g. `RATE_LIMIT_REDIS_URL`)
   - E2E test exercising multiple requests to trigger and recover from a rate-limited state
9. Present summary and seek guidance on recommendations

### Example 2: GitHub Issue URL Mode

User: "https://github.com/agentic-community/mcp-gateway-registry/issues/456"

1. Fetch issue #456 using `gh issue view 456 --repo agentic-community/mcp-gateway-registry --json title,body,labels,state,comments`
2. Analyze the issue content:
   - Extract: "Add support for federated registry syncing"
   - Found: Problem statement, some acceptance criteria
   - Missing: Specific sync frequency, conflict resolution strategy
3. Quick codebase review to understand registry architecture
4. Create `.scratchpad/issue-456/` (folder named after the GitHub issue number)
5. Ask user only about gaps:
   - "The issue mentions syncing but doesn't specify frequency. What sync interval should we design for?"
   - "The issue discusses conflicts but doesn't specify resolution. Should we use last-write-wins or require manual resolution?"
6. Write `github-issue.md` summarizing the issue with clarifications
7. Deep codebase analysis of registry and storage layers
8. Write `lld.md` incorporating:
   - Requirements from original issue
   - Design elements proposed in issue comments
   - Clarifications from user
   - Technical details for implementation
9. Write `review.md` with expert feedback
10. Write `testing.md` with:
    - Functional curl tests for new sync endpoints and existing endpoints that now honor sync state
    - `registry_management.py` tests for any new sync-related CLI commands
    - Backwards-compat tests confirming non-federated deployments are unaffected
    - UX tests for any UI indicators showing federated/synced resources
    - ECS/terraform tests for new sync config vars (e.g. `FEDERATION_PEER_URLS`, `FEDERATION_SYNC_INTERVAL_SECONDS`)
    - E2E test exercising a two-registry sync cycle and conflict resolution
11. Present summary noting:
    - What came from the original issue
    - What was clarified during design
    - Recommendations for implementation

## Step 8: Write Testing Plan (testing.md)

Create a comprehensive testing plan document that captures **executable, copy-pasteable tests** covering every externally observable change introduced by the feature. The goal is for a reviewer or implementer to be able to walk through this document and verify the feature works end-to-end without having to invent test cases.

### When Each Test Category Applies

Include a category only when it is relevant to the feature. Mark a category "Not Applicable" with a short justification when it does not apply.

| Category | Include When |
|----------|--------------|
| Functional Tests (curl) | Feature adds/modifies any HTTP endpoint |
| Functional Tests (registry_management.py) | Feature adds/modifies anything exposed via the management CLI |
| Backwards Compatibility Tests | Feature changes an existing endpoint, schema, CLI command, config default, or data model |
| UX Tests | Feature adds/changes any UI surface (web UI, CLI output formatting, error messages shown to users) |
| ECS / Terraform Deployment Tests | Feature adds or modifies **any** config parameter (env var, setting, Terraform variable, secret) that must flow through the ECS/terraform deployment in `terraform/aws-ecs/` |
| E2E API Tests | Feature adds a new user-visible workflow that spans multiple endpoints or services |

### Testing Plan Template

Use this template verbatim (fill in the placeholders, remove sections marked "Not Applicable" except to record the justification):

```markdown
# Testing Plan: {Feature Name}

*Created: {date}*
*Related LLD: `./lld.md`*
*Related Issue: `./github-issue.md`*

## Overview

### Scope of Testing
{1-2 sentences describing what is being tested and why}

### Test Environments

| Environment | Purpose | How to Reach |
|-------------|---------|--------------|
| Local (docker-compose) | Primary dev verification | `docker-compose up -d` at repo root |
| AWS ECS (terraform) | Staging/prod parity | `terraform/aws-ecs/` deployment |

### Prerequisites

- [ ] Registry and Keycloak running (`docker ps | grep -E 'registry\|keycloak'`)
- [ ] Auth tokens generated: `cd credentials-provider && ./generate_creds.sh`
- [ ] Token file exists: `ls -la .oauth-tokens/ingress.json`
- [ ] {Any feature-specific prereqs, e.g. MongoDB seeded with X}

### Shared Variables

```bash
export REGISTRY_URL="http://localhost"             # or https://registry.<region>.<domain>
export TOKEN_FILE=".oauth-tokens/ingress.json"
export ACCESS_TOKEN=$(jq -r '.access_token' "$TOKEN_FILE")
```

---

## 1. Functional Tests

### 1.1 curl Tests (HTTP API)

For each new or modified endpoint, provide a copy-pasteable curl command, the expected response, and the assertion being verified.

#### Test 1.1.1: {Endpoint description, e.g. "Create resource X"}

**Endpoint:** `POST /api/v1/{path}`

**Command:**
```bash
curl -sS -X POST "$REGISTRY_URL/api/v1/{path}" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "field1": "value1",
        "field2": 42
      }' \
  | jq
```

**Expected Status:** `200 OK` (or `201 Created`)

**Expected Response:**
```json
{
  "id": "{id}",
  "status": "success"
}
```

**Assertions:**
- [ ] Response contains `id` field
- [ ] `status == "success"`
- [ ] {Any side effect to verify, e.g. record exists in DB}

**Negative Case:**
```bash
# Missing required field should return 400
curl -sS -X POST "$REGISTRY_URL/api/v1/{path}" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{}' -w "\n%{http_code}\n"
```
Expected: `400` with validation error body.

#### Test 1.1.2: {Next endpoint}
{Repeat the structure above for each endpoint}

### 1.2 registry_management.py CLI Tests

For each new or modified CLI command exposed via `api/registry_management.py`, provide the exact invocation and expected output.

#### Test 1.2.1: {Command description, e.g. "Register a new feature resource"}

**Command:**
```bash
cd api
uv run python registry_management.py {subcommand} \
  --token-file "../$TOKEN_FILE" \
  --registry-url "$REGISTRY_URL" \
  --{new-param} "{value}"
```

**Expected Output (key fragments):**
```
Successfully created {resource}: <id>
```

**Assertions:**
- [ ] Exit code is `0`
- [ ] Output includes the created resource id
- [ ] Follow-up `list` command shows the resource

**Follow-up Verification:**
```bash
uv run python registry_management.py list \
  --token-file "../$TOKEN_FILE" \
  --registry-url "$REGISTRY_URL"
```

#### Test 1.2.2: {Next command}
{Repeat the structure above}

---

## 2. Backwards Compatibility Tests

*Include this section if the feature touches any existing endpoint, schema, CLI command, default value, or data model. Otherwise replace with: "**Not Applicable** - feature introduces only net-new surface area: {justification}".*

### 2.1 Existing API Contract

Goal: confirm that existing clients continue to work without code changes.

#### Test 2.1.1: Legacy request shape still accepted

**Command (pre-feature request body):**
```bash
curl -sS -X POST "$REGISTRY_URL/api/v1/{existing-endpoint}" \
  -H "Authorization: Bearer $ACCESS_TOKEN" \
  -H "Content-Type: application/json" \
  -d '{
        "legacy_field": "value"
      }' | jq
```

**Assertions:**
- [ ] Returns `200` (not `400`)
- [ ] Response contains all previously documented fields
- [ ] New optional fields either absent or have sensible defaults

### 2.2 Existing CLI Behavior

#### Test 2.2.1: CLI command without new flags behaves as before

```bash
uv run python registry_management.py {existing-command} \
  --token-file "../$TOKEN_FILE" \
  --registry-url "$REGISTRY_URL"
```

**Assertions:**
- [ ] Exit code `0`
- [ ] Output format unchanged (diff against pre-change baseline if available)

### 2.3 Default Behavior When New Config Unset

- [ ] Feature disabled by default (or matches prior behavior) when new env vars are unset
- [ ] No new required config parameter breaks existing deployments

---

## 3. UX Tests

*Include this section if the feature changes any user-facing surface (web UI, CLI output, error messages). Otherwise replace with: "**Not Applicable** - feature is internal only, with no UX changes."*

### 3.1 Web UI Tests

For each UI change, describe the manual test steps.

#### Test 3.1.1: {UI flow, e.g. "Display new resource in listing"}

**Steps:**
1. Log in to the registry UI at `$REGISTRY_URL`
2. Navigate to {page}
3. Verify {visible element} renders correctly
4. {Action, e.g. click "Create"}
5. Verify {expected outcome}

**Assertions:**
- [ ] Element visible at correct position
- [ ] No console errors in browser devtools
- [ ] Responsive behavior at mobile width works
- [ ] Accessible: keyboard navigation and screen-reader labels present

### 3.2 CLI Output / Error Message Tests

#### Test 3.2.1: Error message is actionable

```bash
uv run python registry_management.py {command} --invalid-flag bad-value
```

**Assertions:**
- [ ] Error clearly states what was invalid
- [ ] Error suggests the correct usage
- [ ] No stack trace leaked to user

---

## 4. ECS / Terraform Deployment Tests

*Include this section whenever the feature adds or modifies ANY config parameter (env var, setting, Terraform variable, secret). Otherwise replace with: "**Not Applicable** - feature introduces no new config parameters. Verified by reviewing diff of `registry/core/config.py` and `terraform/aws-ecs/`."*

### 4.1 Terraform Variable Wiring

For each new config parameter, confirm it is plumbed through the ECS deployment.

| Config Parameter | Env Var Name | Added To | Verification Command |
|------------------|--------------|----------|----------------------|
| `feature_enabled` | `FEATURE_ENABLED` | `terraform/aws-ecs/ecs.tf` (task definition env block) | `grep FEATURE_ENABLED terraform/aws-ecs/*.tf` |
| `feature_interval_seconds` | `FEATURE_INTERVAL_SECONDS` | `terraform/aws-ecs/variables.tf` + `ecs.tf` | `grep -n FEATURE_INTERVAL_SECONDS terraform/aws-ecs/` |

### 4.2 Terraform Plan/Apply Verification

```bash
cd terraform/aws-ecs
terraform init
terraform validate
terraform plan -var 'feature_enabled=true' -var 'feature_interval_seconds=60'
```

**Assertions:**
- [ ] `terraform validate` passes
- [ ] `terraform plan` shows the new variable in the ECS task definition env block
- [ ] No unintended changes to unrelated resources

### 4.3 Deploy and Verify on ECS

```bash
cd terraform/aws-ecs
terraform apply -var 'feature_enabled=true' -var 'feature_interval_seconds=60'
```

**Post-Deploy Assertions:**
- [ ] ECS service reaches `RUNNING` state
- [ ] CloudWatch logs show the feature initialized with the configured values
- [ ] Health check endpoint returns `200` against the deployed ALB/CloudFront URL
- [ ] Re-run the Functional Tests from section 1 against the deployed `$REGISTRY_URL`

### 4.4 Rollback Verification

- [ ] `terraform apply` with `feature_enabled=false` disables the feature cleanly
- [ ] Reverting to the previous task definition restores prior behavior (backwards compat holds in deployed env)

---

## 5. End-to-End API Tests

*Include this section if the feature adds a user-visible workflow that spans multiple endpoints or services. Otherwise replace with: "**Not Applicable** - feature is a single-endpoint change, covered by Functional Tests section 1."*

### 5.1 E2E Scenario: {Scenario Name}

**Goal:** {business outcome being exercised, e.g. "A tenant onboards, registers a server, invokes a tool, and tears down the resources"}

**Setup:**
```bash
# Generate token, set env vars (see Shared Variables above)
export RUN_ID=$(date +%s)
```

**Steps (each step is executable):**

1. **Create prerequisite resources**
   ```bash
   curl -sS -X POST "$REGISTRY_URL/api/v1/{pre-req}" \
     -H "Authorization: Bearer $ACCESS_TOKEN" \
     -d '{ "name": "e2e-'$RUN_ID'" }'
   ```

2. **Exercise the new feature**
   ```bash
   curl -sS -X POST "$REGISTRY_URL/api/v1/{new-endpoint}" \
     -H "Authorization: Bearer $ACCESS_TOKEN" \
     -d '{ "ref": "e2e-'$RUN_ID'" }'
   ```

3. **Verify cross-service side effect**
   ```bash
   curl -sS "$REGISTRY_URL/api/v1/{downstream-check}/e2e-$RUN_ID" | jq
   ```

4. **Teardown**
   ```bash
   curl -sS -X DELETE "$REGISTRY_URL/api/v1/{pre-req}/e2e-$RUN_ID"
   ```

**Assertions:**
- [ ] All steps return expected status codes
- [ ] Final state matches expected (resource visible in listing, metrics emitted, logs present)
- [ ] Teardown leaves the system in its original state

### 5.2 Reuse of Existing E2E Harness

If `api/test-management-api-e2e.sh` already exercises adjacent flows, extend it rather than duplicating:

- [ ] Add new test phase to `api/test-management-api-e2e.sh`
- [ ] Document additions in `api/test-management-api-e2e.md`
- [ ] Run full harness locally and against the ECS deployment

```bash
cd api
./test-management-api-e2e.sh --token-file ../$TOKEN_FILE --registry-url "$REGISTRY_URL"
```

---

## 6. Test Execution Checklist

Copy this checklist into the PR description when implementing the feature:

- [ ] Section 1 (Functional) - all curl tests pass against local deployment
- [ ] Section 1 (Functional) - all `registry_management.py` commands succeed
- [ ] Section 2 (Backwards Compat) - verified or marked Not Applicable with justification
- [ ] Section 3 (UX) - verified or marked Not Applicable with justification
- [ ] Section 4 (ECS/Terraform) - verified or marked Not Applicable with justification
- [ ] Section 5 (E2E) - verified or marked Not Applicable with justification
- [ ] Unit tests added in `tests/unit/`
- [ ] Integration tests added in `tests/integration/`
- [ ] `uv run pytest tests/ -n 8` passes with no regressions
```

### Guidance for Generating testing.md

1. **Make tests copy-pasteable.** Use the exact env var conventions (`REGISTRY_URL`, `TOKEN_FILE`, `ACCESS_TOKEN`) and match the style of `api/test-management-api-e2e.sh`.
2. **Cover every new endpoint and every new CLI command.** If the LLD adds three endpoints and two CLI commands, section 1 must have at least five subsections.
3. **Anchor ECS tests on concrete files.** Reference `terraform/aws-ecs/ecs.tf`, `variables.tf`, and any module under `terraform/aws-ecs/modules/` that the parameter flows through. Do not write generic "deploy and verify" wording - name the files.
4. **Mark Not Applicable explicitly.** Do not silently omit sections - always include the heading with a short justification, so reviewers see that the category was considered.
5. **Align with backwards-compat rules.** If the LLD introduces a schema change, the backwards-compat section must test the pre-change request/response shapes.
6. **Do not invent endpoints or flags.** Every curl URL, every `registry_management.py` flag, and every Terraform variable must exist in the LLD or the current codebase.


## Step 9: Present Summary & Seek Guidance

**IMPORTANT:** After completing the design documents and expert review, present a clear summary to the user and ask for guidance on addressing recommendations.

### Summary Format

Present the following information in a clear, tabular format:

```markdown
## Design Summary

### Documents Created

| Document | Location | Description |
|----------|----------|-------------|
| GitHub Issue | `.scratchpad/{feature}/github-issue.md` | Issue specification |
| Low-Level Design | `.scratchpad/{feature}/lld.md` | Technical design |
| Expert Review | `.scratchpad/{feature}/review.md` | Multi-persona review |
| Testing Plan | `.scratchpad/{feature}/testing.md` | Functional, backwards-compat, UX, ECS, and E2E tests |

### Review Verdicts

| Reviewer | Verdict | Blockers | Key Recommendations |
|----------|---------|----------|---------------------|
| Frontend (Pixel) | {verdict} | {count} | {brief summary} |
| Backend (Byte) | {verdict} | {count} | {brief summary} |
| SRE (Circuit) | {verdict} | {count} | {brief summary} |
| Security (Cipher) | {verdict} | {count} | {brief summary} |
| SMTS (Sage) | {verdict} | {count} | {brief summary} |

### Configuration Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `PARAM_NAME` | type | value | {description} |

### New Dependencies

| Package/Resource | Type | Required By |
|------------------|------|-------------|
| `package-name` | Python | Backend |
| None | - | No new dependencies |

### New Tests Required

| Test File | Type | Coverage |
|-----------|------|----------|
| `tests/unit/test_feature.py` | Unit | Service layer |
| `tests/integration/test_feature.py` | Integration | End-to-end |

### Estimated Effort

| Category | Lines of Code |
|----------|---------------|
| New code | ~{X} |
| Tests | ~{X} |
| Modified | ~{X} |
| **Total** | **~{X}** |
```

### Seeking Guidance

After presenting the summary, explicitly ask the user for guidance:

```markdown
## Action Required

### Blockers (Must Address)

The following issues were identified as **blockers** and should be addressed before implementation:

1. {Blocker description from review}
2. {Blocker description from review}

**These will be incorporated into the LLD.**

### Recommendations (Need Your Input)

The following recommendations were made by reviewers. Please indicate which ones to incorporate:

| # | Recommendation | Reviewer | Priority | Incorporate? |
|---|----------------|----------|----------|--------------|
| 1 | {recommendation} | Backend | Should Fix | ? |
| 2 | {recommendation} | SRE | Should Fix | ? |
| 3 | {recommendation} | Security | Nice to Have | ? |

### Questions for You

1. Should I update the LLD to address the blockers and your selected recommendations?
2. Are there any additional requirements or constraints I should consider?
3. Would you like me to proceed with creating the GitHub issue?
```

### Handling User Response

Based on user response:

1. **If user wants LLD updates**: Update the `lld.md` file with the agreed-upon changes
2. **If user approves as-is**: Proceed to offer GitHub issue creation
3. **If user has additional feedback**: Incorporate and regenerate affected sections

### Alternative Approaches Discussion

If reviewers identified better alternatives, explicitly call this out:

```markdown
### Alternative Approaches Identified

Reviewers suggested these alternative approaches that may be worth considering:

| Alternative | Proposed By | Trade-offs |
|-------------|-------------|------------|
| {approach} | {reviewer} | {pros/cons summary} |

Would you like me to explore any of these alternatives further?
```
