# Agent Skills Operational Guide

## Demo Video

https://github.com/user-attachments/assets/5d1f227a-25f8-480d-9ff9-acba2498844b

---

This guide covers registering, managing, and using Agent Skills in MCP Gateway Registry.

## Overview

Agent Skills are reusable instruction sets that enhance AI coding assistants with specialized workflows and behaviors. Skills are defined in SKILL.md files hosted on GitHub, GitLab, or Bitbucket, and registered in the MCP Gateway Registry for discovery and access control.

## Quick Start

### Prerequisites

- MCP Gateway Registry instance running
- Authenticated user account
- SKILL.md file hosted on GitHub, GitLab, or Bitbucket

### Step 1: Create a SKILL.md File

Create a SKILL.md file in your repository following the [agentskills.io](https://agentskills.io) specification:

```markdown
---
name: pdf-processing
description: Convert and manipulate PDF documents using various tools
---

# PDF Processing Skill

This skill helps you work with PDF documents including conversion, extraction, and manipulation.

## When to Use This Skill

- Converting documents to PDF format
- Extracting text or images from PDFs
- Merging or splitting PDF files
- Adding watermarks or annotations

## Workflow

1. Identify the PDF operation needed
2. Check for required tools (pdftk, poppler-utils)
3. Execute the appropriate command
4. Verify the output

## Examples

### Convert HTML to PDF
```bash
wkhtmltopdf input.html output.pdf
```

### Extract text from PDF
```bash
pdftotext document.pdf output.txt
```
```

### Step 2: Register the Skill

**Using the UI:**

1. Navigate to the Skills section in the dashboard
2. Click "Register Skill"
3. Enter the SKILL.md URL (e.g., `https://github.com/org/repo/blob/main/skills/pdf-processing/SKILL.md`)
4. Fill in additional details:
   - Name: Auto-populated from SKILL.md or enter manually
   - Description: Brief description of the skill
   - Visibility: Public, Private, or Group
   - Tags: Add relevant tags for discovery
5. Click "Register"

**Using the API:**

For API details, see the OpenAPI specification at [api/openapi.json](../api/openapi.json). Use the `registry_management.py` CLI for Python-based commands (see [CLI Commands](#cli-commands) section below).

### Step 3: Verify Registration

Check that the skill is registered and healthy using the CLI:

```bash
# Get skill details
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-get --path pdf-processing

# Check skill health
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-health --path pdf-processing
```

## Managing Skills

### List All Skills

**UI:** Navigate to the Skills section in the dashboard.

**CLI:**
```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-list
```

For custom curl commands with query parameters (e.g., `include_disabled`, `tag`), see [api/openapi.json](../api/openapi.json).

### Update a Skill

**UI:** Click the edit (pencil) icon on a skill card.

**API:** For update endpoints, see the OpenAPI specification at [api/openapi.json](../api/openapi.json).

### Enable/Disable Skills

**UI:** Use the toggle switch on the skill card.

**CLI:**
```bash
# Disable a skill
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-toggle --path pdf-processing --enabled false

# Enable a skill
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-toggle --path pdf-processing --enabled true
```

### Delete a Skill

**UI:** Click the delete (trash) icon on a skill card.

**CLI:**
```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-delete --path pdf-processing
```

## Health Monitoring

### Check Skill Health

The registry verifies that SKILL.md files are accessible:

**UI:** Click the refresh icon on a skill card to check health.

**CLI:**
```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-health --path pdf-processing
```

Response example:
```json
{
  "healthy": true,
  "status_code": 200,
  "checked_at": "2025-02-07T15:30:00Z"
}
```

### Health Status Indicators

| Status | Meaning |
|--------|---------|
| Healthy (green) | SKILL.md is accessible |
| Unhealthy (red) | SKILL.md fetch failed |
| Unknown (yellow) | Not yet checked |

## Rating Skills

### Submit a Rating

**UI:** Click the star rating widget on a skill card.

**CLI:**
```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-rate --path pdf-processing --rating 5
```

### View Ratings

**UI:** Rating is displayed on the skill card.

**CLI:**
```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-rating --path pdf-processing
```

Response example:
```json
{
  "num_stars": 4.5,
  "rating_details": [
    {"user": "alice", "rating": 5},
    {"user": "bob", "rating": 4}
  ]
}
```

## Viewing Skill Content

### View SKILL.md Content

**UI:** Click the info (i) icon on a skill card to open the content modal.

The modal displays:
- YAML frontmatter in a table format
- Formatted markdown content
- Links to GitHub source
- Copy and download buttons

**CLI:**
```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-content --path pdf-processing
```

Response example:
```json
{
  "content": "---\nname: pdf-processing\n...",
  "url": "https://raw.githubusercontent.com/org/repo/main/skills/pdf-processing/SKILL.md"
}
```

## Tool Validation

Skills can reference required MCP server tools. Validate tool availability:

**UI:** Click the wrench icon on a skill card.

**API:** For tool validation endpoints, see [api/openapi.json](../api/openapi.json).

Response example:
```json
{
  "all_available": true,
  "tool_results": [
    {
      "tool_name": "Bash",
      "server_path": "/servers/claude-tools",
      "available": true
    }
  ],
  "missing_tools": []
}
```

## Access Control

### Visibility Levels

| Level | Description |
|-------|-------------|
| Public | Visible to all authenticated users |
| Private | Visible only to the owner |
| Group | Visible to specified groups |

### Set Visibility

For visibility update endpoints, see [api/openapi.json](../api/openapi.json). Visibility options are:
- `public` - Visible to all authenticated users
- `private` - Visible only to the owner
- `group` - Visible to specified groups (requires `allowed_groups` parameter)

## Search and Discovery

### Search Skills

**UI:** Use the search bar in the Skills section.

**CLI:**
```bash
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-search --query "pdf"
```

For advanced search with multiple filters, see [api/openapi.json](../api/openapi.json).

## Integration with AI Assistants

### Claude Code Integration

Skills in Claude Code are stored as SKILL.md files in skill directories. To use a skill from the registry:

1. **Download the skill content** to your local skills directory:

```bash
# Global skills directory
mkdir -p ~/.claude/skills/pdf-processing
curl -H "Authorization: Bearer <token>" \
  https://your-registry.com/api/skills/pdf-processing/content \
  | jq -r '.content' > ~/.claude/skills/pdf-processing/SKILL.md

# Or project-level skills
mkdir -p .claude/skills/pdf-processing
curl -H "Authorization: Bearer <token>" \
  https://your-registry.com/api/skills/pdf-processing/content \
  | jq -r '.content' > .claude/skills/pdf-processing/SKILL.md
```

2. **Invoke the skill** using the slash command (folder name becomes the command):

```
/pdf-processing
```

See [Claude Code Skills Documentation](https://code.claude.com/docs/en/skills) for more details.

### Cursor Integration

Cursor uses Agent Skills stored in `.agents/skills/` directories. To use a skill from the registry:

1. **Download the skill content** to your project's skills directory:

```bash
mkdir -p .agents/skills/pdf-processing
curl -H "Authorization: Bearer <token>" \
  https://your-registry.com/api/skills/pdf-processing/content \
  | jq -r '.content' > .agents/skills/pdf-processing/SKILL.md
```

2. **Regenerate AGENTS.md** if using custom rules (required after adding new skills)

See [Cursor Agent Skills Documentation](https://cursor.com/docs/context/skills) for more details.

## Troubleshooting

### Skill Registration Fails

1. **Invalid URL**: Ensure the URL points to a valid SKILL.md file
2. **Name conflict**: Skill names must be unique
3. **Invalid name format**: Names must be lowercase alphanumeric with hyphens

### Skill Shows as Unhealthy

1. **Check URL**: Verify the SKILL.md file is accessible in a browser
2. **Repository access**: Ensure the repository is public or accessible
3. **Raw URL**: The registry uses raw URLs; verify raw content is accessible

### Rating Not Saved

1. **Authentication**: Ensure you're authenticated
2. **Valid range**: Ratings must be between 1 and 5
3. **Refresh**: Try refreshing the page after rating

### Content Not Loading

1. **CORS**: The registry proxies content to avoid CORS issues
2. **Health check**: Verify the skill is healthy first
3. **Network**: Check network connectivity to the source

## Best Practices

### Skill Naming

- Use lowercase letters and hyphens only
- Choose descriptive, specific names
- Avoid generic names like "helper" or "utils"

### SKILL.md Content

- Include clear trigger conditions
- Provide step-by-step workflows
- Add practical examples
- Document required tools

### Tagging

- Use consistent tag conventions
- Include category tags (e.g., "documents", "automation")
- Add technology tags (e.g., "pdf", "python")

### Visibility

- Start with private for testing
- Use group visibility for team-specific skills
- Make public for community sharing

## CLI Commands

The `registry_management.py` CLI provides commands for managing skills from the command line.

### Common Parameters

Global parameters must come **before** the subcommand:

| Parameter | Description |
|-----------|-------------|
| `--registry-url` | Registry base URL (default: http://localhost:8000) |
| `--token-file` | Path to JSON file containing access token |

### Register Skills from Anthropic Skills Repository

Register coding, documentation, and spreadsheet skills from the official [anthropics/skills](https://github.com/anthropics/skills) repository:

```bash
# Set common variables
REGISTRY_URL="https://your-registry.com"
TOKEN_FILE="/path/to/.token"

# Register doc-coauthoring skill (collaborative documentation)
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-register \
  --name doc-coauthoring \
  --url "https://github.com/anthropics/skills/blob/main/skills/doc-coauthoring/SKILL.md" \
  --description "Guide users through structured workflow for co-authoring documentation" \
  --tags docs,authoring,collaboration \
  --visibility public

# Register docx skill (Word document handling)
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-register \
  --name docx \
  --url "https://github.com/anthropics/skills/blob/main/skills/docx/SKILL.md" \
  --description "Create and manipulate Microsoft Word documents" \
  --tags docs,word,docx,documents \
  --visibility public

# Register xlsx skill (Excel spreadsheet handling)
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-register \
  --name xlsx \
  --url "https://github.com/anthropics/skills/blob/main/skills/xlsx/SKILL.md" \
  --description "Create and manipulate Excel spreadsheets" \
  --tags spreadsheet,excel,xlsx,data \
  --visibility public

# Register pdf skill (PDF document handling)
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-register \
  --name pdf \
  --url "https://github.com/anthropics/skills/blob/main/skills/pdf/SKILL.md" \
  --description "Create and manipulate PDF documents" \
  --tags pdf,documents,conversion \
  --visibility public

# Register mcp-builder skill (MCP server development)
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-register \
  --name mcp-builder \
  --url "https://github.com/anthropics/skills/blob/main/skills/mcp-builder/SKILL.md" \
  --description "Build MCP servers and tools for AI assistant integrations" \
  --tags mcp,coding,development,servers \
  --visibility public
```

### Other Skill CLI Commands

```bash
# List all skills
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-list

# Get skill details
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-get --path doc-coauthoring

# Check skill health
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-health --path doc-coauthoring

# Get SKILL.md content
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-content --path doc-coauthoring

# Search for skills
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-search --query "document"

# Toggle skill enabled/disabled
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-toggle --path doc-coauthoring --enabled false

# Rate a skill (1-5 stars)
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-rate --path doc-coauthoring --rating 5

# Get skill rating
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-rating --path doc-coauthoring

# Delete a skill
uv run python api/registry_management.py \
  --registry-url "$REGISTRY_URL" \
  --token-file "$TOKEN_FILE" \
  skill-delete --path doc-coauthoring
```

### Token File Format

The token file should be a JSON file with the following structure:

```json
{
  "tokens": {
    "access_token": "eyJ..."
  }
}
```

Or the simpler format:

```json
{
  "access_token": "eyJ..."
}
```

## API Reference

For complete API endpoint documentation, see:
- **OpenAPI Specification**: [api/openapi.json](../api/openapi.json) - Full API spec for writing custom curl commands
- **API Reference**: [API Reference](api-reference.md) - Human-readable endpoint documentation

## Related Documentation

- [Agent Skills Architecture](design/agent-skills-architecture.md)
- [Authentication](auth.md)
- [Federation](federation.md)
