# Agent Skills Architecture Design

This document describes the architecture and data model for the Agent Skills feature in MCP Gateway Registry.

## Overview

Agent Skills are reusable, shareable instruction sets that augment AI coding assistants with specialized capabilities. Unlike MCP servers (which provide tools), skills provide context, workflows, and behavioral guidance that help AI assistants perform specific tasks more effectively.

The Agent Skills feature follows the [agentskills.io](https://agentskills.io) specification, providing a standardized way to discover, share, and manage skills across AI coding environments.

## Design Principles

### Separation of Concerns

Skills and Servers serve different purposes:

| Aspect | MCP Servers | Agent Skills |
|--------|------------|--------------|
| Primary Function | Provide executable tools | Provide behavioral guidance |
| Content Type | Code, APIs, integrations | Markdown instructions, workflows |
| Execution | Server-side execution | Client-side interpretation |
| State | Stateful (running processes) | Stateless (document-based) |

### URL-Based Discovery

Skills are referenced by a single URL pointing to a `SKILL.md` file:

```
https://github.com/org/repo/blob/main/skills/pdf-processing/SKILL.md
```

The registry:
1. Accepts the user-provided URL (blob URL for GitHub)
2. Auto-translates to raw content URL for fetching
3. Stores both URLs for different use cases

### Progressive Disclosure

Skills support multiple detail tiers to avoid overwhelming AI assistants:

1. **Card View**: Name, description, tags (for discovery)
2. **Summary View**: Plus requirements, tools, target agents
3. **Full View**: Complete SKILL.md content with all details

## Data Model

### SkillCard Entity

The primary entity representing a registered skill:

```
SkillCard
├── Identification
│   ├── path: /skills/{name}         # Unique, immutable path
│   ├── name: string                 # Lowercase alphanumeric with hyphens
│   └── description: string          # What the skill does
│
├── URLs
│   ├── skill_md_url: HttpUrl        # User-provided URL (e.g., GitHub blob)
│   ├── skill_md_raw_url: HttpUrl    # Auto-translated raw content URL
│   └── repository_url: HttpUrl      # Optional git repository
│
├── Metadata
│   ├── version: string              # Skill version
│   ├── author: string               # Skill author
│   ├── license: string              # License identifier
│   ├── compatibility: string        # Human-readable requirements
│   └── tags: string[]               # Categorization tags
│
├── Requirements
│   ├── requirements: CompatibilityRequirement[]  # Machine-readable
│   ├── target_agents: string[]      # Target AI assistants
│   └── allowed_tools: ToolReference[]  # Required MCP tools
│
├── Access Control
│   ├── visibility: public|private|group
│   ├── allowed_groups: string[]     # For group visibility
│   └── owner: string                # For private visibility
│
├── State
│   ├── is_enabled: boolean          # Enable/disable toggle
│   ├── registry_name: string        # Source registry (for federation)
│   ├── health_status: healthy|unhealthy|unknown
│   └── last_checked_time: datetime  # Last health check
│
├── Ratings
│   ├── num_stars: float             # Average rating (0-5)
│   └── rating_details: RatingDetail[]  # Individual ratings
│
└── Timestamps
    ├── created_at: datetime
    └── updated_at: datetime
```

### ToolReference

Links skills to required MCP server tools:

```python
class ToolReference:
    tool_name: str           # Tool name (e.g., "Read", "Bash")
    server_path: str | None  # MCP server path (e.g., "/servers/claude-tools")
    version: str | None      # Optional version constraint
    capabilities: list[str]  # Capability filters (e.g., ["git:*"])
```

### CompatibilityRequirement

Machine-readable compatibility constraints:

```python
class CompatibilityRequirement:
    type: "product" | "tool" | "api" | "environment"
    target: str              # Target identifier
    min_version: str | None  # Minimum version
    max_version: str | None  # Maximum version
    required: bool           # False = optional enhancement
```

## URL Translation

The registry automatically translates user-friendly URLs to raw content URLs:

### GitHub Translation

```
Input:  https://github.com/org/repo/blob/main/skills/name/SKILL.md
Output: https://raw.githubusercontent.com/org/repo/main/skills/name/SKILL.md
```

### GitLab Translation

```
Input:  https://gitlab.com/org/repo/-/blob/main/skills/name/SKILL.md
Output: https://gitlab.com/org/repo/-/raw/main/skills/name/SKILL.md
```

### Bitbucket Translation

```
Input:  https://bitbucket.org/org/repo/src/main/skills/name/SKILL.md
Output: https://bitbucket.org/org/repo/raw/main/skills/name/SKILL.md
```

## Access Control

Skills support three visibility levels:

### Public Skills
- Visible to all authenticated users
- Discoverable via search and listing
- Exportable via federation

### Private Skills
- Visible only to the owner
- Not discoverable by others
- Not exportable via federation

### Group Skills
- Visible to members of specified groups
- Groups are managed via IdP integration (Entra ID, Cognito, etc.)
- Requires `allowed_groups` to be specified

## Health Checking

Skills are health-checked by verifying SKILL.md accessibility:

1. **HEAD Request**: Verify the raw URL is accessible, retrying with a GET whose body is never read when the server answers HEAD with 405 or 501
2. **Status Codes**: 2xx = healthy, others = unhealthy in the response
3. **Trusted Domains**: Only allowed domains are checked (SSRF protection)
4. **Storage**: The result and `last_checked_time` are stored on the skill record in the skills collection, but only when the result is conclusive: reachable, 404 or 410, or blocked by the SSRF guard. Rate limits, other error statuses and timeouts are returned to the caller and not stored, because any user who can view a skill can run the check
5. **Inline Content**: A skill with `skill_md_content` (AgentCore federation) is healthy without a request, since the registry serves its content from the database. The health check and the content route both record it healthy
6. **Content Fetches**: `GET /api/skills/{path}/content` records a 404 or 410 from its SKILL.md fetch as unhealthy and a success (including serving inline content) as healthy, writing only when the status changes; transient failures leave health unchanged, so a caller cannot flip a skill by hammering the route

Semantic search reads `health_status` and `last_checked_time` from the skills collection for each matched skill. The search index does not store health. If that read fails, search still returns its results and reports every skill as `unknown` rather than failing the request.

### Trusted Domains

```python
TRUSTED_DOMAINS = [
    "raw.githubusercontent.com",
    "gitlab.com",
    "bitbucket.org",
    "gist.githubusercontent.com",
]
```

## Rating System

Skills use the same rating system as servers and agents:

1. **Star Rating**: 1-5 stars
2. **Per-User**: One rating per user per skill
3. **Updates**: Users can update their rating
4. **Average**: Displayed as average of all ratings

## Tool Validation

Skills can reference MCP server tools. The registry validates tool availability:

1. **Check Registration**: Verify referenced servers exist
2. **Check Tools**: Verify tools are exposed by servers
3. **Report Status**: Return availability status per tool

## API Endpoints

### Skill Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/skills` | List skills (with visibility filtering) |
| GET | `/api/skills/{path}` | Get skill details |
| POST | `/api/skills` | Register new skill |
| PUT | `/api/skills/{path}` | Update skill |
| DELETE | `/api/skills/{path}` | Delete skill |

### Skill State

| Method | Endpoint | Description |
|--------|----------|-------------|
| PUT | `/api/skills/{path}/enable` | Enable skill |
| PUT | `/api/skills/{path}/disable` | Disable skill |
| GET | `/api/skills/{path}/health` | Check skill health |

### Skill Content

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/skills/{path}/content` | Fetch SKILL.md content |
| GET | `/api/skills/{path}/tools` | Check tool availability |

### Ratings

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/skills/{path}/rating` | Get rating info |
| POST | `/api/skills/{path}/rate` | Submit/update rating |

## Database Schema

Skills are stored in MongoDB/DocumentDB with the following indexes:

```javascript
// Unique index on name
db.agent_skills.createIndex({ "name": 1 }, { unique: true })

// Tags for filtering
db.agent_skills.createIndex({ "tags": 1 })

// Visibility for access control
db.agent_skills.createIndex({ "visibility": 1 })

// Registry name for federation
db.agent_skills.createIndex({ "registry_name": 1 })

// Owner for private skills
db.agent_skills.createIndex({ "owner": 1 })

// Compound index for common queries
db.agent_skills.createIndex({
    "visibility": 1,
    "is_enabled": 1,
    "registry_name": 1
})
```

## Federation Support

Skills participate in peer-to-peer federation:

1. **Export**: Public skills are exported to peer registries
2. **Import**: Skills from peers are imported with `registry_name` set
3. **Sync Modes**: All, whitelist, or tag-based filtering
4. **Ownership**: Federated skills retain original registry attribution

## Future Considerations

### Content Caching
- Cache SKILL.md content to reduce external fetches
- Use `content_version` hash for cache invalidation
- Track `content_updated_at` for freshness

### Skill Bundles
- Group related skills into bundles
- Enable/disable bundles atomically
- Share bundle configurations

### Usage Analytics
- Track skill usage across clients
- Surface popular skills in discovery
- Enable skill recommendations

### Versioning
- Track skill version history
- Support rollback to previous versions
- Version-aware federation sync
