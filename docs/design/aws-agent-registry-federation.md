# AWS Agent Registry Federation -- Design

This document describes the architecture and design decisions for federating Amazon Bedrock AgentCore registries into MCP Gateway Registry.

## Problem Statement

Organizations using Amazon Bedrock AgentCore publish MCP servers, A2A agents, and agent skills into AgentCore registries. These assets need to be discoverable alongside locally registered assets in the MCP Gateway Registry. The federation must support:

- Multiple registries (same account, cross-account, cross-region)
- Four descriptor types (MCP, A2A, CUSTOM, AGENT_SKILLS)
- Automatic sync with stale record cleanup
- Granular add/remove of individual registries from the UI
- Cascade cleanup when a registry is removed

## Architecture

### Component Overview

```
+-------------------------------+
|     MCP Gateway Registry      |
|                               |
|  +-------------------------+  |     +---------------------------+
|  | Federation Config (Mongo)| <---->| Settings UI               |
|  | aws_registry:            |  |     | (ExternalRegistries.tsx)  |
|  |   enabled: true          |  |     +---------------------------+
|  |   registries: [...]      |  |
|  +-------------------------+  |
|            |                  |
|            v                  |
|  +-------------------------+  |     +---------------------------+
|  | AgentCoreFederation     | ----->| AWS Bedrock AgentCore     |
|  | Client (boto3)          |  |     | bedrock-agentcore-control |
|  +-------------------------+  |     +---------------------------+
|            |                  |          |           |
|            v                  |     +---------+ +---------+
|  +-------------------------+  |     | Registry| | Registry|
|  | Server/Agent/Skill      |  |     | Acct A  | | Acct B  |
|  | Repositories (Mongo)    |  |     +---------+ +---------+
|  +-------------------------+  |
+-------------------------------+
```

### Data Flow

```
1. Startup / Manual Sync / Scheduled Sync
   |
   v
2. Load FederationConfig from MongoDB
   |
   v
3. For each registry in config.aws_registry.registries:
   |
   +---> 3a. Create boto3 client (optionally STS AssumeRole for cross-account)
   |
   +---> 3b. Paginate ListRegistryRecords (filtered by descriptor_types, sync_status_filter)
   |
   +---> 3c. Transform each record into MCP Gateway format
   |
   v
4. Register/update assets in MongoDB
   |
   v
5. Reconcile stale records (remove assets no longer in source)
```

## Data Model

### Federation Config (MongoDB: `mcp_federation_config_default`)

The federation config is a single document with `_id: "default"`. The `aws_registry` section stores all AgentCore federation settings:

```json
{
  "_id": "default",
  "aws_registry": {
    "enabled": true,
    "aws_region": "us-east-1",
    "sync_on_startup": true,
    "sync_interval_minutes": 60,
    "sync_timeout_seconds": 300,
    "max_concurrent_fetches": 5,
    "registries": [
      {
        "registry_id": "arn:aws:bedrock-agentcore:us-east-1:123456789012:registry/rXXX",
        "aws_account_id": "123456789012",
        "aws_region": "us-east-1",
        "assume_role_arn": null,
        "descriptor_types": ["MCP", "A2A", "CUSTOM", "AGENT_SKILLS"],
        "sync_status_filter": "APPROVED"
      }
    ]
  },
  "anthropic": { ... },
  "asor": { ... },
  "created_at": "2026-04-10T...",
  "updated_at": "2026-04-11T..."
}
```

### Synced Asset Tracking

Each synced asset carries metadata that links it back to its source registry. This enables cascade cleanup and prevents orphaned records.

**MCP Servers** (`mcp_servers_default`):
```json
{
  "path": "/agentcore-my-server",
  "source": "agentcore",
  "tags": ["agentcore", "bedrock", "federated", "mcp"],
  "metadata": {
    "agentcore_registry_id": "arn:aws:bedrock-agentcore:us-east-1:123456789012:registry/rXXX",
    "agentcore_record_id": "record-abc123",
    "agentcore_descriptor_type": "MCP"
  }
}
```

**A2A Agents** (`mcp_agents_default`):
```json
{
  "path": "/agents/agentcore-my-agent",
  "tags": ["agentcore", "bedrock", "federated", "a2a"],
  "metadata": {
    "agentcore_registry_id": "arn:aws:bedrock-agentcore:us-east-1:123456789012:registry/rXXX",
    "agentcore_record_id": "record-def456",
    "agentcore_descriptor_type": "A2A"
  }
}
```

**Agent Skills** (`agent_skills_default`):
```json
{
  "path": "/skills/agentcore-my-skill",
  "tags": ["agentcore", "bedrock", "federated", "skill"],
  "metadata": {
    "agentcore_registry_id": "arn:aws:bedrock-agentcore:us-east-1:123456789012:registry/rXXX",
    "agentcore_record_id": "record-ghi789",
    "agentcore_descriptor_type": "AGENT_SKILLS"
  }
}
```

## Key Design Decisions

### 1. Single Enable Flag via Environment Variable

**Decision**: Only `AWS_REGISTRY_FEDERATION_ENABLED` is an environment variable. All other settings (region, registries, sync behavior) are managed via the API/UI and stored in MongoDB.

**Rationale**:
- The enable flag is a deployment-level concern (should this instance support AgentCore federation at all?)
- Registry IDs, regions, and descriptor types are operational concerns that change at runtime
- Reduces env var sprawl -- previous design had 7 env vars, most of which were unused by the application

**Implementation**: `_apply_aws_registry_env_vars()` in `registry/main.py` reads the env var on startup and updates the MongoDB federation config before any sync runs.

### 2. Path-Based Naming Convention

**Decision**: Synced assets use a `agentcore-` prefix in their path.

| Asset Type | Path Pattern | Example |
|-----------|-------------|---------|
| MCP Server | `/agentcore-{name}` | `/agentcore-my-mcp-server` |
| A2A Agent | `/agents/agentcore-{name}` | `/agents/agentcore-travel-bot` |
| Agent Skill | `/skills/agentcore-{name}` | `/skills/agentcore-booking-skill` |

**Rationale**:
- Makes federated assets visually distinct in the UI and API
- Enables tag-based fallback matching for cascade cleanup (older records without metadata)
- Avoids path collisions with locally registered assets

### 3. Dual Matching Strategy for Cascade Cleanup

**Decision**: When a registry is removed, the cleanup logic uses two matching strategies:

1. **Primary**: Match by `metadata.agentcore_registry_id` (exact, reliable)
2. **Fallback**: Match by `"agentcore" in tags AND path.startswith("/type/agentcore-")` (for older records)

**Rationale**: Records synced before metadata tracking was added only have tags and path conventions. The fallback ensures these are cleaned up too. The fallback is conservative -- it requires both the tag and the path prefix to match.

### 4. Conditional IAM Policy Creation

**Decision**: The `bedrock_agentcore_access` IAM policy is only created when `aws_registry_federation_enabled = true` in Terraform.

```hcl
resource "aws_iam_policy" "bedrock_agentcore_access" {
  count = var.aws_registry_federation_enabled ? 1 : 0
  ...
}
```

**Rationale**: Follows the principle of least privilege. Deployments that don't use AgentCore federation don't get AgentCore IAM permissions. The policy uses `bedrock-agentcore:*` for simplicity since AgentCore is a new service with a limited action set, and all actions may be needed as the feature evolves.

### 5. Cross-Account Access via Per-Registry Role Assumption

**Decision**: Cross-account access is configured per-registry via `assume_role_arn`, not globally.

**Rationale**: Different registries may be in different accounts, each requiring a different IAM role. The STS AssumeRole call is scoped by a condition requiring `Purpose: agentcore-federation` tag on the target role, preventing the gateway from assuming arbitrary roles.

### 6. Backward Compatibility via Model Validator

**Decision**: A Pydantic `model_validator` transparently renames the old `agentcore` key to `aws_registry` when loading federation config from MongoDB.

```python
@model_validator(mode="before")
@classmethod
def _migrate_agentcore_key(cls, data: Any) -> Any:
    if isinstance(data, dict) and "agentcore" in data and "aws_registry" not in data:
        data["aws_registry"] = data.pop("agentcore")
    return data
```

**Rationale**: Avoids requiring a MongoDB migration script. Existing documents with the old key name deserialize correctly. New saves use the new key name, so documents are gradually migrated.

## API Endpoints

### Federation Config Management

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/api/federation/config/{config_id}` | Get full federation config |
| PUT | `/api/federation/config/{config_id}` | Update full federation config |
| POST | `/api/federation/config/{config_id}/aws_registry/registries` | Add a single registry |
| DELETE | `/api/federation/config/{config_id}/aws_registry/registries/{registry_id}` | Remove a registry (with cascade cleanup) |

### Sync

| Method | Endpoint | Description |
|--------|----------|-------------|
| POST | `/api/federation/sync` | Sync all enabled sources |
| POST | `/api/federation/sync?source=aws_registry` | Sync only AWS Agent Registry |

### Add Registry Request Body

```json
{
  "registry_id": "arn:aws:bedrock-agentcore:us-east-1:123456789012:registry/rXXX",
  "aws_account_id": "123456789012",
  "aws_region": "us-east-1",
  "assume_role_arn": "arn:aws:iam::999888777666:role/FederationReadOnly",
  "descriptor_types": ["MCP", "A2A", "CUSTOM", "AGENT_SKILLS"],
  "sync_status_filter": "APPROVED"
}
```

Only `registry_id` is required. All other fields are optional.

### Delete Registry Response

```json
{
  "message": "Registry removed and 3 server(s), 2 agent(s), 1 skill(s) deregistered",
  "deregistered": {
    "servers": ["/agentcore-my-server"],
    "agents": ["/agents/agentcore-my-agent"],
    "skills": ["/skills/agentcore-my-skill"]
  }
}
```

## Frontend Design

### External Registries Page

The External Registries settings page shows a card for each federation source (AWS Agent Registry, Anthropic, ASOR). Each card has:

- **Header**: Source name, enabled/disabled badge, Sync button, Add (+) button
- **Body**: List of configured entries with remove (X) buttons
- **Empty state**: "No registries configured" with an Add button

### Add Registry Modal

The `AddRegistryEntryModal` component renders different forms based on `sourceType`:

- `aws_registry`: Multi-field form (registry ID, account, region, role ARN, descriptor types, status filter)
- `anthropic`: Single field (server name)
- `asor`: Single field (agent ID)

ARN auto-population: When the user types or pastes a full ARN into the Registry ID field, the region and account ID fields are automatically populated by parsing the ARN structure (`arn:aws:bedrock-agentcore:<region>:<account_id>:registry/...`).

### Confirm Delete Modal

A styled `ConfirmModal` replaces the native browser `window.confirm()` dialog. It supports:
- Destructive (red) and normal (purple) button styles
- Loading state with "Removing..." text
- Warning icon with contextual coloring

## Reconciliation

### Stale Record Cleanup

After each sync, a reconciliation pass removes records that exist locally but are no longer present in the source registry:

1. Collect all paths synced in this run (`synced_paths`)
2. Query local repos for all agentcore-sourced records
3. Delete any record whose path is not in `synced_paths`

This ensures that records deleted from AgentCore are eventually removed from the gateway.

### Timing

- **On startup**: Runs after startup sync if `sync_on_startup: true`
- **On manual sync**: Runs after each sync triggered via API or UI
- **On registry removal**: Cascade cleanup runs immediately (does not wait for sync)

## Security Considerations

### IAM Permissions

The minimum IAM permissions for read-only federation:

```json
{
  "Action": [
    "bedrock-agentcore:ListRegistries",
    "bedrock-agentcore:ListRegistryRecords",
    "bedrock-agentcore:GetRegistryRecord"
  ],
  "Resource": "*"
}
```

The Terraform module uses `bedrock-agentcore:*` for operational flexibility.

### Cross-Account STS

The `sts:AssumeRole` permission is scoped by a condition:

```json
{
  "Condition": {
    "StringLike": {
      "iam:ResourceTag/Purpose": "agentcore-federation"
    }
  }
}
```

This prevents the gateway from assuming arbitrary IAM roles. Remote accounts must explicitly tag their federation role with `Purpose: agentcore-federation`.

### Authentication Chain

1. ECS task role provides base AWS credentials
2. For same-account registries: direct API calls using task role credentials
3. For cross-account registries: STS AssumeRole to get temporary credentials, then API calls

## File Map

| File | Purpose |
|------|---------|
| `registry/schemas/federation_schema.py` | Pydantic models for federation config |
| `registry/services/federation/agentcore_client.py` | boto3 client for AgentCore API |
| `registry/services/federation_reconciliation.py` | Stale record cleanup |
| `registry/api/federation_routes.py` | API endpoints (add/remove/sync) |
| `registry/main.py` | Startup sync and env var override |
| `frontend/src/components/ExternalRegistries.tsx` | Settings page UI |
| `frontend/src/components/AddRegistryEntryModal.tsx` | Add registry modal |
| `frontend/src/components/ConfirmModal.tsx` | Styled confirm dialog |
| `terraform/aws-ecs/modules/mcp-gateway/iam.tf` | AgentCore IAM policy |
| `terraform/aws-ecs/modules/mcp-gateway/ecs-services.tf` | ECS task definition env vars |
