# Federation Operational Guide

This guide covers setting up and operating peer-to-peer federation between MCP Gateway Registry instances.

## Demo

https://github.com/user-attachments/assets/630ce847-b151-4eaa-9cc9-2ec77797f2b5

## Quick Start

### Prerequisites

- Two or more MCP Gateway Registry instances running
- Network connectivity between registries (HTTPS)
- Admin access to both registries

### Step 1: Generate Encryption Key (One-Time)

On the importing registry, generate a Fernet encryption key for storing peer credentials:

```bash
python -c "from cryptography.fernet import Fernet; print(Fernet.generate_key().decode())"
```

Add to your `.env`:

```bash
FEDERATION_ENCRYPTION_KEY=<generated-key>
```

### Step 2: Configure the Exporting Registry

On the registry that will export data, enable federation static token auth:

```bash
# Generate a static token
python -c "import secrets; print(secrets.token_urlsafe(32))"

# Add to .env
FEDERATION_STATIC_TOKEN_AUTH_ENABLED=true
FEDERATION_STATIC_TOKEN=<generated-token>
```

Restart the registry for changes to take effect.

### Step 3: Add Peer Configuration

On the importing registry, add the peer using the UI or API:

**Using the UI:**

1. Navigate to Settings (gear icon in header)
2. Select Federation > Peers
3. Click "Add Peer"
4. Fill in the peer details:
   - Peer ID: A unique identifier (e.g., "lob-a")
   - Name: Human-readable name
   - Endpoint: Base URL of the peer registry
   - Federation Token: The token from Step 2
   - Sync Mode: Select how to filter synced items

**Using the API:**

```bash
curl -X POST https://your-registry.com/api/peers \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin-token>" \
  -d '{
    "peer_id": "lob-a",
    "name": "LOB-A Registry",
    "endpoint": "https://lob-a-registry.corp.com",
    "enabled": true,
    "sync_mode": "all",
    "sync_interval_minutes": 30,
    "federation_token": "<token-from-step-2>"
  }'
```

### Step 4: Set Visibility on Exportable Items

On the exporting registry, mark servers and agents for federation export:

```bash
# Mark a server as public (exportable)
curl -X PUT https://lob-a-registry.corp.com/api/servers/my-tool \
  -H "Authorization: Bearer <admin-token>" \
  -d '{"visibility": "public"}'
```

Or use the UI to edit server/agent settings and set visibility to "public".

### Step 5: Trigger Initial Sync

**Using the UI:**

1. Navigate to Settings > Federation > Peers
2. Click the sync icon next to the peer
3. View sync status and results

**Using the API:**

```bash
curl -X POST https://your-registry.com/api/peers/lob-a/sync \
  -H "Authorization: Bearer <admin-token>"
```

## Common Deployment Topologies

### Hub and Spoke

Central IT maintains a Hub that pulls from all LOB registries.

**Hub Configuration:**

```bash
# Hub .env
FEDERATION_ENCRYPTION_KEY=<key-for-encrypting-peer-tokens>
```

Add each LOB as a peer (UI or API).

**LOB Configuration:**

```bash
# Each LOB .env
FEDERATION_STATIC_TOKEN_AUTH_ENABLED=true
FEDERATION_STATIC_TOKEN=<unique-token-per-lob>
```

No peer configuration needed on LOBs (they only export).

### Bidirectional Sync

Two registries share items with each other.

**Registry A:**

```bash
# .env
FEDERATION_ENCRYPTION_KEY=<key-a>
FEDERATION_STATIC_TOKEN_AUTH_ENABLED=true
FEDERATION_STATIC_TOKEN=<token-a>
```

Add Registry B as a peer with its token.

**Registry B:**

```bash
# .env
FEDERATION_ENCRYPTION_KEY=<key-b>
FEDERATION_STATIC_TOKEN_AUTH_ENABLED=true
FEDERATION_STATIC_TOKEN=<token-b>
```

Add Registry A as a peer with its token.

### Mesh Topology

Multiple registries in a mesh where each can pull from any other.

Each registry:
1. Has its own `FEDERATION_STATIC_TOKEN` for others to pull from it
2. Has `FEDERATION_ENCRYPTION_KEY` to store peer tokens
3. Configures each other registry as a peer

## Sync Mode Configuration

### Sync All

Import all public servers and agents from the peer:

```json
{
  "peer_id": "lob-a",
  "sync_mode": "all"
}
```

### Whitelist Mode

Import only specific servers and agents:

```json
{
  "peer_id": "lob-a",
  "sync_mode": "whitelist",
  "whitelist_servers": ["/production-db", "/shared-api"],
  "whitelist_agents": ["/analytics-agent"]
}
```

### Tag Filter Mode

Import items with specific tags:

```json
{
  "peer_id": "lob-a",
  "sync_mode": "tag_filter",
  "tag_filters": ["production", "shared"]
}
```

## Scheduled Sync

Configure automatic sync at regular intervals:

```json
{
  "peer_id": "lob-a",
  "sync_interval_minutes": 30
}
```

Set to `0` for manual-only sync.

### How Scheduled Sync Works

The registry runs a background scheduler that:

1. **Checks every 60 seconds** for peers that need syncing
2. **Evaluates each enabled peer** with `sync_interval_minutes > 0`
3. **Triggers sync** when the time since `last_successful_sync` exceeds the configured interval
4. **Skips peers** that are disabled, have sync in progress, or have interval set to 0

The scheduler starts automatically when the registry starts and stops gracefully on shutdown.

### Viewing Scheduled Sync Activity

Check the registry logs for scheduled sync activity:

```bash
docker-compose logs registry | grep -i "scheduled sync"
```

Example log output:
```
Scheduled sync triggered for peer 'lob-a' (interval: 30m)
Scheduled sync completed for peer 'lob-a': 15 servers, 3 agents
```

## Managing Peers

### Enable/Disable a Peer

**UI:** Toggle the enabled switch in the peers list.

**API:**

```bash
# Enable
curl -X POST https://registry.com/api/peers/lob-a/enable \
  -H "Authorization: Bearer <token>"

# Disable
curl -X POST https://registry.com/api/peers/lob-a/disable \
  -H "Authorization: Bearer <token>"
```

### Update Peer Configuration

```bash
curl -X PUT https://registry.com/api/peers/lob-a \
  -H "Authorization: Bearer <token>" \
  -H "Content-Type: application/json" \
  -d '{
    "sync_mode": "tag_filter",
    "tag_filters": ["production"]
  }'
```

### Delete a Peer

```bash
curl -X DELETE https://registry.com/api/peers/lob-a \
  -H "Authorization: Bearer <token>"
```

This removes the peer configuration. Synced items are marked as orphaned.

### View Sync Status

**UI:** Click on a peer to view detailed status including:
- Last successful sync
- Total servers/agents synced
- Current generation number
- Health status

**API:**

```bash
curl https://registry.com/api/peers/lob-a/status \
  -H "Authorization: Bearer <token>"
```

## Token Rotation

### Rotating Federation Token (Exporting Registry)

1. Generate a new token:
   ```bash
   python -c "import secrets; print(secrets.token_urlsafe(32))"
   ```

2. Update the exporting registry's `.env`:
   ```bash
   FEDERATION_STATIC_TOKEN=<new-token>
   ```

3. Restart the exporting registry.

4. Update the peer configuration on all importing registries:
   ```bash
   curl -X PUT https://hub-registry.com/api/peers/lob-a \
     -H "Authorization: Bearer <token>" \
     -d '{"federation_token": "<new-token>"}'
   ```

### Rotating Encryption Key (Importing Registry)

If you need to rotate the `FEDERATION_ENCRYPTION_KEY`:

1. Export current peer configurations (tokens will be encrypted)
2. Generate new Fernet key
3. Run migration script to re-encrypt tokens with new key
4. Update `.env` with new key
5. Restart registry

## Troubleshooting

### Connection Refused

**Symptom:** Sync fails with connection error.

**Checks:**
- Verify network connectivity: `curl https://peer-registry.com/api/v1/federation/health`
- Check firewall rules allow HTTPS traffic
- Verify endpoint URL is correct in peer config

### Authentication Failed (401/403)

**Symptom:** Sync fails with authentication error.

**Checks:**
- Verify `FEDERATION_STATIC_TOKEN_AUTH_ENABLED=true` on exporting registry
- Verify token in peer config matches `FEDERATION_STATIC_TOKEN` on exporting registry
- Check token was copied correctly (no extra whitespace)
- Verify `FEDERATION_ENCRYPTION_KEY` is set on importing registry

### No Items Synced

**Symptom:** Sync succeeds but no servers/agents appear.

**Checks:**
- Verify items have `visibility: "public"` on exporting registry
- Check sync_mode and filters are not too restrictive
- Verify items exist on the exporting registry

### Sync Reports 0 Items After Successful Authentication

**Symptom:** Sync completes successfully and authentication passes, but 0 servers/agents are returned even though items exist on the peer registry.

**Root Cause:** This can indicate that the federation token was lost or corrupted during a peer configuration update (issue #561, fixed in version XX.XX).

**Diagnostic Steps:**

1. Check if the peer had items synced previously:
   ```bash
   curl https://registry.com/api/peers/peer-id/status
   ```
   If `total_servers_synced` was > 0 before but is now 0, the token may be lost.

2. Verify the peer registry is actually returning data:
   ```bash
   # Direct test to peer registry (replace with actual token)
   curl -H "Authorization: Bearer <federation-token>" \
        https://peer-registry.com/api/v1/federation/servers
   ```

**Resolution:**

If authentication is working but 0 items are returned, update the federation token using the dedicated endpoint:

```bash
curl -X PATCH https://registry.com/api/peers/peer-id/token \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin-token>" \
  -d '{
    "federation_token": "<correct-token-from-peer-registry>"
  }'
```

Or use the UI:
1. Navigate to Settings > Federation > Peers
2. Click the peer name
3. Update the Federation Token field
4. Save changes
5. Trigger a manual sync to verify

### Federation Token Lost After Peer Update (Fixed in XX.XX)

**Symptom:** After updating a peer's configuration (name, endpoint, sync interval, etc.), all subsequent syncs fail with authentication errors or return 0 items.

**Root Cause:** Bug in versions prior to XX.XX where the `update_peer()` operation would silently drop the encrypted federation token when updating any peer field.

**Who is Affected:**
- Anyone who updated peer configurations between version X.X and X.X

**How to Identify:**
- Sync was working before a peer update
- Sync now returns 0 items or authentication errors
- No changes were made to the token itself

**Recovery:**

Update the federation token using the dedicated token update endpoint:

```bash
curl -X PATCH https://registry.com/api/peers/<peer-id>/token \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer <admin-token>" \
  -d '{
    "federation_token": "<correct-federation-token>"
  }'
```

**Prevention:** This issue is fixed in version XX.XX. After upgrading, peer updates will preserve the federation token correctly.

### Synced Items Are Read-Only

**Expected behavior:** Federated items cannot be modified locally.

If you need to modify a synced item:
1. Modify it on the source registry
2. Wait for next sync or trigger manual sync

### Orphaned Items

**Symptom:** Items show as orphaned in the UI.

This happens when items are removed from the source registry. To resolve:
1. Confirm the items should be removed from the source
2. Delete orphaned items manually, or
3. Re-sync to clear orphaned status if items are restored

## Monitoring

### Health Check Endpoint

Each registry exposes a federation health endpoint:

```bash
curl https://registry.com/api/v1/federation/health
```

Returns:
```json
{
  "status": "healthy",
  "federation_enabled": true,
  "peer_count": 3
}
```

### Sync Status Metrics

Monitor sync status via the API:

```bash
curl https://registry.com/api/peers/lob-a/status
```

Returns:
```json
{
  "peer_id": "lob-a",
  "is_healthy": true,
  "last_successful_sync": "2026-02-05T10:30:00Z",
  "total_servers_synced": 15,
  "total_agents_synced": 3,
  "sync_in_progress": false,
  "consecutive_failures": 0
}
```

### Alerting Recommendations

Set up alerts for:
- `consecutive_failures > 3` - Sync has failed multiple times
- `is_healthy == false` - Peer is unreachable
- Time since `last_successful_sync > 2x sync_interval` - Sync is stale

## Security Best Practices

1. **Use strong tokens**: Generate tokens with `secrets.token_urlsafe(32)` or longer
2. **Rotate tokens periodically**: Rotate federation tokens at least annually
3. **Limit visibility**: Only set `visibility: "public"` on items that should be shared
4. **Use tag filters**: Use tag-based filtering to control what gets synced
5. **Monitor sync activity**: Review sync logs for unexpected patterns
6. **Network isolation**: Use private networks or VPNs between registries when possible

## Related Documentation

- [Federation Architecture](design/federation-architecture.md) - Technical architecture
- [Federation Guide](federation.md) - External registry integration (Anthropic, ASOR)
- [Static Token Auth](static-token-auth.md) - Static token authentication details
- [Configuration Reference](configuration.md) - Environment variable reference
