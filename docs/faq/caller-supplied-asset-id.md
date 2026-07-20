# How do I register an asset with my own id instead of an auto-generated one?

## Question

By default the registry assigns every server, agent, and skill a random `id` (a uuid4) at registration. I want to supply my **own** `id` instead, for example:

- an AWS ARN (`arn:aws:bedrock-agentcore:us-east-1:123456789012:runtime/my-runtime`),
- a URN or peer-registry id from an upstream system,
- any stable identifier my own tooling already tracks.

How do I turn this on, and how do I pass the id when registering?

## Answer

Caller-supplied ids are an **opt-in** feature (issue #1276). They are **off by default** (fail-closed): unless a deployment explicitly enables the flag, a supplied `id` is rejected and the registry keeps auto-generating one. This keeps the default behavior unchanged for every existing caller.

When enabled, a supplied `id` is honored verbatim on the public server / agent / skill registration routes, subject to a safe-character check and a uniqueness check.

### Step 1: Enable the feature

Set the flag for the registry. The variable is `ALLOW_CALLER_SUPPLIED_ASSET_ID` (default `false`). It is wired across all three deployment surfaces:

**Docker Compose** (in `.env`):

```bash
ALLOW_CALLER_SUPPLIED_ASSET_ID=true
```

Then restart the registry so it picks up the new value:

```bash
docker compose -f docker-compose.yml -f docker-compose.override.yml up -d registry
```

**Terraform** (`terraform.tfvars`):

```hcl
allow_caller_supplied_asset_id = true
```

**Helm** (`values.yaml`):

```yaml
registry:
  app:
    allowCallerSuppliedAssetId: true
```

See [docs/unified-parameter-reference.md](../unified-parameter-reference.md) for the full cross-surface reference.

You can confirm the running registry sees the flag:

```bash
docker exec mcp-gateway-registry-registry-1 sh -c 'echo $ALLOW_CALLER_SUPPLIED_ASSET_ID'
# expect: true
```

### Step 1a (existing deployments): run the migration

A fresh deployment needs nothing extra: the registry lazily backfills ids and builds the unique index the first time it touches each collection. But an **existing** deployment that already has assets should run the migration script once, before serving traffic, so the unique index is in place ahead of time:

```bash
# Dry run (default): report duplicate ids, missing ids, and the planned index.
# Makes no changes.
uv run python scripts/migrate-assets-add-unique-id.py

# Apply: backfill a uuid4 onto any legacy doc missing an id, then build the
# unique partial index id_idx on the servers, agents, and skills collections.
uv run python scripts/migrate-assets-add-unique-id.py --apply

# For a remote DocumentDB cluster, pass connection settings, e.g.:
uv run python scripts/migrate-assets-add-unique-id.py --apply --host your-cluster.docdb.amazonaws.com
```

The script is safe by design: it runs a **dry run by default**, backfills missing ids **before** building the index (so the build never fails on legacy rows), and with `--apply` it **refuses and exits non-zero if any collection already has duplicate ids** (resolve those first, so a partial-index build can never fail mid-deploy). The index it creates is named `id_idx`.

Fresh-install index bootstrap is also covered by `scripts/init-documentdb-indexes.py`, which includes `id_idx` alongside the other asset indexes.

### Step 2: Register with a supplied id

All three asset types accept an optional `id` through the management CLI (`api/registry_management.py`). Global arguments (`--registry-url`, `--token-file`) go **before** the subcommand.

**Server** (the `id` goes in the config JSON):

```bash
cat > server.json <<'EOF'
{
  "path": "/my-server",
  "server_name": "my-server",
  "description": "My server",
  "proxy_pass_url": "https://example.com",
  "id": "arn:example:server:my-server"
}
EOF

uv run python api/registry_management.py \
  --registry-url http://localhost --token-file .token \
  register --config server.json
```

**Agent** (the `id` goes in the config JSON):

```bash
cat > agent.json <<'EOF'
{
  "name": "my-agent",
  "description": "My agent",
  "url": "https://example.com/agent",
  "version": "1.0",
  "supportedProtocol": "a2a",
  "id": "urn:example:agent:my-agent"
}
EOF

uv run python api/registry_management.py \
  --registry-url http://localhost --token-file .token \
  agent-register --config agent.json
```

**Skill** (the `id` is a dedicated `--id` flag):

```bash
uv run python api/registry_management.py \
  --registry-url http://localhost --token-file .token \
  skill-register \
  --name my-skill \
  --url "https://raw.githubusercontent.com/org/repo/main/skills/my-skill/SKILL.md" \
  --description "My skill" \
  --id "urn:example:skill:my-skill"
```

Omitting the `id` in any of the above still works and auto-generates a uuid4, so existing configs and scripts need no change.

### Step 3: Verify the stored id

Read the asset back and confirm the `id` is your value, not a uuid4:

```bash
TOKEN=$(python3 -c "import json;d=json.load(open('.token'));print(d.get('access_token') or d['tokens']['access_token'])")
curl -sS "http://localhost/api/agents/my-agent" -H "Authorization: Bearer $TOKEN" \
  | python3 -c "import sys,json;print('id:', json.load(sys.stdin).get('id'))"
# expect: id: urn:example:agent:my-agent
```

## What ids are allowed?

A supplied id must satisfy all of these, or the registration is rejected with **422**:

- **Non-empty** after trimming surrounding whitespace.
- **At most 512 characters.**
- **Safe characters only:** letters, digits, and `. _ - : / @ # = +`. This covers UUID, ARN, and URN shapes while rejecting whitespace, quotes, angle brackets, backslash, and shell/regex metacharacters. Treat the id as an opaque identifier: it is never interpreted as a URL, path, or command.

## Uniqueness and conflicts

Ids must be unique **per asset type**. Registering an asset whose id already belongs to another asset of the same type returns **409 Conflict**:

```json
{
  "detail": "Agent with id 'urn:example:agent:my-agent' already exists",
  "suggestion": "Use a different id or omit it to auto-generate one"
}
```

Uniqueness is enforced by a unique database index in addition to a pre-check, so two registrations racing with the same id still resolve to a single winner and a 409 for the loser.

## What happens when the feature is disabled?

With `ALLOW_CALLER_SUPPLIED_ASSET_ID=false` (the default):

- A request that **supplies** an `id` is rejected with **422** and a message like *"caller-supplied asset id is disabled on this registry; omit 'id' to auto-generate one, or set ALLOW_CALLER_SUPPLIED_ASSET_ID=true to enable it"*.
- A request that **omits** the `id` succeeds and auto-generates a uuid4, exactly as before.

## Note on federation

This flag governs the **public registration routes only**. Federation sync is **not** affected by it: ids arriving from a peer registry are governed by the peer allowlist, which is the trust boundary for federation. A peer asset whose id collides with a local asset is logged and skipped without aborting the sync batch.

## Observability

Three counters expose usage (Prometheus / OpenTelemetry):

- `registry_asset_id_supplied_total{asset_type=...}` — a caller-supplied id was honored.
- `registry_asset_id_conflict_total{asset_type=...}` — a registration was rejected for an id collision.
- `registry_asset_id_federation_conflict_total{asset_type=...}` — a federated asset was skipped for a local id collision.

If you enable the feature, also watch `registry_asset_id_index_build_failed_total`: a non-zero value means the unique index could not be built and the registry is relying on the (racy) service-layer pre-check alone, so id uniqueness is not guaranteed at the database level.
