# Release v1.0.9-patch1 - MongoDB Authentication Compatibility

**January 7, 2026**

---

## Overview

This patch release addresses MongoDB authentication compatibility issues between MongoDB Community Edition and AWS DocumentDB. The changes enable the MCP Gateway Registry to work seamlessly with both MongoDB CE 8.2+ (using SCRAM-SHA-256) and AWS DocumentDB v5.0 (using SCRAM-SHA-1).

**Related Issues:**
- [#334](https://github.com/agentic-community/mcp-gateway-registry/issues/334) - Upgrade MongoDB authentication to SCRAM-SHA-256
- [#336](https://github.com/agentic-community/mcp-gateway-registry/issues/336) - Upgrade AWS DocumentDB authentication to SCRAM-SHA-256 (parking lot)

**Pull Request:**
- [#335](https://github.com/agentic-community/mcp-gateway-registry/pull/335) - Fix MongoDB authentication compatibility for DocumentDB

---

## What's Fixed

### MongoDB Authentication Compatibility

The registry now automatically selects the correct authentication mechanism based on the storage backend:

- **MongoDB CE 8.2+**: Uses SCRAM-SHA-256 (stronger, modern authentication)
- **AWS DocumentDB v5.0**: Uses SCRAM-SHA-1 (only mechanism we could get to work with Amazon DocumentDB although the documentation claims SCRAM-SHA-256 should work, tracking it via [#336](https://github.com/agentic-community/mcp-gateway-registry/issues/336))

This is controlled by the new `STORAGE_BACKEND` environment variable:

```bash
# For MongoDB Community Edition
STORAGE_BACKEND=mongodb-ce

# For AWS DocumentDB (default)
STORAGE_BACKEND=documentdb
```

### Pydantic Validation Fix

Fixed test failures in semantic search API models by adding upper bound validation to relevance scores:

```python
# Before:
relevance_score: float = Field(0.0, ge=0.0)

# After:
relevance_score: float = Field(0.0, ge=0.0, le=1.0)
```

This ensures relevance scores are always bounded between 0.0 and 1.0, as expected.

### Federation Command Fix

Fixed `populate-registry.sh` script federation command syntax:

```bash
# Before (incorrect):
federation-rescan --provider anthropic

# After (correct):
federation-sync --source anthropic
```

### Integration Test Improvements

Added skip markers to MongoDB integration tests that require MongoDB to be running, preventing false failures in CI environments where MongoDB is not available.

---

## Changed Files

### Core Authentication Changes
- `registry/repositories/documentdb/client.py` - Conditional SCRAM authentication based on storage backend
- `scripts/init-documentdb-indexes.py` - Added storage_backend parameter
- `scripts/load-scopes.py` - Conditional SCRAM mechanism selection
- `scripts/manage-documentdb.py` - Conditional SCRAM mechanism selection
- `scripts/debug-scopes.py` - Conditional SCRAM mechanism selection
- `registry/scripts/inspect-documentdb.py` - Conditional SCRAM mechanism selection

### Build and Deployment
- `docker/Dockerfile.registry` - Added scripts directory to container
- `terraform/aws-ecs/documentdb.tf` - Added STORAGE_BACKEND environment variable
- `terraform/aws-ecs/keycloak-ecr.tf` - Version update
- `terraform/aws-ecs/modules/mcp-gateway/ecs-services.tf` - Added STORAGE_BACKEND to all services

### Scripts and Configuration
- `api/populate-registry.sh` - Fixed federation-sync command syntax
- `.env.example` - Added STORAGE_BACKEND documentation

### API and Test Fixes
- `registry/api/search_routes.py` - Added upper bound validation to relevance_score fields
- `tests/integration/test_mongodb_connectivity.py` - Added skip decorators to MongoDB tests

---

## Upgrade Instructions

### For Docker Compose Deployments

1. **Pull the latest changes:**
```bash
cd mcp-gateway-registry
git pull origin main
git checkout v1.0.9-patch1
```

2. **Update environment configuration:**
```bash
# For MongoDB CE deployments, add:
echo "STORAGE_BACKEND=mongodb-ce" >> .env

# For DocumentDB deployments (default), no changes needed
# STORAGE_BACKEND defaults to "documentdb"
```

3. **Rebuild and restart:**
```bash
./build_and_run.sh
```

### For AWS ECS Deployment

1. **Update Terraform configuration:**

The `STORAGE_BACKEND` environment variable is already set to `documentdb` in the Terraform configuration. No changes are required for DocumentDB deployments.

2. **Pull and deploy new images:**
```bash
# Build and push updated images
export AWS_REGION=us-east-1
make build-push

# Force ECS service update
aws ecs update-service \
  --cluster mcp-gateway-ecs-cluster \
  --service mcp-gateway-v2-registry \
  --force-new-deployment

# For auth server
aws ecs update-service \
  --cluster mcp-gateway-ecs-cluster \
  --service mcp-gateway-v2-auth \
  --force-new-deployment
```

### Testing the Upgrade

Verify authentication is working correctly:

```bash
# Check logs for authentication mechanism
aws logs tail /ecs/mcp-gateway-v2-registry --follow | grep "authentication"

# Expected output:
# Using username/password authentication (SCRAM-SHA-1) for documentdb

# Or for MongoDB CE:
# Using username/password authentication (SCRAM-SHA-256) for mongodb-ce
```

---

## Technical Details

### Authentication Mechanism Selection

The conditional authentication logic in `client.py`:

```python
if settings.storage_backend == "mongodb-ce":
    # MongoDB CE 8.2+: Use SCRAM-SHA-256 (stronger, modern authentication)
    auth_mechanism = "SCRAM-SHA-256"
else:
    # AWS DocumentDB v5.0: Only supports SCRAM-SHA-1
    auth_mechanism = "SCRAM-SHA-1"

connection_string = (
    f"mongodb://{settings.documentdb_username}:{settings.documentdb_password}@"
    f"{settings.documentdb_host}:{settings.documentdb_port}/"
    f"{settings.documentdb_database}?authMechanism={auth_mechanism}&authSource=admin"
)
```

### Environment Variables

New environment variable:

- `STORAGE_BACKEND` - Controls authentication mechanism selection
  - `documentdb` (default) - Use SCRAM-SHA-1 for AWS DocumentDB
  - `mongodb-ce` - Use SCRAM-SHA-256 for MongoDB Community Edition

### Why This Change?

AWS DocumentDB v5.0 only supports two authentication mechanisms:
- SCRAM-SHA-1 (username/password)
- MONGODB-AWS (IAM authentication)

MongoDB Community Edition 8.2+ defaults to SCRAM-SHA-256 for improved security. This patch enables seamless operation with both backends without requiring code changes.

---

## Breaking Changes

None. This is a backward-compatible patch release. Existing deployments will continue to work:

- **DocumentDB deployments**: `STORAGE_BACKEND` defaults to `documentdb`, using SCRAM-SHA-1
- **MongoDB CE deployments**: Can now explicitly set `STORAGE_BACKEND=mongodb-ce` to use SCRAM-SHA-256

---

## Known Limitations

- **AWS DocumentDB**: Still uses SCRAM-SHA-1 authentication. Upgrade to SCRAM-SHA-256 is tracked in [#336](https://github.com/agentic-community/mcp-gateway-registry/issues/336) and depends on AWS adding SCRAM-SHA-256 support to DocumentDB.

---

## Resources

### Documentation
- [Environment Configuration](.env.example) - All environment variables documented
- [AWS ECS Deployment Guide](terraform/aws-ecs/README.md)
- [Database Abstraction Layer](docs/database-abstraction-layer.md)

### Related Issues and PRs
- [#334](https://github.com/agentic-community/mcp-gateway-registry/issues/334) - Original issue: Upgrade MongoDB authentication to SCRAM-SHA-256
- [#335](https://github.com/agentic-community/mcp-gateway-registry/pull/335) - Implementation PR
- [#336](https://github.com/agentic-community/mcp-gateway-registry/issues/336) - Future work: DocumentDB SCRAM-SHA-256 support

---

## Commits Included

```
5dc2471 Fix MongoDB authentication compatibility for DocumentDB (#335)
252869c Rewrite roadmap section with milestone-based table format
5761054 Move completed issues #70 and #48 to Completed section
```

**Files Changed:** 16 files changed, 461 insertions(+), 106 deletions(-)

---

## Support

- [GitHub Issues](https://github.com/agentic-community/mcp-gateway-registry/issues)
- [GitHub Discussions](https://github.com/agentic-community/mcp-gateway-registry/discussions)
- [Documentation](https://github.com/agentic-community/mcp-gateway-registry/tree/main/docs)

---

**Full Changelog:** [v1.0.9...v1.0.9-patch1](https://github.com/agentic-community/mcp-gateway-registry/compare/v1.0.9...v1.0.9-patch1)
