# MCP Gateway Registry Scripts

This directory contains utility scripts for building, testing, and deploying MCP Gateway Registry services.

## DocumentDB Initialization Scripts

### Overview

The DocumentDB initialization scripts set up collections and indexes for the MCP Gateway Registry when using AWS DocumentDB Elastic Cluster as the storage backend.

### Quick Start

```bash
# Set environment variables
export DOCUMENTDB_HOST=your-cluster.docdb.amazonaws.com
export DOCUMENTDB_USERNAME=admin
export DOCUMENTDB_PASSWORD=yourpassword

# Run initialization
./scripts/init-documentdb.sh

# Or with namespace
export DOCUMENTDB_NAMESPACE=production
./scripts/init-documentdb.sh
```

### Scripts

#### init-documentdb.sh

Bash wrapper script that downloads the CA bundle (if needed) and runs the Python initialization script.

**Features:**
- Downloads AWS DocumentDB CA bundle automatically if missing
- Validates environment configuration
- Color-coded output for easy readability
- Supports both environment variables and command-line arguments

**Usage:**
```bash
# Using environment variables (recommended)
export DOCUMENTDB_HOST=your-cluster.docdb.amazonaws.com
export DOCUMENTDB_USERNAME=admin
export DOCUMENTDB_PASSWORD=yourpassword
./scripts/init-documentdb.sh

# Pass through command-line arguments to Python script
./scripts/init-documentdb.sh --recreate --namespace production
```

#### init-documentdb-indexes.py

Python script that creates all necessary DocumentDB collections and indexes.

**Features:**
- Creates vector indexes for embeddings (HNSW, 1536 dimensions, cosine similarity)
- Creates standard indexes for servers, agents, scopes, security scans, and federation config
- Supports both IAM and username/password authentication
- Namespace support for multi-tenancy
- Recreate mode to drop and recreate indexes

**Usage:**
```bash
# Using environment variables
uv run python scripts/init-documentdb-indexes.py

# Using command-line arguments
uv run python scripts/init-documentdb-indexes.py \
  --host your-cluster.docdb.amazonaws.com \
  --username admin \
  --password yourpassword

# With IAM authentication
uv run python scripts/init-documentdb-indexes.py \
  --use-iam \
  --host your-cluster.docdb.amazonaws.com

# With namespace
uv run python scripts/init-documentdb-indexes.py --namespace production

# Recreate indexes
uv run python scripts/init-documentdb-indexes.py --recreate
```

#### download-documentdb-ca-bundle.sh

Downloads the AWS DocumentDB global CA bundle certificate required for TLS connections.

**Usage:**
```bash
./scripts/download-documentdb-ca-bundle.sh
```

### Collections and Indexes Created

The initialization script creates the following collections with indexes:

1. **mcp_servers_{namespace}**
   - Unique index on `_id` (path)
   - Index on `server_name`
   - Index on `is_enabled`
   - Index on `version`
   - Index on `tags`

2. **mcp_agents_{namespace}**
   - Unique index on `_id` (path)
   - Index on `name`
   - Index on `is_enabled`
   - Index on `version`
   - Index on `tags`

3. **mcp_scopes_{namespace}**
   - Unique index on `_id` (scope name)
   - Index on `name`

4. **mcp_embeddings_1536_{namespace}**
   - HNSW vector index on `embedding` (1536 dimensions, cosine similarity)
   - Unique index on `path`
   - Index on `name`
   - Index on `entity_type`

5. **mcp_security_scans_{namespace}**
   - Unique index on `_id` (scan ID)
   - Index on `entity_path`
   - Index on `entity_type`
   - Index on `scan_status`
   - Index on `scanned_at`

6. **mcp_federation_config_{namespace}**
   - Unique index on `_id` (config ID)

7. **audit_events_{namespace}**
   - Unique compound index on `request_id` + `log_type`. One request writes both
     an `mcp_server_access` and a `registry_api_access` record, which share a
     `request_id`; the single-field unique index this replaced rejected the
     second write and the record was dropped. The migration builds this index
     before dropping the old one, so uniqueness is never briefly unenforced.
   - Compound index on `identity.username` + `timestamp`
   - Compound index on `action.operation` + `timestamp`
   - Compound index on `action.resource_type` + `timestamp`
   - Index on `mcp_server.name`
   - Compound index on `log_type` + `resource_type` + `resource_id` +
     `timestamp` (the `token_mint` stream keeps these at the top level rather
     than under `action.*`)
   - 13 identity-claim indexes, each `log_type` + one claim + `timestamp`, with a
     `partialFilterExpression` pinning it to the single stream whose record shape
     it serves. `registry_api_access` and `mcp_server_access` nest the claims
     under `identity`; `token_mint` carries them at the top level along with a
     flat `username`. The claims are `principal_name`, `subject`, `canonical_id`
     and `object_id`. Partial rather than sparse: on a compound index `sparse`
     keeps a document when any indexed field exists, and `log_type` always
     exists, so it would be inert.
   - TTL index on `timestamp` (default 7 days, set by
     `AUDIT_LOG_MONGODB_TTL_DAYS`). Reducing it deletes every record older than
     the new window, so the script refuses to shorten retention unless
     `AUDIT_LOG_MONGODB_TTL_ALLOW_SHRINK=true`.

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `DOCUMENTDB_HOST` | `localhost` | DocumentDB cluster endpoint |
| `DOCUMENTDB_PORT` | `27017` | DocumentDB port |
| `DOCUMENTDB_DATABASE` | `mcp_registry` | Database name |
| `DOCUMENTDB_USERNAME` | - | Username for authentication |
| `DOCUMENTDB_PASSWORD` | - | Password for authentication |
| `DOCUMENTDB_USE_IAM` | `false` | Use AWS IAM authentication |
| `DOCUMENTDB_USE_TLS` | `true` | Enable TLS for connections |
| `DOCUMENTDB_TLS_CA_FILE` | `global-bundle.pem` | Path to TLS CA bundle |
| `DOCUMENTDB_NAMESPACE` | `default` | Namespace for multi-tenancy |
| `AUDIT_LOG_MONGODB_TTL_DAYS` | `7` | Audit log retention in days (TTL index) |

### Prerequisites

- Python 3.14+ with motor and boto3 installed
- AWS credentials configured (for IAM authentication or DocumentDB access)
- Network access to DocumentDB cluster
- DocumentDB cluster provisioned via Terraform (see terraform/aws-ecs/documentdb-elastic.tf)

### Authentication Methods

#### Username/Password Authentication

```bash
export DOCUMENTDB_HOST=your-cluster.docdb.amazonaws.com
export DOCUMENTDB_USERNAME=admin
export DOCUMENTDB_PASSWORD=yourpassword
export DOCUMENTDB_USE_TLS=true
./scripts/init-documentdb.sh
```

#### IAM Authentication

```bash
export DOCUMENTDB_HOST=your-cluster.docdb.amazonaws.com
export DOCUMENTDB_USE_IAM=true
export DOCUMENTDB_USE_TLS=true
# AWS credentials from environment or IAM role
./scripts/init-documentdb.sh
```

#### Local Development (No Authentication)

```bash
export DOCUMENTDB_HOST=localhost
export DOCUMENTDB_USE_TLS=false
./scripts/init-documentdb.sh
```

### Troubleshooting

#### "DOCUMENTDB_HOST environment variable is not set"
Set the required environment variables before running:
```bash
export DOCUMENTDB_HOST=your-cluster.docdb.amazonaws.com
```

#### "AWS credentials not found for DocumentDB IAM auth"
Configure AWS credentials:
```bash
aws configure
# Or use IAM role attached to EC2/ECS task
```

#### "Failed to download CA bundle"
- Check network connectivity
- Verify wget or curl is installed
- Download manually from: https://truststore.pki.rds.amazonaws.com/global/global-bundle.pem

#### "Failed to create vector index"
- Ensure DocumentDB cluster version supports vector search
- Check that dimensions (1536) match your embeddings model
- Verify DocumentDB Elastic Cluster (not instance-based cluster)

### Using with Docker Compose

DocumentDB is a managed AWS service and runs outside of Docker. To use DocumentDB with docker-compose services:

1. Initialize DocumentDB:
```bash
export DOCUMENTDB_HOST=your-cluster.docdb.amazonaws.com
export DOCUMENTDB_USERNAME=admin
export DOCUMENTDB_PASSWORD=yourpassword
./scripts/init-documentdb.sh
```

2. Update docker-compose environment:
```yaml
services:
  registry:
    environment:
      - STORAGE_BACKEND=documentdb
      - DOCUMENTDB_HOST=your-cluster.docdb.amazonaws.com
      - DOCUMENTDB_USERNAME=admin
      - DOCUMENTDB_PASSWORD=yourpassword
```

3. Restart services:
```bash
docker-compose up -d
```

### Further Reading

- [AWS DocumentDB Elastic Cluster Documentation](https://docs.aws.amazon.com/documentdb/latest/developerguide/elastic-clusters.html)
- [DocumentDB Vector Search](https://docs.aws.amazon.com/documentdb/latest/developerguide/vector-search.html)
- [Motor AsyncIO MongoDB Driver](https://motor.readthedocs.io/)
- [Terraform Configuration](../terraform/aws-ecs/documentdb-elastic.tf)

## Keycloak Build & Push Script

### Overview

The `build-and-push-keycloak.sh` script automates the process of building a Keycloak Docker image and pushing it to AWS ECR (Elastic Container Registry).

### Quick Start

```bash
# Build and push with defaults (latest tag to us-west-2)
./scripts/build-and-push-keycloak.sh

# Build and push with custom tag
./scripts/build-and-push-keycloak.sh --image-tag v24.0.1

# Build only (don't push)
./scripts/build-and-push-keycloak.sh --no-push
```

### Using with Make

```bash
# Build Keycloak image locally
make build-keycloak

# Build and push to ECR
make build-and-push-keycloak

# Deploy to ECS (after push)
make deploy-keycloak

# Complete workflow: build, push, and deploy
make update-keycloak

# With custom parameters
make build-and-push-keycloak AWS_REGION=us-east-1 IMAGE_TAG=v24.0.1
```

### Options

- `--aws-region REGION` - AWS region (default: us-west-2)
- `--image-tag TAG` - Image tag (default: latest)
- `--aws-profile PROFILE` - AWS profile (default: default)
- `--dockerfile PATH` - Dockerfile path (default: docker/keycloak/Dockerfile)
- `--build-context PATH` - Build context (default: docker/keycloak)
- `--no-push` - Build only, don't push to ECR
- `--help` - Show help message

### Prerequisites

- Docker installed and running
- AWS CLI installed and configured
- AWS credentials with ECR access
- Permission to push to ECR repository `keycloak`

### Features

- Color-coded output for easy readability
- Step-by-step progress tracking
- Error handling with clear error messages
- ECR login automation
- Image verification after push
- Helpful commands for manual deployment

### Workflow Example

```bash
# Build and push image
./scripts/build-and-push-keycloak.sh --image-tag v24.0.1

# Deploy to ECS
aws ecs update-service \
  --cluster keycloak \
  --service keycloak \
  --force-new-deployment \
  --region us-west-2

# Monitor deployment
aws ecs describe-services \
  --cluster keycloak \
  --services keycloak \
  --region us-west-2 \
  --query 'services[0].[serviceName,status,runningCount,desiredCount]' \
  --output table
```

### Troubleshooting

#### "Failed to get AWS account ID"
- Check AWS credentials: `aws sts get-caller-identity`
- Verify AWS profile: `aws configure list --profile <profile-name>`

#### "Failed to login to ECR"
- Verify ECR permissions in IAM
- Check if repository exists: `aws ecr describe-repositories --repository-names keycloak`

#### "Failed to build Docker image"
- Check Docker is running: `docker ps`
- Verify Dockerfile exists: `ls -la docker/keycloak/Dockerfile`

### Further Reading

- [AWS ECR Documentation](https://docs.aws.amazon.com/ecr/)
- [Keycloak Docker Image](https://hub.docker.com/r/keycloak/keycloak)
- [ECS Service Updates](https://docs.aws.amazon.com/AmazonECS/latest/developerguide/update-service.html)
