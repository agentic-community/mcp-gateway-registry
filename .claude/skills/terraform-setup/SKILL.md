---
name: terraform-setup
description: "Install (deploy) MCP Gateway & Registry on AWS using the Terraform aws-ecs stack (ECS Fargate, Aurora, DocumentDB, Keycloak). Asks whether you are running from an EC2 instance or a local laptop, confirms the required AWS IAM permissions are in place, clones the repository, bootstraps the toolchain (uv, AWS CLI, Terraform), configures terraform.tfvars, runs the two-stage terraform apply, and completes post-deployment setup. Does NOT create IAM roles itself — it tells you the permissions you need and offers to guide you through setting them up."
license: Apache-2.0
metadata:
  author: mcp-gateway-registry
  version: "1.0"
---

# MCP Gateway & Registry — Terraform (AWS ECS) Install Skill

**Repository:** https://github.com/agentic-community/mcp-gateway-registry
**This skill:** https://github.com/agentic-community/mcp-gateway-registry/blob/main/.claude/skills/terraform-setup/SKILL.md
**Full Terraform guide:** https://github.com/agentic-community/mcp-gateway-registry/blob/main/terraform/aws-ecs/README.md

## How to run this skill without cloning the repository

This skill is self-contained. You can invoke it from any directory in Claude Code. It will clone the repository for you.

```
/terraform-setup
```

Or reference it remotely if you have not installed this repo:

```
@https://raw.githubusercontent.com/agentic-community/mcp-gateway-registry/main/.claude/skills/terraform-setup/SKILL.md
```

---

## What this skill does

**`/terraform-setup`** — Guided deployment of the MCP Gateway & Registry to AWS via Terraform:
- Confirms where you are running from (EC2 instance vs. local laptop) and what that means for AWS credentials
- Confirms the AWS IAM permissions the deployment requires are in place
- Clones the MCP Gateway & Registry repository
- Bootstraps the toolchain: `uv` + `uv sync` (which provides the AWS CLI in the project venv), Terraform, and supporting tools
- Helps you configure `terraform/aws-ecs/terraform.tfvars`
- Runs `terraform apply` (a single apply in CloudFront Only mode; the two-stage certs-first flow is only needed for custom-domain mode)
- Runs the automated post-deployment setup and registers the AWS KB MCP server (the only server this skill registers)
- Ends with a complete summary of every step taken

By default the three core services (registry, auth-server, mcpgw) pull pre-built images from public ECR, so **no image build step is required**.

### AWS services / resources this deploys (tell the user up front)

Before applying, **state clearly to the user** that this stack creates and pays for the following AWS resources in their account (region `${AWS_REGION}`):

| AWS service | What it is used for |
|-------------|---------------------|
| Amazon VPC | Dedicated VPC across 2 AZs: public/private subnets, Internet Gateway, NAT Gateway, route tables, security groups |
| Amazon ECS (Fargate) | ECS cluster + services/tasks for Registry, Auth Server, and Keycloak (no servers to manage) |
| Amazon RDS — Aurora PostgreSQL Serverless v2 | User/session/Keycloak data; multi-AZ; accessed via RDS Proxy |
| Amazon DocumentDB | MongoDB-compatible store for server/agent metadata, scopes, embeddings (HNSW vector search). **Only when `storage_backend = "documentdb"` (the default) — region-gated, see Step 0.** With an external-mongodb backend, no DocumentDB is provisioned. |
| Elastic Load Balancing (ALB) | Two Application Load Balancers (main + Keycloak) |
| AWS Certificate Manager (ACM) | TLS certificates for HTTPS |
| Amazon CloudFront | CDN / public HTTPS entry point (`*.cloudfront.net`) — this is the default access mode |
| AWS Secrets Manager | Stores DB passwords, Keycloak admin/client secrets, etc. |
| Amazon CloudWatch | Logs (per-service log groups) + Alarms |
| Amazon SNS | Alarm notifications |
| AWS Cloud Map (servicediscovery) | Private service discovery namespace for ECS tasks |
| AWS IAM | Task execution role, task roles, scaling roles (created by Terraform) |
| Application Auto Scaling | CPU/memory-based scaling of ECS tasks |
| AWS KMS | Encryption keys for secrets / data at rest |
| AWS Systems Manager (SSM) | ECS Exec / session access; reads public global-infrastructure parameters |
| Amazon Managed Prometheus (AMP) + metrics-service + ADOT collector + Grafana | Observability pipeline. **`enable_observability` defaults to `true`**, BUT `grafana_image_uri` / `metrics_service_image_uri` default to empty and are **NOT on public ECR** — they must be built (`make build-push`, needs Docker). So a public-image deployment must either build those two images or set `enable_observability = false`. **AMP is AWS-managed** (`aws_prometheus_workspace`); **Grafana here is self-hosted Grafana OSS on ECS** (a custom-built image at `/grafana`), **not** Amazon Managed Grafana. The ADOT collector uses an AWS public image (no build). |
| Amazon Route 53 | **Only if you later switch to a custom domain** (not used in CloudFront-only mode) |

**Cost warning:** this is real, always-on infrastructure — roughly **$170–330/month** in `us-east-1` (Aurora, DocumentDB, ECS, 2 ALBs dominate). Make sure the user understands and agrees before running `terraform apply`. Running `terraform destroy` (with the pre-destroy cleanup) tears it down.

---

## CRITICAL: First action is ALWAYS Step 0

**DO NOT run any Bash commands. DO NOT check prerequisites. DO NOT read any files.**
**The very first action when this skill is invoked MUST be using `AskUserQuestion` to complete Step 0.**
**Nothing else happens until the user has answered the Step 0 questions.**

---

## Step tracking

Throughout the entire execution, Claude must maintain an internal step log. After every phase completes (success, skip, or failure), append an entry to this log. Display the full log as a formatted table in the Final Summary phase.

Step log format: `{ phase, name, status (DONE / SKIPPED / FAILED), notes }`

---

## Step 0: Determine Environment and Mode — MUST BE FIRST, NO EXCEPTIONS

**STOP. Do not run any commands. Use `AskUserQuestion` right now to ask the questions below before taking any other action.**

**Question 1 — Where are you running this deployment from?**
```
  EC2 instance - You are on an EC2 instance inside AWS. AWS credentials come
                 from an IAM role attached to the instance (an instance profile).
                 Recommended: simplest credential handling.

  Local laptop - You are on a local machine (macOS/Linux). AWS credentials come
                 from `aws configure`, an AWS profile, or environment variables.
```
Store the answer as `RUN_ENV` = `ec2` or `local`.

**Question 2 — Installation directory:**
```
Where should the repository be cloned?

Default: ~/mcp-gateway-registry

Enter a path, or accept the default.
```
Store the answer as `INSTALL_DIR`. If the user is already inside a checkout of this repo, offer to use the current directory instead. If the user provides no input, set `INSTALL_DIR=~/mcp-gateway-registry`.

**Expand the path immediately:**
```bash
INSTALL_DIR=$(eval echo "${INSTALL_DIR}")
echo "Installation directory: ${INSTALL_DIR}"
```

**Question 3 — Storage backend:**
```
How should the registry store its data (servers, agents, scopes, embeddings)?

  documentdb (recommended, Terraform DEFAULT) - Provision an Amazon DocumentDB
            cluster in this Terraform state. Simplest option: nothing external
            to set up. Region-gated (see below). Adds cost (~$60-80/month) and
            provisioning time.

  external mongodb (mongodb-ce / mongodb / mongodb-atlas) - Connect to a MongoDB
            you already own (Atlas / self-managed). Requires
            mongodb_connection_string or *_secret_arn. No DocumentDB
            provisioned.
```
Store the answer as `STORAGE_BACKEND` = `documentdb` (default), or one of `mongodb-ce` / `mongodb` / `mongodb-atlas`. **This maps directly to the `storage_backend` variable** (default `documentdb`), whose Terraform validation accepts ONLY these values (`file` is NOT a valid backend here and fails at `terraform plan`). DocumentDB resources are gated `local.is_aws_documentdb = var.storage_backend == "documentdb"`, so they are ONLY created when `documentdb` is chosen. For any external-mongodb value, `mongodb_connection_string` or `mongodb_connection_string_secret_arn` is REQUIRED (collect it in Phase 5).

**Question 4 — AWS region:**
```
Which AWS region should this be deployed into?

Default: us-east-1
```
Store the answer as `AWS_REGION`.

**DocumentDB region gate — ONLY when `STORAGE_BACKEND = documentdb`.** Amazon DocumentDB is not available in every region. If (and only if) the user chose `documentdb`, verify the region supports it and refuse to proceed otherwise. (With an external-mongodb backend, this check does not apply — skip it.)

**Note:** the `/aws/service/global-infrastructure/...` parameters are only served from `us-east-1`, so this query **must** pass `--region us-east-1` regardless of where you are deploying. (Without it the CLI errors "You must specify a region" and the check would wrongly report the region as unsupported.)

```bash
# Run ONLY if STORAGE_BACKEND = documentdb. Needs an AWS CLI (available after Phase 3 uv sync).
AWS="$(command -v aws || echo .venv/bin/aws)"
if ${AWS} ssm get-parameters-by-path \
     --region us-east-1 \
     --path /aws/service/global-infrastructure/services/docdb/regions \
     --query "Parameters[].Value" --output text 2>/dev/null | tr '\t' '\n' | grep -qx "${AWS_REGION}"; then
    echo "DOCDB_SUPPORTED in ${AWS_REGION}"
else
    echo "DOCDB_NOT_SUPPORTED in ${AWS_REGION} — pick a different region"
fi
```

If the region does not support DocumentDB, list the supported regions and ask the user (via `AskUserQuestion`) to choose one before continuing:

```bash
${AWS} ssm get-parameters-by-path \
  --region us-east-1 \
  --path /aws/service/global-infrastructure/services/docdb/regions \
  --query "Parameters[].Value" --output text 2>/dev/null | tr '\t' '\n' | sort
```

**When `documentdb` is selected, do not proceed past Phase 5 with a region that fails this check.**

**Question 5 — Observability pipeline (AMP + metrics-service + ADOT + Grafana):**
```
Enable the observability pipeline?

  No (recommended for a public-image deploy, DEFAULT for this skill) - Sets
     enable_observability = false. No AMP/metrics/Grafana. Avoids building images.

  Yes - Requires building two images NOT on public ECR (mcp-gateway-grafana,
     mcp-gateway-metrics-service) via `make build-push` (needs Docker), pushing
     them to your private ECR, setting grafana_image_uri + metrics_service_image_uri,
     and a grafana_admin_password.
```
Store as `ENABLE_OBSERVABILITY`. **Important:** the Terraform variable `enable_observability` defaults to `true`, but the Grafana/metrics image URIs default to empty — so if you leave it at the default WITHOUT providing those images, `terraform apply` fails with `Container.image should not be null or empty`. Therefore this skill **explicitly writes `enable_observability = false` in `terraform.tfvars` unless the user opts in** and can supply/build the two images.

- AMP is AWS-managed; ADOT uses an AWS public image. Only Grafana + metrics-service need building.
- The bundled Grafana is **self-hosted Grafana OSS on ECS**, not Amazon Managed Grafana (AMG). Pointing **AMG** at the AMP workspace instead would require extending the Terraform module (add `aws_grafana_workspace`) — it is not a config toggle today. Mention this if the user prefers a fully-managed Grafana.
- If `Yes`: also collect `grafana_admin_password` (strong value) in Phase 5, and build/push the images before applying.

**Deployment mode — fixed: CloudFront Only.**

This skill always deploys in **CloudFront Only** mode (`enable_cloudfront = true`, `enable_route53_dns = false`). Do not ask the user to choose a mode. Tell the user:

> "This deployment uses CloudFront. You will get HTTPS URLs like `https://d123abc.cloudfront.net` — no custom domain or Route53 hosted zone is required. If you want to use your own custom domain instead, you can change the Terraform config files; I'll point out exactly which settings to change in Phase 5 and remind you again at the end."

Log: `{ 0, "Environment & Mode Selection", DONE, "Env: ${RUN_ENV}, Region: ${AWS_REGION} (DocumentDB-verified), Mode: CloudFront Only, Dir: ${INSTALL_DIR}" }`

---

## Phase 1: AWS Credentials & IAM Permissions (READ THIS CAREFULLY)

This deployment creates and manages a broad set of AWS resources (VPC, ECS, RDS Aurora, DocumentDB, ALBs, ACM, Route53, Secrets Manager, IAM roles, CloudFront, etc.). The AWS identity you run it with **must** have permissions for all of them.

**The required permissions are the following policy actions** (from `terraform/aws-ecs/README.md`, "IAM Permissions"):

```json
{
    "Version": "2012-10-17",
    "Statement": [
        {
            "Sid": "MCPGatewayDeployment",
            "Effect": "Allow",
            "Action": [
                "secretsmanager:*",
                "bedrock-agentcore:*",
                "iam:PassRole",
                "s3:*",
                "lambda:*",
                "elasticfilesystem:*",
                "ec2:*",
                "ecs:*",
                "rds:*",
                "docdb:*",
                "elasticloadbalancing:*",
                "route53:*",
                "acm:*",
                "iam:*",
                "logs:*",
                "ecr:*",
                "application-autoscaling:*",
                "cloudwatch:*",
                "cloudfront:*",
                "sns:*",
                "ssm:*",
                "kms:*",
                "servicediscovery:*",
                "aps:*"
            ],
            "Resource": "*"
        }
    ]
}
```

Notes:
- **`s3:*` is REQUIRED** — the stack creates S3 buckets for ALB access logs and CloudFront logs. (The `terraform/aws-ecs/README.md` IAM list currently OMITS `s3` — that is a documentation gap; without it `terraform apply` fails with `s3:CreateBucket ... AccessDenied`.)
- `cloudfront:*` is only needed for the CloudFront deployment modes (Mode 1 and Mode 3).
- `aps:*` is only needed when `enable_observability = true` (Amazon Managed Prometheus).
- For production, consider scoping `Resource` down to specific ARNs.

**Account prerequisite — RDS service-linked role.** The stack creates an RDS DB Proxy, which requires the `AWSServiceRoleForRDS` service-linked role to exist in the account. On a brand-new account the first apply may fail with *"RDS is not authorized to assume service-linked role ... AWSServiceRoleForRDS"*. AWS usually auto-creates the role on first RDS use, so **simply re-running `terraform apply` resolves it**. If it persists, create it explicitly: `aws iam create-service-linked-role --aws-service-name rds.amazonaws.com`.

### How the requirement applies to YOUR environment

**If `RUN_ENV = ec2`:**
> This EC2 instance needs an IAM role (instance profile) attached that grants the permissions above. Once that role is attached, the AWS CLI and Terraform pick the credentials up automatically — there is nothing to configure.
>
> **This skill does NOT create that role for you.** Set it up however suits your environment (Terraform, the AWS console, CloudFormation, the CLI, or your platform team). If you want, I can guide you through creating the IAM role + instance profile and associating it with this instance interactively — just ask.

**If `RUN_ENV = local`:**
> Make sure the IAM user or role behind your local AWS credentials has the permissions above. Configure credentials with `aws configure`, an AWS profile (`AWS_PROFILE`), or environment variables.
>
> **Alternatively (recommended for clean credential handling):** launch an EC2 instance that has an instance role carrying these permissions, and run this deployment from there. Again, I can guide you through creating that role and instance if you ask — but this skill will not create it automatically.

### Verify the current identity

The AWS CLI may not be installed yet (it ships inside the project venv after Phase 3's `uv sync`). If an `aws` binary is already on the PATH, confirm who you are:

```bash
aws sts get-caller-identity 2>&1 || echo "AWS CLI not found yet — it will be available after uv sync (Phase 3). Re-verify then."
```

Show the user the identity (Account, Arn). Confirm with the user — via `AskUserQuestion` — that this identity has the required permissions before proceeding. Do not attempt to create or modify IAM roles as part of this skill unless the user explicitly asks you to guide them through it.

Log: `{ 1, "AWS Credentials & IAM Check", DONE/SKIPPED, "Env=${RUN_ENV}, identity=<arn or deferred>" }`

---

## Phase 2: Prerequisites Check

**Announce:** "Checking prerequisites..."

```bash
echo "=== git ==="; git --version 2>/dev/null && echo "GIT_OK" || echo "GIT_FAIL"
echo "=== Docker ==="; docker --version 2>/dev/null && echo "DOCKER_OK" || echo "DOCKER_FAIL (only needed if building custom images)"
echo "=== Terraform ==="; terraform version 2>/dev/null | head -1 && echo "TERRAFORM_OK" || echo "TERRAFORM_FAIL"
echo "=== uv ==="; uv --version 2>/dev/null && echo "UV_OK" || echo "UV_FAIL"
echo "=== jq ==="; jq --version 2>/dev/null && echo "JQ_OK" || echo "JQ_FAIL"
echo "=== AWS CLI (may be absent until uv sync) ==="; aws --version 2>/dev/null && echo "AWS_OK" || echo "AWS_PENDING"
echo "=== Session Manager plugin (for ECS exec / SSH) ==="; session-manager-plugin --version 2>/dev/null && echo "SSM_OK" || echo "SSM_OPTIONAL"
```

Install instructions for missing tools (the README has full Ubuntu/Debian commands):

| Check | Fix |
|-------|-----|
| GIT_FAIL | `sudo apt-get install -y git` (Ubuntu) / `xcode-select --install` (macOS) |
| TERRAFORM_FAIL | Install Terraform >= 1.5 — see README "Install Terraform (Ubuntu/Debian)" or https://www.terraform.io/downloads |
| UV_FAIL | `curl -LsSf https://astral.sh/uv/install.sh \| sh` then `source $HOME/.local/bin/env` |
| JQ_FAIL | `sudo apt-get install -y jq` / `brew install jq` |
| SSM_OPTIONAL | Only needed for `ecs-ssh.sh` / log following — see README "Install AWS Session Manager Plugin" |

`git` and `terraform` and `uv` are required to proceed. `aws` will be provided by `uv sync` in Phase 3 if it is missing now. `docker` is only required if you later choose to build custom images instead of using the public ECR images.

Log: `{ 2, "Prerequisites Check", DONE/FAILED, "what passed/failed" }`

---

## Phase 3: Clone Repository and Bootstrap Toolchain

**Announce:** "Cloning the repository and installing the Python/AWS toolchain via uv..."

### 3a. Clone (or reuse) the repository

```bash
if [ -d "${INSTALL_DIR}/.git" ]; then
    echo "ALREADY_CLONED"
else
    mkdir -p "$(dirname "${INSTALL_DIR}")"
    git clone https://github.com/agentic-community/mcp-gateway-registry.git "${INSTALL_DIR}"
    echo "Clone exit code: $?"
fi
cd "${INSTALL_DIR}"
echo "Working directory: $(pwd)"
ls terraform/aws-ecs/README.md 2>/dev/null && echo "REPO_VERIFIED" || echo "REPO_INVALID"
```

All subsequent phases run from within `${INSTALL_DIR}`.

### 3b. Run `uv sync` (this provides the AWS CLI in the venv)

The project declares `awscli` and `boto3` as dependencies, so `uv sync` gives you a working `aws` at `.venv/bin/aws` even if no system AWS CLI is installed.

```bash
cd "${INSTALL_DIR}"
uv sync 2>&1 | tail -15
echo "uv sync exit code: ${PIPESTATUS[0]}"
ls .venv/bin/aws 2>/dev/null && echo "VENV_AWS_OK" || echo "VENV_AWS_FAIL"
```

**If `uv sync` fails building native dependencies** (e.g. `yara-python` needing a C compiler / Python headers), install the build prerequisites and retry. On Ubuntu/Debian:

```bash
# gcc + matching Python dev headers (adjust the python3.X-dev version to your interpreter)
PYVER=$(uv run python -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>/dev/null || echo "3.14")
sudo apt-get update -qq
sudo apt-get install -y build-essential "python${PYVER}-dev"
uv sync 2>&1 | tail -10
```

On macOS the C toolchain comes from `xcode-select --install`.

### 3c. Make the venv AWS CLI usable

For the rest of this skill, use `.venv/bin/aws` (or `uv run aws`) if there is no system `aws`:

```bash
cd "${INSTALL_DIR}"
AWS="$(command -v aws || echo .venv/bin/aws)"
echo "Using AWS CLI: ${AWS}"
${AWS} --version
${AWS} sts get-caller-identity 2>&1
```

Re-confirm the caller identity now matches an identity with the IAM permissions from Phase 1. If `RUN_ENV=ec2`, the Arn should be an assumed-role tied to this instance's instance profile.

Log: `{ 3, "Clone + Toolchain Bootstrap", DONE/FAILED, "venv aws ready, identity confirmed" }`

---

## Phase 4: Using a Custom Domain Instead (informational — no action by default)

This skill deploys in CloudFront Only mode, so there is nothing to do here by default. **Just inform the user** that switching to a custom domain later is a config-file change, and tell them which files and settings are involved:

> "If you'd rather use your own domain instead of the CloudFront URL, it's a Terraform config change — no code changes. You would:
> 1. Create a **Route53 hosted zone** for your domain and point your registrar's nameservers at it (see `terraform/aws-ecs/README.md` → *Step 1.1: Domain Configuration*).
> 2. Edit **`terraform/aws-ecs/terraform.tfvars`** and switch these settings:
>    - `enable_route53_dns = true`
>    - `base_domain = \"your.domain\"`
>    - `session_cookie_domain = \".your.domain\"`
>    - keep `enable_cloudfront = true` for CloudFront + Custom Domain, or set it to `false` for Custom Domain only.
> 3. Re-run the two-stage `terraform apply`.
>
> The full instructions and the three-mode comparison table are in `terraform/aws-ecs/README.md` (*Deployment Modes* section) and `docs/deployment-modes.md`."

Do not create a hosted zone or modify domain settings unless the user explicitly asks to switch to a custom domain.

Log: `{ 4, "Custom Domain Info (skipped — CloudFront)", SKIPPED, "informed user how to switch" }`

---

## Phase 5: Configure terraform.tfvars

**Announce:** "Creating and configuring terraform/aws-ecs/terraform.tfvars..."

```bash
cd "${INSTALL_DIR}/terraform/aws-ecs"
[ -f terraform.tfvars ] && echo "TFVARS_EXISTS" || cp terraform.tfvars.example terraform.tfvars
```

If `terraform.tfvars` already exists, ask the user whether to keep it or regenerate from the example.

**Set these CloudFront-Only mode values (fixed for this skill):**

```hcl
enable_cloudfront     = true
enable_route53_dns    = false
session_cookie_domain = ""        # empty for CloudFront mode
```

**Also write `enable_observability` explicitly** based on the Step 0 answer — do NOT rely on the variable default (which is `true` and will fail without built Grafana/metrics images):
```hcl
enable_observability  = false     # or true ONLY if you built+set grafana_image_uri & metrics_service_image_uri and set grafana_admin_password
```

Collect the remaining required values (use `AskUserQuestion` for anything not already known):

| Parameter | Notes |
|-----------|-------|
| `aws_region` | Use the DocumentDB-verified `${AWS_REGION}` chosen in Step 0. Must match where any custom ECR images live. |
| `ingress_cidr_blocks` | List of CIDRs allowed to reach the ALB / registry. Always include the deploying EC2 instance's own public IP (`curl -s ifconfig.me` → `["<ip>/32"]`) **and** ask the user for any additional IPs (see below). Avoid `0.0.0.0/0` for production. |
| `keycloak_admin_password` | Min 12 chars. Keycloak master admin console password. |
| `keycloak_database_password` | Min 12 chars. |
| `documentdb_admin_username` / `documentdb_admin_password` | **REQUIRED when `storage_backend = "documentdb"`.** Password min 8 chars; avoid `/ " @` and spaces (DocumentDB constraints). The variable default is empty, so if you forget these `terraform apply` fails with `aws_docdb_cluster ... "master_password" ... required field is not set`. Username e.g. `docdbadmin`. (Can also be supplied via `TF_VAR_documentdb_admin_password`.) |
| `registry_name` / `registry_organization_name` | **REQUIRED — ASK THE USER for both** (use `AskUserQuestion`; offer a sensible default like `"MCP Gateway Registry"` / your org name, but do not silently assume). The Terraform variables default to `""`, and although their descriptions imply an app-side default, **the registry container rejects empty values** and crash-loops with `RuntimeError: Configuration errors: REGISTRY_NAME is required, REGISTRY_ORGANIZATION_NAME is required` (the nginx `unexpected "{"` error that follows is just a downstream symptom of the app failing to write its nginx config). Always write non-empty values. |
| `session_cookie_secure` | `true` (HTTPS — CloudFront always serves HTTPS). |
| `grafana_admin_password` | Required only if `enable_observability = true`. Generate a strong value. |

**Ask about additional access IPs (REQUIRED step).** Before writing `ingress_cidr_blocks`, use `AskUserQuestion` to ask: *"Besides this instance's public IP, are there any other IP addresses that should be allowed to access the registry (e.g. your laptop, a teammate, an office/VPN egress)?"* Add each as a `"<ip>/32"` entry (or a wider CIDR if the user gives one). The instance's own public IP must always be included so the deployment/post-deploy steps can reach the endpoints. Re-confirm the final list with the user before writing it.

**Note on mode flags:** `terraform.tfvars.example` leaves `enable_cloudfront` / `enable_route53_dns` **commented out**, and the variable default for `enable_route53_dns` is `true` (custom-domain). So you MUST explicitly write `enable_cloudfront = true` and `enable_route53_dns = false` — do not rely on the example's commented lines.

`base_domain` is **not required** in CloudFront Only mode — URLs use `*.cloudfront.net`.

By default the core image URIs point at public ECR (`public.ecr.aws/...`) — leave them unless deploying custom images. Demo servers and A2A agents are disabled by default (`enable_demo_servers = false`).

Edit `terraform.tfvars` using a safe method (Python or careful `sed`). Never echo passwords back to the user in plaintext output. Show the user the non-secret settings for confirmation before applying.

### Report the version and image source to the user

The image versions are NOT hardcoded in this skill — read them from the Terraform config so the reported version is always accurate (a `terraform.tfvars` override wins over the `variables.tf` default). Detect and report:

```bash
cd "${INSTALL_DIR}/terraform/aws-ecs"
for SVC in registry auth_server mcpgw; do
    OVERRIDE=$(grep -E "^[[:space:]]*${SVC}_image_uri[[:space:]]*=" terraform.tfvars 2>/dev/null | sed -E 's/.*=[[:space:]]*"([^"]+)".*/\1/')
    DEFAULT=$(grep -A3 "variable \"${SVC}_image_uri\"" variables.tf | grep -E "^[[:space:]]*default[[:space:]]*=" | sed -E 's/.*"([^"]+)".*/\1/')
    echo "${SVC}: ${OVERRIDE:-$DEFAULT}"
done
```

Tell the user explicitly, e.g.:

> "Looking at the Terraform files, this will install **MCP Gateway & Registry version `<TAG>`** (the image tag, e.g. `1.24.2`). The container images are pulled from the project's **public ECR** at `public.ecr.aws/p3v1o3c6/` — specifically `public.ecr.aws/p3v1o3c6/registry:<TAG>`, `.../auth-server:<TAG>`, and `.../mcpgw:<TAG>`. No image build is required; Terraform/ECS pulls these directly."

Use the actual tag and URIs detected by the command above — do not assume a specific version number.

**Also tell the user they can build their own custom images instead** (e.g. to ship local code changes), per `terraform/aws-ecs/README.md` → *Using Custom Images*:

> "If you want to run your own modified build instead of the public images:
> 1. Build and push to your private ECR:
>    ```bash
>    export AWS_REGION=${AWS_REGION}
>    make build-push            # run from the repo root; builds all images
>    ```
> 2. Override the image URIs in **`terraform/aws-ecs/terraform.tfvars`** to point at your account's ECR:
>    ```hcl
>    registry_image_uri    = \"<ACCOUNT_ID>.dkr.ecr.${AWS_REGION}.amazonaws.com/mcp-gateway-registry:latest\"
>    auth_server_image_uri = \"<ACCOUNT_ID>.dkr.ecr.${AWS_REGION}.amazonaws.com/mcp-gateway-auth-server:latest\"
>    mcpgw_image_uri       = \"<ACCOUNT_ID>.dkr.ecr.${AWS_REGION}.amazonaws.com/mcp-gateway-mcpgw:latest\"
>    ```
> 3. Re-run `terraform apply`.
>
> Building requires Docker (and Docker Buildx). The demo servers and A2A agents are not on public ECR — they must be built this way if you enable `enable_demo_servers = true`."

**Remind the user (again):** "These are the CloudFront settings. To switch to a custom domain later, change `enable_route53_dns`, `base_domain`, and `session_cookie_domain` in this same `terraform.tfvars` file — see Phase 4 / `terraform/aws-ecs/README.md` *Deployment Modes*."

Log: `{ 5, "terraform.tfvars Configured", DONE, "CloudFront mode, region, ingress, secrets set" }`

---

## Phase 6: Terraform Deploy (two-stage)

**Before applying, restate the AWS resources, cost, AND expected duration.** Show the user the "AWS services / resources this deploys" table above (VPC, ECS Fargate, Aurora PostgreSQL Serverless v2, DocumentDB, 2 ALBs, ACM, CloudFront, Secrets Manager, CloudWatch, SNS, Cloud Map, IAM, Auto Scaling, KMS, SSM), the ~$170–330/month cost warning, and tell them **the apply typically takes ~20–30 minutes to complete** (Aurora, DocumentDB, RDS Proxy, and ALB provisioning dominate; CloudFront distributions can add a few more minutes). Then use `AskUserQuestion` to get explicit confirmation to proceed. Do not apply until they confirm.

**Announce:** "Deploying infrastructure. This typically takes ~20–30 minutes to complete (RDS/DocumentDB/ALB provisioning is the slowest part). I'll run it in the background and give you a log file to tail."

```bash
cd "${INSTALL_DIR}/terraform/aws-ecs"
AWS="$(command -v aws || echo .venv/bin/aws)"  # ensure AWS creds resolve

terraform init -upgrade
echo "terraform init exit code: $?"
```

**Always redirect apply output to a timestamped `/tmp` log file and echo the path** so the user can `tail -f` it (apply takes ~20 min). Run the apply in the background and report the log path immediately.

**CloudFront Only mode → a single `terraform apply` is sufficient.** The ACM certificate resources are gated `count = var.enable_route53_dns ? 1 : 0`, so with `enable_route53_dns = false` they don't exist and there is **nothing to pre-create**. The two-stage apply below is **only** needed for custom-domain mode (`enable_route53_dns = true`).

```bash
cd "${INSTALL_DIR}/terraform/aws-ecs"
APPLY_LOG="/tmp/terraform-apply-$(date +%Y%m%d-%H%M%S).log"
echo "Terraform apply log: ${APPLY_LOG}   (follow with: tail -f ${APPLY_LOG})"
terraform apply -auto-approve -no-color > "${APPLY_LOG}" 2>&1
TF_RC=$?
echo "TERRAFORM_APPLY_EXIT=${TF_RC}" | tee -a "${APPLY_LOG}"
exit ${TF_RC}   # propagate the REAL terraform exit code
```

**IMPORTANT — do not mask the exit code.** Capture `$?` into `TF_RC` immediately and `exit ${TF_RC}` so a backgrounded apply reports its true status. Do NOT end the command with `echo ... | tee` as the last statement — a successful `echo`/`tee` returns 0 and hides a failed apply (observed: a failed apply was reported as "exit 0" because the trailing pipe succeeded). After it finishes, grep the log for `^Error:` / `Apply complete!` to decide success regardless of the reported code.

Confirm the plan with the user before this step. Use `-auto-approve` once they have confirmed (the redirect makes an interactive prompt impractical anyway). Echo the `${APPLY_LOG}` path to the user before/at launch.

**Custom-domain mode only (skip in CloudFront Only):** first-time custom-domain deploys are two-stage because of ACM/DNS validation dependencies — create the certs first, then the rest (redirect each stage to its own `/tmp` log and echo the path):

```bash
# Stage 1 — certs (custom-domain mode only):
STAGE1_LOG="/tmp/terraform-apply-certs-$(date +%Y%m%d-%H%M%S).log"
echo "Stage 1 log: ${STAGE1_LOG}"
terraform apply -auto-approve -no-color \
  -target=aws_acm_certificate.keycloak \
  -target=aws_acm_certificate.registry \
  -target=aws_acm_certificate_validation.keycloak \
  -target=aws_acm_certificate_validation.registry > "${STAGE1_LOG}" 2>&1
# Stage 2 — everything else:
STAGE2_LOG="/tmp/terraform-apply-$(date +%Y%m%d-%H%M%S).log"
echo "Stage 2 log: ${STAGE2_LOG}"
terraform apply -auto-approve -no-color > "${STAGE2_LOG}" 2>&1
```

If apply fails, surface the error to the user. Common causes: missing IAM permissions (revisit Phase 1), region mismatch with ECR images, or a missing/incorrect Route53 hosted zone.

Log: `{ 6, "Terraform Apply (2-stage)", DONE/FAILED, "certs + full stack applied" }`

---

## Phase 7: Post-Deployment Setup

**Announce:** "Running automated post-deployment setup (Keycloak realm, DocumentDB init, service restarts, endpoint verification)..."

Script: [`terraform/aws-ecs/scripts/post-deployment-setup.sh`](https://github.com/agentic-community/mcp-gateway-registry/blob/main/terraform/aws-ecs/scripts/post-deployment-setup.sh)

```bash
cd "${INSTALL_DIR}/terraform/aws-ecs"

export AWS_REGION="${AWS_REGION}"   # the DocumentDB-verified region from Step 0
# Password for the 'admin' user in the mcp-gateway realm (used to log into the Registry UI):
export INITIAL_ADMIN_PASSWORD="<choose a strong realm admin password>"

./scripts/post-deployment-setup.sh
echo "post-deployment-setup exit code: $?"
```

This saves terraform outputs, validates resources, waits for DNS, verifies ECS services, initializes Keycloak and DocumentDB, restarts registry/auth, and checks endpoints. Watch for `Passed: 8 / Failed: 0`.

**Two known first-deploy failure modes — see "Error Handling Reference" for the exact fix:**
- **Keycloak init times out / Keycloak crash-loops with "Access denied for user 'keycloak'"** — the `keycloak/database` rotation Lambda desynced the secret from the Aurora password during the create-time race. Fix: realign Aurora's master password to the secret and restart Keycloak, then re-run with `--skip-dns-wait --skip-restart`.
- **After this re-run, login fails with `?error=oauth2_callback_failed`** — Keycloak init regenerated the `mcp-gateway-web` client secret, but the running auth-server cached the old one. Fix: force-new-deployment on the auth-server AND registry, then `aws ecs wait services-stable`. (A clean single run restarts services after init and avoids this; you mainly hit it after re-running with `--skip-restart`.)

**Password distinction (state this clearly to the user):**
- `INITIAL_ADMIN_PASSWORD` (realm admin) → logs into the **Registry UI**.
- `keycloak_admin_password` (from terraform.tfvars) → logs into the **Keycloak admin console**.

Log: `{ 7, "Post-Deployment Setup", DONE/FAILED, "Passed X / Failed Y" }`

---

## Phase 8: Register the AWS KB Server

**Announce:** "Registering the AWS KB MCP server so the registry is not empty (this is the only server this skill registers)..."

**URL resolution (IMPORTANT for CloudFront mode):** in CloudFront Only mode the `registry_url` output is `null` — the live registry URL is `cloudfront_mcp_gateway_url`. Resolve in this priority order (same logic as `post-deployment-setup.sh`): `registry_url` → `cloudfront_mcp_gateway_url` → `mcp_gateway_url`.

```bash
cd "${INSTALL_DIR}"
OUTPUTS_FILE="terraform/aws-ecs/scripts/terraform-outputs.json"
export REGISTRY_URL=$(jq -r '.registry_url.value // .cloudfront_mcp_gateway_url.value // .mcp_gateway_url.value // empty' "$OUTPUTS_FILE")
export KEYCLOAK_URL=$(jq -r '.keycloak_url.value // empty' "$OUTPUTS_FILE")
echo "Registry: $REGISTRY_URL"; echo "Keycloak: $KEYCLOAK_URL"

# Register ONLY the AWS KB server (per deployment policy for this account).
# Do NOT register currenttime, cloudflare-docs, context7, or any other example server.
CFG="aws-kb-server.json"
if [ -f "cli/examples/${CFG}" ]; then
    echo "Registering cli/examples/${CFG} ..."
    uv run python api/registry_management.py register --config "cli/examples/${CFG}"
else
    echo "ERROR (not found): cli/examples/${CFG}"
fi
```

**Registration policy for this skill: only the AWS KB server (`cli/examples/aws-kb-server.json`) is registered.** Do not register the currenttime server or any other example servers/agents unless the user explicitly asks. (The README's broader example list — cloudflare-docs, context7, mcpgw, currenttime, demo agents — does not apply here.)

**Authentication:** with `REGISTRY_URL` and `KEYCLOAK_URL` exported, `register` fetches the JWT it needs dynamically (this is the same flow the README's post-deployment registration uses). **Do not reference or pass any local OAuth token files.** This phase is optional — if registration cannot authenticate in your environment, skip it and tell the user they can register the AWS KB server later from the Registry UI.

Verify:
```bash
cd "${INSTALL_DIR}"
uv run python api/registry_management.py list
```

Log: `{ 8, "Register AWS KB Server", DONE/SKIPPED, "aws-kb-server.json registered" }`

---

## Phase 9: Final Verification and Summary

**Announce:** "Running final verification and preparing your summary..."

**Read the URLs from the Terraform state/outputs.** The outputs JSON is written by `post-deployment-setup.sh` (or run `./scripts/save-terraform-outputs.sh`). You can also read straight from state with `terraform output`. Remember: in CloudFront Only mode `registry_url` is `null` — resolve `registry_url → cloudfront_mcp_gateway_url → mcp_gateway_url`.

```bash
cd "${INSTALL_DIR}/terraform/aws-ecs"
OUTPUTS_FILE="scripts/terraform-outputs.json"
# Regenerate the outputs file from current state if it is missing:
[ -f "$OUTPUTS_FILE" ] || terraform output -json > "$OUTPUTS_FILE"

REGISTRY_URL=$(jq -r '.registry_url.value // .cloudfront_mcp_gateway_url.value // .mcp_gateway_url.value // empty' "$OUTPUTS_FILE")
KEYCLOAK_URL=$(jq -r '.keycloak_url.value // empty' "$OUTPUTS_FILE")
KEYCLOAK_ADMIN_URL=$(jq -r '.keycloak_admin_console.value // empty' "$OUTPUTS_FILE")

for NAME_URL in "Registry health:${REGISTRY_URL}/health" "Keycloak:${KEYCLOAK_URL}"; do
    NAME="${NAME_URL%%:*}"; URL="${NAME_URL#*:}"
    STATUS=$(curl -s -o /dev/null -w "%{http_code}" "${URL}" 2>/dev/null || echo "000")
    echo "  ${NAME}: ${URL} — HTTP ${STATUS}"
done
echo "Registry URL:           ${REGISTRY_URL}"
echo "Keycloak Admin Console: ${KEYCLOAK_ADMIN_URL}"
```

**Display the complete step summary table** from the internal step log:

```
========================================
   MCP GATEWAY TERRAFORM DEPLOY COMPLETE
========================================

Environment:   ${RUN_ENV}        Mode: CloudFront Only
Region:        ${AWS_REGION}
Repo:          ${INSTALL_DIR}

+-------+--------------------------------------+----------+----------------------------+
| Phase | Name                                 | Status   | Notes                      |
+-------+--------------------------------------+----------+----------------------------+
|  0    | Environment & Mode Selection         | DONE     | env, CloudFront, dir       |
|  1    | AWS Credentials & IAM Check          | DONE     | identity confirmed         |
|  2    | Prerequisites Check                  | DONE     | tools present              |
|  3    | Clone + Toolchain Bootstrap          | DONE     | venv aws ready             |
|  4    | Custom Domain Info                   | SKIPPED  | CloudFront mode            |
|  5    | terraform.tfvars Configured          | DONE     | CloudFront/region/secrets  |
|  6    | Terraform Apply (2-stage)            | DONE     | full stack                 |
|  7    | Post-Deployment Setup                | DONE     | Passed X / Failed Y        |
|  8    | Register AWS KB Server               | DONE/SKIP| aws-kb-server only         |
|  9    | Final Verification                   | DONE     | endpoints healthy          |
+-------+--------------------------------------+----------+----------------------------+

Access Points:
  Registry UI (login here):  ${REGISTRY_URL}   (a *.cloudfront.net URL)
  Keycloak Admin Console:    ${KEYCLOAK_ADMIN_URL}

Login (Registry UI):
  Username: admin
  Password: the INITIAL_ADMIN_PASSWORD you set in Phase 7

Next steps:
  - Review logs:  cd terraform/aws-ecs && ./scripts/view-cloudwatch-logs.sh --filter "ERROR|FATAL|Exception"
  - See terraform/aws-ecs/README.md for user/group management and operations
```

### Walk the user through opening the UI (do this explicitly)

Using the values you just read from the outputs/state, give the user concrete click-here instructions — substitute the real resolved URLs, never leave placeholders:

> "**Open your registry:**
> 1. Click this URL (it's the CloudFront HTTPS address): **`${REGISTRY_URL}`**
> 2. You'll land on the login page. Log in with:
>    - **Username:** `admin`
>    - **Password:** the value you set as `INITIAL_ADMIN_PASSWORD` in Phase 7 (this is the **mcp-gateway realm** admin password — NOT the Keycloak master password).
> 3. After login you'll see the dashboard. If you ran Phase 8 it already lists the example servers/agents; otherwise it shows 0 servers and 0 agents.
>
> **Keycloak admin console** (for managing users/groups directly): **`${KEYCLOAK_ADMIN_URL}`** — log in with username `admin` and the `keycloak_admin_password` from `terraform.tfvars` (the master admin password, a different one)."

If `${REGISTRY_URL}` is empty, the outputs file is stale — regenerate it with `terraform output -json > scripts/terraform-outputs.json` and re-read, or run `./scripts/save-terraform-outputs.sh`.

### Leave the user with these links / FAQ

Present this as a closing reference list (these paths are in the cloned repo and on GitHub):

> **How do I... ?**
> - **Create human users:** `terraform/aws-ecs/README.md` → *Creating Human Users* (`user-create-human`), and `docs/auth-mgmt.md`.
> - **Create M2M / service accounts (for agents & automation):** `terraform/aws-ecs/README.md` → *Creating M2M Service Accounts* (`user-create-m2m`). Save the client secret immediately — it cannot be retrieved later.
> - **Create groups / control server access:** `terraform/aws-ecs/README.md` → *Creating Groups* (`import-group`).
> - **Generate a JWT token (human):** Log into the Registry UI and click the **Get JWT Token** button (top-left sidebar). Copy the token and save it into a `.token` file, then pass that file to the CLI with `--token-file`. See README → *Generating JWT Tokens*.
>   ```bash
>   # paste the token from the UI into api/.token, then:
>   uv run python api/registry_management.py \
>       --token-file api/.token \
>       --registry-url "${REGISTRY_URL}" \
>       list
>   ```
> - **Register a new MCP server:** save your UI token to a `.token` file, then
>   `uv run python api/registry_management.py --token-file api/.token --registry-url "${REGISTRY_URL}" register --config <your-server-config.json>` — examples in `cli/examples/`. See README → *Register Example MCP Servers*.
> - **Register a new A2A agent:** `uv run python api/registry_management.py --token-file api/.token --registry-url "${REGISTRY_URL}" agent-register --config <your-agent-card.json>`. See README → *Register Example A2A Agents*.
> - **Operations (SSH into tasks, logs, updates, rollbacks):** `terraform/aws-ecs/OPERATIONS.md`.
> - **View logs:** `cd terraform/aws-ecs && ./scripts/view-cloudwatch-logs.sh --filter "ERROR|FATAL|Exception"`.
> - **Full deployment guide:** `terraform/aws-ecs/README.md`.
> - **Project overview:** repo root `README.md`.
> - **GitHub:** https://github.com/agentic-community/mcp-gateway-registry (issues, docs, examples).

### Finally, remind the user how to move to a custom domain

> "This deployment is live on a CloudFront URL. If you later want your own domain, it's a config change only — edit `terraform/aws-ecs/terraform.tfvars`:
> - `enable_route53_dns = true`
> - `base_domain = \"your.domain\"`
> - `session_cookie_domain = \".your.domain\"`
> - (keep `enable_cloudfront = true`)
>
> then create a Route53 hosted zone for the domain and re-run the two-stage `terraform apply`. Full steps: `terraform/aws-ecs/README.md` *Deployment Modes* and `docs/deployment-modes.md`."

---

## Error Handling Reference

### `aws: command not found`
The AWS CLI ships in the project venv. After Phase 3's `uv sync`, use `.venv/bin/aws` or `uv run aws`. If `uv sync` itself failed on a native build, install `build-essential` and the matching `python3.X-dev` headers (Ubuntu) or `xcode-select --install` (macOS), then re-run `uv sync`.

### Terraform apply fails with AccessDenied / not authorized
The running identity is missing one or more permissions from Phase 1. On EC2, confirm the instance profile role carries the full action list; on a laptop, confirm your IAM user/role does. This skill does not create IAM roles — ask and I can guide you through creating the role + instance profile interactively.

### SSL certificate validation stuck
DNS validation can take 5–30 minutes and requires the Route53 hosted zone to be correct. Check `aws acm list-certificates` and `aws acm describe-certificate --certificate-arn <arn>`.

### ECS tasks not starting
Check `aws ecs describe-services` events and `aws ecs describe-tasks` stopped reasons. Common causes: ECR image pull failure (wrong region), insufficient CPU/memory, invalid env vars, or Secrets Manager access denied.

### Keycloak init times out / Keycloak crash-loops with "Access denied for user 'keycloak'"
Post-deployment Step 5 ("Initializing Keycloak") fails with *"Keycloak did not become ready within 5 minutes"*, and the Keycloak task is cycling (`runningCount=0`). This is NOT a warm-up timeout — it is a database auth failure. The stack enables a Secrets Manager rotation Lambda (`<name>-rotate-rds`) on the `keycloak/database` secret with rotate-on-create, which fires while Aurora is still provisioning during the first apply: it advances the **secret** but does not apply the new password to **Aurora**, so Keycloak reads the rotated password and Aurora rejects it.

Confirm and fix:
```bash
# Confirm the cause in the Keycloak app logs (dedicated "keycloak" ECS cluster):
aws logs filter-log-events --region "$AWS_REGION" --log-group-name /ecs/keycloak \
  --start-time $(( ($(date +%s) - 900) ))000 --query 'events[-20:].message' --output text \
  | tr '\t' '\n' | grep -iE 'Access denied|Failed to obtain JDBC'

# Realign Aurora's master password to the CURRENT secret value, then restart Keycloak:
SECRET_PW=$(aws secretsmanager get-secret-value --region "$AWS_REGION" --secret-id keycloak/database \
  --query SecretString --output text | jq -r '.password')
aws rds modify-db-cluster --region "$AWS_REGION" --db-cluster-identifier keycloak \
  --master-user-password "$SECRET_PW" --apply-immediately      # status -> resetting-master-credentials (~1 min)
# wait until PendingModifiedValues.MasterUserPassword clears, then:
aws ecs update-service --region "$AWS_REGION" --cluster keycloak --service keycloak --force-new-deployment
```
Then re-run post-deployment with `./scripts/post-deployment-setup.sh --skip-dns-wait --skip-restart` (with `AWS_REGION` + `INITIAL_ADMIN_PASSWORD` set). The repo now sets `rotate_immediately = false` on these rotation resources (`terraform/aws-ecs/secret-rotation-config.tf`) to prevent the create-time race; a stack deployed before that fix still has a 30-day rotation armed that can re-trigger this — fixing the rotation Lambda or disabling rotation on the secret is the durable remedy.

### Login fails with `?error=oauth2_callback_failed`
After login the browser lands on `https://<cloudfront>/login?error=oauth2_callback_failed`. The auth-server logs (`/ecs/<name>-auth-server`) show its server-side token exchange returning **401** on `POST .../realms/mcp-gateway/protocol/openid-connect/token`. Cause: `init-keycloak.sh` REGENERATES the `mcp-gateway-web` client secret and writes the new value to Secrets Manager (`<name>-keycloak-client-secret`) AND Keycloak — but the already-running auth-server still holds the OLD secret it loaded at startup. This happens whenever Keycloak init runs (or is re-run) AFTER the auth-server task started.

Fix: restart the services so they reload the current secret, then wait for stability:
```bash
aws ecs update-service --region "$AWS_REGION" --cluster <main-cluster> --service <name>-auth --force-new-deployment
aws ecs update-service --region "$AWS_REGION" --cluster <main-cluster> --service <name>-registry --force-new-deployment
aws ecs wait services-stable --region "$AWS_REGION" --cluster <main-cluster> --services <name>-auth <name>-registry
```
Note: a normal single post-deployment run restarts services AFTER Keycloak init, so it self-corrects. You typically only hit this when a Keycloak re-init was done with `--skip-restart`. Always restart auth-server + registry after any Keycloak (re-)init.

### DNS not resolving (custom domain modes)
Allow 5–10 minutes for propagation; test with `dig @8.8.8.8 registry.<region>.<base_domain>`.

---

## Important Rules

- **RUN_ENV, AWS_REGION (DocumentDB-verified), and INSTALL_DIR must be established in Step 0** before any other phase runs. The deployment mode is always CloudFront Only — never ask the user to choose a mode.
- **This skill never creates or modifies IAM roles automatically.** It states the required permissions and, only if the user explicitly asks, guides them through creating a role + instance profile interactively. The actual `aws iam` / instance-profile commands are intentionally not baked into this skill.
- **Always run commands from within `${INSTALL_DIR}`** (or `${INSTALL_DIR}/terraform/aws-ecs` for terraform commands).
- **Use `.venv/bin/aws` / `uv run aws`** when no system AWS CLI is present.
- **Confirm the terraform plan with the user** before approving each `terraform apply`, unless the user explicitly opted into unattended mode.
- **Never print passwords or secrets** in command output; show only non-secret settings for confirmation.
- **Default to public ECR images** — only build/push custom images if the user asks (`make build-push`).
- **Log every phase** to the internal step log and display the full table in Phase 9.
