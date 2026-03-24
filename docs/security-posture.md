# Security Posture - Enterprise-Grade Security for MCP Gateway & Registry

**Last Updated:** March 13, 2026
**Version:** 1.0.16+

---

## Executive Summary

The MCP Gateway & Registry implements defense-in-depth security across all layers of the stack. Our comprehensive security approach ensures that enterprises can safely deploy AI agent infrastructure while maintaining compliance with industry standards and best practices.

This document outlines our security architecture, controls, and practices that make the MCP Gateway & Registry enterprise-ready.

### Security Pillars

1. **Infrastructure Security** - Multi-layered AWS security controls
2. **Data Protection** - Encryption at rest and in transit
3. **Identity & Access Management** - Enterprise SSO and fine-grained authorization
4. **Container Security** - Hardened container images following CIS benchmarks
5. **Application Security** - Secure coding practices with automated scanning
6. **Supply Chain Security** - Automated security analysis of third-party MCP servers
7. **Observability** - Comprehensive audit logging and monitoring

### Deployment Platforms

The MCP Gateway & Registry supports multiple deployment platforms. Security controls are categorized by applicability:

**🟦 ECS Deployment** - AWS ECS with Terraform (uses DocumentDB, ALB, CloudFront, Lambda)
**🟩 EKS Deployment** - Kubernetes/EKS with Helm (uses MongoDB-CE, Kubernetes native features)
**🟨 Universal** - Applies to all deployment platforms (containers, application code, authentication)

---

## Table of Contents

1. [Encryption & Key Management](#encryption--key-management)
2. [Secrets Management & Rotation](#secrets-management--rotation)
3. [Network Security](#network-security)
4. [Access Logging & Audit Trail](#access-logging--audit-trail)
5. [Container Hardening](#container-hardening)
6. [Kubernetes Security](#kubernetes-security)
7. [Application Security](#application-security)
8. [Supply Chain Security](#supply-chain-security)
9. [Identity & Access Management](#identity--access-management)
10. [Monitoring & Alerting](#monitoring--alerting)
11. [Security Testing & Validation](#security-testing--validation)
12. [Compliance & Standards](#compliance--standards)

---

## Encryption & Key Management

**🟦 ECS Deployment** | **🟨 Universal (TLS)**

### Encryption at Rest

**🟦 ECS Deployment Only**

All sensitive data is encrypted at rest using AWS Key Management Service (KMS) with customer-managed keys.

**Encrypted Resources:**
- **AWS Secrets Manager**: All secrets encrypted with dedicated KMS keys
  - DocumentDB database credentials
  - RDS PostgreSQL credentials
  - JWT signing keys
  - Session encryption keys
  - API tokens and service credentials
- **AWS Systems Manager Parameter Store**: All SecureString parameters encrypted
  - Admin passwords
  - Database connection strings
  - Configuration secrets
- **Amazon DocumentDB** (ECS): Cluster encrypted with customer-managed KMS key
- **Amazon RDS PostgreSQL** (ECS): Database encrypted with customer-managed KMS key
- **Amazon S3** (ECS): All buckets use server-side encryption (SSE-S3 or KMS)

**🟩 EKS Deployment:**
- **MongoDB-CE**: Uses Kubernetes secrets for credentials (can be encrypted with KMS via EKS encryption provider)
- **RDS PostgreSQL** (Keycloak): Same as ECS - encrypted with customer-managed KMS key

**KMS Key Architecture:**

Three dedicated KMS keys with distinct purposes:
1. **DocumentDB Key** (`alias/mcp-gateway-documentdb`)
   - Encrypts DocumentDB cluster
   - Encrypts DocumentDB credentials in Secrets Manager
   - Encrypts related SSM parameters

2. **RDS Key** (`alias/keycloak-rds`)
   - Encrypts RDS PostgreSQL database
   - Encrypts RDS credentials in Secrets Manager
   - Encrypts Keycloak configuration parameters

3. **Gateway Secrets Key** (module-specific)
   - Encrypts MCP Gateway application secrets
   - Encrypts JWT signing keys
   - Encrypts session encryption keys

**Key Management Features:**
- ✅ Automatic key rotation enabled (annual rotation)
- ✅ Restrictive key policies following least-privilege principle
- ✅ CloudTrail logging of all key usage
- ✅ Cross-account access controls
- ✅ Key deletion protection with 7-day waiting period

### Encryption in Transit

**🟨 Universal**

All network communication uses TLS encryption:

**TLS Configuration:**
- **External Traffic**: TLS 1.2+ enforced on all ALBs (ECS) / Ingress controllers (EKS) and CloudFront distributions
- **Internal Traffic**: TLS connections to DocumentDB (ECS) / MongoDB-CE (EKS) and RDS
- **API Communication**: HTTPS-only for all REST API endpoints
- **MCP Protocol**: Encrypted SSE (Server-Sent Events) over HTTPS

**S3 Bucket Policies** (ECS):
- TLS enforcement via bucket policies (deny all non-HTTPS requests)
- Applied to all S3 buckets (logs, artifacts, backups)

---

## Secrets Management & Rotation

**🟦 ECS Deployment** | **🟨 Universal (Application-Level)**

### Automated Secret Rotation

**🟦 ECS Deployment Only**

Credentials are automatically rotated on a 30-day schedule using AWS Lambda functions, eliminating manual password management and reducing credential exposure windows.

**Rotation Implementation:**

**DocumentDB Credentials (ECS):**
- Automated rotation Lambda function
- Updates master password in DocumentDB cluster
- Updates stored credentials in Secrets Manager
- Zero-downtime rotation with connection draining

**RDS PostgreSQL Credentials (ECS and EKS):**
- Automated rotation Lambda function
- Updates master password in RDS cluster
- Updates stored credentials in Secrets Manager
- Coordinated updates to application configurations

**Rotation Features (ECS):**
- ✅ 30-day automatic rotation schedule
- ✅ VPC-integrated Lambda functions (secure network access)
- ✅ CloudWatch logging for all rotation events
- ✅ Automatic rollback on rotation failure
- ✅ CloudWatch alarms for rotation failures

**🟩 EKS Deployment:**
- MongoDB-CE credentials stored in Kubernetes secrets
- Manual rotation recommended (can be automated with Kubernetes CronJobs)
- RDS credentials use same AWS Secrets Manager rotation as ECS

### Secrets Access Control

**🟦 ECS Deployment:**

**IAM-Based Access (ECS):**
- Secrets accessible only by authorized ECS task execution roles
- KMS key policies restrict decryption to specific IAM principals
- No secrets stored in environment variables or code

**🟩 EKS Deployment:**
- Kubernetes RBAC controls access to secrets
- IAM Roles for Service Accounts (IRSA) for AWS API access
- Secrets can be encrypted at rest with KMS via EKS encryption provider

**Application-Level Encryption (Universal):**
- Backend MCP server credentials encrypted with Fernet encryption
- JWT tokens signed with cryptographically secure keys
- Session data encrypted before storage

---

## Network Security

**🟦 ECS Deployment (AWS-specific)** | **🟨 Universal (Concepts)**

### Public Access Prevention

**🟦 ECS Deployment**

All storage resources are protected against public exposure:

**S3 Bucket Security:**
- Public access completely blocked on all buckets
- Bucket policies deny any public ACLs or policies
- Applied to:
  - ALB access logs bucket
  - CloudFront access logs bucket
  - CodeBuild artifacts bucket
  - Backup storage buckets

**Database Access:**
- **DocumentDB** (ECS): Cluster deployed in private subnets (no public endpoint)
- **MongoDB-CE** (EKS): Pod-to-pod communication within cluster, no external exposure
- **RDS PostgreSQL** (ECS/EKS): Deployed in private subnets (no public endpoint)
- Security groups (ECS) / Network Policies (EKS) allow connections only from authorized workloads

### Security Groups & Network Segmentation

**🟦 ECS Deployment**

**Principle of Least Privilege:**
- Dedicated security groups per service layer
- Ingress rules limited to specific ports and source security groups
- Egress rules restricted to required destinations only

**Security Group Architecture:**
```
[ALB Security Group]
  ↓ TCP 8080 (HTTP)
[Registry ECS Security Group]
  ↓ TCP 27017 (MongoDB)
[DocumentDB Security Group]

[ALB Security Group]
  ↓ TCP 8080 (HTTP)
[Auth Server ECS Security Group]
  ↓ TCP 5432 (PostgreSQL)
[RDS Security Group]
```

**Lambda Function Security (ECS):**
- Secret rotation Lambdas deployed in VPC
- Dedicated security group with minimal permissions
- Access to databases via security group rules only

**🟩 EKS Deployment**

**Kubernetes Network Policies:**
- Define ingress/egress rules for pods
- Restrict pod-to-pod communication
- Isolate application tiers (frontend, backend, database)
- Default deny-all with explicit allow rules

---

## Access Logging & Audit Trail

**🟦 ECS (Infrastructure Logs)** | **🟨 Universal (Application Logs)**

### Comprehensive Access Logging

All traffic to the platform is logged for security analysis and compliance.

**🟦 Application Load Balancer Logging (ECS):**
- **MCP Gateway ALB**: All HTTP/HTTPS requests logged to S3
- **Keycloak ALB**: All authentication traffic logged to S3
- **Log Format**: W3C Extended Log Format
- **Storage**: Dedicated S3 bucket with 90-day retention
- **Encryption**: SSE-S3 (AES-256) encryption

**🟦 CloudFront Access Logging (ECS):**
- **MCP Gateway Distribution**: All CDN requests logged
- **Keycloak Distribution**: All auth-related CDN traffic logged
- **Log Format**: W3C Extended Log Format (compressed .gz)
- **Storage**: Dedicated S3 bucket with separate prefixes per distribution
- **Retention**: 90-day lifecycle policy

**🟦 DocumentDB Audit Logging (ECS):**
- **Audit Events Captured**:
  - Authentication events (login attempts, failures)
  - Authorization decisions (access control checks)
  - DDL operations (schema changes, index creation)
  - User management (user creation, role assignments)
  - Administrative commands (cluster configuration changes)
- **Destination**: CloudWatch Logs (`/aws/docdb/mcp-gateway-registry/audit`)
- **Query**: CloudWatch Logs Insights for analysis

### Application Audit Logging

**🟨 Universal (All Deployments)**

**Registry Audit Log:**
- All API requests logged to DocumentDB (ECS) or MongoDB-CE (EKS)
- All MCP tool invocations logged
- User authentication events tracked
- Configuration changes recorded

**Audit Log Fields:**
- Timestamp (UTC with timezone)
- Username and session ID
- HTTP method and status code
- Request path and query parameters
- Response time and size
- User agent and source IP
- Error details (if applicable)

**Audit Features:**
- ✅ Searchable filters (username, method, status code, date range)
- ✅ Statistics dashboard (event counts, unique users, timelines)
- ✅ Export to CSV/JSONL for external analysis
- ✅ Automatic TTL-based retention (configurable, default 7 days)
- ✅ DocumentDB indexing for fast queries

---

## Container Hardening

**🟨 Universal (All Deployments)**

### CIS Docker Benchmark Compliance

All container images are hardened following CIS Docker Benchmark 4.1 requirements, regardless of deployment platform (ECS, EKS, Docker Compose).

**Non-Root User Execution:**

Every container runs as a non-privileged user (UID 1000):

```dockerfile
# Create non-root user early for security
RUN groupadd -g 1000 appuser && useradd -u 1000 -g appuser appuser

# Copy files with correct ownership (fast, secure)
COPY --from=builder --chown=appuser:appuser /app/.venv /app/.venv

# Switch to non-root user
USER appuser
```

**Container Images Secured (12 total):**
- Registry service (with nginx)
- Auth server
- MCP servers (3 variants: GPU, CPU, lightweight)
- Metrics service
- Keycloak
- Database initialization containers
- Grafana

**Security Controls Per Container:**
- ✅ Non-root user execution (CIS 4.1)
- ✅ No sudo package installed
- ✅ Health checks configured (CIS 4.6)
- ✅ Multi-stage builds (minimal attack surface)
- ✅ No build tools in runtime images
- ✅ Minimal base images (python:3.12-slim)

### Container Runtime Security

**Docker Compose Security Options:**

```yaml
services:
  registry:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    ports:
      - "80:8080"    # High port for non-root
      - "443:8443"   # High port for non-root
```

**Security Features:**
- `no-new-privileges:true` - Prevents privilege escalation
- `cap_drop: ALL` - Drops all Linux capabilities
- High port binding (8080, 8443) - Non-root operation
- Read-only root filesystem (where possible)

**MongoDB Capability Exception:**

MongoDB requires `SETUID` and `SETGID` capabilities because its entrypoint uses `gosu` to drop privileges from `root` to the `mongodb` user at startup. Without these capabilities, MongoDB fails with:

```text
error: failed switching to 'mongodb': operation not permitted
```

The correct least-privilege pattern is to drop all capabilities and then explicitly add back only the minimum required:

```yaml
  mongodb:
    security_opt:
      - no-new-privileges:true
    cap_drop:
      - ALL
    cap_add:
      - SETUID   # Required by gosu to switch to mongodb user at startup
      - SETGID   # Required by gosu to switch to mongodb group at startup
```

This follows CIS Docker Benchmark guidance: explicitly enumerate the minimum capabilities a container needs rather than leaving it with a broad default set.

### Image Supply Chain

**Image Signing & Verification:**
- Official images published to Docker Hub
- Versioned releases with semantic versioning
- Automated builds via GitHub Actions
- Container vulnerability scanning in CI/CD

---

## Kubernetes Security

**🟩 EKS Deployment Only**

### Pod Security Standards

All Kubernetes Pods implement Pod Security Standards (PSS) at the **Restricted** level - the most stringent security profile.

**Pod-Level Security Context:**

```yaml
spec:
  securityContext:
    runAsNonRoot: true
    runAsUser: 1000
    runAsGroup: 1000
    fsGroup: 1000
    seccompProfile:
      type: RuntimeDefault
```

**Container-Level Security Context:**

```yaml
containers:
  - name: container
    securityContext:
      allowPrivilegeEscalation: false
      runAsNonRoot: true
      runAsUser: 1000
      capabilities:
        drop:
          - ALL
```

**Security Controls:**
- ✅ **runAsNonRoot**: Prevents containers from running as root
- ✅ **Drop ALL Capabilities**: Removes all Linux capabilities
- ✅ **No Privilege Escalation**: Blocks privilege escalation attempts
- ✅ **Seccomp Profile**: Restricts system calls
- ✅ **Read-Only Root Filesystem**: Where application permits

**Helm Charts Secured:**
- Registry Deployment
- Auth Server Deployment
- MCP Gateway (mcpgw) Deployment
- MongoDB Configuration Job
- Keycloak Configuration Job

### EKS-Specific Security

When deployed to Amazon EKS:
- IAM Roles for Service Accounts (IRSA) for AWS API access
- EKS security group policies
- Pod Security Policy (PSP) enforcement (EKS < 1.25)
- Pod Security Standards (PSS) enforcement (EKS ≥ 1.25)
- Network policies for pod-to-pod communication

---

## Application Security

**🟨 Universal (All Deployments)**

### Secure Coding Practices

The application codebase follows secure coding standards validated by automated security scanning, regardless of deployment platform.

**Bandit Static Analysis:**

All Python code is continuously scanned with Bandit security linter to detect:
- SQL injection vulnerabilities
- Command injection risks
- Hardcoded credentials
- Insecure cryptographic functions
- Subprocess misuse
- Unsafe deserialization
- And 50+ other security patterns

**Security Issues Addressed:**

**Subprocess Security:**
- Always use list form (never `shell=True`)
- Validate command arguments against allowlists
- Add timeouts to prevent DoS
- Proper error handling with logging

```python
# Secure subprocess pattern
result = subprocess.run(
    ["nginx", "-s", "reload"],
    capture_output=True,
    text=True,
    timeout=5,
)
```

**SQL Injection Prevention:**
- Parameterized queries for all database operations
- Table/column name validation against allowlists
- No string interpolation in SQL statements

```python
# Secure SQL pattern
table = validate_table_name(table)  # Allowlist check
query = f"DELETE FROM {table} WHERE created_at < ?"
cursor.execute(query, (cutoff,))
```

**Request Timeout Protection:**
- All HTTP requests include timeout parameters
- Prevents resource exhaustion DoS attacks
- Default 30-second timeout for external APIs

**Secure Configuration:**
- No hardcoded credentials in code
- All sensitive config via environment variables
- Bind addresses configurable (default 127.0.0.1)
- TLS-only communication in production

### Dependency Management

**Vulnerability Scanning:**
- Automated dependency vulnerability scanning in CI/CD
- Regular updates for security patches
- Pinned versions for reproducible builds

**Python Dependencies:**
- `uv` package manager for fast, reproducible installs
- `pyproject.toml` for dependency management
- No pip cache to reduce image size

---

## Supply Chain Security

**🟨 Universal (All Deployments)**

### Automated Security Scanning

Third-party MCP servers, A2A agents, and Agent Skills are automatically scanned before being made available to users, regardless of deployment platform.

**Scanning Infrastructure:**

**MCP Server Scanning:**
- Scanner: [Cisco AI Defense MCP Scanner](https://github.com/cisco-ai-defense/mcp-scanner)
- Analyzers: YARA (pattern-based), LLM (semantic analysis)
- Detection: SQL injection, command injection, XSS, path traversal, hardcoded secrets

**A2A Agent Scanning:**
- Scanner: [Cisco AI Defense A2A Scanner](https://github.com/cisco-ai-defense/a2a-scanner)
- Analyzers: YARA, Heuristic, Spec validation, Endpoint analysis
- Detection: Protocol violations, malicious behaviors, security misconfigurations

**Agent Skills Scanning:**
- Scanner: [Cisco AI Defense Skill Scanner](https://github.com/cisco-ai-defense/cisco-ai-skill-scanner)
- Analyzers: Static analysis, Behavioral analysis, LLM semantic analysis
- Detection: Prompt injection, command injection, data exfiltration, social engineering

### Scanning Workflows

**1. Automatic Registration-Time Scanning:**

Every new MCP server/agent/skill is scanned before being enabled:
- Scan triggered automatically on registration
- Results analyzed for severity (Critical, High, Medium, Low)
- Safe items: Enabled immediately
- Unsafe items: Disabled with `security-pending` tag
- Detailed report saved for administrator review

**2. Manual On-Demand Scanning:**

Administrators can trigger scans via API or CLI:
```bash
# Rescan MCP server
curl -X POST /api/servers/{path}/rescan -H "Authorization: Bearer $TOKEN"

# Rescan A2A agent
curl -X POST /api/agents/{path}/rescan -H "Authorization: Bearer $TOKEN"

# Rescan Agent Skill
curl -X POST /api/skills/{path}/rescan -H "Authorization: Bearer $TOKEN"
```

**3. Periodic Registry Scanning:**

Comprehensive scans of all enabled servers on a schedule:
- Detects newly discovered vulnerabilities
- Generates executive security reports
- Tracks vulnerability trends over time

### Threat Detection

**Security Threats Detected:**
- SQL injection patterns
- Command injection vulnerabilities
- Cross-site scripting (XSS) vectors
- Path traversal attempts
- Hardcoded credentials and secrets
- Malicious code patterns
- Prompt injection attacks (skills)
- Data exfiltration risks
- Privilege escalation patterns
- SSRF vulnerabilities

**Automated Response:**
- Critical/High severity: Server/agent/skill automatically disabled
- Security-pending tag applied for admin review
- Detailed JSON report saved to `security_scans/` directory
- UI indicators (shield icons) show security status

For complete details, see [Security Scanner Documentation](security-scanner.md).

---

## Identity & Access Management

**🟨 Universal (All Deployments)**

### Enterprise Identity Integration

**Supported Identity Providers (All Deployments):**
- **Keycloak** (default, self-hosted)
- **Microsoft Entra ID** (Azure AD)
- **AWS Cognito**
- Any OIDC-compliant provider

**SSO Features:**
- Single Sign-On (SSO) with identity provider session
- Proper OIDC logout flow with `id_token_hint`
- Multi-factor authentication (MFA) support
- Conditional access policies (Entra ID)

### Authorization Model

**Role-Based Access Control (RBAC):**

**Admin Role:**
- Full system access and configuration
- User and group management
- Security scan triggers
- Audit log access
- System health monitoring

**User Role:**
- MCP server registration (own servers)
- Tool discovery and execution
- Dashboard and API access
- Limited configuration access

**Service Role:**
- API authentication with static tokens
- Registry API access (federation)
- Metrics collection and export

### Fine-Grained Access Control

**Scope-Based Permissions:**
- OAuth scopes for granular API access control
- Tool-level permissions (read, execute)
- Resource-level isolation (user can only manage own servers)
- IAM group-based tool access control

**Token Security:**
- JWT tokens signed with SECRET_KEY
- Short expiration windows (configurable)
- Secure cookie transmission (HttpOnly, Secure, SameSite)
- Rate limiting: 100 tokens per user per hour

**Session Security:**
- Session data encrypted with SECRET_KEY (Fernet)
- Secure cookie domain configuration
- HTTPS-only transmission (production)
- SameSite=Lax CSRF protection

For complete details, see [Fine-Grained Access Control](scopes.md).

---

## Monitoring & Alerting

**🟦 ECS (CloudWatch Alarms)** | **🟨 Universal (Metrics & Dashboards)**

### CloudWatch Alarms

**🟦 ECS Deployment Only**

Proactive monitoring with automated alerts for security-critical resources.

**KMS Monitoring (2 alarms):**
- KMS API throttling detection (DocumentDB key)
- KMS API throttling detection (RDS key)
- Threshold: >10 errors in 1 minute
- Impact: Prevents secret decryption failures

**DocumentDB Monitoring (1 alarm):**
- Audit log failure detection
- Threshold: >10 failures in 5 minutes
- Impact: Identifies compliance gaps

**S3 Cost Control (2 alarms):**
- ALB logs bucket size monitoring
- CloudFront logs bucket size monitoring
- Threshold: >100 GB
- Impact: Prevents unexpected costs

**WAF Attack Detection (4 alarms):**
- Blocked requests monitoring (both ALBs)
- Rate limit trigger detection (both ALBs)
- Threshold: Configurable per alarm type
- Impact: Early warning of attacks/DDoS

**Alarm Configuration:**
- Optional SNS topic for email/SMS notifications
- Alarms created but not intrusive if SNS not configured
- Multiple evaluation periods to reduce false positives
- `treat_missing_data: notBreaching` for newly created resources

**🟩 EKS Deployment:**
- Uses Kubernetes-native monitoring (Prometheus, Alertmanager)
- Pod resource monitoring via Kubernetes metrics server
- Custom Prometheus alerts for application and infrastructure

### Metrics & Observability

**🟨 Universal (All Deployments)**

**Prometheus Metrics:**
- Tool execution counters and duration histograms
- System resource usage (CPU, memory, connections)
- Authentication metrics (login, logout, token vending)
- Error rates and response times

**Grafana Dashboards (All Deployments):**
- MCP data-plane performance metrics
- System health and resource utilization
- Tool usage analytics
- Real-time performance monitoring

**🟦 Amazon Managed Prometheus (AMP) - ECS Deployment:**
- Native AWS integration for ECS deployments
- Metrics service collects and exports to AMP
- OpenTelemetry support for external platforms (Datadog, etc.)

**🟩 Prometheus - EKS Deployment:**
- Self-hosted Prometheus in Kubernetes cluster
- Metrics scraped from pods via ServiceMonitor CRDs
- Persistent storage for metrics retention

---

## Security Testing & Validation

**🟨 Universal (All Deployments)**

### Automated Security Testing

**Container Security Tests (All Deployments):**
- Test suite: `tests/security/test_container_security.py`
- Validates: USER directive, no sudo, HEALTHCHECK, environment config
- Coverage: 12 Dockerfiles × 7 test categories = 84 test cases

**Pre-Commit Hooks:**

Automated security checks before every commit:
```bash
# Hooks include:
- Ruff linter (security rules enabled)
- Bandit security scan
- MyPy type checking
- Trailing whitespace removal
- YAML/JSON validation
- Python syntax validation
- Shell script syntax validation
```

**Semgrep Static Analysis:**

Comprehensive multi-language static code analysis:
- **Languages**: Python, JavaScript/TypeScript, YAML, Terraform, Dockerfile
- **Rule Sets**:
  - SQL injection detection
  - JWT security validation
  - Secret detection (credentials, tokens, API keys)
  - Docker Compose security best practices
  - Terraform infrastructure security
  - Path traversal prevention
  - CSRF protection validation
- **Scan Coverage**: 162 initial findings → 25 actionable items (84% reduction)
- **Resolution Status**:
  - ✅ SQL injection - Column validation implemented in metrics service
  - ✅ Docker Compose - `security_opt` and `cap_drop` added to all services
  - ✅ Terraform secrets - KMS encryption enabled for all AWS Secrets Manager secrets
  - ✅ JWT verification - Confirmed secure (two-step validation pattern)
  - ✅ Path traversal - Fixed in CLI and API endpoints
- **False Positive Filtering**: `.semgrepignore` excludes docs and tests
- **Tracking**: GitHub Issue [#650](https://github.com/agentic-community/mcp-gateway-registry/issues/650)

**CI/CD Pipeline:**

GitHub Actions run on every pull request:
- Bandit security scan (fail on high/critical)
- Ruff linting with security rules
- Unit tests (701 tests)
- Integration tests (57 tests)
- Type checking with MyPy
- Container security validation

### Manual Security Testing

**Penetration Testing:**
- Recommended: Annual third-party penetration testing
- Internal security reviews before major releases
- Vulnerability disclosure program

**Security Audits:**
- Code review with security focus
- Infrastructure security assessment
- Compliance gap analysis

---

## Compliance & Standards

**🟨 Universal (All Deployments)**

### Industry Standards

**CIS Docker Benchmark (All Deployments):**
- ✅ 4.1: Non-root user execution
- ✅ 4.2: Health checks configured
- ✅ 4.3: No unnecessary packages
- ✅ 4.5: Environment security (PIP_NO_CACHE_DIR)
- ✅ 4.6: Security options in orchestration

**OWASP Top 10 (2021):**
- ✅ A01: Broken Access Control - IAM, RBAC, fine-grained permissions
- ✅ A02: Cryptographic Failures - KMS encryption, TLS everywhere
- ✅ A03: Injection - Parameterized queries, subprocess validation
- ✅ A05: Security Misconfiguration - Hardened defaults, security contexts
- ✅ A07: Authentication Failures - Enterprise SSO, MFA, proper session management
- ✅ A09: Logging Failures - Comprehensive audit logging, CloudWatch
- ✅ A10: SSRF - Input validation, URL allowlists

**Kubernetes Pod Security Standards (PSS):**
- ✅ Restricted level compliance (most stringent)
- ✅ runAsNonRoot enforcement
- ✅ All capabilities dropped
- ✅ No privilege escalation
- ✅ Seccomp profiles applied

### Compliance Frameworks

**SOC 2 Controls:**
- Encryption at rest and in transit
- Access control and authentication
- Audit logging and monitoring
- Change management and versioning
- Incident response procedures

**PCI-DSS:**
- Encryption of sensitive data
- Secure authentication mechanisms
- Network segmentation and firewalls
- Audit logging and monitoring
- Access control and least privilege

**HIPAA (Healthcare):**
- Data encryption (at rest and in transit)
- Access controls and authentication
- Audit controls and logging
- Integrity controls
- Transmission security

**GDPR (Data Protection):**
- Data encryption
- Access controls and consent management
- Audit trails
- Data retention policies (TTL-based)
- Right to erasure (data deletion capabilities)

---

## Verification & Validation

### Infrastructure Verification

**🟦 ECS Deployment**

**Verify KMS Encryption:**
```bash
# Check secret encryption
aws secretsmanager describe-secret \
  --secret-id mcp-gateway/documentdb/credentials \
  --query 'KmsKeyId'

# Check KMS key rotation
aws kms get-key-rotation-status \
  --key-id alias/mcp-gateway-documentdb
```

**Verify Access Logging:**
```bash
# Check ALB logs
aws s3 ls s3://mcp-gateway-{region}-{account}-alb-logs/ --recursive | head -20

# Check CloudFront logs
aws s3 ls s3://mcp-gateway-{region}-{account}-cloudfront-logs/ --recursive | head -20

# Check DocumentDB audit logs
aws logs describe-log-groups --log-group-name-prefix /aws/docdb
```

**Verify CloudWatch Alarms:**
```bash
# List all security alarms
aws cloudwatch describe-alarms \
  --alarm-name-prefix mcp-gateway \
  --query 'MetricAlarms[*].[AlarmName,StateValue]' \
  --output table
```

**🟩 EKS Deployment**

**Verify Pod Security Standards:**
```bash
# Check pod security context
kubectl get pod -n mcp-gateway <pod-name> -o jsonpath='{.spec.securityContext}'

# Check container security context
kubectl get pod -n mcp-gateway <pod-name> -o jsonpath='{.spec.containers[0].securityContext}'

# Verify non-root user
kubectl exec -n mcp-gateway <pod-name> -- whoami
# Expected output: appuser
```

**Verify Network Policies:**
```bash
# List network policies
kubectl get networkpolicies -n mcp-gateway

# Describe specific policy
kubectl describe networkpolicy <policy-name> -n mcp-gateway
```

**Verify Kubernetes Secrets:**
```bash
# Check if secrets are encrypted at rest (EKS encryption provider)
kubectl get secret -n mcp-gateway <secret-name> -o jsonpath='{.metadata.annotations}'
```

### Application Verification

**🟨 Universal (All Deployments)**

**Run Security Tests:**
```bash
# Container security tests
pytest tests/security/test_container_security.py -v

# Bandit security scan
uv run bandit -r registry/ auth_server/ api/ -ll

# Pre-commit checks
pre-commit run --all-files
```

**Verify Container Security:**
```bash
# Check non-root user
docker compose exec registry whoami
# Expected output: appuser

# Check security options
docker compose config | grep -A 5 "security_opt"
```

**Verify Supply Chain Security:**
```bash
# Check MCP server scan results
cat security_scans/{server-url}.json | jq '.tool_results[].is_safe'

# Trigger manual scan
curl -X POST /api/servers/{path}/rescan -H "Authorization: Bearer $TOKEN"
```

---

## Security Incident Response

### Incident Detection

**Monitoring Channels:**
- CloudWatch Alarms (immediate notification)
- Audit log anomaly detection
- Security scan failure alerts
- WAF blocked request spikes

### Response Procedures

**Severity Levels:**
- **Critical**: Data breach, system compromise, authentication bypass
- **High**: Unauthorized access, privilege escalation, DoS attack
- **Medium**: Suspicious activity, failed authentication spike, misconfiguration
- **Low**: Policy violation, informational security event

**Response Steps:**
1. **Detection**: Alert received via CloudWatch, logs, or monitoring
2. **Triage**: Assess severity and impact
3. **Containment**: Isolate affected resources, disable compromised accounts
4. **Investigation**: Review audit logs, analyze attack patterns
5. **Remediation**: Patch vulnerabilities, rotate credentials, update policies
6. **Recovery**: Restore services, verify security posture
7. **Post-Mortem**: Document incident, update procedures, implement preventions

### Security Contacts

**Report Security Vulnerabilities:**
- AWS Security: http://aws.amazon.com/security/vulnerability-reporting/
- Email: aws-security@amazon.com
- **Do NOT create public GitHub issues for security vulnerabilities**

**Security Updates:**
- Monitor [release notes](../release-notes/) for security patches
- Subscribe to [GitHub Security Advisories](https://github.com/agentic-community/mcp-gateway-registry/security/advisories)

---

## Summary

The MCP Gateway & Registry implements enterprise-grade security across all layers:

✅ **Encryption Everywhere** - At rest (KMS) and in transit (TLS)
✅ **Zero-Trust Architecture** - Identity verification, least-privilege access
✅ **Defense-in-Depth** - Multiple security layers at infrastructure, application, and container levels
✅ **Automated Secrets Management** - 30-day rotation, encrypted storage
✅ **Comprehensive Logging** - ALB, CloudFront, DocumentDB, application audit logs
✅ **Supply Chain Security** - Automated scanning of third-party MCP servers
✅ **Container Hardening** - CIS benchmark compliance, non-root execution
✅ **Proactive Monitoring** - CloudWatch alarms, Prometheus metrics, Grafana dashboards
✅ **Compliance Ready** - SOC 2, PCI-DSS, HIPAA, GDPR controls

This security posture enables enterprises to confidently deploy AI agent infrastructure while maintaining regulatory compliance and protecting sensitive data.

### Security Controls by Deployment Platform

| Security Control | ECS | EKS | Universal |
|------------------|-----|-----|-----------|
| **KMS Encryption (AWS Secrets Manager, SSM)** | ✅ | ⚠️ Optional* | ❌ |
| **Automated Secret Rotation (Lambda)** | ✅ | ⚠️ RDS only | ❌ |
| **ALB Access Logging** | ✅ | ⚠️ Ingress logs | ❌ |
| **CloudFront Logging** | ✅ | ✅ | ❌ |
| **DocumentDB Audit Logging** | ✅ | ❌ | ❌ |
| **MongoDB-CE Audit Logging** | ❌ | ⚠️ Optional* | ❌ |
| **CloudWatch Alarms** | ✅ | ⚠️ Custom | ❌ |
| **S3 Security (Public Block, TLS)** | ✅ | ⚠️ If used | ❌ |
| **Security Groups** | ✅ | ❌ | ❌ |
| **Kubernetes Network Policies** | ❌ | ✅ | ❌ |
| **Pod Security Standards (PSS)** | ❌ | ✅ | ❌ |
| **Container Hardening (CIS)** | ✅ | ✅ | ✅ |
| **Non-Root Containers** | ✅ | ✅ | ✅ |
| **Application Security (Bandit)** | ✅ | ✅ | ✅ |
| **Supply Chain Security (Scanners)** | ✅ | ✅ | ✅ |
| **IAM / RBAC** | ✅ | ✅ | ✅ |
| **Enterprise SSO (OIDC)** | ✅ | ✅ | ✅ |
| **Application Audit Logging** | ✅ | ✅ | ✅ |
| **Prometheus Metrics** | ✅ | ✅ | ✅ |
| **Grafana Dashboards** | ✅ | ✅ | ✅ |

**Legend:**
- ✅ Fully supported and implemented
- ⚠️ Partially supported or requires configuration
- ❌ Not applicable for this platform
- *EKS can optionally use KMS for Kubernetes secrets encryption via encryption provider
- *MongoDB-CE audit logging can be enabled in configuration

**Key Differences:**
- **ECS**: Uses AWS-native services (ALB, DocumentDB, Secrets Manager, Lambda, CloudWatch)
- **EKS**: Uses Kubernetes-native features (Network Policies, PSS, Ingress, MongoDB-CE)
- **Universal**: Application-level controls work across all platforms

---

## References

### Documentation
- [Security Scanner Documentation](security-scanner.md) - Supply chain security for MCP servers
- [Fine-Grained Access Control](scopes.md) - Permission management
- [Audit Logging](audit-logging.md) - Comprehensive event tracking
- [Authentication Guide](auth.md) - Identity provider integration
- [Configuration Reference](configuration.md) - Security configuration options

### Standards & Frameworks
- [CIS Docker Benchmark](https://www.cisecurity.org/benchmark/docker) - Container security standards
- [OWASP Top 10](https://owasp.org/www-project-top-ten/) - Application security risks
- [Kubernetes Pod Security Standards](https://kubernetes.io/docs/concepts/security/pod-security-standards/) - Pod security profiles
- [AWS Security Best Practices](https://docs.aws.amazon.com/security/) - Cloud security guidance
- [Bandit Security Linter](https://bandit.readthedocs.io/) - Python security scanning

### Security Tools
- [Cisco AI Defense MCP Scanner](https://github.com/cisco-ai-defense/mcp-scanner) - MCP server security analysis
- [Cisco AI Defense A2A Scanner](https://github.com/cisco-ai-defense/a2a-scanner) - Agent security analysis
- [Cisco AI Defense Skill Scanner](https://github.com/cisco-ai-defense/cisco-ai-skill-scanner) - Agent Skills security analysis

---
