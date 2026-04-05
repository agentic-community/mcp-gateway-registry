# AgentCore Auto-Registration -- Low-Level Design

*Created: 2026-04-03*
*Updated: 2026-04-04*

## Purpose

Discover AWS Bedrock AgentCore Gateways and Agent Runtimes in one or more AWS accounts and register them with the MCP Gateway Registry. Auth tokens for CUSTOM_JWT gateways are managed by a separate token refresher process that runs as a cron job or sidecar.

## Components

| Component | File | Runs |
|-----------|------|------|
| **Scanner + Registrar** | `cli/agentcore/` | On-demand or scheduled |
| **Token Refresher** | `cli/agentcore/token_refresher.py` | Cron every 45 min or sidecar |

```
                       Phase 1: Registration                    Phase 2: Token Refresh
                       (on-demand)                              (cron every 45 min)

  +----------------+     +--------------------+               +--------------------+
  | AgentCore API  |---->| Scanner+Registrar  |--register-->  | MCP Gateway        |
  | (AWS)          |     | cli/agentcore/     |               | Registry           |
  +----------------+     +--------------------+               +--------------------+
                              |                                      ^
                              | writes                               | PATCH auth_credential
                              v                                      |
                    +-------------------------+              +--------------------+
                    | token_refresh_manifest  |--read------->| Token Refresher    |
                    | .json (gitignored)      |              | token_refresher.py |
                    +-------------------------+              +--------------------+
                                                                     |
                                                              +------+------+
                                                              |             |
                                                         GET OIDC      POST token
                                                         discovery     endpoint
                                                              |             |
                                                         +----v-------------v----+
                                                         | IdP (Cognito, Auth0,  |
                                                         |  Okta, Entra, etc.)   |
                                                         +-----------------------+
```

---

## Background: What AgentCore Returns

Every CUSTOM_JWT gateway from the AgentCore API includes OIDC metadata:

```json
{
  "name": "customersupport-gw",
  "gatewayUrl": "https://gateway.example.com",
  "authorizerType": "CUSTOM_JWT",
  "authorizerConfiguration": {
    "customJWTAuthorizer": {
      "discoveryUrl": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_pnikLWYzO/.well-known/openid-configuration",
      "allowedClients": ["7kqi2l0n47mnfmhfapsf29ch4h"]
    }
  }
}
```

The `discoveryUrl` is an OIDC Discovery endpoint (standard across all providers). GETting it returns:

```json
{
  "issuer": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_pnikLWYzO",
  "token_endpoint": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_pnikLWYzO/oauth2/token",
  "jwks_uri": "..."
}
```

The `token_endpoint` works for standard OAuth2 `client_credentials` grant. This is identical across Cognito, Auth0, Okta, Entra, Keycloak -- no provider-specific code needed for token generation.

The IdP vendor is always identifiable from the `discoveryUrl`:

| IdP | Pattern in discoveryUrl |
|-----|------------------------|
| Cognito | `cognito-idp` |
| Auth0 | `auth0.com` |
| Okta | `okta.com` |
| Entra | `microsoftonline.com` |
| Keycloak | `/realms/` |

This matters because Cognito allows auto-retrieval of `client_secret` via the AWS API (`describe_user_pool_client`), while other providers require the secret to be configured as an environment variable.

---

## Phase 1: Scanner + Registrar

### CLI Interface

```bash
# List resources (discovery only, no registration)
uv run python -m cli.agentcore list \
    --region us-east-1 \
    --output json

# Dry run
uv run python -m cli.agentcore sync \
    --registry-url https://registry.example.com \
    --token-file .token \
    --region us-east-1 \
    --dry-run

# Register
uv run python -m cli.agentcore sync \
    --registry-url https://registry.example.com \
    --token-file .token \
    --region us-east-1

# Cross-account
uv run python -m cli.agentcore sync \
    --registry-url https://registry.example.com \
    --token-file .token \
    --region us-east-1 \
    --accounts 111122223333,444455556666
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--registry-url` | `REGISTRY_URL` env or `http://localhost` | Registry base URL |
| `--token-file` | `REGISTRY_TOKEN_FILE` env or `.token` | Path to registry auth token file |
| `--region` | `AWS_REGION` env or `us-east-1` | AWS region |
| `--timeout` | `30` | AWS API call timeout (seconds) |
| `--dry-run` | false | Preview without registering |
| `--overwrite` | false | Overwrite existing registrations |
| `--gateways-only` | false | Skip runtimes |
| `--runtimes-only` | false | Skip gateways |
| `--include-mcp-targets` | false | Register mcpServer gateway targets as separate servers |
| `--accounts` | current account | Comma-separated account IDs for cross-account |
| `--assume-role-name` | `AgentCoreSyncRole` | IAM role to assume in target accounts |
| `--output` | `text` | Output format: `text` or `json` |
| `--manifest` | `token_refresh_manifest.json` | Output path for token refresh manifest |
| `--visibility` | `internal` | Registration visibility |
| `--debug` | false | Enable DEBUG logging |

### Sequence Diagram

```
User              cmd_sync          Scanner         AgentCore API      RegistryClient
 |                   |                 |                  |                  |
 |-- sync ---------->|                 |                  |                  |
 |                   |                 |                  |                  |
 |                   |-- scan_gateways()                  |                  |
 |                   |---------------->|                  |                  |
 |                   |                 |-- list_gateways->|                  |
 |                   |                 |<-- gw summaries -|                  |
 |                   |                 |-- get_gateway() ->|                  |
 |                   |                 |<-- full gateway --|                  |
 |                   |                 |   (includes authorizerConfiguration)|
 |                   |                 |-- list_targets() ->|                |
 |                   |                 |<-- targets --------|                |
 |                   |<-- gateways ----|                  |                  |
 |                   |                                    |                  |
 |                   |  For each gateway:                 |                  |
 |                   |                                    |                  |
 |                   |  1. Build registration model       |                  |
 |                   |     - proxy_pass_url = gatewayUrl  |                  |
 |                   |     - auth_scheme = "bearer"       |                  |
 |                   |       (for CUSTOM_JWT and AWS_IAM) |                  |
 |                   |     - auth_credential = null       |                  |
 |                   |     - metadata includes:           |                  |
 |                   |       discovery_url                |                  |
 |                   |       allowed_clients              |                  |
 |                   |       idp_vendor                   |                  |
 |                   |                                    |                  |
 |                   |  2. Register with registry         |                  |
 |                   |-- POST /internal/services ---------------------->     |
 |                   |<-- 201 Created ----------------------------------|    |
 |                   |                                    |                  |
 |                   |  3. If CUSTOM_JWT: add to manifest entries            |
 |                   |                                    |                  |
 |                   |-- scan_runtimes()                  |                  |
 |                   |---------------->|                  |                  |
 |                   |                 |-- list_runtimes->|                  |
 |                   |                 |<-- runtimes -----|                  |
 |                   |<-- runtimes ----|                  |                  |
 |                   |                                    |                  |
 |                   |  For each runtime:                 |                  |
 |                   |  - MCP protocol -> register as MCP Server             |
 |                   |  - HTTP/A2A -> register as Agent                      |
 |                   |  (no token needed, health check    |                  |
 |                   |   falls back to ping)              |                  |
 |                   |                                    |                  |
 |                   |-- write token_refresh_manifest.json|                  |
 |                   |-- print summary                    |                  |
 |                   |                                    |                  |
 |<-- summary -------|                                    |                  |
```

### Files

| File | Lines (approx) | Responsibility |
|------|----------------|----------------|
| `cli/agentcore/__init__.py` | 10 | Package init |
| `cli/agentcore/__main__.py` | 7 | `python -m cli.agentcore` entry point |
| `cli/agentcore/sync.py` | ~300 | CLI parsing (`argparse`), `cmd_sync()`, `cmd_list()` |
| `cli/agentcore/discovery.py` | ~200 | `AgentCoreScanner` -- paginated AWS API calls |
| `cli/agentcore/registration.py` | ~500 | `RegistrationBuilder`, `SyncOrchestrator` |
| `cli/agentcore/models.py` | ~200 | Pydantic models, helper functions |

### Key Data Structures

#### AgentCoreScanner

```python
class AgentCoreScanner:
    """Scans AgentCore resources via boto3 bedrock-agentcore-control client."""

    def __init__(
        self,
        region: str,
        timeout: int = 30,
        session: boto3.Session | None = None,
    ) -> None: ...

    def scan_gateways(self) -> list[dict[str, Any]]:
        """List gateways, filter to READY, get details + targets."""
        ...

    def scan_runtimes(self) -> list[dict[str, Any]]:
        """List runtimes, filter to READY, get details + endpoints."""
        ...
```

Both methods paginate via `nextToken` and only return resources with `status == "READY"`.

#### RegistrationBuilder

Converts raw AWS dicts into registry registration models.

```python
class RegistrationBuilder:

    def __init__(
        self,
        region: str,
        visibility: str = "internal",
        session: boto3.Session | None = None,
    ) -> None:
        self.region = region
        self.visibility = visibility
        self.account_id = self._get_account_id()

    def build_gateway_registration(
        self,
        gateway: dict[str, Any],
    ) -> InternalServiceRegistration:
        """Build MCP Server registration from a gateway.

        Extracts OIDC metadata (discovery_url, allowed_clients, idp_vendor)
        from authorizerConfiguration and stores in metadata field.
        """
        name = gateway.get("name", gateway["gatewayId"])
        gateway_url = gateway.get("gatewayUrl", "")
        authorizer_type = gateway.get("authorizerType", "NONE")

        # Extract OIDC metadata for CUSTOM_JWT gateways
        authorizer_config = gateway.get("authorizerConfiguration", {})
        jwt_config = authorizer_config.get("customJWTAuthorizer", {})
        discovery_url = jwt_config.get("discoveryUrl", "")
        allowed_clients = jwt_config.get("allowedClients", [])
        idp_vendor = _detect_idp_vendor(discovery_url) if discovery_url else ""

        metadata = {
            "source": "agentcore-sync",
            "gateway_arn": gateway.get("gatewayArn"),
            "gateway_id": gateway.get("gatewayId"),
            "authorizer_type": authorizer_type,
            "region": self.region,
            "account_id": self.account_id,
        }

        if authorizer_type == "CUSTOM_JWT" and discovery_url:
            metadata["discovery_url"] = discovery_url
            metadata["allowed_clients"] = allowed_clients
            metadata["idp_vendor"] = idp_vendor

        return InternalServiceRegistration(
            path=f"/{_slugify(name)}",
            name=name,
            description=gateway.get("description", f"AgentCore Gateway: {name}"),
            proxy_pass_url=gateway_url,
            mcp_endpoint=gateway_url,
            auth_provider="bedrock-agentcore",
            auth_scheme=_get_auth_scheme(authorizer_type),
            supported_transports=["streamable-http"],
            tags=["agentcore", "gateway", "auto-registered"],
            overwrite=False,
            metadata=metadata,
        )

    def build_runtime_mcp_registration(
        self,
        runtime: dict[str, Any],
    ) -> InternalServiceRegistration:
        """Build MCP Server registration from a MCP-protocol runtime."""
        ...

    def build_runtime_agent_registration(
        self,
        runtime: dict[str, Any],
    ) -> AgentRegistration:
        """Build A2A Agent registration from an HTTP/A2A-protocol runtime."""
        ...
```

#### SyncOrchestrator

Coordinates scan, build, register, and manifest output.

```python
class SyncOrchestrator:
    """Orchestrates discovery, registration, and manifest generation.

    1. Scan gateways / runtimes via AgentCoreScanner
    2. Build registrations via RegistrationBuilder
    3. Register with the registry via RegistryClient
    4. Write token_refresh_manifest.json for CUSTOM_JWT gateways
    """

    def __init__(
        self,
        scanner: AgentCoreScanner,
        builder: RegistrationBuilder,
        registry_client: RegistryClient,
        dry_run: bool = False,
        overwrite: bool = False,
        include_mcp_targets: bool = False,
        output_format: str = "text",
        manifest_path: str = "token_refresh_manifest.json",
    ) -> None:
        self.scanner = scanner
        self.builder = builder
        self.registry = registry_client
        self.dry_run = dry_run
        self.overwrite = overwrite
        self.include_mcp_targets = include_mcp_targets
        self.output_format = output_format
        self.manifest_path = manifest_path
        self.results: list[dict[str, Any]] = []
        self._manifest_entries: list[dict[str, Any]] = []

    def sync_gateways(self) -> None:
        """Scan and register all gateways."""
        gateways = self.scanner.scan_gateways()
        for gateway in gateways:
            self._register_gateway(gateway)
            if self.include_mcp_targets:
                for target in gateway.get("targets", []):
                    self._register_target(gateway, target)

    def sync_runtimes(self) -> None:
        """Scan and register all runtimes."""
        runtimes = self.scanner.scan_runtimes()
        for runtime in runtimes:
            self._register_runtime(runtime)

    def write_manifest(self) -> None:
        """Write token_refresh_manifest.json for CUSTOM_JWT gateways."""
        if self.dry_run:
            logger.info(
                f"[DRY-RUN] Would write manifest with "
                f"{len(self._manifest_entries)} entries"
            )
            return

        if not self._manifest_entries:
            logger.info("No CUSTOM_JWT gateways -- skipping manifest")
            return

        with open(self.manifest_path, "w") as f:
            json.dump(self._manifest_entries, f, indent=2)

        logger.info(
            f"Wrote {len(self._manifest_entries)} entries "
            f"to {self.manifest_path}"
        )

    def print_summary(self) -> None:
        """Print sync summary in text or JSON format."""
        ...
```

#### `_register_gateway()` -- core registration logic

```python
def _register_gateway(
    self,
    gateway: dict[str, Any],
) -> None:
    """Register a single gateway with the registry."""
    gateway_name = gateway.get("name", gateway["gatewayId"])
    gateway_url = gateway.get("gatewayUrl", "")
    gateway_arn = gateway.get("gatewayArn", "")

    if not _validate_https_url(gateway_url, gateway_name):
        self.results.append({
            "resource_type": "gateway",
            "resource_name": gateway_name,
            "resource_arn": gateway_arn,
            "registration_type": "mcp_server",
            "path": f"/{_slugify(gateway_name)}",
            "status": "skipped",
            "message": "Invalid URL (must be HTTPS)",
        })
        return

    registration = self.builder.build_gateway_registration(gateway)
    registration.overwrite = self.overwrite

    result: dict[str, Any] = {
        "resource_type": "gateway",
        "resource_name": gateway_name,
        "resource_arn": gateway_arn,
        "registration_type": "mcp_server",
        "path": registration.service_path,
    }

    if self.dry_run:
        result["status"] = "dry_run"
        result["message"] = "Would register as MCP Server"
        self.results.append(result)
        self._collect_manifest_entry(gateway, registration.service_path)
        return

    try:
        self._register_service_with_retry(registration)
        result["status"] = "registered"
        result["message"] = "Successfully registered"
    except Exception as e:
        if _is_conflict_error(e) and not self.overwrite:
            result["status"] = "skipped"
            result["message"] = "Already registered (use --overwrite)"
        else:
            result["status"] = "failed"
            result["message"] = str(e)
            logger.error(f"Failed to register gateway: {e}")
        self.results.append(result)
        return

    self.results.append(result)
    self._collect_manifest_entry(gateway, registration.service_path)


def _collect_manifest_entry(
    self,
    gateway: dict[str, Any],
    server_path: str,
) -> None:
    """Add a CUSTOM_JWT gateway to the token refresh manifest."""
    if gateway.get("authorizerType") != "CUSTOM_JWT":
        return

    jwt_config = gateway.get("authorizerConfiguration", {}).get(
        "customJWTAuthorizer", {}
    )
    discovery_url = jwt_config.get("discoveryUrl", "")
    if not discovery_url:
        return

    self._manifest_entries.append({
        "server_path": server_path,
        "gateway_arn": gateway.get("gatewayArn", ""),
        "discovery_url": discovery_url,
        "allowed_clients": jwt_config.get("allowedClients", []),
        "idp_vendor": _detect_idp_vendor(discovery_url),
    })
```

### Helper Functions

```python
IDP_PATTERNS: dict[str, str] = {
    "cognito-idp": "cognito",
    "auth0.com": "auth0",
    "okta.com": "okta",
    "microsoftonline.com": "entra",
    "/realms/": "keycloak",
}


def _detect_idp_vendor(
    discovery_url: str,
) -> str:
    """Detect IdP vendor from OIDC discovery URL."""
    for pattern, vendor in IDP_PATTERNS.items():
        if pattern in discovery_url:
            return vendor
    return "unknown"


def _slugify(name: str) -> str:
    """Convert name to URL-safe slug."""
    ...


def _validate_https_url(url: str, resource_name: str) -> bool:
    """Validate that URL uses HTTPS."""
    ...


def _get_auth_scheme(authorizer_type: str) -> str:
    """Map AgentCore authorizer type to registry auth scheme.
    CUSTOM_JWT -> bearer, AWS_IAM -> bearer, NONE -> none.
    """
    ...
```

### Manifest File Format

Output: `token_refresh_manifest.json` (add to `.gitignore`)

```json
[
  {
    "server_path": "/customersupport-gw",
    "gateway_arn": "arn:aws:bedrock:us-east-1:015469603702:gateway/gw-abc",
    "discovery_url": "https://cognito-idp.us-east-1.amazonaws.com/us-east-1_pnikLWYzO/.well-known/openid-configuration",
    "allowed_clients": ["7kqi2l0n47mnfmhfapsf29ch4h"],
    "idp_vendor": "cognito"
  },
  {
    "server_path": "/enterprise-gw",
    "gateway_arn": "arn:aws:bedrock:us-east-1:015469603702:gateway/gw-def",
    "discovery_url": "https://myorg.okta.com/.well-known/openid-configuration",
    "allowed_clients": ["0oa1234567abcdefg"],
    "idp_vendor": "okta"
  }
]
```

### Runtime Registration

Runtimes are registered without tokens. The registry health check falls back to a ping for servers without `auth_credential`.

- **MCP protocol runtime** -> registered as MCP Server (via `InternalServiceRegistration`)
- **HTTP/A2A protocol runtime** -> registered as A2A Agent (via `AgentRegistration` with SigV4 security scheme)

No manifest entry is created for runtimes.

> **Note:** Agents imported from runtimes are registered with an empty skills array. To add skills, use the agent edit dialog in the UI or the `PUT /api/agents/{path}` API endpoint. Updating skills triggers a security rescan of the agent.

#### Agent Overwrite Handling

`AgentRegistration` does not have an `overwrite` field (unlike `InternalServiceRegistration`). When `--overwrite` is used and an agent already exists (409 Conflict), the orchestrator catches the conflict and calls `update_agent()` (PUT) to update the existing registration:

1. Attempt `register_agent()` (POST)
2. If 409 Conflict and `--overwrite` is set -> call `update_agent()` (PUT)
3. If 409 Conflict without `--overwrite` -> mark as "skipped"

### Cross-Account Support

For multi-account scanning, the CLI accepts `--accounts 111122223333,444455556666`. For each account:

1. `sts:AssumeRole` into `arn:aws:iam::{account}:role/{assume_role_name}`
2. Create a boto3 Session from the assumed role credentials
3. Pass that session to `AgentCoreScanner` and `RegistrationBuilder`
4. Register all discovered resources into the same registry

---

## Phase 2: Token Refresher

### File: `cli/agentcore/token_refresher.py`

Standalone script (~250 lines) that reads the manifest, resolves client secrets, fetches tokens, and updates the registry.

### CLI Interface

```bash
# One-time refresh (Cognito auto-retrieval needs no env vars)
uv run python -m cli.agentcore.token_refresher \
    --manifest token_refresh_manifest.json \
    --registry-url https://registry.example.com \
    --token-file .token

# With per-client env vars (highest priority)
OAUTH_CLIENT_SECRET_49ujl0b9ser72gnp6q1ph9v6vs=mysecret \
    uv run python -m cli.agentcore.token_refresher \
    --manifest token_refresh_manifest.json \
    --registry-url https://registry.example.com \
    --token-file .token

# With vendor-level env vars (fallback for non-Cognito IdPs)
AUTH0_CLIENT_SECRET=xxx OKTA_CLIENT_SECRET=yyy \
    uv run python -m cli.agentcore.token_refresher \
    --manifest token_refresh_manifest.json \
    --registry-url https://registry.example.com \
    --token-file .token

# Continuous mode (run as sidecar)
uv run python -m cli.agentcore.token_refresher \
    --manifest token_refresh_manifest.json \
    --registry-url https://registry.example.com \
    --token-file .token \
    --loop --interval 2700
```

**Flags:**

| Flag | Default | Description |
|------|---------|-------------|
| `--manifest` | `token_refresh_manifest.json` | Path to manifest file |
| `--registry-url` | `REGISTRY_URL` env or `http://localhost` | Registry base URL |
| `--token-file` | `REGISTRY_TOKEN_FILE` env or `.token` | Registry auth token file |
| `--loop` | false | Run continuously |
| `--interval` | `2700` (45 min) | Refresh interval in seconds |
| `--scan` / `--no-scan` | `--scan` (enabled) | Trigger security rescan after each credential update |
| `--debug` | false | Enable DEBUG logging |

### Sequence Diagram

```
token_refresher.py    Manifest       Env Vars       Cognito API    OIDC Discovery    Registry
       |                 |               |               |               |              |
       |-- read -------->|               |               |               |              |
       |<-- entries[] ---|               |               |               |              |
       |                                 |               |               |              |
       |  For each entry:                |               |               |              |
       |                                 |               |               |              |
       |  1. Resolve client_secret (3-tier priority)     |               |              |
       |     [Priority 1: per-client]    |               |               |              |
       |-- check OAUTH_CLIENT_SECRET_<id>|               |               |              |
       |     if found -> use it          |               |               |              |
       |                                 |               |               |              |
       |     [Priority 2: cognito auto]  |               |               |              |
       |     (if cognito + no per-client)---------------->|              |              |
       |     describe_user_pool_client() |               |               |              |
       |     <-- ClientSecret -----------|---------------|               |              |
       |                                 |               |               |              |
       |     [Priority 3: vendor env]    |               |               |              |
       |-- check AUTH0_/OKTA_/etc ------>|               |               |              |
       |                                 |               |               |              |
       |  2. Get token_endpoint          |               |               |              |
       |-- GET discovery_url ------------------------------------------>|              |
       |<-- {token_endpoint: "..."} ------------------------------------|              |
       |                                 |               |               |              |
       |  3. Get token (OAuth2 client_credentials)       |               |              |
       |-- POST token_endpoint ---------------------------------------->|              |
       |   {grant_type: client_credentials,              |               |              |
       |    client_id: allowed_clients[0],               |               |              |
       |    client_secret: from step 1}                  |               |              |
       |<-- {access_token: "eyJ..."} -----------------------------------|              |
       |                                 |               |               |              |
       |  4. Update registry             |               |               |              |
       |-- PATCH /api/servers/{path}/auth-credential --------------------------->      |
       |   {auth_scheme: "bearer", auth_credential: "eyJ..."}                          |
       |<-- 200 OK --------------------------------------------------------------------|
       |                                 |               |               |              |
       |  5. Trigger security rescan     |               |               |              |
       |-- POST /api/servers/{path}/rescan ---------------------------------------->   |
       |<-- scan results (is_safe, severity counts) ------------------------------------|
       |                                 |               |               |              |
       |  Write last_refreshed timestamp to manifest     |               |              |
```

### Client Secret Resolution (3-Tier Priority)

For each manifest entry, the token refresher resolves the client secret using this priority order:

| Priority | Method | Env Var / Mechanism | When Used |
|----------|--------|---------------------|-----------|
| **1** | Per-client env var | `OAUTH_CLIENT_SECRET_<client_id>=<secret>` | Any IdP -- overrides all other methods |
| **2** | Cognito auto-retrieval | `boto3.describe_user_pool_client()` | `cognito` only -- parses pool_id/region from discovery URL |
| **3** | Vendor-specific env var | `AUTH0_CLIENT_SECRET`, `OKTA_CLIENT_SECRET`, `ENTRA_CLIENT_SECRET`, `KEYCLOAK_CLIENT_SECRET` | Non-Cognito IdPs -- one secret shared across all gateways for that vendor |

If none of the tiers produce a secret, the entry is skipped with a warning.

**Per-client env var** (Priority 1) is useful when multiple gateways use the same IdP but have different client secrets. The env var name is `OAUTH_CLIENT_SECRET_` followed by the `client_id` (from `allowed_clients[0]` in the manifest).

**Cognito auto-retrieval** (Priority 2) parses region and pool_id from the discovery URL:
```
https://cognito-idp.us-east-1.amazonaws.com/us-east-1_pnikLWYzO/.well-known/openid-configuration
                    ^^^^^^^^^                ^^^^^^^^^^^^^^^^^
                    region                   user_pool_id
```
Then calls `describe_user_pool_client(UserPoolId=pool_id, ClientId=client_id)` to auto-retrieve the secret. Requires IAM permissions for `cognito-idp:DescribeUserPoolClient`.

**Vendor env vars** (Priority 3) are shared across all gateways for a given IdP. One secret per vendor.

| IdP Vendor | Env Var |
|------------|---------|
| `auth0` | `AUTH0_CLIENT_SECRET` |
| `okta` | `OKTA_CLIENT_SECRET` |
| `entra` | `ENTRA_CLIENT_SECRET` |
| `keycloak` | `KEYCLOAK_CLIENT_SECRET` |
| `unknown` | Skipped with warning |

### Code Structure

All private functions at the top, public functions below. One parameter per line. Modular functions (30-50 lines max).

#### Constants

```python
OIDC_DISCOVERY_TIMEOUT: int = 10
TOKEN_REQUEST_TIMEOUT: int = 15
REGISTRY_REQUEST_TIMEOUT: int = 15

IDP_PATTERNS: dict[str, str] = {
    "cognito-idp": "cognito",
    "auth0.com": "auth0",
    "okta.com": "okta",
    "microsoftonline.com": "entra",
    "/realms/": "keycloak",
}

IDP_SECRET_ENV_VARS: dict[str, str] = {
    "auth0": "AUTH0_CLIENT_SECRET",
    "okta": "OKTA_CLIENT_SECRET",
    "entra": "ENTRA_CLIENT_SECRET",
    "keycloak": "KEYCLOAK_CLIENT_SECRET",
}

ENV_VAR_PREFIX: str = "OAUTH_CLIENT_SECRET_"
```

#### Private Functions

```python
def _read_manifest(
    manifest_path: str,
) -> list[dict[str, Any]]:
    """Read token refresh manifest from JSON file."""
    ...


def _detect_idp_vendor(
    discovery_url: str,
) -> str:
    """Detect IdP vendor from OIDC discovery URL.
    Matches known patterns in the URL string.
    """
    for pattern, vendor in IDP_PATTERNS.items():
        if pattern in discovery_url:
            return vendor
    return "unknown"


def _get_cognito_client_secret(
    discovery_url: str,
    client_id: str,
) -> str | None:
    """Auto-retrieve client secret from Cognito.

    Parses user_pool_id and region from the discoveryUrl,
    calls describe_user_pool_client() via boto3.
    """
    # Parse: https://cognito-idp.{region}.amazonaws.com/{pool_id}/...
    region = discovery_url.split("cognito-idp.")[1].split(".amazonaws")[0]
    pool_id = discovery_url.split("amazonaws.com/")[1].split("/")[0]

    client = boto3.client("cognito-idp", region_name=region)
    response = client.describe_user_pool_client(
        UserPoolId=pool_id,
        ClientId=client_id,
    )
    return response["UserPoolClient"].get("ClientSecret")


def _get_client_secret(
    idp_vendor: str,
    discovery_url: str,
    client_id: str,
) -> str | None:
    """Resolve client secret using 3-tier priority:

    1. Per-client env var: OAUTH_CLIENT_SECRET_<client_id>
    2. Cognito auto-retrieval via AWS API (cognito only)
    3. Vendor env var: AUTH0_CLIENT_SECRET, OKTA_CLIENT_SECRET, etc.
    """
    # Priority 1: per-client env var (OAUTH_CLIENT_SECRET_<client_id>)
    env_var_name = f"{ENV_VAR_PREFIX}{client_id}"
    secret = os.environ.get(env_var_name)
    if secret:
        logger.info(f"Using client secret from env var {env_var_name}")
        return secret

    # Priority 2: Cognito auto-retrieval via AWS API
    if idp_vendor == "cognito":
        return _get_cognito_client_secret(discovery_url, client_id)

    # Priority 3: vendor-specific env var
    vendor_env_var = IDP_SECRET_ENV_VARS.get(idp_vendor)
    if not vendor_env_var:
        logger.warning(f"No env var mapping for IdP vendor: {idp_vendor}")
        return None

    secret = os.environ.get(vendor_env_var)
    if not secret:
        logger.warning(f"Env var {vendor_env_var} not set for {idp_vendor}")
    return secret


def _get_token_endpoint(
    discovery_url: str,
) -> str | None:
    """Fetch token_endpoint from OIDC discovery document.

    GETs the discoveryUrl and extracts the token_endpoint field.
    Standard OIDC -- works for all providers.
    """
    response = requests.get(discovery_url, timeout=OIDC_DISCOVERY_TIMEOUT)
    response.raise_for_status()
    return response.json().get("token_endpoint")


def _request_token(
    token_endpoint: str,
    client_id: str,
    client_secret: str,
) -> str | None:
    """Request access token via OAuth2 client_credentials grant."""
    response = requests.post(
        token_endpoint,
        headers={"Content-Type": "application/x-www-form-urlencoded"},
        data={
            "grant_type": "client_credentials",
            "client_id": client_id,
            "client_secret": client_secret,
        },
        timeout=TOKEN_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return response.json().get("access_token")


def _update_registry_credential(
    registry_url: str,
    registry_token: str,
    server_path: str,
    auth_credential: str,
) -> bool:
    """PATCH auth_credential for a server in the registry."""
    url = f"{registry_url.rstrip('/')}/api/servers{server_path}/auth-credential"
    response = requests.patch(
        url,
        headers={
            "Authorization": f"Bearer {registry_token}",
            "Content-Type": "application/json",
        },
        json={
            "auth_scheme": "bearer",
            "auth_credential": auth_credential,
        },
        timeout=REGISTRY_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    return True


def _load_registry_token(
    token_file: str,
) -> str:
    """Load registry auth token from JSON file.

    Supports two formats:
    - Flat: {"access_token": "..."} or {"token": "..."}
    - Nested: {"tokens": {"access_token": "..."}}
    """
    ...
```

#### Public Function

```python
def refresh_all(
    manifest_path: str,
    registry_url: str,
    registry_token: str,
    run_scan: bool = True,
) -> dict[str, Any]:
    """Refresh tokens for all entries in the manifest.

    For each CUSTOM_JWT gateway:
    1. Resolve client_secret (per-client env -> Cognito auto -> vendor env)
    2. GET discoveryUrl -> extract token_endpoint
    3. POST client_credentials grant -> get access_token
    4. PATCH auth_credential in the registry
    5. Trigger security rescan (if run_scan is True)

    Returns summary dict with success/failure/skipped/scan counts.
    """
    entries = _read_manifest(manifest_path)
    start_time = time.time()

    success_count = 0
    failure_count = 0
    skipped_count = 0

    for entry in entries:
        server_path = entry["server_path"]
        discovery_url = entry["discovery_url"]
        allowed_clients = entry.get("allowed_clients", [])
        idp_vendor = entry.get("idp_vendor") or _detect_idp_vendor(discovery_url)

        if not allowed_clients:
            logger.warning(f"No allowed_clients for {server_path} -- skipping")
            skipped_count += 1
            continue

        client_id = allowed_clients[0]

        # Step 1: Resolve client_secret
        client_secret = _get_client_secret(idp_vendor, discovery_url, client_id)
        if not client_secret:
            skipped_count += 1
            continue

        # Step 2: Get token_endpoint via OIDC discovery
        token_endpoint = _get_token_endpoint(discovery_url)
        if not token_endpoint:
            failure_count += 1
            continue

        # Step 3: Request token
        token = _request_token(token_endpoint, client_id, client_secret)
        if not token:
            failure_count += 1
            continue

        # Step 4: Update registry
        updated = _update_registry_credential(
            registry_url, registry_token, server_path, token
        )
        if updated:
            success_count += 1
            entry["last_refreshed"] = datetime.now(timezone.utc).isoformat()
        else:
            failure_count += 1

    # Update manifest with timestamps
    with open(manifest_path, "w") as f:
        json.dump(entries, f, indent=2)

    elapsed = time.time() - start_time
    summary = {
        "total": len(entries),
        "success": success_count,
        "failed": failure_count,
        "skipped": skipped_count,
        "elapsed_seconds": round(elapsed, 1),
    }
    logger.info(f"Token refresh complete: {json.dumps(summary)}")
    return summary
```

#### Main Function

```python
def main() -> None:
    """Parse arguments and run token refresh."""
    parser = argparse.ArgumentParser(
        description="Refresh auth tokens for AgentCore CUSTOM_JWT gateways",
    )
    parser.add_argument("--manifest", default="token_refresh_manifest.json")
    parser.add_argument("--registry-url", default=os.environ.get("REGISTRY_URL", "http://localhost"))
    parser.add_argument("--token-file", default=os.environ.get("REGISTRY_TOKEN_FILE", ".token"))
    parser.add_argument("--loop", action="store_true")
    parser.add_argument("--interval", type=int, default=2700)
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)

    registry_token = _load_registry_token(args.token_file)

    if args.loop:
        while True:
            refresh_all(args.manifest, args.registry_url, registry_token)
            time.sleep(args.interval)
    else:
        refresh_all(args.manifest, args.registry_url, registry_token)
```

### Cron Setup

```bash
# Refresh every 45 minutes (tokens typically expire in 60 min)
*/45 * * * * cd /app && uv run python -m cli.agentcore.token_refresher \
    --manifest token_refresh_manifest.json \
    --registry-url https://registry.example.com \
    --token-file .token \
    >> /var/log/token-refresher.log 2>&1
```

Or run as a sidecar with `--loop --interval 2700`.

### Registry API Requirement

The token refresher uses the PATCH endpoint:

```
PATCH /api/servers/{path}/auth-credential
Authorization: Bearer {registry_token}
Content-Type: application/json

{"auth_scheme": "bearer", "auth_credential": "eyJhbGciOiJSUzI1NiIs..."}
```

The registry encrypts the credential before storing (existing behavior for `auth_credential` on POST).

**Note:** If the registry token has expired, the PATCH will fail with an HTTP 500 from nginx (HTML response, not JSON). The token refresher detects this and logs a diagnostic message suggesting token regeneration.

---

## Example: 12 Gateways Across 3 IdPs

```
Gateways 1-5:   Cognito  (discoveryUrl contains "cognito-idp")
Gateways 6-8:   Auth0    (discoveryUrl contains "auth0.com")
Gateways 9-12:  Entra    (discoveryUrl contains "microsoftonline.com")
```

**Phase 1** -- `sync` registers all 12 gateways without tokens. Outputs manifest with 12 entries.

**Phase 2** -- `token_refresher` processes the manifest using 3-tier secret resolution:
- Gateways 1-5: detects `cognito`, auto-retrieves secret via `describe_user_pool_client()` (zero config)
- Gateways 6-8: detects `auth0`, reads `AUTH0_CLIENT_SECRET` from env (one secret for all 3)
- Gateways 9-12: detects `entra`, reads `ENTRA_CLIENT_SECRET` from env (one secret for all 4)
- For all 12: GETs discoveryUrl -> token_endpoint, POSTs client_credentials, PATCHes registry

**Total config needed**: two env vars (`AUTH0_CLIENT_SECRET`, `ENTRA_CLIENT_SECRET`). Cognito needs nothing.

**Override example**: If Auth0 gateway #7 uses a different client secret than gateways #6 and #8:
```bash
# Per-client override takes priority over AUTH0_CLIENT_SECRET
OAUTH_CLIENT_SECRET_gw7clientid=different-secret AUTH0_CLIENT_SECRET=shared-secret \
    uv run python -m cli.agentcore.token_refresher --manifest token_refresh_manifest.json
```

---

## Test Plan

| Test | What It Validates |
|------|-------------------|
| `test_detect_idp_vendor_cognito` | `_detect_idp_vendor()` returns `"cognito"` for cognito-idp URLs |
| `test_detect_idp_vendor_auth0` | Returns `"auth0"` for auth0.com URLs |
| `test_detect_idp_vendor_unknown` | Returns `"unknown"` for unrecognized URLs |
| `test_build_gateway_registration_custom_jwt` | Metadata includes `discovery_url`, `allowed_clients`, `idp_vendor` |
| `test_build_gateway_registration_none_auth` | Metadata does not include OIDC fields |
| `test_register_gateway_collects_manifest` | `_manifest_entries` populated for CUSTOM_JWT gateways |
| `test_register_gateway_no_manifest_for_iam` | `_manifest_entries` empty for AWS_IAM gateways |
| `test_write_manifest_creates_file` | JSON file written with correct structure |
| `test_write_manifest_dry_run_skips` | No file written in dry-run mode |
| `test_sync_gateways_end_to_end` | Full flow: scan -> register -> manifest (mocked AWS + registry) |
| `test_per_client_env_var_takes_priority` | `OAUTH_CLIENT_SECRET_<id>` takes priority over Cognito auto and vendor env |
| `test_get_cognito_client_secret` | Parses pool_id/region from URL, calls describe_user_pool_client |
| `test_get_client_secret_auth0_from_env` | Reads AUTH0_CLIENT_SECRET |
| `test_get_client_secret_missing_env` | Returns None, logs warning |
| `test_get_token_endpoint_from_discovery` | GETs discovery URL, extracts token_endpoint |
| `test_request_token_success` | Standard client_credentials grant |
| `test_update_registry_credential` | PATCHes auth_credential via `/api/servers/{path}/auth-credential` |
| `test_refresh_all_mixed_idps` | End-to-end: Cognito auto + Auth0 env + skip unknown |
| `test_refresh_all_writes_timestamps` | Manifest updated with last_refreshed |
| `test_runtime_no_manifest_entry` | Runtimes do not appear in manifest |
| `test_agent_conflict_with_overwrite_calls_update` | Agent `--overwrite` uses `update_agent()` PUT on conflict |
| `test_agent_conflict_without_overwrite_skips` | Agent conflict without `--overwrite` shows "skipped" |
