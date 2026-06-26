import logging
from datetime import UTC
from enum import Enum
from pathlib import Path
from typing import Annotated

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

# Accepted values for STORAGE_BACKEND. Keep in sync with the Terraform allowlist
# at terraform/aws-ecs/variables.tf (issue #955) so both layers reject the same
# typos with the same messages.
ALLOWED_STORAGE_BACKENDS: frozenset[str] = frozenset(
    {
        "documentdb",
        "mongodb-ce",
        "mongodb",
        "mongodb-atlas",
    }
)


# MongoDB-compatible backends. All values in this set route to the same
# DocumentDB/MongoDB repositories via the factory; documentdb is retained
# only to preserve AWS DocumentDB-specific SCRAM-SHA-1 auth selection in
# utils/mongodb_connection.py.
MONGODB_BACKENDS: frozenset[str] = frozenset(
    {
        "documentdb",
        "mongodb-ce",
        "mongodb",
        "mongodb-atlas",
    }
)


class DeploymentMode(str, Enum):
    """Deployment mode options."""

    WITH_GATEWAY = "with-gateway"
    REGISTRY_ONLY = "registry-only"


class RegistryMode(str, Enum):
    """Registry operating modes."""

    FULL = "full"
    SKILLS_ONLY = "skills-only"
    MCP_SERVERS_ONLY = "mcp-servers-only"
    AGENTS_ONLY = "agents-only"


class InternalDeploymentType(str, Enum):
    """Classification of an internal/workshop deployment (telemetry label only)."""

    NONE = "none"
    DEV = "dev"
    WORKSHOP = "workshop"
    OTHER = "other"


class Settings(BaseSettings):
    """Application settings with environment variable support."""

    model_config = SettingsConfigDict(
        env_file=".env",
        case_sensitive=False,
        extra="ignore",  # Ignore extra environment variables
    )

    # Network binding: default "0.0.0.0" binds to all IPv4 interfaces inside
    # the container (works everywhere). Opt into IPv6 dual-stack by setting
    # BIND_HOST=:: — but note that requires net.ipv6.bindv6only=0 on the host
    # AND an IPv6 loopback in the container image, which the stock Docker
    # image does not have. See docs/TELEMETRY.md and
    # docs/unified-parameter-reference.md (Group 4) for details.
    bind_host: str = "0.0.0.0"  # nosec B104 - bind to all IPv4 interfaces inside container

    # Auth settings
    secret_key: str = ""
    session_cookie_name: str = "mcp_gateway_session"
    session_max_age_seconds: int = 60 * 60 * 8  # 8 hours
    session_cookie_secure: bool = False  # Set to True in production with HTTPS
    session_cookie_domain: str | None = None  # e.g., ".example.com" for cross-subdomain sharing
    auth_server_url: str = "http://localhost:8888"
    auth_server_external_url: str = "http://localhost:8888"  # External URL for OAuth redirects
    auth_provider: str = "cognito"  # Auth provider: cognito, keycloak, entra, github
    registry_static_token_auth_enabled: bool = False  # Enable static token auth (IdP-independent)
    registry_api_token: str = ""  # Static API token for registry access
    registry_api_keys: str = ""  # Multi-key static tokens JSON (Issue #779)
    max_tokens_per_user_per_hour: int = 100  # JWT token vending rate limit
    ide_oauth_client_id: str = Field(
        default="",
        description=(
            "Pre-registered public OAuth client_id that IDEs (e.g. Cursor) use to "
            "initiate the gateway login flow. When set, the server's Connect config "
            "advertises this client_id and omits the static gateway token, so the IDE "
            "shows a login button and runs the OAuth/PKCE flow. Use this when anonymous "
            "Dynamic Client Registration is disabled on the auth provider and a fixed "
            "public client is registered instead. Empty (default) keeps the legacy "
            "static-token Connect config."
        ),
    )
    ide_oauth_callback_port: int = Field(
        default=0,
        description=(
            "Fixed loopback callback port the IDE should use for the OAuth login "
            "redirect (http://localhost:<port>/callback). Needed for IdPs that match "
            "the redirect_uri literally including the port (Okta, Entra, Cognito): the "
            "operator registers http://localhost:<port>/callback on the public client, "
            "and the Connect dialog emits the matching --callback-port so the IDE does "
            "not pick a random port. 0 (default) omits it, which is correct for Keycloak "
            "(wildcard loopback redirect) and DCR flows. Only used when "
            "ide_oauth_client_id (or a per-server oauth_client_id) is set."
        ),
        ge=0,
        le=65535,
    )

    # Registration webhook settings (Issue #742)
    registration_webhook_url: str | None = Field(
        default=None,
        description="Webhook URL to POST to on successful registration or deletion. Disabled if not set.",
    )
    registration_webhook_auth_header: str = Field(
        default="Authorization",
        description="Auth header name for webhook requests (e.g., Authorization, X-API-Key)",
    )
    registration_webhook_auth_token: str | None = Field(
        default=None,
        description="Auth token for webhook. If header is Authorization, Bearer is auto-prepended.",
    )
    registration_webhook_timeout_seconds: int = Field(
        default=10,
        description="Timeout for webhook HTTP calls in seconds",
    )
    registration_webhook_signing_secret: str | None = Field(
        default=None,
        description=(
            "Shared secret for HMAC-SHA256 signing of outbound registration "
            "webhook payloads. When set, an X-Registry-Signature header is added. "
            "When unset, no signature is sent (current behavior)."
        ),
    )

    # Lifecycle workflow settings (Issue #1330)
    registration_enforced_status: str | None = Field(
        default=None,
        description=(
            "When set (e.g. 'draft'), mandates the initial lifecycle status for all "
            "new asset registrations. A registration with no status is forced to this "
            "value; a registration with a different explicit status fails with 4xx. "
            "Unset = current behavior (new assets default to 'active')."
        ),
    )

    # Registration Gate Configuration (Admission Control, Issue #809)
    registration_gate_enabled: bool = Field(
        default=False,
        description="Enable registration gate (admission control webhook) for all asset registrations and updates",
    )
    registration_gate_url: str = Field(
        default="",
        description="URL of the registration gate endpoint (HTTPS recommended, HTTP triggers warning)",
    )
    registration_gate_auth_type: str = Field(
        default="none",
        description="Auth type for gate endpoint: 'none', 'api_key', 'bearer', or 'oauth2_client_credentials'",
    )
    registration_gate_auth_credential: str = Field(
        default="",
        description="API key or Bearer token for authenticating with the gate endpoint",
    )
    registration_gate_auth_header_name: str = Field(
        default="X-Api-Key",
        description="HTTP header name for API key auth (only used when auth_type='api_key')",
    )
    registration_gate_timeout_seconds: int = Field(
        default=5,
        description="HTTP request timeout in seconds for each gate call attempt",
    )
    registration_gate_max_retries: int = Field(
        default=2,
        description="Maximum retry attempts for gate calls on transient failures",
    )

    # Registration Gate OAuth2 Client Credentials (Issue #917)
    registration_gate_oauth2_token_url: str = Field(
        default="",
        description="OAuth2 token endpoint URL for client credentials flow",
    )
    registration_gate_oauth2_client_id: str = Field(
        default="",
        description="OAuth2 client ID for client credentials flow",
    )
    registration_gate_oauth2_client_secret: str = Field(
        default="",
        description="OAuth2 client secret for client credentials flow",
    )
    registration_gate_oauth2_scope: str = Field(
        default="",
        description="OAuth2 scope parameter (e.g., api://app-id/.default for Entra)",
    )

    # Embeddings settings [Default]
    embeddings_provider: str = "sentence-transformers"  # 'sentence-transformers' or 'litellm'
    embeddings_model_name: str = "all-MiniLM-L6-v2"
    embeddings_model_dimensions: int = 384  # 384 for default and 1024 for bedrock titan v2

    # HNSW vector search tuning (only used with DocumentDB backend)
    # Higher efSearch improves recall at the cost of query latency.
    # Default 40 may miss documents in small collections; 100 gives near-exact recall.
    vector_search_ef_search: int = 100

    # Search fusion method: 'rrf' (Reciprocal Rank Fusion, industry standard)
    # or 'legacy' (previous additive formula). RRF avoids score saturation and
    # handles missing embeddings gracefully.
    search_fusion_method: str = "rrf"

    # Custom entity types (admin-defined schema-driven catalog types)
    custom_entity_types_enabled: bool = Field(
        default=False,
        description=(
            "Enable admin-defined custom catalog entity types. This is a "
            "feature-invisible kill-switch, not a pause-new-usage flag: when "
            "false the CRUD routers are unregistered (404), the tabs/config "
            "list is empty, and custom types are excluded from both the search "
            "scope and result processing. Existing records remain in the DB but "
            "are unreachable until the flag is re-enabled."
        ),
    )
    custom_type_cache_ttl_seconds: float = Field(
        default=60.0,
        description="TTL for the in-process custom-type descriptor cache (seconds)",
    )
    max_custom_records_per_type: int = Field(
        default=1000,
        description=(
            "Soft cap on records per custom type (0 = unlimited). Best-effort: "
            "the count-then-create check may overshoot slightly under concurrent "
            "creates; it is not a hard transactional guarantee. Guards against "
            "runaway imports hitting the embedding-collection scaling ceiling."
        ),
    )
    max_custom_types: int = Field(
        default=50,
        description=(
            "Cap on the number of custom entity TYPES an admin can define "
            "(0 = unlimited). Guards against unbounded type creation, each of "
            "which carries its own embedding collection. Enforced at type create."
        ),
    )

    # LiteLLM-specific settings (only used when embeddings_provider='litellm')
    # For Bedrock: Set to None and configure AWS credentials via standard methods
    # (IAM roles, AWS_ACCESS_KEY_ID/AWS_SECRET_ACCESS_KEY env vars, or ~/.aws/credentials)
    embeddings_api_key: str | None = None
    embeddings_secret_key: str | None = None
    embeddings_api_base: str | None = None
    embeddings_aws_region: str | None = "us-east-1"

    # Health check settings
    health_check_interval_seconds: int = (
        300  # 5 minutes for automatic background checks (configurable via env var)
    )
    health_check_timeout_seconds: int = 2  # Very fast timeout for user-driven actions

    # WebSocket performance settings
    max_websocket_connections: int = 100  # Reasonable limit for development/testing
    websocket_send_timeout_seconds: float = 2.0  # Allow slightly more time per connection
    websocket_broadcast_interval_ms: int = 10  # Very responsive - 10ms minimum between broadcasts
    websocket_max_batch_size: int = 20  # Smaller batches for faster updates
    websocket_cache_ttl_seconds: int = 1  # 1 second cache for near real-time user feedback

    # Well-known discovery settings
    enable_wellknown_discovery: bool = True
    wellknown_cache_ttl: int = 300  # 5 minutes

    # ARD Catalog Publisher settings (issue #1294)
    # Publishes /.well-known/ai-catalog.json per the Agentic Resource Discovery spec.
    ard_catalog_enabled: bool = Field(
        default=True,
        description="Enable the /.well-known/ai-catalog.json ARD publisher endpoint",
    )
    ard_registry_enabled: bool = Field(
        default=True,
        description=(
            "Enable the ARD Registry adapter (POST /api/ard/search, GET /api/ard/agents) "
            "and the self application/ai-registry+json catalog entry. Independent of the "
            "Phase 1 publisher toggle ard_catalog_enabled."
        ),
    )
    ard_publisher_domain: str = Field(
        default="",
        description=(
            "FQDN used as the ARD URN publisher (urn:air:<domain>:...). "
            "Empty means derive from the host of registry_url; "
            "falls back to 'example.com' if no resolvable host is available."
        ),
    )
    ard_catalog_default_namespace: str = Field(
        default="",
        description=(
            "Optional override for the URN namespace segment. "
            "Empty uses the entity type (server/agent/skill)."
        ),
    )

    # ARD Phase 3 ingestion (issue #1296): all behavior config lives in the DB
    # (FederationConfig.ai_catalog) and is managed via the federation API / External
    # Registries UI, mirroring the Anthropic/ASOR/AWS upstream registries. No
    # per-knob env vars here by design.

    # MCP OAuth discovery settings (RFC 9728 / RFC 8414)
    mcp_https_required: bool = Field(
        default=True,
        description=(
            "Require HTTPS for the canonical MCP resource URL advertised in PRM. "
            "Set to false only in local/dev environments."
        ),
    )
    mcp_resource_documentation_url: str | None = Field(
        default=None,
        description=(
            "Override URL for the `resource_documentation` field in the PRM document. "
            "Defaults to <registry_url>/docs/oauth when unset."
        ),
    )
    mcp_advertised_scopes: str = Field(
        default="",
        description=(
            "Space-separated override for the `scopes_supported` array in the PRM "
            "document. When set, only these scopes are advertised to discovery "
            "clients. Useful when the IdP performs RFC 7591 DCR and rejects "
            "registration requests containing scope names it doesn't recognize. "
            "Example: `openid email profile offline_access`. "
            "When unset, all scopes from the registry's authorization config are "
            "advertised (default)."
        ),
    )

    # OpenTelemetry / OTLP settings (metrics-service)
    otel_otlp_endpoint: str | None = None  # OTLP HTTP endpoint (e.g. https://otlp.example.com)
    otel_otlp_export_interval_ms: int = 30000  # OTLP export interval in milliseconds
    otel_exporter_otlp_metrics_temporality_preference: str = "cumulative"  # cumulative or delta

    # OTel-native metric emission migration (issue #1122)
    metrics_legacy_http_post: bool = Field(
        default=False,
        description=(
            "When True, ALSO emit metrics via the legacy HTTP POST path to "
            "metrics-service:8890 in addition to the native OTel path. "
            "For one-release Compose migration only; removed in 1.26.0."
        ),
    )
    otel_metric_export_interval_ms: int = Field(
        default=15000,
        ge=1000,
        description=(
            "OTel SDK metric export push interval in milliseconds. "
            "Default 15s gives near-real-time dashboards during incident "
            "response. Raise to 30000+ for high-traffic production."
        ),
    )

    # Security scanning settings (MCP Servers)
    security_scan_enabled: bool = True
    security_scan_on_registration: bool = True
    security_block_unsafe_servers: bool = True
    security_analyzers: str = "yara"  # Comma-separated: yara, llm, or yara,llm
    security_scan_timeout: int = 60  # 1 minute
    security_add_pending_tag: bool = True
    mcp_scanner_llm_api_key: str = ""  # Optional LLM API key for advanced analysis

    # Agent security scanning settings (A2A Agents)
    agent_security_scan_enabled: bool = True
    agent_security_scan_on_registration: bool = True
    agent_security_block_unsafe_agents: bool = True
    agent_security_analyzers: str = (
        "yara,spec"  # Comma-separated: yara, spec, heuristic, llm, endpoint
    )
    agent_security_scan_timeout: int = 60  # 1 minute
    agent_security_add_pending_tag: bool = True
    a2a_scanner_llm_api_key: str = ""  # Optional Azure OpenAI API key for LLM-based analysis

    # Skill security scanning settings (AI Agent Skills)
    skill_security_scan_enabled: bool = True
    skill_security_scan_on_registration: bool = True
    skill_security_block_unsafe_skills: bool = True
    skill_security_analyzers: str = (
        "static"  # Comma-separated: static, behavioral, llm, meta, virustotal, ai-defense
    )
    skill_security_scan_timeout: int = 120  # 2 minutes
    skill_security_add_pending_tag: bool = True
    skill_scanner_llm_api_key: str = ""  # Optional LLM API key for LLM-based analysis
    skill_scanner_virustotal_api_key: str = ""  # Optional VirusTotal API key
    skill_scanner_ai_defense_api_key: str = ""  # Optional Cisco AI Defense API key

    # GitHub Private Repository Access (SKILL.md fetching)
    github_pat: str = Field(
        default="",
        description="GitHub Personal Access Token for private repo SKILL.md access",
    )
    github_app_id: str = Field(
        default="",
        description="GitHub App ID for installation-based auth",
    )
    github_app_installation_id: str = Field(
        default="",
        description="GitHub App Installation ID",
    )
    github_app_private_key: str = Field(
        default="",
        description="GitHub App private key (PEM format, newlines as \\n)",
    )
    github_extra_hosts: str = Field(
        default="",
        description=(
            "Comma-separated extra GitHub hosts (e.g. github.mycompany.com,raw.github.mycompany.com). "
            "Hosts here receive GitHub auth headers AND bypass the SKILL.md SSRF private-IP check, "
            "so GHES instances on internal networks remain reachable. Keep the list tight."
        ),
    )
    github_api_base_url: str = Field(
        default="https://api.github.com",
        description="GitHub API base URL for App token exchange (for GHES: https://github.mycompany.com/api/v3)",
    )

    # Update check (GitHub Releases API for newer registry versions)
    update_check_enabled: bool = Field(
        default=True,
        description=(
            "Enable background polling of the GitHub Releases API to surface "
            "newer registry versions to admins. Set false for air-gapped "
            "deployments or to silence the admin banner."
        ),
    )
    update_check_interval_hours: int = Field(
        default=24,
        ge=1,
        description="Polling interval in hours for the update-check background task.",
    )

    # Federation settings
    registry_id: str | None = None  # Unique identifier for this registry instance in federation
    federation_static_token_auth_enabled: bool = False  # Enable federation static token auth
    federation_static_token: str = ""  # Federation static token for peer registry access
    workday_token_url: str = Field(
        default="https://your-tenant.workday.com/ccx/oauth2/your_instance/token",
        description="Workday OAuth token endpoint URL for ASOR federation (must use HTTPS in production)",
    )

    # Registry Card configuration
    registry_url: str = Field(
        default="http://localhost:8000",
        description="Base URL of this registry instance (HTTPS required in production)",
    )
    registry_organization_name: str = Field(
        default="ACME Inc.",
        description="Organization that operates this registry",
    )
    registry_name: str = Field(
        default="AI Registry",
        description="Human-readable display name for this registry instance",
    )
    registry_description: str | None = Field(
        default=None,
        description="Description of this registry instance",
    )
    registry_contact_email: str | None = Field(
        default=None,
        description="Contact email for registry operators",
    )
    registry_contact_url: str | None = Field(
        default=None,
        description="Documentation or support URL",
    )

    # Keycloak Configuration
    keycloak_enabled: bool = Field(
        default=False,
        description="Enable Keycloak as the identity provider",
    )
    keycloak_url: str = Field(
        default="http://keycloak:8080",
        description="Internal Keycloak URL",
    )
    keycloak_external_url: str = Field(
        default="http://localhost:8080",
        description="External Keycloak URL for browser redirects",
    )
    keycloak_realm: str = Field(
        default="mcp-gateway",
        description="Keycloak realm name",
    )
    keycloak_client_id: str = Field(
        default="mcp-gateway-web",
        description="Keycloak OAuth2 client ID",
    )
    keycloak_client_secret: str = Field(
        default="",
        description="Keycloak OAuth2 client secret",
    )
    keycloak_admin: str = Field(
        default="admin",
        description="Keycloak admin username",
    )
    keycloak_admin_password: str = Field(
        default="",
        description="Keycloak admin password",
    )
    keycloak_m2m_client_id: str = Field(
        default="",
        description="Keycloak M2M (machine-to-machine) client ID",
    )
    keycloak_m2m_client_secret: str = Field(
        default="",
        description="Keycloak M2M (machine-to-machine) client secret",
    )

    # Okta Configuration
    okta_enabled: bool = Field(
        default=False,
        description="Enable Okta as the identity provider",
    )
    okta_domain: str = Field(
        default="",
        description="Okta organization domain (e.g., dev-123456.okta.com)",
    )
    okta_client_id: str = Field(
        default="",
        description="Okta OAuth2 client ID",
    )
    okta_client_secret: str = Field(
        default="",
        description="Okta OAuth2 client secret",
    )
    okta_m2m_client_id: str = Field(
        default="",
        description="Okta M2M (machine-to-machine) client ID",
    )
    okta_m2m_client_secret: str = Field(
        default="",
        description="Okta M2M (machine-to-machine) client secret",
    )
    okta_api_token: str = Field(
        default="",
        description="Okta API token for admin operations",
    )
    okta_auth_server_id: str = Field(
        default="",
        description="Okta authorization server ID",
    )

    # Entra ID Configuration
    entra_enabled: bool = Field(
        default=False,
        description="Enable Microsoft Entra ID as the identity provider",
    )
    entra_tenant_id: str = Field(
        default="",
        description="Microsoft Entra ID tenant ID",
    )
    entra_client_id: str = Field(
        default="",
        description="Microsoft Entra ID client ID",
    )
    entra_client_secret: str = Field(
        default="",
        description="Microsoft Entra ID client secret",
    )
    entra_group_admin_id: str = Field(
        default="",
        description="Microsoft Entra ID admin group ID",
    )

    # IdP Group Filtering (applies to all identity providers)
    idp_group_filter_prefix: str = Field(
        default="",
        description="Comma-separated prefixes to filter IdP groups in IAM > Groups page",
    )

    # Login-time group allowlist (applies to all identity providers). When set,
    # only these exact group names/IDs are stored in a user's session at login.
    # When empty, the registry auto-derives the allowlist from scope mappings.
    allowed_idp_groups: str = Field(
        default="",
        description=(
            "Comma-separated exact IdP group names/IDs to keep in the session at "
            "login. Empty means auto-derive from scope mappings (recommended)."
        ),
    )

    # M2M direct registration (issue #851)
    m2m_direct_registration_enabled: bool = Field(
        default=True,
        description=(
            "Enable direct M2M client registration API at /api/iam/m2m-clients. "
            "This feature lets admins register M2M client_ids and group mappings "
            "without an IdP Admin API token."
        ),
    )

    # User-to-group fallback for IdPs that don't carry groups in JWTs (issue #1127).
    # Auth server consults the idp_user_groups collection only when the JWT's
    # groups claim is empty AND the token's provider name appears in this list.
    # Annotated with NoDecode so pydantic-settings does NOT try to JSON-parse
    # the env var; the field_validator below handles the CSV split itself.
    idp_user_group_fallback_enabled_providers: Annotated[list[str], NoDecode] = Field(
        default_factory=lambda: ["pingfederate"],
        description=(
            "Comma-separated list of IdP providers for which the auth server should "
            "consult the idp_user_groups collection when the JWT's groups claim is "
            "empty. Read from IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS. "
            "Default: 'pingfederate'."
        ),
    )

    @field_validator("idp_user_group_fallback_enabled_providers", mode="before")
    @classmethod
    def _parse_idp_user_group_fallback_enabled_providers(
        cls,
        v: object,
    ) -> list[str]:
        """Accept either a CSV string (from env var) or a list (from defaults).

        Empty/whitespace entries are dropped, and values are lowercased to
        match the case of provider names emitted by the auth layer
        (e.g. 'pingfederate', 'okta').
        """
        if v is None or v == "":
            return []
        if isinstance(v, list):
            return [str(item).strip().lower() for item in v if str(item).strip()]
        if isinstance(v, str):
            return [item.strip().lower() for item in v.split(",") if item.strip()]
        raise ValueError(
            "IDP_USER_GROUP_FALLBACK_ENABLED_PROVIDERS must be a CSV string or list, "
            f"got {type(v).__name__}"
        )

    # ANS Integration
    ans_integration_enabled: bool = Field(
        default=False,
        description="Enable ANS (Agent Name Service) integration",
    )
    ans_api_endpoint: str = Field(
        default="https://api.godaddy.com",
        description="ANS API base URL",
    )
    ans_api_key: str = Field(
        default="",
        description="GoDaddy API key for ANS",
    )
    ans_api_secret: str = Field(
        default="",
        description="GoDaddy API secret for ANS",
    )
    ans_api_timeout_seconds: int = Field(
        default=30,
        description="ANS API request timeout in seconds",
    )
    ans_sync_interval_hours: int = Field(
        default=6,
        description="ANS background sync interval in hours",
    )
    ans_verification_cache_ttl_seconds: int = Field(
        default=3600,
        description="ANS verification cache TTL in seconds",
    )

    # Application Log Configuration (Issue #886)
    app_log_max_bytes: int = Field(
        default=50 * 1024 * 1024,
        description="Max size per log file in bytes before rotation (default 50 MB)",
    )
    app_log_backup_count: int = Field(
        default=5,
        description="Number of rotated backup log files to keep",
    )
    app_log_centralized_enabled: bool = Field(
        default=True,
        description="Write application logs to centralized application_logs collection",
    )
    app_log_centralized_ttl_days: int = Field(
        default=1,
        description="Days to retain application log entries in centralized store (TTL index)",
    )
    app_log_mongodb_buffer_size: int = Field(
        default=50,
        description="Number of log records to buffer before flushing to MongoDB",
    )
    app_log_mongodb_flush_interval_seconds: float = Field(
        default=5.0,
        description="Seconds between periodic flushes of buffered log records to MongoDB",
    )
    app_log_level: str = Field(
        default="INFO",
        description="Application log level (DEBUG, INFO, WARNING, ERROR, CRITICAL)",
    )
    app_log_excluded_loggers: str = Field(
        default="uvicorn.access,httpx,pymongo,motor",
        description="Comma-separated logger names to exclude from MongoDB log writes",
    )
    app_log_dir: str | None = Field(
        default=None,
        description=(
            "Directory where service log files are written. "
            "When unset, defaults to /var/log/containers/ai-registry in "
            "containers and ./logs in local dev mode. "
            "Each service writes {app_log_dir}/{service_name}.log."
        ),
    )
    app_log_file_format: str = Field(
        default="json",
        description=(
            "Format for service log files. 'json' emits JSON Lines per "
            "docs/logging-standard.md (Splunk-friendly). 'text' emits the "
            "legacy comma-separated format. Console/stdout format is not "
            "affected by this setting."
        ),
    )
    app_log_console_format: str = Field(
        default="json",
        description=(
            "Format for STDOUT/console output. 'json' (default) emits the "
            "same JSON Lines schema as APP_LOG_FILE_FORMAT=json so a "
            "sidecar/agent scraping STDOUT receives structured records. "
            "'text' emits the human-readable comma-separated format if you "
            "want `docker logs` / `kubectl logs` to stay skimmable."
        ),
    )

    # Audit Logging Configuration
    audit_log_enabled: bool = True  # Enable/disable audit logging globally
    audit_log_dir: str = "logs/audit"  # Directory for local audit log files
    audit_log_rotation_hours: int = 1  # Hours between time-based file rotations
    audit_log_rotation_max_mb: int = 100  # Maximum file size in MB before rotation
    audit_log_local_retention_hours: int = (
        1  # Hours to retain local files (default 1 hour, configurable)
    )
    audit_log_health_checks: bool = False  # Whether to log health check requests
    audit_log_static_assets: bool = False  # Whether to log static asset requests

    # Audit Logging MongoDB Configuration
    audit_log_mongodb_enabled: bool = True  # Enable/disable MongoDB storage for audit logs
    audit_log_mongodb_ttl_days: int = 7  # Days to retain audit events in MongoDB (default 7 days)

    # Deployment Mode Configuration
    deployment_mode: DeploymentMode = Field(
        default=DeploymentMode.WITH_GATEWAY,
        description="Deployment mode: with-gateway or registry-only",
    )
    registry_mode: RegistryMode = Field(
        default=RegistryMode.FULL, description="Registry operating mode"
    )

    # Internal/workshop deployment classification (telemetry labels only; see issue
    # #1216). These are self-reported labels for tracing our own internal and workshop
    # deployments in telemetry. They do NOT change access control or network behavior.
    # The cross-field default/correction logic (force to "none" when not internal,
    # default to "dev" when internal and unset) runs at startup in main.py, mirroring
    # the deployment_mode/registry_mode correction pattern.
    internal_only_deployment: bool = Field(
        default=False,
        description=(
            "Marks this as one of our own internal/workshop deployments (not a "
            "community install). Telemetry label only; does not change access control."
        ),
    )
    internal_deployment_type: InternalDeploymentType = Field(
        default=InternalDeploymentType.NONE,
        description=(
            "Classification of an internal deployment: none, dev, workshop, or other. "
            "Forced to 'none' when internal_only_deployment is false; defaults to 'dev' "
            "when internal_only_deployment is true and left unset."
        ),
    )

    # Coding assistant selection for the Server Configuration modal.
    # Empty (default) means all supported assistants are shown.
    # Comma-separated list from the CODING_ASSISTANTS env var, e.g.
    # CODING_ASSISTANTS=cursor,claude-code
    coding_assistants: str = Field(
        default="",
        description=(
            "Comma-separated allowlist of coding assistants shown in the UI config modal. "
            "Empty means show all. Supported values: cursor, roo-code, claude-code, kiro."
        ),
    )

    @property
    def coding_assistants_list(self) -> list[str]:
        """Parse coding_assistants CSV into a list, stripping whitespace."""
        if not self.coding_assistants:
            return []
        return [item.strip() for item in self.coding_assistants.split(",") if item.strip()]

    # Tab visibility overrides (AND-ed with REGISTRY_MODE feature flags)
    show_servers_tab: bool = Field(default=True, description="Show MCP Servers tab in UI")
    show_virtual_servers_tab: bool = Field(
        default=True, description="Show Virtual MCP Servers tab in UI"
    )
    show_skills_tab: bool = Field(default=True, description="Show Skills tab in UI")
    show_agents_tab: bool = Field(default=True, description="Show Agents tab in UI")

    # Telemetry settings (anonymous usage tracking)
    telemetry_enabled: bool = Field(
        default=True,
        description="Enable anonymous telemetry (startup ping). Opt-out: MCP_TELEMETRY_DISABLED=1",
    )
    telemetry_opt_out: bool = Field(
        default=False,
        description="Disable daily heartbeat telemetry only. Opt-out: MCP_TELEMETRY_OPT_OUT=1",
    )
    telemetry_heartbeat_interval_minutes: int = Field(
        default=1440,
        description="Heartbeat telemetry interval in minutes (default: 1440 = 24 hours). MCP_TELEMETRY_HEARTBEAT_INTERVAL_MINUTES=1440",
    )
    telemetry_endpoint: str = Field(
        default="https://m3ijrhd020.execute-api.us-east-1.amazonaws.com/v1/collect",
        description="HTTPS endpoint for telemetry collector (must be HTTPS; supports self-hosted)",
    )
    telemetry_debug: bool = Field(
        default=False,
        description="Log telemetry payloads instead of sending (for debugging)",
    )
    telemetry_imds_probe_disabled: bool = Field(
        default=False,
        description=(
            "Disable IMDS probing in cloud detection (opt-out). "
            "When true, registry will only use env vars, DMI files, ECS "
            "metadata URI, and k8s node-name heuristics to detect the cloud "
            "provider. MCP_TELEMETRY_IMDS_PROBE_DISABLED=1"
        ),
    )
    mcp_cloud_provider: str | None = Field(
        default=None,
        description=(
            "Operator-supplied cloud provider override. Takes precedence over "
            "the auto-detection cascade and the admin-UI hint. "
            "Allowed: aws, azure, gcp, on_premises, other."
        ),
    )

    @field_validator("mcp_cloud_provider")
    @classmethod
    def _validate_cloud_provider(cls, v: str | None) -> str | None:
        if v is None or v == "":
            return None
        v_lower = v.strip().lower()
        if v_lower not in {"aws", "azure", "gcp", "on_premises", "other"}:
            display = v[:16] + ("..." if len(v) > 16 else "")
            import logging as _logging

            _logging.getLogger(__name__).warning(
                f"MCP_CLOUD_PROVIDER={display!r} is not a valid value; ignoring"
            )
            return None
        return v_lower

    # Demo server configuration
    disable_ai_registry_tools_server: bool = Field(
        default=False,
        description="Disable auto-registration of the built-in airegistry-tools server on startup. Set DISABLE_AI_REGISTRY_TOOLS_SERVER=true to opt out.",
    )

    # Tool-level access enforcement (Issue #1026)
    mcp_tools_list_filter_enabled: bool = Field(
        default=True,
        description=(
            "Enable filtering of MCP tools/list JSON-RPC responses per "
            "user tool scope. REST endpoints always filter regardless."
        ),
    )
    mcp_proxy_max_body_bytes: int = Field(
        default=2 * 1024 * 1024,
        ge=1024,
        description=(
            "Maximum buffered size for MCP tools/list upstream responses "
            "before the proxy hop returns 413. Raise only for servers "
            "with unusually large tool catalogs."
        ),
    )
    tool_filter_audit_log_level: str = Field(
        default="INFO",
        description=(
            "Launch-window override for tool-pruning audit log verbosity. "
            "Valid values: DEBUG, INFO, WARNING."
        ),
    )

    internal_token_ttl_seconds: int = Field(
        default=30,
        ge=5,
        description=(
            "Lifetime (seconds) of the /validate-minted /mcp-proxy internal "
            "token; the replay-window cap. Short by design."
        ),
    )
    internal_token_leeway_seconds: int = Field(
        default=5,
        ge=0,
        description=(
            "Clock-skew leeway (seconds) on the /mcp-proxy internal token " "exp/iat checks."
        ),
    )

    @property
    def nginx_updates_enabled(self) -> bool:
        """Check if nginx updates should be performed."""
        return self.deployment_mode == DeploymentMode.WITH_GATEWAY

    # UI Title Configuration
    ui_title: str | None = Field(
        default=None,
        description=(
            "Override for the UI title shown in the header, login, and logout pages. "
            "When unset (or empty/whitespace-only), the title defaults to "
            "'AI Gateway & Registry' for with-gateway mode and 'AI Registry' for "
            "registry-only mode."
        ),
    )

    @property
    def effective_ui_title(self) -> str:
        """Return the resolved UI title.

        Reads ``self.deployment_mode``, which has already been auto-corrected by
        ``_apply_mode_corrections`` in ``registry/main.py`` at startup, so this
        property always sees the post-correction value.
        """
        if self.ui_title and self.ui_title.strip():
            return self.ui_title
        if self.deployment_mode == DeploymentMode.REGISTRY_ONLY:
            return "AI Registry"
        return "AI Gateway & Registry"

    # Registration deduplication. Advisory checks that surface
    # likely-duplicate entities (servers, agents, skills) before a user
    # registers a new one. The /check-duplicates endpoints are always
    # available; this setting only controls whether the registration
    # form pre-flights the check and renders the modal hint.
    dedup_registration_hint_enabled: bool = Field(
        default=True,
        description=(
            "When true, the registration form pre-flights "
            "/api/<entity>/check-duplicates and renders a hint modal "
            "if matches are found. When false, the form submits "
            "straight to /register without checking. The endpoint and "
            "service themselves remain available regardless."
        ),
    )
    dedup_score_threshold: float = Field(
        default=0.7,
        ge=0.0,
        le=1.0,
        description="Minimum semantic-search score (0..1) for an advisory match to be returned.",
    )
    dedup_max_suggestions: int = Field(
        default=3,
        ge=1,
        le=10,
        description="Cap on the number of duplicate suggestions returned.",
    )

    # Storage Backend Configuration
    storage_backend: str = Field(
        default="mongodb-ce",
        description=(
            "Storage backend selection. Accepted values: "
            "documentdb, mongodb-ce, mongodb, mongodb-atlas. "
            "mongodb, mongodb-atlas, and mongodb-ce are aliases that route to "
            "the same MongoDB/DocumentDB repositories. documentdb is retained "
            "for AWS DocumentDB-specific auth (SCRAM-SHA-1). Unknown values "
            "fail startup with a clear error listing accepted values. "
            "The 'file' backend was removed in v1.24.8."
        ),
    )

    @field_validator("app_log_dir", mode="before")
    @classmethod
    def _validate_app_log_dir(
        cls,
        v: str | None,
    ) -> str | None:
        """Reject non-absolute APP_LOG_DIR and paths containing '..'.

        Operator-supplied config; we validate at startup to fail fast with a
        clear message instead of letting the container silently fall back to
        console-only logging after a mkdir error.
        """
        if v is None or v == "":
            return None
        if not isinstance(v, str):
            raise ValueError(f"APP_LOG_DIR must be a string, got {type(v).__name__}")
        if not v.startswith("/"):
            raise ValueError(f"APP_LOG_DIR must be an absolute path, got {v!r}")
        if ".." in Path(v).parts:
            raise ValueError(f"APP_LOG_DIR must not contain '..' segments, got {v!r}")
        return v

    @field_validator("app_log_file_format", mode="before")
    @classmethod
    def _validate_app_log_file_format(
        cls,
        v: str,
    ) -> str:
        """Accept only 'json' (new default) or 'text' (legacy back-compat)."""
        if v is None:
            return "json"
        if not isinstance(v, str):
            raise ValueError(f"APP_LOG_FILE_FORMAT must be a string, got {type(v).__name__}")
        normalized = v.strip().lower()
        if normalized not in ("json", "text"):
            raise ValueError(f"APP_LOG_FILE_FORMAT must be 'json' or 'text', got {v!r}")
        return normalized

    @field_validator("app_log_console_format", mode="before")
    @classmethod
    def _validate_app_log_console_format(
        cls,
        v: str,
    ) -> str:
        """Accept only 'text' (default, human-readable) or 'json' (JSONL)."""
        if v is None:
            return "text"
        if not isinstance(v, str):
            raise ValueError(f"APP_LOG_CONSOLE_FORMAT must be a string, got {type(v).__name__}")
        normalized = v.strip().lower()
        if normalized not in ("json", "text"):
            raise ValueError(f"APP_LOG_CONSOLE_FORMAT must be 'json' or 'text', got {v!r}")
        return normalized

    @field_validator("storage_backend", mode="before")
    @classmethod
    def _validate_storage_backend(
        cls,
        v: str | None,
    ) -> str:
        """Reject unknown STORAGE_BACKEND values at startup.

        Empty string and None coerce to "mongodb-ce" (the default since
        v1.24.8). Any other value is normalized (stripped, lowercased) and
        compared against ALLOWED_STORAGE_BACKENDS. The legacy "file" value
        is rejected with an actionable migration message.

        Safe to echo v in the error: storage_backend is a non-secret config
        name. Do not copy this pattern for fields that could hold credentials.
        """
        if v is None or v == "":
            return "mongodb-ce"
        if not isinstance(v, str):
            raise ValueError(f"STORAGE_BACKEND must be a string, got {type(v).__name__}")
        normalized = v.strip().lower()
        if normalized == "file":
            raise ValueError(
                "The 'file' storage backend was removed in v1.24.8. "
                "Set STORAGE_BACKEND to 'mongodb-ce' or 'documentdb'. "
                "To migrate existing data, run: "
                "python scripts/migrate-file-to-mongodb.py"
            )
        if normalized not in ALLOWED_STORAGE_BACKENDS:
            accepted = ", ".join(sorted(ALLOWED_STORAGE_BACKENDS))
            raise ValueError(f"Invalid STORAGE_BACKEND={v!r}. Accepted values: {accepted}.")
        return normalized

    @field_validator("internal_deployment_type", mode="before")
    @classmethod
    def _validate_internal_deployment_type(
        cls,
        v: str | None,
    ) -> str:
        """Reject unknown INTERNAL_DEPLOYMENT_TYPE values at startup.

        Empty string and None coerce to "none". Any other value is normalized
        (stripped, lowercased) and compared against the InternalDeploymentType
        members. Unknown values raise ValueError listing the accepted values.

        Safe to echo v in the error: this is a non-secret config name.
        """
        if v is None or v == "":
            return InternalDeploymentType.NONE.value
        if isinstance(v, InternalDeploymentType):
            return v.value
        if not isinstance(v, str):
            raise ValueError(f"INTERNAL_DEPLOYMENT_TYPE must be a string, got {type(v).__name__}")
        normalized = v.strip().lower()
        valid = {t.value for t in InternalDeploymentType}
        if normalized not in valid:
            accepted = ", ".join(sorted(valid))
            raise ValueError(
                f"Invalid INTERNAL_DEPLOYMENT_TYPE={v!r}. Accepted values: {accepted}."
            )
        return normalized

    # DocumentDB Configuration (only used when storage_backend="documentdb" or "mongodb-ce")
    documentdb_host: str = "localhost"
    documentdb_port: int = 27017
    documentdb_database: str = "mcp_registry"
    documentdb_username: str | None = None
    documentdb_password: str | None = None
    documentdb_use_tls: bool = True
    documentdb_tls_ca_file: str = "/app/certs/global-bundle.pem"
    documentdb_use_iam: bool = False
    documentdb_replica_set: str | None = None
    documentdb_read_preference: str = "secondaryPreferred"
    documentdb_direct_connection: bool = False  # Set to True only for single-node MongoDB (tests)

    # Full MongoDB connection URI override. When set, bypasses host/port/user/password
    # assembly and is passed verbatim to the MongoDB client. Required for MongoDB Atlas
    # (mongodb+srv://...) and any externally-managed MongoDB where the caller wants to
    # own the full URI (replica sets, TLS params, retryWrites, etc.).
    mongodb_connection_string: str | None = None

    # DocumentDB Namespace (for multi-tenancy support)
    documentdb_namespace: str = "default"

    # Agent batch API (issue #956)
    batch_max_operations_per_job: int = Field(
        default=1000,
        ge=1,
        description="Maximum number of items allowed in a single agent batch submission.",
    )
    batch_max_concurrent_jobs_per_user: int = Field(
        default=3,
        ge=1,
        description="Maximum number of active (queued or running) batch jobs per submitter.",
    )
    batch_job_retention_days: int = Field(
        default=7,
        ge=1,
        description="Retention window for agent batch jobs in MongoDB (TTL index on updated_at).",
    )
    batch_worker_poll_interval_seconds: float = Field(
        default=1.0,
        gt=0,
        description="How often the batch worker polls MongoDB for queued jobs.",
    )
    batch_worker_enabled: bool = Field(
        default=True,
        description=(
            "Enable the in-process agent batch worker loop. Lease-based claiming "
            "makes multi-worker operation safe: any number of replicas may run "
            "with this true and cooperatively drain the queue."
        ),
    )
    batch_worker_lease_ttl_seconds: float = Field(
        default=60.0,
        gt=0,
        description=(
            "How long a claimed batch job stays owned before its lease expires "
            "and another worker may reclaim it. Must exceed the worst-case time "
            "between lease renewals (slowest single item + heartbeat interval + "
            "clock-skew slack), or a slow-but-healthy worker risks having its job "
            "reclaimed and processed concurrently."
        ),
    )
    batch_worker_lease_heartbeat_seconds: float = Field(
        default=15.0,
        gt=0,
        description=(
            "Interval at which a worker renews the lease on its in-flight job. "
            "Should be comfortably below batch_worker_lease_ttl_seconds (a common "
            "shape is TTL = 3-4x the heartbeat) so a renewal is never missed by a "
            "healthy worker."
        ),
    )
    batch_max_request_bytes: int = Field(
        default=4 * 1024 * 1024,
        ge=1024,
        description="Maximum request body size (bytes) accepted by POST /api/agents/batch.",
    )

    # Container paths - adjust for local development
    container_app_dir: Path = Path("/app")
    container_registry_dir: Path = Path("/app/registry")
    container_log_dir: Path = Path("/app/logs")

    # Local development mode detection
    @property
    def is_local_dev(self) -> bool:
        """Check if running in local development mode."""
        return not Path("/app").exists()

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        if not self.secret_key:
            raise RuntimeError(
                "SECRET_KEY environment variable is required. "
                "Set it to a value at least 32 bytes long, identical across all auth_server "
                "and registry replicas (see chart values.yaml: global.secretKey)."
            )

    @property
    def embeddings_model_dir(self) -> Path:
        if self.is_local_dev:
            return Path.cwd() / "registry" / "models" / self.embeddings_model_name
        return self.container_registry_dir / "models" / self.embeddings_model_name

    @property
    def servers_dir(self) -> Path:
        if self.is_local_dev:
            return Path.cwd() / "registry" / "servers"
        return self.container_registry_dir / "servers"

    @property
    def static_dir(self) -> Path:
        if self.is_local_dev:
            return Path.cwd() / "registry" / "static"
        return self.container_registry_dir / "static"

    @property
    def templates_dir(self) -> Path:
        if self.is_local_dev:
            return Path.cwd() / "registry" / "templates"
        return self.container_registry_dir / "templates"

    @property
    def nginx_config_path(self) -> Path:
        return Path("/etc/nginx/conf.d/nginx_rev_proxy.conf")

    @property
    def state_file_path(self) -> Path:
        return self.servers_dir / "server_state.json"

    @property
    def log_dir(self) -> Path:
        """Resolve the directory where application .log files are written.

        Resolution order:
        1. APP_LOG_DIR override (if set) - used verbatim.
        2. ./logs in local dev (when /app does not exist).
        3. /var/log/containers/ai-registry in containers (issue #987).

        Note: this is for the Python application logs (registry.log,
        auth-server.log, ai-registry-tools.log). Audit logs still use
        container_log_dir via the audit_log_path property and are not
        affected by this setting.
        """
        if self.app_log_dir:
            return Path(self.app_log_dir)
        if self.is_local_dev:
            return Path.cwd() / "logs"
        return Path("/var/log/containers/ai-registry")

    @property
    def log_file_path(self) -> Path:
        """Resolve the full path to this service's .log file.

        Kept for backwards compatibility with callers that import log_file_path
        directly; most code should call setup_logging(service_name=...) instead,
        which computes the same path via log_dir.
        """
        return self.log_dir / "registry.log"

    @property
    def dotenv_path(self) -> Path:
        if self.is_local_dev:
            return Path.cwd() / ".env"
        return self.container_registry_dir / ".env"

    @property
    def agents_dir(self) -> Path:
        """Directory for agent card storage."""
        if self.is_local_dev:
            return Path.cwd() / "registry" / "agents"
        return self.container_registry_dir / "agents"

    @property
    def agent_state_file_path(self) -> Path:
        """Path to agent state file (enabled/disabled tracking)."""
        return self.agents_dir / "agent_state.json"

    @property
    def peers_dir(self) -> Path:
        """Directory for peer federation config storage."""
        home_dir = Path.home()
        return home_dir / "mcp-gateway" / "peers"

    @property
    def peer_sync_state_file_path(self) -> Path:
        """Path to peer sync state file."""
        home_dir = Path.home()
        return home_dir / "mcp-gateway" / "peer_sync_state.json"

    @property
    def audit_log_path(self) -> Path:
        """Get audit log directory based on environment."""
        if self.is_local_dev:
            return Path.cwd() / self.audit_log_dir
        return self.container_log_dir / "audit"

    @property
    def data_dir(self) -> Path:
        """Get data directory for persistent storage (telemetry ID, etc.)."""
        if self.is_local_dev:
            return Path.cwd() / "registry" / "data"
        return self.container_registry_dir / "data"


class EmbeddingConfig:
    """Helper class for embedding configuration and metadata generation."""

    def __init__(self, settings_instance: Settings):
        self.settings = settings_instance

    @property
    def model_family(self) -> str:
        """Extract model family from model name.

        Examples:
            - "openai/text-embedding-ada-002" -> "openai"
            - "all-MiniLM-L6-v2" -> "sentence-transformers"
            - "amazon.titan-embed-text-v2:0" -> "amazon-bedrock"
        """
        model_name = self.settings.embeddings_model_name

        if "/" in model_name:
            # Format: "provider/model-name"
            return model_name.split("/")[0]
        elif "amazon." in model_name or "titan" in model_name.lower():
            return "amazon-bedrock"
        elif self.settings.embeddings_provider == "litellm":
            return "litellm"
        else:
            return self.settings.embeddings_provider

    @property
    def index_name(self) -> str:
        """Generate dimension-specific collection/index name.

        Returns index name in format: mcp-embeddings-{dimensions}-{namespace}
        Example: mcp-embeddings-1536-default
        """
        base_name = "mcp-embeddings"
        dimensions = self.settings.embeddings_model_dimensions
        namespace = self.settings.documentdb_namespace

        # Replace base name with dimension-specific name
        return f"{base_name}-{dimensions}-{namespace}"

    def get_embedding_metadata(self) -> dict:
        """Generate embedding metadata for document storage.

        Returns:
            Dictionary with embedding metadata including:
            - provider: Embedding provider (e.g., "litellm", "sentence-transformers")
            - model: Full model name
            - model_family: Extracted model family
            - dimensions: Embedding dimension count
            - version: Model version (extracted if available, else "v1")
            - created_at: Current timestamp in ISO format
            - indexing_strategy: Search strategy (currently "hybrid")
        """
        from datetime import datetime

        model_name = self.settings.embeddings_model_name

        # Extract version if present in model name
        version = "v1"
        if "v2" in model_name.lower():
            version = "v2"
        elif "v3" in model_name.lower():
            version = "v3"
        elif "ada-002" in model_name:
            version = "ada-002"

        return {
            "provider": self.settings.embeddings_provider,
            "model": model_name,
            "model_family": self.model_family,
            "dimensions": self.settings.embeddings_model_dimensions,
            "version": version,
            "created_at": datetime.now(UTC).isoformat(),
            "indexing_strategy": "hybrid",
        }


logger = logging.getLogger(__name__)


def _validate_mode_combination(
    deployment_mode: DeploymentMode, registry_mode: RegistryMode
) -> tuple[DeploymentMode, RegistryMode, bool]:
    """
    Validate and potentially correct deployment/registry mode combination.

    Args:
        deployment_mode: Current deployment mode setting
        registry_mode: Current registry mode setting

    Returns:
        Tuple of (corrected_deployment_mode, corrected_registry_mode, was_corrected)
    """
    # Invalid: with-gateway + skills-only
    # Skills don't need gateway, auto-convert to registry-only
    if deployment_mode == DeploymentMode.WITH_GATEWAY and registry_mode == RegistryMode.SKILLS_ONLY:
        return (DeploymentMode.REGISTRY_ONLY, RegistryMode.SKILLS_ONLY, True)

    return (deployment_mode, registry_mode, False)


def _print_config_warning_banner(
    original_deployment: DeploymentMode,
    original_registry: RegistryMode,
    corrected_deployment: DeploymentMode,
    corrected_registry: RegistryMode,
) -> None:
    """Print conspicuous warning banner for invalid configuration."""
    banner = f"""
================================================================================
WARNING: Invalid configuration detected!

DEPLOYMENT_MODE={original_deployment.value} is incompatible with REGISTRY_MODE={original_registry.value}
Skills do not require gateway integration.

Auto-converting to:
  DEPLOYMENT_MODE={corrected_deployment.value}
  REGISTRY_MODE={corrected_registry.value}
================================================================================
"""
    logger.warning(banner)
    print(banner)


def log_tab_visibility_warnings(s: Settings) -> None:
    """Log warnings for SHOW_*_TAB parameters that are ineffective given REGISTRY_MODE."""
    mode = s.registry_mode
    checks = [
        (
            s.show_servers_tab,
            "SHOW_SERVERS_TAB",
            mode in (RegistryMode.FULL, RegistryMode.MCP_SERVERS_ONLY),
        ),
        (
            s.show_agents_tab,
            "SHOW_AGENTS_TAB",
            mode in (RegistryMode.FULL, RegistryMode.AGENTS_ONLY),
        ),
        (
            s.show_skills_tab,
            "SHOW_SKILLS_TAB",
            mode in (RegistryMode.FULL, RegistryMode.SKILLS_ONLY),
        ),
        (
            s.show_virtual_servers_tab,
            "SHOW_VIRTUAL_SERVERS_TAB",
            mode in (RegistryMode.FULL, RegistryMode.MCP_SERVERS_ONLY),
        ),
    ]
    for show_tab, param_name, mode_enables in checks:
        if show_tab and not mode_enables:
            logger.warning(
                "%s is true but REGISTRY_MODE=%s does not enable this feature; "
                "the tab will remain hidden.",
                param_name,
                mode.value,
            )


# Global settings instance
settings = Settings()

# Global embedding config instance
embedding_config = EmbeddingConfig(settings)
