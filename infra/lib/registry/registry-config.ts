/**
 * RegistryConfig - TypeScript interface mapping all Terraform variables
 * for the MCP Gateway Registry infrastructure.
 *
 * Sensitive values are loaded from environment variables (CDK_* prefix).
 * All other values can be specified in config.yaml.
 */

import * as fs from 'fs';
import * as yaml from 'js-yaml';

// ---------------------------------------------------------------------------
// Nested configuration interfaces
// ---------------------------------------------------------------------------

export interface DocumentDbConfig {
  /** DocumentDB Elastic Cluster admin username */
  adminUsername: string;
  /** DocumentDB Elastic Cluster admin password (from env CDK_DOCUMENTDB_ADMIN_PASSWORD) */
  adminPassword: string;
  /** vCPU capacity per shard (2, 4, 8, 16, 32, or 64) */
  shardCapacity: number;
  /** Number of shards (1-32) */
  shardCount: number;
  /** Instance class for DocumentDB cluster instances */
  instanceClass: string;
  /** Number of read replica instances (0-15) */
  replicaCount: number;
  /** DocumentDB database name */
  database: string;
  /** DocumentDB namespace for collections */
  namespace: string;
  /** Use TLS for DocumentDB connections */
  useTls: boolean;
  /** Use IAM authentication for DocumentDB */
  useIam: boolean;
}

export interface KeycloakConfig {
  /** Keycloak admin username */
  adminUser: string;
  /** Keycloak admin password (from env CDK_KEYCLOAK_ADMIN_PASSWORD) */
  adminPassword: string;
  /** Keycloak database username */
  databaseUsername: string;
  /** Keycloak database password (from env CDK_KEYCLOAK_DATABASE_PASSWORD) */
  databasePassword: string;
  /** Minimum Aurora Capacity Units */
  databaseMinAcu: number;
  /** Maximum Aurora Capacity Units */
  databaseMaxAcu: number;
  /** Keycloak log level */
  logLevel: string;
  /** Full domain for Keycloak when use_regional_domains is false */
  domain: string;
  /** Root domain when use_regional_domains is false */
  rootDomain: string;
}

export interface ImagesConfig {
  /** Container image URI for registry service */
  registry: string;
  /** Container image URI for auth server service */
  authServer: string;
  /** Container image URI for Keycloak (used by Auth stack) */
  keycloak: string;
  /** Container image URI for currenttime MCP server */
  currenttime: string;
  /** Container image URI for mcpgw MCP server */
  mcpgw: string;
  /** Container image URI for realserverfaketools MCP server */
  realserverfaketools: string;
  /** Container image URI for flight booking A2A agent */
  flightBookingAgent: string;
  /** Container image URI for travel assistant A2A agent */
  travelAssistantAgent: string;
  /** Container image URI for metrics-service */
  metricsService: string;
  /** Container image URI for Grafana OSS */
  grafana: string;
}

export interface ReplicasConfig {
  /** Number of replicas for registry main service */
  registry: number;
  /** Number of replicas for auth service */
  auth: number;
  /** Number of replicas for CurrentTime MCP server */
  currenttime: number;
  /** Number of replicas for MCPGW MCP server */
  mcpgw: number;
  /** Number of replicas for RealServerFakeTools MCP server */
  realserverfaketools: number;
  /** Number of replicas for Flight Booking A2A agent */
  flightBookingAgent: number;
  /** Number of replicas for Travel Assistant A2A agent */
  travelAssistantAgent: number;
}

export interface EmbeddingsConfig {
  /** Embeddings provider: 'sentence-transformers' or 'litellm' */
  provider: string;
  /** Name of the embeddings model */
  modelName: string;
  /** Dimension of the embeddings model */
  modelDimensions: number;
  /** AWS region for Bedrock embeddings */
  awsRegion: string;
  /** API key for embeddings provider (from env CDK_EMBEDDINGS_API_KEY) */
  apiKey: string;
}

export interface SecurityConfig {
  /** Enable security scanning for MCP servers */
  scanEnabled: boolean;
  /** Automatically scan servers when they are registered */
  scanOnRegistration: boolean;
  /** Block servers that fail security scans */
  blockUnsafeServers: boolean;
  /** Analyzers to use for security scanning (comma-separated) */
  analyzers: string;
  /** Security scan timeout in seconds */
  scanTimeout: number;
  /** Add 'security-pending' tag to servers that fail security scan */
  addPendingTag: boolean;
}

export interface HealthChecksConfig {
  /** Seconds between health checks of every enabled server; also one tool refresh cycle */
  intervalSeconds: number;
  /** Fetch tools again from servers that stay healthy, on a schedule (issue #2050) */
  toolRefreshEnabled: boolean;
  /** Fetch each healthy server once every this many health cycles (1 to 1000) */
  toolRefreshEveryCycles: number;
  /** Most background tool fetches one registry task runs at once (1 to 100) */
  toolFetchMaxConcurrency: number;
}

export interface FederationConfig {
  /** Unique identifier for this registry instance in federation */
  registryId: string;
  /** Enable static token auth for Federation API endpoints */
  staticTokenAuthEnabled: boolean;
  /** Static token for Federation API access (from env CDK_FEDERATION_STATIC_TOKEN) */
  staticToken: string;
  /** Fernet encryption key for storing federation tokens (from env CDK_FEDERATION_ENCRYPTION_KEY) */
  encryptionKey: string;
  /** Enable AWS Agent Registry federation */
  awsRegistryFederationEnabled: boolean;
  /**
   * IAM role ARNs the registry task may assume for cross-account AWS Agent
   * Registry federation. Empty (default) disables cross-account access: the
   * sts:AssumeRole grant is omitted entirely and only same-account federation
   * works. Fail-closed: an unset list grants no cross-account trust.
   */
  awsRegistryFederationAssumeRoleArns: string[];
}

export interface AuditConfig {
  /** Enable audit logging for all API and MCP requests */
  logEnabled: boolean;
  /** Audit log retention period in days */
  logTtlDays: number;
}

export interface OtelConfig {
  /** OTLP endpoint for pushing metrics to an external platform */
  otlpEndpoint: string;
  /** Headers for OTLP exporter (from env CDK_OTEL_EXPORTER_OTLP_HEADERS) */
  exporterOtlpHeaders: string;
  /** OTLP export interval in milliseconds */
  otlpExportIntervalMs: number;
  /** OTLP metrics temporality preference */
  exporterOtlpMetricsTemporalityPreference: string;
  /**
   * Image for the ADOT collector sidecars. Pin this to a released tag for any
   * deployment you need to be able to roll back: the default tracks `latest`,
   * matching the Terraform stack, so the collector version can change under a
   * redeploy that changed nothing else.
   */
  adotCollectorImage: string;
}

export interface EntraConfig {
  /** Enable Microsoft Entra ID as authentication provider */
  enabled: boolean;
  /** Azure AD Tenant ID */
  tenantId: string;
  /** Entra ID Application (client) ID */
  clientId: string;
  /** Entra ID Client Secret (from env CDK_ENTRA_CLIENT_SECRET) */
  clientSecret: string;
  /**
   * PRM scope-advertisement form: 'v1' (api://<app-id>/<scope>) or 'v2'
   * (bare, default). Set 'v1' only if the Entra app exposes v1-style api://
   * scopes (issue #990). Empty string uses the provider default (v2).
   */
  scopeFormat: string;
  /**
   * Application ID URI (e.g. api://<app-id>) registered on the Entra app.
   * Used as the v1 scope prefix and accepted as a token audience. Empty
   * defaults to api://<clientId>.
   */
  applicationIdUri: string;
}

export interface OktaConfig {
  /** Enable Okta as authentication provider */
  enabled: boolean;
  /** Okta domain */
  domain: string;
  /** Okta Web Application (client) ID */
  clientId: string;
  /** Okta Client Secret (from env CDK_OKTA_CLIENT_SECRET) */
  clientSecret: string;
  /** Okta M2M Client ID */
  m2mClientId: string;
  /** Okta M2M Client Secret (from env CDK_OKTA_M2M_CLIENT_SECRET) */
  m2mClientSecret: string;
  /** Okta API Token (from env CDK_OKTA_API_TOKEN) */
  apiToken: string;
  /** Okta Custom Authorization Server ID */
  authServerId: string;
}

export interface Auth0Config {
  /** Enable Auth0 as authentication provider */
  enabled: boolean;
  /** Auth0 domain */
  domain: string;
  /** Auth0 Web Application (client) ID */
  clientId: string;
  /** Auth0 Client Secret (from env CDK_AUTH0_CLIENT_SECRET) */
  clientSecret: string;
  /** Auth0 API Audience */
  audience: string;
  /** Auth0 custom claim for group memberships */
  groupsClaim: string;
  /** Auth0 M2M Client ID */
  m2mClientId: string;
  /** Auth0 M2M Client Secret (from env CDK_AUTH0_M2M_CLIENT_SECRET) */
  m2mClientSecret: string;
  /** Auth0 Management API Token (from env CDK_AUTH0_MANAGEMENT_API_TOKEN) */
  managementApiToken: string;
}

export interface CognitoConfig {
  /** Enable Amazon Cognito as authentication provider */
  enabled: boolean;
  /** Cognito user pool id, for example us-west-2_AbC123 */
  userPoolId: string;
  /** Cognito app client id for the web (authorization-code) flow */
  clientId: string;
  /**
   * ARN of an existing Secrets Manager secret holding the app client secret.
   *
   * An ARN rather than the value: a literal is written into the synthesized
   * template, so it would sit in plaintext in `cdk.out` and in the deployed
   * CloudFormation stack. Create the secret out of band, then point here.
   * Must be a complete ARN, including the six-character suffix.
   */
  clientSecretArn: string;
  /**
   * Cognito hosted-UI domain PREFIX, not a hostname. The auth-server appends
   * `.auth.<region>.amazoncognito.com`, so `my-pool` yields
   * `my-pool.auth.us-west-2.amazoncognito.com`. Empty derives the prefix from
   * the user pool id (auth_server/providers/cognito.py,
   * resolve_cognito_domain_prefix), which works only if the pool's domain uses
   * that prefix.
   */
  domain: string;
  /**
   * Comma-separated allowlist of M2M app client ids whose access tokens the
   * auth-server accepts. Read by the auth-server only. Empty denies every
   * client-credentials token, which is the fail-closed default.
   */
  m2mClientIds: string;
}

export interface PingFederateConfig {
  /** Enable PingFederate as authentication provider */
  enabled: boolean;
  /** Internal PingFederate base URL used for server-to-server calls */
  baseUrl: string;
  /** Browser-facing PingFederate URL used in redirects */
  externalUrl: string;
  /** OIDC client id for the web (authorization-code) flow */
  clientId: string;
  /**
   * ARN of an existing Secrets Manager secret holding the OIDC client secret.
   * An ARN rather than the value, for the reason given on
   * {@link CognitoConfig.clientSecretArn}.
   */
  clientSecretArn: string;
  /** M2M (client-credentials) client id */
  m2mClientId: string;
  /**
   * ARN of an existing secret holding a SEPARATE M2M client secret. Optional:
   * the provider falls back to the web client secret when this is unset
   * (auth_server/providers/pingfederate.py), so leaving it empty injects
   * nothing rather than injecting an empty value.
   */
  m2mClientSecretArn: string;
  /** Application ID URI accepted as a token audience */
  applicationIdUri: string;
  /** Claim carrying group membership */
  groupsClaim: string;
  /**
   * EXTRA audiences accepted on an M2M (client-credentials) token, comma or
   * whitespace separated.
   *
   * Optional, and empty is already fail-closed: the provider always validates
   * `aud` against the union of `clientId`, `m2mClientId`,
   * `applicationIdUri` and this list, so an empty list accepts only the
   * configured client ids and application id URI
   * (auth_server/providers/pingfederate.py). Add to it when an M2M token's
   * audience is some other resource identifier.
   */
  m2mAllowedAudiences: string;
  /**
   * PingFederate admin API URL. Registry only: it creates OAuth clients and
   * PCV users through this API.
   */
  adminUrl: string;
  /** PingFederate admin API username */
  adminUser: string;
  /**
   * ARN of an existing secret holding the admin API password. Registry only.
   *
   * Required when PingFederate is enabled: registry/utils/pingfederate_manager.py
   * raises without it, so user administration and service-account creation
   * fail even though login works. It also rejects the vendor default
   * `2FederateM0re`.
   */
  adminPasswordArn: string;
}

export interface EgressConfig {
  /** Enable egress auth (per-user credential vault, OAuth consent, OBO) */
  enabled: boolean;
  /**
   * ARN of an existing secret holding the AES-256-GCM input keying material
   * the registry uses to encrypt vaulted egress credentials at rest.
   *
   * An ARN rather than the value, for the reason given on
   * {@link CognitoConfig.clientSecretArn}. Never generated either: rotating
   * this key needs a coordinated re-encryption of the whole vault, so a
   * CDK-generated value a later synth replaced would strand it. Empty leaves
   * credentials in plaintext, which is the application's legacy behavior.
   */
  credentialEncryptionKeyArn: string;
  /**
   * Credential store backend. Only 'secrets-manager' is supported here.
   *
   * The application accepts 'secrets-manager' or 'openbao'
   * (registry/core/config.py, `secret_store_backend`), and this stack
   * provisions no OpenBao, so there is nothing else to offer.
   */
  secretStoreBackend: string;
  /** Secrets Manager path prefix owning the per-user egress secrets */
  secretsManagerPathPrefix: string;
  /**
   * KEY ARN of a customer-managed KMS key encrypting the vault. Empty uses
   * the AWS-managed Secrets Manager key.
   *
   * Must be a `key/<id>` ARN. Not a key id, not an alias name, and not an
   * alias ARN: this value is used as an IAM policy Resource, and IAM accepts
   * only a key ARN there. An alias ARN is valid solely in a statement
   * governing alias operations, so a policy written with one grants nothing
   * and every vault write fails with AccessDenied.
   *
   * See https://docs.aws.amazon.com/kms/latest/developerguide/cmks-in-iam-policies.html
   */
  secretsManagerKmsKeyArn: string;
  /**
   * URL the auth-server uses to reach the registry's internal egress-token
   * vend listener. Defaults to the dedicated nginx port that the ALB never
   * fronts.
   */
  registryInternalUrl: string;
  /** Browser-facing base URL for the egress OAuth callback */
  oauthCallbackBaseUrl: string;
  /** Comma-separated allowlist of trusted IdP hosts for egress OAuth */
  oauthTrustedIdpHosts: string;
  /**
   * WHITESPACE-separated allowlist of audiences an obo_exchange server may
   * register as its target (`raw.split()` in registry/core/schemas.py, so a
   * comma-separated value parses as one audience and matches nothing).
   *
   * Optional, and empty is not "allow anything": the application falls back
   * to a shape rule that admits only an internal MCP server's own IdP
   * audience (an `api://` App ID URI or a bare non-GUID client id) and rejects
   * every `https://` host URL and every bare GUID, so shared first-party
   * resources cannot be targeted. A bare-GUID target has to be listed here.
   * Set this to narrow that to an explicit, audited set.
   */
  oboAllowedAudiences: string;
  /** Seconds before expiry at which a vaulted token is refreshed */
  tokenRefreshSkewSeconds: number;
  /** Lifetime of a pending egress OAuth consent state */
  stateTtlSeconds: number;
  /** Enable the OBO exchange result cache */
  oboCacheEnabled: boolean;
  /** Upper bound on how long an OBO result may be reused */
  oboCacheMaxTtlSeconds: number;
  /** Safety margin subtracted from a cached OBO token's expiry */
  oboCacheExpirySkewSeconds: number;
  /** Max connections in the shared outbound HTTP pool */
  httpPoolMaxConnections: number;
  /** Max keep-alive connections in the shared outbound HTTP pool */
  httpPoolMaxKeepalive: number;
  /** Keep-alive expiry for the shared outbound HTTP pool */
  httpPoolKeepaliveExpirySeconds: number;
  /** Connect retries for the shared outbound HTTP pool */
  httpPoolConnectRetries: number;
  /** Route outbound egress traffic through a forward proxy */
  forwardProxyEnabled: boolean;
  /** CA bundle path for the forward proxy's TLS interception */
  forwardProxyCaBundle: string;
}

export interface GithubConfig {
  /** GitHub Personal Access Token (from env CDK_GITHUB_PAT) */
  pat: string;
  /** GitHub App ID */
  appId: string;
  /** GitHub App Installation ID */
  appInstallationId: string;
  /** GitHub App private key PEM (from env CDK_GITHUB_APP_PRIVATE_KEY) */
  appPrivateKey: string;
  /** Comma-separated extra GitHub hosts for enterprise instances */
  extraHosts: string;
  /** GitHub API base URL */
  apiBaseUrl: string;
}

export interface AnsConfig {
  /** Enable ANS integration for agent identity verification */
  integrationEnabled: boolean;
  /** ANS API endpoint URL */
  apiEndpoint: string;
  /** ANS API key (from env CDK_ANS_API_KEY) */
  apiKey: string;
  /** ANS API secret (from env CDK_ANS_API_SECRET) */
  apiSecret: string;
  /** ANS API request timeout in seconds */
  apiTimeoutSeconds: number;
  /** How often to re-sync ANS verification status (in hours) */
  syncIntervalHours: number;
  /** Cache TTL for ANS verification results (in seconds) */
  verificationCacheTtlSeconds: number;
}

export interface TelemetryConfig {
  /** Disable anonymous startup telemetry ('1' to opt out) */
  disabled: string;
  /** Disable daily heartbeat telemetry only ('1' to opt out) */
  optOut: string;
  /** Heartbeat telemetry interval in minutes */
  heartbeatIntervalMinutes: string;
  /** Enable telemetry debug mode */
  debug: string;
}

export interface RegistryCardConfig {
  /** Human-readable registry name for federation and discovery */
  name: string;
  /** Organization that operates this registry */
  organizationName: string;
  /** Registry description for federation discovery */
  description: string;
  /** Contact email for registry administrators */
  contactEmail: string;
  /** Documentation or support URL for this registry */
  contactUrl: string;
}

export interface CloudFrontConfig {
  /** Enable CloudFront distributions for HTTPS */
  enabled: boolean;
  /** Name of the managed prefix list for ALB ingress */
  prefixListName: string;
}

export interface SessionConfig {
  /** Enable secure flag on session cookies */
  cookieSecure: boolean;
  /** Domain for session cookies */
  cookieDomain: string;
  /**
   * Name the session cookie __Host-<name> (Secure, Path=/, no Domain). Requires
   * cookieSecure and an empty cookieDomain; cdk synth fails otherwise, as both
   * services would refuse to start.
   */
  cookieHostPrefix: boolean;
  /**
   * Browser session lifetime in seconds: the auth server's session store TTL,
   * the session cookie's Max-Age, and the registry's cookie signature max age.
   * Rendered into both services' environments so the two always match. Must be
   * a positive integer; both services refuse to start otherwise.
   */
  maxAgeSeconds: number;
  /**
   * Comma-separated exact-match allowlist of OAuth login/logout redirect URIs
   * (open-redirect hardening). When set, an absolute redirect_uri is accepted
   * only if it exactly matches an entry; relative paths are always allowed.
   * Empty falls back to the weaker cookie-domain heuristic.
   */
  oauth2AllowedRedirectUris: string;
  /** Store OAuth provider tokens in session cookies */
  oauthStoreTokensInSession: boolean;
}

export interface UiTabsConfig {
  /** Show the MCP Servers tab in the UI */
  showServersTab: boolean;
  /** Show the Virtual MCP Servers tab in the UI */
  showVirtualServersTab: boolean;
  /** Show the Skills tab in the UI */
  showSkillsTab: boolean;
  /** Show the Agents tab in the UI */
  showAgentsTab: boolean;
}

export interface StaticTokenAuthConfig {
  /** Enable static token auth for Registry API endpoints */
  registryStaticTokenAuthEnabled: boolean;
  /** Static API key for Registry API (from env CDK_REGISTRY_API_TOKEN) */
  registryApiToken: string;
  /** Maximum JWT tokens that can be vended per user per hour */
  maxTokensPerUserPerHour: number;
  /** Enable M2M direct client registration */
  m2mDirectRegistrationEnabled: boolean;
}

export interface MonitoringConfig {
  /** Enable CloudWatch monitoring and alarms */
  enabled: boolean;
  /** Email address for CloudWatch alarm notifications */
  alarmEmail: string;
  /** SNS topic ARN for CloudWatch alarm notifications */
  alarmSnsTopicArn: string;
}

// ---------------------------------------------------------------------------
// Top-level RegistryConfig interface
// ---------------------------------------------------------------------------

export interface RegistryConfig {
  /** Whether this registry stack is enabled for deployment */
  enabled: boolean;
  /** Name of the deployment */
  name: string;
  /** AWS region for deployment */
  awsRegion: string;
  /** AWS account ID for deployment */
  awsAccountId: string;
  /** CIDR block for VPC */
  vpcCidr: string;
  /** List of CIDR blocks allowed to access the ALB */
  ingressCidrBlocks: string[];
  /** Enable Route53 DNS records and ACM certificates */
  enableRoute53Dns: boolean;
  /** Use region-based domains */
  useRegionalDomains: boolean;
  /** Base domain for regional domains */
  baseDomain: string;
  /** ARN of ACM certificate for HTTPS */
  certificateArn: string;
  /** Storage backend: 'file' or 'documentdb' */
  storageBackend: string;
  /** Deployment mode: 'with-gateway' or 'registry-only' */
  deploymentMode: string;
  /** Registry mode: 'full', 'skills-only', 'mcp-servers-only', 'agents-only' */
  registryMode: string;
  /** Enable full observability pipeline */
  enableObservability: boolean;
  /** Enable WAFv2 Web ACLs for ALBs */
  enableWaf: boolean;
  /** Whether to create CodeBuild resources */
  createCodebuild: boolean;
  /** Comma-separated list of prefixes to filter IdP groups */
  idpGroupFilterPrefix: string;
  /** Disable auto-registration of the built-in airegistry-tools server */
  disableAiRegistryToolsServer: string;
  /** Grafana admin password (from env CDK_GRAFANA_ADMIN_PASSWORD) */
  grafanaAdminPassword: string;
  /**
   * Comma-separated CIDRs (or bare IPs) of the trusted proxy hop in front of
   * the bundled nginx, used for nginx's set_real_ip_from. Empty resolves to
   * this stack's VPC CIDR, because an ECS task always sits behind the ALB.
   * Without it, audit records and the per-IP rate-limit zones see the ALB's
   * address and collapse every client into one bucket.
   */
  trustedRealIpCidrs: string;

  // Nested config sections
  documentdb: DocumentDbConfig;
  keycloak: KeycloakConfig;
  images: ImagesConfig;
  replicas: ReplicasConfig;
  embeddings: EmbeddingsConfig;
  security: SecurityConfig;
  healthChecks: HealthChecksConfig;
  federation: FederationConfig;
  audit: AuditConfig;
  otel: OtelConfig;
  entra: EntraConfig;
  okta: OktaConfig;
  auth0: Auth0Config;
  cognito: CognitoConfig;
  pingfederate: PingFederateConfig;
  egress: EgressConfig;
  github: GithubConfig;
  ans: AnsConfig;
  telemetry: TelemetryConfig;
  registryCard: RegistryCardConfig;
  cloudfront: CloudFrontConfig;
  session: SessionConfig;
  uiTabs: UiTabsConfig;
  staticTokenAuth: StaticTokenAuthConfig;
  monitoring: MonitoringConfig;
}

// ---------------------------------------------------------------------------
// Default configuration matching Terraform variable defaults
// ---------------------------------------------------------------------------

export const DEFAULT_REGISTRY_CONFIG: RegistryConfig = {
  enabled: true,
  name: 'mcp-gateway',
  awsRegion: 'us-west-2',
  awsAccountId: '',
  trustedRealIpCidrs: '',
  vpcCidr: '10.0.0.0/16',
  ingressCidrBlocks: ['0.0.0.0/0'],
  enableRoute53Dns: true,
  useRegionalDomains: true,
  baseDomain: 'mycorp.click',
  certificateArn: '',
  storageBackend: 'file',
  deploymentMode: 'with-gateway',
  registryMode: 'full',
  enableObservability: true,
  enableWaf: true,
  createCodebuild: false,
  idpGroupFilterPrefix: '',
  disableAiRegistryToolsServer: 'false',
  grafanaAdminPassword: '',

  documentdb: {
    adminUsername: 'docdbadmin',
    adminPassword: '',
    shardCapacity: 2,
    shardCount: 1,
    instanceClass: 'db.t3.medium',
    replicaCount: 0,
    database: 'mcp_registry',
    namespace: 'default',
    useTls: true,
    useIam: false,
  },

  keycloak: {
    adminUser: 'admin',
    adminPassword: '',
    databaseUsername: 'keycloak',
    databasePassword: '',
    databaseMinAcu: 0.5,
    databaseMaxAcu: 2,
    logLevel: 'INFO',
    domain: '',
    rootDomain: '',
  },

  images: {
    registry: '',
    authServer: 'mcpgateway/auth-server:latest',
    keycloak: 'quay.io/keycloak/keycloak:26.2.4',
    currenttime: '',
    mcpgw: '',
    realserverfaketools: '',
    flightBookingAgent: '',
    travelAssistantAgent: '',
    metricsService: '',
    grafana: '',
  },

  replicas: {
    registry: 1,
    auth: 1,
    currenttime: 1,
    mcpgw: 1,
    realserverfaketools: 1,
    flightBookingAgent: 1,
    travelAssistantAgent: 1,
  },

  embeddings: {
    provider: 'sentence-transformers',
    modelName: 'all-MiniLM-L6-v2',
    modelDimensions: 384,
    awsRegion: 'us-east-1',
    apiKey: '',
  },

  security: {
    scanEnabled: false,
    scanOnRegistration: false,
    blockUnsafeServers: false,
    analyzers: 'yara',
    scanTimeout: 60,
    addPendingTag: false,
  },

  healthChecks: {
    intervalSeconds: 300,
    toolRefreshEnabled: false,
    toolRefreshEveryCycles: 3,
    toolFetchMaxConcurrency: 5,
  },

  federation: {
    registryId: '',
    staticTokenAuthEnabled: false,
    staticToken: '',
    encryptionKey: '',
    awsRegistryFederationEnabled: false,
    awsRegistryFederationAssumeRoleArns: [],
  },

  audit: {
    logEnabled: true,
    logTtlDays: 7,
  },

  otel: {
    otlpEndpoint: '',
    exporterOtlpHeaders: '',
    otlpExportIntervalMs: 30000,
    exporterOtlpMetricsTemporalityPreference: 'cumulative',
    adotCollectorImage: 'public.ecr.aws/aws-observability/aws-otel-collector:latest',
  },

  entra: {
    enabled: false,
    tenantId: '',
    clientId: '',
    clientSecret: '',
    scopeFormat: '',
    applicationIdUri: '',
  },

  okta: {
    enabled: false,
    domain: '',
    clientId: '',
    clientSecret: '',
    m2mClientId: '',
    m2mClientSecret: '',
    apiToken: '',
    authServerId: '',
  },

  auth0: {
    enabled: false,
    domain: '',
    clientId: '',
    clientSecret: '',
    audience: '',
    groupsClaim: 'https://mcp-gateway/groups',
    m2mClientId: '',
    m2mClientSecret: '',
    managementApiToken: '',
  },

  cognito: {
    enabled: false,
    userPoolId: '',
    clientId: '',
    clientSecretArn: '',
    domain: '',
    m2mClientIds: '',
  },

  pingfederate: {
    enabled: false,
    baseUrl: '',
    externalUrl: '',
    clientId: '',
    clientSecretArn: '',
    m2mClientId: '',
    m2mClientSecretArn: '',
    applicationIdUri: '',
    groupsClaim: 'groups',
    m2mAllowedAudiences: '',
    adminUrl: 'https://pingfederate:9999',
    adminUser: 'administrator',
    adminPasswordArn: '',
  },

  egress: {
    enabled: false,
    credentialEncryptionKeyArn: '',
    secretStoreBackend: 'secrets-manager',  // pragma: allowlist secret
    secretsManagerPathPrefix: 'mcp/egress',  // pragma: allowlist secret
    secretsManagerKmsKeyArn: '',
    registryInternalUrl: 'http://registry:8091',
    oauthCallbackBaseUrl: '',
    oauthTrustedIdpHosts: '',
    oboAllowedAudiences: '',
    tokenRefreshSkewSeconds: 300,
    stateTtlSeconds: 600,
    oboCacheEnabled: false,
    oboCacheMaxTtlSeconds: 300,
    oboCacheExpirySkewSeconds: 30,
    httpPoolMaxConnections: 100,
    httpPoolMaxKeepalive: 20,
    httpPoolKeepaliveExpirySeconds: 30,
    httpPoolConnectRetries: 1,
    forwardProxyEnabled: false,
    forwardProxyCaBundle: '',
  },

  github: {
    pat: '',
    appId: '',
    appInstallationId: '',
    appPrivateKey: '',
    extraHosts: '',
    apiBaseUrl: 'https://api.github.com',
  },

  ans: {
    integrationEnabled: false,
    apiEndpoint: 'https://api.godaddy.com',
    apiKey: '',
    apiSecret: '',
    apiTimeoutSeconds: 30,
    syncIntervalHours: 6,
    verificationCacheTtlSeconds: 3600,
  },

  telemetry: {
    disabled: '',
    optOut: '',
    heartbeatIntervalMinutes: '1440',
    debug: 'false',
  },

  registryCard: {
    name: '',
    organizationName: '',
    description: '',
    contactEmail: '',
    contactUrl: '',
  },

  cloudfront: {
    enabled: false,
    prefixListName: '',
  },

  session: {
    cookieSecure: true,
    cookieDomain: '',
    cookieHostPrefix: false,
    maxAgeSeconds: 28800,
    oauth2AllowedRedirectUris: '',
    oauthStoreTokensInSession: false,
  },

  uiTabs: {
    showServersTab: true,
    showVirtualServersTab: true,
    showSkillsTab: true,
    showAgentsTab: true,
  },

  staticTokenAuth: {
    registryStaticTokenAuthEnabled: false,
    registryApiToken: '',
    maxTokensPerUserPerHour: 100,
    m2mDirectRegistrationEnabled: true,
  },

  monitoring: {
    enabled: true,
    alarmEmail: '',
    alarmSnsTopicArn: '',
  },
};

// ---------------------------------------------------------------------------
// Helper: deep merge objects (target is mutated)
// ---------------------------------------------------------------------------

function _deepMerge(target: Record<string, any>, source: Record<string, any>): Record<string, any> {
  for (const key of Object.keys(source)) {
    if (key === '__proto__' || key === 'constructor' || key === 'prototype') continue;
    const sourceVal = source[key];
    const targetVal = target[key];

    if (
      sourceVal !== null &&
      typeof sourceVal === 'object' &&
      !Array.isArray(sourceVal) &&
      targetVal !== null &&
      typeof targetVal === 'object' &&
      !Array.isArray(targetVal)
    ) {
      _deepMerge(targetVal, sourceVal);
    } else if (sourceVal === null) {
      // A YAML key written with no value parses as null: `oboAllowedAudiences:`
      // is a natural way to mean "leave it empty", and it is one keystroke
      // from a commented-out example. Assigning null would either crash a
      // string method during validation or reach a container as the literal
      // `Value: null`, so keep the default instead. A key an operator wants
      // genuinely empty is written `key: ""`, which is not null and is
      // assigned below.
      continue;
    } else if (sourceVal !== undefined) {
      target[key] = sourceVal;
    }
  }
  return target;
}

// ---------------------------------------------------------------------------
// Helper: inject sensitive values from environment variables
// ---------------------------------------------------------------------------

// Map of env var name to dotted path within RegistryConfig.
const SECRET_ENV_PATHS: Record<string, string> = {
  CDK_KEYCLOAK_ADMIN_PASSWORD: 'keycloak.adminPassword',
  CDK_KEYCLOAK_DATABASE_PASSWORD: 'keycloak.databasePassword',
  CDK_DOCUMENTDB_ADMIN_PASSWORD: 'documentdb.adminPassword',
  CDK_EMBEDDINGS_API_KEY: 'embeddings.apiKey',
  CDK_ENTRA_CLIENT_SECRET: 'entra.clientSecret',
  CDK_OKTA_CLIENT_SECRET: 'okta.clientSecret',
  CDK_OKTA_M2M_CLIENT_SECRET: 'okta.m2mClientSecret',
  CDK_OKTA_API_TOKEN: 'okta.apiToken',
  CDK_AUTH0_CLIENT_SECRET: 'auth0.clientSecret',
  CDK_AUTH0_M2M_CLIENT_SECRET: 'auth0.m2mClientSecret',
  CDK_AUTH0_MANAGEMENT_API_TOKEN: 'auth0.managementApiToken',
  // No CDK_* entry for the Cognito, PingFederate or egress secrets: they are
  // referenced by ARN, never by value. A literal would be written into the
  // synthesized template and sit in plaintext in cdk.out and in the deployed
  // stack. The entries above predate that decision; see the follow-up issue
  // linked from infra/docs/CDK-INFRASTRUCTURE.md.
  CDK_REGISTRY_API_TOKEN: 'staticTokenAuth.registryApiToken',
  CDK_FEDERATION_STATIC_TOKEN: 'federation.staticToken',
  CDK_FEDERATION_ENCRYPTION_KEY: 'federation.encryptionKey',
  CDK_ANS_API_KEY: 'ans.apiKey',
  CDK_ANS_API_SECRET: 'ans.apiSecret',
  CDK_GITHUB_PAT: 'github.pat',
  CDK_GITHUB_APP_PRIVATE_KEY: 'github.appPrivateKey',
  CDK_GRAFANA_ADMIN_PASSWORD: 'grafanaAdminPassword',
  CDK_OTEL_EXPORTER_OTLP_HEADERS: 'otel.exporterOtlpHeaders',
};

function _injectSecrets(config: RegistryConfig): void {
  for (const [envVar, dottedPath] of Object.entries(SECRET_ENV_PATHS)) {
    const value = process.env[envVar];
    if (!value) continue;
    const keys = dottedPath.split('.');
    const last = keys.pop()!;
    let target: any = config;
    for (const k of keys) target = target[k];
    target[last] = value;
  }
}

// ---------------------------------------------------------------------------
// Helpers: provider validation, run at synth so a misconfiguration fails here
// rather than as a crash-looping task
// ---------------------------------------------------------------------------

// ---------------------------------------------------------------------------
// Values that reach an IAM policy Resource
// ---------------------------------------------------------------------------
//
// Every config value interpolated into an ARN inside an IAM policy goes
// through _requireIamResourceSegment below. Validating one such field and
// trusting the rest is how a wildcard gets in: `*` and `?` are IAM wildcards,
// so one in any of these positions silently widens the grant, and a `:` adds
// ARN fields the policy author never wrote.
//
// The fields, each covered by a case in the "values that reach an IAM policy
// Resource" block of test/registry-config.test.ts:
//
//   cognito.userPoolId                   -> cognito-idp:...:userpool/<id>
//   egress.secretsManagerPathPrefix      -> secretsmanager:...:secret:<prefix>/*
//   egress.secretsManagerKmsKeyArn       -> the Resource itself
//   cognito.clientSecretArn              -> secretsmanager GetSecretValue
//   pingfederate.clientSecretArn         -> same
//   pingfederate.m2mClientSecretArn      -> same
//   pingfederate.adminPasswordArn        -> same
//   egress.credentialEncryptionKeyArn    -> same
//
// Add a field here and to that test block in the same change.

/** IAM wildcards and the ARN field separator. Never valid inside a segment. */
const IAM_RESOURCE_UNSAFE_PATTERN = /[*?:\s]/;

/**
 * Characters AWS allows in a Secrets Manager secret name. Also the legal
 * alphabet for a name prefix, since a prefix is just a name's leading part.
 * https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_CreateSecret.html
 */
const SECRET_NAME_ALPHABET = 'A-Za-z0-9/_+=.@\\-';  // pragma: allowlist secret

/**
 * Reject a value that becomes part of an ARN in an IAM policy.
 *
 * Checks the never-safe characters first so the error names the real problem,
 * then the field's own alphabet.
 */
function _requireIamResourceSegment(
  value: string,
  settingName: string,
  pattern: RegExp,
  expected: string,
): void {
  const unsafe = IAM_RESOURCE_UNSAFE_PATTERN.exec(value);
  if (unsafe) {
    throw new Error(
      `${settingName} must not contain ${JSON.stringify(unsafe[0])}, but got ` +
      `${JSON.stringify(value)}. This value is interpolated into an ARN in an IAM policy ` +
      'Resource, where `*` and `?` are wildcards that widen the grant and `:` adds ARN ' +
      'fields. A wildcard here would grant access well beyond the intended resource.',
    );
  }
  if (!pattern.test(value)) {
    throw new Error(
      `${settingName} must be ${expected}, but got ${JSON.stringify(value)}. It is ` +
      'interpolated into an ARN in an IAM policy Resource.',
    );
  }
}

/** Cognito user pool ids look like us-west-2_AbC123. */
const COGNITO_USER_POOL_ID_PATTERN = /^[a-z]{2}(-[a-z]+)+-\d+_[A-Za-z0-9]+$/;

/**
 * A Secrets Manager name prefix: the secret-name alphabet, with no leading,
 * trailing or doubled `/`.
 *
 * Edge slashes are rejected rather than normalized because the prefix is used
 * in two places that would then disagree. `mcp/egress/` yields the IAM
 * Resource `secret:mcp/egress//*`, while the application normalizes the
 * prefix before building a secret name, so the paths it actually writes never
 * match the grant and every vault operation fails with AccessDenied. Failing
 * at synth is clearer than either guessing the application's normalization or
 * shipping a grant that silently matches nothing.
 */
const SECRETS_PATH_PREFIX_PATTERN =
  new RegExp(`^[${SECRET_NAME_ALPHABET}]+$`);

/** True when a prefix has a leading, trailing or doubled slash. */
function _hasSlashProblem(value: string): boolean {
  return value.startsWith('/') || value.endsWith('/') || value.includes('//');
}

/**
 * Storage backends the application treats as Mongo-family.
 *
 * Source of truth: `MONGODB_BACKENDS` in registry/core/config.py. Asserted in
 * test/registry-config.test.ts so a drift here is visible.
 */
const MONGO_STORAGE_BACKENDS = ['documentdb', 'mongodb', 'mongodb-ce', 'mongodb-atlas'];

/**
 * Egress secret-store backends this stack supports.
 *
 * The application accepts `secrets-manager` and `openbao`
 * (registry/core/config.py, `secret_store_backend`). This stack provisions no
 * OpenBao, so `secrets-manager` is the only value it can honor. Anything else,
 * including a Mongo backend name, is rejected here rather than passed through
 * for the application to reject at startup.
 */
const EGRESS_SECRET_STORE_BACKENDS = ['secrets-manager'];

/**
 * A complete Secrets Manager secret ARN, including the six-character suffix
 * the service appends. `Secret.fromSecretCompleteArn` needs the complete form,
 * and a partial ARN would silently grant and read the wrong secret.
 *
 * The name portion is restricted to the characters AWS allows in a secret
 * name, which is what keeps an IAM wildcard out of it. A permissive `.+` here
 * accepted `secret:*-AbCdEf`, and because every Secrets Manager ARN ends in a
 * dash plus six characters, that single value as an IAM Resource matches every
 * secret in the account and region.
 */
const SECRET_COMPLETE_ARN_PATTERN = new RegExp(
  `^arn:aws[a-z-]*:secretsmanager:[a-z0-9-]+:\\d{12}:secret:` +
  `[${SECRET_NAME_ALPHABET}]+-[A-Za-z0-9]{6}$`,
);

/**
 * A KMS KEY ARN. Deliberately not an alias ARN, a bare key id or an alias
 * name: IAM accepts only a key ARN in the Resource element of a statement
 * granting cryptographic operations, and an alias ARN there grants nothing.
 * https://docs.aws.amazon.com/kms/latest/developerguide/cmks-in-iam-policies.html
 */
const KMS_KEY_ARN_PATTERN =
  /^arn:aws[a-z-]*:kms:[a-z0-9-]+:\d{12}:key\/[A-Za-z0-9-]+$/;

function _requireSecretArn(value: string, settingName: string): void {
  if (SECRET_COMPLETE_ARN_PATTERN.test(value)) return;

  // Echo the value only when it is recognizably an ARN. The most likely
  // mistake in a field named *Arn is pasting the secret itself, and an error
  // message reaches logs and CI output, so describe a non-ARN rather than
  // printing it.
  const detail = value.startsWith('arn:')
    ? `got ${JSON.stringify(value)}`
    : `got a ${value.length}-character value that is not an ARN (not shown, in case it is ` +
      'the secret itself)';

  throw new Error(
    `${settingName} must be a complete Secrets Manager secret ARN, ending in the ` +
    'six-character suffix AWS appends ' +
    '(arn:aws:secretsmanager:<region>:<account>:secret:<name>-AbCdEf), ' +
    `${detail}. The secret is referenced, never embedded, so that its value does not end ` +
    'up in plaintext in the synthesized template. Create it out of band, for example with ' +
    '`aws secretsmanager create-secret`, then put the returned ARN here.',
  );
}

function _validateCognito(config: RegistryConfig): void {
  if (!config.cognito.enabled) return;

  // cognito.domain is not required: the auth-server derives the prefix from
  // the user pool id when it is empty, for both the browser login URLs and
  // the token endpoints (resolve_cognito_domain_prefix in
  // auth_server/providers/cognito.py).
  const missing: string[] = [];
  if (!config.cognito.userPoolId) missing.push('cognito.userPoolId');
  if (!config.cognito.clientId) missing.push('cognito.clientId');
  if (!config.cognito.clientSecretArn) missing.push('cognito.clientSecretArn');
  if (missing.length > 0) {
    throw new Error(
      `cognito.enabled=true needs ${missing.join(', ')}. The auth-server resolves the pool, ` +
      'client and issuer from stack config only, never from a request header, so an ' +
      'incomplete config has no safe fallback.',
    );
  }

  _requireSecretArn(config.cognito.clientSecretArn, 'cognito.clientSecretArn');

  _requireIamResourceSegment(
    config.cognito.userPoolId,
    'cognito.userPoolId',
    COGNITO_USER_POOL_ID_PATTERN,
    'a Cognito user pool id of the form us-west-2_AbC123',
  );

  // COGNITO_DOMAIN is a PREFIX. The auth-server builds
  // https://<domain>.auth.<region>.amazoncognito.com, so a full hostname here
  // yields a doubled one and every OAuth endpoint 404s.
  if (config.cognito.domain.includes('.')) {
    throw new Error(
      `cognito.domain must be the hosted-UI domain PREFIX, not a hostname, got ` +
      `${JSON.stringify(config.cognito.domain)}. The auth-server appends ` +
      '".auth.<region>.amazoncognito.com", so pass "my-pool", not ' +
      '"my-pool.auth.us-west-2.amazoncognito.com". Leave it empty to derive the domain ' +
      'from the user pool id.',
    );
  }
}

function _validatePingFederate(config: RegistryConfig): void {
  if (!config.pingfederate.enabled) return;

  const missing: string[] = [];
  if (!config.pingfederate.baseUrl) missing.push('pingfederate.baseUrl');
  if (!config.pingfederate.clientId) missing.push('pingfederate.clientId');
  if (!config.pingfederate.clientSecretArn) missing.push('pingfederate.clientSecretArn');
  // The registry's PingFederate admin client raises without a password, so
  // login would work while user administration and service-account creation
  // failed. Fail here instead.
  if (!config.pingfederate.adminPasswordArn) missing.push('pingfederate.adminPasswordArn');
  if (!config.pingfederate.adminUrl) missing.push('pingfederate.adminUrl');
  if (!config.pingfederate.adminUser) missing.push('pingfederate.adminUser');
  if (missing.length > 0) {
    throw new Error(`pingfederate.enabled=true needs ${missing.join(', ')}.`);
  }

  _requireSecretArn(config.pingfederate.clientSecretArn, 'pingfederate.clientSecretArn');
  _requireSecretArn(config.pingfederate.adminPasswordArn, 'pingfederate.adminPasswordArn');

  // Optional: the provider falls back to the web client secret when no
  // separate M2M secret is set, so only validate a value that is present.
  if (config.pingfederate.m2mClientSecretArn) {
    _requireSecretArn(config.pingfederate.m2mClientSecretArn, 'pingfederate.m2mClientSecretArn');
  }

  // pingfederate.m2mAllowedAudiences is deliberately NOT required alongside
  // m2mClientId. The provider always validates `aud` against the union of the
  // client ids, applicationIdUri and this list, so an empty list accepts only
  // the configured client ids and is already fail-closed
  // (auth_server/providers/pingfederate.py). Requiring it would block a
  // configuration both Terraform and the application support.

  if (!config.pingfederate.groupsClaim) {
    throw new Error(
      'pingfederate.groupsClaim must name the claim carrying group membership (for ' +
      'example "groups"). Group membership drives authorization, so it cannot be empty.',
    );
  }
}

function _validateEgress(config: RegistryConfig): void {
  if (!EGRESS_SECRET_STORE_BACKENDS.includes(config.egress.secretStoreBackend)) {
    throw new Error(
      `egress.secretStoreBackend must be one of ${EGRESS_SECRET_STORE_BACKENDS.join(', ')}, ` +
      `got ${JSON.stringify(config.egress.secretStoreBackend)}.`,
    );
  }
  _requireIamResourceSegment(
    config.egress.secretsManagerPathPrefix,
    'egress.secretsManagerPathPrefix',
    SECRETS_PATH_PREFIX_PATTERN,
    'a Secrets Manager name prefix using only letters, digits and /_+=.@-',
  );
  if (_hasSlashProblem(config.egress.secretsManagerPathPrefix)) {
    throw new Error(
      'egress.secretsManagerPathPrefix must not start or end with "/" or contain "//", but ' +
      `got ${JSON.stringify(config.egress.secretsManagerPathPrefix)}. The IAM grant is built ` +
      'as "secret:<prefix>/*", so an edge slash yields a doubled separator that no secret ' +
      'the application writes will match, and every vault operation fails with AccessDenied. ' +
      'Write it as "mcp/egress".',
    );
  }

  if (config.egress.credentialEncryptionKeyArn) {
    // Only the reference is checked here. The application enforces the key's
    // own rules (at least 32 bytes, no placeholder markers) at startup, and it
    // is the only side that ever sees the value.
    _requireSecretArn(
      config.egress.credentialEncryptionKeyArn,
      'egress.credentialEncryptionKeyArn',
    );
  }

  // The value is both passed to the application as SECRETS_MANAGER_KMS_KEY_ID
  // and used as an IAM policy Resource. A wildcard would grant kms:Decrypt on
  // every key in the account, and a bare key id is not a valid IAM Resource.
  if (config.egress.secretsManagerKmsKeyArn) {
    // Not routed through _requireIamResourceSegment: this value IS the whole
    // Resource, so it legitimately contains the ARN's own colons. The pattern
    // below is strict enough to exclude a wildcard on its own.
    if (!KMS_KEY_ARN_PATTERN.test(config.egress.secretsManagerKmsKeyArn)) {
      throw new Error(
        'egress.secretsManagerKmsKeyArn must be a KMS KEY ARN ' +
        '(arn:aws:kms:<region>:<account>:key/<id>), got ' +
        `${JSON.stringify(config.egress.secretsManagerKmsKeyArn)}. IAM accepts only a key ` +
        'ARN in the Resource element of a statement granting cryptographic operations: an ' +
        'alias ARN, an alias name or a bare key id grants nothing there, so every vault ' +
        'write would fail with AccessDenied. A wildcard is rejected for the opposite ' +
        'reason, since it would grant decrypt on every key in the account. Resolve an alias ' +
        'with `aws kms describe-key --key-id alias/<name> --query KeyMetadata.Arn`.',
      );
    }
  }

  if (!config.egress.enabled) {
    if (config.egress.oboCacheEnabled) {
      throw new Error(
        'egress.oboCacheEnabled=true needs egress.enabled=true; the cache only applies to ' +
        'obo_exchange servers and the registry refuses to start on the pair.',
      );
    }
    return;
  }

  // The vault's refresh lease lock lives in Mongo, so the file backend has no
  // home for it and cross-replica single-flight refresh would silently drop.
  if (!MONGO_STORAGE_BACKENDS.includes(config.storageBackend)) {
    throw new Error(
      `egress.enabled=true needs a Mongo-family storageBackend (one of ` +
      `${MONGO_STORAGE_BACKENDS.join(', ')}), got ${JSON.stringify(config.storageBackend)}. ` +
      'The registry refuses to start otherwise.',
    );
  }

  // egress.oboAllowedAudiences is deliberately NOT required. Empty is not
  // "allow anything": the application falls back to a shape rule that admits
  // only an internal MCP server's own IdP audience and rejects every
  // https:// host URL. Requiring it here would block a configuration both
  // Terraform and the application support.
  //
  // A comma-separated value, though, parses as ONE audience and matches
  // nothing, because the application splits on whitespace
  // (registry/core/schemas.py). Catch that rather than let it fail silently.
  if (config.egress.oboAllowedAudiences.includes(',')) {
    throw new Error(
      'egress.oboAllowedAudiences is whitespace-separated, not comma-separated, but got ' +
      `${JSON.stringify(config.egress.oboAllowedAudiences)}. The application splits on ` +
      'whitespace, so a comma-separated list is read as a single audience and matches ' +
      'nothing, silently rejecting every obo_exchange registration.',
    );
  }

  if (config.egress.oboCacheEnabled) {
    if (config.egress.oboCacheExpirySkewSeconds >= config.egress.oboCacheMaxTtlSeconds) {
      throw new Error(
        `egress.oboCacheExpirySkewSeconds (${config.egress.oboCacheExpirySkewSeconds}) must be ` +
        `below egress.oboCacheMaxTtlSeconds (${config.egress.oboCacheMaxTtlSeconds}); ` +
        'otherwise every cached token is born expired.',
      );
    }
    if (config.egress.secretStoreBackend !== 'secrets-manager') {  // pragma: allowlist secret
      throw new Error(
        'egress.oboCacheEnabled=true needs egress.secretStoreBackend="secrets-manager" on ECS ' +  // pragma: allowlist secret
        `(the only backend this stack provisions), got ${JSON.stringify(config.egress.secretStoreBackend)}.`,
      );
    }
  }
}

// ---------------------------------------------------------------------------
// Public: load and validate config
// ---------------------------------------------------------------------------

/**
 * Load registry configuration from a YAML file, merge with defaults,
 * and inject sensitive values from environment variables.
 *
 * The YAML file should have a top-level `registry` key whose structure
 * mirrors RegistryConfig. Missing keys fall back to DEFAULT_REGISTRY_CONFIG.
 *
 * @param configPath - Absolute path to the YAML configuration file
 * @returns Fully resolved RegistryConfig
 */
export function loadRegistryConfig(configPath: string): RegistryConfig {
  // Start with a deep copy of defaults
  const config: RegistryConfig = JSON.parse(JSON.stringify(DEFAULT_REGISTRY_CONFIG));

  // Read and parse YAML
  if (fs.existsSync(configPath)) {
    const raw = fs.readFileSync(configPath, 'utf8');
    let parsed: Record<string, any> | null;
    try {
      parsed = yaml.load(raw) as Record<string, any> | null;
    } catch (err) {
      throw new Error(`Failed to parse config YAML at ${configPath}: ${err}`);
    }

    if (parsed && typeof parsed === 'object' && parsed.registry) {
      _deepMerge(config as unknown as Record<string, any>, parsed.registry);
    }
  } else {
    console.warn(`Config file not found at ${configPath}, using defaults`);
  }

  // Inject secrets from environment variables (overrides YAML values)
  _injectSecrets(config);

  // Validate required secrets at synth time to fail fast
  if (config.enabled) {
    const requiredSecrets: Array<[string, string]> = [
      [config.keycloak.adminPassword, 'CDK_KEYCLOAK_ADMIN_PASSWORD'],
      [config.keycloak.databasePassword, 'CDK_KEYCLOAK_DATABASE_PASSWORD'],
      [config.documentdb.adminPassword, 'CDK_DOCUMENTDB_ADMIN_PASSWORD'],
      // Required once observability is on, which the shipped config.yaml
      // enables. The secret is created from this value, and CloudFormation
      // rejects a SecretString shorter than one character, so an unset
      // password fails the deploy after a clean synth. Fail here instead.
      ...(config.enableObservability
        ? [[config.grafanaAdminPassword, 'CDK_GRAFANA_ADMIN_PASSWORD'] as [string, string]]
        : []),
    ];
    const missing = requiredSecrets
      .filter(([val]) => !val || val.length < 8)
      .map(([, name]) => name);
    if (missing.length > 0) {
      throw new Error(
        `Required secrets missing or too short (min 8 chars): ${missing.join(', ')}. ` +
        'Set them as environment variables before running cdk synth/deploy.',
      );
    }
  }

  // Both services refuse to start on a non-positive session lifetime; fail
  // at synth instead of as crash-looping tasks.
  const sessionMaxAge = config.session.maxAgeSeconds;
  if (!Number.isInteger(sessionMaxAge) || sessionMaxAge <= 0) {
    throw new Error(
      `session.maxAgeSeconds must be a positive integer number of seconds (for example 28800 for 8 hours), got ${JSON.stringify(sessionMaxAge)}. ` +
      'The registry and auth-server refuse to start on 0, a negative, or a non-integer value.',
    );
  }

  // Browsers drop a __Host- cookie that is not Secure or has a Domain, and both
  // services refuse to start on that pair; fail at synth, as Terraform's
  // preconditions do at plan.
  if (config.session.cookieHostPrefix) {
    const problems: string[] = [];
    if (!config.session.cookieSecure) {
      problems.push('session.cookieSecure must be true');
    }
    if ((config.session.cookieDomain ?? '').trim() !== '') {
      problems.push('session.cookieDomain must be empty');
    }
    if (problems.length > 0) {
      throw new Error(
        `session.cookieHostPrefix=true needs a Secure session cookie with no Domain, but ${problems.join(' and ')}. ` +
        'Browsers drop a __Host- cookie otherwise, and the registry and auth-server refuse to start. ' +
        'Fix those values in config.yaml, or set session.cookieHostPrefix=false.',
      );
    }
  }

  _validateCognito(config);
  _validatePingFederate(config);
  _validateEgress(config);

  // Keycloak sslRequired=external blocks admin ops over plain HTTP, so a
  // public HTTPS front is required. Fail fast at synth if neither option is
  // enabled — otherwise KeycloakService throws late during construct build.
  if (!config.enableRoute53Dns && !config.cloudfront.enabled) {
    throw new Error(
      'Keycloak needs a public HTTPS front: set enableRoute53Dns=true (custom ' +
      'domain + ACM) or cloudfront.enabled=true (CloudFront default cert) in config.yaml. ' +
      'Plain HTTP ALB is not supported because Keycloak sslRequired=external ' +
      'blocks admin/OIDC endpoints over unencrypted connections.',
    );
  }

  return config;
}
