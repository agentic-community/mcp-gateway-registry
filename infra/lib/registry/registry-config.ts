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

  // Nested config sections
  documentdb: DocumentDbConfig;
  keycloak: KeycloakConfig;
  images: ImagesConfig;
  replicas: ReplicasConfig;
  embeddings: EmbeddingsConfig;
  security: SecurityConfig;
  federation: FederationConfig;
  audit: AuditConfig;
  otel: OtelConfig;
  entra: EntraConfig;
  okta: OktaConfig;
  auth0: Auth0Config;
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
  },

  entra: {
    enabled: false,
    tenantId: '',
    clientId: '',
    clientSecret: '',
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
