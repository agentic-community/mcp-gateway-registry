/**
 * Shared API types used across the application
 */

// ============================================================================
// Auth Types
// ============================================================================

export interface User {
  user_id: string;
  email?: string;
  username?: string;
  auth_method: 'oidc' | 'password';
  is_admin: boolean;
  roles?: string[];
  groups?: string[];
  session_id?: string;
}

export interface AuthProviders {
  providers: Array<{
    name: string;
    display_name: string;
  }>;
}

export interface CsrfToken {
  csrf_token: string;
}

export interface LoginRequest {
  username: string;
  password: string;
}

// ============================================================================
// Registry: Server Types
// ============================================================================

/** Upstream authentication mode */
export type UpstreamAuthMode = 'gateway-managed' | 'none';

/** Upstream authentication type */
export type UpstreamAuthType =
  | 'none'
  | 'api-key'
  | 'oauth2'
  | 'oidc'
  | 'provider-oauth'
  | 'jwt'
  | 'mtls'
  | 'header-trust';

/** Credential binding strategy */
export type CredentialBinding = 'service' | 'user' | 'agent' | 'user+agent';

/** Upstream credential status for the current principal */
export type UpstreamCredentialStatus = 'configured' | 'missing' | 'expired' | 'revoked';

/** Upstream authentication configuration */
export interface UpstreamAuthConfig {
  mode: UpstreamAuthMode;
  type: UpstreamAuthType;
  provider?: string;
  credential_binding: CredentialBinding;
}

export interface Server {
  display_name: string;
  path: string;
  proxy_pass_url: string;
  description?: string;
  tags?: string[];
  is_enabled: boolean;
  health_status?: 'healthy' | 'unhealthy' | 'unknown';
  last_checked_iso?: string;
  num_tools?: number;
  num_stars?: number;
  is_python?: boolean;
  license?: string;
  upstream_auth?: UpstreamAuthConfig;
  upstream_credential_status?: UpstreamCredentialStatus;
}

export interface ServerDetails extends Server {
  tools?: Tool[];
  supported_transports?: string[];
  metadata?: Record<string, unknown>;
}

export interface Tool {
  name: string;
  description?: string;
  input_schema?: Record<string, unknown>;
}

export interface RegisterServerRequest {
  name: string;
  path: string;
  proxy_pass_url: string;
  description?: string;
  tags?: string[];
}

export interface EditServerRequest {
  name?: string;
  proxy_pass_url?: string;
  description?: string;
  tags?: string[];
}

// ============================================================================
// Registry: A2A Agent Types
// ============================================================================

export interface A2AAgent {
  name: string;
  path: string;
  url?: string;
  description?: string;
  skills?: string[];
  tags?: string[];
  num_skills?: number;
  num_stars?: number;
  is_enabled: boolean;
  provider?: string;
  streaming?: boolean;
  trust_level?: string;
  visibility?: 'public' | 'private';
  health_status?: 'healthy' | 'unhealthy' | 'unknown';
}

export interface RegisterA2AAgentRequest {
  name: string;
  path: string;
  description?: string;
  skills?: string[];
  tags?: string[];
  visibility?: 'public' | 'private';
}

// ============================================================================
// EnforceAI: Agent Types
// ============================================================================

export interface EnforceAIAgent {
  user_id: string;
  agent_id: string;
  scopes: string[];
  allowed_tools?: string[] | null;
  alias?: string | null;
  metadata?: Record<string, unknown> | null;
  revoked_at?: string | null;
  tokens_valid_after?: string | null;
  created_at: string;
  updated_at: string;
}

export interface CreateAgentRequest {
  scopes: string[];
  allowed_tools?: string[] | null;
  alias?: string | null;
  metadata?: Record<string, unknown> | null;
}

export interface UpdateAgentRequest {
  scopes?: string[] | null;
  allowed_tools?: string[] | null;
  alias?: string | null;
  metadata?: Record<string, unknown> | null;
}

// ============================================================================
// EnforceAI: API Key Types
// ============================================================================

export interface ApiKeySummary {
  key_id: string;
  user_id: string;
  agent_id: string;
  scopes?: string[] | null;
  expires_at?: string | null;
  revoked_at?: string | null;
  created_at: string;
  last_used_at?: string | null;
}

export interface CreateApiKeyRequest {
  scopes?: string[] | null;
  expires_at?: string | null;
}

export interface CreateApiKeyResponse {
  key_id: string;
  secret: string;
  api_key_value: string;
}

// ============================================================================
// EnforceAI: Gateway Token Types
// ============================================================================

export interface MintTokenRequest {
  scopes: string[];
  ttl_seconds?: number | null;
  expires_at?: string | null;
}

export interface MintTokenResponse {
  token: string;
}

export interface RevokeTokenRequest {
  gateway_token?: string;
  agent_id?: string;
  jti?: string;
  reason?: string;
}

export interface TokenRevocationRecord {
  jti: string;
  user_id: string;
  agent_id: string;
  revoked_at: string;
  expires_at?: string | null;
  reason?: string | null;
}

// ============================================================================
// EnforceAI: Admin Types
// ============================================================================

export interface AdminUser {
  user_id: string;
  email?: string;
  username?: string;
  auth_method: string;
  role?: string;
  last_seen_at?: string;
  created_at?: string;
}

// ============================================================================
// Scopes Types
// ============================================================================

/** Method access policy for a server */
export interface MethodPolicy {
  all_methods: boolean;
  methods: string[];
}

/** Tool access policy for a server */
export interface ToolPolicy {
  all_tools: boolean;
  tools: string[];
}

/** Permission to access a specific server */
export interface ServerPermission {
  server: string;
  methods: MethodPolicy;
  tools?: ToolPolicy | null;
}

/** Permission for agent actions */
export interface AgentActionPermission {
  action: string;
  resources: string[];
}

/** UI action permission (for scopes that control UI visibility) */
export interface UIActionPermission {
  action: string;
  resources: string[];
}

/** Complete definition of a scope */
export interface ScopeDefinition {
  name: string;
  server_permissions: ServerPermission[];
  agent_permissions: AgentActionPermission[];
}

/** Full scope catalog containing all scopes and mappings */
export interface ScopeCatalog {
  version: string;
  generated_at: string;
  etag?: string;
  last_modified?: string;
  group_mappings: Record<string, string[]>;
  scopes: Record<string, ScopeDefinition>;
}

// ============================================================================
// Scopes Management (Admin) Types
// ============================================================================

export interface ScopeMutationResponse {
  ok: boolean;
  scope_name: string;
  etag: string;
  last_modified?: string;
}

export interface MethodPolicyUpsert {
  all_methods: boolean;
  methods: string[];
}

export interface ToolPolicyUpsert {
  all_tools: boolean;
  tools: string[];
}

export interface ServerPermissionUpsert {
  server: string;
  methods: MethodPolicyUpsert;
  tools?: ToolPolicyUpsert | null;
}

export interface AgentPermissionUpsert {
  action: string;
  resources: string[];
}

export interface CreateScopeRequest {
  name: string;
  server_permissions: ServerPermissionUpsert[];
  agent_permissions: AgentPermissionUpsert[];
}

export interface ReplaceScopeRequest {
  name?: string | null;
  server_permissions: ServerPermissionUpsert[];
  agent_permissions: AgentPermissionUpsert[];
}

/** Simplified scope info for display */
export interface ScopeInfo {
  name: string;
  servers: string[];
  methods: string[];
  tools: string[];
  all_servers: boolean;
  all_methods: boolean;
  all_tools: boolean;
  agent_actions: string[];
}

// ============================================================================
// Upstream Credential Types
// ============================================================================

/** Upstream credential metadata (never includes secret) */
export interface UpstreamCredential {
  credential_id: string;
  server_path: string;
  credential_type: UpstreamAuthType;
  binding: CredentialBinding;
  user_id?: string;
  agent_id?: string;
  status: UpstreamCredentialStatus;
  expires_at?: string | null;
  created_at: string;
  updated_at: string;
  last_used_at?: string | null;
}

/** Request to create an upstream credential (API key or static JWT) */
export interface CreateUpstreamCredentialRequest {
  credential_type: 'api-key' | 'jwt';
  /** The secret value (API key or JWT token) */
  secret: string;
  /** Optional expiration date */
  expires_at?: string | null;
}

/** Response after creating an upstream credential - includes secret once */
export interface CreateUpstreamCredentialResponse {
  credential_id: string;
  server_path: string;
  credential_type: UpstreamAuthType;
  /** The secret value - only returned once at creation time */
  secret: string;
  created_at: string;
}

/** Request to revoke an upstream credential */
export interface RevokeUpstreamCredentialRequest {
  reason?: string;
}

// ============================================================================
// Upstream OAuth Credential Types
// ============================================================================

/** OAuth credential metadata (for oauth2, oidc, provider-oauth) */
export interface UpstreamOAuthCredential {
  credential_id: string;
  server_path: string;
  credential_type: 'oauth2' | 'oidc' | 'provider-oauth';
  binding: CredentialBinding;
  user_id?: string;
  agent_id?: string;
  status: UpstreamCredentialStatus;
  /** OAuth provider name (e.g., 'github', 'google', 'slack') */
  provider?: string;
  /** OAuth scopes granted */
  oauth_scopes?: string[];
  /** Token expiration time */
  token_expires_at?: string | null;
  /** Whether a refresh token is available */
  has_refresh_token?: boolean;
  expires_at?: string | null;
  created_at: string;
  updated_at: string;
  last_used_at?: string | null;
}

/** Response from starting an OAuth flow */
export interface StartOAuthFlowResponse {
  /** The authorization URL to redirect the user to */
  authorization_url: string;
  /** State parameter for CSRF protection */
  state: string;
}

/** Request to disconnect an OAuth credential */
export interface DisconnectOAuthRequest {
  /** Optional reason for disconnecting */
  reason?: string;
}

// ============================================================================
// Egress Allowlist Types (Admin)
// ============================================================================

export interface EgressAllowlistEntry {
  entry_id: string;
  pattern: string;
  description?: string;
  expires_at?: string | null;
  created_at: string;
  updated_at: string;
}

export interface CreateEgressAllowlistEntryRequest {
  pattern: string;
  description?: string;
  expires_at?: string | null;
}

export interface UpdateEgressAllowlistEntryRequest {
  pattern?: string;
  description?: string;
  expires_at?: string | null;
}

// ============================================================================
// Common Types
// ============================================================================

export interface PaginatedResponse<T> {
  items: T[];
  total: number;
  page: number;
  page_size: number;
}

export interface ApiErrorResponse {
  detail: string;
  status_code?: number;
  error_code?: string;
}
