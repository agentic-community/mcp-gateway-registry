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
  injection?: {
    kind: 'header';
    header_name: string;
    scheme?: string | null;
  } | null;
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
  upstream_auth?: UpstreamAuthConfig;
}

export interface EditServerRequest {
  name?: string;
  proxy_pass_url?: string;
  description?: string;
  tags?: string[];
  upstream_auth?: UpstreamAuthConfig;
  enabled?: boolean;
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

/** Upstream credential record returned by EnforceAI management APIs (never includes secret). */
export interface UpstreamCredential {
  credential_id: string;
  server_path: string;
  credential_type: UpstreamAuthType;
  credential_binding: CredentialBinding;
  user_id?: string | null;
  agent_id?: string | null;
  provider?: string | null;
  scopes?: string[] | null;
  token_type?: string | null;
  expires_at?: string | null;
  revoked_at?: string | null;
  created_at: string;
  updated_at: string;
  last_used_at?: string | null;
}

/** Request to create an upstream credential (secret is in `secret_payload`). */
export interface CreateUpstreamCredentialRequest {
  credential_type: UpstreamAuthType;
  credential_binding: CredentialBinding;
  agent_id?: string | null;
  provider?: string | null;
  scopes?: string[] | null;
  token_type?: string | null;
  expires_at?: string | null;
  secret_payload?: Record<string, unknown> | null;
}

/** Response after creating an upstream credential - includes `secret_payload` once. */
export interface CreateUpstreamCredentialResponse {
  credential: UpstreamCredential;
  secret_payload?: Record<string, unknown> | null;
}

/** Request to revoke an upstream credential */
export interface RevokeUpstreamCredentialRequest {
  reason?: string;
}

// ============================================================================
// Upstream OAuth Credential Types
// ============================================================================

export type UpstreamOAuthCredentialType = 'oauth2' | 'oidc' | 'provider-oauth';

/** Response from starting an OAuth flow */
export interface StartOAuthFlowResponse {
  /** The authorization URL to redirect the user to */
  authorization_url: string;
  /** State id for tracking callback */
  state_id: string;
  /** When the state expires */
  expires_at: string;
}

/** Request to disconnect an OAuth credential */
export interface DisconnectOAuthRequest {
  credential_type: UpstreamOAuthCredentialType;
  credential_binding: 'user' | 'user+agent';
  agent_id?: string | null;
  provider: string;
}

export interface DisconnectOAuthResponse {
  revoked_count: number;
}

export interface StartOAuthFlowRequest {
  credential_type: UpstreamOAuthCredentialType;
  credential_binding: 'user' | 'user+agent';
  agent_id?: string | null;
  provider: string;
  scopes?: string[] | null;
  ui_return_url: string;
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
// Upstream OAuth Provider Registry Types (Admin)
// ============================================================================

export interface UpstreamOAuthProviderRecord {
  provider_id: string;
  authorization_endpoint: string;
  token_endpoint: string;
  client_id: string;
  default_scopes: string[];
  extra_authorize_params: Record<string, string>;
  created_at: string;
  updated_at: string;
}

export interface UpstreamOAuthProviderPublic {
  provider: UpstreamOAuthProviderRecord;
  secret_present: boolean;
}

export interface UpstreamOAuthProviderCreateRequest {
  provider_id: string;
  authorization_endpoint: string;
  token_endpoint: string;
  client_id: string;
  client_secret: string;
  default_scopes?: string[];
  extra_authorize_params?: Record<string, string>;
}

export interface UpstreamOAuthProviderUpdateRequest {
  authorization_endpoint?: string;
  token_endpoint?: string;
  client_id?: string;
  client_secret?: string;
  default_scopes?: string[];
  extra_authorize_params?: Record<string, string>;
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
