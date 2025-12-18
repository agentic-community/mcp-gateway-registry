/**
 * EnforceAI API functions
 */

import { apiGet, apiPost, apiPut, apiDelete, type ApiError } from './client';
import type {
  EnforceAIAgent,
  CreateAgentRequest,
  UpdateAgentRequest,
  ApiKeySummary,
  CreateApiKeyRequest,
  CreateApiKeyResponse,
  MintTokenRequest,
  MintTokenResponse,
  RevokeTokenRequest,
  TokenRevocationRecord,
  AdminUser,
  CreateScopeRequest,
  ReplaceScopeRequest,
  ScopeMutationResponse,
  UpstreamCredential,
  CreateUpstreamCredentialRequest,
  CreateUpstreamCredentialResponse,
  UpstreamOAuthCredential,
  StartOAuthFlowResponse,
} from './types';

// ============================================================================
// EnforceAI Agents API
// ============================================================================

/**
 * Fetch all EnforceAI agents for the current user
 */
export async function getEnforceAIAgents(): Promise<EnforceAIAgent[]> {
  return apiGet<EnforceAIAgent[]>('/enforceai/agents');
}

/**
 * Fetch a single EnforceAI agent by ID
 */
export async function getEnforceAIAgent(agentId: string): Promise<EnforceAIAgent> {
  return apiGet<EnforceAIAgent>(`/enforceai/agents/${encodeURIComponent(agentId)}`);
}

/**
 * Create a new EnforceAI agent
 */
export async function createEnforceAIAgent(
  data: CreateAgentRequest
): Promise<EnforceAIAgent> {
  return apiPost<EnforceAIAgent>('/enforceai/agents', data);
}

/**
 * Update an existing EnforceAI agent
 */
export async function updateEnforceAIAgent(
  agentId: string,
  data: UpdateAgentRequest
): Promise<EnforceAIAgent> {
  return apiPut<EnforceAIAgent>(
    `/enforceai/agents/${encodeURIComponent(agentId)}`,
    data
  );
}

/**
 * Revoke an EnforceAI agent
 */
export async function revokeEnforceAIAgent(agentId: string): Promise<EnforceAIAgent> {
  return apiPost<EnforceAIAgent>(
    `/enforceai/agents/${encodeURIComponent(agentId)}/revoke`
  );
}

/**
 * Revoke all tokens for an agent (sets tokens_valid_after)
 */
export async function revokeAllAgentTokens(agentId: string): Promise<EnforceAIAgent> {
  return apiPost<EnforceAIAgent>(
    `/enforceai/agents/${encodeURIComponent(agentId)}/revoke-all-tokens`
  );
}

// ============================================================================
// EnforceAI API Keys API
// ============================================================================

/**
 * Fetch all API keys for an agent
 */
export async function getApiKeys(agentId: string): Promise<ApiKeySummary[]> {
  return apiGet<ApiKeySummary[]>(
    `/enforceai/agents/${encodeURIComponent(agentId)}/api-keys`
  );
}

/**
 * Create a new API key for an agent
 */
export async function createApiKey(
  agentId: string,
  data: CreateApiKeyRequest
): Promise<CreateApiKeyResponse> {
  return apiPost<CreateApiKeyResponse>(
    `/enforceai/agents/${encodeURIComponent(agentId)}/api-keys`,
    data
  );
}

/**
 * Revoke an API key
 */
export async function revokeApiKey(keyId: string): Promise<void> {
  return apiPost<void>(`/enforceai/api-keys/${encodeURIComponent(keyId)}/revoke`);
}

// ============================================================================
// EnforceAI Gateway Tokens API
// ============================================================================

/**
 * Mint a new gateway token for an agent
 */
export async function mintGatewayToken(
  agentId: string,
  data: MintTokenRequest
): Promise<MintTokenResponse> {
  return apiPost<MintTokenResponse>(
    `/enforceai/agents/${encodeURIComponent(agentId)}/tokens/mint`,
    data
  );
}

/**
 * Revoke a gateway token
 */
export async function revokeGatewayToken(
  data: RevokeTokenRequest
): Promise<TokenRevocationRecord> {
  return apiPost<TokenRevocationRecord>('/enforceai/tokens/revoke', data);
}

// ============================================================================
// Admin API
// ============================================================================

/**
 * Search for users (admin only)
 * @param query - Search query to match against email/username
 */
export async function searchAdminUsers(query: string): Promise<AdminUser[]> {
  return apiGet<AdminUser[]>(`/enforceai/admin/users?query=${encodeURIComponent(query)}`);
}

/**
 * Get a single user by ID (admin only)
 * @param userId - The canonical user_id
 */
export async function getAdminUser(userId: string): Promise<AdminUser> {
  return apiGet<AdminUser>(`/enforceai/admin/users/${encodeURIComponent(userId)}`);
}

/**
 * Get agents for a specific user (admin only)
 * @param userId - The canonical user_id
 */
export async function getAdminUserAgents(userId: string): Promise<EnforceAIAgent[]> {
  return apiGet<EnforceAIAgent[]>(
    `/enforceai/admin/users/${encodeURIComponent(userId)}/agents`
  );
}

// ============================================================================
// Admin Cross-User Operations API
// ============================================================================

/**
 * Create an agent for another user (admin only)
 */
export async function adminCreateAgentForUser(
  userId: string,
  data: CreateAgentRequest
): Promise<EnforceAIAgent> {
  return apiPost<EnforceAIAgent>(
    `/enforceai/admin/users/${encodeURIComponent(userId)}/agents`,
    data
  );
}

/**
 * Revoke an agent for another user (admin only)
 */
export async function adminRevokeAgentForUser(
  userId: string,
  agentId: string
): Promise<EnforceAIAgent> {
  return apiPost<EnforceAIAgent>(
    `/enforceai/admin/users/${encodeURIComponent(userId)}/agents/${encodeURIComponent(agentId)}/revoke`
  );
}

/**
 * Revoke all tokens for a user's agent (admin only)
 */
export async function adminRevokeAllTokensForUser(
  userId: string,
  agentId: string
): Promise<EnforceAIAgent> {
  return apiPost<EnforceAIAgent>(
    `/enforceai/admin/users/${encodeURIComponent(userId)}/agents/${encodeURIComponent(agentId)}/tokens/revoke-all`
  );
}

/**
 * Get API keys for a user's agent (admin only)
 */
export async function adminGetApiKeysForUser(
  userId: string,
  agentId: string
): Promise<ApiKeySummary[]> {
  return apiGet<ApiKeySummary[]>(
    `/enforceai/admin/users/${encodeURIComponent(userId)}/agents/${encodeURIComponent(agentId)}/api-keys`
  );
}

/**
 * Create an API key for another user's agent (admin only)
 */
export async function adminCreateApiKeyForUser(
  userId: string,
  agentId: string,
  data: CreateApiKeyRequest
): Promise<CreateApiKeyResponse> {
  return apiPost<CreateApiKeyResponse>(
    `/enforceai/admin/users/${encodeURIComponent(userId)}/agents/${encodeURIComponent(agentId)}/api-keys`,
    data
  );
}

/**
 * Revoke an API key for another user (admin only)
 */
export async function adminRevokeApiKeyForUser(
  userId: string,
  keyId: string
): Promise<ApiKeySummary> {
  return apiPost<ApiKeySummary>(
    `/enforceai/admin/users/${encodeURIComponent(userId)}/api-keys/${encodeURIComponent(keyId)}/revoke`
  );
}

export interface AdminRevokeTokenRequest {
  agent_id: string;
  jti: string;
  reason?: string;
}

/**
 * Revoke a gateway token for another user (admin only)
 */
export async function adminRevokeTokenForUser(
  userId: string,
  data: AdminRevokeTokenRequest
): Promise<TokenRevocationRecord> {
  return apiPost<TokenRevocationRecord>(
    `/enforceai/admin/users/${encodeURIComponent(userId)}/tokens/revoke`,
    data
  );
}

// ============================================================================
// Connection Test
// ============================================================================

/**
 * Test EnforceAI connectivity
 * Returns timing and status information
 *
 * A service is considered "reachable" if it responds with any HTTP status code,
 * including 401/403 (which indicate the service is up but auth is required).
 * Only network errors or 5xx status codes indicate the service is unreachable.
 */
export async function testEnforceAIConnection(): Promise<{
  reachable: boolean;
  elapsed_ms: number;
}> {
  const start = Date.now();
  try {
    await apiGet<EnforceAIAgent[]>('/enforceai/agents');
    return {
      reachable: true,
      elapsed_ms: Date.now() - start,
    };
  } catch (error) {
    const elapsed_ms = Date.now() - start;

    // The error is normalized to ApiError by the client interceptor
    const apiError = error as ApiError;

    // If it's a network error or timeout, the service is unreachable
    if (apiError.isNetworkError || apiError.isTimeout) {
      return {
        reachable: false,
        elapsed_ms,
      };
    }

    // If we got an HTTP status code, check if it's a server error (5xx)
    // Any 1xx-4xx response means the service IS reachable (just auth/permission issues)
    if (apiError.status > 0 && apiError.status < 500) {
      return {
        reachable: true,
        elapsed_ms,
      };
    }

    // 5xx or unknown errors mean unreachable
    return {
      reachable: false,
      elapsed_ms,
    };
  }
}

// ============================================================================
// Scopes Management (Admin)
// ============================================================================

export async function adminCreateScope(
  data: CreateScopeRequest,
  ifMatch?: string
): Promise<ScopeMutationResponse> {
  return apiPost<ScopeMutationResponse>(
    '/enforceai/admin/scopes',
    data,
    ifMatch ? { headers: { 'If-Match': ifMatch } } : undefined
  );
}

export async function adminReplaceScope(
  scopeName: string,
  data: ReplaceScopeRequest,
  ifMatch: string
): Promise<ScopeMutationResponse> {
  return apiPut<ScopeMutationResponse>(
    `/enforceai/admin/scopes/${encodeURIComponent(scopeName)}`,
    data,
    { headers: { 'If-Match': ifMatch } }
  );
}

export async function adminDeleteScope(
  scopeName: string,
  ifMatch: string
): Promise<ScopeMutationResponse> {
  return apiDelete<ScopeMutationResponse>(
    `/enforceai/admin/scopes/${encodeURIComponent(scopeName)}`,
    { headers: { 'If-Match': ifMatch } }
  );
}

// ============================================================================
// Upstream Credentials API
// ============================================================================

/**
 * Get upstream credential for a server
 * Returns metadata only (never includes secret)
 */
export async function getUpstreamCredential(
  serverPath: string
): Promise<UpstreamCredential | null> {
  try {
    return await apiGet<UpstreamCredential>(
      `/enforceai/upstream/servers/${encodeURIComponent(serverPath)}/credentials`
    );
  } catch (error) {
    const apiError = error as ApiError;
    // 404 means no credential configured - return null
    if (apiError.status === 404) {
      return null;
    }
    throw error;
  }
}

/**
 * Create or replace an upstream credential for a server
 * Returns the credential with secret (only returned once at creation)
 */
export async function createUpstreamCredential(
  serverPath: string,
  data: CreateUpstreamCredentialRequest
): Promise<CreateUpstreamCredentialResponse> {
  return apiPost<CreateUpstreamCredentialResponse>(
    `/enforceai/upstream/servers/${encodeURIComponent(serverPath)}/credentials`,
    data
  );
}

/**
 * Revoke an upstream credential
 */
export async function revokeUpstreamCredential(
  credentialId: string,
  reason?: string
): Promise<void> {
  return apiPost<void>(
    `/enforceai/upstream/credentials/${encodeURIComponent(credentialId)}/revoke`,
    reason ? { reason } : {}
  );
}

// ============================================================================
// Upstream OAuth Credentials API
// ============================================================================

/**
 * Get upstream OAuth credential for a server
 * Returns OAuth-specific metadata (provider, scopes, expiry)
 */
export async function getUpstreamOAuthCredential(
  serverPath: string
): Promise<UpstreamOAuthCredential | null> {
  try {
    return await apiGet<UpstreamOAuthCredential>(
      `/enforceai/upstream/servers/${encodeURIComponent(serverPath)}/credentials`
    );
  } catch (error) {
    const apiError = error as ApiError;
    // 404 means no credential configured - return null
    if (apiError.status === 404) {
      return null;
    }
    throw error;
  }
}

/**
 * Start an OAuth flow for upstream authentication
 * Returns the authorization URL to redirect the user to
 */
export async function startUpstreamOAuth(
  serverPath: string,
  returnUrl?: string
): Promise<StartOAuthFlowResponse> {
  return apiPost<StartOAuthFlowResponse>(
    `/enforceai/upstream/servers/${encodeURIComponent(serverPath)}/oauth/start`,
    { return_url: returnUrl }
  );
}

/**
 * Disconnect an OAuth credential (revoke and remove stored tokens)
 */
export async function disconnectUpstreamOAuth(
  serverPath: string,
  reason?: string
): Promise<void> {
  return apiPost<void>(
    `/enforceai/upstream/servers/${encodeURIComponent(serverPath)}/oauth/disconnect`,
    reason ? { reason } : {}
  );
}
