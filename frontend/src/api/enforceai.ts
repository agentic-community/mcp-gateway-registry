/**
 * EnforceAI API functions
 */

import { apiGet, apiPost, apiPut, apiDelete } from './client';
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
// Connection Test
// ============================================================================

/**
 * Test EnforceAI connectivity
 * Returns timing and status information
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
  } catch {
    return {
      reachable: false,
      elapsed_ms: Date.now() - start,
    };
  }
}
