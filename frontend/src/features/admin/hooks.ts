/**
 * React Query hooks for Admin user management
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  searchAdminUsers,
  getAdminUser,
  getAdminUserAgents,
  adminCreateAgentForUser,
  adminRevokeAgentForUser,
  adminRevokeAllTokensForUser,
  adminGetApiKeysForUser,
  adminCreateApiKeyForUser,
  adminRevokeApiKeyForUser,
  adminRevokeTokenForUser,
} from '@/api/enforceai';
import type {
  AdminUser,
  EnforceAIAgent,
  CreateAgentRequest,
  ApiKeySummary,
  CreateApiKeyRequest,
  CreateApiKeyResponse,
  TokenRevocationRecord,
} from '@/api/types';
import type { AdminRevokeTokenRequest } from '@/api/enforceai';

// Query keys for cache management
export const adminQueryKeys = {
  all: ['admin'] as const,
  users: (query: string) => ['admin', 'users', query] as const,
  user: (userId: string) => ['admin', 'users', userId] as const,
  userAgents: (userId: string) => ['admin', 'users', userId, 'agents'] as const,
  userAgentApiKeys: (userId: string, agentId: string) =>
    ['admin', 'users', userId, 'agents', agentId, 'api-keys'] as const,
};

// ============================================================================
// Admin Users Search Hook
// ============================================================================

export interface UseAdminUsersResult {
  users: AdminUser[];
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Search for users (admin only)
 * @param query - Search query to match against email/username
 * @param enabled - Whether to run the query (defaults to true if query is not empty)
 */
export function useAdminUsers(query: string, enabled?: boolean): UseAdminUsersResult {
  const queryResult = useQuery({
    queryKey: adminQueryKeys.users(query),
    queryFn: () => searchAdminUsers(query),
    enabled: enabled !== undefined ? enabled : query.length > 0,
    staleTime: 1000 * 30, // 30 seconds
  });

  return {
    users: queryResult.data ?? [],
    isLoading: queryResult.isLoading,
    isError: queryResult.isError,
    error: queryResult.error,
    refetch: queryResult.refetch,
  };
}

// ============================================================================
// Admin User Detail Hook
// ============================================================================

export interface UseAdminUserResult {
  user: AdminUser | null;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Fetch a single user by ID (admin only)
 * @param userId - The canonical user_id
 */
export function useAdminUser(userId: string): UseAdminUserResult {
  const queryResult = useQuery({
    queryKey: adminQueryKeys.user(userId),
    queryFn: () => getAdminUser(userId),
    enabled: Boolean(userId),
    staleTime: 1000 * 60, // 1 minute
  });

  return {
    user: queryResult.data ?? null,
    isLoading: queryResult.isLoading,
    isError: queryResult.isError,
    error: queryResult.error,
    refetch: queryResult.refetch,
  };
}

// ============================================================================
// Admin User Agents Hook
// ============================================================================

export interface UseAdminUserAgentsResult {
  agents: EnforceAIAgent[];
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Fetch all agents for a specific user (admin only)
 * @param userId - The canonical user_id
 */
export function useAdminUserAgents(userId: string): UseAdminUserAgentsResult {
  const queryResult = useQuery({
    queryKey: adminQueryKeys.userAgents(userId),
    queryFn: () => getAdminUserAgents(userId),
    enabled: Boolean(userId),
    staleTime: 1000 * 60, // 1 minute
  });

  return {
    agents: queryResult.data ?? [],
    isLoading: queryResult.isLoading,
    isError: queryResult.isError,
    error: queryResult.error,
    refetch: queryResult.refetch,
  };
}

// ============================================================================
// Admin User Agent API Keys Hook
// ============================================================================

export interface UseAdminUserAgentApiKeysResult {
  apiKeys: ApiKeySummary[];
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Fetch all API keys for a specific user's agent (admin only)
 * @param userId - The canonical user_id
 * @param agentId - The agent ID
 */
export function useAdminUserAgentApiKeys(
  userId: string,
  agentId: string
): UseAdminUserAgentApiKeysResult {
  const queryResult = useQuery({
    queryKey: adminQueryKeys.userAgentApiKeys(userId, agentId),
    queryFn: () => adminGetApiKeysForUser(userId, agentId),
    enabled: Boolean(userId) && Boolean(agentId),
    staleTime: 1000 * 60, // 1 minute
  });

  return {
    apiKeys: queryResult.data ?? [],
    isLoading: queryResult.isLoading,
    isError: queryResult.isError,
    error: queryResult.error,
    refetch: queryResult.refetch,
  };
}

// ============================================================================
// Admin Create Agent Mutation Hook
// ============================================================================

export interface UseAdminCreateAgentResult {
  createAgent: (data: CreateAgentRequest) => Promise<EnforceAIAgent>;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

/**
 * Create an agent for another user (admin only)
 * @param userId - The canonical user_id of the target user
 */
export function useAdminCreateAgent(userId: string): UseAdminCreateAgentResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: (data: CreateAgentRequest) => adminCreateAgentForUser(userId, data),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: adminQueryKeys.userAgents(userId) });
      queryClient.invalidateQueries({ queryKey: adminQueryKeys.user(userId) });
    },
  });

  return {
    createAgent: mutation.mutateAsync,
    isLoading: mutation.isPending,
    isError: mutation.isError,
    error: mutation.error,
  };
}

// ============================================================================
// Admin Revoke Agent Mutation Hook
// ============================================================================

export interface UseAdminRevokeAgentResult {
  revokeAgent: (agentId: string) => Promise<EnforceAIAgent>;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

/**
 * Revoke an agent for another user (admin only)
 * @param userId - The canonical user_id of the target user
 */
export function useAdminRevokeAgent(userId: string): UseAdminRevokeAgentResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: (agentId: string) => adminRevokeAgentForUser(userId, agentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: adminQueryKeys.userAgents(userId) });
    },
  });

  return {
    revokeAgent: mutation.mutateAsync,
    isLoading: mutation.isPending,
    isError: mutation.isError,
    error: mutation.error,
  };
}

// ============================================================================
// Admin Revoke All Tokens Mutation Hook
// ============================================================================

export interface UseAdminRevokeAllTokensResult {
  revokeAllTokens: (agentId: string) => Promise<EnforceAIAgent>;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

/**
 * Revoke all tokens for a user's agent (admin only)
 * @param userId - The canonical user_id of the target user
 */
export function useAdminRevokeAllTokens(userId: string): UseAdminRevokeAllTokensResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: (agentId: string) => adminRevokeAllTokensForUser(userId, agentId),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: adminQueryKeys.userAgents(userId) });
    },
  });

  return {
    revokeAllTokens: mutation.mutateAsync,
    isLoading: mutation.isPending,
    isError: mutation.isError,
    error: mutation.error,
  };
}

// ============================================================================
// Admin Create API Key Mutation Hook
// ============================================================================

export interface UseAdminCreateApiKeyResult {
  createApiKey: (agentId: string, data: CreateApiKeyRequest) => Promise<CreateApiKeyResponse>;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

/**
 * Create an API key for another user's agent (admin only)
 * @param userId - The canonical user_id of the target user
 */
export function useAdminCreateApiKey(userId: string): UseAdminCreateApiKeyResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: ({ agentId, data }: { agentId: string; data: CreateApiKeyRequest }) =>
      adminCreateApiKeyForUser(userId, agentId, data),
    onSuccess: (_, variables) => {
      queryClient.invalidateQueries({
        queryKey: adminQueryKeys.userAgentApiKeys(userId, variables.agentId),
      });
    },
  });

  return {
    createApiKey: (agentId: string, data: CreateApiKeyRequest) =>
      mutation.mutateAsync({ agentId, data }),
    isLoading: mutation.isPending,
    isError: mutation.isError,
    error: mutation.error,
  };
}

// ============================================================================
// Admin Revoke API Key Mutation Hook
// ============================================================================

export interface UseAdminRevokeApiKeyResult {
  revokeApiKey: (keyId: string) => Promise<ApiKeySummary>;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

/**
 * Revoke an API key for another user (admin only)
 * @param userId - The canonical user_id of the target user
 */
export function useAdminRevokeApiKey(userId: string): UseAdminRevokeApiKeyResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: (keyId: string) => adminRevokeApiKeyForUser(userId, keyId),
    onSuccess: () => {
      // Invalidate all API keys queries for this user
      queryClient.invalidateQueries({
        queryKey: ['admin', 'users', userId],
        predicate: (query) => query.queryKey.includes('api-keys'),
      });
    },
  });

  return {
    revokeApiKey: mutation.mutateAsync,
    isLoading: mutation.isPending,
    isError: mutation.isError,
    error: mutation.error,
  };
}

// ============================================================================
// Admin Revoke Token Mutation Hook
// ============================================================================

export interface UseAdminRevokeTokenResult {
  revokeToken: (data: AdminRevokeTokenRequest) => Promise<TokenRevocationRecord>;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
}

/**
 * Revoke a gateway token for another user (admin only)
 * @param userId - The canonical user_id of the target user
 */
export function useAdminRevokeToken(userId: string): UseAdminRevokeTokenResult {
  const mutation = useMutation({
    mutationFn: (data: AdminRevokeTokenRequest) => adminRevokeTokenForUser(userId, data),
  });

  return {
    revokeToken: mutation.mutateAsync,
    isLoading: mutation.isPending,
    isError: mutation.isError,
    error: mutation.error,
  };
}
