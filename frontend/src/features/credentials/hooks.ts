/**
 * React Query hooks for credentials management (API Keys + Gateway Tokens)
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  getApiKeys,
  createApiKey,
  revokeApiKey,
  mintGatewayToken,
  revokeGatewayToken,
} from '@/api/enforceai';
import type {
  ApiKeySummary,
  CreateApiKeyRequest,
  CreateApiKeyResponse,
  MintTokenRequest,
  MintTokenResponse,
  RevokeTokenRequest,
  TokenRevocationRecord,
} from '@/api/types';

// ============================================================================
// Query Keys
// ============================================================================

export const credentialsQueryKeys = {
  apiKeys: (agentId: string) => ['api-keys', agentId] as const,
  allApiKeys: ['api-keys'] as const,
};

// ============================================================================
// API Keys Hooks
// ============================================================================

interface UseApiKeysResult {
  apiKeys: ApiKeySummary[];
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
}

export function useApiKeys(agentId: string): UseApiKeysResult {
  const query = useQuery({
    queryKey: credentialsQueryKeys.apiKeys(agentId),
    queryFn: () => getApiKeys(agentId),
    enabled: !!agentId,
  });

  return {
    apiKeys: query.data ?? [],
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
  };
}

interface UseCreateApiKeyResult {
  createApiKey: (
    agentId: string,
    data: CreateApiKeyRequest
  ) => Promise<CreateApiKeyResponse>;
  isCreating: boolean;
  error: Error | null;
}

export function useCreateApiKey(): UseCreateApiKeyResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: ({
      agentId,
      data,
    }: {
      agentId: string;
      data: CreateApiKeyRequest;
    }) => createApiKey(agentId, data),
    onSettled: (_data, _error, variables) => {
      queryClient.invalidateQueries({
        queryKey: credentialsQueryKeys.apiKeys(variables.agentId),
      });
    },
  });

  return {
    createApiKey: (agentId: string, data: CreateApiKeyRequest) =>
      mutation.mutateAsync({ agentId, data }),
    isCreating: mutation.isPending,
    error: mutation.error,
  };
}

interface UseRevokeApiKeyResult {
  revokeApiKey: (keyId: string, agentId: string) => Promise<void>;
  isRevoking: boolean;
  error: Error | null;
}

export function useRevokeApiKey(): UseRevokeApiKeyResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: ({ keyId }: { keyId: string; agentId: string }) =>
      revokeApiKey(keyId),
    onSuccess: (_data, variables) => {
      // Update cache to mark as revoked
      queryClient.setQueryData<ApiKeySummary[]>(
        credentialsQueryKeys.apiKeys(variables.agentId),
        (old) =>
          old?.map((key) =>
            key.key_id === variables.keyId
              ? { ...key, revoked_at: new Date().toISOString() }
              : key
          )
      );
    },
    onSettled: (_data, _error, variables) => {
      queryClient.invalidateQueries({
        queryKey: credentialsQueryKeys.apiKeys(variables.agentId),
      });
    },
  });

  return {
    revokeApiKey: (keyId: string, agentId: string) =>
      mutation.mutateAsync({ keyId, agentId }),
    isRevoking: mutation.isPending,
    error: mutation.error,
  };
}

// ============================================================================
// Gateway Token Hooks
// ============================================================================

interface UseMintTokenResult {
  mintToken: (
    agentId: string,
    data: MintTokenRequest
  ) => Promise<MintTokenResponse>;
  isMinting: boolean;
  error: Error | null;
}

export function useMintToken(): UseMintTokenResult {
  const mutation = useMutation({
    mutationFn: ({
      agentId,
      data,
    }: {
      agentId: string;
      data: MintTokenRequest;
    }) => mintGatewayToken(agentId, data),
  });

  return {
    mintToken: (agentId: string, data: MintTokenRequest) =>
      mutation.mutateAsync({ agentId, data }),
    isMinting: mutation.isPending,
    error: mutation.error,
  };
}

interface UseRevokeTokenResult {
  revokeToken: (data: RevokeTokenRequest) => Promise<TokenRevocationRecord>;
  isRevoking: boolean;
  error: Error | null;
}

export function useRevokeToken(): UseRevokeTokenResult {
  const mutation = useMutation({
    mutationFn: revokeGatewayToken,
  });

  return {
    revokeToken: mutation.mutateAsync,
    isRevoking: mutation.isPending,
    error: mutation.error,
  };
}

// ============================================================================
// Token Decoding Utilities
// ============================================================================

export interface DecodedToken {
  header: {
    alg: string;
    typ: string;
    kid?: string;
  };
  payload: {
    iss: string;
    sub: string;
    aud: string[];
    exp: number;
    iat: number;
    jti: string;
    type: string;
    name?: string;
    scopes: string[];
  };
  signature: string;
}

/**
 * Decode a JWT token locally (no validation, just base64 decode)
 */
export function decodeToken(token: string): DecodedToken | null {
  try {
    const parts = token.split('.');
    if (parts.length !== 3) {
      return null;
    }

    const decodeBase64 = (str: string): string => {
      // Handle URL-safe base64
      const base64 = str.replace(/-/g, '+').replace(/_/g, '/');
      const padded = base64 + '='.repeat((4 - (base64.length % 4)) % 4);
      return atob(padded);
    };

    const header = JSON.parse(decodeBase64(parts[0]));
    const payload = JSON.parse(decodeBase64(parts[1]));

    return {
      header,
      payload,
      signature: parts[2],
    };
  } catch {
    return null;
  }
}

/**
 * Check if a token is expired
 */
export function isTokenExpired(token: string): boolean {
  const decoded = decodeToken(token);
  if (!decoded) return true;
  return decoded.payload.exp * 1000 < Date.now();
}

/**
 * Get token expiration date
 */
export function getTokenExpiration(token: string): Date | null {
  const decoded = decodeToken(token);
  if (!decoded) return null;
  return new Date(decoded.payload.exp * 1000);
}
