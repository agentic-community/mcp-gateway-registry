/**
 * React Query hooks for the Overview page
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { getServers } from '@/api/registry';
import { getA2AAgents } from '@/api/registry';
import { getEnforceAIAgents, testEnforceAIConnection } from '@/api/enforceai';
import type { Server, A2AAgent, EnforceAIAgent } from '@/api/types';

// Query keys for cache management
export const overviewQueryKeys = {
  servers: ['servers'] as const,
  a2aAgents: ['a2a-agents'] as const,
  enforceaiAgents: ['enforceai-agents'] as const,
  connectionTest: ['connection-test'] as const,
};

// ============================================================================
// Server Hooks
// ============================================================================

export interface UseServersResult {
  servers: Server[];
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Fetch all MCP servers with summary stats
 */
export function useServers(): UseServersResult {
  const query = useQuery({
    queryKey: overviewQueryKeys.servers,
    queryFn: getServers,
    staleTime: 1000 * 30, // 30 seconds
  });

  return {
    servers: query.data ?? [],
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
  };
}

export interface ServerStats {
  total: number;
  enabled: number;
  disabled: number;
}

/**
 * Calculate server statistics
 */
export function useServerStats(): ServerStats & { isLoading: boolean } {
  const { servers, isLoading } = useServers();

  const enabled = servers.filter((s) => s.is_enabled).length;
  const disabled = servers.length - enabled;

  return {
    total: servers.length,
    enabled,
    disabled,
    isLoading,
  };
}

// ============================================================================
// A2A Agent Hooks
// ============================================================================

export interface UseA2AAgentsResult {
  agents: A2AAgent[];
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Fetch all A2A agents
 */
export function useA2AAgents(): UseA2AAgentsResult {
  const query = useQuery({
    queryKey: overviewQueryKeys.a2aAgents,
    queryFn: getA2AAgents,
    staleTime: 1000 * 30, // 30 seconds
  });

  return {
    agents: query.data ?? [],
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
  };
}

export interface A2AAgentStats {
  total: number;
  enabled: number;
  disabled: number;
}

/**
 * Calculate A2A agent statistics
 */
export function useA2AAgentStats(): A2AAgentStats & { isLoading: boolean } {
  const { agents, isLoading } = useA2AAgents();

  const enabled = agents.filter((a) => a.is_enabled).length;
  const disabled = agents.length - enabled;

  return {
    total: agents.length,
    enabled,
    disabled,
    isLoading,
  };
}

// ============================================================================
// EnforceAI Agent Hooks
// ============================================================================

export interface UseEnforceAIAgentsResult {
  agents: EnforceAIAgent[];
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  errorCode?: number;
  refetch: () => void;
}

/**
 * Fetch all EnforceAI agents
 * Handles 404 gracefully (EnforceAI not enabled)
 */
export function useEnforceAIAgents(): UseEnforceAIAgentsResult {
  const query = useQuery({
    queryKey: overviewQueryKeys.enforceaiAgents,
    queryFn: getEnforceAIAgents,
    staleTime: 1000 * 30, // 30 seconds
    retry: (failureCount, error) => {
      // Don't retry on 404 (EnforceAI not enabled)
      if (error && 'status' in error && error.status === 404) {
        return false;
      }
      return failureCount < 2;
    },
  });

  // Extract status code from error
  const errorCode =
    query.error && 'status' in query.error
      ? (query.error.status as number)
      : undefined;

  return {
    agents: query.data ?? [],
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    errorCode,
    refetch: query.refetch,
  };
}

export interface EnforceAIAgentStats {
  total: number;
  active: number;
  revoked: number;
  enforceAIEnabled: boolean;
}

/**
 * Calculate EnforceAI agent statistics
 */
export function useEnforceAIAgentStats(): EnforceAIAgentStats & {
  isLoading: boolean;
  isError: boolean;
} {
  const { agents, isLoading, isError, errorCode } = useEnforceAIAgents();

  // EnforceAI is not enabled if we get 404
  const enforceAIEnabled = !(isError && errorCode === 404);

  const active = agents.filter((a) => !a.revoked_at).length;
  const revoked = agents.length - active;

  return {
    total: agents.length,
    active,
    revoked,
    enforceAIEnabled,
    isLoading,
    isError: isError && errorCode !== 404, // Don't treat 404 as error
  };
}

// ============================================================================
// Connection Test Hook
// ============================================================================

export interface ConnectionStatus {
  registry: {
    reachable: boolean;
    elapsed_ms: number;
    tested: boolean;
  };
  enforceai: {
    reachable: boolean;
    elapsed_ms: number;
    tested: boolean;
    enabled: boolean;
  };
}

interface ConnectionTestResult {
  status: ConnectionStatus;
  isTesting: boolean;
  testConnections: () => void;
}

/**
 * Test connectivity to Registry and EnforceAI
 */
export function useConnectionTest(): ConnectionTestResult {
  const queryClient = useQueryClient();

  const testMutation = useMutation({
    mutationFn: async (): Promise<ConnectionStatus> => {
      // Test registry
      const registryStart = Date.now();
      let registryReachable = false;
      try {
        await getServers();
        registryReachable = true;
      } catch {
        registryReachable = false;
      }
      const registryElapsed = Date.now() - registryStart;

      // Test EnforceAI
      const enforceaiResult = await testEnforceAIConnection();

      return {
        registry: {
          reachable: registryReachable,
          elapsed_ms: registryElapsed,
          tested: true,
        },
        enforceai: {
          reachable: enforceaiResult.reachable,
          elapsed_ms: enforceaiResult.elapsed_ms,
          tested: true,
          enabled: enforceaiResult.reachable,
        },
      };
    },
    onSuccess: () => {
      // Invalidate queries to refresh data after connection test
      queryClient.invalidateQueries({ queryKey: overviewQueryKeys.servers });
      queryClient.invalidateQueries({ queryKey: overviewQueryKeys.a2aAgents });
      queryClient.invalidateQueries({ queryKey: overviewQueryKeys.enforceaiAgents });
    },
  });

  return {
    status: testMutation.data ?? {
      registry: { reachable: false, elapsed_ms: 0, tested: false },
      enforceai: { reachable: false, elapsed_ms: 0, tested: false, enabled: true },
    },
    isTesting: testMutation.isPending,
    testConnections: () => testMutation.mutate(),
  };
}
