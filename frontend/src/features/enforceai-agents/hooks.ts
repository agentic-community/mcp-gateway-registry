/**
 * React Query hooks for EnforceAI agent management
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  getEnforceAIAgents,
  getEnforceAIAgent,
  createEnforceAIAgent,
  updateEnforceAIAgent,
  revokeEnforceAIAgent,
  revokeAllAgentTokens,
} from '@/api/enforceai';
import type {
  EnforceAIAgent,
  CreateAgentRequest,
  UpdateAgentRequest,
} from '@/api/types';

// ============================================================================
// Query Keys
// ============================================================================

export const enforceAIAgentQueryKeys = {
  all: ['enforceai-agents'] as const,
  detail: (agentId: string) => ['enforceai-agents', agentId] as const,
};

// ============================================================================
// Filter Types
// ============================================================================

export type AgentStatusFilter = 'all' | 'active' | 'revoked';

export interface EnforceAIAgentFilter {
  search: string;
  status: AgentStatusFilter;
}

// ============================================================================
// Filter Function
// ============================================================================

export function filterEnforceAIAgents(
  agents: EnforceAIAgent[],
  filter: EnforceAIAgentFilter
): EnforceAIAgent[] {
  return agents.filter((agent) => {
    // Search filter (alias, agent_id)
    if (filter.search) {
      const searchLower = filter.search.toLowerCase();
      const matchesAlias = agent.alias?.toLowerCase().includes(searchLower);
      const matchesId = agent.agent_id.toLowerCase().includes(searchLower);
      const matchesScope = agent.scopes.some((s) =>
        s.toLowerCase().includes(searchLower)
      );
      if (!matchesAlias && !matchesId && !matchesScope) {
        return false;
      }
    }

    // Status filter
    if (filter.status !== 'all') {
      const isRevoked = agent.revoked_at !== null;
      if (filter.status === 'active' && isRevoked) return false;
      if (filter.status === 'revoked' && !isRevoked) return false;
    }

    return true;
  });
}

// ============================================================================
// Hooks: Query
// ============================================================================

interface UseEnforceAIAgentsResult {
  agents: EnforceAIAgent[];
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
}

export function useEnforceAIAgents(): UseEnforceAIAgentsResult {
  const query = useQuery({
    queryKey: enforceAIAgentQueryKeys.all,
    queryFn: getEnforceAIAgents,
  });

  return {
    agents: query.data ?? [],
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
  };
}

interface UseEnforceAIAgentResult {
  agent: EnforceAIAgent | null;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
}

export function useEnforceAIAgent(agentId: string): UseEnforceAIAgentResult {
  const query = useQuery({
    queryKey: enforceAIAgentQueryKeys.detail(agentId),
    queryFn: () => getEnforceAIAgent(agentId),
    enabled: !!agentId,
  });

  return {
    agent: query.data ?? null,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
  };
}

// ============================================================================
// Hooks: Mutations
// ============================================================================

interface UseCreateEnforceAIAgentResult {
  createAgent: (data: CreateAgentRequest) => Promise<EnforceAIAgent>;
  isCreating: boolean;
  error: Error | null;
}

export function useCreateEnforceAIAgent(): UseCreateEnforceAIAgentResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: createEnforceAIAgent,
    onSuccess: (newAgent) => {
      // Add to cache
      queryClient.setQueryData<EnforceAIAgent[]>(
        enforceAIAgentQueryKeys.all,
        (old) => [...(old ?? []), newAgent]
      );
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: enforceAIAgentQueryKeys.all });
    },
  });

  return {
    createAgent: mutation.mutateAsync,
    isCreating: mutation.isPending,
    error: mutation.error,
  };
}

interface UseUpdateEnforceAIAgentResult {
  updateAgent: (
    agentId: string,
    data: UpdateAgentRequest
  ) => Promise<EnforceAIAgent>;
  isUpdating: boolean;
  error: Error | null;
}

export function useUpdateEnforceAIAgent(): UseUpdateEnforceAIAgentResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: ({
      agentId,
      data,
    }: {
      agentId: string;
      data: UpdateAgentRequest;
    }) => updateEnforceAIAgent(agentId, data),
    onSuccess: (updatedAgent) => {
      // Update cache
      queryClient.setQueryData<EnforceAIAgent[]>(
        enforceAIAgentQueryKeys.all,
        (old) =>
          old?.map((a) =>
            a.agent_id === updatedAgent.agent_id ? updatedAgent : a
          )
      );
      queryClient.setQueryData<EnforceAIAgent>(
        enforceAIAgentQueryKeys.detail(updatedAgent.agent_id),
        updatedAgent
      );
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: enforceAIAgentQueryKeys.all });
    },
  });

  return {
    updateAgent: (agentId: string, data: UpdateAgentRequest) =>
      mutation.mutateAsync({ agentId, data }),
    isUpdating: mutation.isPending,
    error: mutation.error,
  };
}

interface UseRevokeEnforceAIAgentResult {
  revokeAgent: (agentId: string) => Promise<EnforceAIAgent>;
  isRevoking: boolean;
  error: Error | null;
}

export function useRevokeEnforceAIAgent(): UseRevokeEnforceAIAgentResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: revokeEnforceAIAgent,
    onSuccess: (revokedAgent) => {
      // Update cache
      queryClient.setQueryData<EnforceAIAgent[]>(
        enforceAIAgentQueryKeys.all,
        (old) =>
          old?.map((a) =>
            a.agent_id === revokedAgent.agent_id ? revokedAgent : a
          )
      );
      queryClient.setQueryData<EnforceAIAgent>(
        enforceAIAgentQueryKeys.detail(revokedAgent.agent_id),
        revokedAgent
      );
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: enforceAIAgentQueryKeys.all });
    },
  });

  return {
    revokeAgent: mutation.mutateAsync,
    isRevoking: mutation.isPending,
    error: mutation.error,
  };
}

interface UseRevokeAllTokensResult {
  revokeAllTokens: (agentId: string) => Promise<EnforceAIAgent>;
  isRevoking: boolean;
  error: Error | null;
}

export function useRevokeAllTokens(): UseRevokeAllTokensResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: revokeAllAgentTokens,
    onSuccess: (updatedAgent) => {
      // Update cache
      queryClient.setQueryData<EnforceAIAgent[]>(
        enforceAIAgentQueryKeys.all,
        (old) =>
          old?.map((a) =>
            a.agent_id === updatedAgent.agent_id ? updatedAgent : a
          )
      );
      queryClient.setQueryData<EnforceAIAgent>(
        enforceAIAgentQueryKeys.detail(updatedAgent.agent_id),
        updatedAgent
      );
    },
    onSettled: () => {
      queryClient.invalidateQueries({ queryKey: enforceAIAgentQueryKeys.all });
    },
  });

  return {
    revokeAllTokens: mutation.mutateAsync,
    isRevoking: mutation.isPending,
    error: mutation.error,
  };
}
