/**
 * React Query hooks for A2A Agent management
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  getA2AAgents,
  getA2AAgent,
  registerA2AAgent,
  updateA2AAgent,
  deleteA2AAgent,
  toggleA2AAgent,
} from '@/api/registry';
import type { A2AAgent, RegisterA2AAgentRequest } from '@/api/types';

// Query keys for cache management
export const agentQueryKeys = {
  all: ['a2a-agents'] as const,
  detail: (path: string) => ['a2a-agents', path] as const,
};

// ============================================================================
// Agent List Hook
// ============================================================================

export interface UseAgentsResult {
  agents: A2AAgent[];
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Fetch all A2A agents
 */
export function useAgents(): UseAgentsResult {
  const query = useQuery({
    queryKey: agentQueryKeys.all,
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

// ============================================================================
// Agent Detail Hook
// ============================================================================

export interface UseAgentResult {
  agent: A2AAgent | null;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Fetch a single agent by path
 */
export function useAgent(path: string): UseAgentResult {
  const query = useQuery({
    queryKey: agentQueryKeys.detail(path),
    queryFn: () => getA2AAgent(path),
    enabled: Boolean(path),
    staleTime: 1000 * 30, // 30 seconds
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
// Agent Registration Mutation
// ============================================================================

export interface UseRegisterAgentResult {
  registerAgent: (data: RegisterA2AAgentRequest) => Promise<A2AAgent>;
  isRegistering: boolean;
  error: Error | null;
  reset: () => void;
}

/**
 * Register a new A2A agent
 */
export function useRegisterAgent(): UseRegisterAgentResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: registerA2AAgent,
    onSuccess: () => {
      // Invalidate the agents list to refetch
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.all });
    },
  });

  return {
    registerAgent: mutation.mutateAsync,
    isRegistering: mutation.isPending,
    error: mutation.error,
    reset: mutation.reset,
  };
}

// ============================================================================
// Agent Update Mutation
// ============================================================================

export interface UseUpdateAgentResult {
  updateAgent: (path: string, data: Partial<RegisterA2AAgentRequest>) => Promise<A2AAgent>;
  isUpdating: boolean;
  error: Error | null;
  reset: () => void;
}

/**
 * Update an existing A2A agent
 */
export function useUpdateAgent(): UseUpdateAgentResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: ({ path, data }: { path: string; data: Partial<RegisterA2AAgentRequest> }) =>
      updateA2AAgent(path, data),
    onSuccess: (_, variables) => {
      // Invalidate both the list and the specific agent
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.all });
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.detail(variables.path) });
    },
  });

  return {
    updateAgent: (path: string, data: Partial<RegisterA2AAgentRequest>) =>
      mutation.mutateAsync({ path, data }),
    isUpdating: mutation.isPending,
    error: mutation.error,
    reset: mutation.reset,
  };
}

// ============================================================================
// Agent Delete Mutation
// ============================================================================

export interface UseDeleteAgentResult {
  deleteAgent: (path: string) => Promise<void>;
  isDeleting: boolean;
  error: Error | null;
  reset: () => void;
}

/**
 * Delete an A2A agent
 */
export function useDeleteAgent(): UseDeleteAgentResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: deleteA2AAgent,
    onSuccess: (_, path) => {
      // Invalidate the agents list
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.all });
      // Remove the specific agent from cache
      queryClient.removeQueries({ queryKey: agentQueryKeys.detail(path) });
    },
  });

  return {
    deleteAgent: mutation.mutateAsync,
    isDeleting: mutation.isPending,
    error: mutation.error,
    reset: mutation.reset,
  };
}

// ============================================================================
// Agent Toggle Mutation
// ============================================================================

export interface UseToggleAgentResult {
  toggleAgent: (path: string, enabled: boolean) => Promise<A2AAgent>;
  isToggling: boolean;
  error: Error | null;
  reset: () => void;
}

/**
 * Toggle agent enabled/disabled state
 */
export function useToggleAgent(): UseToggleAgentResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: ({ path, enabled }: { path: string; enabled: boolean }) =>
      toggleA2AAgent(path, enabled),
    onMutate: async ({ path, enabled }) => {
      // Cancel any outgoing refetches
      await queryClient.cancelQueries({ queryKey: agentQueryKeys.all });

      // Snapshot the previous value
      const previousAgents = queryClient.getQueryData<A2AAgent[]>(agentQueryKeys.all);

      // Optimistically update
      if (previousAgents) {
        queryClient.setQueryData<A2AAgent[]>(
          agentQueryKeys.all,
          previousAgents.map((a) =>
            a.path === path ? { ...a, is_enabled: enabled } : a
          )
        );
      }

      return { previousAgents };
    },
    onError: (_, __, context) => {
      // Rollback on error
      if (context?.previousAgents) {
        queryClient.setQueryData(agentQueryKeys.all, context.previousAgents);
      }
    },
    onSettled: (_, __, variables) => {
      // Always refetch after error or success
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.all });
      queryClient.invalidateQueries({ queryKey: agentQueryKeys.detail(variables.path) });
    },
  });

  return {
    toggleAgent: (path: string, enabled: boolean) =>
      mutation.mutateAsync({ path, enabled }),
    isToggling: mutation.isPending,
    error: mutation.error,
    reset: mutation.reset,
  };
}

// ============================================================================
// Agent Filtering Utilities
// ============================================================================

export type AgentFilter = {
  search: string;
  status: 'all' | 'enabled' | 'disabled';
  healthStatus: 'all' | 'healthy' | 'unhealthy' | 'unknown';
  visibility: 'all' | 'public' | 'private';
};

/**
 * Filter agents based on search and status criteria
 */
export function filterAgents(agents: A2AAgent[], filter: AgentFilter): A2AAgent[] {
  return agents.filter((agent) => {
    // Search filter
    if (filter.search) {
      const searchLower = filter.search.toLowerCase();
      const matchesSearch =
        agent.name.toLowerCase().includes(searchLower) ||
        agent.path.toLowerCase().includes(searchLower) ||
        agent.description?.toLowerCase().includes(searchLower) ||
        agent.tags?.some((tag) => tag.toLowerCase().includes(searchLower)) ||
        agent.skills?.some((skill) => skill.toLowerCase().includes(searchLower));
      if (!matchesSearch) return false;
    }

    // Status filter
    if (filter.status !== 'all') {
      if (filter.status === 'enabled' && !agent.is_enabled) return false;
      if (filter.status === 'disabled' && agent.is_enabled) return false;
    }

    // Health status filter
    if (filter.healthStatus !== 'all') {
      if (agent.health_status !== filter.healthStatus) return false;
    }

    // Visibility filter
    if (filter.visibility !== 'all') {
      if (agent.visibility !== filter.visibility) return false;
    }

    return true;
  });
}
