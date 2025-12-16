/**
 * React Query hooks for MCP Server management
 */

import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import {
  getServers,
  getServer,
  registerServer,
  updateServer,
  deleteServer,
  toggleServer,
  refreshServerHealth,
} from '@/api/registry';
import type { Server, ServerDetails, RegisterServerRequest, EditServerRequest } from '@/api/types';

// Query keys for cache management
export const serverQueryKeys = {
  all: ['servers'] as const,
  detail: (path: string) => ['servers', path] as const,
};

// ============================================================================
// Server List Hook
// ============================================================================

export interface UseServersResult {
  servers: Server[];
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Fetch all MCP servers
 */
export function useServers(): UseServersResult {
  const query = useQuery({
    queryKey: serverQueryKeys.all,
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

// ============================================================================
// Server Detail Hook
// ============================================================================

export interface UseServerResult {
  server: ServerDetails | null;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
}

/**
 * Fetch a single server by path
 */
export function useServer(path: string): UseServerResult {
  const query = useQuery({
    queryKey: serverQueryKeys.detail(path),
    queryFn: () => getServer(path),
    enabled: Boolean(path),
    staleTime: 1000 * 30, // 30 seconds
  });

  return {
    server: query.data ?? null,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
  };
}

// ============================================================================
// Server Registration Mutation
// ============================================================================

export interface UseRegisterServerResult {
  registerServer: (data: RegisterServerRequest) => Promise<Server>;
  isRegistering: boolean;
  error: Error | null;
  reset: () => void;
}

/**
 * Register a new MCP server
 */
export function useRegisterServer(): UseRegisterServerResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: registerServer,
    onSuccess: () => {
      // Invalidate the servers list to refetch
      queryClient.invalidateQueries({ queryKey: serverQueryKeys.all });
    },
  });

  return {
    registerServer: mutation.mutateAsync,
    isRegistering: mutation.isPending,
    error: mutation.error,
    reset: mutation.reset,
  };
}

// ============================================================================
// Server Update Mutation
// ============================================================================

export interface UseUpdateServerResult {
  updateServer: (path: string, data: EditServerRequest) => Promise<Server>;
  isUpdating: boolean;
  error: Error | null;
  reset: () => void;
}

/**
 * Update an existing MCP server
 */
export function useUpdateServer(): UseUpdateServerResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: ({ path, data }: { path: string; data: EditServerRequest }) =>
      updateServer(path, data),
    onSuccess: (_, variables) => {
      // Invalidate both the list and the specific server
      queryClient.invalidateQueries({ queryKey: serverQueryKeys.all });
      queryClient.invalidateQueries({ queryKey: serverQueryKeys.detail(variables.path) });
    },
  });

  return {
    updateServer: (path: string, data: EditServerRequest) =>
      mutation.mutateAsync({ path, data }),
    isUpdating: mutation.isPending,
    error: mutation.error,
    reset: mutation.reset,
  };
}

// ============================================================================
// Server Delete Mutation
// ============================================================================

export interface UseDeleteServerResult {
  deleteServer: (path: string) => Promise<void>;
  isDeleting: boolean;
  error: Error | null;
  reset: () => void;
}

/**
 * Delete an MCP server
 */
export function useDeleteServer(): UseDeleteServerResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: deleteServer,
    onSuccess: (_, path) => {
      // Invalidate the servers list
      queryClient.invalidateQueries({ queryKey: serverQueryKeys.all });
      // Remove the specific server from cache
      queryClient.removeQueries({ queryKey: serverQueryKeys.detail(path) });
    },
  });

  return {
    deleteServer: mutation.mutateAsync,
    isDeleting: mutation.isPending,
    error: mutation.error,
    reset: mutation.reset,
  };
}

// ============================================================================
// Server Toggle Mutation
// ============================================================================

export interface UseToggleServerResult {
  toggleServer: (path: string, enabled: boolean) => Promise<Server>;
  isToggling: boolean;
  error: Error | null;
  reset: () => void;
}

/**
 * Toggle server enabled/disabled state
 */
export function useToggleServer(): UseToggleServerResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: ({ path, enabled }: { path: string; enabled: boolean }) =>
      toggleServer(path, enabled),
    onMutate: async ({ path, enabled }) => {
      // Cancel any outgoing refetches
      await queryClient.cancelQueries({ queryKey: serverQueryKeys.all });

      // Snapshot the previous value
      const previousServers = queryClient.getQueryData<Server[]>(serverQueryKeys.all);

      // Optimistically update
      if (previousServers) {
        queryClient.setQueryData<Server[]>(
          serverQueryKeys.all,
          previousServers.map((s) =>
            s.path === path ? { ...s, is_enabled: enabled } : s
          )
        );
      }

      return { previousServers };
    },
    onError: (_, __, context) => {
      // Rollback on error
      if (context?.previousServers) {
        queryClient.setQueryData(serverQueryKeys.all, context.previousServers);
      }
    },
    onSettled: (_, __, variables) => {
      // Always refetch after error or success
      queryClient.invalidateQueries({ queryKey: serverQueryKeys.all });
      queryClient.invalidateQueries({ queryKey: serverQueryKeys.detail(variables.path) });
    },
  });

  return {
    toggleServer: (path: string, enabled: boolean) =>
      mutation.mutateAsync({ path, enabled }),
    isToggling: mutation.isPending,
    error: mutation.error,
    reset: mutation.reset,
  };
}

// ============================================================================
// Server Refresh Health Mutation
// ============================================================================

export interface UseRefreshServerResult {
  refreshServer: (path: string) => Promise<Server>;
  isRefreshing: boolean;
  refreshingPath: string | null;
  error: Error | null;
  reset: () => void;
}

/**
 * Refresh server health status
 */
export function useRefreshServer(): UseRefreshServerResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: refreshServerHealth,
    onSuccess: (updatedServer, path) => {
      // Update the server in the list cache
      const previousServers = queryClient.getQueryData<Server[]>(serverQueryKeys.all);
      if (previousServers) {
        queryClient.setQueryData<Server[]>(
          serverQueryKeys.all,
          previousServers.map((s) =>
            s.path === path ? { ...s, ...updatedServer } : s
          )
        );
      }

      // Update the specific server cache if it exists
      queryClient.setQueryData(serverQueryKeys.detail(path), updatedServer);
    },
  });

  return {
    refreshServer: mutation.mutateAsync,
    isRefreshing: mutation.isPending,
    refreshingPath: mutation.isPending ? (mutation.variables as string) : null,
    error: mutation.error,
    reset: mutation.reset,
  };
}

// ============================================================================
// Server Filtering Utilities
// ============================================================================

export type ServerFilter = {
  search: string;
  status: 'all' | 'enabled' | 'disabled';
  healthStatus: 'all' | 'healthy' | 'unhealthy' | 'unknown';
};

/**
 * Filter servers based on search and status criteria
 */
export function filterServers(servers: Server[], filter: ServerFilter): Server[] {
  return servers.filter((server) => {
    // Search filter
    if (filter.search) {
      const searchLower = filter.search.toLowerCase();
      const matchesSearch =
        server.display_name.toLowerCase().includes(searchLower) ||
        server.path.toLowerCase().includes(searchLower) ||
        server.description?.toLowerCase().includes(searchLower) ||
        server.tags?.some((tag) => tag.toLowerCase().includes(searchLower));
      if (!matchesSearch) return false;
    }

    // Status filter
    if (filter.status !== 'all') {
      if (filter.status === 'enabled' && !server.is_enabled) return false;
      if (filter.status === 'disabled' && server.is_enabled) return false;
    }

    // Health status filter
    if (filter.healthStatus !== 'all') {
      if (server.health_status !== filter.healthStatus) return false;
    }

    return true;
  });
}
