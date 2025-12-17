import { useQuery, useMutation, useQueryClient } from '@tanstack/react-query';
import { apiClient } from '@/api/client';
import { adminCreateScope, adminDeleteScope, adminReplaceScope } from '@/api/enforceai';
import type {
  ScopeCatalog,
  ScopeDefinition,
  ScopeInfo,
  ServerPermission,
  CreateScopeRequest,
  ReplaceScopeRequest,
  ScopeMutationResponse,
} from '@/api/types';

// ============================================================================
// Query Keys
// ============================================================================

export const scopesKeys = {
  all: ['scopes'] as const,
  catalog: () => [...scopesKeys.all, 'catalog'] as const,
};

// ============================================================================
// API Functions
// ============================================================================

async function fetchScopeCatalog(): Promise<ScopeCatalog> {
  const response = await apiClient.get<ScopeCatalog>('/enforceai/scopes/catalog');
  return response.data;
}

// ============================================================================
// Utility Functions
// ============================================================================

/**
 * Transform a ScopeDefinition into a simplified ScopeInfo for display
 */
export function transformScopeToInfo(scope: ScopeDefinition): ScopeInfo {
  const servers = new Set<string>();
  const methods = new Set<string>();
  const tools = new Set<string>();
  let allServers = false;
  let allMethods = false;
  let allTools = false;

  for (const perm of scope.server_permissions) {
    if (perm.server === '*') {
      allServers = true;
    } else {
      servers.add(perm.server);
    }

    if (perm.methods.all_methods) {
      allMethods = true;
    } else {
      perm.methods.methods.forEach((m) => methods.add(m));
    }

    if (perm.tools) {
      if (perm.tools.all_tools) {
        allTools = true;
      } else {
        perm.tools.tools.forEach((t) => tools.add(t));
      }
    }
  }

  const agentActions = scope.agent_permissions.map((p) => p.action);

  return {
    name: scope.name,
    servers: Array.from(servers),
    methods: Array.from(methods),
    tools: Array.from(tools),
    all_servers: allServers,
    all_methods: allMethods,
    all_tools: allTools,
    agent_actions: agentActions,
  };
}

/**
 * Get display text for servers in a scope
 */
export function getServersDisplay(scope: ScopeInfo): string {
  if (scope.all_servers) return 'All servers';
  if (scope.servers.length === 0) return 'None';
  if (scope.servers.length <= 3) return scope.servers.join(', ');
  return `${scope.servers.slice(0, 3).join(', ')} +${scope.servers.length - 3} more`;
}

/**
 * Get display text for methods in a scope
 */
export function getMethodsDisplay(scope: ScopeInfo): string {
  if (scope.all_methods) return 'All methods';
  if (scope.methods.length === 0) return 'None';
  if (scope.methods.length <= 3) return scope.methods.join(', ');
  return `${scope.methods.slice(0, 3).join(', ')} +${scope.methods.length - 3} more`;
}

/**
 * Get display text for tools in a scope
 */
export function getToolsDisplay(scope: ScopeInfo): string {
  if (scope.all_tools) return 'All tools';
  if (scope.tools.length === 0) return 'Inherits from server';
  if (scope.tools.length <= 3) return scope.tools.join(', ');
  return `${scope.tools.slice(0, 3).join(', ')} +${scope.tools.length - 3} more`;
}

/**
 * Filter scopes by search query
 */
export function filterScopes(
  scopes: ScopeInfo[],
  query: string
): ScopeInfo[] {
  if (!query.trim()) return scopes;

  const lowerQuery = query.toLowerCase();
  return scopes.filter((scope) => {
    // Search by scope name
    if (scope.name.toLowerCase().includes(lowerQuery)) return true;

    // Search by server name
    if (scope.servers.some((s) => s.toLowerCase().includes(lowerQuery)))
      return true;

    // Search by tool name
    if (scope.tools.some((t) => t.toLowerCase().includes(lowerQuery)))
      return true;

    // Search by agent action
    if (scope.agent_actions.some((a) => a.toLowerCase().includes(lowerQuery)))
      return true;

    return false;
  });
}

/**
 * Get server permissions grouped by server for a scope definition
 */
export function getServerPermissionsByServer(
  scope: ScopeDefinition
): Map<string, ServerPermission[]> {
  const grouped = new Map<string, ServerPermission[]>();
  for (const perm of scope.server_permissions) {
    const existing = grouped.get(perm.server) || [];
    existing.push(perm);
    grouped.set(perm.server, existing);
  }
  return grouped;
}

// ============================================================================
// Hooks
// ============================================================================

/**
 * Fetch the complete scope catalog
 */
export function useScopeCatalog() {
  const {
    data: catalog,
    isLoading,
    isError,
    error,
    refetch,
  } = useQuery({
    queryKey: scopesKeys.catalog(),
    queryFn: fetchScopeCatalog,
    staleTime: 5 * 60 * 1000, // 5 minutes
    retry: 1, // Only retry once since catalog might not be available
  });

  // Transform scopes to ScopeInfo for easier display
  const scopeInfos: ScopeInfo[] = catalog
    ? Object.values(catalog.scopes).map(transformScopeToInfo)
    : [];

  // Get unique servers across all scopes
  const allServers = catalog
    ? [...new Set(
        Object.values(catalog.scopes).flatMap((scope) =>
          scope.server_permissions.map((p) => p.server)
        )
      )].filter((s) => s !== '*')
    : [];

  // Get group mappings for reference
  const groupMappings = catalog?.group_mappings || {};

  return {
    catalog,
    scopes: scopeInfos,
    allServers,
    groupMappings,
    isLoading,
    isError,
    error,
    refetch,
    isAvailable: !!catalog,
  };
}

// ============================================================================
// Hooks: Mutations
// ============================================================================

interface UseCreateScopeResult {
  createScope: (
    data: CreateScopeRequest,
    ifMatch?: string
  ) => Promise<ScopeMutationResponse>;
  isCreating: boolean;
  error: Error | null;
}

export function useCreateScope(): UseCreateScopeResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: ({ data, ifMatch }: { data: CreateScopeRequest; ifMatch?: string }) =>
      adminCreateScope(data, ifMatch),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: scopesKeys.catalog() });
    },
  });

  return {
    createScope: async (data: CreateScopeRequest, ifMatch?: string) =>
      mutation.mutateAsync({ data, ifMatch }),
    isCreating: mutation.isPending,
    error: mutation.error as Error | null,
  };
}

interface UseReplaceScopeResult {
  replaceScope: (
    scopeName: string,
    data: ReplaceScopeRequest,
    ifMatch: string
  ) => Promise<ScopeMutationResponse>;
  isUpdating: boolean;
  error: Error | null;
}

export function useReplaceScope(): UseReplaceScopeResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: ({
      scopeName,
      data,
      ifMatch,
    }: {
      scopeName: string;
      data: ReplaceScopeRequest;
      ifMatch: string;
    }) => adminReplaceScope(scopeName, data, ifMatch),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: scopesKeys.catalog() });
    },
  });

  return {
    replaceScope: async (scopeName: string, data: ReplaceScopeRequest, ifMatch: string) =>
      mutation.mutateAsync({ scopeName, data, ifMatch }),
    isUpdating: mutation.isPending,
    error: mutation.error as Error | null,
  };
}

interface UseDeleteScopeResult {
  deleteScope: (scopeName: string, ifMatch: string) => Promise<ScopeMutationResponse>;
  isDeleting: boolean;
  error: Error | null;
}

export function useDeleteScope(): UseDeleteScopeResult {
  const queryClient = useQueryClient();

  const mutation = useMutation({
    mutationFn: ({ scopeName, ifMatch }: { scopeName: string; ifMatch: string }) =>
      adminDeleteScope(scopeName, ifMatch),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: scopesKeys.catalog() });
    },
  });

  return {
    deleteScope: async (scopeName: string, ifMatch: string) =>
      mutation.mutateAsync({ scopeName, ifMatch }),
    isDeleting: mutation.isPending,
    error: mutation.error as Error | null,
  };
}

/**
 * Get a specific scope by name
 */
export function useScope(scopeName: string) {
  const { catalog, isLoading, isError, error } = useScopeCatalog();

  const scope = catalog?.scopes[scopeName] || null;
  const scopeInfo = scope ? transformScopeToInfo(scope) : null;

  return {
    scope,
    scopeInfo,
    isLoading,
    isError,
    error,
    isAvailable: !!scope,
  };
}

/**
 * Get all scopes that grant access to a specific server
 */
export function useScopesByServer(serverName: string) {
  const { catalog, isLoading, isError } = useScopeCatalog();

  const matchingScopes: ScopeInfo[] = [];

  if (catalog) {
    for (const [, scope] of Object.entries(catalog.scopes)) {
      const hasServer = scope.server_permissions.some(
        (p) => p.server === serverName || p.server === '*'
      );
      if (hasServer) {
        matchingScopes.push(transformScopeToInfo(scope));
      }
    }
  }

  return {
    scopes: matchingScopes,
    isLoading,
    isError,
  };
}

/**
 * Get all scope names for use in forms (e.g., agent creation)
 */
export function useScopeNames() {
  const { catalog, isLoading, isError } = useScopeCatalog();

  const scopeNames = catalog ? Object.keys(catalog.scopes) : [];

  return {
    scopeNames,
    isLoading,
    isError,
  };
}
