/**
 * Audit hooks for React Query
 */

import { useQuery } from '@tanstack/react-query';
import { getAuditEvents, getAdminAuditEvents } from '@/api/enforceai';
import type { AuditEventsQuery, AdminAuditEventsQuery, AuditEventsListResponse } from '@/api/types';

interface AuditQueryOptions {
  enabled?: boolean;
}

/**
 * Query key factory for audit events
 */
export const auditQueryKeys = {
  all: ['audit'] as const,
  events: () => [...auditQueryKeys.all, 'events'] as const,
  eventsList: (filters: AuditEventsQuery) => [...auditQueryKeys.events(), filters] as const,
  adminEvents: () => [...auditQueryKeys.all, 'admin-events'] as const,
  adminEventsList: (filters: AdminAuditEventsQuery) => [...auditQueryKeys.adminEvents(), filters] as const,
};

/**
 * Default query parameters
 */
export const DEFAULT_AUDIT_QUERY: AuditEventsQuery = {
  limit: 50,
};

/**
 * Hook for fetching audit events with filtering
 */
export function useAuditEvents(
  filters: AuditEventsQuery = {},
  options: AuditQueryOptions = {}
): {
  data: AuditEventsListResponse | undefined;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
  isFetching: boolean;
} {
  const mergedFilters = { ...DEFAULT_AUDIT_QUERY, ...filters };

  const query = useQuery({
    queryKey: auditQueryKeys.eventsList(mergedFilters),
    queryFn: () => getAuditEvents(mergedFilters),
    staleTime: 30 * 1000, // 30 seconds - audit data can be slightly stale
    refetchOnWindowFocus: true,
    enabled: options.enabled ?? true,
  });

  return {
    data: query.data,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
    isFetching: query.isFetching,
  };
}

/**
 * Hook for fetching admin audit events (all users)
 */
export function useAdminAuditEvents(
  filters: AdminAuditEventsQuery = {},
  options: AuditQueryOptions = {}
): {
  data: AuditEventsListResponse | undefined;
  isLoading: boolean;
  isError: boolean;
  error: Error | null;
  refetch: () => void;
  isFetching: boolean;
} {
  const mergedFilters = { ...DEFAULT_AUDIT_QUERY, ...filters };

  const query = useQuery({
    queryKey: auditQueryKeys.adminEventsList(mergedFilters),
    queryFn: () => getAdminAuditEvents(mergedFilters),
    staleTime: 30 * 1000, // 30 seconds
    refetchOnWindowFocus: true,
    enabled: options.enabled ?? true,
  });

  return {
    data: query.data,
    isLoading: query.isLoading,
    isError: query.isError,
    error: query.error,
    refetch: query.refetch,
    isFetching: query.isFetching,
  };
}

/**
 * Time presets for quick filtering
 */
export const TIME_PRESETS = [
  { label: '15 minutes', value: '15m', minutes: 15 },
  { label: '1 hour', value: '1h', minutes: 60 },
  { label: '24 hours', value: '24h', minutes: 60 * 24 },
  { label: '7 days', value: '7d', minutes: 60 * 24 * 7 },
] as const;

export type TimePresetValue = (typeof TIME_PRESETS)[number]['value'];

/**
 * Calculate the 'since' datetime from a preset
 */
export function getTimePresetSince(preset: TimePresetValue): string {
  const presetConfig = TIME_PRESETS.find((p) => p.value === preset);
  if (!presetConfig) {
    throw new Error(`Unknown time preset: ${preset}`);
  }

  const since = new Date(Date.now() - presetConfig.minutes * 60 * 1000);
  return since.toISOString();
}

/**
 * Outcome options for filtering
 */
export const OUTCOME_OPTIONS = [
  { value: 'allow', label: 'Allowed', variant: 'success' as const },
  { value: 'deny', label: 'Denied', variant: 'error' as const },
] as const;

/**
 * Common action types for filtering
 */
export const ACTION_CATEGORIES = [
  {
    category: 'MCP Operations',
    actions: ['tools/list', 'tools/call', 'resources/list', 'resources/read', 'prompts/list', 'prompts/get'],
  },
  {
    category: 'Agent Management',
    actions: [
      'management/agents/create',
      'management/agents/update',
      'management/agents/revoke',
      'management/agents/list',
      'management/agents/get',
    ],
  },
  {
    category: 'API Key Management',
    actions: ['management/api-keys/create', 'management/api-keys/revoke', 'management/api-keys/list'],
  },
  {
    category: 'Token Management',
    actions: ['management/tokens/mint', 'management/tokens/revoke', 'management/tokens/revoke-all'],
  },
] as const;

/**
 * Flattened list of all action types
 */
export const ALL_ACTIONS = ACTION_CATEGORIES.flatMap((cat) => cat.actions);

/**
 * Interface for advanced filter state
 */
export interface AdvancedFilters {
  agentId: string;
  actions: string[];
  requestId: string;
  server: string;
  tool: string;
  userId: string; // For admin mode
}

/**
 * Initial advanced filter state
 */
export const INITIAL_ADVANCED_FILTERS: AdvancedFilters = {
  agentId: '',
  actions: [],
  requestId: '',
  server: '',
  tool: '',
  userId: '',
};

/**
 * Check if any advanced filters are active
 */
export function hasActiveAdvancedFilters(filters: AdvancedFilters, isAdmin: boolean = false): boolean {
  return (
    filters.agentId.trim() !== '' ||
    filters.actions.length > 0 ||
    filters.requestId.trim() !== '' ||
    filters.server.trim() !== '' ||
    filters.tool.trim() !== '' ||
    (isAdmin && filters.userId.trim() !== '')
  );
}

/**
 * Count active advanced filters
 */
export function countActiveAdvancedFilters(filters: AdvancedFilters, isAdmin: boolean = false): number {
  let count = 0;
  if (filters.agentId.trim()) count++;
  if (filters.actions.length > 0) count++;
  if (filters.requestId.trim()) count++;
  if (filters.server.trim()) count++;
  if (filters.tool.trim()) count++;
  if (isAdmin && filters.userId.trim()) count++;
  return count;
}
