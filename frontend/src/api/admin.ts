/**
 * Admin API functions
 */

import { apiGet, apiPost, apiPut, apiDelete } from './client';
import type {
  EgressAllowlistEntry,
  CreateEgressAllowlistEntryRequest,
  UpdateEgressAllowlistEntryRequest,
} from './types';

// ============================================================================
// Egress Allowlist API
// ============================================================================

/**
 * Get all egress allowlist entries
 */
export async function getEgressAllowlistEntries(): Promise<EgressAllowlistEntry[]> {
  return apiGet<EgressAllowlistEntry[]>('/enforceai/admin/egress-allowlist');
}

/**
 * Create a new egress allowlist entry
 */
export async function createEgressAllowlistEntry(
  data: CreateEgressAllowlistEntryRequest
): Promise<EgressAllowlistEntry> {
  return apiPost<EgressAllowlistEntry, CreateEgressAllowlistEntryRequest>(
    '/enforceai/admin/egress-allowlist',
    data
  );
}

/**
 * Update an existing egress allowlist entry
 */
export async function updateEgressAllowlistEntry(
  entryId: string,
  data: UpdateEgressAllowlistEntryRequest
): Promise<EgressAllowlistEntry> {
  return apiPut<EgressAllowlistEntry, UpdateEgressAllowlistEntryRequest>(
    `/enforceai/admin/egress-allowlist/${entryId}`,
    data
  );
}

/**
 * Delete an egress allowlist entry
 */
export async function deleteEgressAllowlistEntry(entryId: string): Promise<void> {
  return apiDelete<void>(`/enforceai/admin/egress-allowlist/${entryId}`);
}

/**
 * Check if a pattern is allowed
 */
export async function checkEgressPattern(pattern: string): Promise<{ allowed: boolean; reason?: string }> {
  return apiPost<{ allowed: boolean; reason?: string }, { pattern: string }>(
    '/enforceai/admin/egress-allowlist/check',
    { pattern }
  );
}
