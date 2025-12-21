/**
 * Admin API functions
 */

import { apiGet, apiPost, apiPut, apiDelete } from './client';
import type {
  EgressAllowlistEntry,
  CreateEgressAllowlistEntryRequest,
  UpdateEgressAllowlistEntryRequest,
  UpstreamOAuthProviderPublic,
  UpstreamOAuthProviderCreateRequest,
  UpstreamOAuthProviderUpdateRequest,
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

// ============================================================================
// Upstream OAuth Provider Registry API (Admin)
// ============================================================================

export async function listUpstreamOAuthProviders(): Promise<UpstreamOAuthProviderPublic[]> {
  return apiGet<UpstreamOAuthProviderPublic[]>('/enforceai/admin/upstream-oauth-providers');
}

export async function getUpstreamOAuthProvider(
  providerId: string
): Promise<UpstreamOAuthProviderPublic> {
  return apiGet<UpstreamOAuthProviderPublic>(
    `/enforceai/admin/upstream-oauth-providers/${encodeURIComponent(providerId)}`
  );
}

export async function createUpstreamOAuthProvider(
  data: UpstreamOAuthProviderCreateRequest
): Promise<UpstreamOAuthProviderPublic> {
  return apiPost<UpstreamOAuthProviderPublic, UpstreamOAuthProviderCreateRequest>(
    '/enforceai/admin/upstream-oauth-providers',
    data
  );
}

export async function updateUpstreamOAuthProvider(
  providerId: string,
  data: UpstreamOAuthProviderUpdateRequest
): Promise<UpstreamOAuthProviderPublic> {
  return apiPut<UpstreamOAuthProviderPublic, UpstreamOAuthProviderUpdateRequest>(
    `/enforceai/admin/upstream-oauth-providers/${encodeURIComponent(providerId)}`,
    data
  );
}

export async function deleteUpstreamOAuthProvider(providerId: string): Promise<void> {
  await apiDelete<{ ok: boolean }>(
    `/enforceai/admin/upstream-oauth-providers/${encodeURIComponent(providerId)}`
  );
}
