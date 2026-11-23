import type { AuditFilters } from '../components/AuditFilterBar';

/**
 * Translate the audit filter bar state into the query parameters shared by
 * `/api/audit/events` and `/api/audit/export`, so the grid and the export
 * always agree on which events match.
 */
export function buildAuditQueryParams(filters: AuditFilters): URLSearchParams {
  const params = new URLSearchParams();
  params.set('stream', filters.stream);

  if (filters.from) {
    params.set('from', new Date(filters.from).toISOString());
  }
  if (filters.to) {
    params.set('to', new Date(filters.to).toISOString());
  }
  if (filters.username) {
    params.set('username', filters.username);
  }
  if (filters.operation) {
    params.set('operation', filters.operation);
  }
  if (filters.resourceType) {
    params.set('resource_type', filters.resourceType);
  }
  if (filters.statusMin !== undefined) {
    params.set('status_min', filters.statusMin.toString());
  }
  if (filters.statusMax !== undefined) {
    params.set('status_max', filters.statusMax.toString());
  }
  return params;
}
