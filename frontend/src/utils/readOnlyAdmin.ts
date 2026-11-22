/**
 * Read-only admin (view_all, issue #2099) helpers.
 *
 * The registry refuses a few reads to view_all holders on purpose (connection
 * settings, SKILL.md content), because they use stored credentials. Those 403s
 * start with this phrase, set in registry/auth/asset_permissions.py
 * (VIEW_ALL_DENIED_PREFIX), so the UI can show them as a note, not an error.
 */
export const VIEW_ALL_DENIED_PREFIX = 'Not available with read-only admin access';

/** Return the 403 detail when it is a read-only-admin refusal, else null. */
export function readOnlyAdminDenial(error: unknown): string | null {
  const response = (error as { response?: { status?: number; data?: { detail?: unknown } } })
    ?.response;
  const detail = response?.data?.detail;
  if (response?.status === 403 && typeof detail === 'string'
      && detail.startsWith(VIEW_ALL_DENIED_PREFIX)) {
    return detail;
  }
  return null;
}
