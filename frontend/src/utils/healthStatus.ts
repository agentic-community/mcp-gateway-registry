// Backend `registry/constants.py` HealthStatus emits 9 distinct values
// including `local`, `checking`, and granular `unhealthy: <reason>` strings.
// The UI groups these into a smaller display set for icons / labels.
export type DisplayStatus =
  | 'healthy'
  | 'healthy-auth-expired'
  | 'unhealthy'
  | 'unknown'
  | 'local';


export function normalizeHealthStatus(raw: string | undefined | null): DisplayStatus {
  if (!raw) return 'unknown';
  if (raw === 'healthy') return 'healthy';
  if (raw === 'healthy-auth-expired') return 'healthy-auth-expired';
  if (raw === 'local') return 'local';
  if (raw.startsWith('unhealthy')) return 'unhealthy';
  return 'unknown';
}
