// Lightweight security-scan summary used to colour the shield icon on cards.
// The list endpoints return this shape inline (avoiding a per-card fetch); a
// rescan returns the full scan document, from which we extract the same fields
// so the parent list entry — and thus the badge — stays in sync after a rescan.

export interface SecurityScanSummary {
  scan_failed?: boolean;
  critical_issues?: number;
  high_severity?: number;
  medium_severity?: number;
  low_severity?: number;
}

/**
 * Pick the icon-relevant fields out of a full or partial scan result. Returns
 * null for a missing result so callers can clear the badge to its "unscanned"
 * (gray) state.
 */
export function toScanSummary(
  scan: SecurityScanSummary | null | undefined,
): SecurityScanSummary | null {
  if (!scan) return null;
  return {
    scan_failed: scan.scan_failed ?? false,
    critical_issues: scan.critical_issues ?? 0,
    high_severity: scan.high_severity ?? 0,
    medium_severity: scan.medium_severity ?? 0,
    low_severity: scan.low_severity ?? 0,
  };
}
