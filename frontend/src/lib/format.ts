import {
  format,
  formatDistanceToNow,
  parseISO,
  isValid,
  differenceInSeconds,
  differenceInMinutes,
  differenceInHours,
  differenceInDays,
} from 'date-fns';

/**
 * Parse a date string safely
 */
export function parseDate(dateString: string | null | undefined): Date | null {
  if (!dateString) return null;

  try {
    const date = parseISO(dateString);
    return isValid(date) ? date : null;
  } catch {
    return null;
  }
}

/**
 * Format a date string for display
 * @example formatDate('2024-01-15T10:30:00Z') => 'Jan 15, 2024 10:30 AM'
 */
export function formatDate(
  dateString: string | null | undefined,
  formatStr: string = 'MMM d, yyyy h:mm a'
): string {
  const date = parseDate(dateString);
  if (!date) return '-';
  return format(date, formatStr);
}

/**
 * Format a date as relative time
 * @example formatRelativeTime('2024-01-15T10:30:00Z') => '2 hours ago'
 */
export function formatRelativeTime(
  dateString: string | null | undefined,
  options?: { addSuffix?: boolean }
): string {
  const date = parseDate(dateString);
  if (!date) return '-';
  return formatDistanceToNow(date, { addSuffix: options?.addSuffix ?? true });
}

/**
 * Format a date as a short relative time (e.g., "2h ago", "3d")
 */
export function formatShortRelativeTime(
  dateString: string | null | undefined
): string {
  const date = parseDate(dateString);
  if (!date) return '-';

  const now = new Date();
  const seconds = differenceInSeconds(now, date);

  if (seconds < 60) {
    return 'just now';
  }

  const minutes = differenceInMinutes(now, date);
  if (minutes < 60) {
    return `${minutes}m ago`;
  }

  const hours = differenceInHours(now, date);
  if (hours < 24) {
    return `${hours}h ago`;
  }

  const days = differenceInDays(now, date);
  if (days < 30) {
    return `${days}d ago`;
  }

  return format(date, 'MMM d');
}

/**
 * Format a date for date-only display
 * @example formatDateOnly('2024-01-15T10:30:00Z') => 'Jan 15, 2024'
 */
export function formatDateOnly(dateString: string | null | undefined): string {
  return formatDate(dateString, 'MMM d, yyyy');
}

/**
 * Format a date for datetime display
 * @example formatDateTime('2024-01-15T10:30:00Z') => '2024-01-15 10:30:00'
 */
export function formatDateTime(dateString: string | null | undefined): string {
  return formatDate(dateString, 'yyyy-MM-dd HH:mm:ss');
}

/**
 * Format a date for ISO format (useful for form inputs)
 */
export function formatISODate(dateString: string | null | undefined): string {
  const date = parseDate(dateString);
  if (!date) return '';
  return format(date, "yyyy-MM-dd'T'HH:mm");
}

/**
 * Check if a date is in the past
 */
export function isPast(dateString: string | null | undefined): boolean {
  const date = parseDate(dateString);
  if (!date) return false;
  return date < new Date();
}

/**
 * Check if a date is in the future
 */
export function isFuture(dateString: string | null | undefined): boolean {
  const date = parseDate(dateString);
  if (!date) return false;
  return date > new Date();
}

/**
 * Format duration in seconds to human readable
 * @example formatDuration(3665) => '1h 1m 5s'
 */
export function formatDuration(seconds: number): string {
  if (seconds < 0) return '-';

  const hours = Math.floor(seconds / 3600);
  const minutes = Math.floor((seconds % 3600) / 60);
  const secs = Math.floor(seconds % 60);

  const parts: string[] = [];
  if (hours > 0) parts.push(`${hours}h`);
  if (minutes > 0) parts.push(`${minutes}m`);
  if (secs > 0 || parts.length === 0) parts.push(`${secs}s`);

  return parts.join(' ');
}
