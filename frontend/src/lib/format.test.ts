import { describe, it, expect, vi, beforeEach, afterEach } from 'vitest';
import {
  parseDate,
  formatDate,
  formatRelativeTime,
  formatShortRelativeTime,
  formatDateOnly,
  formatDateTime,
  formatISODate,
  isPast,
  isFuture,
  formatDuration,
} from './format';

describe('Date formatting utilities', () => {
  // Mock current date for consistent tests
  const mockDate = new Date('2024-06-15T12:00:00Z');

  beforeEach(() => {
    vi.useFakeTimers();
    vi.setSystemTime(mockDate);
  });

  afterEach(() => {
    vi.useRealTimers();
  });

  describe('parseDate', () => {
    it('parses valid ISO date strings', () => {
      const result = parseDate('2024-01-15T10:30:00Z');
      expect(result).toBeInstanceOf(Date);
      expect(result?.toISOString()).toBe('2024-01-15T10:30:00.000Z');
    });

    it('returns null for null input', () => {
      expect(parseDate(null)).toBeNull();
    });

    it('returns null for undefined input', () => {
      expect(parseDate(undefined)).toBeNull();
    });

    it('returns null for invalid date strings', () => {
      expect(parseDate('not-a-date')).toBeNull();
    });

    it('returns null for empty string', () => {
      expect(parseDate('')).toBeNull();
    });
  });

  describe('formatDate', () => {
    it('formats dates with default format', () => {
      const result = formatDate('2024-01-15T10:30:00Z');
      expect(result).toMatch(/Jan 15, 2024/);
    });

    it('formats dates with custom format', () => {
      const result = formatDate('2024-01-15T10:30:00Z', 'yyyy-MM-dd');
      expect(result).toBe('2024-01-15');
    });

    it('returns dash for null input', () => {
      expect(formatDate(null)).toBe('-');
    });

    it('returns dash for invalid date', () => {
      expect(formatDate('invalid')).toBe('-');
    });
  });

  describe('formatRelativeTime', () => {
    it('formats past dates with suffix', () => {
      const pastDate = new Date(mockDate.getTime() - 2 * 60 * 60 * 1000); // 2 hours ago
      const result = formatRelativeTime(pastDate.toISOString());
      expect(result).toContain('ago');
    });

    it('formats without suffix when specified', () => {
      const pastDate = new Date(mockDate.getTime() - 2 * 60 * 60 * 1000);
      const result = formatRelativeTime(pastDate.toISOString(), { addSuffix: false });
      expect(result).not.toContain('ago');
    });

    it('returns dash for null input', () => {
      expect(formatRelativeTime(null)).toBe('-');
    });
  });

  describe('formatShortRelativeTime', () => {
    it('returns "just now" for very recent times', () => {
      const recentDate = new Date(mockDate.getTime() - 30 * 1000); // 30 seconds ago
      expect(formatShortRelativeTime(recentDate.toISOString())).toBe('just now');
    });

    it('returns minutes for times under an hour', () => {
      const minutesAgo = new Date(mockDate.getTime() - 30 * 60 * 1000); // 30 minutes ago
      expect(formatShortRelativeTime(minutesAgo.toISOString())).toBe('30m ago');
    });

    it('returns hours for times under a day', () => {
      const hoursAgo = new Date(mockDate.getTime() - 5 * 60 * 60 * 1000); // 5 hours ago
      expect(formatShortRelativeTime(hoursAgo.toISOString())).toBe('5h ago');
    });

    it('returns days for times under a month', () => {
      const daysAgo = new Date(mockDate.getTime() - 7 * 24 * 60 * 60 * 1000); // 7 days ago
      expect(formatShortRelativeTime(daysAgo.toISOString())).toBe('7d ago');
    });

    it('returns formatted date for older times', () => {
      const oldDate = new Date(mockDate.getTime() - 60 * 24 * 60 * 60 * 1000); // 60 days ago
      const result = formatShortRelativeTime(oldDate.toISOString());
      expect(result).toMatch(/Apr \d+/);
    });

    it('returns dash for null input', () => {
      expect(formatShortRelativeTime(null)).toBe('-');
    });
  });

  describe('formatDateOnly', () => {
    it('formats date without time', () => {
      const result = formatDateOnly('2024-01-15T10:30:00Z');
      expect(result).toBe('Jan 15, 2024');
    });

    it('returns dash for null input', () => {
      expect(formatDateOnly(null)).toBe('-');
    });
  });

  describe('formatDateTime', () => {
    it('formats date with time in ISO-like format', () => {
      const result = formatDateTime('2024-01-15T10:30:45Z');
      expect(result).toMatch(/2024-01-15 \d{2}:\d{2}:\d{2}/);
    });

    it('returns dash for null input', () => {
      expect(formatDateTime(null)).toBe('-');
    });
  });

  describe('formatISODate', () => {
    it('formats date for datetime-local inputs', () => {
      const result = formatISODate('2024-01-15T10:30:00Z');
      expect(result).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}$/);
    });

    it('returns empty string for null input', () => {
      expect(formatISODate(null)).toBe('');
    });
  });

  describe('isPast', () => {
    it('returns true for past dates', () => {
      const pastDate = new Date(mockDate.getTime() - 1000);
      expect(isPast(pastDate.toISOString())).toBe(true);
    });

    it('returns false for future dates', () => {
      const futureDate = new Date(mockDate.getTime() + 1000);
      expect(isPast(futureDate.toISOString())).toBe(false);
    });

    it('returns false for null input', () => {
      expect(isPast(null)).toBe(false);
    });
  });

  describe('isFuture', () => {
    it('returns true for future dates', () => {
      const futureDate = new Date(mockDate.getTime() + 1000);
      expect(isFuture(futureDate.toISOString())).toBe(true);
    });

    it('returns false for past dates', () => {
      const pastDate = new Date(mockDate.getTime() - 1000);
      expect(isFuture(pastDate.toISOString())).toBe(false);
    });

    it('returns false for null input', () => {
      expect(isFuture(null)).toBe(false);
    });
  });

  describe('formatDuration', () => {
    it('formats seconds only', () => {
      expect(formatDuration(45)).toBe('45s');
    });

    it('formats minutes and seconds', () => {
      expect(formatDuration(125)).toBe('2m 5s');
    });

    it('formats hours, minutes, and seconds', () => {
      expect(formatDuration(3665)).toBe('1h 1m 5s');
    });

    it('formats hours and minutes without seconds', () => {
      expect(formatDuration(3660)).toBe('1h 1m');
    });

    it('returns 0s for zero duration', () => {
      expect(formatDuration(0)).toBe('0s');
    });

    it('returns dash for negative duration', () => {
      expect(formatDuration(-10)).toBe('-');
    });
  });
});
