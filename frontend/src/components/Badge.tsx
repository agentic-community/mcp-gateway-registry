import React from 'react';
import clsx from 'clsx';

/**
 * Color tone for a Badge. Each maps to the standard
 * `bg-<c>-100 text-<c>-700 dark:bg-<c>-900/30 dark:text-<c>-300` pill the app
 * repeats ~37 times for status/label pills (VIRTUAL, SKILL, OFFICIAL, BEARER
 * AUTH, ORPHANED, …). Centralizing the per-tone classes means a contrast or
 * color tweak is one edit here, not a sweep across call sites.
 *
 * Gradient/bespoke badges (ANTHROPIC, ASOR) and the domain badges that already
 * have their own components (StatusBadge, VersionBadge, ANSBadge, ProviderBadge)
 * stay as-is.
 */
export type BadgeTone =
  | 'gray'
  | 'blue'
  | 'cyan'
  | 'teal'
  | 'green'
  | 'emerald'
  | 'amber'
  | 'yellow'
  | 'red'
  | 'purple'
  | 'violet'
  | 'indigo';

const TONE_CLASSES: Record<BadgeTone, string> = {
  gray: 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300',
  blue: 'bg-info-100 text-info-700 dark:bg-info-900/30 dark:text-info-300',
  cyan: 'bg-agent-100 text-agent-700 dark:bg-agent-900/30 dark:text-agent-300',
  teal: 'bg-virtual-100 text-virtual-700 dark:bg-virtual-900/30 dark:text-virtual-300',
  green: 'bg-success-100 text-success-700 dark:bg-success-900/30 dark:text-success-300',
  emerald:
    'bg-success-100 text-success-700 dark:bg-success-900/30 dark:text-success-300',
  amber: 'bg-warning-100 text-warning-700 dark:bg-warning-900/30 dark:text-warning-300',
  yellow:
    'bg-warning-100 text-warning-700 dark:bg-warning-900/30 dark:text-warning-300',
  red: 'bg-danger-100 text-danger-700 dark:bg-danger-900/30 dark:text-danger-300',
  purple:
    'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300',
  violet:
    'bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300',
  indigo:
    'bg-info-100 text-info-700 dark:bg-info-900/30 dark:text-info-300',
};

interface BadgeProps {
  tone?: BadgeTone;
  /** Pill (rounded-full) by default; `square` uses `rounded`. */
  shape?: 'pill' | 'square';
  /** Adds a subtle border in the tone color. */
  bordered?: boolean;
  className?: string;
  title?: string;
  children: React.ReactNode;
}

const BORDER_CLASSES: Record<BadgeTone, string> = {
  gray: 'border border-gray-200 dark:border-gray-600',
  blue: 'border border-info-200 dark:border-info-600',
  cyan: 'border border-agent-200 dark:border-agent-600',
  teal: 'border border-virtual-200 dark:border-virtual-600',
  green: 'border border-success-200 dark:border-success-700',
  emerald: 'border border-success-200 dark:border-success-600',
  amber: 'border border-warning-200 dark:border-warning-600',
  yellow: 'border border-warning-200 dark:border-warning-600',
  red: 'border border-danger-200 dark:border-danger-600',
  purple: 'border border-purple-200 dark:border-purple-600',
  violet: 'border border-purple-200 dark:border-purple-600',
  indigo: 'border border-info-200 dark:border-info-600',
};

/**
 * Small status/label pill. Use for ad-hoc colored badges; reach for the
 * dedicated StatusBadge/VersionBadge/ANSBadge/ProviderBadge for those domains.
 */
const Badge: React.FC<BadgeProps> = ({
  tone = 'gray',
  shape = 'pill',
  bordered = false,
  className,
  title,
  children,
}) => {
  return (
    <span
      title={title}
      className={clsx(
        'inline-flex items-center px-2 py-0.5 text-xs font-semibold flex-shrink-0',
        shape === 'pill' ? 'rounded-full' : 'rounded',
        TONE_CLASSES[tone],
        bordered && BORDER_CLASSES[tone],
        className,
      )}
    >
      {children}
    </span>
  );
};

export default Badge;
