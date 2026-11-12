import React from 'react';
import clsx from 'clsx';
import { ACCENTS, AccentToken } from '../../theme/accents';

/**
 * Color tone for a status dot. `off` is the muted gray used for disabled /
 * inactive states; the rest map to the glow colors the card footers use.
 *
 * Tones describe *meaning* (healthy/degraded/down), not card identity — a
 * green "Enabled" dot looks the same on every entity type for consistency.
 */
export type StatusTone =
  | 'green'
  | 'emerald'
  | 'orange'
  | 'red'
  | 'amber'
  | 'off';

const TONE_CLASSES: Record<StatusTone, string> = {
  green: 'bg-success-400 shadow-lg shadow-success-400/30',
  emerald: 'bg-success-400 shadow-lg shadow-success-400/30',
  orange: 'bg-warning-400 shadow-lg shadow-warning-400/30',
  red: 'bg-danger-400 shadow-lg shadow-danger-400/30',
  amber: 'bg-warning-400 shadow-lg shadow-warning-400/30',
  off: 'bg-gray-300 dark:bg-gray-600',
};

interface StatusDotProps {
  tone: StatusTone;
  label: string;
  title?: string;
  className?: string;
}

/**
 * A glowing status dot plus its label, used across every card footer for
 * enabled/disabled and health states. Render two side by side with a
 * StatusDivider between them to match the existing footer layout.
 */
const StatusDot: React.FC<StatusDotProps> = ({
  tone,
  label,
  title,
  className,
}) => {
  return (
    <div className={clsx('flex items-center gap-2', className)}>
      <div className={clsx('w-3 h-3 rounded-full', TONE_CLASSES[tone])} />
      <span
        className="text-sm font-medium text-gray-700 dark:text-gray-300"
        title={title}
      >
        {label}
      </span>
    </div>
  );
};

/**
 * Thin vertical rule placed between two StatusDots in a footer. Tinted by the
 * card accent so the divider matches the footer it sits in.
 */
export const StatusDivider: React.FC<{ accent?: AccentToken }> = ({
  accent = 'neutral',
}) => <div className={clsx('w-px h-4', ACCENTS[accent].divider)} />;

export default StatusDot;
