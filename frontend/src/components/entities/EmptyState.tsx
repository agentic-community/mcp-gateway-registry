import React from 'react';
import clsx from 'clsx';

/**
 * Visual tone for the empty-state panel. Each entity collection tints its
 * empty state to match its accent (servers neutral, agents cyan, skills amber,
 * virtual teal); `error` is the red failure panel.
 */
export type EmptyStateTone = 'neutral' | 'cyan' | 'amber' | 'teal' | 'error';

const TONE_CLASSES: Record<EmptyStateTone, string> = {
  neutral: 'bg-gray-50 dark:bg-gray-800',
  cyan: 'bg-cyan-50 dark:bg-cyan-900/20 border border-cyan-200 dark:border-cyan-800',
  amber: 'bg-amber-50 dark:bg-amber-900/20 border border-amber-200 dark:border-amber-800',
  teal: 'bg-teal-50 dark:bg-teal-900/20 border border-teal-200 dark:border-teal-800',
  error: 'bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800',
};

interface EmptyStateProps {
  title: string;
  subtitle?: React.ReactNode;
  tone?: EmptyStateTone;
  /** Optional call-to-action button rendered under the message. */
  cta?: React.ReactNode;
  className?: string;
}

/**
 * The "No X found" panel shared by every entity collection on the Dashboard.
 * Replaces the repeated text-center / rounded / tinted blocks. The `error`
 * tone is used for load-failure messages (red title).
 */
const EmptyState: React.FC<EmptyStateProps> = ({
  title,
  subtitle,
  tone = 'neutral',
  cta,
  className,
}) => {
  return (
    <div
      className={clsx('text-center py-12 rounded-lg', TONE_CLASSES[tone], className)}
    >
      <div
        className={clsx(
          'text-lg mb-2',
          tone === 'error' ? 'text-red-500' : 'text-gray-400',
        )}
      >
        {title}
      </div>
      {subtitle && (
        <p
          className={clsx(
            'text-sm',
            tone === 'error'
              ? 'text-red-600 dark:text-red-400'
              : 'text-gray-500 dark:text-gray-300',
          )}
        >
          {subtitle}
        </p>
      )}
      {cta && <div className="mt-6">{cta}</div>}
    </div>
  );
};

export default EmptyState;
