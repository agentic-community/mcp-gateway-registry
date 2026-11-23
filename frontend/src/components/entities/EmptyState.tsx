import React from 'react';
import clsx from 'clsx';

/**
 * Visual tone for the empty-state panel. Each entity collection tints its
 * empty state to match its accent (servers neutral, agents cyan, skills amber,
 * virtual teal); `error` is the red failure panel.
 */
export type EmptyStateTone = 'neutral' | 'cyan' | 'amber' | 'teal' | 'error';

const TONE_CLASSES: Record<EmptyStateTone, string> = {
  neutral: 'bg-surface-muted border border-line',
  cyan: 'bg-agent-50 dark:bg-agent-900/20 border border-agent-200 dark:border-agent-800',
  amber: 'bg-skill-50 dark:bg-skill-900/20 border border-skill-200 dark:border-skill-800',
  teal: 'bg-virtual-50 dark:bg-virtual-900/20 border border-virtual-200 dark:border-virtual-800',
  error: 'bg-danger-50 dark:bg-danger-900/20 border border-danger-200 dark:border-danger-800',
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
          tone === 'error' ? 'text-danger-500' : 'text-fg-subtle',
        )}
      >
        {title}
      </div>
      {subtitle && (
        <p
          className={clsx(
            'text-sm',
            tone === 'error'
              ? 'text-danger-600 dark:text-danger-400'
              : 'text-fg-muted',
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
