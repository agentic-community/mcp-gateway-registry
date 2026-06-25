import React from 'react';
import clsx from 'clsx';

interface CardHeaderProps {
  title: string;
  /** Optional monospace path/code line under the title. */
  path?: string;
  /** Badge slot rendered inline after the title (StatusBadge, custom pills). */
  badges?: React.ReactNode;
  /** Right-aligned action buttons (info, security scan, edit, delete). */
  actions?: React.ReactNode;
  /**
   * Allow the title+badges row to wrap to a second line. ServerCard and
   * SkillCard wrap; Agent/VirtualServer/Custom keep a single row. Off by default.
   */
  wrapBadges?: boolean;
  /**
   * Apply a 120px min-width to the title so badges keep a stable left edge.
   * Only ServerCard uses this on main. Off by default.
   */
  minTitleWidth?: boolean;
  /**
   * Wrap the action buttons in a `flex items-center gap-1 flex-shrink-0`
   * container. Only VirtualServerCard does this on main; the other cards render
   * action buttons directly in the header row. Off by default.
   */
  wrapActions?: boolean;
  /**
   * Body content (description, tags, trust bar) rendered inside the same
   * padded block as the header row, below it. Main's cards keep the
   * description and tags inside the header's `p-5 pb-4` block rather than in a
   * separate padded section, so passing them here reproduces that structure.
   */
  children?: React.ReactNode;
  className?: string;
}

/**
 * Shared card header: a wrapping row of title + badges on the left and an
 * action-button cluster on the right, with an optional monospace path beneath.
 *
 * Cards pass their entity-specific badges and action buttons as slots; the
 * layout (flex, wrapping, truncation) is owned here so all cards align.
 */
const CardHeader: React.FC<CardHeaderProps> = ({
  title,
  path,
  badges,
  actions,
  children,
  wrapBadges = false,
  minTitleWidth = false,
  wrapActions = false,
  className,
}) => {
  return (
    <div className={clsx('p-5 pb-4', className)}>
      <div className="flex items-start justify-between mb-4">
        <div className="flex-1 min-w-0">
          <div className={clsx('flex items-center gap-2 mb-3', wrapBadges && 'flex-wrap')}>
            <h3
              className={clsx(
                'text-lg font-bold text-gray-900 dark:text-white truncate',
                minTitleWidth && 'min-w-[120px]',
              )}
            >
              {title}
            </h3>
            {badges}
          </div>
          {path && (
            <code className="text-xs text-gray-600 dark:text-gray-300 bg-gray-50 dark:bg-gray-800/50 px-2 py-1 rounded font-mono">
              {path}
            </code>
          )}
        </div>
        {actions &&
          (wrapActions ? (
            <div className="flex items-center gap-1 flex-shrink-0">{actions}</div>
          ) : (
            actions
          ))}
      </div>
      {children}
    </div>
  );
};

export default CardHeader;
