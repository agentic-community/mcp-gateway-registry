import React from 'react';
import clsx from 'clsx';
import { ACCENTS, AccentToken } from '../../theme/accents';

interface TagListProps {
  tags: string[];
  /** Accent token — the default tag pill color follows the card accent. */
  accent?: AccentToken;
  /** Max tags to show before collapsing the rest into a "+N" pill. */
  max?: number;
  /** Prefix each tag (e.g. "#"). All entity cards use a hash for consistency. */
  prefix?: string;
  /** Pill shape. Square `rounded` by default; custom entities use `rounded-full`. */
  rounded?: 'rounded' | 'rounded-full';
  /** Optional per-tag class override (e.g. highlight security-pending tags). */
  tagClassName?: (tag: string) => string | undefined;
  className?: string;
}

/**
 * Renders the first `max` tags as pills plus a "+N" overflow pill. Shared by
 * every card. The default pill color comes from the card accent so tags match
 * the rest of the card; per-tag overrides (tagClassName) handle special states.
 *
 * Returns null when there are no tags so callers don't need a guard.
 */
const TagList: React.FC<TagListProps> = ({
  tags,
  accent = 'neutral',
  max = 3,
  prefix = '',
  rounded = 'rounded',
  tagClassName,
  className,
}) => {
  if (!tags || tags.length === 0) {
    return null;
  }

  const visible = tags.slice(0, max);
  const overflow = tags.length - visible.length;

  return (
    <div className={clsx('flex flex-wrap gap-1.5', className)}>
      {visible.map((tag) => (
        <span
          key={tag}
          className={clsx(
            'px-2 py-1 text-xs font-medium',
            rounded,
            tagClassName?.(tag) ?? ACCENTS[accent].tag,
          )}
        >
          {prefix}
          {tag}
        </span>
      ))}
      {overflow > 0 && (
        <span
          className={clsx(
            'px-2 py-1 text-xs font-medium bg-gray-500/10 text-fg-muted',
            rounded,
          )}
        >
          +{overflow}
        </span>
      )}
    </div>
  );
};

export default TagList;
