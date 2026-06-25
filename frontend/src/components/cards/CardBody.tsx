import React from 'react';
import clsx from 'clsx';

interface CardBodyProps {
  /** Line-clamped description. Falls back to a muted placeholder when empty. */
  description?: string | null;
  /** Placeholder shown when description is empty. Pass null to render nothing. */
  emptyText?: string | null;
  /** Number of lines before truncation. */
  clamp?: 2 | 3;
  /** Extra content under the description (ANS bar, tags, target agents, etc.). */
  children?: React.ReactNode;
  /**
   * When true, render the description + children directly without the outer
   * `px-5 pb-4` padding wrapper. Used when the body lives inside CardHeader's
   * padded block (main keeps description + tags in the header block).
   */
  unwrapped?: boolean;
  className?: string;
}

/**
 * The card's main content region: a line-clamped description followed by an
 * arbitrary slot for entity-specific rows (tags, trust bars, attribute lists).
 *
 * Pads horizontally to align with the header/footer. Cards that need a stats
 * grid render <CardStatsRow> as a child or sibling.
 */
const CardBody: React.FC<CardBodyProps> = ({
  description,
  emptyText = 'No description available',
  clamp = 2,
  children,
  unwrapped = false,
  className,
}) => {
  const text = description || emptyText;

  const content = (
    <>
      {text && (
        <p
          className={clsx(
            'text-gray-600 dark:text-gray-300 text-sm leading-relaxed mb-4',
            clamp === 2 ? 'line-clamp-2' : 'line-clamp-3',
          )}
        >
          {text}
        </p>
      )}
      {children}
    </>
  );

  if (unwrapped) {
    return content;
  }

  return <div className={clsx('px-5 pb-4', className)}>{content}</div>;
};

export default CardBody;
