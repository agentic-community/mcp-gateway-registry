import React from 'react';
import { ArrowPathIcon } from '@heroicons/react/24/outline';

interface ListStateBoundaryProps {
  isLoading: boolean;
  error: string | null;
  /** True when the (filtered) list has no items. */
  isEmpty: boolean;
  /** Message shown for the empty state. */
  emptyMessage: React.ReactNode;
  /** Extra classes for the empty-state container (e.g. max-width). */
  emptyClassName?: string;
  /** The list/table, rendered only when not loading, not errored, not empty. */
  children: React.ReactNode;
}

/**
 * The loading / error / empty / content state machine shared by the IAM list
 * views (groups, users, m2m, user-groups), which each repeated the same four
 * mutually-exclusive branches. Renders a centered spinner, an error line, an
 * empty message, or the children.
 */
const ListStateBoundary: React.FC<ListStateBoundaryProps> = ({
  isLoading,
  error,
  isEmpty,
  emptyMessage,
  emptyClassName,
  children,
}) => {
  if (isLoading) {
    return (
      <div className="flex justify-center py-12">
        <ArrowPathIcon className="h-6 w-6 text-gray-400 animate-spin" />
      </div>
    );
  }
  if (error) {
    return (
      <div className="text-center py-8 text-red-500 dark:text-red-400 text-sm">{error}</div>
    );
  }
  if (isEmpty) {
    return (
      <div
        className={`text-center py-12 text-fg-subtle ${emptyClassName ?? ''}`}
      >
        {emptyMessage}
      </div>
    );
  }
  return <>{children}</>;
};

export default ListStateBoundary;
