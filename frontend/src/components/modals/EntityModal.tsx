import React from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import clsx from 'clsx';
import useEscapeKey from '../../hooks/useEscapeKey';

export type ModalMaxWidth = 'sm' | 'md' | 'lg' | 'xl' | '2xl' | '3xl' | '4xl';

const MAX_WIDTH_CLASSES: Record<ModalMaxWidth, string> = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  '2xl': 'max-w-2xl',
  '3xl': 'max-w-3xl',
  '4xl': 'max-w-4xl',
};

interface EntityModalProps {
  isOpen: boolean;
  onClose: () => void;
  /**
   * Header title. A string renders as the standard heading; pass a node to
   * supply a richer header (e.g. name + badge + path) for entity modals that
   * need more than a plain title.
   */
  title: React.ReactNode;
  /** Right-aligned header content placed before the close button (e.g. CopyButton). */
  headerActions?: React.ReactNode;
  loading?: boolean;
  error?: string | null;
  maxWidth?: ModalMaxWidth;
  /**
   * Tailwind z-index class for the backdrop. Default ``z-50`` suits a top-level
   * modal; pass ``z-[60]`` when stacking on top of another open modal.
   */
  zIndexClass?: string;
  /**
   * Body layout. ``padded`` (default) keeps the original DetailsModal feel
   * (p-6, content flows, max-h 80vh). ``flush`` gives a header/scroll-body
   * column (border-separated header, independently scrolling body) for the
   * richer standalone modals.
   */
  layout?: 'padded' | 'flush';
  children: React.ReactNode;
}

/**
 * Composable modal shell shared by every entity detail modal.
 *
 * Owns the backdrop, blur, escape handling, max-width, dark mode, and the
 * loading/error states that DetailsModal provided — plus a header-actions slot
 * and a node-capable title so the previously-standalone modals (virtual server,
 * custom entity, skill SKILL.md viewer) can drop their hand-rolled skeletons.
 *
 * DetailsModal re-exports this with a string-title default for back-compat.
 */
const EntityModal: React.FC<EntityModalProps> = ({
  isOpen,
  onClose,
  title,
  headerActions,
  loading = false,
  error = null,
  maxWidth = '4xl',
  zIndexClass = 'z-50',
  layout = 'padded',
  children,
}) => {
  useEscapeKey(onClose, isOpen);

  if (!isOpen) {
    return null;
  }

  const flush = layout === 'flush';

  const titleNode =
    typeof title === 'string' ? (
      <h3 className="text-lg font-semibold text-fg truncate">
        {title}
      </h3>
    ) : (
      title
    );

  return (
    <div
      className={clsx(
        'fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center',
        zIndexClass,
      )}
    >
      <div
        className={clsx(
          'bg-surface-overlay border border-line rounded-xl shadow-xl w-full mx-4',
          MAX_WIDTH_CLASSES[maxWidth],
          flush
            ? 'max-h-[85vh] flex flex-col'
            : 'p-6 max-h-[80vh] overflow-auto',
        )}
        role="dialog"
        aria-modal="true"
      >
        {/* Header */}
        <div
          className={clsx(
            'flex items-center justify-between',
            flush
              ? 'px-6 py-4 border-b border-line flex-shrink-0'
              : 'mb-4',
          )}
        >
          {titleNode}
          <div className="flex items-center gap-2 flex-shrink-0">
            {headerActions}
            <button
              onClick={onClose}
              className="btn-icon"
              aria-label="Close"
            >
              <XMarkIcon className="h-5 w-5" />
            </button>
          </div>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className="flex flex-col items-center gap-3">
              <div className="animate-spin rounded-full h-10 w-10 border-2 border-line border-t-accent"></div>
              <p className="text-sm text-fg-muted">Loading details...</p>
            </div>
          </div>
        )}

        {/* Error State */}
        {!loading && error && (
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 mb-4">
            <h4 className="font-medium text-red-900 dark:text-red-100 mb-1">
              Error Loading Details
            </h4>
            <p className="text-sm text-red-800 dark:text-red-200">{error}</p>
          </div>
        )}

        {/* Content */}
        {!loading &&
          !error &&
          (flush ? (
            <div className="flex-1 overflow-y-auto p-6">{children}</div>
          ) : (
            children
          ))}
      </div>
    </div>
  );
};

export default EntityModal;
