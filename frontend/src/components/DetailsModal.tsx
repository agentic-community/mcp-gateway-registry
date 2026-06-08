import React from 'react';
import useEscapeKey from '../hooks/useEscapeKey';

interface DetailsModalProps {
  title: string;
  isOpen: boolean;
  onClose: () => void;
  loading?: boolean;
  error?: string | null;
  children: React.ReactNode;
  maxWidth?: 'sm' | 'md' | 'lg' | 'xl' | '2xl' | '3xl' | '4xl';
  /**
   * Tailwind z-index class for the backdrop. Default ``z-50`` is fine
   * for a top-level page modal. Pass ``z-[60]`` (or higher) when the
   * modal is rendered on top of another open modal — e.g. a duplicate
   * check that pre-flights from inside an open registration form.
   */
  zIndexClass?: string;
}

const MAX_WIDTH_CLASSES = {
  sm: 'max-w-sm',
  md: 'max-w-md',
  lg: 'max-w-lg',
  xl: 'max-w-xl',
  '2xl': 'max-w-2xl',
  '3xl': 'max-w-3xl',
  '4xl': 'max-w-4xl',
};

/**
 * Shared DetailsModal component with loading and error states.
 *
 * Features:
 * - Backdrop with blur effect
 * - Escape key handler
 * - Configurable max width
 * - Built-in loading spinner
 * - Built-in error display
 * - Dark mode support
 *
 * Usage:
 * ```tsx
 * <DetailsModal
 *   title="Server Details"
 *   isOpen={isOpen}
 *   onClose={handleClose}
 *   loading={loading}
 *   error={error}
 *   maxWidth="4xl"
 * >
 *   <YourContent />
 * </DetailsModal>
 * ```
 */
const DetailsModal: React.FC<DetailsModalProps> = ({
  title,
  isOpen,
  onClose,
  loading = false,
  error = null,
  children,
  maxWidth = '4xl',
  zIndexClass = 'z-50',
}) => {
  useEscapeKey(onClose, isOpen);

  if (!isOpen) {
    return null;
  }

  return (
    <div className={`fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center ${zIndexClass}`}>
      <div
        className={`bg-white dark:bg-gray-800 rounded-xl p-6 ${MAX_WIDTH_CLASSES[maxWidth]} w-full mx-4 max-h-[80vh] overflow-auto`}
      >
        {/* Header */}
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            {title}
          </h3>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 transition-colors"
            aria-label="Close"
          >
            ✕
          </button>
        </div>

        {/* Loading State */}
        {loading && (
          <div className="flex items-center justify-center py-12">
            <div className="flex flex-col items-center gap-3">
              <div className="animate-spin rounded-full h-10 w-10 border-b-2 border-blue-600 dark:border-blue-400"></div>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Loading details...
              </p>
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
        {!loading && !error && children}
      </div>
    </div>
  );
};

export default DetailsModal;
