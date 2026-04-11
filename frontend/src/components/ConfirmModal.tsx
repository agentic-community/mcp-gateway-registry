import React from 'react';
import DetailsModal from './DetailsModal';
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline';


/**
 * Props for the ConfirmModal component.
 */
interface ConfirmModalProps {
  isOpen: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  isDestructive?: boolean;
  isLoading?: boolean;
}


/**
 * A styled confirmation modal that replaces window.confirm().
 *
 * Renders a centered dialog with a warning icon, message, and
 * Cancel / Confirm action buttons. Supports destructive (red)
 * and normal (purple) confirm button styles.
 */
const ConfirmModal: React.FC<ConfirmModalProps> = ({
  isOpen,
  onClose,
  onConfirm,
  title,
  message,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  isDestructive = false,
  isLoading = false,
}) => {
  return (
    <DetailsModal title={title} isOpen={isOpen} onClose={onClose} maxWidth="sm">
      <div className="flex flex-col items-center text-center space-y-4">
        <div className={`p-3 rounded-full ${
          isDestructive
            ? 'bg-red-100 dark:bg-red-900/30'
            : 'bg-yellow-100 dark:bg-yellow-900/30'
        }`}>
          <ExclamationTriangleIcon className={`h-6 w-6 ${
            isDestructive
              ? 'text-red-600 dark:text-red-400'
              : 'text-yellow-600 dark:text-yellow-400'
          }`} />
        </div>

        <p className="text-sm text-gray-600 dark:text-gray-300">
          {message}
        </p>

        <div className="flex justify-center space-x-3 pt-2 w-full">
          <button
            type="button"
            onClick={onClose}
            disabled={isLoading}
            className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300
                       bg-white dark:bg-gray-800 border border-gray-300 dark:border-gray-600
                       rounded-lg hover:bg-gray-50 dark:hover:bg-gray-700
                       disabled:opacity-50 transition-colors"
          >
            {cancelLabel}
          </button>
          <button
            type="button"
            onClick={onConfirm}
            disabled={isLoading}
            className={`px-4 py-2 text-sm font-medium text-white rounded-lg
                       disabled:opacity-50 disabled:cursor-not-allowed transition-colors ${
              isDestructive
                ? 'bg-red-600 hover:bg-red-700'
                : 'bg-purple-600 hover:bg-purple-700'
            }`}
          >
            {isLoading ? 'Removing...' : confirmLabel}
          </button>
        </div>
      </div>
    </DetailsModal>
  );
};


export default ConfirmModal;
