import React from 'react';
import DetailsModal from './DetailsModal';
import Button from './Button';
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
  /** Label shown on the confirm button while `isLoading` is true. Defaults to "Removing..." for backward compatibility with the original delete-only callsites. */
  loadingLabel?: string;
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
  loadingLabel = 'Removing...',
  isDestructive = false,
  isLoading = false,
}) => {
  return (
    <DetailsModal title={title} isOpen={isOpen} onClose={onClose} maxWidth="sm">
      <div className="flex flex-col items-center text-center space-y-4">
        <div className={`p-3 rounded-full ${
          isDestructive
            ? 'bg-danger-100 dark:bg-danger-900/30'
            : 'bg-warning-100 dark:bg-warning-900/30'
        }`}>
          <ExclamationTriangleIcon className={`h-6 w-6 ${
            isDestructive
              ? 'text-danger-600 dark:text-danger-400'
              : 'text-warning-600 dark:text-warning-400'
          }`} />
        </div>

        <p className="text-sm text-gray-600 dark:text-gray-300">
          {message}
        </p>

        <div className="flex justify-center space-x-3 pt-2 w-full">
          <Button variant="secondary" onClick={onClose} disabled={isLoading}>
            {cancelLabel}
          </Button>
          <Button
            variant={isDestructive ? 'danger' : 'primary'}
            onClick={onConfirm}
            disabled={isLoading}
          >
            {isLoading ? loadingLabel : confirmLabel}
          </Button>
        </div>
      </div>
    </DetailsModal>
  );
};


export default ConfirmModal;
