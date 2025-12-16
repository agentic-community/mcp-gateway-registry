import { ExclamationTriangleIcon } from '@heroicons/react/24/outline';
import { cn } from '../../lib/cn';
import { Button } from './Button';
import { Modal, ModalFooter } from './Modal';

export interface ConfirmDialogProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => void;
  title: string;
  message: string;
  confirmLabel?: string;
  cancelLabel?: string;
  variant?: 'danger' | 'warning' | 'info';
  loading?: boolean;
}

const iconVariants = {
  danger: cn(
    'bg-red-100 text-red-600',
    'dark:bg-red-900/30 dark:text-red-400'
  ),
  warning: cn(
    'bg-yellow-100 text-yellow-600',
    'dark:bg-yellow-900/30 dark:text-yellow-400'
  ),
  info: cn(
    'bg-blue-100 text-blue-600',
    'dark:bg-blue-900/30 dark:text-blue-400'
  ),
};

export function ConfirmDialog({
  open,
  onClose,
  onConfirm,
  title,
  message,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  variant = 'danger',
  loading = false,
}: ConfirmDialogProps) {
  const handleConfirm = () => {
    onConfirm();
  };

  return (
    <Modal open={open} onClose={onClose} size="sm" showCloseButton={false}>
      <div className="flex items-start gap-4">
        <div
          className={cn(
            'flex-shrink-0 flex items-center justify-center',
            'w-10 h-10 rounded-full',
            iconVariants[variant]
          )}
        >
          <ExclamationTriangleIcon className="h-6 w-6" aria-hidden="true" />
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
            {title}
          </h3>
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
            {message}
          </p>
        </div>
      </div>
      <ModalFooter>
        <Button variant="secondary" onClick={onClose} disabled={loading}>
          {cancelLabel}
        </Button>
        <Button
          variant={variant === 'danger' ? 'danger' : 'primary'}
          onClick={handleConfirm}
          loading={loading}
        >
          {confirmLabel}
        </Button>
      </ModalFooter>
    </Modal>
  );
}
