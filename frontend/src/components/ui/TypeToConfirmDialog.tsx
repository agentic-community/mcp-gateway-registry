/**
 * TypeToConfirmDialog - Confirmation dialog that requires typing a value to confirm
 */

import { useState } from 'react';
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline';
import { cn } from '../../lib/cn';
import { Button } from './Button';
import { Input } from './Input';
import { Modal, ModalFooter } from './Modal';

export interface TypeToConfirmDialogProps {
  open: boolean;
  onClose: () => void;
  onConfirm: (reason?: string) => void;
  title: string;
  message: string;
  confirmValue: string;
  confirmLabel?: string;
  cancelLabel?: string;
  loading?: boolean;
  requireReason?: boolean;
  reasonLabel?: string;
  reasonPlaceholder?: string;
}

export function TypeToConfirmDialog({
  open,
  onClose,
  onConfirm,
  title,
  message,
  confirmValue,
  confirmLabel = 'Confirm',
  cancelLabel = 'Cancel',
  loading = false,
  requireReason = false,
  reasonLabel = 'Reason (required)',
  reasonPlaceholder = 'Enter reason for this action...',
}: TypeToConfirmDialogProps) {
  const [typedValue, setTypedValue] = useState('');
  const [reason, setReason] = useState('');

  const isValid =
    typedValue === confirmValue && (!requireReason || reason.trim().length > 0);

  const handleConfirm = () => {
    if (isValid) {
      onConfirm(requireReason ? reason.trim() : undefined);
    }
  };

  const handleClose = () => {
    setTypedValue('');
    setReason('');
    onClose();
  };

  return (
    <Modal open={open} onClose={handleClose} size="md" showCloseButton={false}>
      <div className="flex items-start gap-4">
        <div
          className={cn(
            'flex-shrink-0 flex items-center justify-center',
            'w-10 h-10 rounded-full',
            'bg-red-100 text-red-600',
            'dark:bg-red-900/30 dark:text-red-400'
          )}
        >
          <ExclamationTriangleIcon className="h-6 w-6" aria-hidden="true" />
        </div>
        <div className="flex-1">
          <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">{title}</h3>
          <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">{message}</p>
        </div>
      </div>

      <div className="mt-6 space-y-4">
        <div>
          <label
            htmlFor="confirm-input"
            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
          >
            Type <code className="text-red-600 dark:text-red-400">{confirmValue}</code> to
            confirm
          </label>
          <Input
            id="confirm-input"
            value={typedValue}
            onChange={(e) => setTypedValue(e.target.value)}
            placeholder={confirmValue}
            autoComplete="off"
            autoFocus
          />
        </div>

        {requireReason && (
          <div>
            <label
              htmlFor="reason-input"
              className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
            >
              {reasonLabel}
            </label>
            <Input
              id="reason-input"
              value={reason}
              onChange={(e) => setReason(e.target.value)}
              placeholder={reasonPlaceholder}
            />
          </div>
        )}
      </div>

      <ModalFooter>
        <Button variant="secondary" onClick={handleClose} disabled={loading}>
          {cancelLabel}
        </Button>
        <Button variant="danger" onClick={handleConfirm} loading={loading} disabled={!isValid}>
          {confirmLabel}
        </Button>
      </ModalFooter>
    </Modal>
  );
}
