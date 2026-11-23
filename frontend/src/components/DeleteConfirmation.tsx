import React, { useState } from 'react';
import { ArrowPathIcon } from '@heroicons/react/24/outline';

/**
 * Props for the DeleteConfirmation component.
 */
export interface DeleteConfirmationProps {
  entityType: 'server' | 'agent' | 'group' | 'user' | 'm2m' | 'rate-limit';
  entityName: string;
  entityPath: string;
  onConfirm: (path: string) => Promise<void>;
  onCancel: () => void;
}

/**
 * DeleteConfirmation component provides an inline confirmation UI for delete operations.
 *
 * Displays a red-tinted container with warning text, requiring users to type the entity
 * name exactly before the delete button becomes enabled. Shows loading state during
 * API calls and displays error messages on failure.
 *
 * Requirements: 4.1, 4.2, 4.3, 4.4, 4.5, 4.6, 4.7, 4.8
 */
const DeleteConfirmation: React.FC<DeleteConfirmationProps> = ({
  entityType,
  entityName,
  entityPath,
  onConfirm,
  onCancel,
}) => {
  const [typedName, setTypedName] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const isConfirmed = typedName === entityName;

  const handleDelete = async () => {
    if (!isConfirmed || isDeleting) return;

    setIsDeleting(true);
    setError(null);

    try {
      await onConfirm(entityPath);
      onCancel(); // Close on success - parent handles list refresh + toast
    } catch (err: any) {
      setError(
        err.response?.data?.detail ||
        err.response?.data?.reason ||
        `Failed to delete ${entityType}`
      );
    } finally {
      setIsDeleting(false);
    }
  };

  const entityTypeLabels: Record<string, string> = {
    server: 'Server',
    agent: 'Agent',
    group: 'Group',
    user: 'User',
    m2m: 'M2M Account',
    'rate-limit': 'Rate Limit',
  };
  const entityTypeLabel = entityTypeLabels[entityType] || entityType;

  return (
    <div className="p-4 bg-danger-50 dark:bg-danger-900/20 rounded-lg border border-danger-200 dark:border-danger-800">
      <h4 className="text-danger-800 dark:text-danger-200 font-semibold mb-2">
        Delete {entityTypeLabel}
      </h4>
      <p className="text-sm text-danger-700 dark:text-danger-300 mb-2">
        This action is irreversible. This will permanently delete the {entityType}{' '}
        "<strong>{entityName}</strong>" and remove it from the registry.
      </p>
      <p className="text-sm text-danger-700 dark:text-danger-300 mb-3">
        Type <strong>{entityName}</strong> to confirm:
      </p>
      <input
        type="text"
        value={typedName}
        onChange={(e) => setTypedName(e.target.value)}
        className="input mb-3 border-danger-300 dark:border-danger-700"
        placeholder={entityName}
        disabled={isDeleting}
      />
      {error && (
        <p className="text-sm text-danger-600 dark:text-danger-400 mb-3">{error}</p>
      )}
      <div className="flex gap-2 justify-end">
        <button
          onClick={onCancel}
          disabled={isDeleting}
          className="btn-secondary"
        >
          Cancel
        </button>
        <button
          onClick={handleDelete}
          disabled={!isConfirmed || isDeleting}
          className="btn-danger"
        >
          {isDeleting && <ArrowPathIcon className="h-4 w-4 animate-spin" />}
          Delete {entityTypeLabel}
        </button>
      </div>
    </div>
  );
};

export default DeleteConfirmation;
