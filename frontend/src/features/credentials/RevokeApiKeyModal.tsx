import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Button } from '@/components/ui';
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline';
import { useRevokeApiKey } from './hooks';
import { formatRelativeTime } from '@/lib/format';
import type { ApiKeySummary } from '@/api/types';

// ============================================================================
// Component
// ============================================================================

export interface RevokeApiKeyModalProps {
  open: boolean;
  apiKey: ApiKeySummary;
  onClose: () => void;
  onSuccess: () => void;
}

export function RevokeApiKeyModal({
  open,
  apiKey,
  onClose,
  onSuccess,
}: RevokeApiKeyModalProps) {
  const { revokeApiKey, isRevoking, error } = useRevokeApiKey();

  const handleRevoke = async () => {
    try {
      await revokeApiKey(apiKey.key_id, apiKey.agent_id);
      onSuccess();
    } catch {
      // Error handled by hook
    }
  };

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Revoke API Key"
      size="md"
    >
      <div className="space-y-4">
        <div className="flex items-start space-x-3">
          <div className="flex-shrink-0">
            <div className="w-10 h-10 rounded-full bg-red-100 dark:bg-red-900/30 flex items-center justify-center">
              <ExclamationTriangleIcon className="w-5 h-5 text-red-600 dark:text-red-400" />
            </div>
          </div>
          <div>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              You are about to permanently revoke this API key:
            </p>
            <p className="mt-1 text-sm font-mono text-gray-900 dark:text-gray-100">
              {apiKey.key_id}
            </p>
          </div>
        </div>

        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-3">
          <h4 className="text-sm font-medium text-red-800 dark:text-red-300">
            Warning: This action cannot be undone
          </h4>
          <ul className="mt-2 text-sm text-red-700 dark:text-red-400 list-disc list-inside space-y-1">
            <li>The API key will be immediately invalidated</li>
            <li>Any applications using this key will lose access</li>
            <li>You will need to create a new key if access is needed</li>
          </ul>
        </div>

        {/* Key details */}
        <div className="bg-gray-50 dark:bg-gray-900 rounded-md p-3 space-y-2">
          <div className="text-sm">
            <span className="text-gray-500 dark:text-gray-400">Key ID:</span>{' '}
            <code className="text-gray-900 dark:text-gray-100">{apiKey.key_id}</code>
          </div>
          <div className="text-sm">
            <span className="text-gray-500 dark:text-gray-400">Created:</span>{' '}
            <span className="text-gray-900 dark:text-gray-100">
              {formatRelativeTime(apiKey.created_at)}
            </span>
          </div>
          {apiKey.scopes && apiKey.scopes.length > 0 && (
            <div className="text-sm">
              <span className="text-gray-500 dark:text-gray-400">Scopes:</span>{' '}
              <span className="text-gray-900 dark:text-gray-100">
                {apiKey.scopes.join(', ')}
              </span>
            </div>
          )}
          {apiKey.last_used_at && (
            <div className="text-sm">
              <span className="text-gray-500 dark:text-gray-400">Last Used:</span>{' '}
              <span className="text-gray-900 dark:text-gray-100">
                {formatRelativeTime(apiKey.last_used_at)}
              </span>
            </div>
          )}
        </div>

        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
            <p className="text-sm text-red-600 dark:text-red-400">
              Failed to revoke API key. Please try again.
            </p>
          </div>
        )}
      </div>

      <ModalFooter>
        <Button
          type="button"
          variant="secondary"
          onClick={onClose}
          disabled={isRevoking}
        >
          Cancel
        </Button>
        <Button
          type="button"
          variant="danger"
          onClick={handleRevoke}
          loading={isRevoking}
        >
          Revoke API Key
        </Button>
      </ModalFooter>
    </Modal>
  );
}
