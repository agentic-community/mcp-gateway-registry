import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Button } from '@/components/ui';
import { ExclamationTriangleIcon, ClockIcon } from '@heroicons/react/24/outline';
import { useRevokeAllTokens } from './hooks';
import { formatRelativeTime } from '@/lib/format';
import type { EnforceAIAgent } from '@/api/types';

// ============================================================================
// Component
// ============================================================================

export interface RevokeAllTokensModalProps {
  open: boolean;
  agent: EnforceAIAgent;
  onClose: () => void;
  onSuccess: () => void;
}

export function RevokeAllTokensModal({
  open,
  agent,
  onClose,
  onSuccess,
}: RevokeAllTokensModalProps) {
  const { revokeAllTokens, isRevoking, error } = useRevokeAllTokens();

  const handleRevoke = async () => {
    try {
      await revokeAllTokens(agent.agent_id);
      onSuccess();
    } catch {
      // Error is handled by the hook
    }
  };

  const hasExistingTokensValidAfter = agent.tokens_valid_after !== null;

  return (
    <Modal
      open={open}
      onClose={onClose}
      title="Revoke All Tokens"
      size="md"
    >
      <div className="space-y-4">
        <div className="flex items-start space-x-3">
          <div className="flex-shrink-0">
            <div className="w-10 h-10 rounded-full bg-yellow-100 dark:bg-yellow-900/30 flex items-center justify-center">
              <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 dark:text-yellow-400" />
            </div>
          </div>
          <div>
            <p className="text-sm text-gray-700 dark:text-gray-300">
              You are about to revoke all existing tokens for agent:
            </p>
            <p className="mt-1 text-sm font-mono text-gray-900 dark:text-gray-100">
              {agent.alias ? `${agent.alias} (${agent.agent_id.slice(0, 8)}...)` : agent.agent_id}
            </p>
          </div>
        </div>

        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md p-3">
          <h4 className="text-sm font-medium text-yellow-800 dark:text-yellow-300">
            What happens when you revoke all tokens?
          </h4>
          <ul className="mt-2 text-sm text-yellow-700 dark:text-yellow-400 list-disc list-inside space-y-1">
            <li>Sets <code className="font-mono text-xs">tokens_valid_after</code> to current time</li>
            <li>All gateway tokens issued before this time become invalid</li>
            <li>API keys remain valid (they are independent of this setting)</li>
            <li>New tokens minted after this action will work normally</li>
          </ul>
        </div>

        {hasExistingTokensValidAfter && (
          <div className="flex items-center space-x-2 p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md">
            <ClockIcon className="w-5 h-5 text-blue-500" />
            <div className="text-sm text-blue-700 dark:text-blue-400">
              <strong>Current tokens_valid_after:</strong>{' '}
              {formatRelativeTime(agent.tokens_valid_after!)}
              <span className="block text-xs mt-0.5 opacity-75">
                ({new Date(agent.tokens_valid_after!).toISOString()})
              </span>
            </div>
          </div>
        )}

        <p className="text-sm text-gray-600 dark:text-gray-400">
          This is useful when you suspect a token has been compromised or when
          rotating credentials. The agent remains active and can issue new tokens.
        </p>

        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
            <p className="text-sm text-red-600 dark:text-red-400">
              Failed to revoke tokens. Please try again.
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
          Revoke All Tokens
        </Button>
      </ModalFooter>
    </Modal>
  );
}
