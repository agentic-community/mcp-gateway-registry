import { useState } from 'react';
import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Button, Input } from '@/components/ui';
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline';
import { useRevokeEnforceAIAgent } from './hooks';
import type { EnforceAIAgent } from '@/api/types';

// ============================================================================
// Component
// ============================================================================

export interface RevokeAgentModalProps {
  open: boolean;
  agent: EnforceAIAgent;
  onClose: () => void;
  onSuccess: () => void;
}

export function RevokeAgentModal({
  open,
  agent,
  onClose,
  onSuccess,
}: RevokeAgentModalProps) {
  const { revokeAgent, isRevoking, error } = useRevokeEnforceAIAgent();
  const [confirmInput, setConfirmInput] = useState('');

  const shortId = agent.agent_id.slice(0, 8);
  const isConfirmValid = confirmInput === shortId;

  const handleClose = () => {
    setConfirmInput('');
    onClose();
  };

  const handleRevoke = async () => {
    if (!isConfirmValid) return;

    try {
      await revokeAgent(agent.agent_id);
      setConfirmInput('');
      onSuccess();
    } catch {
      // Error is handled by the hook
    }
  };

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Revoke Agent"
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
              You are about to <strong>permanently revoke</strong> this agent:
            </p>
            <p className="mt-1 text-sm font-mono text-gray-900 dark:text-gray-100">
              {agent.alias ? `${agent.alias} (${agent.agent_id})` : agent.agent_id}
            </p>
          </div>
        </div>

        <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-3">
          <h4 className="text-sm font-medium text-red-800 dark:text-red-300">
            This action cannot be undone
          </h4>
          <ul className="mt-2 text-sm text-red-700 dark:text-red-400 list-disc list-inside space-y-1">
            <li>All tokens for this agent will immediately become invalid</li>
            <li>All API keys for this agent will be revoked</li>
            <li>The agent cannot be re-activated</li>
            <li>Any services using this agent will lose access</li>
          </ul>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            To confirm, type <span className="font-mono font-bold">{shortId}</span>
          </label>
          <Input
            value={confirmInput}
            onChange={(e) => setConfirmInput(e.target.value)}
            placeholder={shortId}
            autoComplete="off"
          />
        </div>

        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
            <p className="text-sm text-red-600 dark:text-red-400">
              Failed to revoke agent. Please try again.
            </p>
          </div>
        )}
      </div>

      <ModalFooter>
        <Button
          type="button"
          variant="secondary"
          onClick={handleClose}
          disabled={isRevoking}
        >
          Cancel
        </Button>
        <Button
          type="button"
          variant="danger"
          onClick={handleRevoke}
          disabled={!isConfirmValid}
          loading={isRevoking}
        >
          Revoke Agent
        </Button>
      </ModalFooter>
    </Modal>
  );
}
