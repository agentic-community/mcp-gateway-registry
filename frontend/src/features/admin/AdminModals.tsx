/**
 * AdminModals - Modal dialogs for admin cross-user operations
 */

import { useState } from 'react';
import { Modal, ModalFooter, Button, Input, TypeToConfirmDialog } from '@/components/ui';
import { ScopePicker } from '@/features/scopes/ScopePicker';
import type { CreateAgentRequest, EnforceAIAgent } from '@/api/types';

// ============================================================================
// Admin Create Agent Modal
// ============================================================================

export interface AdminCreateAgentModalProps {
  open: boolean;
  onClose: () => void;
  onSubmit: (data: CreateAgentRequest) => Promise<void>;
  targetUserEmail: string;
  targetUserId: string;
  loading?: boolean;
}

export function AdminCreateAgentModal({
  open,
  onClose,
  onSubmit,
  targetUserEmail,
  targetUserId,
  loading = false,
}: AdminCreateAgentModalProps) {
  const [scopes, setScopes] = useState<string[]>([]);
  const [alias, setAlias] = useState('');
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    setError(null);

    if (scopes.length === 0) {
      setError('At least one scope is required');
      return;
    }

    try {
      await onSubmit({
        scopes,
        alias: alias.trim() || null,
      });
      handleClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to create agent');
    }
  };

  const handleClose = () => {
    setScopes([]);
    setAlias('');
    setError(null);
    onClose();
  };

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Create Agent for User"
      description={`Creating agent for ${targetUserEmail}`}
      size="lg"
    >
      <div className="space-y-4">
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md p-3">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            <strong>Admin Action:</strong> You are creating an agent on behalf of user{' '}
            <code className="bg-yellow-100 dark:bg-yellow-900 px-1 rounded">{targetUserId}</code>
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Alias (optional)
          </label>
          <Input
            value={alias}
            onChange={(e) => setAlias(e.target.value)}
            placeholder="e.g., production-agent"
          />
        </div>

        <ScopePicker
          value={scopes}
          onChange={setScopes}
          label="Scopes (required)"
          helperText="Select the scopes this agent should have access to"
          error={scopes.length === 0 ? 'At least one scope is required' : undefined}
        />

        {error && (
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        )}
      </div>

      <ModalFooter>
        <Button variant="secondary" onClick={handleClose} disabled={loading}>
          Cancel
        </Button>
        <Button
          variant="primary"
          onClick={handleSubmit}
          loading={loading}
          disabled={scopes.length === 0}
        >
          Create Agent
        </Button>
      </ModalFooter>
    </Modal>
  );
}

// ============================================================================
// Admin Revoke Agent Modal
// ============================================================================

export interface AdminRevokeAgentModalProps {
  open: boolean;
  onClose: () => void;
  onConfirm: (reason?: string) => Promise<void>;
  agent: EnforceAIAgent | null;
  targetUserEmail: string;
  loading?: boolean;
}

export function AdminRevokeAgentModal({
  open,
  onClose,
  onConfirm,
  agent,
  targetUserEmail,
  loading = false,
}: AdminRevokeAgentModalProps) {
  const [error, setError] = useState<string | null>(null);

  const handleConfirm = async (reason?: string) => {
    setError(null);
    try {
      await onConfirm(reason);
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to revoke agent');
    }
  };

  if (!agent) return null;

  return (
    <TypeToConfirmDialog
      open={open}
      onClose={onClose}
      onConfirm={handleConfirm}
      title="Revoke Agent"
      message={`You are about to revoke the agent ${agent.alias || agent.agent_id} for user ${targetUserEmail}. This action will permanently disable the agent and invalidate all associated tokens. This cannot be undone.`}
      confirmValue={agent.agent_id}
      confirmLabel="Revoke Agent"
      loading={loading}
      requireReason
      reasonLabel="Reason for revocation (required)"
      reasonPlaceholder="e.g., Security concern, user request, policy violation..."
    />
  );
}

// ============================================================================
// Admin Revoke All Tokens Modal
// ============================================================================

export interface AdminRevokeAllTokensModalProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void>;
  agent: EnforceAIAgent | null;
  targetUserEmail: string;
  loading?: boolean;
}

export function AdminRevokeAllTokensModal({
  open,
  onClose,
  onConfirm,
  agent,
  targetUserEmail,
  loading = false,
}: AdminRevokeAllTokensModalProps) {
  const [error, setError] = useState<string | null>(null);

  const handleConfirm = async () => {
    setError(null);
    try {
      await onConfirm();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to revoke tokens');
    }
  };

  if (!agent) return null;

  return (
    <TypeToConfirmDialog
      open={open}
      onClose={onClose}
      onConfirm={handleConfirm}
      title="Revoke All Tokens"
      message={`You are about to revoke all gateway tokens for agent ${agent.alias || agent.agent_id} belonging to user ${targetUserEmail}. This will immediately invalidate all active tokens and require new tokens to be minted.`}
      confirmValue={agent.agent_id}
      confirmLabel="Revoke All Tokens"
      loading={loading}
    />
  );
}

// ============================================================================
// Admin Revoke API Key Modal
// ============================================================================

export interface AdminRevokeApiKeyModalProps {
  open: boolean;
  onClose: () => void;
  onConfirm: () => Promise<void>;
  keyId: string | null;
  targetUserEmail: string;
  loading?: boolean;
}

export function AdminRevokeApiKeyModal({
  open,
  onClose,
  onConfirm,
  keyId,
  targetUserEmail,
  loading = false,
}: AdminRevokeApiKeyModalProps) {
  const [error, setError] = useState<string | null>(null);

  const handleConfirm = async () => {
    setError(null);
    try {
      await onConfirm();
      onClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to revoke API key');
    }
  };

  if (!keyId) return null;

  return (
    <TypeToConfirmDialog
      open={open}
      onClose={onClose}
      onConfirm={handleConfirm}
      title="Revoke API Key"
      message={`You are about to revoke API key ${keyId} for user ${targetUserEmail}. This will immediately invalidate the key and any applications using it will lose access.`}
      confirmValue={keyId}
      confirmLabel="Revoke API Key"
      loading={loading}
    />
  );
}

// ============================================================================
// Admin Revoke Token Modal (by JTI)
// ============================================================================

export interface AdminRevokeTokenModalProps {
  open: boolean;
  onClose: () => void;
  onConfirm: (agentId: string, jti: string, reason?: string) => Promise<void>;
  targetUserEmail: string;
  targetUserId: string;
  loading?: boolean;
}

export function AdminRevokeTokenModal({
  open,
  onClose,
  onConfirm,
  targetUserEmail,
  targetUserId,
  loading = false,
}: AdminRevokeTokenModalProps) {
  const [agentId, setAgentId] = useState('');
  const [jti, setJti] = useState('');
  const [reason, setReason] = useState('');
  const [error, setError] = useState<string | null>(null);

  const handleSubmit = async () => {
    setError(null);

    if (!agentId.trim()) {
      setError('Agent ID is required');
      return;
    }
    if (!jti.trim()) {
      setError('JTI is required');
      return;
    }

    try {
      await onConfirm(agentId.trim(), jti.trim(), reason.trim() || undefined);
      handleClose();
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Failed to revoke token');
    }
  };

  const handleClose = () => {
    setAgentId('');
    setJti('');
    setReason('');
    setError(null);
    onClose();
  };

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Revoke Gateway Token"
      description={`Revoking token for user ${targetUserEmail}`}
      size="md"
    >
      <div className="space-y-4">
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md p-3">
          <p className="text-sm text-yellow-800 dark:text-yellow-200">
            <strong>Admin Action:</strong> You are revoking a token on behalf of user{' '}
            <code className="bg-yellow-100 dark:bg-yellow-900 px-1 rounded">{targetUserId}</code>
          </p>
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Agent ID
          </label>
          <Input
            value={agentId}
            onChange={(e) => setAgentId(e.target.value)}
            placeholder="UUID of the agent"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            JTI (JWT ID)
          </label>
          <Input
            value={jti}
            onChange={(e) => setJti(e.target.value)}
            placeholder="Token JTI to revoke"
          />
        </div>

        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Reason (optional)
          </label>
          <Input
            value={reason}
            onChange={(e) => setReason(e.target.value)}
            placeholder="Reason for revocation..."
          />
        </div>

        {error && (
          <p className="text-sm text-red-600 dark:text-red-400">{error}</p>
        )}
      </div>

      <ModalFooter>
        <Button variant="secondary" onClick={handleClose} disabled={loading}>
          Cancel
        </Button>
        <Button
          variant="danger"
          onClick={handleSubmit}
          loading={loading}
          disabled={!agentId.trim() || !jti.trim()}
        >
          Revoke Token
        </Button>
      </ModalFooter>
    </Modal>
  );
}
