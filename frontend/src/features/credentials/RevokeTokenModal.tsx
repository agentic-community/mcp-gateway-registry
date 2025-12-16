import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Button, Input, Textarea } from '@/components/ui';
import {
  NoSymbolIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
} from '@heroicons/react/24/outline';
import { useRevokeToken, decodeToken } from './hooks';
import { formatRelativeTime } from '@/lib/format';
import type { TokenRevocationRecord } from '@/api/types';

// ============================================================================
// Form Schemas
// ============================================================================

const revokeByJtiSchema = z.object({
  agent_id: z.string().min(1, 'Agent ID is required'),
  jti: z.string().min(1, 'JTI is required'),
  reason: z.string().optional(),
});

const revokeByTokenSchema = z.object({
  gateway_token: z.string().min(1, 'Token is required'),
  reason: z.string().optional(),
});

type RevokeByJtiFormData = z.infer<typeof revokeByJtiSchema>;
type RevokeByTokenFormData = z.infer<typeof revokeByTokenSchema>;

// ============================================================================
// Component
// ============================================================================

export interface RevokeTokenModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
  defaultAgentId?: string;
}

type RevokeMode = 'jti' | 'token';

export function RevokeTokenModal({
  open,
  onClose,
  onSuccess,
  defaultAgentId,
}: RevokeTokenModalProps) {
  const { revokeToken, isRevoking, error } = useRevokeToken();
  const [mode, setMode] = useState<RevokeMode>('jti');
  const [revocationRecord, setRevocationRecord] = useState<TokenRevocationRecord | null>(null);

  const jtiForm = useForm<RevokeByJtiFormData>({
    resolver: zodResolver(revokeByJtiSchema),
    defaultValues: {
      agent_id: defaultAgentId || '',
      jti: '',
      reason: '',
    },
  });

  const tokenForm = useForm<RevokeByTokenFormData>({
    resolver: zodResolver(revokeByTokenSchema),
    defaultValues: {
      gateway_token: '',
      reason: '',
    },
  });

  // Extract JTI from pasted token for preview
  const pastedToken = tokenForm.watch('gateway_token');
  const decodedToken = pastedToken ? decodeToken(pastedToken) : null;

  const onSubmitJti = async (data: RevokeByJtiFormData) => {
    try {
      const result = await revokeToken({
        agent_id: data.agent_id,
        jti: data.jti,
        reason: data.reason || undefined,
      });
      setRevocationRecord(result);
    } catch {
      // Error handled by hook
    }
  };

  const onSubmitToken = async (data: RevokeByTokenFormData) => {
    try {
      const result = await revokeToken({
        gateway_token: data.gateway_token,
        reason: data.reason || undefined,
      });
      // Clear the token from form immediately after submission (security)
      tokenForm.setValue('gateway_token', '');
      setRevocationRecord(result);
    } catch {
      // Error handled by hook
    }
  };

  const handleClose = () => {
    jtiForm.reset();
    tokenForm.reset();
    setRevocationRecord(null);
    setMode('jti');
    onClose();
  };

  const handleDone = () => {
    jtiForm.reset();
    tokenForm.reset();
    setRevocationRecord(null);
    setMode('jti');
    onSuccess();
  };

  // If token was revoked, show the result
  if (revocationRecord) {
    return (
      <Modal
        open={open}
        onClose={handleClose}
        title="Token Revoked"
        size="md"
      >
        <div className="space-y-4">
          {/* Success message */}
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <div className="w-10 h-10 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                <CheckCircleIcon className="w-5 h-5 text-green-600 dark:text-green-400" />
              </div>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                Token revoked successfully
              </p>
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                The token can no longer be used for authentication.
              </p>
            </div>
          </div>

          {/* Revocation details */}
          <div className="bg-gray-50 dark:bg-gray-900 rounded-md p-4 space-y-2">
            <div>
              <span className="text-xs font-medium text-gray-500 dark:text-gray-400">JTI:</span>
              <code className="ml-2 text-sm font-mono text-gray-900 dark:text-gray-100">
                {revocationRecord.jti}
              </code>
            </div>
            <div>
              <span className="text-xs font-medium text-gray-500 dark:text-gray-400">Agent ID:</span>
              <code className="ml-2 text-sm font-mono text-gray-900 dark:text-gray-100">
                {revocationRecord.agent_id}
              </code>
            </div>
            <div>
              <span className="text-xs font-medium text-gray-500 dark:text-gray-400">Revoked At:</span>
              <span className="ml-2 text-sm text-gray-900 dark:text-gray-100">
                {formatRelativeTime(revocationRecord.revoked_at)}
              </span>
            </div>
            {revocationRecord.reason && (
              <div>
                <span className="text-xs font-medium text-gray-500 dark:text-gray-400">Reason:</span>
                <span className="ml-2 text-sm text-gray-900 dark:text-gray-100">
                  {revocationRecord.reason}
                </span>
              </div>
            )}
          </div>
        </div>

        <ModalFooter>
          <Button type="button" variant="primary" onClick={handleDone}>
            Done
          </Button>
        </ModalFooter>
      </Modal>
    );
  }

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Revoke Gateway Token"
      size="md"
    >
      <div className="space-y-4">
        {/* Warning */}
        <div className="flex items-start space-x-3 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md">
          <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
          <div>
            <p className="text-sm text-yellow-800 dark:text-yellow-300">
              Revoking a token immediately invalidates it. Any requests using this token will be rejected.
            </p>
          </div>
        </div>

        {/* Mode selector */}
        <div className="flex space-x-2">
          <Button
            type="button"
            variant={mode === 'jti' ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => setMode('jti')}
          >
            By Agent ID + JTI
          </Button>
          <Button
            type="button"
            variant={mode === 'token' ? 'primary' : 'secondary'}
            size="sm"
            onClick={() => setMode('token')}
          >
            By Pasting Token
          </Button>
        </div>

        {/* JTI mode form */}
        {mode === 'jti' && (
          <form onSubmit={jtiForm.handleSubmit(onSubmitJti)} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Agent ID <span className="text-red-500">*</span>
              </label>
              <Input
                {...jtiForm.register('agent_id')}
                placeholder="UUID of the agent"
                error={jtiForm.formState.errors.agent_id?.message}
              />
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                JTI (JWT ID) <span className="text-red-500">*</span>
              </label>
              <Input
                {...jtiForm.register('jti')}
                placeholder="Token's jti claim"
                error={jtiForm.formState.errors.jti?.message}
              />
              <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                The jti claim from the token's payload.
              </p>
            </div>

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Reason (optional)
              </label>
              <Input
                {...jtiForm.register('reason')}
                placeholder="Why is this token being revoked?"
              />
            </div>

            {error && (
              <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
                <p className="text-sm text-red-600 dark:text-red-400">
                  Failed to revoke token. Please verify the agent ID and JTI.
                </p>
              </div>
            )}

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
                type="submit"
                variant="danger"
                loading={isRevoking}
                leftIcon={<NoSymbolIcon className="w-4 h-4" />}
              >
                Revoke Token
              </Button>
            </ModalFooter>
          </form>
        )}

        {/* Token paste mode form */}
        {mode === 'token' && (
          <form onSubmit={tokenForm.handleSubmit(onSubmitToken)} className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Gateway Token <span className="text-red-500">*</span>
              </label>
              <Textarea
                {...tokenForm.register('gateway_token')}
                placeholder="Paste the full JWT token here"
                rows={4}
                className="font-mono text-xs"
                error={tokenForm.formState.errors.gateway_token?.message}
              />
              <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                The token will be used to extract agent_id and jti, then cleared.
              </p>
            </div>

            {/* Token preview */}
            {decodedToken && (
              <div className="bg-gray-50 dark:bg-gray-900 rounded-md p-3 space-y-1">
                <p className="text-xs font-medium text-gray-500 dark:text-gray-400">Token Preview:</p>
                <div className="text-xs">
                  <span className="text-gray-500 dark:text-gray-400">JTI:</span>{' '}
                  <code className="text-gray-900 dark:text-gray-100">{decodedToken.payload.jti}</code>
                </div>
                <div className="text-xs">
                  <span className="text-gray-500 dark:text-gray-400">Subject:</span>{' '}
                  <code className="text-gray-900 dark:text-gray-100">{decodedToken.payload.sub}</code>
                </div>
                <div className="text-xs">
                  <span className="text-gray-500 dark:text-gray-400">Expires:</span>{' '}
                  <span className="text-gray-900 dark:text-gray-100">
                    {new Date(decodedToken.payload.exp * 1000).toLocaleString()}
                  </span>
                </div>
              </div>
            )}

            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Reason (optional)
              </label>
              <Input
                {...tokenForm.register('reason')}
                placeholder="Why is this token being revoked?"
              />
            </div>

            {error && (
              <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
                <p className="text-sm text-red-600 dark:text-red-400">
                  Failed to revoke token. The token may be invalid or already revoked.
                </p>
              </div>
            )}

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
                type="submit"
                variant="danger"
                loading={isRevoking}
                leftIcon={<NoSymbolIcon className="w-4 h-4" />}
              >
                Revoke Token
              </Button>
            </ModalFooter>
          </form>
        )}
      </div>
    </Modal>
  );
}
