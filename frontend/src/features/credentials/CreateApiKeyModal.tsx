import { useState } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Button, Input, Checkbox } from '@/components/ui';
import { CopyButton } from '@/components/ui/CopyButton';
import {
  KeyIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ArrowDownTrayIcon,
  ClipboardDocumentIcon,
} from '@heroicons/react/24/outline';
import { useCreateApiKey } from './hooks';
import type { EnforceAIAgent, CreateApiKeyResponse } from '@/api/types';

// ============================================================================
// Form Schema
// ============================================================================

const createApiKeySchema = z.object({
  scopes: z.string().optional(),
  expires_at: z.string().optional(),
});

type CreateApiKeyFormData = z.infer<typeof createApiKeySchema>;

// ============================================================================
// Component
// ============================================================================

export interface CreateApiKeyModalProps {
  open: boolean;
  agent: EnforceAIAgent;
  onClose: () => void;
  onSuccess: () => void;
}

export function CreateApiKeyModal({
  open,
  agent,
  onClose,
  onSuccess,
}: CreateApiKeyModalProps) {
  const { createApiKey, isCreating, error } = useCreateApiKey();
  const [createdKey, setCreatedKey] = useState<CreateApiKeyResponse | null>(null);
  const [acknowledged, setAcknowledged] = useState(false);

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset,
  } = useForm<CreateApiKeyFormData>({
    resolver: zodResolver(createApiKeySchema),
    defaultValues: {
      scopes: '',
      expires_at: '',
    },
  });

  const onSubmit = async (data: CreateApiKeyFormData) => {
    try {
      const scopes = data.scopes
        ? data.scopes.split(',').map((s) => s.trim()).filter(Boolean)
        : undefined;

      const result = await createApiKey(agent.agent_id, {
        scopes: scopes?.length ? scopes : null,
        expires_at: data.expires_at || null,
      });

      setCreatedKey(result);
    } catch {
      // Error handled by hook
    }
  };

  const handleClose = () => {
    if (createdKey && !acknowledged) {
      // Don't allow closing without acknowledgment
      return;
    }
    reset();
    setCreatedKey(null);
    setAcknowledged(false);
    onClose();
  };

  const handleDone = () => {
    reset();
    setCreatedKey(null);
    setAcknowledged(false);
    onSuccess();
  };

  const downloadAsFile = () => {
    if (!createdKey) return;

    const content = `# EnforceAI API Key
# Agent: ${agent.alias || agent.agent_id}
# Created: ${new Date().toISOString()}
#
# IMPORTANT: Store this securely. This secret cannot be retrieved again.

API Key: ${createdKey.api_key_value}

# Usage:
# Add this header to your requests:
# Authorization: Bearer ${createdKey.api_key_value}
`;

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `api-key-${createdKey.key_id.slice(0, 8)}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // If key was created, show the secret display
  if (createdKey) {
    return (
      <Modal
        open={open}
        onClose={handleClose}
        title="API Key Created"
        size="lg"
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
                API key created successfully
              </p>
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                For agent: {agent.alias || agent.agent_id.slice(0, 8)}
              </p>
            </div>
          </div>

          {/* Critical warning */}
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md p-4">
            <div className="flex">
              <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0" />
              <div className="ml-3">
                <h3 className="text-sm font-medium text-yellow-800 dark:text-yellow-300">
                  Save this secret now
                </h3>
                <p className="mt-1 text-sm text-yellow-700 dark:text-yellow-400">
                  This is the only time you will see this API key secret. It cannot be retrieved later.
                  Store it securely before closing this dialog.
                </p>
              </div>
            </div>
          </div>

          {/* Secret display */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              API Key Value
            </label>
            <div className="relative">
              <code className="block w-full p-3 pr-24 bg-gray-100 dark:bg-gray-800 rounded-md text-sm font-mono text-gray-900 dark:text-gray-100 break-all border border-gray-200 dark:border-gray-700">
                {createdKey.api_key_value}
              </code>
              <div className="absolute right-2 top-1/2 -translate-y-1/2">
                <CopyButton text={createdKey.api_key_value} />
              </div>
            </div>
          </div>

          {/* Key ID for reference */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Key ID (for reference)
            </label>
            <code className="block w-full p-2 bg-gray-50 dark:bg-gray-900 rounded-md text-sm font-mono text-gray-600 dark:text-gray-400">
              {createdKey.key_id}
            </code>
          </div>

          {/* Action buttons */}
          <div className="flex space-x-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={downloadAsFile}
              leftIcon={<ArrowDownTrayIcon className="w-4 h-4" />}
            >
              Download as .txt
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                navigator.clipboard.writeText(
                  `Authorization: Bearer ${createdKey.api_key_value}`
                );
              }}
              leftIcon={<ClipboardDocumentIcon className="w-4 h-4" />}
            >
              Copy as Header
            </Button>
          </div>

          {/* Usage instructions */}
          <div className="bg-gray-50 dark:bg-gray-900 rounded-md p-3">
            <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
              How to use
            </h4>
            <code className="block text-xs font-mono text-gray-600 dark:text-gray-400">
              Authorization: Bearer {createdKey.api_key_value.slice(0, 20)}...
            </code>
          </div>

          {/* Acknowledgment checkbox */}
          <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
            <Checkbox
              label="I have securely saved this API key"
              checked={acknowledged}
              onChange={(e) => setAcknowledged(e.target.checked)}
            />
          </div>
        </div>

        <ModalFooter>
          <Button
            type="button"
            variant="primary"
            onClick={handleDone}
            disabled={!acknowledged}
          >
            Done
          </Button>
        </ModalFooter>
      </Modal>
    );
  }

  // Form to create API key
  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Create API Key"
      size="md"
    >
      <form onSubmit={handleSubmit(onSubmit)} className="space-y-4">
        {/* Agent info */}
        <div className="flex items-center space-x-3 p-3 bg-gray-50 dark:bg-gray-900 rounded-md">
          <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
            <KeyIcon className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
              Creating key for: {agent.alias || 'Unnamed Agent'}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400 font-mono">
              {agent.agent_id}
            </p>
          </div>
        </div>

        {/* Scopes */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Scopes (optional)
          </label>
          <Input
            {...register('scopes')}
            placeholder="scope1, scope2 (subset of agent scopes)"
            error={errors.scopes?.message}
          />
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Leave empty to inherit all agent scopes. Comma-separated list.
          </p>
          {agent.scopes.length > 0 && (
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Agent scopes: {agent.scopes.join(', ')}
            </p>
          )}
        </div>

        {/* Expiration */}
        <div>
          <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
            Expires At (optional)
          </label>
          <Input
            type="datetime-local"
            {...register('expires_at')}
            error={errors.expires_at?.message}
          />
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Leave empty for no expiration.
          </p>
        </div>

        {error && (
          <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
            <p className="text-sm text-red-600 dark:text-red-400">
              Failed to create API key. Please try again.
            </p>
          </div>
        )}

        <ModalFooter>
          <Button
            type="button"
            variant="secondary"
            onClick={handleClose}
            disabled={isCreating}
          >
            Cancel
          </Button>
          <Button
            type="submit"
            variant="primary"
            loading={isCreating}
          >
            Create API Key
          </Button>
        </ModalFooter>
      </form>
    </Modal>
  );
}
