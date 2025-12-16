import { useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Button, Input, Textarea } from '@/components/ui';
import { useUpdateEnforceAIAgent } from './hooks';
import type { EnforceAIAgent } from '@/api/types';

// ============================================================================
// Validation Schema
// ============================================================================

const editAgentSchema = z.object({
  alias: z
    .string()
    .max(100, 'Alias must be 100 characters or less')
    .optional()
    .or(z.literal('')),
  scopes: z
    .string()
    .min(1, 'At least one scope is required')
    .refine(
      (val) =>
        val
          .split(',')
          .map((s) => s.trim())
          .filter((s) => s.length > 0).length > 0,
      { message: 'At least one valid scope is required' }
    ),
  allowed_tools: z.string().optional(),
  metadata: z.string().optional(),
});

type EditAgentFormData = z.infer<typeof editAgentSchema>;

// ============================================================================
// Component
// ============================================================================

export interface EditAgentModalProps {
  open: boolean;
  agent: EnforceAIAgent;
  onClose: () => void;
  onSuccess: () => void;
}

export function EditAgentModal({
  open,
  agent,
  onClose,
  onSuccess,
}: EditAgentModalProps) {
  const { updateAgent, isUpdating, error } = useUpdateEnforceAIAgent();

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset: resetForm,
    setError,
  } = useForm<EditAgentFormData>({
    resolver: zodResolver(editAgentSchema),
    defaultValues: {
      alias: agent.alias || '',
      scopes: agent.scopes.join(', '),
      allowed_tools: agent.allowed_tools?.join(', ') || '',
      metadata: agent.metadata ? JSON.stringify(agent.metadata, null, 2) : '',
    },
  });

  // Reset form when agent changes
  useEffect(() => {
    resetForm({
      alias: agent.alias || '',
      scopes: agent.scopes.join(', '),
      allowed_tools: agent.allowed_tools?.join(', ') || '',
      metadata: agent.metadata ? JSON.stringify(agent.metadata, null, 2) : '',
    });
  }, [agent, resetForm]);

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const onSubmit = async (data: EditAgentFormData) => {
    try {
      // Convert scopes from comma-separated string to array
      const scopes = data.scopes
        .split(',')
        .map((scope) => scope.trim())
        .filter((scope) => scope.length > 0);

      // Convert allowed_tools from comma-separated string to array (or null)
      const allowed_tools = data.allowed_tools
        ? data.allowed_tools
            .split(',')
            .map((tool) => tool.trim())
            .filter((tool) => tool.length > 0)
        : null;

      // Parse metadata JSON (or null)
      let metadata: Record<string, unknown> | null = null;
      if (data.metadata && data.metadata.trim()) {
        try {
          metadata = JSON.parse(data.metadata);
        } catch {
          setError('metadata', {
            type: 'manual',
            message: 'Invalid JSON format',
          });
          return;
        }
      }

      await updateAgent(agent.agent_id, {
        scopes,
        alias: data.alias || null,
        allowed_tools: allowed_tools && allowed_tools.length > 0 ? allowed_tools : null,
        metadata,
      });

      onSuccess();
    } catch {
      // Error is handled by the hook
    }
  };

  // Check if scopes are being changed for warning
  const currentScopes = agent.scopes.join(', ');

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Edit EnforceAI Agent"
      description={`Update agent ${agent.alias || agent.agent_id.slice(0, 8)}...`}
      size="lg"
    >
      <form onSubmit={handleSubmit(onSubmit)}>
        <div className="space-y-4">
          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Agent ID
            </label>
            <div className="p-2 bg-gray-50 dark:bg-gray-700 rounded-md font-mono text-sm text-gray-600 dark:text-gray-400">
              {agent.agent_id}
            </div>
          </div>

          <Input
            label="Alias"
            placeholder="my-agent"
            helperText="A human-readable name for this agent"
            error={errors.alias?.message}
            {...register('alias')}
          />

          <Input
            label="Scopes"
            placeholder="sqlite.manage, filesystem.read"
            helperText="Comma-separated list of scopes (required)"
            error={errors.scopes?.message}
            {...register('scopes')}
          />

          {currentScopes && (
            <div className="p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md">
              <p className="text-sm text-blue-700 dark:text-blue-400">
                <strong>Current scopes:</strong> {currentScopes}
              </p>
              <p className="text-xs text-blue-600 dark:text-blue-500 mt-1">
                Removing scopes may affect existing tokens and API keys for this agent.
              </p>
            </div>
          )}

          <Input
            label="Allowed Tools (optional)"
            placeholder="read_query, list_tables"
            helperText="Comma-separated list of specific tools this agent can use. Leave empty to allow all tools within scope permissions."
            error={errors.allowed_tools?.message}
            {...register('allowed_tools')}
          />

          <Textarea
            label="Metadata (optional)"
            placeholder='{"team": "platform", "purpose": "database-access"}'
            helperText="Optional JSON metadata for this agent"
            rows={3}
            error={errors.metadata?.message}
            {...register('metadata')}
          />

          {error && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
              <p className="text-sm text-red-600 dark:text-red-400">
                Failed to update agent. Please check your inputs and try again.
              </p>
            </div>
          )}
        </div>

        <ModalFooter>
          <Button
            type="button"
            variant="secondary"
            onClick={handleClose}
            disabled={isUpdating}
          >
            Cancel
          </Button>
          <Button type="submit" variant="primary" loading={isUpdating}>
            Save Changes
          </Button>
        </ModalFooter>
      </form>
    </Modal>
  );
}
