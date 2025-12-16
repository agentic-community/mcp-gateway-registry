import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Button, Input, Textarea } from '@/components/ui';
import { useCreateEnforceAIAgent } from './hooks';

// ============================================================================
// Validation Schema
// ============================================================================

const createAgentSchema = z.object({
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

type CreateAgentFormData = z.infer<typeof createAgentSchema>;

// ============================================================================
// Component
// ============================================================================

export interface CreateAgentModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export function CreateAgentModal({
  open,
  onClose,
  onSuccess,
}: CreateAgentModalProps) {
  const { createAgent, isCreating, error } = useCreateEnforceAIAgent();

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset: resetForm,
    setError,
  } = useForm<CreateAgentFormData>({
    resolver: zodResolver(createAgentSchema),
    defaultValues: {
      alias: '',
      scopes: '',
      allowed_tools: '',
      metadata: '',
    },
  });

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const onSubmit = async (data: CreateAgentFormData) => {
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

      await createAgent({
        scopes,
        alias: data.alias || null,
        allowed_tools: allowed_tools && allowed_tools.length > 0 ? allowed_tools : null,
        metadata,
      });

      resetForm();
      onSuccess();
    } catch {
      // Error is handled by the hook
    }
  };

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Create EnforceAI Agent"
      description="Create a new agent identity for tool access authorization"
      size="lg"
    >
      <form onSubmit={handleSubmit(onSubmit)}>
        <div className="space-y-4">
          <Input
            label="Alias (optional)"
            placeholder="my-agent"
            helperText="A human-readable name for this agent"
            error={errors.alias?.message}
            {...register('alias')}
          />

          <Input
            label="Scopes"
            placeholder="sqlite.manage, filesystem.read"
            helperText="Comma-separated list of scopes (required). These define what this agent is allowed to do."
            error={errors.scopes?.message}
            {...register('scopes')}
          />

          <div className="p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md">
            <p className="text-sm text-yellow-700 dark:text-yellow-400">
              Scopes must be defined in the scope catalog. Invalid scopes will
              result in authorization failures.
            </p>
          </div>

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
                Failed to create agent. Please check your inputs and try again.
              </p>
            </div>
          )}
        </div>

        <ModalFooter>
          <Button
            type="button"
            variant="secondary"
            onClick={handleClose}
            disabled={isCreating}
          >
            Cancel
          </Button>
          <Button type="submit" variant="primary" loading={isCreating}>
            Create Agent
          </Button>
        </ModalFooter>
      </form>
    </Modal>
  );
}
