import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Button, Input, Textarea } from '@/components/ui';
import { useRegisterAgent } from './hooks';

// ============================================================================
// Validation Schema
// ============================================================================

const registerAgentSchema = z.object({
  name: z
    .string()
    .min(1, 'Name is required')
    .max(100, 'Name must be 100 characters or less'),
  path: z
    .string()
    .min(1, 'Path is required')
    .max(50, 'Path must be 50 characters or less')
    .regex(
      /^[a-z0-9-]+$/,
      'Path must contain only lowercase letters, numbers, and hyphens'
    ),
  description: z.string().max(500, 'Description must be 500 characters or less').optional(),
  skills: z.string().optional(),
  tags: z.string().optional(),
  visibility: z.enum(['public', 'private']).optional(),
});

type RegisterAgentFormData = z.infer<typeof registerAgentSchema>;

// ============================================================================
// Component
// ============================================================================

export interface AgentRegisterModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export function AgentRegisterModal({
  open,
  onClose,
  onSuccess,
}: AgentRegisterModalProps) {
  const { registerAgent, isRegistering, error, reset } = useRegisterAgent();

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset: resetForm,
  } = useForm<RegisterAgentFormData>({
    resolver: zodResolver(registerAgentSchema),
    defaultValues: {
      name: '',
      path: '',
      description: '',
      skills: '',
      tags: '',
      visibility: 'public',
    },
  });

  const handleClose = () => {
    resetForm();
    reset();
    onClose();
  };

  const onSubmit = async (data: RegisterAgentFormData) => {
    try {
      // Convert skills from comma-separated string to array
      const skills = data.skills
        ? data.skills
            .split(',')
            .map((skill) => skill.trim())
            .filter((skill) => skill.length > 0)
        : undefined;

      // Convert tags from comma-separated string to array
      const tags = data.tags
        ? data.tags
            .split(',')
            .map((tag) => tag.trim())
            .filter((tag) => tag.length > 0)
        : undefined;

      await registerAgent({
        name: data.name,
        path: data.path,
        description: data.description || undefined,
        skills,
        tags,
        visibility: data.visibility,
      });

      resetForm();
      reset();
      onSuccess();
    } catch {
      // Error is handled by the hook
    }
  };

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Register A2A Agent"
      description="Add a new Agent-to-Agent agent to the registry"
      size="lg"
    >
      <form onSubmit={handleSubmit(onSubmit)}>
        <div className="space-y-4">
          <Input
            label="Name"
            placeholder="My AI Agent"
            error={errors.name?.message}
            {...register('name')}
          />

          <Input
            label="Path"
            placeholder="my-agent"
            helperText="Unique identifier for the agent (lowercase, no spaces)"
            error={errors.path?.message}
            {...register('path')}
          />

          <Textarea
            label="Description"
            placeholder="A brief description of what this agent does..."
            rows={3}
            error={errors.description?.message}
            {...register('description')}
          />

          <Input
            label="Skills"
            placeholder="code-review, refactoring, testing"
            helperText="Comma-separated list of skills this agent provides"
            error={errors.skills?.message}
            {...register('skills')}
          />

          <Input
            label="Tags"
            placeholder="ai, automation, code"
            helperText="Comma-separated list of tags"
            error={errors.tags?.message}
            {...register('tags')}
          />

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Visibility
            </label>
            <select
              {...register('visibility')}
              className="block w-full rounded-md border-gray-300 dark:border-gray-600 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm dark:bg-gray-700 dark:text-gray-100"
            >
              <option value="public">Public</option>
              <option value="private">Private</option>
            </select>
          </div>

          {error && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
              <p className="text-sm text-red-600 dark:text-red-400">
                Failed to register agent. Please check your inputs and try again.
              </p>
            </div>
          )}
        </div>

        <ModalFooter>
          <Button
            type="button"
            variant="secondary"
            onClick={handleClose}
            disabled={isRegistering}
          >
            Cancel
          </Button>
          <Button type="submit" variant="primary" loading={isRegistering}>
            Register Agent
          </Button>
        </ModalFooter>
      </form>
    </Modal>
  );
}
