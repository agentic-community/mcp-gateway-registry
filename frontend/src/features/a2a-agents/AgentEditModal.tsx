import { useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Button, Input, Textarea } from '@/components/ui';
import { useUpdateAgent } from './hooks';
import type { A2AAgent } from '@/api/types';

// ============================================================================
// Validation Schema
// ============================================================================

const editAgentSchema = z.object({
  name: z
    .string()
    .min(1, 'Name is required')
    .max(100, 'Name must be 100 characters or less'),
  description: z.string().max(500, 'Description must be 500 characters or less').optional(),
  skills: z.string().optional(),
  tags: z.string().optional(),
  visibility: z.enum(['public', 'private']).optional(),
});

type EditAgentFormData = z.infer<typeof editAgentSchema>;

// ============================================================================
// Component
// ============================================================================

export interface AgentEditModalProps {
  open: boolean;
  agent: A2AAgent;
  onClose: () => void;
  onSuccess: () => void;
}

export function AgentEditModal({
  open,
  agent,
  onClose,
  onSuccess,
}: AgentEditModalProps) {
  const { updateAgent, isUpdating, error, reset } = useUpdateAgent();

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset: resetForm,
  } = useForm<EditAgentFormData>({
    resolver: zodResolver(editAgentSchema),
    defaultValues: {
      name: agent.name,
      description: agent.description || '',
      skills: agent.skills?.join(', ') || '',
      tags: agent.tags?.join(', ') || '',
      visibility: agent.visibility || 'public',
    },
  });

  // Reset form when agent changes
  useEffect(() => {
    resetForm({
      name: agent.name,
      description: agent.description || '',
      skills: agent.skills?.join(', ') || '',
      tags: agent.tags?.join(', ') || '',
      visibility: agent.visibility || 'public',
    });
  }, [agent, resetForm]);

  const handleClose = () => {
    reset();
    onClose();
  };

  const onSubmit = async (data: EditAgentFormData) => {
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

      await updateAgent(agent.path, {
        name: data.name,
        description: data.description || undefined,
        skills,
        tags,
        visibility: data.visibility,
      });

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
      title="Edit A2A Agent"
      description={`Update settings for ${agent.name}`}
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

          <div>
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
              Path
            </label>
            <p className="text-sm text-gray-500 dark:text-gray-400 font-mono bg-gray-50 dark:bg-gray-700 px-3 py-2 rounded-md">
              {agent.path}
            </p>
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              The path cannot be changed after registration
            </p>
          </div>

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
