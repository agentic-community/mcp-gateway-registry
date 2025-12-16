import { useEffect } from 'react';
import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Button, Input, Textarea } from '@/components/ui';
import { useUpdateServer } from './hooks';
import type { Server } from '@/api/types';

// ============================================================================
// Validation Schema
// ============================================================================

const editServerSchema = z.object({
  name: z
    .string()
    .min(1, 'Display name is required')
    .max(100, 'Display name must be 100 characters or less'),
  proxy_pass_url: z
    .string()
    .min(1, 'URL is required')
    .url('Must be a valid URL'),
  description: z.string().max(500, 'Description must be 500 characters or less').optional(),
  tags: z.string().optional(),
});

type EditServerFormData = z.infer<typeof editServerSchema>;

// ============================================================================
// Component
// ============================================================================

export interface ServerEditModalProps {
  open: boolean;
  server: Server;
  onClose: () => void;
  onSuccess: () => void;
}

export function ServerEditModal({
  open,
  server,
  onClose,
  onSuccess,
}: ServerEditModalProps) {
  const { updateServer, isUpdating, error, reset } = useUpdateServer();

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset: resetForm,
  } = useForm<EditServerFormData>({
    resolver: zodResolver(editServerSchema),
    defaultValues: {
      name: server.display_name,
      proxy_pass_url: server.proxy_pass_url,
      description: server.description || '',
      tags: server.tags?.join(', ') || '',
    },
  });

  // Reset form when server changes
  useEffect(() => {
    resetForm({
      name: server.display_name,
      proxy_pass_url: server.proxy_pass_url,
      description: server.description || '',
      tags: server.tags?.join(', ') || '',
    });
  }, [server, resetForm]);

  const handleClose = () => {
    resetForm();
    reset();
    onClose();
  };

  const onSubmit = async (data: EditServerFormData) => {
    try {
      // Convert tags from comma-separated string to array
      const tags = data.tags
        ? data.tags
            .split(',')
            .map((tag) => tag.trim())
            .filter((tag) => tag.length > 0)
        : undefined;

      await updateServer(server.path, {
        name: data.name,
        proxy_pass_url: data.proxy_pass_url,
        description: data.description || undefined,
        tags,
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
      title="Edit MCP Server"
      description={`Editing ${server.display_name}`}
      size="lg"
    >
      <form onSubmit={handleSubmit(onSubmit)}>
        <div className="space-y-4">
          <div className="p-3 bg-gray-50 dark:bg-gray-700/50 rounded-md">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              <span className="font-medium">Path:</span>{' '}
              <code className="text-xs bg-gray-200 dark:bg-gray-600 px-1 py-0.5 rounded">
                {server.path}
              </code>
            </p>
          </div>

          <Input
            label="Display Name"
            placeholder="My MCP Server"
            error={errors.name?.message}
            {...register('name')}
          />

          <Input
            label="Proxy URL"
            placeholder="http://localhost:3000/mcp"
            helperText="The URL where the MCP server is running"
            error={errors.proxy_pass_url?.message}
            {...register('proxy_pass_url')}
          />

          <Textarea
            label="Description"
            placeholder="A brief description of what this server does..."
            rows={3}
            error={errors.description?.message}
            {...register('description')}
          />

          <Input
            label="Tags"
            placeholder="database, sql, analytics"
            helperText="Comma-separated list of tags"
            error={errors.tags?.message}
            {...register('tags')}
          />

          {error && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
              <p className="text-sm text-red-600 dark:text-red-400">
                Failed to update server. Please check your inputs and try again.
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
