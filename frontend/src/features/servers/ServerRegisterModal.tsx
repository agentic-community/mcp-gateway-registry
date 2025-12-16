import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Button, Input, Textarea } from '@/components/ui';
import { useRegisterServer } from './hooks';

// ============================================================================
// Validation Schema
// ============================================================================

const registerServerSchema = z.object({
  name: z
    .string()
    .min(1, 'Display name is required')
    .max(100, 'Display name must be 100 characters or less'),
  path: z
    .string()
    .min(1, 'Path is required')
    .max(50, 'Path must be 50 characters or less')
    .regex(
      /^[a-z0-9-]+$/,
      'Path must contain only lowercase letters, numbers, and hyphens'
    ),
  proxy_pass_url: z
    .string()
    .min(1, 'URL is required')
    .url('Must be a valid URL'),
  description: z.string().max(500, 'Description must be 500 characters or less').optional(),
  tags: z.string().optional(),
});

type RegisterServerFormData = z.infer<typeof registerServerSchema>;

// ============================================================================
// Component
// ============================================================================

export interface ServerRegisterModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess: () => void;
}

export function ServerRegisterModal({
  open,
  onClose,
  onSuccess,
}: ServerRegisterModalProps) {
  const { registerServer, isRegistering, error, reset } = useRegisterServer();

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset: resetForm,
  } = useForm<RegisterServerFormData>({
    resolver: zodResolver(registerServerSchema),
    defaultValues: {
      name: '',
      path: '',
      proxy_pass_url: '',
      description: '',
      tags: '',
    },
  });

  const handleClose = () => {
    resetForm();
    reset();
    onClose();
  };

  const onSubmit = async (data: RegisterServerFormData) => {
    try {
      // Convert tags from comma-separated string to array
      const tags = data.tags
        ? data.tags
            .split(',')
            .map((tag) => tag.trim())
            .filter((tag) => tag.length > 0)
        : undefined;

      await registerServer({
        name: data.name,
        path: data.path,
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
      title="Register MCP Server"
      description="Add a new MCP server to the registry"
      size="lg"
    >
      <form onSubmit={handleSubmit(onSubmit)}>
        <div className="space-y-4">
          <Input
            label="Display Name"
            placeholder="My MCP Server"
            error={errors.name?.message}
            {...register('name')}
          />

          <Input
            label="Path"
            placeholder="my-server"
            helperText="Unique identifier for the server (lowercase, no spaces)"
            error={errors.path?.message}
            {...register('path')}
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
                Failed to register server. Please check your inputs and try again.
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
            Register Server
          </Button>
        </ModalFooter>
      </form>
    </Modal>
  );
}
