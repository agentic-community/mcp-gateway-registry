import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Button, Input, Textarea } from '@/components/ui';
import { useRegisterServer } from './hooks';
import type { CredentialBinding, UpstreamAuthConfig } from '@/api/types';

// ============================================================================
// Validation Schema
// ============================================================================

const upstreamAuthTypeOptions = [
  'none',
  'api-key',
  'oauth2',
  'oidc',
  'provider-oauth',
  'jwt',
  'header-trust',
] as const;

type UpstreamAuthTypeOption = (typeof upstreamAuthTypeOptions)[number];

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
  upstream_auth_type: z.enum(upstreamAuthTypeOptions),
  upstream_auth_provider: z.string().max(100).optional(),
  upstream_credential_binding: z.enum(['service', 'user', 'agent', 'user+agent']),
  upstream_injection_header_name: z.string().max(100).optional(),
  upstream_injection_scheme: z.string().max(50).optional(),
}).superRefine((data, ctx) => {
  if (data.upstream_auth_type === 'provider-oauth') {
    const provider = (data.upstream_auth_provider || '').trim();
    if (!provider) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Provider is required for provider OAuth',
        path: ['upstream_auth_provider'],
      });
    }
  }

  const needsHeaderInjection =
    data.upstream_auth_type === 'api-key' ||
    data.upstream_auth_type === 'oauth2' ||
    data.upstream_auth_type === 'oidc' ||
    data.upstream_auth_type === 'provider-oauth' ||
    data.upstream_auth_type === 'jwt';
  if (needsHeaderInjection) {
    const headerName = (data.upstream_injection_header_name || '').trim();
    if (headerName && /[\r\n]/.test(headerName)) {
      ctx.addIssue({
        code: z.ZodIssueCode.custom,
        message: 'Header name must not contain newline characters',
        path: ['upstream_injection_header_name'],
      });
    }
  }
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
      upstream_auth_type: 'none',
      upstream_auth_provider: '',
      upstream_credential_binding: 'service',
      upstream_injection_header_name: '',
      upstream_injection_scheme: '',
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

      const upstreamAuthType = data.upstream_auth_type as UpstreamAuthTypeOption;
      const upstreamCredentialBinding = data.upstream_credential_binding as CredentialBinding;
      const upstreamProvider = (data.upstream_auth_provider || '').trim() || undefined;
      const upstreamHeaderName = (data.upstream_injection_header_name || '').trim() || undefined;
      const upstreamScheme = (data.upstream_injection_scheme || '').trim() || undefined;

      const upstream_auth: UpstreamAuthConfig =
        upstreamAuthType === 'none'
          ? {
              mode: 'none',
              type: 'none',
              credential_binding: 'service',
              injection: null,
            }
          : {
              mode: 'gateway-managed',
              type: upstreamAuthType,
              provider: upstreamProvider,
              credential_binding: upstreamCredentialBinding,
              injection: upstreamHeaderName
                ? {
                    kind: 'header',
                    header_name: upstreamHeaderName,
                    scheme: upstreamScheme || null,
                  }
                : null,
            };

      await registerServer({
        name: data.name,
        path: data.path,
        proxy_pass_url: data.proxy_pass_url,
        description: data.description || undefined,
        tags,
        upstream_auth,
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

          <div className="p-4 bg-gray-50 dark:bg-gray-700/50 rounded-md space-y-3">
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                Upstream Authentication
              </p>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-1">
                Upstream credentials are gateway-terminated. Configure secrets in “Upstream
                Credentials” (not in this form).
              </p>
            </div>

            <div>
              <label
                htmlFor="upstream_auth_type"
                className="block text-sm font-medium text-gray-700 dark:text-gray-200"
              >
                Auth Type
              </label>
              <select
                id="upstream_auth_type"
                className="mt-1 block w-full rounded-md border-gray-300 text-sm focus:border-primary-500 focus:ring-primary-500 dark:bg-gray-800 dark:border-gray-600 dark:text-gray-200"
                {...register('upstream_auth_type')}
              >
                <option value="none">None</option>
                <option value="api-key">API Key</option>
                <option value="oauth2">OAuth2</option>
                <option value="oidc">OIDC</option>
                <option value="provider-oauth">Provider OAuth</option>
                <option value="jwt">JWT (Static Bearer)</option>
                <option value="header-trust">Header Trust (No Secret)</option>
              </select>
            </div>

            <Input
              label="Provider (optional)"
              placeholder="github"
              helperText="Required for Provider OAuth; optional for OAuth2/OIDC"
              error={errors.upstream_auth_provider?.message}
              {...register('upstream_auth_provider')}
            />

            <div>
              <label
                htmlFor="upstream_credential_binding"
                className="block text-sm font-medium text-gray-700 dark:text-gray-200"
              >
                Credential Binding
              </label>
              <select
                id="upstream_credential_binding"
                className="mt-1 block w-full rounded-md border-gray-300 text-sm focus:border-primary-500 focus:ring-primary-500 dark:bg-gray-800 dark:border-gray-600 dark:text-gray-200"
                {...register('upstream_credential_binding')}
              >
                <option value="service">Service</option>
                <option value="user">User</option>
                <option value="agent">Agent</option>
                <option value="user+agent">User + Agent</option>
              </select>
            </div>

            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3">
              <Input
                label="Injection Header Override (optional)"
                placeholder="Authorization"
                helperText="Leave blank to use defaults"
                error={errors.upstream_injection_header_name?.message}
                {...register('upstream_injection_header_name')}
              />
              <Input
                label="Scheme (optional)"
                placeholder="Bearer"
                helperText="Applies only for Authorization-like headers"
                error={errors.upstream_injection_scheme?.message}
                {...register('upstream_injection_scheme')}
              />
            </div>
          </div>

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
