import { useForm } from 'react-hook-form';
import { zodResolver } from '@hookform/resolvers/zod';
import { z } from 'zod';
import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Button, Input, Textarea } from '@/components/ui';
import { useCreateScope } from './hooks';

const RESERVED_SCOPE_NAMES = new Set(['UI-Scopes', 'group_mappings']);

const createScopeSchema = z
  .object({
    name: z
      .string()
      .min(1, 'Scope name is required')
      .refine((val) => val.trim().length > 0, {
        message: 'Scope name is required',
      })
      .refine((val) => !RESERVED_SCOPE_NAMES.has(val.trim()), {
        message: 'This scope name is reserved',
      }),
    mode: z.enum(['blank', 'server-tools']),
    server: z.string().optional(),
    tools_mode: z.enum(['all', 'list']).optional(),
    tools: z.string().optional(),
    agent_actions_json: z.string().optional(),
  })
  .superRefine((data, ctx) => {
    if (data.mode === 'server-tools') {
      if (!data.server || !data.server.trim()) {
        ctx.addIssue({
          code: z.ZodIssueCode.custom,
          path: ['server'],
          message: 'Server is required for this mode',
        });
      }

      const toolsMode = data.tools_mode ?? 'all';
      if (toolsMode === 'list') {
        const parsed = (data.tools ?? '')
          .split(',')
          .map((t) => t.trim())
          .filter((t) => t.length > 0);

        if (parsed.length === 0) {
          ctx.addIssue({
            code: z.ZodIssueCode.custom,
            path: ['tools'],
            message: 'Provide at least one tool name',
          });
        }
      }
    }
  });

type CreateScopeFormData = z.infer<typeof createScopeSchema>;

export interface CreateScopeModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess: (scopeName: string) => void;
  catalogEtag?: string;
}

export function CreateScopeModal({
  open,
  onClose,
  onSuccess,
  catalogEtag,
}: CreateScopeModalProps) {
  const { createScope, isCreating, error } = useCreateScope();

  const {
    register,
    handleSubmit,
    formState: { errors },
    reset: resetForm,
    setError,
    watch,
  } = useForm<CreateScopeFormData>({
    resolver: zodResolver(createScopeSchema),
    defaultValues: {
      name: '',
      mode: 'blank',
      server: '',
      tools_mode: 'all',
      tools: '',
      agent_actions_json: '',
    },
  });

  const mode = watch('mode');
  const toolsMode = watch('tools_mode');

  const handleClose = () => {
    resetForm();
    onClose();
  };

  const onSubmit = async (data: CreateScopeFormData) => {
    const name = data.name.trim();

    let agentPermissions: Array<{ action: string; resources: string[] }> = [];
    if (data.agent_actions_json && data.agent_actions_json.trim()) {
      try {
        const parsed = JSON.parse(data.agent_actions_json) as unknown;
        if (!Array.isArray(parsed)) {
          throw new Error('agent_permissions must be a JSON array');
        }
        agentPermissions = parsed as Array<{ action: string; resources: string[] }>;
      } catch {
        setError('agent_actions_json', {
          type: 'manual',
          message: 'Invalid JSON format',
        });
        return;
      }
    }

    try {
      if (data.mode === 'blank') {
        await createScope(
          {
            name,
            server_permissions: [],
            agent_permissions: agentPermissions,
          },
          catalogEtag
        );
        resetForm();
        onSuccess(name);
        return;
      }

      const server = (data.server ?? '').trim();
      const resolvedToolsMode = data.tools_mode ?? 'all';
      const tools =
        resolvedToolsMode === 'list'
          ? (data.tools ?? '')
              .split(',')
              .map((t) => t.trim())
              .filter((t) => t.length > 0)
          : [];

      await createScope(
        {
          name,
          server_permissions: [
            {
              server,
              methods: {
                all_methods: false,
                methods: ['tools/list', 'tools/call'],
              },
              tools:
                resolvedToolsMode === 'all'
                  ? { all_tools: true, tools: [] }
                  : { all_tools: false, tools },
            },
          ],
          agent_permissions: agentPermissions,
        },
        catalogEtag
      );

      resetForm();
      onSuccess(name);
    } catch {
      // Error is handled by the hook
    }
  };

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Create Scope"
      description="Create a new scope in the policy catalog (admin only)"
      size="lg"
    >
      <form onSubmit={handleSubmit(onSubmit)}>
        <div className="space-y-4">
          <Input
            label="Scope name"
            placeholder="sqlite.manage"
            helperText="Scope names must be unique within the catalog."
            error={errors.name?.message}
            {...register('name')}
          />

          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              Creation mode
            </label>
            <div className="space-y-2">
              <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
                <input
                  type="radio"
                  value="blank"
                  {...register('mode')}
                />
                Blank (deny-by-default)
              </label>
              <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
                <input
                  type="radio"
                  value="server-tools"
                  {...register('mode')}
                />
                Simple server tools scope (tools/list + tools/call)
              </label>
            </div>
          </div>

          {mode === 'server-tools' && (
            <>
              <Input
                label="Server"
                placeholder="sqlite"
                helperText="Server name as used by the gateway (e.g., 'sqlite', 'mcpgw')."
                error={errors.server?.message}
                {...register('server')}
              />

              <div className="space-y-2">
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
                  Tools policy
                </label>
                <div className="space-y-2">
                  <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
                    <input
                      type="radio"
                      value="all"
                      {...register('tools_mode')}
                    />
                    All tools (*)
                  </label>
                  <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
                    <input
                      type="radio"
                      value="list"
                      {...register('tools_mode')}
                    />
                    Allowlist (comma-separated)
                  </label>
                </div>
              </div>

              {toolsMode === 'list' && (
                <Input
                  label="Tools allowlist"
                  placeholder="read_query, list_tables"
                  helperText="Comma-separated list of tool names."
                  error={errors.tools?.message}
                  {...register('tools')}
                />
              )}
            </>
          )}

          <Textarea
            label="Agent permissions (optional JSON)"
            placeholder='[{"action":"list_agents","resources":["all"]}]'
            helperText="Advanced: optional agent action permissions in the scopes.yml format."
            rows={3}
            error={errors.agent_actions_json?.message}
            {...register('agent_actions_json')}
          />

          {error && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
              <p className="text-sm text-red-600 dark:text-red-400">
                Failed to create scope. Please check your inputs and try again.
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
            Create Scope
          </Button>
        </ModalFooter>
      </form>
    </Modal>
  );
}

