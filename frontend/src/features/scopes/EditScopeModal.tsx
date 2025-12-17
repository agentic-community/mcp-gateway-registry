import { useEffect, useMemo, useState } from 'react';
import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Badge, Button, Card, Checkbox, Input } from '@/components/ui';
import { normalizeError } from '@/lib/errors';
import { useReplaceScope } from './hooks';
import type {
  AgentActionPermission,
  ReplaceScopeRequest,
  ScopeDefinition,
  ServerPermission,
} from '@/api/types';

type _ToolsPolicyMode = 'none' | 'all' | 'list';

function _validateServerPermissionsOrError(
  serverPermissions: ServerPermission[]
): string | null {
  for (const perm of serverPermissions) {
    const methods = perm.methods.all_methods ? ['*'] : perm.methods.methods;
    const methodsLower = methods.map((method) => method.toLowerCase().trim());
    const allowsToolsCall =
      methodsLower.includes('tools/call') ||
      methodsLower.includes('*') ||
      methodsLower.includes('all');

    if (allowsToolsCall && !perm.tools) {
      return `Server '${perm.server}' includes tools/call but has no tools policy`;
    }
  }
  return null;
}

function _allowsToolsCall(
  serverPermission: ServerPermission
): boolean {
  const methods = serverPermission.methods.all_methods
    ? ['*']
    : serverPermission.methods.methods;
  const methodsLower = methods.map((method) => method.toLowerCase().trim());
  return (
    methodsLower.includes('tools/call') ||
    methodsLower.includes('*') ||
    methodsLower.includes('all')
  );
}

function _normalizeServerPermissions(
  serverPermissions: ServerPermission[]
): ServerPermission[] {
  return serverPermissions.map((permission) => {
    const server = permission.server.trim();
    const methods = permission.methods.all_methods
      ? []
      : permission.methods.methods
          .map((method) => method.trim())
          .filter((method) => method.length > 0);

    const tools =
      permission.tools == null
        ? undefined
        : permission.tools.all_tools
          ? { all_tools: true, tools: [] }
          : {
              all_tools: false,
              tools: permission.tools.tools
                .map((tool) => tool.trim())
                .filter((tool) => tool.length > 0),
            };

    return {
      server,
      methods: {
        all_methods: permission.methods.all_methods,
        methods,
      },
      tools,
    };
  });
}

function _normalizeAgentPermissions(
  agentPermissions: AgentActionPermission[]
): AgentActionPermission[] {
  return agentPermissions.map((permission) => {
    return {
      action: permission.action.trim(),
      resources: permission.resources
        .map((resource) => resource.trim())
        .filter((resource) => resource.length > 0),
    };
  });
}

function _summarize(
  serverPermissions: ServerPermission[] | null,
  agentPermissions: AgentActionPermission[] | null
) {
  if (!serverPermissions) return null;

  const servers = Array.from(
    new Set(serverPermissions.map((permission) => permission.server).filter(Boolean))
  ).sort();

  let hasAllMethods = false;
  let methodCount = 0;
  let hasAllTools = false;
  let toolCount = 0;

  for (const permission of serverPermissions) {
    if (permission.methods.all_methods) {
      hasAllMethods = true;
    } else {
      methodCount += permission.methods.methods.length;
    }

    if (permission.tools?.all_tools) {
      hasAllTools = true;
    } else if (permission.tools?.tools) {
      toolCount += permission.tools.tools.length;
    }
  }

  return {
    servers,
    methods: hasAllMethods ? 'all' : String(methodCount),
    tools: hasAllTools ? 'all' : String(toolCount),
    agentActions: String(agentPermissions?.length ?? 0),
  };
}

function _diffLists(before: string[], after: string[]) {
  const beforeSet = new Set(before);
  const afterSet = new Set(after);

  const added = after.filter((item) => !beforeSet.has(item));
  const removed = before.filter((item) => !afterSet.has(item));

  return { added, removed };
}

function _getToolsPolicyMode(
  permission: ServerPermission
): _ToolsPolicyMode {
  if (!permission.tools) return 'none';
  if (permission.tools.all_tools) return 'all';
  return 'list';
}

export interface EditScopeModalProps {
  open: boolean;
  onClose: () => void;
  onSuccess: (scopeName: string) => void;
  scopeName: string;
  scope: ScopeDefinition;
  catalogEtag?: string;
}

export function EditScopeModal({
  open,
  onClose,
  onSuccess,
  scopeName,
  scope,
  catalogEtag,
}: EditScopeModalProps) {
  const { replaceScope, isUpdating, error } = useReplaceScope();

  const [serverPermissionsDraft, setServerPermissionsDraft] = useState<ServerPermission[]>(
    scope.server_permissions
  );
  const [agentPermissionsDraft, setAgentPermissionsDraft] = useState<
    AgentActionPermission[]
  >(scope.agent_permissions);
  const [serverPermissionsError, setServerPermissionsError] = useState<string | null>(
    null
  );
  const [agentPermissionsError, setAgentPermissionsError] = useState<string | null>(
    null
  );
  const [formError, setFormError] = useState<string | null>(null);

  useEffect(() => {
    if (!open) return;
    setServerPermissionsDraft(scope.server_permissions);
    setAgentPermissionsDraft(scope.agent_permissions);
    setServerPermissionsError(null);
    setAgentPermissionsError(null);
    setFormError(null);
  }, [open, scope.server_permissions, scope.agent_permissions]);

  const originalSummary = useMemo(
    () => _summarize(scope.server_permissions, scope.agent_permissions),
    [scope.server_permissions, scope.agent_permissions]
  );

  const draftSummary = useMemo(
    () => _summarize(serverPermissionsDraft, agentPermissionsDraft),
    [serverPermissionsDraft, agentPermissionsDraft]
  );

  const serverDiff = useMemo(() => {
    if (!originalSummary || !draftSummary) return null;
    return _diffLists(originalSummary.servers, draftSummary.servers);
  }, [originalSummary, draftSummary]);

  const handleClose = () => {
    setServerPermissionsDraft(scope.server_permissions);
    setAgentPermissionsDraft(scope.agent_permissions);
    setServerPermissionsError(null);
    setAgentPermissionsError(null);
    setFormError(null);
    onClose();
  };

  const _updateServerPermission = (
    index: number,
    updater: (current: ServerPermission) => ServerPermission
  ) => {
    setServerPermissionsDraft((prev) => {
      const updated = [...prev];
      const current = updated[index];
      if (!current) return prev;
      updated[index] = updater(current);
      return updated;
    });
  };

  const _updateAgentPermission = (
    index: number,
    updater: (current: AgentActionPermission) => AgentActionPermission
  ) => {
    setAgentPermissionsDraft((prev) => {
      const updated = [...prev];
      const current = updated[index];
      if (!current) return prev;
      updated[index] = updater(current);
      return updated;
    });
  };

  const _setToolsPolicyMode = (
    index: number,
    mode: _ToolsPolicyMode
  ) => {
    setServerPermissionsDraft((prev) => {
      const updated = [...prev];
      const current = updated[index];
      if (!current) return prev;

      if (mode === 'none') {
        updated[index] = { ...current, tools: undefined };
        return updated;
      }

      if (mode === 'all') {
        updated[index] = { ...current, tools: { all_tools: true, tools: [] } };
        return updated;
      }

      const existing = current.tools?.all_tools ? [] : (current.tools?.tools ?? []);
      updated[index] = {
        ...current,
        tools: { all_tools: false, tools: existing },
      };
      return updated;
    });
  };

  const _ensureToolsPolicyWhenRequired = (
    index: number,
    permission: ServerPermission
  ) => {
    if (!_allowsToolsCall(permission)) return;
    if (permission.tools) return;
    _setToolsPolicyMode(index, 'all');
  };

  const _addServerPermission = () => {
    setServerPermissionsDraft((prev) => [
      ...prev,
      {
        server: '',
        methods: { all_methods: false, methods: [] },
        tools: undefined,
      },
    ]);
  };

  const _removeServerPermission = (index: number) => {
    setServerPermissionsDraft((prev) => prev.filter((_, idx) => idx !== index));
  };

  const _addAgentPermission = () => {
    setAgentPermissionsDraft((prev) => [
      ...prev,
      {
        action: '',
        resources: [],
      },
    ]);
  };

  const _removeAgentPermission = (index: number) => {
    setAgentPermissionsDraft((prev) => prev.filter((_, idx) => idx !== index));
  };

  const _validateAndBuildPayload = (): ReplaceScopeRequest | null => {
    setServerPermissionsError(null);
    setAgentPermissionsError(null);
    setFormError(null);

    if (!catalogEtag) {
      setFormError('Catalog ETag missing. Refresh and try again.');
      return null;
    }

    const normalizedServerPermissions = _normalizeServerPermissions(serverPermissionsDraft);
    const normalizedAgentPermissions = _normalizeAgentPermissions(agentPermissionsDraft);

    const missingServer = normalizedServerPermissions.find((permission) => !permission.server);
    if (missingServer) {
      setServerPermissionsError('Each server permission requires a server name.');
      return null;
    }

    const toolsCallError = _validateServerPermissionsOrError(normalizedServerPermissions);
    if (toolsCallError) {
      setServerPermissionsError(toolsCallError);
      return null;
    }

    const missingAction = normalizedAgentPermissions.find(
      (permission) => permission.action.length === 0 && permission.resources.length > 0
    );
    if (missingAction) {
      setAgentPermissionsError('Each agent permission requires an action name.');
      return null;
    }

    return {
      name: scopeName,
      server_permissions: normalizedServerPermissions,
      agent_permissions: normalizedAgentPermissions,
    };
  };

  const onSubmit = async () => {
    const payload = _validateAndBuildPayload();
    if (!payload || !catalogEtag) return;

    try {
      await replaceScope(scopeName, payload, catalogEtag);
      onSuccess(scopeName);
      handleClose();
    } catch {
      // Error handled by hook
    }
  };

  const normalizedError = error ? normalizeError(error) : null;
  const showEtagGuidance = normalizedError?.status === 412;

  return (
    <Modal
      open={open}
      onClose={handleClose}
      title="Edit Scope"
      description={`Update policy scope ${scopeName} (admin only)`}
      size="xl"
    >
      <form
        onSubmit={(event) => {
          event.preventDefault();
          onSubmit();
        }}
      >
        <div className="space-y-4">
          {originalSummary && draftSummary && (
            <div className="p-3 bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md">
              <p className="text-sm text-blue-700 dark:text-blue-400">
                <strong>Changes summary:</strong> servers {originalSummary.servers.length} →{' '}
                {draftSummary.servers.length}, methods {originalSummary.methods} →{' '}
                {draftSummary.methods}, tools {originalSummary.tools} → {draftSummary.tools},
                agent actions {originalSummary.agentActions} → {draftSummary.agentActions}
              </p>
              {serverDiff && (serverDiff.added.length > 0 || serverDiff.removed.length > 0) && (
                <p className="mt-1 text-xs text-blue-600 dark:text-blue-500">
                  Servers: {serverDiff.added.length > 0 ? `+${serverDiff.added.join(', ')}` : ''}
                  {serverDiff.added.length > 0 && serverDiff.removed.length > 0 ? ' ' : ''}
                  {serverDiff.removed.length > 0 ? `-${serverDiff.removed.join(', ')}` : ''}
                </p>
              )}
            </div>
          )}

          <div className="space-y-2">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  Server permissions
                </h3>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  Define server/method/tool access for this scope.
                </p>
              </div>
              <Button type="button" variant="secondary" size="sm" onClick={_addServerPermission}>
                Add server permission
              </Button>
            </div>

            {serverPermissionsError && (
              <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
                <p className="text-sm text-red-600 dark:text-red-400">{serverPermissionsError}</p>
              </div>
            )}

            {serverPermissionsDraft.length === 0 ? (
              <div className="p-3 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md">
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  No server permissions. This scope will deny all server access.
                </p>
              </div>
            ) : (
              <div className="space-y-3">
                {serverPermissionsDraft.map((permission, index) => {
                  const toolsRequired = _allowsToolsCall(permission);
                  const showToolsPolicy = toolsRequired || permission.tools != null;
                  const toolsPolicyMode = _getToolsPolicyMode(permission);

                  return (
                    <Card key={`${permission.server}-${index}`}>
                      <div className="p-4 space-y-4">
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex-1 space-y-2">
                            <Input
                              label="Server"
                              placeholder="mcpgw"
                              value={permission.server}
                              onChange={(event) => {
                                const value = event.target.value;
                                _updateServerPermission(index, (current) => ({
                                  ...current,
                                  server: value,
                                }));
                              }}
                            />
                          </div>
                          <Button
                            type="button"
                            variant="danger"
                            size="sm"
                            onClick={() => _removeServerPermission(index)}
                          >
                            Remove
                          </Button>
                        </div>

                        <div className="space-y-2">
                          <div className="flex items-center justify-between gap-2">
                            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                              Methods
                            </p>
                            {toolsRequired && (
                              <Badge variant="warning" size="sm">
                                tools/call requires tools policy
                              </Badge>
                            )}
                          </div>
                          <Checkbox
                            label="Allow all methods"
                            checked={permission.methods.all_methods}
                            onChange={(event) => {
                              const checked = event.target.checked;
                              const nextPermission: ServerPermission = {
                                ...permission,
                                methods: { all_methods: checked, methods: [] },
                              };
                              _updateServerPermission(index, () => nextPermission);
                              _ensureToolsPolicyWhenRequired(index, nextPermission);
                            }}
                          />

                          {!permission.methods.all_methods && (
                            <div className="space-y-2">
                              {permission.methods.methods.length === 0 ? (
                                <p className="text-xs text-gray-600 dark:text-gray-400">
                                  No methods specified (deny all methods).
                                </p>
                              ) : null}

                              <div className="space-y-2">
                                {permission.methods.methods.map((method, methodIndex) => (
                                  <div key={`${methodIndex}`} className="flex items-center gap-2">
                                    <Input
                                      aria-label={`Method ${index + 1}.${methodIndex + 1}`}
                                      placeholder="tools/list"
                                      value={method}
                                      onChange={(event) => {
                                        const value = event.target.value;
                                        const nextPermission: ServerPermission = {
                                          ...permission,
                                          methods: {
                                            ...permission.methods,
                                            methods: permission.methods.methods.map((existing, idx) =>
                                              idx === methodIndex ? value : existing
                                            ),
                                          },
                                        };
                                        _updateServerPermission(index, () => nextPermission);
                                        _ensureToolsPolicyWhenRequired(index, nextPermission);
                                      }}
                                    />
                                    <Button
                                      type="button"
                                      variant="secondary"
                                      size="sm"
                                      onClick={() => {
                                        _updateServerPermission(index, (current) => {
                                          const nextMethods = current.methods.methods.filter(
                                            (_, idx) => idx !== methodIndex
                                          );
                                          return {
                                            ...current,
                                            methods: { ...current.methods, methods: nextMethods },
                                          };
                                        });
                                      }}
                                    >
                                      Remove
                                    </Button>
                                  </div>
                                ))}
                              </div>

                              <Button
                                type="button"
                                variant="secondary"
                                size="sm"
                                onClick={() => {
                                  _updateServerPermission(index, (current) => ({
                                    ...current,
                                    methods: {
                                      ...current.methods,
                                      methods: [...current.methods.methods, ''],
                                    },
                                  }));
                                }}
                              >
                                Add method
                              </Button>
                            </div>
                          )}
                        </div>

                        {showToolsPolicy && (
                          <div className="space-y-2">
                            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                              Tools policy
                            </p>
                            <div className="space-y-2">
                              <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
                                <input
                                  type="radio"
                                  name={`tools-policy-${index}`}
                                  checked={toolsPolicyMode === 'none'}
                                  disabled={toolsRequired}
                                  onChange={() => _setToolsPolicyMode(index, 'none')}
                                />
                                No tools policy
                              </label>
                              <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
                                <input
                                  type="radio"
                                  name={`tools-policy-${index}`}
                                  checked={toolsPolicyMode === 'all'}
                                  onChange={() => _setToolsPolicyMode(index, 'all')}
                                />
                                Allow all tools (*)
                              </label>
                              <label className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300">
                                <input
                                  type="radio"
                                  name={`tools-policy-${index}`}
                                  checked={toolsPolicyMode === 'list'}
                                  onChange={() => _setToolsPolicyMode(index, 'list')}
                                />
                                Allowlist
                              </label>
                            </div>

                            {toolsPolicyMode === 'list' && (
                              <div className="space-y-2">
                                {(permission.tools?.tools ?? []).map((tool, toolIndex) => (
                                  <div key={`${toolIndex}`} className="flex items-center gap-2">
                                    <Input
                                      aria-label={`Tool ${index + 1}.${toolIndex + 1}`}
                                      placeholder="tool-name"
                                      value={tool}
                                      onChange={(event) => {
                                        const value = event.target.value;
                                        _updateServerPermission(index, (current) => {
                                          const existingTools = current.tools?.tools ?? [];
                                          const nextTools = [...existingTools];
                                          nextTools[toolIndex] = value;
                                          return {
                                            ...current,
                                            tools: { all_tools: false, tools: nextTools },
                                          };
                                        });
                                      }}
                                    />
                                    <Button
                                      type="button"
                                      variant="secondary"
                                      size="sm"
                                      onClick={() => {
                                        _updateServerPermission(index, (current) => {
                                          const existingTools = current.tools?.tools ?? [];
                                          const nextTools = existingTools.filter(
                                            (_, idx) => idx !== toolIndex
                                          );
                                          return {
                                            ...current,
                                            tools: { all_tools: false, tools: nextTools },
                                          };
                                        });
                                      }}
                                    >
                                      Remove
                                    </Button>
                                  </div>
                                ))}

                                <Button
                                  type="button"
                                  variant="secondary"
                                  size="sm"
                                  onClick={() => {
                                    _updateServerPermission(index, (current) => {
                                      const existingTools = current.tools?.tools ?? [];
                                      return {
                                        ...current,
                                        tools: {
                                          all_tools: false,
                                          tools: [...existingTools, ''],
                                        },
                                      };
                                    });
                                  }}
                                >
                                  Add tool
                                </Button>
                              </div>
                            )}
                          </div>
                        )}
                      </div>
                    </Card>
                  );
                })}
              </div>
            )}
          </div>

          <div className="space-y-2">
            <div className="flex items-center justify-between gap-3">
              <div>
                <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  Agent permissions
                </h3>
                <p className="text-xs text-gray-600 dark:text-gray-400">
                  Define agent actions for this scope (optional).
                </p>
              </div>
              <Button type="button" variant="secondary" size="sm" onClick={_addAgentPermission}>
                Add agent permission
              </Button>
            </div>

            {agentPermissionsError && (
              <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
                <p className="text-sm text-red-600 dark:text-red-400">{agentPermissionsError}</p>
              </div>
            )}

            {agentPermissionsDraft.length === 0 ? (
              <div className="p-3 bg-gray-50 dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md">
                <p className="text-sm text-gray-600 dark:text-gray-400">No agent permissions.</p>
              </div>
            ) : (
              <div className="space-y-3">
                {agentPermissionsDraft.map((permission, index) => {
                  return (
                    <Card key={`${permission.action}-${index}`}>
                      <div className="p-4 space-y-4">
                        <div className="flex items-start justify-between gap-3">
                          <div className="flex-1 space-y-2">
                            <Input
                              label="Action"
                              placeholder="agents:create"
                              value={permission.action}
                              onChange={(event) => {
                                const value = event.target.value;
                                _updateAgentPermission(index, (current) => ({
                                  ...current,
                                  action: value,
                                }));
                              }}
                            />
                          </div>
                          <Button
                            type="button"
                            variant="danger"
                            size="sm"
                            onClick={() => _removeAgentPermission(index)}
                          >
                            Remove
                          </Button>
                        </div>

                        <div className="space-y-2">
                          <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                            Resources
                          </p>
                          {(permission.resources ?? []).map((resource, resourceIndex) => (
                            <div
                              key={`${resourceIndex}`}
                              className="flex items-center gap-2"
                            >
                              <Input
                                aria-label={`Resource ${index + 1}.${resourceIndex + 1}`}
                                placeholder="*"
                                value={resource}
                                onChange={(event) => {
                                  const value = event.target.value;
                                  _updateAgentPermission(index, (current) => {
                                    const nextResources = [...current.resources];
                                    nextResources[resourceIndex] = value;
                                    return { ...current, resources: nextResources };
                                  });
                                }}
                              />
                              <Button
                                type="button"
                                variant="secondary"
                                size="sm"
                                onClick={() => {
                                  _updateAgentPermission(index, (current) => {
                                    const nextResources = current.resources.filter(
                                      (_, idx) => idx !== resourceIndex
                                    );
                                    return { ...current, resources: nextResources };
                                  });
                                }}
                              >
                                Remove
                              </Button>
                            </div>
                          ))}

                          <Button
                            type="button"
                            variant="secondary"
                            size="sm"
                            onClick={() => {
                              _updateAgentPermission(index, (current) => ({
                                ...current,
                                resources: [...current.resources, ''],
                              }));
                            }}
                          >
                            Add resource
                          </Button>
                          <p className="text-xs text-gray-600 dark:text-gray-400">
                            Use <code>*</code> to match all resources.
                          </p>
                        </div>
                      </div>
                    </Card>
                  );
                })}
              </div>
            )}
          </div>

          {normalizedError && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
              <p className="text-sm text-red-600 dark:text-red-400">{normalizedError.message}</p>
              {showEtagGuidance && (
                <p className="mt-1 text-xs text-red-600 dark:text-red-400">
                  The catalog changed since you loaded it. Refresh the page and try again.
                </p>
              )}
            </div>
          )}

          {formError && !normalizedError && (
            <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
              <p className="text-sm text-red-600 dark:text-red-400">{formError}</p>
            </div>
          )}
        </div>

        <ModalFooter>
          <Button type="button" variant="secondary" onClick={handleClose} disabled={isUpdating}>
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
