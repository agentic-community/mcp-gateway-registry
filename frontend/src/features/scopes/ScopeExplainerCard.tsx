import {
  InformationCircleIcon,
  ServerIcon,
  WrenchScrewdriverIcon,
  EyeIcon,
  PlayIcon,
} from '@heroicons/react/24/outline';
import { Badge } from '@/components/ui';
import type { ScopeDefinition, ServerPermission } from '@/api/types';

// ============================================================================
// Types
// ============================================================================

interface ScopeExplainerCardProps {
  scope: ScopeDefinition;
}

// ============================================================================
// Helper Components
// ============================================================================

interface ServerPermissionRowProps {
  permission: ServerPermission;
}

function ServerPermissionRow({ permission }: ServerPermissionRowProps) {
  const isWildcardServer = permission.server === '*';
  const hasAllMethods = permission.methods.all_methods;
  const hasAllTools = permission.tools?.all_tools;

  return (
    <div className="bg-gray-50 dark:bg-gray-900 rounded-md p-3 space-y-2">
      {/* Server */}
      <div className="flex items-center space-x-2">
        <ServerIcon className="w-4 h-4 text-gray-500" />
        <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
          {isWildcardServer ? (
            <span className="text-amber-600 dark:text-amber-400">
              All Servers (*)
            </span>
          ) : (
            permission.server
          )}
        </span>
      </div>

      {/* Methods */}
      <div className="ml-6">
        <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Methods:</p>
        <div className="flex flex-wrap gap-1">
          {hasAllMethods ? (
            <Badge variant="warning" size="sm">
              All methods
            </Badge>
          ) : (
            permission.methods.methods.map((method) => (
              <Badge key={method} variant="neutral" size="sm">
                {method}
              </Badge>
            ))
          )}
        </div>
      </div>

      {/* Tools (if specified) */}
      {permission.tools && (
        <div className="ml-6">
          <p className="text-xs text-gray-500 dark:text-gray-400 mb-1">Tools:</p>
          <div className="flex flex-wrap gap-1">
            {hasAllTools ? (
              <Badge variant="warning" size="sm">
                All tools (*)
              </Badge>
            ) : permission.tools.tools.length > 0 ? (
              permission.tools.tools.map((tool) => (
                <Badge key={tool} variant="info" size="sm">
                  {tool}
                </Badge>
              ))
            ) : (
              <span className="text-xs text-gray-400">No specific tools</span>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function ScopeExplainerCard({ scope }: ScopeExplainerCardProps) {
  const hasServerPermissions = scope.server_permissions.length > 0;
  const hasAgentPermissions = scope.agent_permissions.length > 0;

  // Determine what capabilities this scope grants
  const capabilities: { icon: typeof EyeIcon; label: string; color: string }[] = [];

  if (hasServerPermissions) {
    const hasToolsList = scope.server_permissions.some(
      (p) => p.methods.all_methods || p.methods.methods.includes('tools/list')
    );
    const hasToolsCall = scope.server_permissions.some(
      (p) => p.methods.all_methods || p.methods.methods.includes('tools/call')
    );

    if (hasToolsList) {
      capabilities.push({
        icon: EyeIcon,
        label: 'Can discover tools (tools/list)',
        color: 'text-blue-600 dark:text-blue-400',
      });
    }
    if (hasToolsCall) {
      capabilities.push({
        icon: PlayIcon,
        label: 'Can execute tools (tools/call)',
        color: 'text-green-600 dark:text-green-400',
      });
    }
  }

  return (
    <div className="space-y-4">
      {/* What does this scope allow? */}
      <div>
        <div className="flex items-center space-x-2 mb-3">
          <InformationCircleIcon className="w-5 h-5 text-blue-500" />
          <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100">
            What does this scope allow?
          </h4>
        </div>

        {/* Capabilities Summary */}
        {capabilities.length > 0 && (
          <div className="mb-4 space-y-1">
            {capabilities.map((cap, idx) => (
              <div key={idx} className="flex items-center space-x-2">
                <cap.icon className={`w-4 h-4 ${cap.color}`} />
                <span className="text-sm text-gray-700 dark:text-gray-300">
                  {cap.label}
                </span>
              </div>
            ))}
          </div>
        )}

        {/* Server Permissions */}
        {hasServerPermissions && (
          <div className="mb-4">
            <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
              Server Access ({scope.server_permissions.length})
            </h5>
            <div className="space-y-2">
              {scope.server_permissions.map((perm, idx) => (
                <ServerPermissionRow key={`${perm.server}-${idx}`} permission={perm} />
              ))}
            </div>
          </div>
        )}

        {/* Agent Permissions */}
        {hasAgentPermissions && (
          <div>
            <h5 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-2">
              Agent Actions ({scope.agent_permissions.length})
            </h5>
            <div className="bg-gray-50 dark:bg-gray-900 rounded-md p-3">
              {scope.agent_permissions.map((perm, idx) => (
                <div key={idx} className="mb-2 last:mb-0">
                  <div className="flex items-center space-x-2 mb-1">
                    <WrenchScrewdriverIcon className="w-4 h-4 text-gray-500" />
                    <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                      {perm.action}
                    </span>
                  </div>
                  <div className="ml-6 flex flex-wrap gap-1">
                    {perm.resources.map((resource) => (
                      <Badge
                        key={resource}
                        variant={resource === 'all' ? 'warning' : 'neutral'}
                        size="sm"
                      >
                        {resource}
                      </Badge>
                    ))}
                  </div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* No permissions case */}
        {!hasServerPermissions && !hasAgentPermissions && (
          <p className="text-sm text-gray-500 dark:text-gray-400 italic">
            This scope does not grant any direct permissions. It may be used for
            grouping or inheritance purposes.
          </p>
        )}
      </div>

      {/* Usage Note */}
      <div className="border-t border-gray-200 dark:border-gray-700 pt-3">
        <p className="text-xs text-gray-500 dark:text-gray-400">
          Assign this scope to agents to grant these permissions. Scopes can be
          combined to provide cumulative access.
        </p>
      </div>
    </div>
  );
}
