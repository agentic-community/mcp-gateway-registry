/**
 * EffectiveAccessPreview - Shows what tools an agent can actually access
 * Based on agent scopes (from catalog) and allowed_tools configuration
 */

import { useMemo, useState } from 'react';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { EmptyState } from '@/components/ui/EmptyState';
import { Spinner } from '@/components/ui/Spinner';
import { Input } from '@/components/ui/Input';
import { ServerWithTools } from '@/features/servers/hooks';
import { useScopeCatalog } from '@/features/scopes/hooks';
import { EnforceAIAgent } from '@/api/types';
import {
  InformationCircleIcon,
  ExclamationTriangleIcon,
  MagnifyingGlassIcon,
  ShieldCheckIcon,
} from '@heroicons/react/24/outline';

export interface EffectiveAccessPreviewProps {
  agent: EnforceAIAgent | null;
  serversWithTools: ServerWithTools[];
}

interface EffectiveServerAccess {
  serverPath: string;
  serverName: string;
  hasAccess: boolean;
  visibleTools: string[];
  reason: string;
}

/**
 * Calculate effective tool access for an agent
 */
function calculateEffectiveAccess(
  agent: EnforceAIAgent | null,
  serversWithTools: ServerWithTools[],
  scopeCatalog: any
): EffectiveServerAccess[] {
  if (!agent || !scopeCatalog) {
    return serversWithTools.map((swt) => ({
      serverPath: swt.server.path,
      serverName: swt.server.display_name,
      hasAccess: false,
      visibleTools: [],
      reason: 'No agent or catalog available',
    }));
  }

  const agentScopes = agent.scopes || [];
  const agentAllowedTools = agent.allowed_tools;

  return serversWithTools.map((swt) => {
    const { server, tools } = swt;
    const toolNames = tools?.map((t) => t.name) || [];

    // Check if any agent scope grants access to this server
    let hasServerAccess = false;
    let scopeGrantedTools: string[] = [];
    let allToolsGranted = false;

    for (const scopeName of agentScopes) {
      const scopeDef = scopeCatalog.scopes[scopeName];
      if (!scopeDef) continue;

      // Check server permissions
      for (const perm of scopeDef.server_permissions) {
        const matchesServer = perm.server === '*' || perm.server === server.path;
        if (!matchesServer) continue;

        hasServerAccess = true;

        // Check tool permissions within this scope
        if (perm.tools) {
          if (perm.tools.all_tools) {
            allToolsGranted = true;
            scopeGrantedTools = [...toolNames];
          } else {
            scopeGrantedTools = [
              ...new Set([...scopeGrantedTools, ...perm.tools.tools]),
            ];
          }
        }
      }
    }

    if (!hasServerAccess) {
      return {
        serverPath: server.path,
        serverName: server.display_name,
        hasAccess: false,
        visibleTools: [],
        reason: 'No scope grants access to this server',
      };
    }

    // Apply allowed_tools filter if present
    let visibleTools: string[] = [];
    if (allToolsGranted || scopeGrantedTools.length === 0) {
      // All tools from scope
      visibleTools = [...toolNames];
    } else {
      // Only tools permitted by scopes
      visibleTools = toolNames.filter((t) => scopeGrantedTools.includes(t));
    }

    // Further filter by allowed_tools if specified
    if (agentAllowedTools && agentAllowedTools.length > 0) {
      visibleTools = visibleTools.filter((t) => agentAllowedTools.includes(t));
    }

    let reason = 'Access granted by scopes';
    if (agentAllowedTools && agentAllowedTools.length > 0) {
      reason += ', filtered by allowed_tools';
    }

    return {
      serverPath: server.path,
      serverName: server.display_name,
      hasAccess: true,
      visibleTools,
      reason,
    };
  });
}

/**
 * EffectiveAccessPreview component
 */
export function EffectiveAccessPreview({
  agent,
  serversWithTools,
}: EffectiveAccessPreviewProps) {
  const { catalog, isLoading: catalogLoading } = useScopeCatalog();
  const [search, setSearch] = useState('');

  const effectiveAccess = useMemo(
    () => calculateEffectiveAccess(agent, serversWithTools, catalog),
    [agent, serversWithTools, catalog]
  );

  const filteredAccess = useMemo(() => {
    if (!search) return effectiveAccess;
    const searchLower = search.toLowerCase();
    return effectiveAccess.filter(
      (ea) =>
        ea.serverName.toLowerCase().includes(searchLower) ||
        ea.serverPath.toLowerCase().includes(searchLower) ||
        ea.visibleTools.some((t) => t.toLowerCase().includes(searchLower))
    );
  }, [effectiveAccess, search]);

  const totalVisibleTools = useMemo(
    () => effectiveAccess.reduce((sum, ea) => sum + ea.visibleTools.length, 0),
    [effectiveAccess]
  );

  const serversWithAccess = useMemo(
    () => effectiveAccess.filter((ea) => ea.hasAccess).length,
    [effectiveAccess]
  );

  if (catalogLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Spinner />
        <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
          Loading scope catalog...
        </span>
      </div>
    );
  }

  if (!catalog) {
    return (
      <EmptyState
        title="Scope catalog not available"
        description="The scope catalog is needed to preview effective access."
        icon={<InformationCircleIcon className="h-12 w-12" />}
      />
    );
  }

  if (!agent) {
    return (
      <EmptyState
        title="Select an agent"
        description="Choose an agent to preview their effective tool access."
        icon={<ShieldCheckIcon className="h-12 w-12" />}
      />
    );
  }

  return (
    <div className="space-y-6">
      {/* Disclaimer */}
      <div className="rounded-md bg-blue-50 dark:bg-blue-900/20 p-4 border border-blue-200 dark:border-blue-800">
        <div className="flex">
          <InformationCircleIcon className="h-5 w-5 text-blue-400" />
          <div className="ml-3">
            <h3 className="text-sm font-medium text-blue-800 dark:text-blue-200">
              Preview Only
            </h3>
            <div className="mt-2 text-sm text-blue-700 dark:text-blue-300">
              This preview is based on client-side calculations. Final enforcement is performed
              server-side and may differ due to additional policies or runtime conditions.
            </div>
          </div>
        </div>
      </div>

      {/* Summary */}
      <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
        <Card>
          <div className="p-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">Agent Scopes</div>
            <div className="text-2xl font-bold text-gray-900 dark:text-gray-100 mt-1">
              {agent.scopes?.length || 0}
            </div>
          </div>
        </Card>
        <Card>
          <div className="p-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">Servers with Access</div>
            <div className="text-2xl font-bold text-gray-900 dark:text-gray-100 mt-1">
              {serversWithAccess} / {serversWithTools.length}
            </div>
          </div>
        </Card>
        <Card>
          <div className="p-4">
            <div className="text-sm text-gray-500 dark:text-gray-400">Total Visible Tools</div>
            <div className="text-2xl font-bold text-gray-900 dark:text-gray-100 mt-1">
              {totalVisibleTools}
            </div>
          </div>
        </Card>
      </div>

      {/* Search */}
      <Input
        type="text"
        placeholder="Search servers or tools..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        leftAddon={<MagnifyingGlassIcon className="h-5 w-5" />}
      />

      {/* Effective access list */}
      {filteredAccess.length === 0 ? (
        <EmptyState
          title="No servers match"
          description="Try adjusting your search query."
          icon={<MagnifyingGlassIcon className="h-12 w-12" />}
        />
      ) : (
        <div className="space-y-3">
          {filteredAccess.map((ea) => (
            <Card key={ea.serverPath}>
              <div className="p-4">
                <div className="flex items-start justify-between mb-3">
                  <div>
                    <div className="flex items-center gap-2">
                      <h3 className="text-base font-medium text-gray-900 dark:text-gray-100">
                        {ea.serverName}
                      </h3>
                      <Badge variant={ea.hasAccess ? 'success' : 'error'} size="sm">
                        {ea.hasAccess ? 'accessible' : 'no access'}
                      </Badge>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {ea.serverPath}
                    </p>
                  </div>
                  <span className="text-sm text-gray-500 dark:text-gray-400">
                    {ea.visibleTools.length} tool{ea.visibleTools.length !== 1 ? 's' : ''}
                  </span>
                </div>

                <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">{ea.reason}</p>

                {ea.visibleTools.length > 0 ? (
                  <div className="flex flex-wrap gap-2">
                    {ea.visibleTools.map((toolName) => (
                      <Badge key={toolName} variant="neutral" size="sm">
                        {toolName}
                      </Badge>
                    ))}
                  </div>
                ) : (
                  ea.hasAccess && (
                    <div className="flex items-center gap-2 text-yellow-600 dark:text-yellow-400">
                      <ExclamationTriangleIcon className="h-4 w-4" />
                      <span className="text-sm">No tools visible with current configuration</span>
                    </div>
                  )
                )}
              </div>
            </Card>
          ))}
        </div>
      )}
    </div>
  );
}
