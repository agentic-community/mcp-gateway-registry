/**
 * ToolsPage - Discovery page for all MCP server tools
 * Helps users understand what tools are available and configure allowed_tools
 */

import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { PageHeader } from '@/components/layout/PageHeader';
import { PageContent } from '@/components/layout/PageContent';
import { Input } from '@/components/ui/Input';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Card } from '@/components/ui/Card';
import { Spinner } from '@/components/ui/Spinner';
import { EmptyState } from '@/components/ui/EmptyState';
import { useAllServerTools } from '@/features/servers/hooks';
import {
  MagnifyingGlassIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  WrenchScrewdriverIcon,
} from '@heroicons/react/24/outline';

/**
 * ToolsPage component
 */
export default function ToolsPage() {
  const navigate = useNavigate();
  const { serversWithTools, isLoading, isError, error } = useAllServerTools();
  const [search, setSearch] = useState('');
  const [expandedServers, setExpandedServers] = useState<Set<string>>(new Set());

  // Calculate total tools count
  const totalTools = useMemo(() => {
    return serversWithTools.reduce((sum, { tools }) => sum + (tools?.length || 0), 0);
  }, [serversWithTools]);

  // Filter servers and tools by search
  const filteredServers = useMemo(() => {
    if (!search) return serversWithTools;

    const searchLower = search.toLowerCase();
    return serversWithTools
      .map((swt) => ({
        ...swt,
        tools: swt.tools?.filter(
          (tool) =>
            tool.name.toLowerCase().includes(searchLower) ||
            tool.description?.toLowerCase().includes(searchLower)
        ),
        matchesServer:
          swt.server.display_name.toLowerCase().includes(searchLower) ||
          swt.server.path.toLowerCase().includes(searchLower),
      }))
      .filter((swt) => swt.matchesServer || (swt.tools && swt.tools.length > 0));
  }, [serversWithTools, search]);

  // Toggle server expansion
  const toggleServer = (path: string) => {
    const newExpanded = new Set(expandedServers);
    if (newExpanded.has(path)) {
      newExpanded.delete(path);
    } else {
      newExpanded.add(path);
    }
    setExpandedServers(newExpanded);
  };

  // Expand all servers
  const expandAll = () => {
    setExpandedServers(new Set(filteredServers.map((swt) => swt.server.path)));
  };

  // Collapse all servers
  const collapseAll = () => {
    setExpandedServers(new Set());
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center h-full">
        <Spinner size="lg" />
        <span className="ml-3 text-gray-600 dark:text-gray-400">
          Loading tools from all servers...
        </span>
      </div>
    );
  }

  if (isError) {
    return (
      <PageContent>
        <EmptyState
          title="Failed to load tools"
          description={error?.message || 'An error occurred while loading tools'}
          icon={<WrenchScrewdriverIcon className="h-12 w-12" />}
        />
      </PageContent>
    );
  }

  return (
    <>
      <PageHeader
        title="Tools Discovery"
        description="Browse all available MCP server tools and configure agent access"
      />

      <PageContent>
        <div className="space-y-6">
          {/* Summary stats */}
          <div className="grid grid-cols-1 sm:grid-cols-3 gap-4">
            <Card>
              <div className="p-4">
                <div className="text-sm text-gray-500 dark:text-gray-400">Total Servers</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100 mt-1">
                  {serversWithTools.length}
                </div>
              </div>
            </Card>
            <Card>
              <div className="p-4">
                <div className="text-sm text-gray-500 dark:text-gray-400">Total Tools</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100 mt-1">
                  {totalTools}
                </div>
              </div>
            </Card>
            <Card>
              <div className="p-4">
                <div className="text-sm text-gray-500 dark:text-gray-400">Enabled Servers</div>
                <div className="text-2xl font-bold text-gray-900 dark:text-gray-100 mt-1">
                  {serversWithTools.filter((swt) => swt.server.is_enabled).length}
                </div>
              </div>
            </Card>
          </div>

          {/* Search and controls */}
          <div className="flex flex-col sm:flex-row gap-4">
            <div className="flex-1">
              <Input
                type="text"
                placeholder="Search tools or servers..."
                value={search}
                onChange={(e) => setSearch(e.target.value)}
                leftAddon={<MagnifyingGlassIcon className="h-5 w-5" />}
              />
            </div>
            <div className="flex gap-2">
              <Button variant="secondary" size="sm" onClick={expandAll}>
                Expand All
              </Button>
              <Button variant="secondary" size="sm" onClick={collapseAll}>
                Collapse All
              </Button>
              <Button
                variant="primary"
                size="sm"
                onClick={() => navigate('/agents')}
              >
                Configure Agent
              </Button>
            </div>
          </div>

          {/* Servers and tools list */}
          {filteredServers.length === 0 ? (
            <EmptyState
              title="No tools found"
              description={
                search
                  ? 'Try adjusting your search query.'
                  : 'Register MCP servers to see their tools here.'
              }
              icon={<MagnifyingGlassIcon className="h-12 w-12" />}
            />
          ) : (
            <div className="space-y-3">
              {filteredServers.map(({ server, tools }) => {
                const isExpanded = expandedServers.has(server.path);
                const toolCount = tools?.length || 0;

                return (
                  <Card key={server.path}>
                    <div
                      className="p-4 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-800"
                      onClick={() => toggleServer(server.path)}
                    >
                      {/* Server header */}
                      <div className="flex items-start justify-between">
                        <div className="flex items-start gap-3 flex-1">
                          <button
                            type="button"
                            className="mt-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                            onClick={(e) => {
                              e.stopPropagation();
                              toggleServer(server.path);
                            }}
                          >
                            {isExpanded ? (
                              <ChevronDownIcon className="h-5 w-5" />
                            ) : (
                              <ChevronRightIcon className="h-5 w-5" />
                            )}
                          </button>
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2 flex-wrap">
                              <h3 className="text-base font-medium text-gray-900 dark:text-gray-100">
                                {server.display_name}
                              </h3>
                              <Badge
                                variant={server.is_enabled ? 'success' : 'neutral'}
                                size="sm"
                              >
                                {server.is_enabled ? 'enabled' : 'disabled'}
                              </Badge>
                              {server.health_status && (
                                <Badge
                                  variant={
                                    server.health_status === 'healthy'
                                      ? 'success'
                                      : server.health_status === 'unhealthy'
                                      ? 'error'
                                      : 'neutral'
                                  }
                                  size="sm"
                                >
                                  {server.health_status}
                                </Badge>
                              )}
                            </div>
                            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                              {server.path}
                            </p>
                            {server.description && (
                              <p className="text-sm text-gray-600 dark:text-gray-300 mt-2">
                                {server.description}
                              </p>
                            )}
                          </div>
                        </div>
                        <div className="ml-4 text-sm text-gray-500 dark:text-gray-400">
                          {toolCount} tool{toolCount !== 1 ? 's' : ''}
                        </div>
                      </div>
                    </div>

                    {/* Tools list (expanded) */}
                    {isExpanded && (
                      <div className="border-t border-gray-200 dark:border-gray-700 px-4 py-3">
                        {tools && tools.length > 0 ? (
                          <div className="space-y-3">
                            {tools.map((tool) => (
                              <div
                                key={tool.name}
                                className="pl-8 py-2 border-l-2 border-gray-200 dark:border-gray-700"
                              >
                                <div className="font-mono text-sm text-gray-900 dark:text-gray-100">
                                  {tool.name}
                                </div>
                                {tool.description && (
                                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                                    {tool.description}
                                  </p>
                                )}
                                {tool.input_schema && (
                                  <details className="mt-2">
                                    <summary className="text-xs text-blue-600 dark:text-blue-400 cursor-pointer hover:underline">
                                      View input schema
                                    </summary>
                                    <pre className="mt-2 p-2 bg-gray-100 dark:bg-gray-900 rounded text-xs overflow-x-auto">
                                      {JSON.stringify(tool.input_schema, null, 2)}
                                    </pre>
                                  </details>
                                )}
                              </div>
                            ))}
                          </div>
                        ) : (
                          <p className="pl-8 text-sm text-gray-500 dark:text-gray-400 italic">
                            No tools available for this server
                          </p>
                        )}
                      </div>
                    )}
                  </Card>
                );
              })}
            </div>
          )}
        </div>
      </PageContent>
    </>
  );
}
