/**
 * ToolPicker - Reusable multi-select component for choosing tools from servers
 * Groups tools by server and allows selection across multiple servers
 */

import { useState, useMemo } from 'react';
import { Checkbox } from '@/components/ui/Checkbox';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { Spinner } from '@/components/ui/Spinner';
import { EmptyState } from '@/components/ui/EmptyState';
import { cn } from '@/lib/cn';
import { MagnifyingGlassIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';
import { ServerWithTools } from '@/features/servers/hooks';

export interface ToolPickerProps {
  serversWithTools: ServerWithTools[];
  selectedTools: string[];
  onChange: (tools: string[]) => void;
  isLoading?: boolean;
  error?: Error | null;
  showDuplicateWarning?: boolean;
}

/**
 * ToolPicker component
 */
export function ToolPicker({
  serversWithTools,
  selectedTools,
  onChange,
  isLoading = false,
  error = null,
  showDuplicateWarning = true,
}: ToolPickerProps) {
  const [search, setSearch] = useState('');

  // Find duplicate tool names across servers
  const duplicateToolNames = useMemo(() => {
    const toolCounts = new Map<string, number>();
    serversWithTools.forEach(({ tools }) => {
      tools?.forEach((tool) => {
        toolCounts.set(tool.name, (toolCounts.get(tool.name) || 0) + 1);
      });
    });
    return new Set(
      Array.from(toolCounts.entries())
        .filter(([, count]) => count > 1)
        .map(([name]) => name)
    );
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
      }))
      .filter((swt) => swt.tools && swt.tools.length > 0);
  }, [serversWithTools, search]);

  // Handle tool selection
  const handleToolToggle = (toolName: string) => {
    if (selectedTools.includes(toolName)) {
      onChange(selectedTools.filter((t) => t !== toolName));
    } else {
      onChange([...selectedTools, toolName]);
    }
  };

  // Handle select all for a server
  const handleServerSelectAll = (serverTools: string[]) => {
    const allSelected = serverTools.every((tool) => selectedTools.includes(tool));
    if (allSelected) {
      onChange(selectedTools.filter((t) => !serverTools.includes(t)));
    } else {
      const newSelected = new Set([...selectedTools, ...serverTools]);
      onChange(Array.from(newSelected));
    }
  };

  if (isLoading) {
    return (
      <div className="flex items-center justify-center py-8">
        <Spinner />
        <span className="ml-2 text-sm text-gray-600 dark:text-gray-400">
          Loading tools...
        </span>
      </div>
    );
  }

  if (error) {
    return (
      <EmptyState
        title="Failed to load tools"
        description={error.message}
        icon={<ExclamationTriangleIcon className="h-12 w-12" />}
      />
    );
  }

  if (serversWithTools.length === 0) {
    return (
      <EmptyState
        title="No servers available"
        description="Register MCP servers to see their tools here."
      />
    );
  }

  return (
    <div className="space-y-4">
      {/* Search */}
      <Input
        type="text"
        placeholder="Search tools..."
        value={search}
        onChange={(e) => setSearch(e.target.value)}
        leftAddon={<MagnifyingGlassIcon className="h-5 w-5" />}
      />

      {/* Duplicate warning */}
      {showDuplicateWarning && duplicateToolNames.size > 0 && (
        <div className="rounded-md bg-yellow-50 dark:bg-yellow-900/20 p-3 border border-yellow-200 dark:border-yellow-800">
          <div className="flex">
            <ExclamationTriangleIcon className="h-5 w-5 text-yellow-400" />
            <div className="ml-3">
              <h3 className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                Duplicate tool names detected
              </h3>
              <div className="mt-2 text-sm text-yellow-700 dark:text-yellow-300">
                <p>The following tools appear in multiple servers:</p>
                <div className="mt-1 flex flex-wrap gap-1">
                  {Array.from(duplicateToolNames).map((name) => (
                    <Badge key={name} variant="warning" size="sm">
                      {name}
                    </Badge>
                  ))}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Selected count */}
      {selectedTools.length > 0 && (
        <div className="text-sm text-gray-600 dark:text-gray-400">
          {selectedTools.length} tool{selectedTools.length !== 1 ? 's' : ''} selected
        </div>
      )}

      {/* Tools grouped by server */}
      <div className="space-y-4 max-h-96 overflow-y-auto">
        {filteredServers.length === 0 ? (
          <EmptyState
            title="No tools found"
            description="Try adjusting your search query."
            icon={<MagnifyingGlassIcon className="h-12 w-12" />}
          />
        ) : (
          filteredServers.map(({ server, tools }) => {
            const serverToolNames = tools?.map((t) => t.name) || [];
            const allSelected = serverToolNames.every((tool) =>
              selectedTools.includes(tool)
            );
            const someSelected = serverToolNames.some((tool) =>
              selectedTools.includes(tool)
            );

            return (
              <div
                key={server.path}
                className="border border-gray-200 dark:border-gray-700 rounded-lg p-4"
              >
                {/* Server header */}
                <div className="flex items-start justify-between mb-3">
                  <div className="flex-1">
                    <div className="flex items-center gap-2">
                      <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                        {server.display_name}
                      </h3>
                      <Badge variant={server.is_enabled ? 'success' : 'neutral'} size="sm">
                        {server.is_enabled ? 'enabled' : 'disabled'}
                      </Badge>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                      {server.path}
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => handleServerSelectAll(serverToolNames)}
                    className="text-xs text-blue-600 dark:text-blue-400 hover:underline"
                  >
                    {allSelected ? 'Deselect all' : 'Select all'}
                  </button>
                </div>

                {/* Tools list */}
                {tools && tools.length > 0 ? (
                  <div className="space-y-2">
                    {tools.map((tool) => {
                      const isDuplicate = duplicateToolNames.has(tool.name);
                      return (
                        <label
                          key={tool.name}
                          className={cn(
                            'flex items-start gap-3 p-2 rounded hover:bg-gray-50 dark:hover:bg-gray-800 cursor-pointer',
                            isDuplicate && 'border-l-2 border-yellow-400'
                          )}
                        >
                          <Checkbox
                            checked={selectedTools.includes(tool.name)}
                            onChange={() => handleToolToggle(tool.name)}
                          />
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center gap-2">
                              <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                                {tool.name}
                              </span>
                              {isDuplicate && (
                                <Badge variant="warning" size="sm">
                                  duplicate
                                </Badge>
                              )}
                            </div>
                            {tool.description && (
                              <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
                                {tool.description}
                              </p>
                            )}
                          </div>
                        </label>
                      );
                    })}
                  </div>
                ) : (
                  <p className="text-sm text-gray-500 dark:text-gray-400 italic">
                    No tools available for this server
                  </p>
                )}
              </div>
            );
          })
        )}
      </div>
    </div>
  );
}
