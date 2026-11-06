import React, { useState, useMemo } from 'react';
import {
  MagnifyingGlassIcon,
  PlusIcon,
  XMarkIcon,
  PencilIcon,
  ChevronDownIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';
import { ToolMapping, ToolCatalogEntry } from '../types/virtualServer';
import { useToolCatalog } from '../hooks/useVirtualServers';


/**
 * Props for the ToolSelector component.
 */
interface ToolSelectorProps {
  selectedTools: ToolMapping[];
  onToolsChange: (tools: ToolMapping[]) => void;
}


/**
 * Group catalog entries by server for display.
 */
interface ServerGroup {
  serverPath: string;
  serverName: string;
  tools: ToolCatalogEntry[];
}


/**
 * ToolSelector provides a two-panel picker for selecting tools
 * from the global tool catalog and configuring them as ToolMappings.
 *
 * Left panel: available tools grouped by server with search.
 * Right panel: selected tools with alias and version configuration.
 */
const ToolSelector: React.FC<ToolSelectorProps> = ({
  selectedTools,
  onToolsChange,
}) => {
  const { catalog, loading, error } = useToolCatalog();
  const [searchQuery, setSearchQuery] = useState('');
  const [expandedServers, setExpandedServers] = useState<Set<string>>(new Set());
  const [editingAlias, setEditingAlias] = useState<number | null>(null);

  // Group catalog tools by server
  const serverGroups: ServerGroup[] = useMemo(() => {
    const groupMap = new Map<string, ServerGroup>();

    for (const entry of catalog) {
      const existing = groupMap.get(entry.server_path);
      if (existing) {
        existing.tools.push(entry);
      } else {
        groupMap.set(entry.server_path, {
          serverPath: entry.server_path,
          serverName: entry.server_name,
          tools: [entry],
        });
      }
    }

    return Array.from(groupMap.values()).sort((a, b) =>
      a.serverName.localeCompare(b.serverName)
    );
  }, [catalog]);

  // Filter groups and tools by search
  const filteredGroups = useMemo(() => {
    if (!searchQuery) return serverGroups;
    const query = searchQuery.toLowerCase();

    return serverGroups
      .map((group) => ({
        ...group,
        tools: group.tools.filter(
          (tool) =>
            tool.tool_name.toLowerCase().includes(query) ||
            tool.description.toLowerCase().includes(query) ||
            tool.server_name.toLowerCase().includes(query)
        ),
      }))
      .filter((group) => group.tools.length > 0);
  }, [serverGroups, searchQuery]);

  const toggleServerGroup = (serverPath: string) => {
    setExpandedServers((prev) => {
      const next = new Set(prev);
      if (next.has(serverPath)) {
        next.delete(serverPath);
      } else {
        next.add(serverPath);
      }
      return next;
    });
  };

  const isToolSelected = (entry: ToolCatalogEntry): boolean => {
    return selectedTools.some(
      (t) =>
        t.tool_name === entry.tool_name &&
        t.backend_server_path === entry.server_path
    );
  };

  const areAllGroupToolsSelected = (group: ServerGroup): boolean => {
    return group.tools.every((tool) => isToolSelected(tool));
  };

  const handleAddTool = (entry: ToolCatalogEntry) => {
    if (isToolSelected(entry)) return;

    const newMapping: ToolMapping = {
      tool_name: entry.tool_name,
      backend_server_path: entry.server_path,
      alias: null,
      backend_version: null,
    };

    onToolsChange([...selectedTools, newMapping]);
  };

  const handleSelectAllFromGroup = (group: ServerGroup) => {
    const newMappings: ToolMapping[] = [];
    for (const tool of group.tools) {
      if (!isToolSelected(tool)) {
        newMappings.push({
          tool_name: tool.tool_name,
          backend_server_path: tool.server_path,
          alias: null,
          backend_version: null,
        });
      }
    }
    if (newMappings.length > 0) {
      onToolsChange([...selectedTools, ...newMappings]);
    }
  };

  const handleRemoveTool = (index: number) => {
    const updated = selectedTools.filter((_, i) => i !== index);
    onToolsChange(updated);
  };

  const handleAliasChange = (index: number, alias: string) => {
    const updated = selectedTools.map((tool, i) =>
      i === index ? { ...tool, alias: alias || null } : tool
    );
    onToolsChange(updated);
  };

  const handleVersionChange = (index: number, version: string) => {
    const updated = selectedTools.map((tool, i) =>
      i === index ? { ...tool, backend_version: version || null } : tool
    );
    onToolsChange(updated);
  };

  // Find catalog entry for a selected tool to get available versions
  const findCatalogEntry = (mapping: ToolMapping): ToolCatalogEntry | undefined => {
    return catalog.find(
      (entry) =>
        entry.tool_name === mapping.tool_name &&
        entry.server_path === mapping.backend_server_path
    );
  };

  return (
    <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
      {/* Left Panel: Available Tools */}
      <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
        <div className="bg-gray-50 dark:bg-gray-900/50 px-4 py-3 border-b border-gray-200 dark:border-gray-700">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
            Available Tools
          </h4>
          <div className="relative">
            <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
            <input
              type="text"
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              placeholder="Search tools..."
              aria-label="Search available tools"
              className="input pl-9 pr-4 py-1.5"
            />
          </div>
        </div>

        <div className="max-h-80 overflow-y-auto" role="listbox" aria-label="Available tools">
          {loading && (
            <div className="p-4 text-center text-sm text-gray-500 dark:text-gray-400">
              Loading tool catalog...
            </div>
          )}

          {error && (
            <div className="p-4 text-center text-sm text-red-500 dark:text-red-400">
              {error}
            </div>
          )}

          {!loading && !error && filteredGroups.length === 0 && (
            <div className="p-4 text-center text-sm text-gray-500 dark:text-gray-400">
              {searchQuery ? 'No matching tools found' : 'No tools available'}
            </div>
          )}

          {filteredGroups.map((group) => (
            <div key={group.serverPath}>
              <button
                onClick={() => toggleServerGroup(group.serverPath)}
                className="w-full flex items-center justify-between px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 transition-colors"
              >
                <div className="flex items-center gap-2">
                  {expandedServers.has(group.serverPath) ? (
                    <ChevronDownIcon className="h-4 w-4" />
                  ) : (
                    <ChevronRightIcon className="h-4 w-4" />
                  )}
                  <span>{group.serverName}</span>
                </div>
                <span className="text-xs bg-gray-200 dark:bg-gray-600 px-2 py-0.5 rounded-full">
                  {group.tools.length}
                </span>
              </button>

              {expandedServers.has(group.serverPath) && (
                <div className="pl-8 pr-2 pb-1">
                  {!areAllGroupToolsSelected(group) && (
                    <button
                      onClick={() => handleSelectAllFromGroup(group)}
                      className="w-full text-left px-3 py-1.5 text-xs font-medium text-teal-600 dark:text-teal-400 hover:bg-teal-50 dark:hover:bg-teal-900/20 rounded transition-colors mb-1"
                    >
                      Select All ({group.tools.length} tools)
                    </button>
                  )}
                  {group.tools.map((tool) => {
                    const selected = isToolSelected(tool);
                    return (
                      <button
                        key={`${tool.server_path}-${tool.tool_name}`}
                        onClick={() => handleAddTool(tool)}
                        disabled={selected}
                        role="option"
                        aria-selected={selected}
                        className={`w-full text-left px-3 py-2 text-sm rounded transition-colors mb-1 ${
                          selected
                            ? 'bg-teal-50 dark:bg-teal-900/20 text-teal-600 dark:text-teal-400 cursor-default'
                            : 'hover:bg-gray-100 dark:hover:bg-gray-700 text-gray-700 dark:text-gray-300'
                        }`}
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-mono text-xs">{tool.tool_name}</span>
                          {!selected && (
                            <PlusIcon className="h-4 w-4 text-gray-400" />
                          )}
                          {selected && (
                            <span className="text-xs text-teal-500">Added</span>
                          )}
                        </div>
                        {tool.description && (
                          <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5 line-clamp-1">
                            {tool.description}
                          </p>
                        )}
                      </button>
                    );
                  })}
                </div>
              )}
            </div>
          ))}
        </div>
      </div>

      {/* Right Panel: Selected Tools */}
      <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
        <div className="bg-gray-50 dark:bg-gray-900/50 px-4 py-3 border-b border-gray-200 dark:border-gray-700">
          <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Selected Tools ({selectedTools.length})
          </h4>
        </div>

        <div className="max-h-80 overflow-y-auto" role="listbox" aria-label="Selected tools">
          {selectedTools.length === 0 && (
            <div className="p-4 text-center text-sm text-gray-500 dark:text-gray-400">
              No tools selected. Click on tools from the left panel to add them.
            </div>
          )}

          {selectedTools.map((mapping, index) => {
            const catalogEntry = findCatalogEntry(mapping);
            const hasMultipleVersions =
              catalogEntry && catalogEntry.available_versions.length > 1;

            return (
              <div
                key={`${mapping.backend_server_path}-${mapping.tool_name}-${index}`}
                className="px-4 py-3 border-b border-gray-100 dark:border-gray-700 last:border-b-0"
              >
                <div className="flex items-center justify-between mb-1">
                  <div className="flex-1 min-w-0">
                    <span className="font-mono text-sm text-gray-900 dark:text-white">
                      {mapping.alias || mapping.tool_name}
                    </span>
                    {mapping.alias && (
                      <span className="text-xs text-gray-500 dark:text-gray-400 ml-2">
                        (from {mapping.tool_name})
                      </span>
                    )}
                  </div>
                  <div className="flex items-center gap-1">
                    <button
                      onClick={() =>
                        setEditingAlias(editingAlias === index ? null : index)
                      }
                      className="p-1 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded transition-colors"
                      title="Set alias"
                    >
                      <PencilIcon className="h-3.5 w-3.5" />
                    </button>
                    <button
                      onClick={() => handleRemoveTool(index)}
                      className="p-1 text-gray-400 hover:text-red-500 rounded transition-colors"
                      title="Remove tool"
                    >
                      <XMarkIcon className="h-3.5 w-3.5" />
                    </button>
                  </div>
                </div>

                <div className="text-xs text-gray-500 dark:text-gray-400">
                  {mapping.backend_server_path}
                </div>

                {/* Alias input */}
                {editingAlias === index && (
                  <div className="mt-2">
                    <input
                      type="text"
                      value={mapping.alias || ''}
                      onChange={(e) => handleAliasChange(index, e.target.value)}
                      placeholder="Tool alias (optional)"
                      className="input px-2 py-1 text-xs"
                    />
                  </div>
                )}

                {/* Version selector */}
                {hasMultipleVersions && (
                  <div className="mt-2">
                    <select
                      value={mapping.backend_version || ''}
                      onChange={(e) => handleVersionChange(index, e.target.value)}
                      className="input pl-2 py-1 text-xs"
                    >
                      <option value="">Default version</option>
                      {catalogEntry.available_versions.map((v) => (
                        <option key={v} value={v}>
                          {v}
                        </option>
                      ))}
                    </select>
                  </div>
                )}
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export default ToolSelector;
