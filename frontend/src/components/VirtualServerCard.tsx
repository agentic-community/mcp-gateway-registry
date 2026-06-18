import React, { useState, useCallback } from 'react';
import axios from 'axios';
import {
  PencilIcon,
  TrashIcon,
  LinkIcon,
  WrenchScrewdriverIcon,
  XMarkIcon,
  ChevronDownIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';
import { VirtualServerInfo, ResolvedTool } from '../types/virtualServer';
import ServerConfigModal from './ServerConfigModal';
import StarRatingWidget from './StarRatingWidget';
import useEscapeKey from '../hooks/useEscapeKey';


/**
 * Props for the VirtualServerCard component.
 */
interface VirtualServerCardProps {
  virtualServer: VirtualServerInfo;
  canModify: boolean;
  onToggle: (path: string, enabled: boolean) => void;
  onEdit: (server: VirtualServerInfo) => void;
  onDelete: (path: string) => void;
  onShowToast?: (message: string, type: 'success' | 'error' | 'info') => void;
  onServerUpdate?: (path: string, updates: Partial<VirtualServerInfo>) => void;
  authToken?: string | null;
}


/**
 * VirtualServerCard renders a dashboard card for a virtual MCP server.
 *
 * Uses a teal/cyan gradient for visual distinction from regular ServerCard.
 * Matches the layout and UI elements of the regular ServerCard.
 */
const VirtualServerCard: React.FC<VirtualServerCardProps> = ({
  virtualServer: server,
  canModify,
  onToggle,
  onEdit,
  onDelete,
  onShowToast,
  onServerUpdate,
  authToken,
}) => {
  const [showTools, setShowTools] = useState(false);
  const [tools, setTools] = useState<ResolvedTool[]>([]);
  const [loadingTools, setLoadingTools] = useState(false);
  const [expandedBackends, setExpandedBackends] = useState<Record<string, boolean>>({});
  const [expandedTools, setExpandedTools] = useState<Record<string, boolean>>({});
  const [showConfig, setShowConfig] = useState(false);

  useEscapeKey(() => setShowTools(false), showTools);

  const handleViewTools = useCallback(async () => {
    if (loadingTools) return;

    setShowTools(true);
    setLoadingTools(true);

    try {
      // Fetch resolved tools with full details (description, schema)
      const response = await axios.get<{ tools: ResolvedTool[] }>(
        `/api/virtual-servers${server.path}/tools`
      );
      const resolvedTools = response.data.tools || [];
      setTools(resolvedTools);

      // Group tools by backend to determine collapse state
      const toolsByBackend: Record<string, ResolvedTool[]> = {};
      for (const tool of resolvedTools) {
        const backend = tool.backend_server_path;
        if (!toolsByBackend[backend]) {
          toolsByBackend[backend] = [];
        }
        toolsByBackend[backend].push(tool);
      }

      // Auto-expand first backend, collapse tools if more than 3 in any backend
      const backends = Object.keys(toolsByBackend);
      if (backends.length > 0) {
        setExpandedBackends({ [backends[0]]: true });
      }

      // If any backend has more than 3 tools, collapse all tools by default
      // Otherwise expand all tools
      const hasLargeBackend = Object.values(toolsByBackend).some(t => t.length > 3);
      if (!hasLargeBackend) {
        // Expand all tools if small number of tools
        const allToolsExpanded: Record<string, boolean> = {};
        for (const tool of resolvedTools) {
          allToolsExpanded[tool.name] = true;
        }
        setExpandedTools(allToolsExpanded);
      } else {
        setExpandedTools({});
      }
    } catch (error) {
      console.error('Failed to fetch tools:', error);
      onShowToast?.('Failed to load tools', 'error');
      setTools([]);
    } finally {
      setLoadingTools(false);
    }
  }, [server.path, loadingTools, onShowToast]);

  const toggleBackend = (backend: string) => {
    setExpandedBackends(prev => ({
      ...prev,
      [backend]: !prev[backend]
    }));
  };

  const toggleTool = (toolName: string) => {
    setExpandedTools(prev => ({
      ...prev,
      [toolName]: !prev[toolName]
    }));
  };

  // Group tools by backend server
  const toolsByBackend = tools.reduce<Record<string, ResolvedTool[]>>((acc, tool) => {
    const backend = tool.backend_server_path;
    if (!acc[backend]) {
      acc[backend] = [];
    }
    acc[backend].push(tool);
    return acc;
  }, {});

  const backendPaths = Object.keys(toolsByBackend);

  // Create a Server-like object for ServerConfigModal
  const serverForConfig = {
    name: server.server_name,
    path: server.path,
    description: server.description,
    enabled: server.is_enabled,
    tags: server.tags,
  };

  return (
    <>
      <div className="group rounded-2xl shadow-sm hover:shadow-xl transition-all duration-300 h-full flex flex-col bg-gradient-to-br from-teal-50 to-cyan-50 dark:from-teal-900/20 dark:to-cyan-900/20 border-2 border-teal-200 dark:border-teal-700 hover:border-teal-300 dark:hover:border-teal-600">
        {/* Header */}
        <div className="p-5 pb-4">
          <div className="flex items-start justify-between mb-4">
            <div className="flex-1 min-w-0">
              <div className="flex items-center gap-2 mb-3">
                <h3 className="text-lg font-bold text-gray-900 dark:text-white truncate">
                  {server.server_name}
                </h3>
                <span className="px-2 py-0.5 text-xs font-semibold bg-teal-100 text-teal-700 dark:bg-teal-900/30 dark:text-teal-300 rounded-full flex-shrink-0 border border-teal-200 dark:border-teal-600">
                  VIRTUAL
                </span>
              </div>

              <code className="text-xs text-gray-600 dark:text-gray-300 bg-gray-50 dark:bg-gray-800/50 px-2 py-1 rounded font-mono">
                {server.path}
              </code>
            </div>

            <div className="flex items-center gap-1 flex-shrink-0">
              {canModify && (
                <button
                  className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50 rounded-lg transition-all duration-200"
                  onClick={() => onEdit(server)}
                  title="Edit virtual server"
                >
                  <PencilIcon className="h-4 w-4" />
                </button>
              )}

              {/* Connect Button */}
              <button
                onClick={() => setShowConfig(true)}
                className="flex items-center gap-1 px-2 py-1.5 text-xs font-medium text-green-600 dark:text-green-400 hover:bg-green-50 dark:hover:bg-green-700/50 rounded-lg transition-all duration-200 flex-shrink-0 border border-green-200 dark:border-green-700"
                title="Get connection details and mcp.json configuration"
                aria-label={`Connect to ${server.server_name}`}
              >
                <LinkIcon className="h-3.5 w-3.5" />
                Connect
              </button>

              {canModify && (
                <button
                  onClick={() => onDelete(server.path)}
                  className="p-2 text-gray-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-700/50 rounded-lg transition-all duration-200"
                  title="Delete virtual server"
                >
                  <TrashIcon className="h-4 w-4" />
                </button>
              )}
            </div>
          </div>

          {/* Description */}
          <p className="text-gray-600 dark:text-gray-300 text-sm leading-relaxed line-clamp-2 mb-4">
            {server.description || 'No description available'}
          </p>

          {/* Tags */}
          {server.tags && server.tags.length > 0 && (
            <div className="flex flex-wrap gap-1.5 mb-4">
              {server.tags.slice(0, 3).map((tag) => (
                <span
                  key={tag}
                  className="px-2 py-1 text-xs font-medium bg-teal-50 dark:bg-teal-900/30 text-teal-700 dark:text-teal-300 rounded"
                >
                  #{tag}
                </span>
              ))}
              {server.tags.length > 3 && (
                <span className="px-2 py-1 text-xs font-medium bg-gray-50 dark:bg-gray-800 text-gray-600 dark:text-gray-300 rounded">
                  +{server.tags.length - 3}
                </span>
              )}
            </div>
          )}
        </div>

        {/* Stats - 2-column layout */}
        <div className="px-5 pb-4">
          <div className="grid grid-cols-2 gap-4">
            {/* Rating */}
            <StarRatingWidget
              resourceType="virtual-servers"
              path={server.path}
              initialRating={server.num_stars || 0}
              initialCount={server.rating_details?.length || 0}
              ratingDetails={server.rating_details}
              authToken={authToken}
              onShowToast={onShowToast}
              onRatingUpdate={(newRating) => {
                onServerUpdate?.(server.path, { num_stars: newRating });
              }}
            />

            {/* Tools - clickable */}
            <div className="flex items-center gap-2">
              {server.tool_count > 0 ? (
                <button
                  onClick={handleViewTools}
                  disabled={loadingTools}
                  className="flex items-center gap-2 text-teal-600 hover:text-teal-700 dark:text-teal-400 dark:hover:text-teal-300 disabled:opacity-50 hover:bg-teal-50 dark:hover:bg-teal-900/20 px-2 py-1 -mx-2 -my-1 rounded transition-all"
                  title="View tools"
                >
                  <div className="p-1.5 bg-teal-50 dark:bg-teal-900/30 rounded">
                    <WrenchScrewdriverIcon className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="text-sm font-semibold">{server.tool_count}</div>
                    <div className="text-xs">Tools</div>
                  </div>
                </button>
              ) : (
                <div className="flex items-center gap-2 text-gray-400 dark:text-gray-500">
                  <div className="p-1.5 bg-gray-50 dark:bg-gray-800 rounded">
                    <WrenchScrewdriverIcon className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="text-sm font-semibold">0</div>
                    <div className="text-xs">Tools</div>
                  </div>
                </div>
              )}
            </div>

          </div>
        </div>

        {/* Footer */}
        <div className="mt-auto px-5 py-4 border-t border-teal-100 dark:border-teal-800 bg-teal-50/50 dark:bg-teal-900/10 rounded-b-2xl">
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${
                server.is_enabled
                  ? 'bg-green-400 shadow-lg shadow-green-400/30'
                  : 'bg-gray-300 dark:bg-gray-600'
              }`} />
              <span className="text-sm font-medium text-gray-700 dark:text-gray-300">
                {server.is_enabled ? 'Enabled' : 'Disabled'}
              </span>
            </div>

            {/* Toggle Switch */}
            {canModify && (
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={server.is_enabled}
                  onChange={(e) => onToggle(server.path, e.target.checked)}
                  className="sr-only peer"
                  aria-label={`Enable ${server.server_name}`}
                />
                <div className={`relative w-12 h-6 rounded-full transition-colors duration-200 ease-in-out ${
                  server.is_enabled
                    ? 'bg-teal-600'
                    : 'bg-gray-300 dark:bg-gray-600'
                }`}>
                  <div className={`absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform duration-200 ease-in-out ${
                    server.is_enabled ? 'translate-x-6' : 'translate-x-0'
                  }`} />
                </div>
              </label>
            )}
          </div>
        </div>
      </div>

      {/* Tools Modal */}
      {showTools && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
          <div className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-auto">
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Tools for {server.server_name}
              </h3>
              <button
                onClick={() => setShowTools(false)}
                className="p-2 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>

            {loadingTools ? (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-teal-600"></div>
                <span className="ml-3 text-gray-500">Loading tools...</span>
              </div>
            ) : tools.length > 0 ? (
              <div className="space-y-3">
                {backendPaths.map((backend) => {
                  const backendTools = toolsByBackend[backend];
                  const isBackendExpanded = expandedBackends[backend];

                  return (
                    <div key={backend} className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
                      <button
                        onClick={() => toggleBackend(backend)}
                        className="w-full flex items-center justify-between px-4 py-3 bg-gray-50 dark:bg-gray-900/50 hover:bg-gray-100 dark:hover:bg-gray-800 transition-colors text-left"
                      >
                        <div className="flex items-center gap-2">
                          {isBackendExpanded ? (
                            <ChevronDownIcon className="h-4 w-4 text-gray-500" />
                          ) : (
                            <ChevronRightIcon className="h-4 w-4 text-gray-500" />
                          )}
                          <span className="text-sm font-mono text-gray-700 dark:text-gray-200">
                            {backend}
                          </span>
                        </div>
                        <span className="px-2 py-0.5 text-xs bg-teal-100 dark:bg-teal-900/40 text-teal-700 dark:text-teal-300 rounded-full">
                          {backendTools.length} tool{backendTools.length !== 1 ? 's' : ''}
                        </span>
                      </button>

                      {isBackendExpanded && (
                        <ul className="border-t border-gray-200 dark:border-gray-700 divide-y divide-gray-100 dark:divide-gray-800">
                          {backendTools.map((tool) => {
                            const isToolExpanded = expandedTools[tool.name];
                            const hasDetails = tool.description || (tool.input_schema && Object.keys(tool.input_schema).length > 0);

                            return (
                              <li
                                key={tool.name}
                                className="bg-white dark:bg-gray-800"
                              >
                                {/* Tool header - clickable to expand */}
                                <button
                                  onClick={() => hasDetails && toggleTool(tool.name)}
                                  className={`w-full px-4 py-3 text-left ${hasDetails ? 'cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-700/50' : 'cursor-default'}`}
                                  disabled={!hasDetails}
                                >
                                  <div className="flex items-start justify-between gap-2">
                                    <div className="flex items-center gap-2 flex-1 min-w-0">
                                      {hasDetails && (
                                        isToolExpanded ? (
                                          <ChevronDownIcon className="h-3 w-3 text-gray-400 flex-shrink-0" />
                                        ) : (
                                          <ChevronRightIcon className="h-3 w-3 text-gray-400 flex-shrink-0" />
                                        )
                                      )}
                                      {!hasDetails && <div className="w-3" />}
                                      <span className="font-medium text-sm text-gray-900 dark:text-white">
                                        {tool.name}
                                      </span>
                                      {tool.original_name && tool.name !== tool.original_name && (
                                        <span className="text-xs text-gray-400 dark:text-gray-500">
                                          (original: {tool.original_name})
                                        </span>
                                      )}
                                    </div>
                                    {tool.backend_version && (
                                      <span className="px-1.5 py-0.5 text-[10px] bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 rounded font-mono flex-shrink-0">
                                        v{tool.backend_version}
                                      </span>
                                    )}
                                  </div>
                                </button>

                                {/* Expanded tool details */}
                                {isToolExpanded && hasDetails && (
                                  <div className="px-4 pb-3 pt-0 space-y-3">
                                    {/* Description */}
                                    {tool.description && (
                                      <div className="ml-5">
                                        <p className="text-xs text-gray-600 dark:text-gray-400 leading-relaxed whitespace-pre-wrap">
                                          {tool.description}
                                        </p>
                                      </div>
                                    )}

                                    {/* Schema */}
                                    {tool.input_schema && Object.keys(tool.input_schema).length > 0 && (
                                      <div className="ml-5">
                                        <details className="text-xs">
                                          <summary className="cursor-pointer text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-300 font-medium">
                                            View Schema
                                          </summary>
                                          <pre className="mt-2 p-3 bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700 rounded overflow-x-auto text-gray-800 dark:text-gray-200">
                                            {JSON.stringify(tool.input_schema, null, 2)}
                                          </pre>
                                        </details>
                                      </div>
                                    )}

                                    {/* Required scopes */}
                                    {tool.required_scopes && tool.required_scopes.length > 0 && (
                                      <div className="ml-5 flex flex-wrap gap-1">
                                        {tool.required_scopes.map((scope) => (
                                          <span
                                            key={scope}
                                            className="px-1.5 py-0.5 text-[10px] bg-amber-50 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 rounded font-mono"
                                          >
                                            {scope}
                                          </span>
                                        ))}
                                      </div>
                                    )}
                                  </div>
                                )}
                              </li>
                            );
                          })}
                        </ul>
                      )}
                    </div>
                  );
                })}
              </div>
            ) : (
              <p className="text-gray-500 dark:text-gray-300 text-center py-8">
                No tools available for this virtual server.
              </p>
            )}
          </div>
        </div>
      )}

      {/* ServerConfigModal - reusing exact same component as ServerCard */}
      <ServerConfigModal
        server={serverForConfig as any}
        isOpen={showConfig}
        onClose={() => setShowConfig(false)}
        onShowToast={onShowToast}
        resourceType="virtual_server"
      />
    </>
  );
};

export default VirtualServerCard;
