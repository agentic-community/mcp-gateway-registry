import React, { useState, useEffect, useMemo } from 'react';
import { VirtualServerInfo, VirtualServerConfig, ToolMapping } from '../types/virtualServer';
import axios from 'axios';
import ResourceBoundTokenButton from './ResourceBoundTokenButton';
import { EntityModal, CollapsibleSection } from './modals';


interface VirtualServerDetailsModalProps {
  virtualServer: VirtualServerInfo;
  isOpen: boolean;
  onClose: () => void;
}


const VirtualServerDetailsModal: React.FC<VirtualServerDetailsModalProps> = ({
  virtualServer,
  isOpen,
  onClose
}) => {
  const [fullConfig, setFullConfig] = useState<VirtualServerConfig | null>(null);
  const [loading, setLoading] = useState(false);
  const [expandedBackends, setExpandedBackends] = useState<Record<string, boolean>>({});

  // Fetch full config when modal opens
  useEffect(() => {
    if (!isOpen || !virtualServer?.path) {
      setFullConfig(null);
      return;
    }

    const fetchConfig = async () => {
      setLoading(true);
      try {
        const response = await axios.get<VirtualServerConfig>(
          `/api/virtual-servers${virtualServer.path}`
        );
        setFullConfig(response.data);
        // Auto-expand first backend
        if (response.data.tool_mappings?.length > 0) {
          const firstBackend = response.data.tool_mappings[0].backend_server_path;
          setExpandedBackends({ [firstBackend]: true });
        }
      } catch (err) {
        console.error('Failed to fetch virtual server config:', err);
      } finally {
        setLoading(false);
      }
    };

    fetchConfig();
  }, [isOpen, virtualServer?.path]);

  // Group tools by backend server
  const toolsByBackend = useMemo(() => {
    const tools = fullConfig?.tool_mappings || virtualServer.tool_mappings || [];
    const grouped: Record<string, ToolMapping[]> = {};

    for (const tool of tools) {
      const backend = tool.backend_server_path;
      if (!grouped[backend]) {
        grouped[backend] = [];
      }
      grouped[backend].push(tool);
    }

    return grouped;
  }, [fullConfig, virtualServer.tool_mappings]);

  const toggleBackend = (backend: string) => {
    setExpandedBackends(prev => ({
      ...prev,
      [backend]: !prev[backend]
    }));
  };

  const backendPaths = virtualServer.backend_paths || [];
  const hasToolDetails = Object.keys(toolsByBackend).length > 0;

  return (
    <EntityModal
      isOpen={isOpen}
      onClose={onClose}
      maxWidth="2xl"
      layout="flush"
      title={
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              {virtualServer.server_name}
            </h3>
            <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-virtual-100 text-virtual-700 dark:bg-virtual-900/40 dark:text-virtual-200 border border-virtual-200 dark:border-virtual-600">
              VIRTUAL
            </span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">{virtualServer.path}</p>
        </div>
      }
    >
      <div className="space-y-4">
          {/* Description */}
          <div>
            <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
              Description
            </p>
            <p className="text-sm text-gray-700 dark:text-gray-200">
              {virtualServer.description || 'No description available.'}
            </p>
          </div>

          {virtualServer.path && (
            <ResourceBoundTokenButton
              resourceType="virtual_server"
              resourceId={virtualServer.path}
              resourceName={virtualServer.server_name}
            />
          )}

          {/* Tags */}
          {virtualServer.tags && virtualServer.tags.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
                Tags
              </p>
              <div className="flex flex-wrap gap-2">
                {virtualServer.tags.map((tag) => (
                  <span
                    key={tag}
                    className="px-2.5 py-1 text-xs rounded-full bg-virtual-50 text-virtual-700 dark:bg-virtual-900/40 dark:text-virtual-200"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Backend Servers with Tools */}
          {backendPaths.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
                Backend Servers ({backendPaths.length}) - Tools ({virtualServer.tool_count})
              </p>
              {loading ? (
                <div className="flex items-center justify-center py-4">
                  <div className="animate-spin rounded-full h-5 w-5 border-b-2 border-virtual-600 dark:border-virtual-400"></div>
                  <span className="ml-2 text-sm text-fg-subtle">Loading tool details...</span>
                </div>
              ) : (
                <ul className="space-y-2">
                  {backendPaths.map((path) => {
                    const backendTools = toolsByBackend[path] || [];
                    const toolCount = backendTools.length;

                    return (
                      <li key={path}>
                        <CollapsibleSection
                          collapsible={hasToolDetails}
                          expanded={!!expandedBackends[path]}
                          onToggle={() => toggleBackend(path)}
                          title={
                            <span className="text-sm font-mono text-gray-700 dark:text-gray-200 truncate">
                              {path}
                            </span>
                          }
                          badge={
                            hasToolDetails ? (
                              <span className="px-2 py-0.5 text-xs bg-virtual-100 dark:bg-virtual-900/40 text-virtual-700 dark:text-virtual-300 rounded-full">
                                {toolCount} tool{toolCount !== 1 ? 's' : ''}
                              </span>
                            ) : undefined
                          }
                        >
                          {backendTools.length > 0 && (
                            <ul className="border-t border-gray-200 dark:border-gray-700 divide-y divide-gray-100 dark:divide-gray-800">
                              {backendTools.map((tool) => (
                                <li
                                  key={tool.alias || tool.tool_name}
                                  className="px-4 py-3 bg-surface"
                                >
                                  <div className="flex items-start justify-between gap-2">
                                    <div className="flex-1 min-w-0">
                                      <span className="font-medium text-sm text-gray-900 dark:text-white">
                                        {tool.alias || tool.tool_name}
                                      </span>
                                      {tool.alias && tool.alias !== tool.tool_name && (
                                        <span className="ml-2 text-xs text-gray-400 dark:text-gray-500">
                                          (original: {tool.tool_name})
                                        </span>
                                      )}
                                    </div>
                                    {tool.backend_version && (
                                      <span className="px-1.5 py-0.5 text-[10px] bg-gray-100 dark:bg-gray-700 text-gray-500 dark:text-gray-400 rounded font-mono">
                                        v{tool.backend_version}
                                      </span>
                                    )}
                                  </div>
                                  {tool.description_override && (
                                    <p className="mt-1 text-xs text-gray-600 dark:text-gray-400 leading-relaxed">
                                      {tool.description_override}
                                    </p>
                                  )}
                                </li>
                              ))}
                            </ul>
                          )}
                        </CollapsibleSection>
                      </li>
                    );
                  })}
                </ul>
              )}
            </div>
          )}

          {/* Status */}
          <div>
            <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
              Status
            </p>
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${
                virtualServer.is_enabled
                  ? 'bg-success-400 shadow-lg shadow-success-400/30'
                  : 'bg-gray-300 dark:bg-gray-600'
              }`} />
              <span className="text-sm text-gray-700 dark:text-gray-300">
                {virtualServer.is_enabled ? 'Enabled' : 'Disabled'}
              </span>
            </div>
          </div>

          {/* Required Scopes */}
          {virtualServer.required_scopes && virtualServer.required_scopes.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
                Required Scopes
              </p>
              <div className="flex flex-wrap gap-2">
                {virtualServer.required_scopes.map((scope) => (
                  <span
                    key={scope}
                    className="px-2.5 py-1 text-xs rounded-full bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-200 font-mono"
                  >
                    {scope}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Supported Transports */}
          {virtualServer.supported_transports && virtualServer.supported_transports.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
                Supported Transports
              </p>
              <div className="flex flex-wrap gap-2">
                {virtualServer.supported_transports.map((transport) => (
                  <span
                    key={transport}
                    className="px-2.5 py-1 text-xs rounded-full bg-info-50 text-info-700 dark:bg-info-900/40 dark:text-info-200"
                  >
                    {transport}
                  </span>
                ))}
              </div>
            </div>
          )}
      </div>
    </EntityModal>
  );
};

export default VirtualServerDetailsModal;
