import React, { useState, useMemo, useCallback } from 'react';
import {
  ArrowPathIcon,
  CogIcon,
  InformationCircleIcon,
  XMarkIcon,
  ArrowTopRightOnSquareIcon,
  ClipboardIcon,
  ArrowDownTrayIcon
} from '@heroicons/react/24/outline';
import axios from 'axios';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  SemanticServerHit,
  SemanticToolHit,
  SemanticAgentHit,
  SemanticSkillHit,
  SemanticVirtualServerHit
} from '../hooks/useSemanticSearch';
import ServerConfigModal from './ServerConfigModal';
import AgentDetailsModal from './AgentDetailsModal';
import type { Server } from './ServerCard';
import type { Agent as AgentType } from './AgentCard';
import useEscapeKey from '../hooks/useEscapeKey';
import ANSBadge from './ANSBadge';

interface SemanticSearchResultsProps {
  query: string;
  loading: boolean;
  error: string | null;
  servers: SemanticServerHit[];
  tools: SemanticToolHit[];
  agents: SemanticAgentHit[];
  skills: SemanticSkillHit[];
  virtualServers?: SemanticVirtualServerHit[];
}

interface ToolSchemaModalProps {
  toolName: string;
  serverName: string;
  schema: Record<string, any> | null;
  isOpen: boolean;
  onClose: () => void;
}

const ToolSchemaModal: React.FC<ToolSchemaModalProps> = ({
  toolName,
  serverName,
  schema,
  isOpen,
  onClose
}) => {
  useEscapeKey(onClose, isOpen);
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-2xl w-full max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
          <div>
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              {toolName}
            </h3>
            <p className="text-sm text-gray-500 dark:text-gray-400">{serverName}</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 rounded-lg transition-colors"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>
        <div className="p-4 overflow-auto flex-1">
          <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
            Input Schema
          </p>
          {schema && Object.keys(schema).length > 0 ? (
            <pre className="text-xs bg-gray-100 dark:bg-gray-900 p-3 rounded-lg overflow-auto text-gray-800 dark:text-gray-200">
              {JSON.stringify(schema, null, 2)}
            </pre>
          ) : (
            <p className="text-sm text-gray-500 dark:text-gray-400 italic">
              No input schema available for this tool.
            </p>
          )}
        </div>
      </div>
    </div>
  );
};

// Helper function to parse YAML frontmatter from markdown
const parseYamlFrontmatter = (content: string): { frontmatter: Record<string, string> | null; body: string } => {
  const frontmatterRegex = /^---\s*\n([\s\S]*?)\n---\s*\n([\s\S]*)$/;
  const match = content.match(frontmatterRegex);

  if (match) {
    const yamlContent = match[1];
    const body = match[2];
    const frontmatter: Record<string, string> = {};
    const lines = yamlContent.split('\n');
    for (const line of lines) {
      const colonIndex = line.indexOf(':');
      if (colonIndex > 0) {
        const key = line.substring(0, colonIndex).trim();
        const value = line.substring(colonIndex + 1).trim();
        if (key && value) {
          frontmatter[key] = value;
        }
      }
    }
    return { frontmatter: Object.keys(frontmatter).length > 0 ? frontmatter : null, body };
  }
  return { frontmatter: null, body: content };
};


interface ServerDetailsModalProps {
  server: SemanticServerHit;
  isOpen: boolean;
  onClose: () => void;
}

const ServerDetailsModal: React.FC<ServerDetailsModalProps> = ({
  server,
  isOpen,
  onClose
}) => {
  useEscapeKey(onClose, isOpen);
  if (!isOpen) return null;

  const isFederatedServer = server.sync_metadata?.is_federated === true;
  const peerRegistryId = isFederatedServer && server.sync_metadata?.source_peer_id
    ? server.sync_metadata.source_peer_id.replace('peer-registry-', '').replace('peer-', '').toUpperCase()
    : null;

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-2xl w-full max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                {server.server_name}
              </h3>
              {isFederatedServer && peerRegistryId && (
                <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-cyan-100 text-cyan-700 dark:bg-cyan-900/40 dark:text-cyan-200 border border-cyan-200 dark:border-cyan-700">
                  {peerRegistryId}
                </span>
              )}
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400">{server.path}</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 rounded-lg transition-colors"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>
        <div className="p-4 overflow-auto flex-1 space-y-4">
          {/* Description */}
          <div>
            <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
              Description
            </p>
            <p className="text-sm text-gray-700 dark:text-gray-200">
              {server.description || 'No description available.'}
            </p>
          </div>

          {/* Tags */}
          {server.tags && server.tags.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
                Tags
              </p>
              <div className="flex flex-wrap gap-2">
                {server.tags.map((tag) => (
                  <span
                    key={tag}
                    className="px-2.5 py-1 text-xs rounded-full bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-200"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Tools */}
          {server.matching_tools && server.matching_tools.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
                Tools ({server.matching_tools.length})
              </p>
              <ul className="space-y-2">
                {server.matching_tools.map((tool) => (
                  <li key={tool.tool_name} className="text-sm text-gray-700 dark:text-gray-200 bg-gray-50 dark:bg-gray-900/50 p-3 rounded-lg">
                    <span className="font-medium text-gray-900 dark:text-white">{tool.tool_name}</span>
                    {tool.description && (
                      <p className="text-gray-600 dark:text-gray-300 mt-1 text-xs">
                        {tool.description}
                      </p>
                    )}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Status */}
          <div>
            <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
              Status
            </p>
            <div className="flex items-center gap-2">
              <div className={`w-3 h-3 rounded-full ${
                server.is_enabled
                  ? 'bg-green-400 shadow-lg shadow-green-400/30'
                  : 'bg-gray-300 dark:bg-gray-600'
              }`} />
              <span className="text-sm text-gray-700 dark:text-gray-300">
                {server.is_enabled ? 'Enabled' : 'Disabled'}
              </span>
            </div>
          </div>

          {/* Relevance Score */}
          <div>
            <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
              Match Score
            </p>
            <span className="inline-flex items-center rounded-full bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-200 px-3 py-1 text-xs font-semibold">
              {Math.round(Math.min(server.relevance_score, 1) * 100)}% match
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};


interface SkillContentModalProps {
  skill: SemanticSkillHit;
  isOpen: boolean;
  onClose: () => void;
}

const SkillContentModal: React.FC<SkillContentModalProps> = ({
  skill,
  isOpen,
  onClose
}) => {
  const [loading, setLoading] = useState(false);
  const [content, setContent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEscapeKey(onClose, isOpen);

  // Fetch content when modal opens
  React.useEffect(() => {
    if (!isOpen) {
      setContent(null);
      setError(null);
      return;
    }

    const fetchContent = async () => {
      setLoading(true);
      setError(null);
      try {
        // skill.path is like "/skills/doc-coauthoring", need just "/doc-coauthoring"
        const apiPath = skill.path.startsWith('/skills/')
          ? skill.path.replace('/skills/', '/')
          : skill.path;
        const response = await axios.get(`/api/skills${apiPath}/content`);
        setContent(response.data.content);
      } catch (err: any) {
        console.error('Failed to fetch SKILL.md content:', err);
        setError(err.response?.data?.detail || 'Failed to load SKILL.md content');
      } finally {
        setLoading(false);
      }
    };

    fetchContent();
  }, [isOpen, skill.path]);

  if (!isOpen) return null;

  const handleCopy = () => {
    if (content) {
      navigator.clipboard.writeText(content);
    }
  };

  const handleDownload = () => {
    if (content) {
      const blob = new Blob([content], { type: 'text/markdown' });
      const url = URL.createObjectURL(blob);
      const a = document.createElement('a');
      a.href = url;
      a.download = `${skill.skill_name || 'skill'}.md`;
      document.body.appendChild(a);
      a.click();
      document.body.removeChild(a);
      URL.revokeObjectURL(url);
    }
  };

  const { frontmatter, body } = content ? parseYamlFrontmatter(content) : { frontmatter: null, body: '' };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-4xl w-full max-h-[90vh] flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
          <div className="flex items-center gap-2">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
              {skill.skill_name}
            </h3>
            <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-200 border border-amber-200 dark:border-amber-600">
              SKILL
            </span>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 rounded-lg transition-colors"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>

        {/* Action buttons */}
        <div className="flex items-center gap-4 px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50">
          {skill.skill_md_url && (
            <a
              href={skill.skill_md_url}
              target="_blank"
              rel="noopener noreferrer"
              className="flex items-center gap-1 text-sm text-amber-700 dark:text-amber-300 hover:underline"
            >
              <ArrowTopRightOnSquareIcon className="h-4 w-4" />
              View on GitHub
            </a>
          )}
          {content && (
            <>
              <button
                onClick={handleCopy}
                className="flex items-center gap-1 text-sm text-gray-600 dark:text-gray-400 hover:text-amber-700 dark:hover:text-amber-300 transition-colors"
                title="Copy to clipboard"
              >
                <ClipboardIcon className="h-4 w-4" />
                Copy
              </button>
              <button
                onClick={handleDownload}
                className="flex items-center gap-1 text-sm text-gray-600 dark:text-gray-400 hover:text-amber-700 dark:hover:text-amber-300 transition-colors"
                title="Download SKILL.md"
              >
                <ArrowDownTrayIcon className="h-4 w-4" />
                Download
              </button>
            </>
          )}
        </div>

        <div className="p-4 overflow-auto flex-1">
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-amber-600"></div>
            </div>
          ) : error ? (
            <div className="text-center py-12 text-gray-500">
              <p className="text-red-500">{error}</p>
              {skill.skill_md_url && (
                <p className="mt-2 text-sm">
                  Try visiting the{' '}
                  <a
                    href={skill.skill_md_url}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="text-amber-600 hover:underline"
                  >
                    source URL
                  </a>{' '}
                  directly.
                </p>
              )}
            </div>
          ) : content ? (
            <>
              {/* YAML Frontmatter Table */}
              {frontmatter && (
                <div className="mb-6 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
                  <table className="w-full text-sm">
                    <tbody>
                      {Object.entries(frontmatter).map(([key, value]) => (
                        <tr key={key} className="border-b border-gray-200 dark:border-gray-700 last:border-b-0">
                          <td className="px-4 py-2 bg-gray-50 dark:bg-gray-900/50 font-medium text-gray-700 dark:text-gray-300 w-1/4">
                            {key}
                          </td>
                          <td className="px-4 py-2 text-gray-900 dark:text-white">
                            {value}
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              )}
              {/* Markdown Body */}
              <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:text-amber-800 dark:prose-headings:text-amber-200 prose-a:text-amber-600 dark:prose-a:text-amber-400 prose-code:bg-gray-100 dark:prose-code:bg-gray-900 prose-code:px-1 prose-code:py-0.5 prose-code:rounded prose-pre:bg-gray-100 dark:prose-pre:bg-gray-900">
                <ReactMarkdown remarkPlugins={[remarkGfm]}>{body}</ReactMarkdown>
              </div>
            </>
          ) : (
            <div className="text-center py-12 text-gray-500">
              <p>Could not load SKILL.md content.</p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};


interface VirtualServerDetailsModalProps {
  virtualServer: SemanticVirtualServerHit;
  isOpen: boolean;
  onClose: () => void;
}

const VirtualServerDetailsModal: React.FC<VirtualServerDetailsModalProps> = ({
  virtualServer,
  isOpen,
  onClose
}) => {
  const [copiedEndpoint, setCopiedEndpoint] = useState(false);
  const [expandedTools, setExpandedTools] = useState<Set<string>>(new Set());

  useEscapeKey(onClose, isOpen);
  if (!isOpen) return null;

  const tools = virtualServer.matching_tools || [];
  const backendPaths = virtualServer.backend_paths || [];

  const handleCopyEndpoint = () => {
    if (virtualServer.endpoint_url) {
      navigator.clipboard.writeText(virtualServer.endpoint_url);
      setCopiedEndpoint(true);
      setTimeout(() => setCopiedEndpoint(false), 2000);
    }
  };

  const toggleToolExpand = (toolName: string) => {
    setExpandedTools(prev => {
      const next = new Set(prev);
      if (next.has(toolName)) {
        next.delete(toolName);
      } else {
        next.add(toolName);
      }
      return next;
    });
  };

  return (
    <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/50">
      <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-2xl w-full max-h-[80vh] flex flex-col">
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
          <div>
            <div className="flex items-center gap-2">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                {virtualServer.server_name}
              </h3>
              <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-indigo-100 text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-200 border border-indigo-200 dark:border-indigo-600">
                VIRTUAL
              </span>
            </div>
            <p className="text-sm text-gray-500 dark:text-gray-400">{virtualServer.path}</p>
          </div>
          <button
            onClick={onClose}
            className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200 rounded-lg transition-colors"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>
        <div className="p-4 overflow-auto flex-1 space-y-4">
          {/* Endpoint URL */}
          {virtualServer.endpoint_url && (
            <div>
              <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
                Endpoint URL
              </p>
              <div className="flex items-center gap-2 bg-gray-50 dark:bg-gray-900/50 rounded-lg p-2">
                <code className="flex-1 text-sm text-indigo-600 dark:text-indigo-400 font-mono break-all">
                  {virtualServer.endpoint_url}
                </code>
                <button
                  onClick={handleCopyEndpoint}
                  className="flex-shrink-0 p-2 text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-400 hover:bg-indigo-50 dark:hover:bg-indigo-900/30 rounded-lg transition-colors"
                  title="Copy endpoint URL"
                >
                  {copiedEndpoint ? (
                    <span className="text-xs text-green-600 dark:text-green-400 font-medium">Copied!</span>
                  ) : (
                    <ClipboardIcon className="h-4 w-4" />
                  )}
                </button>
              </div>
            </div>
          )}

          {/* Description */}
          <div>
            <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
              Description
            </p>
            <p className="text-sm text-gray-700 dark:text-gray-200">
              {virtualServer.description || 'No description available.'}
            </p>
          </div>

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
                    className="px-2.5 py-1 text-xs rounded-full bg-indigo-50 text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-200"
                  >
                    {tag}
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Backend Servers */}
          {backendPaths.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
                Backend Servers ({backendPaths.length})
              </p>
              <ul className="space-y-1">
                {backendPaths.map((path) => (
                  <li key={path} className="text-sm text-gray-700 dark:text-gray-200 font-mono bg-gray-50 dark:bg-gray-900/50 px-2 py-1 rounded">
                    {path}
                  </li>
                ))}
              </ul>
            </div>
          )}

          {/* Tools */}
          {tools.length > 0 && (
            <div>
              <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
                Tools ({tools.length})
              </p>
              <ul className="space-y-2">
                {tools.map((tool) => {
                  const isExpanded = expandedTools.has(tool.tool_name);
                  return (
                    <li key={tool.tool_name} className="text-sm text-gray-700 dark:text-gray-200 bg-gray-50 dark:bg-gray-900/50 rounded-lg overflow-hidden">
                      <button
                        type="button"
                        onClick={() => toggleToolExpand(tool.tool_name)}
                        className="w-full p-3 text-left hover:bg-gray-100 dark:hover:bg-gray-800/50 transition-colors"
                      >
                        <div className="flex items-center justify-between">
                          <span className="font-medium text-gray-900 dark:text-white">{tool.tool_name}</span>
                          <div className="flex items-center gap-2">
                            {tool.relevance_score !== undefined && (
                              <span className="text-xs text-indigo-600 dark:text-indigo-400">
                                {Math.round(tool.relevance_score * 100)}%
                              </span>
                            )}
                            <InformationCircleIcon className={`h-4 w-4 text-gray-400 transition-transform ${isExpanded ? 'rotate-180' : ''}`} />
                          </div>
                        </div>
                        {(tool.description || tool.match_context) && (
                          <p className="text-gray-600 dark:text-gray-300 mt-1 text-xs">
                            {tool.description || tool.match_context}
                          </p>
                        )}
                      </button>
                      {isExpanded && (
                        <div className="px-3 pb-3 border-t border-gray-200 dark:border-gray-700 pt-2">
                          {tool.inputSchema && Object.keys(tool.inputSchema).length > 0 ? (
                            <>
                              <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
                                Input Schema
                              </p>
                              <pre className="text-xs bg-gray-100 dark:bg-gray-900 p-3 rounded-lg overflow-auto text-gray-800 dark:text-gray-200 max-h-48">
                                {JSON.stringify(tool.inputSchema, null, 2)}
                              </pre>
                            </>
                          ) : (
                            <p className="text-xs text-gray-500 dark:text-gray-400 italic">
                              No input schema available for this tool.
                            </p>
                          )}
                        </div>
                      )}
                    </li>
                  );
                })}
              </ul>
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
                  ? 'bg-green-400 shadow-lg shadow-green-400/30'
                  : 'bg-gray-300 dark:bg-gray-600'
              }`} />
              <span className="text-sm text-gray-700 dark:text-gray-300">
                {virtualServer.is_enabled ? 'Enabled' : 'Disabled'}
              </span>
            </div>
          </div>

          {/* Relevance Score */}
          <div>
            <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
              Match Score
            </p>
            <span className="inline-flex items-center rounded-full bg-indigo-100 text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-200 px-3 py-1 text-xs font-semibold">
              {Math.round(Math.min(virtualServer.relevance_score, 1) * 100)}% match
            </span>
          </div>
        </div>
      </div>
    </div>
  );
};


interface VirtualServerResultCardProps {
  virtualServer: SemanticVirtualServerHit;
  onViewDetails: () => void;
}

const VirtualServerResultCard: React.FC<VirtualServerResultCardProps> = ({
  virtualServer,
  onViewDetails
}) => {
  const [showAllTools, setShowAllTools] = useState(false);
  const tools = virtualServer.matching_tools || [];
  const visibleTools = showAllTools ? tools : tools.slice(0, 3);
  const hasMoreTools = tools.length > 3;

  return (
    <div className="rounded-2xl border-2 border-indigo-200 dark:border-indigo-700 bg-gradient-to-br from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 p-5 shadow-sm hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between gap-4">
        <div>
          <div className="flex items-center gap-2">
            <p className="text-base font-semibold text-gray-900 dark:text-white">
              {virtualServer.server_name}
            </p>
            <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-indigo-100 text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-200 border border-indigo-200 dark:border-indigo-600">
              VIRTUAL
            </span>
          </div>
          <p className="text-sm text-gray-500 dark:text-gray-400">{virtualServer.path}</p>
        </div>
        <div className="flex items-center gap-2">
          <button
            type="button"
            onClick={onViewDetails}
            className="p-2 text-gray-400 hover:text-indigo-600 dark:hover:text-indigo-300 hover:bg-indigo-50 dark:hover:bg-indigo-700/30 rounded-lg transition-colors"
            title="View virtual server details"
          >
            <InformationCircleIcon className="h-4 w-4" />
          </button>
          <span className="inline-flex items-center rounded-full bg-indigo-100 text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-200 px-3 py-1 text-xs font-semibold">
            {Math.round(Math.min(virtualServer.relevance_score, 1) * 100)}% match
          </span>
        </div>
      </div>

      <p className="mt-3 text-sm text-gray-600 dark:text-gray-300 line-clamp-3">
        {virtualServer.description || virtualServer.match_context || 'No description available.'}
      </p>

      {virtualServer.tags && virtualServer.tags.length > 0 && (
        <div className="mt-4 flex flex-wrap gap-2">
          {virtualServer.tags.slice(0, 6).map((tag) => (
            <span
              key={tag}
              className="px-2.5 py-1 text-[11px] rounded-full bg-indigo-50 text-indigo-700 dark:bg-indigo-900/40 dark:text-indigo-200"
            >
              {tag}
            </span>
          ))}
        </div>
      )}

      {/* Tools Section */}
      {tools.length > 0 && (
        <div className="mt-4 border-t border-dashed border-indigo-200 dark:border-indigo-700 pt-3">
          <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
            Tools ({tools.length})
          </p>
          <ul className="space-y-2">
            {visibleTools.map((tool) => (
              <li key={tool.tool_name} className="text-sm text-gray-700 dark:text-gray-200 flex items-start gap-2">
                <div className="flex-1 min-w-0">
                  <span className="font-medium text-gray-900 dark:text-white">{tool.tool_name}</span>
                  {tool.relevance_score !== undefined && (
                    <span className="ml-2 text-xs text-indigo-600 dark:text-indigo-400">
                      {Math.round(tool.relevance_score * 100)}%
                    </span>
                  )}
                  {(tool.description || tool.match_context) && (
                    <p className="text-gray-600 dark:text-gray-300 text-xs mt-0.5 line-clamp-1">
                      {tool.description || tool.match_context}
                    </p>
                  )}
                </div>
              </li>
            ))}
          </ul>
          {hasMoreTools && (
            <button
              type="button"
              onClick={() => setShowAllTools(!showAllTools)}
              className="mt-2 text-xs text-indigo-600 dark:text-indigo-400 hover:underline"
            >
              {showAllTools ? 'Show less' : `+${tools.length - 3} more tools...`}
            </button>
          )}
        </div>
      )}

      <div className="mt-4 flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
        <span>{virtualServer.backend_count || 0} backends</span>
        <span>{virtualServer.is_enabled ? 'Enabled' : 'Disabled'}</span>
      </div>
    </div>
  );
};


const formatPercent = (value: number) => `${Math.round(Math.min(value, 1) * 100)}%`;

const SemanticSearchResults: React.FC<SemanticSearchResultsProps> = ({
  query,
  loading,
  error,
  servers,
  tools,
  agents,
  skills,
  virtualServers = []
}) => {
  const hasResults = servers.length > 0 || tools.length > 0 || agents.length > 0 || skills.length > 0 || virtualServers.length > 0;
  const [configServer, setConfigServer] = useState<SemanticServerHit | null>(null);
  const [detailsServer, setDetailsServer] = useState<SemanticServerHit | null>(null);
  const [detailsSkill, setDetailsSkill] = useState<SemanticSkillHit | null>(null);
  const [detailsAgent, setDetailsAgent] = useState<SemanticAgentHit | null>(null);
  const [detailsVirtualServer, setDetailsVirtualServer] = useState<SemanticVirtualServerHit | null>(null);
  const [agentDetailsData, setAgentDetailsData] = useState<any>(null);
  const [agentDetailsLoading, setAgentDetailsLoading] = useState(false);
  const [selectedToolSchema, setSelectedToolSchema] = useState<{
    toolName: string;
    serverName: string;
    schema: Record<string, any> | null;
  } | null>(null);

  // Build a lookup map from server_path + tool_name to inputSchema
  const toolSchemaMap = useMemo(() => {
    const map = new Map<string, Record<string, any>>();
    for (const tool of tools) {
      const key = `${tool.server_path}:${tool.tool_name}`;
      if (tool.inputSchema) {
        map.set(key, tool.inputSchema);
      }
    }
    return map;
  }, [tools]);

  const openToolSchema = (
    serverPath: string,
    serverName: string,
    toolName: string
  ) => {
    const key = `${serverPath}:${toolName}`;
    const schema = toolSchemaMap.get(key) || null;
    setSelectedToolSchema({ toolName, serverName, schema });
  };

  const openAgentDetails = async (agentHit: SemanticAgentHit) => {
    setDetailsAgent(agentHit);
    setAgentDetailsData(null);
    setAgentDetailsLoading(true);
    try {
      const response = await axios.get(`/api/agents${agentHit.path}`);
      setAgentDetailsData(response.data);
    } catch (error) {
      console.error('Failed to fetch agent details:', error);
    } finally {
      setAgentDetailsLoading(false);
    }
  };

  const mapHitToAgent = (hit: SemanticAgentHit): AgentType => {
    const card = hit.agent_card || {};
    // Derive trust_level from the top-level trust_verified field returned by search API
    const trustVerified = hit.trust_verified || 'none';
    let trustLevel: AgentType['trust_level'] = card.trust_level || 'unverified';
    if (trustVerified === 'verified') {
      trustLevel = 'verified';
    }
    return {
      name: card.name || hit.path.replace(/^\//, ''),
      path: hit.path,
      url: card.url,
      description: card.description,
      version: card.version,
      visibility: (card.visibility as AgentType['visibility']) ?? 'public',
      trust_level: trustLevel,
      enabled: card.is_enabled ?? true,
      tags: card.tags || [],
      status: 'unknown',
      ans_metadata: card.ans_metadata || card.ansMetadata || undefined,
    };
  };

  return (
    <>
    <div className="space-y-8">
      <div className="flex flex-col gap-2 sm:flex-row sm:items-center sm:justify-between">
        <div>
          <p className="text-sm font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wide">
            Semantic Search
          </p>
          <h3 className="text-xl font-semibold text-gray-900 dark:text-white">
            Results for <span className="text-purple-600 dark:text-purple-300">“{query}”</span>
          </h3>
        </div>
        {loading && (
          <div className="inline-flex items-center text-sm text-purple-600 dark:text-purple-300">
            <ArrowPathIcon className="h-5 w-5 animate-spin mr-2" />
            Searching…
          </div>
        )}
      </div>

      {error && (
        <div className="rounded-lg border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700 dark:border-red-500/40 dark:bg-red-900/30 dark:text-red-200">
          {error}
        </div>
      )}

      {!loading && !error && !hasResults && (
        <div className="text-center py-16 border border-dashed border-gray-200 dark:border-gray-700 rounded-xl">
          <p className="text-lg font-medium text-gray-700 dark:text-gray-200 mb-2">
            No semantic matches found
          </p>
          <p className="text-sm text-gray-500 dark:text-gray-400 max-w-xl mx-auto">
            Try refining your query or describing the tools or capabilities you need. Semantic
            search understands natural language — phrases like “servers that handle authentication”
            or “tools for syncing calendars” work great.
          </p>
        </div>
      )}

      {servers.length > 0 && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Matching Servers <span className="text-sm font-normal text-gray-500">({servers.length})</span>
            </h4>
          </div>
          <div
            className="grid"
            style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.5rem' }}
          >
            {servers.map((server) => {
              // Detect if server is from a peer registry using sync_metadata
              const isFederatedServer = server.sync_metadata?.is_federated === true;
              const peerRegistryId = isFederatedServer && server.sync_metadata?.source_peer_id
                ? server.sync_metadata.source_peer_id.replace('peer-registry-', '').replace('peer-', '').toUpperCase()
                : null;
              const isOrphanedServer = server.sync_metadata?.is_orphaned === true;

              return (
              <div
                key={server.path}
                className="rounded-2xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-5 shadow-sm hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <p className="text-base font-semibold text-gray-900 dark:text-white">
                        {server.server_name}
                      </p>
                      {/* Registry source badge - only show for federated (peer registry) items */}
                      {isFederatedServer && (
                        <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-cyan-100 text-cyan-700 dark:bg-cyan-900/40 dark:text-cyan-200 border border-cyan-200 dark:border-cyan-700">
                          {peerRegistryId}
                        </span>
                      )}
                      {/* Orphaned badge */}
                      {isOrphanedServer && (
                        <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-200 border border-red-200 dark:border-red-700" title="No longer exists on peer registry">
                          ORPHANED
                        </span>
                      )}
                    </div>
                    <p className="text-sm text-gray-500 dark:text-gray-300">{server.path}</p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setDetailsServer(server)}
                      className="p-2 text-gray-400 hover:text-purple-600 dark:hover:text-purple-300 hover:bg-purple-50 dark:hover:bg-purple-700/30 rounded-lg transition-colors"
                      title="View server details"
                    >
                      <InformationCircleIcon className="h-4 w-4" />
                    </button>
                    <button
                      type="button"
                      onClick={() => setConfigServer(server)}
                      className="p-2 text-gray-400 hover:text-green-600 dark:hover:text-green-300 hover:bg-green-50 dark:hover:bg-green-700/30 rounded-lg transition-colors"
                      title="Open MCP configuration"
                    >
                      <CogIcon className="h-4 w-4" />
                    </button>
                    <span className="inline-flex items-center rounded-full bg-purple-100 text-purple-700 dark:bg-purple-900/40 dark:text-purple-200 px-3 py-1 text-xs font-semibold">
                      {formatPercent(server.relevance_score)} match
                    </span>
                  </div>
                </div>
                <p className="mt-3 text-sm text-gray-600 dark:text-gray-300 line-clamp-3">
                  {server.description || server.match_context || 'No description available.'}
                </p>

                {server.tags?.length > 0 && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {server.tags.slice(0, 6).map((tag) => (
                      <span
                        key={tag}
                        className="px-2.5 py-1 text-xs rounded-full bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-200"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}

                {server.matching_tools?.length > 0 && (
                  <div className="mt-4 border-t border-dashed border-gray-200 dark:border-gray-700 pt-3">
                    <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-2">
                      Relevant tools
                    </p>
                    <ul className="space-y-2">
                      {server.matching_tools.slice(0, 3).map((tool) => (
                        <li key={tool.tool_name} className="text-sm text-gray-700 dark:text-gray-200 flex items-start gap-2">
                          <div className="flex-1 min-w-0">
                            <span className="font-medium text-gray-900 dark:text-white">{tool.tool_name}</span>
                            <span className="mx-2 text-gray-400">-</span>
                            <span className="text-gray-600 dark:text-gray-300 line-clamp-1">
                              {tool.description || tool.match_context || 'No description'}
                            </span>
                          </div>
                          <button
                            type="button"
                            onClick={() => openToolSchema(server.path, server.server_name, tool.tool_name)}
                            className="flex-shrink-0 p-1 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 rounded transition-colors"
                            title="View input schema"
                          >
                            <InformationCircleIcon className="h-4 w-4" />
                          </button>
                        </li>
                      ))}
                    </ul>
                  </div>
                )}
              </div>
            );
            })}
          </div>
        </section>
      )}

      {tools.length > 0 && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Matching Tools <span className="text-sm font-normal text-gray-500">({tools.length})</span>
            </h4>
          </div>
          <div
            className="grid"
            style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}
          >
            {tools.map((tool) => (
              <div
                key={`${tool.server_path}-${tool.tool_name}`}
                className="rounded-xl border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 p-4 flex flex-col gap-2 sm:flex-row sm:items-start sm:justify-between"
              >
                <div className="flex-1 min-w-0">
                  <p className="text-sm font-semibold text-gray-900 dark:text-white">
                    {tool.tool_name}
                    <span className="ml-2 text-xs font-normal text-gray-500 dark:text-gray-400">
                      ({tool.server_name})
                    </span>
                  </p>
                  <p className="text-sm text-gray-600 dark:text-gray-300 line-clamp-2">
                    {tool.description || tool.match_context || 'No description available.'}
                  </p>
                </div>
                <div className="flex items-center gap-2 flex-shrink-0">
                  <button
                    type="button"
                    onClick={() => setSelectedToolSchema({
                      toolName: tool.tool_name,
                      serverName: tool.server_name,
                      schema: tool.inputSchema || null
                    })}
                    className="p-1.5 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/30 rounded-lg transition-colors"
                    title="View input schema"
                  >
                    <InformationCircleIcon className="h-4 w-4" />
                  </button>
                  <span className="inline-flex items-center rounded-full bg-blue-100 text-blue-700 dark:bg-blue-900/40 dark:text-blue-200 px-3 py-1 text-xs font-semibold">
                    {formatPercent(tool.relevance_score)} match
                  </span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {agents.length > 0 && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Matching Agents <span className="text-sm font-normal text-gray-500">({agents.length})</span>
            </h4>
          </div>
          <div
            className="grid"
            style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}
          >
            {agents.map((agent) => {
              // Extract agent details from agent_card
              const card = agent.agent_card || {};
              const agentName = card.name || agent.path.replace(/^\//, '');
              const agentDescription = card.description;
              const agentTags = card.tags || [];
              const agentVisibility = card.visibility || 'public';
              const trustVerified = agent.trust_verified || 'none';
              const agentTrustLevel = trustVerified === 'verified' ? 'verified' : (card.trust_level || 'unverified');
              const agentIsEnabled = card.is_enabled ?? false;
              const syncMetadata = card.sync_metadata;

              // Extract skill names from agent_card.skills (array of skill objects)
              const rawSkills = card.skills || [];
              const skillNames = rawSkills.map((s: any) =>
                typeof s === 'string' ? s : s?.name || s?.id
              ).filter(Boolean);

              // Detect if agent is from a peer registry using sync_metadata
              const isFederatedAgent = syncMetadata?.is_federated === true;
              const peerRegistryId = isFederatedAgent && syncMetadata?.source_peer_id
                ? syncMetadata.source_peer_id.replace('peer-registry-', '').replace('peer-', '').toUpperCase()
                : null;
              const isOrphanedAgent = syncMetadata?.is_orphaned === true;

              return (
              <div
                key={agent.path}
                className="rounded-2xl border border-cyan-200 dark:border-cyan-900/40 bg-white dark:bg-gray-800 p-5 shadow-sm hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <p className="text-base font-semibold text-gray-900 dark:text-white">
                        {agentName}
                      </p>
                      {/* Registry source badge - only show for federated (peer registry) items */}
                      {isFederatedAgent && (
                        <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-violet-100 text-violet-700 dark:bg-violet-900/40 dark:text-violet-200 border border-violet-200 dark:border-violet-700">
                          {peerRegistryId}
                        </span>
                      )}
                      {/* Orphaned badge */}
                      {isOrphanedAgent && (
                        <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-red-100 text-red-700 dark:bg-red-900/40 dark:text-red-200 border border-red-200 dark:border-red-700" title="No longer exists on peer registry">
                          ORPHANED
                        </span>
                      )}
                    </div>
                    <p className="text-xs uppercase tracking-wide text-gray-400 dark:text-gray-500">
                      {agentVisibility}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => openAgentDetails(agent)}
                      className="p-2 text-gray-400 hover:text-cyan-600 dark:hover:text-cyan-300 hover:bg-cyan-50 dark:hover:bg-cyan-700/30 rounded-lg transition-colors"
                      title="View full agent details"
                    >
                      <InformationCircleIcon className="h-4 w-4" />
                    </button>
                    <span className="inline-flex items-center rounded-full bg-cyan-100 text-cyan-700 dark:bg-cyan-900/40 dark:text-cyan-200 px-3 py-1 text-xs font-semibold">
                      {formatPercent(agent.relevance_score)} match
                    </span>
                  </div>
                </div>

                <p className="mt-3 text-sm text-gray-600 dark:text-gray-300 line-clamp-3">
                  {agentDescription || agent.match_context || 'No description available.'}
                </p>

                {skillNames.length > 0 && (
                  <div className="mt-4">
                    <p className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase tracking-wide mb-1">
                      Key Skills
                    </p>
                    <p className="text-xs text-gray-600 dark:text-gray-300">
                      {skillNames.slice(0, 4).join(', ')}
                      {skillNames.length > 4 && '…'}
                    </p>
                  </div>
                )}

                {agentTags.length > 0 && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {agentTags.slice(0, 6).map((tag: string) => (
                      <span
                        key={tag}
                        className="px-2.5 py-1 text-[11px] rounded-full bg-cyan-50 text-cyan-700 dark:bg-cyan-900/40 dark:text-cyan-200"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}

                <div className="mt-4 flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                  {(card.ans_metadata || card.ansMetadata) ? (
                    <ANSBadge ansMetadata={card.ans_metadata || card.ansMetadata} compact />
                  ) : (
                    <span className="font-semibold text-cyan-700 dark:text-cyan-200">
                      {agentTrustLevel}
                    </span>
                  )}
                  <span>{agentIsEnabled ? 'Enabled' : 'Disabled'}</span>
                </div>
              </div>
            );
            })}
          </div>
        </section>
      )}

      {skills.length > 0 && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Matching Skills <span className="text-sm font-normal text-gray-500">({skills.length})</span>
            </h4>
          </div>
          <div
            className="grid"
            style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}
          >
            {skills.map((skill) => (
              <div
                key={skill.path}
                className="rounded-2xl border-2 border-amber-200 dark:border-amber-700 bg-gradient-to-br from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 p-5 shadow-sm hover:shadow-md transition-shadow"
              >
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <div className="flex items-center gap-2">
                      <p className="text-base font-semibold text-gray-900 dark:text-white">
                        {skill.skill_name}
                      </p>
                      <span className="px-2 py-0.5 text-[10px] font-semibold rounded-full bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-200 border border-amber-200 dark:border-amber-600">
                        SKILL
                      </span>
                    </div>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      {skill.visibility || 'public'}
                    </p>
                  </div>
                  <div className="flex items-center gap-2">
                    <button
                      type="button"
                      onClick={() => setDetailsSkill(skill)}
                      className="p-2 text-gray-400 hover:text-amber-600 dark:hover:text-amber-300 hover:bg-amber-50 dark:hover:bg-amber-700/30 rounded-lg transition-colors"
                      title="View SKILL.md content"
                    >
                      <InformationCircleIcon className="h-4 w-4" />
                    </button>
                    <span className="inline-flex items-center rounded-full bg-amber-100 text-amber-700 dark:bg-amber-900/40 dark:text-amber-200 px-3 py-1 text-xs font-semibold">
                      {formatPercent(skill.relevance_score)} match
                    </span>
                  </div>
                </div>

                <p className="mt-3 text-sm text-gray-600 dark:text-gray-300 line-clamp-3">
                  {skill.description || skill.match_context || 'No description available.'}
                </p>

                {skill.tags && skill.tags.length > 0 && (
                  <div className="mt-4 flex flex-wrap gap-2">
                    {skill.tags.slice(0, 6).map((tag) => (
                      <span
                        key={tag}
                        className="px-2.5 py-1 text-[11px] rounded-full bg-amber-50 text-amber-700 dark:bg-amber-900/40 dark:text-amber-200"
                      >
                        {tag}
                      </span>
                    ))}
                  </div>
                )}

                <div className="mt-4 flex items-center justify-between text-xs text-gray-500 dark:text-gray-400">
                  <div className="flex items-center gap-2">
                    {skill.author && (
                      <span>by {skill.author}</span>
                    )}
                    {skill.version && (
                      <span className="text-amber-600 dark:text-amber-400">v{skill.version}</span>
                    )}
                  </div>
                  <span>{skill.is_enabled ? 'Enabled' : 'Disabled'}</span>
                </div>
              </div>
            ))}
          </div>
        </section>
      )}

      {virtualServers.length > 0 && (
        <section className="space-y-4">
          <div className="flex items-center justify-between">
            <h4 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
              Matching Virtual Servers <span className="text-sm font-normal text-gray-500">({virtualServers.length})</span>
            </h4>
          </div>
          <div
            className="grid"
            style={{ gridTemplateColumns: 'repeat(auto-fit, minmax(320px, 1fr))', gap: '1.25rem' }}
          >
            {virtualServers.map((vs) => (
              <VirtualServerResultCard
                key={vs.path}
                virtualServer={vs}
                onViewDetails={() => setDetailsVirtualServer(vs)}
              />
            ))}
          </div>
        </section>
      )}
    </div>

    {configServer && (
      <ServerConfigModal
        server={
          {
            name: configServer.server_name,
            path: configServer.path,
            description: configServer.description,
            enabled: configServer.is_enabled ?? true,
            tags: configServer.tags,
            num_tools: configServer.num_tools,
          } as Server
        }
        isOpen
        onClose={() => setConfigServer(null)}
      />
    )}

    {detailsAgent && (
      <AgentDetailsModal
        agent={mapHitToAgent(detailsAgent)}
        isOpen
        onClose={() => setDetailsAgent(null)}
        loading={agentDetailsLoading}
        fullDetails={agentDetailsData}
      />
    )}

    {selectedToolSchema && (
      <ToolSchemaModal
        toolName={selectedToolSchema.toolName}
        serverName={selectedToolSchema.serverName}
        schema={selectedToolSchema.schema}
        isOpen
        onClose={() => setSelectedToolSchema(null)}
      />
    )}

    {detailsServer && (
      <ServerDetailsModal
        server={detailsServer}
        isOpen
        onClose={() => setDetailsServer(null)}
      />
    )}

    {detailsSkill && (
      <SkillContentModal
        skill={detailsSkill}
        isOpen
        onClose={() => setDetailsSkill(null)}
      />
    )}

    {detailsVirtualServer && (
      <VirtualServerDetailsModal
        virtualServer={detailsVirtualServer}
        isOpen
        onClose={() => setDetailsVirtualServer(null)}
      />
    )}
    </>
  );
};

export default SemanticSearchResults;
