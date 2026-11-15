import React, { useState } from 'react';
import {
  PencilIcon,
  TrashIcon,
  Cog6ToothIcon,
  LinkIcon,
} from '@heroicons/react/24/outline';
import { OpenApiServer } from '../types/openApiServer';
import ServerConfigModal from './ServerConfigModal';


/** The status a server card shows. */
export type OpenApiDisplayStatus = 'applied' | 'pending' | 'error';


/**
 * The status to show for a server: `error` when any target failed in the
 * container, `pending` while the container has not confirmed the latest
 * generation, otherwise `applied`.
 */
export function openApiServerStatus(server: OpenApiServer): OpenApiDisplayStatus {
  const targets = server.targets || [];
  if (targets.some((t) => t.status === 'error')) {
    return 'error';
  }
  if (server.sync_status === 'pending' || targets.some((t) => t.status === 'pending')) {
    return 'pending';
  }
  return 'applied';
}


/** Total tools across a server's targets. */
export function openApiToolCount(server: OpenApiServer): number {
  let total = 0;
  for (const target of server.targets || []) {
    total += target.tool_names?.length || 0;
  }
  return total;
}


const STATUS_CLASSES: Record<OpenApiDisplayStatus, string> = {
  applied: 'bg-success-100 dark:bg-success-900/30 text-success-800 dark:text-success-400',
  pending: 'bg-warning-100 dark:bg-warning-900/30 text-warning-800 dark:text-warning-400',
  error: 'bg-danger-100 dark:bg-danger-900/30 text-danger-800 dark:text-danger-400',
};


/** A small badge for `applied`, `pending`, `error` or a target's `ok`. */
export const OpenApiStatusBadge: React.FC<{ status: string }> = ({ status }) => {
  const key: OpenApiDisplayStatus =
    status === 'error' ? 'error' : status === 'pending' ? 'pending' : 'applied';
  return (
    <span
      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${STATUS_CLASSES[key]}`}
      data-testid="openapi-status-badge"
    >
      {status}
    </span>
  );
};


interface OpenApiServerCardProps {
  server: OpenApiServer;
  onManage: (server: OpenApiServer) => void;
  onEdit: (server: OpenApiServer) => void;
  onDelete: (server: OpenApiServer) => void;
  onShowToast?: (message: string, type: 'success' | 'error') => void;
}


/**
 * OpenApiServerCard shows one OpenAPI MCP server in the admin OpenAPI view:
 * name, description, path, target and tool counts, a status badge, and a
 * Connect button that opens the same connection dialog (client configs and
 * the gateway URL) as the other server cards.
 */
const OpenApiServerCard: React.FC<OpenApiServerCardProps> = ({
  server,
  onManage,
  onEdit,
  onDelete,
  onShowToast,
}) => {
  const [showConfig, setShowConfig] = useState(false);
  const targetCount = server.targets?.length || 0;
  const toolCount = openApiToolCount(server);
  // Each OpenAPI MCP server is an ordinary registry server at its path, so the
  // standard connection dialog works with these fields.
  const serverForConfig = {
    name: server.name,
    path: server.path,
    description: server.description,
    enabled: true,
    tags: server.tags || [],
  };

  return (
    <div className="rounded-2xl shadow-sm hover:shadow-xl transition-all duration-300 h-full flex flex-col bg-surface border-2 border-server-200 dark:border-server-700">
      <div className="p-5 pb-4 flex-1">
        <div className="flex items-start justify-between mb-3 gap-2">
          <div className="flex-1 min-w-0">
            <div className="flex items-center gap-2 mb-2">
              <h3 className="text-lg font-bold text-fg truncate">{server.name}</h3>
              <span className="px-2 py-0.5 text-xs font-semibold bg-server-100 text-server-700 dark:bg-server-900/30 dark:text-server-300 rounded-full flex-shrink-0 border border-server-200 dark:border-server-600">
                OPENAPI
              </span>
            </div>
            <code className="text-xs text-fg-muted bg-gray-500/10 px-2 py-1 rounded font-mono">
              {server.path}
            </code>
          </div>
          <div className="flex items-center gap-1 flex-shrink-0">
            <button
              onClick={() => setShowConfig(true)}
              className="flex items-center gap-1 px-2 py-1.5 text-xs font-medium text-success-600 dark:text-success-400 hover:bg-success-50 dark:hover:bg-success-700/50 rounded-lg transition-all duration-200 flex-shrink-0 border border-success-200 dark:border-success-700"
              title="Get connection details and mcp.json configuration"
              aria-label={`Connect to ${server.name}`}
            >
              <LinkIcon className="h-3.5 w-3.5" />
              Connect
            </button>
            <button onClick={() => onEdit(server)} className="btn-icon" title="Edit server"
              aria-label={`Edit ${server.name}`}>
              <PencilIcon className="h-4 w-4" />
            </button>
            <button onClick={() => onDelete(server)} className="btn-icon" title="Delete server"
              aria-label={`Delete ${server.name}`}>
              <TrashIcon className="h-4 w-4" />
            </button>
          </div>
        </div>

        <p className="text-sm text-fg-muted leading-relaxed line-clamp-2 mb-3">
          {server.description || 'No description available'}
        </p>

        <div className="flex items-center gap-4 text-sm text-fg-muted">
          <span>{targetCount} API{targetCount !== 1 ? 's' : ''}</span>
          <span>{toolCount} tool{toolCount !== 1 ? 's' : ''}</span>
          <OpenApiStatusBadge status={openApiServerStatus(server)} />
        </div>
      </div>

      <div className="px-5 py-3 border-t border-line">
        <button onClick={() => onManage(server)}
          className="btn-secondary flex items-center gap-2 text-sm">
          <Cog6ToothIcon className="h-4 w-4" />
          Manage APIs
        </button>
      </div>

      <ServerConfigModal
        server={serverForConfig as any}
        isOpen={showConfig}
        onClose={() => setShowConfig(false)}
        onShowToast={onShowToast}
        resourceType="server"
      />
    </div>
  );
};


export default OpenApiServerCard;
