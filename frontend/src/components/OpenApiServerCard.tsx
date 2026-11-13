import React from 'react';
import {
  PencilIcon,
  TrashIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline';
import { OpenApiServer } from '../types/openApiServer';
import { SafeLink } from './SafeLink';
import { CopyButton } from './modals';


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
}


/**
 * OpenApiServerCard shows one OpenAPI MCP server in the admin OpenAPI view:
 * name, description, path, MCP URL with a copy button, target and tool
 * counts, and a status badge.
 */
const OpenApiServerCard: React.FC<OpenApiServerCardProps> = ({
  server,
  onManage,
  onEdit,
  onDelete,
}) => {
  const targetCount = server.targets?.length || 0;
  const toolCount = openApiToolCount(server);

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

        {server.mcp_url && (
          <div className="flex items-center gap-2 mb-3 min-w-0">
            <SafeLink href={server.mcp_url} target="_blank" rel="noopener noreferrer"
              className="text-xs font-mono text-primary-600 dark:text-primary-400 truncate">
              {server.mcp_url}
            </SafeLink>
            <CopyButton getText={() => server.mcp_url || ''} label="Copy" variant="subtle"
              title="Copy the MCP URL" />
          </div>
        )}

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
    </div>
  );
};


export default OpenApiServerCard;
