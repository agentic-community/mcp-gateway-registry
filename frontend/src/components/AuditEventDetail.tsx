import React, { useState } from 'react';
import {
  XMarkIcon,
  ClipboardDocumentIcon,
  CheckIcon,
} from '@heroicons/react/24/outline';
import type { AuditEvent } from '../types/audit';
import { CopyButton } from './modals';

interface AuditEventDetailProps {
  event: AuditEvent;
  onClose: () => void;
}

interface IdentityClaimRow {
  label: string;
  value: string;
  mono: boolean;
}

const AuditEventDetail: React.FC<AuditEventDetailProps> = ({ event, onClose }) => {
  const [copied, setCopied] = useState(false);

  const isMcpEvent = event.log_type === 'mcp_server_access';

  // Durable / IdP identity claims: nested under `identity` on the registry_api
  // and mcp_access streams, flat on token_mint records (no `identity` block).
  // A claim the token did not carry arrives as an explicit `null`.
  const claim = (
    field: 'principal_name' | 'canonical_id' | 'subject' | 'object_id' | 'tenant_id' | 'app_id',
  ): string | null | undefined => event.identity?.[field] || event[field];

  // Only the claims the token actually carried; most tokens carry none.
  const identityClaims = [
    { label: 'Principal', value: claim('principal_name'), mono: false },
    { label: 'Canonical ID', value: claim('canonical_id'), mono: true },
    { label: 'Subject', value: claim('subject'), mono: true },
    { label: 'Object ID', value: claim('object_id'), mono: true },
    { label: 'Tenant ID', value: claim('tenant_id'), mono: true },
    { label: 'App ID', value: claim('app_id'), mono: true },
  ].filter((c): c is IdentityClaimRow => Boolean(c.value));

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(JSON.stringify(event, null, 2));
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy to clipboard:', err);
    }
  };

  const formatJson = (obj: unknown): string => {
    return JSON.stringify(obj, null, 2);
  };

  const getStatusColor = (statusCode: number): string => {
    if (statusCode >= 200 && statusCode < 300) return 'text-success-600 dark:text-success-400';
    if (statusCode >= 400 && statusCode < 500) return 'text-warning-600 dark:text-warning-400';
    if (statusCode >= 500) return 'text-danger-600 dark:text-danger-400';
    return 'text-fg-muted';
  };

  const getMcpStatusColor = (status: string): string => {
    switch (status?.toLowerCase()) {
      case 'success':
        return 'text-success-600 dark:text-success-400';
      case 'error':
        return 'text-danger-600 dark:text-danger-400';
      case 'timeout':
        return 'text-warning-600 dark:text-warning-400';
      default:
        return 'text-fg-muted';
    }
  };

  return (
    <div className="bg-surface rounded-lg shadow-lg border border-line overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-line bg-surface-muted flex items-center justify-between gap-2">
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <h3 className="text-sm font-medium text-fg flex-shrink-0">
            Event Details
          </h3>
          <span
            className="text-xs text-fg-subtle font-mono truncate"
            title={event.request_id}
          >
            {event.request_id}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="btn-secondary gap-1.5 px-2.5 py-1.5 text-xs"
            title="Copy JSON to clipboard"
          >
            {copied ? (
              <>
                <CheckIcon className="h-4 w-4 text-success-500" />
                <span>Copied!</span>
              </>
            ) : (
              <>
                <ClipboardDocumentIcon className="h-4 w-4" />
                <span>Copy JSON</span>
              </>
            )}
          </button>
          <button
            onClick={onClose}
            className="btn-icon h-8 w-8"
            title="Close"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Summary */}
      <div className="px-4 py-3 border-b border-line grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="min-w-0">
          <div className="text-xs text-fg-subtle uppercase tracking-wider mb-1">
            Timestamp
          </div>
          <div className="text-sm text-fg truncate">
            {new Date(event.timestamp).toLocaleString()}
          </div>
        </div>
        <div className="min-w-0">
          <div className="text-xs text-fg-subtle uppercase tracking-wider mb-1">
            User
          </div>
          <div className="text-sm text-fg flex items-center gap-1 min-w-0">
            <span
              className="truncate"
              title={event.identity?.username || event.username || event.username_hash || '-'}
            >
              {event.identity?.username || event.username || event.username_hash || '-'}
            </span>
            {event.identity?.is_admin && (
              <span className="px-1.5 py-0.5 text-xs font-medium bg-accent-soft text-accent-soft-fg rounded flex-shrink-0">
                Admin
              </span>
            )}
          </div>
        </div>
        <div className="min-w-0">
          <div className="text-xs text-fg-subtle uppercase tracking-wider mb-1">
            Status
          </div>
          {isMcpEvent ? (
            <div className={`text-sm font-medium ${getMcpStatusColor(event.mcp_response?.status || '')}`}>
              {event.mcp_response?.status || '-'}
            </div>
          ) : (
            <div className={`text-sm font-medium ${getStatusColor(event.response?.status_code || 0)}`}>
              {event.response?.status_code || '-'}
            </div>
          )}
        </div>
        <div className="min-w-0">
          <div className="text-xs text-fg-subtle uppercase tracking-wider mb-1">
            Duration
          </div>
          <div className="text-sm text-fg">
            {isMcpEvent
              ? `${(event.mcp_response?.duration_ms || 0).toFixed(2)} ms`
              : `${(event.response?.duration_ms || 0).toFixed(2)} ms`
            }
          </div>
        </div>
      </div>

      {/* Durable / IdP identity claims (admin-only detail panel; omitted entirely
          when the token carried none) */}
      {identityClaims.length > 0 && (
        <div className="px-4 py-3 border-b border-line bg-surface-muted/50">
          <div className="text-xs font-medium text-fg-subtle uppercase tracking-wider mb-2">
            Identity Claims
          </div>
          {/* These values run 36-73 chars while this panel is one third of the
              page, so they wrap (break-all) instead of truncating: truncated,
              `canonical_id` and `object_id` render as the same string and the
              operator cannot tell them apart. The copy control sits on the label
              line so the value keeps the full column width, and each value is
              copyable on its own -- hover-only `title` is dead on touch.
              The column count tracks the PAGE grid, which drops this panel from
              full width to a one-third sidebar at `lg`: two columns while the
              panel is wide, one once it narrows (measured at 1024px: a 36-char
              GUID needs ~300px and a two-column cell is 127px). */}
          <dl className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-1 gap-x-4 gap-y-3">
            {identityClaims.map(({ label, value, mono }) => (
              <div key={label} className="min-w-0">
                <dt className="flex items-center justify-between gap-1 text-xs text-fg-subtle uppercase tracking-wider mb-1">
                  <span>{label}</span>
                  <CopyButton
                    variant="subtle"
                    label=""
                    copiedLabel=""
                    getText={() => value}
                    title={`Copy ${label}`}
                  />
                </dt>
                <dd
                  className={`text-sm text-fg break-all ${mono ? 'font-mono' : ''}`}
                >
                  {value}
                </dd>
              </div>
            ))}
          </dl>
        </div>
      )}

      {/* MCP-specific summary row */}
      {isMcpEvent && (
        <div className="px-4 py-3 border-b border-line grid grid-cols-2 md:grid-cols-4 gap-2 bg-info-50/50 dark:bg-info-900/10">
          <div className="min-w-0 overflow-hidden">
            <div className="text-xs text-fg-subtle uppercase tracking-wider mb-1 truncate" title="MCP Server">
              Server
            </div>
            <div
              className="text-sm text-fg truncate"
              title={event.mcp_server?.name || '-'}
            >
              {event.mcp_server?.name || '-'}
            </div>
          </div>
          <div className="min-w-0 overflow-hidden">
            <div className="text-xs text-fg-subtle uppercase tracking-wider mb-1 truncate" title="MCP Method">
              Method
            </div>
            <div
              className="text-sm font-mono text-fg truncate"
              title={event.mcp_request?.method || '-'}
            >
              {event.mcp_request?.method || '-'}
            </div>
          </div>
          <div className="min-w-0 overflow-hidden">
            <div className="text-xs text-fg-subtle uppercase tracking-wider mb-1 truncate" title="Tool/Resource">
              Tool
            </div>
            <div
              className="text-sm text-fg truncate"
              title={event.mcp_request?.tool_name || event.mcp_request?.resource_uri || '-'}
            >
              {event.mcp_request?.tool_name || event.mcp_request?.resource_uri || '-'}
            </div>
          </div>
          <div className="min-w-0 overflow-hidden">
            <div className="text-xs text-fg-subtle uppercase tracking-wider mb-1 truncate">
              Transport
            </div>
            <div
              className="text-sm text-fg truncate"
              title={event.mcp_request?.transport || '-'}
            >
              {event.mcp_request?.transport || '-'}
            </div>
          </div>
        </div>
      )}

      {/* JSON Content */}
      <div className="p-4 max-h-[60vh] overflow-auto">
        <pre className="text-xs font-mono text-fg-muted whitespace-pre-wrap break-words bg-surface-muted p-4 rounded-lg border border-line">
          {formatJson(event)}
        </pre>
      </div>
    </div>
  );
};

export default AuditEventDetail;
