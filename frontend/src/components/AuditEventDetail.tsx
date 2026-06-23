import React, { useState } from 'react';
import {
  XMarkIcon,
  ClipboardDocumentIcon,
  CheckIcon,
} from '@heroicons/react/24/outline';
import { AuditEvent } from './AuditLogTable';

interface AuditEventDetailProps {
  event: AuditEvent;
  onClose: () => void;
}

const AuditEventDetail: React.FC<AuditEventDetailProps> = ({ event, onClose }) => {
  const [copied, setCopied] = useState(false);

  const isMcpEvent = event.log_type === 'mcp_server_access';

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
    if (statusCode >= 200 && statusCode < 300) return 'text-green-600 dark:text-green-400';
    if (statusCode >= 400 && statusCode < 500) return 'text-yellow-600 dark:text-yellow-400';
    if (statusCode >= 500) return 'text-red-600 dark:text-red-400';
    return 'text-gray-600 dark:text-gray-400';
  };

  const getMcpStatusColor = (status: string): string => {
    switch (status?.toLowerCase()) {
      case 'success':
        return 'text-green-600 dark:text-green-400';
      case 'error':
        return 'text-red-600 dark:text-red-400';
      case 'timeout':
        return 'text-yellow-600 dark:text-yellow-400';
      default:
        return 'text-gray-600 dark:text-gray-400';
    }
  };

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
      {/* Header */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50 flex items-center justify-between gap-2">
        <div className="flex items-center gap-3 min-w-0 flex-1">
          <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 flex-shrink-0">
            Event Details
          </h3>
          <span
            className="text-xs text-gray-500 dark:text-gray-400 font-mono truncate"
            title={event.request_id}
          >
            {event.request_id}
          </span>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleCopy}
            className="flex items-center gap-1.5 px-2.5 py-1.5 text-xs font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors"
            title="Copy JSON to clipboard"
          >
            {copied ? (
              <>
                <CheckIcon className="h-4 w-4 text-green-500" />
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
            className="p-1.5 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
            title="Close"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>
      </div>

      {/* Summary */}
      <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 grid grid-cols-2 md:grid-cols-4 gap-4">
        <div className="min-w-0">
          <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
            Timestamp
          </div>
          <div className="text-sm text-gray-900 dark:text-gray-100 truncate">
            {new Date(event.timestamp).toLocaleString()}
          </div>
        </div>
        <div className="min-w-0">
          <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
            User
          </div>
          <div className="text-sm text-gray-900 dark:text-gray-100 flex items-center gap-1 min-w-0">
            <span className="truncate" title={event.identity?.username || event.username_hash || '-'}>
              {event.identity?.username || event.username_hash || '-'}
            </span>
            {event.identity?.is_admin && (
              <span className="px-1.5 py-0.5 text-xs font-medium bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300 rounded flex-shrink-0">
                Admin
              </span>
            )}
          </div>
        </div>
        <div className="min-w-0">
          <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
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
          <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
            Duration
          </div>
          <div className="text-sm text-gray-900 dark:text-gray-100">
            {isMcpEvent
              ? `${(event.mcp_response?.duration_ms || 0).toFixed(2)} ms`
              : `${(event.response?.duration_ms || 0).toFixed(2)} ms`
            }
          </div>
        </div>
      </div>

      {/* MCP-specific summary row */}
      {isMcpEvent && (
        <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700 grid grid-cols-2 md:grid-cols-4 gap-2 bg-blue-50/50 dark:bg-blue-900/10">
          <div className="min-w-0 overflow-hidden">
            <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1 truncate" title="MCP Server">
              Server
            </div>
            <div
              className="text-sm text-gray-900 dark:text-gray-100 truncate"
              title={event.mcp_server?.name || '-'}
            >
              {event.mcp_server?.name || '-'}
            </div>
          </div>
          <div className="min-w-0 overflow-hidden">
            <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1 truncate" title="MCP Method">
              Method
            </div>
            <div
              className="text-sm font-mono text-gray-900 dark:text-gray-100 truncate"
              title={event.mcp_request?.method || '-'}
            >
              {event.mcp_request?.method || '-'}
            </div>
          </div>
          <div className="min-w-0 overflow-hidden">
            <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1 truncate" title="Tool/Resource">
              Tool
            </div>
            <div
              className="text-sm text-gray-900 dark:text-gray-100 truncate"
              title={event.mcp_request?.tool_name || event.mcp_request?.resource_uri || '-'}
            >
              {event.mcp_request?.tool_name || event.mcp_request?.resource_uri || '-'}
            </div>
          </div>
          <div className="min-w-0 overflow-hidden">
            <div className="text-xs text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1 truncate" title="Transport">
              Transport
            </div>
            <div
              className="text-sm text-gray-900 dark:text-gray-100 truncate"
              title={event.mcp_request?.transport || '-'}
            >
              {event.mcp_request?.transport || '-'}
            </div>
          </div>
        </div>
      )}

      {/* JSON Content */}
      <div className="p-4 max-h-[60vh] overflow-auto">
        <pre className="text-xs font-mono text-gray-800 dark:text-gray-200 whitespace-pre-wrap break-words bg-gray-50 dark:bg-gray-900/50 p-4 rounded-lg border border-gray-200 dark:border-gray-700">
          {formatJson(event)}
        </pre>
      </div>
    </div>
  );
};

export default AuditEventDetail;
