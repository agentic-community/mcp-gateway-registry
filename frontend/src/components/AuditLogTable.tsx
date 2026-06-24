import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  ChevronLeftIcon,
  ChevronRightIcon,
  ChevronDoubleLeftIcon,
  ChevronDoubleRightIcon,
  ChevronDownIcon,
  ChevronUpIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';
import { AuditFilters } from './AuditFilterBar';

export interface AuditEvent {
  _id?: string;
  timestamp: string;
  request_id: string;
  log_type: string;
  version?: string;
  correlation_id?: string;
  identity: {
    username: string;
    auth_method: string;
    provider?: string;
    groups?: string[];
    scopes?: string[];
    is_admin: boolean;
    credential_type: string;
    credential_hint?: string;
  };
  request?: {
    method: string;
    path: string;
    query_params?: Record<string, unknown>;
    client_ip: string;
    forwarded_for?: string;
    user_agent?: string;
    content_length?: number;
  };
  response?: {
    status_code: number;
    duration_ms: number;
    content_length?: number;
  };
  action?: {
    operation: string;
    resource_type: string;
    resource_id?: string;
    description?: string;
  };
  authorization?: {
    decision: string;
    required_permission?: string;
    evaluated_scopes?: string[];
  };
  // MCP-specific fields
  mcp_server?: {
    name: string;
    path: string;
    version?: string;
    proxy_target: string;
  };
  mcp_request?: {
    method: string;
    tool_name?: string;
    resource_uri?: string;
    mcp_session_id?: string;
    transport: string;
    jsonrpc_id?: string;
  };
  mcp_response?: {
    status: string;
    duration_ms: number;
    error_code?: number;
    error_message?: string;
  };
}

interface AuditLogTableProps {
  filters: AuditFilters;
  onEventSelect?: (event: AuditEvent) => void;
  selectedEventId?: string;
}

interface PaginationState {
  total: number;
  limit: number;
  offset: number;
}

const getStatusColor = (statusCode: number): string => {
  if (statusCode >= 200 && statusCode < 300) {
    return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400';
  }
  if (statusCode >= 300 && statusCode < 400) {
    return 'bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-400';
  }
  if (statusCode >= 400 && statusCode < 500) {
    return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400';
  }
  if (statusCode >= 500) {
    return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
  }
  return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300';
};

const getMethodColor = (method: string): string => {
  switch (method.toUpperCase()) {
    case 'GET':
      return 'text-blue-600 dark:text-blue-400';
    case 'POST':
      return 'text-green-600 dark:text-green-400';
    case 'PUT':
    case 'PATCH':
      return 'text-yellow-600 dark:text-yellow-400';
    case 'DELETE':
      return 'text-red-600 dark:text-red-400';
    default:
      return 'text-gray-600 dark:text-gray-400';
  }
};

const getMcpStatusColor = (status: string): string => {
  switch (status.toLowerCase()) {
    case 'success':
      return 'bg-green-100 text-green-800 dark:bg-green-900/30 dark:text-green-400';
    case 'error':
      return 'bg-red-100 text-red-800 dark:bg-red-900/30 dark:text-red-400';
    case 'timeout':
      return 'bg-yellow-100 text-yellow-800 dark:bg-yellow-900/30 dark:text-yellow-400';
    default:
      return 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-300';
  }
};

const formatTimestamp = (timestamp: string): string => {
  try {
    const date = new Date(timestamp);
    return date.toLocaleString(undefined, {
      year: 'numeric',
      month: 'short',
      day: '2-digit',
      hour: '2-digit',
      minute: '2-digit',
      second: '2-digit',
    });
  } catch {
    return timestamp;
  }
};

const AuditLogTable: React.FC<AuditLogTableProps> = ({
  filters,
  onEventSelect,
  selectedEventId,
}) => {
  const [events, setEvents] = useState<AuditEvent[]>([]);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [pagination, setPagination] = useState<PaginationState>({
    total: 0,
    limit: 50,
    offset: 0,
  });
  // Sort order: -1 = descending (newest first), 1 = ascending (oldest first)
  const [sortOrder, setSortOrder] = useState<-1 | 1>(-1);

  const fetchEvents = useCallback(async (offset: number = 0, currentSortOrder: -1 | 1 = sortOrder) => {
    setLoading(true);
    setError(null);

    try {
      const params = new URLSearchParams();
      params.set('stream', filters.stream);
      params.set('limit', pagination.limit.toString());
      params.set('offset', offset.toString());
      params.set('sort_order', currentSortOrder.toString());

      if (filters.from) {
        params.set('from', new Date(filters.from).toISOString());
      }
      if (filters.to) {
        params.set('to', new Date(filters.to).toISOString());
      }
      if (filters.username) {
        params.set('username', filters.username);
      }
      if (filters.operation) {
        params.set('operation', filters.operation);
      }
      if (filters.resourceType) {
        params.set('resource_type', filters.resourceType);
      }
      if (filters.statusMin !== undefined) {
        params.set('status_min', filters.statusMin.toString());
      }
      if (filters.statusMax !== undefined) {
        params.set('status_max', filters.statusMax.toString());
      }

      const response = await axios.get(`/api/audit/events?${params.toString()}`);
      const data = response.data;

      setEvents(data.events || []);
      setPagination({
        total: data.total || 0,
        limit: data.limit || 50,
        offset: data.offset || 0,
      });
    } catch (err: any) {
      console.error('Failed to fetch audit events:', err);
      if (err.response?.status === 403) {
        setError('Access denied. Admin permissions required.');
      } else {
        setError(err.response?.data?.detail || 'Failed to load audit events');
      }
      setEvents([]);
    } finally {
      setLoading(false);
    }
  }, [filters, pagination.limit, sortOrder]);

  useEffect(() => {
    fetchEvents(0, sortOrder);
  }, [filters, sortOrder]); // eslint-disable-line react-hooks/exhaustive-deps

  const handlePageChange = (newOffset: number) => {
    fetchEvents(newOffset, sortOrder);
  };

  const handleSortToggle = () => {
    const newSortOrder = sortOrder === -1 ? 1 : -1;
    setSortOrder(newSortOrder);
  };

  const totalPages = Math.ceil(pagination.total / pagination.limit);
  const currentPage = Math.floor(pagination.offset / pagination.limit) + 1;

  const handleFirstPage = () => handlePageChange(0);
  const handlePrevPage = () => handlePageChange(Math.max(0, pagination.offset - pagination.limit));
  const handleNextPage = () => handlePageChange(pagination.offset + pagination.limit);
  const handleLastPage = () => handlePageChange((totalPages - 1) * pagination.limit);

  const isMcpStream = filters.stream === 'mcp_access';
  const isTokenMintStream = filters.stream === 'token_mint';

  if (error) {
    return (
      <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4">
        <div className="flex items-center gap-2 text-red-700 dark:text-red-400">
          <ExclamationTriangleIcon className="h-5 w-5" />
          <span>{error}</span>
        </div>
      </div>
    );
  }

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 overflow-hidden">
      {/* Table */}
      <div className="overflow-x-auto">
        <table className="w-full">
          <thead>
            <tr className="bg-gray-50 dark:bg-gray-900/50 border-b border-gray-200 dark:border-gray-700">
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                <button
                  onClick={handleSortToggle}
                  className="flex items-center gap-1 hover:text-gray-700 dark:hover:text-gray-200 transition-colors"
                  title={sortOrder === -1 ? "Sorted newest first - click for oldest first" : "Sorted oldest first - click for newest first"}
                >
                  Timestamp
                  {sortOrder === -1 ? (
                    <ChevronDownIcon className="h-3 w-3" />
                  ) : (
                    <ChevronUpIcon className="h-3 w-3" />
                  )}
                </button>
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                {isTokenMintStream ? 'User (hash)' : 'User'}
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                {isTokenMintStream ? 'Token Kind' : isMcpStream ? 'MCP Method' : 'Method'}
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                {isTokenMintStream ? 'Path' : isMcpStream ? 'Tool/Resource' : 'Operation'}
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                {isTokenMintStream ? 'Resource' : isMcpStream ? 'MCP Server' : 'Resource'}
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                {isTokenMintStream ? 'Outcome' : 'Status'}
              </th>
              <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                Duration
              </th>
            </tr>
          </thead>
          <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
            {loading ? (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center">
                  <div className="flex items-center justify-center gap-2 text-gray-500 dark:text-gray-400">
                    <div className="animate-spin h-5 w-5 border-2 border-blue-500 border-t-transparent rounded-full" />
                    <span>Loading events...</span>
                  </div>
                </td>
              </tr>
            ) : events.length === 0 ? (
              <tr>
                <td colSpan={7} className="px-4 py-8 text-center text-gray-500 dark:text-gray-400">
                  No audit events found matching the current filters.
                </td>
              </tr>
            ) : (
              events.map((event) => (
                <tr
                  key={event.request_id}
                  onClick={() => onEventSelect?.(event)}
                  className={`cursor-pointer transition-colors ${
                    selectedEventId === event.request_id
                      ? 'bg-blue-50 dark:bg-blue-900/20'
                      : 'hover:bg-gray-50 dark:hover:bg-gray-700/50'
                  }`}
                >
                  <td className="px-4 py-3 text-sm text-gray-900 dark:text-gray-100 whitespace-nowrap">
                    {formatTimestamp(event.timestamp)}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    <div className="flex items-center gap-1">
                      <span className="text-gray-900 dark:text-gray-100">
                        {isTokenMintStream ? event.username_hash : event.identity?.username || '-'}
                      </span>
                      {!isTokenMintStream && event.identity?.is_admin && (
                        <span className="px-1.5 py-0.5 text-xs font-medium bg-purple-100 text-purple-700 dark:bg-purple-900/30 dark:text-purple-300 rounded">
                          Admin
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-3 text-sm">
                    {isTokenMintStream ? (
                      <span className={`px-2 py-1 text-xs font-medium rounded ${event.token_kind === 'resource' ? 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300' : 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300'}`}>
                        {event.token_kind || '-'}
                      </span>
                    ) : isMcpStream ? (
                      <span className="font-mono text-gray-700 dark:text-gray-300">
                        {event.mcp_request?.method || '-'}
                      </span>
                    ) : (
                      <span className={`font-mono font-medium ${getMethodColor(event.request?.method || '')}`}>
                        {event.request?.method || '-'}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">
                    {isTokenMintStream ? (
                      <span className="font-mono">{event.token_path || '-'}</span>
                    ) : isMcpStream ? (
                      event.mcp_request?.tool_name || event.mcp_request?.resource_uri || '-'
                    ) : (
                      event.action?.operation || '-'
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300">
                    {isTokenMintStream ? (
                      event.resource_type ? (
                        <span>
                          {event.resource_type}
                          {event.resource_id && (
                            <span className="text-gray-500 dark:text-gray-400">
                              /{event.resource_id}
                            </span>
                          )}
                        </span>
                      ) : (
                        <span className="text-gray-400 dark:text-gray-500">-</span>
                      )
                    ) : isMcpStream ? (
                      event.mcp_server?.name || '-'
                    ) : event.action ? (
                      <span>
                        {event.action.resource_type}
                        {event.action.resource_id && (
                          <span className="text-gray-500 dark:text-gray-400">
                            /{event.action.resource_id}
                          </span>
                        )}
                      </span>
                    ) : (
                      <span className="text-gray-400 dark:text-gray-500">-</span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm">
                    {isTokenMintStream ? (
                      <span className={`px-2 py-1 text-xs font-medium rounded ${event.outcome === 'success' ? 'bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-300' : 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300'}`}>
                        {event.outcome || '-'}
                      </span>
                    ) : isMcpStream ? (
                      <span className={`px-2 py-1 text-xs font-medium rounded ${getMcpStatusColor(event.mcp_response?.status || '')}`}>
                        {event.mcp_response?.status || '-'}
                      </span>
                    ) : (
                      <span className={`px-2 py-1 text-xs font-medium rounded ${getStatusColor(event.response?.status_code || 0)}`}>
                        {event.response?.status_code || '-'}
                      </span>
                    )}
                  </td>
                  <td className="px-4 py-3 text-sm text-gray-700 dark:text-gray-300 whitespace-nowrap">
                    {isTokenMintStream
                      ? (event.expires_in_seconds ? `${Math.round(event.expires_in_seconds / 3600)}h` : '-')
                      : isMcpStream
                        ? `${(event.mcp_response?.duration_ms || 0).toFixed(1)} ms`
                        : `${(event.response?.duration_ms || 0).toFixed(1)} ms`
                    }
                  </td>
                </tr>
              ))
            )}
          </tbody>
        </table>
      </div>

      {/* Pagination */}
      {!loading && events.length > 0 && (
        <div className="px-4 py-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50">
          <div className="flex items-center justify-between">
            <div className="text-sm text-gray-700 dark:text-gray-300">
              Showing{' '}
              <span className="font-medium">{pagination.offset + 1}</span>
              {' '}-{' '}
              <span className="font-medium">
                {Math.min(pagination.offset + pagination.limit, pagination.total)}
              </span>
              {' '}of{' '}
              <span className="font-medium">{pagination.total}</span>
              {' '}events
            </div>
            <div className="flex items-center gap-1">
              <button
                onClick={handleFirstPage}
                disabled={currentPage === 1}
                className="p-1.5 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="First page"
              >
                <ChevronDoubleLeftIcon className="h-4 w-4" />
              </button>
              <button
                onClick={handlePrevPage}
                disabled={currentPage === 1}
                className="p-1.5 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="Previous page"
              >
                <ChevronLeftIcon className="h-4 w-4" />
              </button>
              <span className="px-3 py-1 text-sm text-gray-700 dark:text-gray-300">
                Page {currentPage} of {totalPages}
              </span>
              <button
                onClick={handleNextPage}
                disabled={currentPage === totalPages}
                className="p-1.5 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="Next page"
              >
                <ChevronRightIcon className="h-4 w-4" />
              </button>
              <button
                onClick={handleLastPage}
                disabled={currentPage === totalPages}
                className="p-1.5 text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded disabled:opacity-50 disabled:cursor-not-allowed"
                title="Last page"
              >
                <ChevronDoubleRightIcon className="h-4 w-4" />
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AuditLogTable;
