import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  ArrowDownTrayIcon,
  ArrowPathIcon,
  FunnelIcon,
  XMarkIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';


interface LogEntry {
  timestamp: string;
  hostname: string;
  service: string;
  level: string;
  logger: string;
  filename: string;
  lineno: number;
  message: string;
}

interface LogQueryResponse {
  entries: LogEntry[];
  total_count: number;
  limit: number;
  offset: number;
  has_next: boolean;
}

interface LogMetadata {
  services: string[];
  hostnames: string[];
  levels: string[];
}

interface LogFilters {
  service: string;
  level: string;
  hostname: string;
  search: string;
  start: string;
  end: string;
}

interface ApplicationLogsProps {
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
}

const LEVEL_COLORS: Record<string, string> = {
  DEBUG: 'bg-gray-100 text-gray-700 dark:bg-gray-700 dark:text-gray-300',
  INFO: 'bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300',
  WARNING: 'bg-yellow-100 text-yellow-700 dark:bg-yellow-900/30 dark:text-yellow-300',
  ERROR: 'bg-red-100 text-red-700 dark:bg-red-900/30 dark:text-red-300',
  CRITICAL: 'bg-red-200 text-red-900 dark:bg-red-900/50 dark:text-red-200',
};

const PAGE_SIZE = 50;


const ApplicationLogs: React.FC<ApplicationLogsProps> = ({ onShowToast }) => {
  const [entries, setEntries] = useState<LogEntry[]>([]);
  const [totalCount, setTotalCount] = useState(0);
  const [hasNext, setHasNext] = useState(false);
  const [offset, setOffset] = useState(0);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [metadata, setMetadata] = useState<LogMetadata | null>(null);
  const [showFilters, setShowFilters] = useState(false);
  const [expandedRow, setExpandedRow] = useState<number | null>(null);
  const [fetchTrigger, setFetchTrigger] = useState(0);
  const [filters, setFilters] = useState<LogFilters>({
    service: '',
    level: '',
    hostname: '',
    search: '',
    start: '',
    end: '',
  });

  const _buildParams = useCallback((extraOffset?: number): URLSearchParams => {
    const params = new URLSearchParams();
    params.set('limit', PAGE_SIZE.toString());
    params.set('offset', (extraOffset ?? offset).toString());
    if (filters.service) params.set('service', filters.service);
    if (filters.level) params.set('level', filters.level);
    if (filters.hostname) params.set('hostname', filters.hostname);
    if (filters.search) params.set('search', filters.search);
    if (filters.start) params.set('start', new Date(filters.start).toISOString());
    if (filters.end) params.set('end', new Date(filters.end).toISOString());
    return params;
  }, [filters, offset]);

  const _fetchLogs = useCallback(async (currentOffset: number) => {
    setLoading(true);
    setError(null);
    try {
      const params = _buildParams(currentOffset);
      const response = await axios.get<LogQueryResponse>(
        `/api/admin/logs?${params.toString()}`
      );
      setEntries(response.data.entries);
      setTotalCount(response.data.total_count);
      setHasNext(response.data.has_next);
    } catch (err: any) {
      if (err.response?.status === 403) {
        setError('Access denied. Admin permissions required.');
      } else if (err.response?.status === 503) {
        setError('Centralized application logging is not enabled. Set APP_LOG_CENTRALIZED_ENABLED=true.');
      } else {
        setError(err.response?.data?.detail || 'Failed to load application logs.');
      }
    } finally {
      setLoading(false);
    }
  }, [_buildParams]);

  const _fetchMetadata = useCallback(async () => {
    try {
      const response = await axios.get<LogMetadata>('/api/admin/logs/metadata');
      setMetadata(response.data);
    } catch {
      // Metadata is optional; silently ignore
    }
  }, []);

  useEffect(() => {
    _fetchLogs(offset);
  }, [offset, fetchTrigger, _fetchLogs]);

  useEffect(() => {
    _fetchMetadata();
  }, [_fetchMetadata]);

  const handleApplyFilters = () => {
    setExpandedRow(null);
    setOffset(0);
    setFetchTrigger((prev) => prev + 1);
  };

  const handleClearFilters = () => {
    setFilters({ service: '', level: '', hostname: '', search: '', start: '', end: '' });
    setExpandedRow(null);
    setOffset(0);
    setFetchTrigger((prev) => prev + 1);
  };

  const handleExport = useCallback(() => {
    const params = _buildParams(0);
    params.set('limit', '50000');
    params.delete('offset');
    window.open(`/api/admin/logs/export?${params.toString()}`, '_blank');
    onShowToast('Log export started', 'info');
  }, [_buildParams, onShowToast]);

  const handleRefresh = () => {
    setExpandedRow(null);
    _fetchLogs(offset);
  };

  const handlePrevPage = () => {
    const newOffset = Math.max(0, offset - PAGE_SIZE);
    setOffset(newOffset);
    setExpandedRow(null);
  };

  const handleNextPage = () => {
    if (hasNext) {
      setOffset(offset + PAGE_SIZE);
      setExpandedRow(null);
    }
  };

  const _activeFilterCount = (): number => {
    let count = 0;
    if (filters.service) count++;
    if (filters.level) count++;
    if (filters.hostname) count++;
    if (filters.search) count++;
    if (filters.start) count++;
    if (filters.end) count++;
    return count;
  };

  const _formatTimestamp = (ts: string): string => {
    try {
      const d = new Date(ts);
      return d.toLocaleString(undefined, {
        year: 'numeric',
        month: '2-digit',
        day: '2-digit',
        hour: '2-digit',
        minute: '2-digit',
        second: '2-digit',
        hour12: false,
      });
    } catch {
      return ts;
    }
  };

  const _truncateMessage = (msg: string, maxLen: number = 120): string => {
    const firstLine = msg.split('\n')[0];
    if (firstLine.length <= maxLen) return firstLine;
    return firstLine.substring(0, maxLen) + '...';
  };

  const currentPage = Math.floor(offset / PAGE_SIZE) + 1;
  const totalPages = Math.ceil(totalCount / PAGE_SIZE);
  const filterCount = _activeFilterCount();

  return (
    <div>
      {/* Header */}
      <div className="mb-6 flex items-center justify-between">
        <div>
          <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
            Application Logs
          </h2>
          <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
            View and download centralized application logs from all services.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={handleRefresh}
            className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors"
            title="Refresh"
          >
            <ArrowPathIcon className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <button
            onClick={handleExport}
            className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors"
            title="Export as JSONL"
          >
            <ArrowDownTrayIcon className="h-4 w-4" />
            <span>Download JSONL</span>
          </button>
        </div>
      </div>

      {/* Filter Toggle */}
      <div className="mb-4">
        <button
          onClick={() => setShowFilters(!showFilters)}
          className={`flex items-center gap-2 px-3 py-2 text-sm font-medium rounded-lg border transition-colors ${
            filterCount > 0
              ? 'text-purple-700 dark:text-purple-300 bg-purple-50 dark:bg-purple-900/20 border-purple-300 dark:border-purple-700'
              : 'text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 border-gray-300 dark:border-gray-600 hover:bg-gray-50 dark:hover:bg-gray-600'
          }`}
        >
          <FunnelIcon className="h-4 w-4" />
          <span>Filters{filterCount > 0 ? ` (${filterCount})` : ''}</span>
        </button>
      </div>

      {/* Filter Panel */}
      {showFilters && (
        <div className="mb-6 bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 p-4">
          <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
            {/* Service */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Service
              </label>
              <select
                value={filters.service}
                onChange={(e) => setFilters({ ...filters, service: e.target.value })}
                className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              >
                <option value="">All Services</option>
                {metadata?.services.map((s) => (
                  <option key={s} value={s}>{s}</option>
                ))}
              </select>
            </div>

            {/* Level */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Level
              </label>
              <select
                value={filters.level}
                onChange={(e) => setFilters({ ...filters, level: e.target.value })}
                className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              >
                <option value="">All Levels</option>
                {metadata?.levels.map((l) => (
                  <option key={l} value={l}>{l}</option>
                ))}
              </select>
            </div>

            {/* Hostname */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Hostname / Pod
              </label>
              <select
                value={filters.hostname}
                onChange={(e) => setFilters({ ...filters, hostname: e.target.value })}
                className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              >
                <option value="">All Hosts</option>
                {metadata?.hostnames.map((h) => (
                  <option key={h} value={h}>{h}</option>
                ))}
              </select>
            </div>

            {/* Search */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                Search in message
              </label>
              <input
                type="text"
                value={filters.search}
                onChange={(e) => setFilters({ ...filters, search: e.target.value })}
                onKeyDown={(e) => { if (e.key === 'Enter') handleApplyFilters(); }}
                placeholder="e.g. timeout, connection refused"
                className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              />
            </div>

            {/* Start time */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                From
              </label>
              <input
                type="datetime-local"
                value={filters.start}
                onChange={(e) => setFilters({ ...filters, start: e.target.value })}
                className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              />
            </div>

            {/* End time */}
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
                To
              </label>
              <input
                type="datetime-local"
                value={filters.end}
                onChange={(e) => setFilters({ ...filters, end: e.target.value })}
                className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-purple-500 focus:border-transparent"
              />
            </div>
          </div>

          {/* Filter actions */}
          <div className="flex items-center gap-3 mt-4">
            <button
              onClick={handleApplyFilters}
              className="px-4 py-2 text-sm font-medium text-white bg-purple-600 rounded-lg hover:bg-purple-700 transition-colors"
            >
              Apply Filters
            </button>
            {filterCount > 0 && (
              <button
                onClick={handleClearFilters}
                className="flex items-center gap-1 px-3 py-2 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200 transition-colors"
              >
                <XMarkIcon className="h-4 w-4" />
                Clear All
              </button>
            )}
          </div>
        </div>
      )}

      {/* Error State */}
      {error && (
        <div className="mb-6 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg p-4 flex items-center gap-3">
          <ExclamationTriangleIcon className="h-5 w-5 text-red-500 flex-shrink-0" />
          <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
        </div>
      )}

      {/* Summary bar */}
      {!error && (
        <div className="mb-4 flex items-center justify-between text-sm text-gray-600 dark:text-gray-400">
          <span>
            {totalCount.toLocaleString()} log entries
            {filterCount > 0 && ' (filtered)'}
          </span>
          <span>
            Page {currentPage} of {totalPages || 1}
          </span>
        </div>
      )}

      {/* Log Table */}
      {!error && (
        <div className="bg-white dark:bg-gray-800 rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
          {loading && entries.length === 0 ? (
            <div className="flex justify-center items-center py-16">
              <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-purple-600"></div>
            </div>
          ) : entries.length === 0 ? (
            <div className="text-center py-16 text-gray-500 dark:text-gray-400">
              No log entries found.
            </div>
          ) : (
            <div className="overflow-x-auto">
              <table className="w-full text-sm" aria-label="Application log entries">
                <thead>
                  <tr className="border-b border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/50">
                    <th className="px-4 py-3 text-left font-medium text-gray-600 dark:text-gray-400 w-40">Timestamp</th>
                    <th className="px-4 py-3 text-left font-medium text-gray-600 dark:text-gray-400 w-24">Level</th>
                    <th className="px-4 py-3 text-left font-medium text-gray-600 dark:text-gray-400 w-28">Service</th>
                    <th className="px-4 py-3 text-left font-medium text-gray-600 dark:text-gray-400 w-36">Source</th>
                    <th className="px-4 py-3 text-left font-medium text-gray-600 dark:text-gray-400">Message</th>
                  </tr>
                </thead>
                <tbody className="divide-y divide-gray-100 dark:divide-gray-700/50">
                  {entries.map((entry, idx) => (
                    <React.Fragment key={idx}>
                      <tr
                        className={`hover:bg-gray-50 dark:hover:bg-gray-700/50 cursor-pointer transition-colors ${
                          expandedRow === idx ? 'bg-gray-50 dark:bg-gray-700/50' : ''
                        }`}
                        role="button"
                        tabIndex={0}
                        aria-expanded={expandedRow === idx}
                        aria-label={`${entry.level} log from ${entry.service} at ${_formatTimestamp(entry.timestamp)}`}
                        onClick={() => setExpandedRow(expandedRow === idx ? null : idx)}
                        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); setExpandedRow(expandedRow === idx ? null : idx); } }}
                      >
                        <td className="px-4 py-2.5 text-gray-600 dark:text-gray-400 font-mono text-xs whitespace-nowrap">
                          {_formatTimestamp(entry.timestamp)}
                        </td>
                        <td className="px-4 py-2.5">
                          <span className={`inline-block px-2 py-0.5 text-xs font-medium rounded ${LEVEL_COLORS[entry.level] || LEVEL_COLORS.INFO}`}>
                            {entry.level}
                          </span>
                        </td>
                        <td className="px-4 py-2.5 text-gray-700 dark:text-gray-300 font-mono text-xs">
                          {entry.service}
                        </td>
                        <td className="px-4 py-2.5 text-gray-500 dark:text-gray-500 font-mono text-xs">
                          {entry.filename}:{entry.lineno}
                        </td>
                        <td className="px-4 py-2.5 text-gray-800 dark:text-gray-200 text-xs">
                          {_truncateMessage(entry.message)}
                        </td>
                      </tr>
                      {expandedRow === idx && (
                        <tr>
                          <td colSpan={5} className="px-4 py-3 bg-gray-50 dark:bg-gray-900/30">
                            <div className="space-y-2">
                              <div className="flex gap-6 text-xs text-gray-500 dark:text-gray-400">
                                <span><strong>Hostname:</strong> {entry.hostname}</span>
                                <span><strong>Logger:</strong> {entry.logger}</span>
                              </div>
                              <pre className="text-xs font-mono text-gray-700 dark:text-gray-300 whitespace-pre-wrap break-words bg-white dark:bg-gray-800 rounded p-3 border border-gray-200 dark:border-gray-700 max-h-64 overflow-y-auto">
                                {entry.message}
                              </pre>
                            </div>
                          </td>
                        </tr>
                      )}
                    </React.Fragment>
                  ))}
                </tbody>
              </table>
            </div>
          )}

          {/* Pagination */}
          {entries.length > 0 && (
            <div className="flex items-center justify-between px-4 py-3 border-t border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900/30">
              <div className="text-xs text-gray-500 dark:text-gray-400">
                Showing {offset + 1}-{Math.min(offset + entries.length, totalCount)} of {totalCount.toLocaleString()}
              </div>
              <div className="flex items-center gap-2">
                <button
                  onClick={handlePrevPage}
                  disabled={offset === 0}
                  className="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronLeftIcon className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                </button>
                <span className="text-xs text-gray-600 dark:text-gray-400">
                  {currentPage} / {totalPages}
                </span>
                <button
                  onClick={handleNextPage}
                  disabled={!hasNext}
                  className="p-1.5 rounded hover:bg-gray-200 dark:hover:bg-gray-600 disabled:opacity-30 disabled:cursor-not-allowed transition-colors"
                >
                  <ChevronRightIcon className="h-4 w-4 text-gray-600 dark:text-gray-400" />
                </button>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
};

export default ApplicationLogs;
