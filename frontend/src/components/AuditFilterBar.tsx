import React, { useState, useEffect, useRef } from 'react';
import SearchableSelect, { SelectOption } from './SearchableSelect';
import axios from 'axios';
import {
  FunnelIcon,
  XMarkIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';

export interface AuditFilters {
  stream: 'registry_api' | 'mcp_access';
  from?: string;
  to?: string;
  username?: string;
  operation?: string;
  resourceType?: string;
  statusMin?: number;
  statusMax?: number;
}

interface AuditFilterBarProps {
  filters: AuditFilters;
  onFilterChange: (filters: AuditFilters) => void;
  onRefresh?: () => void;
  loading?: boolean;
}

const REGISTRY_OPERATION_OPTIONS = [
  { value: '', label: 'All Operations' },
  { value: 'create', label: 'Create' },
  { value: 'read', label: 'Read' },
  { value: 'update', label: 'Update' },
  { value: 'delete', label: 'Delete' },
  { value: 'list', label: 'List' },
  { value: 'toggle', label: 'Toggle' },
  { value: 'rate', label: 'Rate' },
  { value: 'login', label: 'Login' },
  { value: 'logout', label: 'Logout' },
  { value: 'search', label: 'Search' },
];

const MCP_OPERATION_OPTIONS = [
  { value: '', label: 'All Methods' },
  { value: 'initialize', label: 'Initialize' },
  { value: 'tools/list', label: 'Tools List' },
  { value: 'tools/call', label: 'Tools Call' },
  { value: 'resources/list', label: 'Resources List' },
  { value: 'resources/templates/list', label: 'Resource Templates' },
  { value: 'notifications/initialized', label: 'Notifications' },
];

const REGISTRY_RESOURCE_TYPE_OPTIONS = [
  { value: '', label: 'All Resources' },
  { value: 'server', label: 'Server' },
  { value: 'agent', label: 'Agent' },
  { value: 'auth', label: 'Auth' },
  { value: 'federation', label: 'Federation' },
  { value: 'health', label: 'Health' },
  { value: 'search', label: 'Search' },
];

const MCP_RESOURCE_TYPE_OPTIONS = [
  { value: '', label: 'All Servers' },
];

const STATUS_PRESETS = [
  { value: '', label: 'All Status Codes' },
  { value: '2xx', label: '2xx Success' },
  { value: '4xx', label: '4xx Client Error' },
  { value: '5xx', label: '5xx Server Error' },
  { value: 'error', label: 'All Errors (4xx & 5xx)' },
];

interface FilterOptionsCache {
  registry_api?: { usernames: SelectOption[]; serverNames: SelectOption[] };
  mcp_access?: { usernames: SelectOption[]; serverNames: SelectOption[] };
}

const AuditFilterBar: React.FC<AuditFilterBarProps> = ({
  filters,
  onFilterChange,
  onRefresh,
  loading = false,
}) => {
  const isMcpStream = filters.stream === 'mcp_access';
  const operationOptions = isMcpStream ? MCP_OPERATION_OPTIONS : REGISTRY_OPERATION_OPTIONS;
  const resourceTypeOptions = isMcpStream ? MCP_RESOURCE_TYPE_OPTIONS : REGISTRY_RESOURCE_TYPE_OPTIONS;

  const [usernameOptions, setUsernameOptions] = useState<SelectOption[]>([]);
  const [serverNameOptions, setServerNameOptions] = useState<SelectOption[]>([]);
  const [optionsLoading, setOptionsLoading] = useState(false);
  const optionsCacheRef = useRef<FilterOptionsCache>({});

  // Prefetch both streams' filter options on mount
  useEffect(() => {
    const fetchAllOptions = async () => {
      setOptionsLoading(true);
      try {
        const [registryRes, mcpRes] = await Promise.all([
          axios.get('/api/audit/filter-options', { params: { stream: 'registry_api' } }),
          axios.get('/api/audit/filter-options', { params: { stream: 'mcp_access' } }),
        ]);

        optionsCacheRef.current = {
          registry_api: {
            usernames: registryRes.data.usernames.map((u: string) => ({ value: u, label: u })),
            serverNames: [],
          },
          mcp_access: {
            usernames: mcpRes.data.usernames.map((u: string) => ({ value: u, label: u })),
            serverNames: mcpRes.data.server_names.map((s: string) => ({ value: s, label: s })),
          },
        };

        // Set current stream's options
        const current = optionsCacheRef.current[filters.stream];
        if (current) {
          setUsernameOptions(current.usernames);
          setServerNameOptions(current.serverNames);
        }
      } catch (error) {
        console.error('Failed to fetch filter options:', error);
      } finally {
        setOptionsLoading(false);
      }
    };
    fetchAllOptions();
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // When stream changes, serve from cache
  useEffect(() => {
    const cached = optionsCacheRef.current[filters.stream];
    if (cached) {
      setUsernameOptions(cached.usernames);
      setServerNameOptions(cached.serverNames);
    }
  }, [filters.stream]);

  const handleStreamChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    // Clear operation and resource type filters when switching streams
    onFilterChange({
      ...filters,
      stream: e.target.value as 'registry_api' | 'mcp_access',
      operation: undefined,
      resourceType: undefined,
    });
  };

  const handleFromChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onFilterChange({
      ...filters,
      from: e.target.value || undefined,
    });
  };

  const handleToChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    onFilterChange({
      ...filters,
      to: e.target.value || undefined,
    });
  };

  const handleUsernameSelect = (value: string) => {
    onFilterChange({
      ...filters,
      username: value || undefined,
    });
  };

  const handleOperationChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onFilterChange({
      ...filters,
      operation: e.target.value || undefined,
    });
  };

  const handleResourceTypeChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    onFilterChange({
      ...filters,
      resourceType: e.target.value || undefined,
    });
  };

  const handleServerNameSelect = (value: string) => {
    onFilterChange({
      ...filters,
      resourceType: value || undefined,
    });
  };

  const handleStatusPresetChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    const value = e.target.value;
    let statusMin: number | undefined;
    let statusMax: number | undefined;

    switch (value) {
      case '2xx':
        statusMin = 200;
        statusMax = 299;
        break;
      case '4xx':
        statusMin = 400;
        statusMax = 499;
        break;
      case '5xx':
        statusMin = 500;
        statusMax = 599;
        break;
      case 'error':
        statusMin = 400;
        statusMax = 599;
        break;
      default:
        statusMin = undefined;
        statusMax = undefined;
    }

    onFilterChange({
      ...filters,
      statusMin,
      statusMax,
    });
  };

  const getStatusPresetValue = (): string => {
    const { statusMin, statusMax } = filters;
    if (statusMin === 200 && statusMax === 299) return '2xx';
    if (statusMin === 400 && statusMax === 499) return '4xx';
    if (statusMin === 500 && statusMax === 599) return '5xx';
    if (statusMin === 400 && statusMax === 599) return 'error';
    return '';
  };

  const handleClearFilters = () => {
    onFilterChange({
      stream: filters.stream,
      from: undefined,
      to: undefined,
      username: undefined,
      operation: undefined,
      resourceType: undefined,
      statusMin: undefined,
      statusMax: undefined,
    });
  };

  const hasActiveFilters = !!(
    filters.from ||
    filters.to ||
    filters.username ||
    filters.operation ||
    filters.resourceType ||
    filters.statusMin ||
    filters.statusMax
  );

  return (
    <div className="bg-white dark:bg-gray-800 rounded-lg shadow-sm border border-gray-200 dark:border-gray-700 p-4 mb-4">
      <div className="flex items-center gap-2 mb-4">
        <FunnelIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
        <h3 className="text-sm font-medium text-gray-700 dark:text-gray-300">
          Filters
        </h3>
        {hasActiveFilters && (
          <button
            onClick={handleClearFilters}
            className="ml-auto flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          >
            <XMarkIcon className="h-4 w-4" />
            Clear filters
          </button>
        )}
        {onRefresh && (
          <button
            onClick={onRefresh}
            disabled={loading}
            className="ml-2 p-1.5 text-gray-500 hover:text-blue-600 dark:text-gray-400 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded transition-colors disabled:opacity-50"
            title="Refresh"
          >
            <ArrowPathIcon className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
        {/* Stream Selector */}
        <div>
          <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
            Log Stream
          </label>
          <select
            value={filters.stream}
            onChange={handleStreamChange}
            className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            <option value="registry_api">Registry API</option>
            <option value="mcp_access">MCP Access</option>
          </select>
        </div>

        {/* Date Range - From */}
        <div>
          <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
            From Date
          </label>
          <input
            type="datetime-local"
            value={filters.from || ''}
            onChange={handleFromChange}
            className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        {/* Date Range - To */}
        <div>
          <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
            To Date
          </label>
          <input
            type="datetime-local"
            value={filters.to || ''}
            onChange={handleToChange}
            className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          />
        </div>

        {/* Username Filter */}
        <div>
          <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
            Username
          </label>
          <SearchableSelect
            options={usernameOptions}
            value={filters.username || ''}
            onChange={handleUsernameSelect}
            placeholder="Search username..."
            isLoading={optionsLoading}
            allowCustom={true}
            specialOptions={[{ value: '', label: 'All Users' }]}
            focusColor="focus:ring-blue-500"
          />
        </div>

        {/* Operation / MCP Method Filter */}
        <div>
          <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
            {isMcpStream ? 'MCP Method' : 'Operation'}
          </label>
          <select
            value={filters.operation || ''}
            onChange={handleOperationChange}
            className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            {operationOptions.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>

        {/* Resource Type / Server Name Filter */}
        <div>
          <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
            {isMcpStream ? 'Server Name' : 'Resource Type'}
          </label>
          {isMcpStream ? (
            <SearchableSelect
              options={serverNameOptions}
              value={filters.resourceType || ''}
              onChange={handleServerNameSelect}
              placeholder="Search server..."
              isLoading={optionsLoading}
              allowCustom={true}
              specialOptions={[{ value: '', label: 'All Servers' }]}
              focusColor="focus:ring-blue-500"
            />
          ) : (
            <select
              value={filters.resourceType || ''}
              onChange={handleResourceTypeChange}
              className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
            >
              {resourceTypeOptions.map((opt) => (
                <option key={opt.value} value={opt.value}>
                  {opt.label}
                </option>
              ))}
            </select>
          )}
        </div>

        {/* Status Code Range Filter */}
        <div>
          <label className="block text-xs font-medium text-gray-600 dark:text-gray-400 mb-1">
            Status Code
          </label>
          <select
            value={getStatusPresetValue()}
            onChange={handleStatusPresetChange}
            className="w-full px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 focus:ring-2 focus:ring-blue-500 focus:border-blue-500"
          >
            {STATUS_PRESETS.map((opt) => (
              <option key={opt.value} value={opt.value}>
                {opt.label}
              </option>
            ))}
          </select>
        </div>
      </div>
    </div>
  );
};

export default AuditFilterBar;
