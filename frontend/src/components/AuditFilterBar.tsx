import React, { useState, useEffect, useRef } from 'react';
import SearchableSelect, { SelectOption } from './SearchableSelect';
import axios from 'axios';
import {
  FunnelIcon,
  XMarkIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';

export interface AuditFilters {
  stream: 'registry_api' | 'mcp_access' | 'token_mint';
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

const TOKEN_MINT_OPERATION_OPTIONS = [
  { value: '', label: 'All Token Kinds' },
  { value: 'resource', label: 'Resource-bound' },
  { value: 'user', label: 'User' },
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
  // The token_mint stream is selectable but its options are never fetched: only
  // registry_api and mcp_access are prefetched on mount, so this key stays
  // absent and the username control falls back to free-text entry.
  token_mint?: { usernames: SelectOption[]; serverNames: SelectOption[] };
}

const AuditFilterBar: React.FC<AuditFilterBarProps> = ({
  filters,
  onFilterChange,
  onRefresh,
  loading = false,
}) => {
  const isMcpStream = filters.stream === 'mcp_access';
  const isTokenMintStream = filters.stream === 'token_mint';
  const operationOptions = isTokenMintStream
    ? TOKEN_MINT_OPERATION_OPTIONS
    : isMcpStream
      ? MCP_OPERATION_OPTIONS
      : REGISTRY_OPERATION_OPTIONS;
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

  // When stream changes, serve from cache. token_mint has no prefetched options
  // (its identities are not enumerated), so CLEAR rather than leave the previous
  // stream's usernames on screen -- offering registry_api identities while
  // querying token_mint is worse than an empty list. Free-text entry still
  // works: the control allows custom values.
  useEffect(() => {
    const cached = optionsCacheRef.current[filters.stream];
    setUsernameOptions(cached?.usernames ?? []);
    setServerNameOptions(cached?.serverNames ?? []);
  }, [filters.stream]);

  const handleStreamChange = (e: React.ChangeEvent<HTMLSelectElement>) => {
    // Clear operation and resource type filters when switching streams
    onFilterChange({
      ...filters,
      stream: e.target.value as 'registry_api' | 'mcp_access' | 'token_mint',
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
    <div className="card p-4">
      <div className="mb-4 flex items-center gap-2">
        <FunnelIcon className="h-4 w-4 text-fg-subtle" />
        <h3 className="text-sm font-semibold text-fg">
          Server filters
        </h3>
        {hasActiveFilters && (
          <button
            onClick={handleClearFilters}
            type="button"
            className="ml-auto flex items-center gap-1 rounded-md px-2 py-1 text-xs font-medium text-fg-muted transition-colors hover:bg-surface-hover hover:text-fg"
          >
            <XMarkIcon className="h-4 w-4" />
            Clear filters
          </button>
        )}
        {onRefresh && (
          <button
            type="button"
            onClick={onRefresh}
            disabled={loading}
            className={`btn-icon h-8 w-8 ${hasActiveFilters ? '' : 'ml-auto'}`}
            title="Refresh"
            aria-label="Refresh audit data"
          >
            <ArrowPathIcon className={`h-4 w-4 ${loading ? 'animate-spin' : ''}`} />
          </button>
        )}
      </div>

      <div className="grid grid-cols-1 gap-4 md:grid-cols-2 lg:grid-cols-4">
        {/* Stream Selector */}
        <div>
          <label htmlFor="audit-filter-stream" className="label">
            Log Stream
          </label>
          <select
            id="audit-filter-stream"
            value={filters.stream}
            onChange={handleStreamChange}
            className="input"
          >
            <option value="registry_api">Registry API</option>
            <option value="mcp_access">MCP Access</option>
            <option value="token_mint">Token Mints</option>
          </select>
        </div>

        {/* Date Range - From */}
        <div>
          <label htmlFor="audit-filter-from" className="label">
            From Date
          </label>
          <input
            id="audit-filter-from"
            type="datetime-local"
            value={filters.from || ''}
            onChange={handleFromChange}
            className="input"
          />
        </div>

        {/* Date Range - To */}
        <div>
          <label htmlFor="audit-filter-to" className="label">
            To Date
          </label>
          <input
            id="audit-filter-to"
            type="datetime-local"
            value={filters.to || ''}
            onChange={handleToChange}
            className="input"
          />
        </div>

        {/* Username / identity filter. The token_mint stream's identities are
            never enumerated (that aggregation is expensive), so for that stream
            this control is a free-text search: the dropdown says so and the
            typed value is committed with Enter. */}
        <div>
          <label className="label">
            Username
          </label>
          <SearchableSelect
            options={usernameOptions}
            value={filters.username || ''}
            onChange={handleUsernameSelect}
            placeholder={
              isTokenMintStream ? 'Search identity or paste an IdP id...' : 'Search username...'
            }
            isLoading={optionsLoading}
            allowCustom={true}
            specialOptions={[{ value: '', label: 'All Users' }]}
            focusColor="focus:ring-focus/40"
            emptyMessage={
              isTokenMintStream
                ? 'Token mint identities are not listed. Type a username, UPN, object id or sub and press Enter to search.'
                : undefined
            }
          />
          {isTokenMintStream && (
            <p className="mt-1 text-xs text-fg-subtle">
              Free-text search: press Enter to search for the identity you typed.
            </p>
          )}
        </div>

        {/* Operation / MCP Method Filter */}
        <div>
          <label htmlFor="audit-filter-operation" className="label">
            {isTokenMintStream ? 'Token Kind' : isMcpStream ? 'MCP Method' : 'Operation'}
          </label>
          <select
            id="audit-filter-operation"
            value={filters.operation || ''}
            onChange={handleOperationChange}
            className="input"
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
          <label htmlFor={isMcpStream ? undefined : 'audit-filter-resource'} className="label">
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
              focusColor="focus:ring-focus/40"
            />
          ) : (
            <select
              id="audit-filter-resource"
              value={filters.resourceType || ''}
              onChange={handleResourceTypeChange}
              className="input"
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
          <label htmlFor="audit-filter-status" className="label">
            Status Code
          </label>
          <select
            id="audit-filter-status"
            value={getStatusPresetValue()}
            onChange={handleStatusPresetChange}
            className="input"
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
