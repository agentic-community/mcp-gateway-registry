import React, { useState, useEffect, useMemo, useCallback } from 'react';
import axios from 'axios';
import {
  MagnifyingGlassIcon,
  ArrowPathIcon,
  ClipboardIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  ArrowDownTrayIcon,
  XMarkIcon,
  CheckIcon,
  ExclamationCircleIcon,
} from '@heroicons/react/24/outline';

/* ------------------------------------------------------------------ */
/*  Types                                                              */
/* ------------------------------------------------------------------ */

interface ConfigField {
  key: string;
  label: string;
  value: string;
  raw_value: string | null;
  is_masked: boolean;
  unit: string | null;
}

interface ConfigSubgroup {
  id: string;
  title: string;
  fields: ConfigField[];
}

interface ConfigGroup {
  id: string;
  title: string;
  order: number;
  fields: ConfigField[];
  subgroups?: ConfigSubgroup[];
}

interface ConfigResponse {
  groups: ConfigGroup[];
  total_groups: number;
  is_local_dev: boolean;
}

type ExportFormat = 'env' | 'json' | 'tfvars' | 'yaml';

interface ConfigPanelProps {
  onError?: (error: string) => void;
  showToast?: (message: string, type: 'success' | 'error') => void;
}

/* ------------------------------------------------------------------ */
/*  Helpers                                                            */
/* ------------------------------------------------------------------ */

const EXPORT_OPTIONS: { format: ExportFormat; label: string }[] = [
  { format: 'env', label: '.env' },
  { format: 'json', label: 'JSON' },
  { format: 'tfvars', label: 'Terraform (.tfvars)' },
  { format: 'yaml', label: 'YAML' },
];

const DEFAULT_EXPANDED: Set<string> = new Set(['deployment', 'storage']);

/**
 * Highlight occurrences of `term` inside `text` using <mark> tags.
 */
function highlightMatch(text: string, term: string): React.ReactNode {
  if (!term) return text;
  const idx = text.toLowerCase().indexOf(term.toLowerCase());
  if (idx === -1) return text;
  return (
    <>
      {text.slice(0, idx)}
      <mark className="bg-yellow-200 dark:bg-yellow-700 rounded px-0.5">{text.slice(idx, idx + term.length)}</mark>
      {text.slice(idx + term.length)}
    </>
  );
}

/* ------------------------------------------------------------------ */
/*  ConfigGroupPanel sub-component                                     */
/* ------------------------------------------------------------------ */

interface ConfigGroupPanelProps {
  group: ConfigGroup;
  expanded: boolean;
  onToggle: () => void;
  searchTerm: string;
  copiedKey: string | null;
  onCopy: (key: string, value: string) => void;
}

/**
 * Render a single field row (reused in top-level fields and subgroups).
 */
const FieldRow: React.FC<{
  field: ConfigField;
  searchTerm: string;
  copiedKey: string | null;
  onCopy: (key: string, value: string) => void;
}> = ({ field, searchTerm, copiedKey, onCopy }) => (
  <div
    key={field.key}
    className="flex items-center justify-between px-4 py-2.5 hover:bg-gray-50 dark:hover:bg-gray-800/50"
  >
    <div className="flex-1 min-w-0 mr-4">
      <div className="text-xs text-gray-500 dark:text-gray-400 font-mono truncate">
        {highlightMatch(field.key, searchTerm)}
      </div>
      <div className="text-sm text-gray-900 dark:text-white">
        {highlightMatch(field.label, searchTerm)}
      </div>
    </div>
    <div className="flex items-center space-x-2 flex-shrink-0">
      <span
        className={`text-sm font-mono ${
          field.is_masked
            ? 'text-gray-400 dark:text-gray-500 italic'
            : 'text-gray-700 dark:text-gray-300'
        }`}
      >
        {highlightMatch(field.value, searchTerm)}
        {field.unit && !field.is_masked && (
          <span className="text-xs text-gray-400 dark:text-gray-500 ml-1">
            {field.unit}
          </span>
        )}
      </span>
      {!field.is_masked && field.raw_value !== null && (
        <button
          onClick={() => onCopy(field.key, String(field.raw_value))}
          className="p-1 rounded hover:bg-gray-200 dark:hover:bg-gray-700 transition-colors"
          aria-label={`Copy ${field.label} value`}
          title="Copy value"
        >
          {copiedKey === field.key ? (
            <CheckIcon className="h-4 w-4 text-green-500" />
          ) : (
            <ClipboardIcon className="h-4 w-4 text-gray-400 dark:text-gray-500" />
          )}
        </button>
      )}
    </div>
  </div>
);

const ConfigGroupPanel: React.FC<ConfigGroupPanelProps> = ({
  group,
  expanded,
  onToggle,
  searchTerm,
  copiedKey,
  onCopy,
}) => {
  const panelId = `config-group-${group.id}`;
  const totalFields = group.fields.length +
    (group.subgroups?.reduce((s, sg) => s + sg.fields.length, 0) || 0);

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
      {/* Group header */}
      <button
        onClick={onToggle}
        aria-expanded={expanded}
        aria-controls={panelId}
        className="w-full flex items-center justify-between px-4 py-3
                   bg-gray-50 dark:bg-gray-900/50 hover:bg-gray-100 dark:hover:bg-gray-700/50
                   transition-colors text-left"
      >
        <div className="flex items-center space-x-2">
          {expanded ? (
            <ChevronDownIcon className="h-4 w-4 text-gray-500 dark:text-gray-400" />
          ) : (
            <ChevronRightIcon className="h-4 w-4 text-gray-500 dark:text-gray-400" />
          )}
          <span className="text-sm font-medium text-gray-900 dark:text-white">
            {highlightMatch(group.title, searchTerm)}
          </span>
        </div>
        <div className="flex items-center space-x-2">
          {group.subgroups && group.subgroups.length > 0 && (
            <span className="text-xs text-gray-500 dark:text-gray-400 bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded-full">
              {group.subgroups.length} {group.subgroups.length === 1 ? 'provider' : 'providers'}
            </span>
          )}
          <span className="text-xs text-gray-500 dark:text-gray-400 bg-gray-200 dark:bg-gray-700 px-2 py-0.5 rounded-full">
            {totalFields} {totalFields === 1 ? 'field' : 'fields'}
          </span>
        </div>
      </button>

      {/* Group fields and subgroups */}
      {expanded && (
        <div id={panelId} role="region">
          {/* Top-level fields */}
          {group.fields.length > 0 && (
            <div className="divide-y divide-gray-100 dark:divide-gray-700/50">
              {group.fields.map((field) => (
                <FieldRow
                  key={field.key}
                  field={field}
                  searchTerm={searchTerm}
                  copiedKey={copiedKey}
                  onCopy={onCopy}
                />
              ))}
            </div>
          )}

          {/* Subgroups */}
          {group.subgroups?.map((sg) => (
            <div key={sg.id}>
              <div className="px-4 py-2 bg-gray-100/50 dark:bg-gray-800/50 border-t border-gray-200 dark:border-gray-700">
                <span className="text-xs font-semibold text-gray-600 dark:text-gray-300 uppercase tracking-wider">
                  {highlightMatch(sg.title, searchTerm)}
                </span>
                <span className="text-xs text-gray-400 dark:text-gray-500 ml-2">
                  {sg.fields.length} {sg.fields.length === 1 ? 'field' : 'fields'}
                </span>
              </div>
              <div className="divide-y divide-gray-100 dark:divide-gray-700/50">
                {sg.fields.map((field) => (
                  <FieldRow
                    key={field.key}
                    field={field}
                    searchTerm={searchTerm}
                    copiedKey={copiedKey}
                    onCopy={onCopy}
                  />
                ))}
              </div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

/* ------------------------------------------------------------------ */
/*  ConfigPanel main component                                         */
/* ------------------------------------------------------------------ */

const ConfigPanel: React.FC<ConfigPanelProps> = ({ onError, showToast }) => {
  const [config, setConfig] = useState<ConfigResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [expandedGroups, setExpandedGroups] = useState<Set<string>>(new Set(DEFAULT_EXPANDED));
  const [searchTerm, setSearchTerm] = useState('');
  const [copiedKey, setCopiedKey] = useState<string | null>(null);
  const [exportOpen, setExportOpen] = useState(false);

  /* ---- Data fetching ---- */

  const fetchConfig = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const res = await axios.get<ConfigResponse>('/api/config/full');
      setConfig(res.data);
    } catch (err: any) {
      const msg = err.response?.data?.detail || 'Failed to load configuration';
      setError(msg);
      onError?.(msg);
    } finally {
      setLoading(false);
    }
  }, [onError]);

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  /* ---- Filtering ---- */

  const filteredGroups = useMemo(() => {
    if (!config) return [];
    if (!searchTerm.trim()) return config.groups;

    const term = searchTerm.toLowerCase();
    const matchField = (f: ConfigField) =>
      f.key.toLowerCase().includes(term) ||
      f.label.toLowerCase().includes(term) ||
      f.value.toLowerCase().includes(term);

    return config.groups
      .map((group) => ({
        ...group,
        fields: group.fields.filter(matchField),
        subgroups: group.subgroups?.map((sg) => ({
          ...sg,
          fields: sg.fields.filter(matchField),
        })).filter((sg) => sg.fields.length > 0),
      }))
      .filter((group) => group.fields.length > 0 || (group.subgroups && group.subgroups.length > 0));
  }, [config, searchTerm]);

  const totalMatchingFields = useMemo(
    () => filteredGroups.reduce((sum, g) => {
      const sgFields = g.subgroups?.reduce((s, sg) => s + sg.fields.length, 0) || 0;
      return sum + g.fields.length + sgFields;
    }, 0),
    [filteredGroups]
  );

  /* ---- Group expand/collapse ---- */

  const toggleGroup = useCallback((groupId: string) => {
    setExpandedGroups((prev) => {
      const next = new Set(prev);
      if (next.has(groupId)) next.delete(groupId);
      else next.add(groupId);
      return next;
    });
  }, []);

  const expandAll = useCallback(() => {
    if (!config) return;
    setExpandedGroups(new Set(config.groups.map((g) => g.id)));
  }, [config]);

  const collapseAll = useCallback(() => {
    setExpandedGroups(new Set());
  }, []);

  /* ---- Clipboard ---- */

  const copyToClipboard = useCallback(
    async (key: string, value: string) => {
      try {
        await navigator.clipboard.writeText(value);
        setCopiedKey(key);
        showToast?.('Copied to clipboard', 'success');
        setTimeout(() => setCopiedKey(null), 2000);
      } catch {
        showToast?.('Failed to copy', 'error');
      }
    },
    [showToast]
  );

  /* ---- Export ---- */

  const handleExport = useCallback(
    async (format: ExportFormat) => {
      setExportOpen(false);
      try {
        const res = await axios.get(`/api/config/export`, {
          params: { format },
          responseType: 'blob',
        });

        const disposition = res.headers['content-disposition'];
        let filename = `mcp-registry-config.${format}`;
        if (disposition) {
          const match = disposition.match(/filename="?([^"]+)"?/);
          if (match) filename = match[1];
        }

        const url = window.URL.createObjectURL(new Blob([res.data]));
        const link = document.createElement('a');
        link.href = url;
        link.setAttribute('download', filename);
        document.body.appendChild(link);
        link.click();
        link.remove();
        window.URL.revokeObjectURL(url);
      } catch (err: any) {
        const msg = err.response?.data?.detail || 'Export failed';
        showToast?.(msg, 'error');
      }
    },
    [showToast]
  );

  /* ---- Skeleton loading ---- */

  if (loading) {
    return (
      <div className="space-y-4" data-testid="config-skeleton">
        <div className="flex items-center justify-between">
          <div className="h-7 w-56 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
          <div className="flex space-x-2">
            <div className="h-9 w-24 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
            <div className="h-9 w-9 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
          </div>
        </div>
        <div className="h-10 w-full bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
        {[1, 2, 3, 4].map((i) => (
          <div key={i} className="h-14 bg-gray-200 dark:bg-gray-700 rounded animate-pulse" />
        ))}
      </div>
    );
  }

  /* ---- Error state ---- */

  if (error) {
    return (
      <div className="text-center py-12" data-testid="config-error">
        <ExclamationCircleIcon className="h-12 w-12 mx-auto text-red-500 mb-4" />
        <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-2">
          Failed to Load Configuration
        </h3>
        <p className="text-gray-500 dark:text-gray-400 mb-4">{error}</p>
        <button
          onClick={fetchConfig}
          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
        >
          Retry
        </button>
      </div>
    );
  }

  if (!config) return null;

  /* ---- Main render ---- */

  return (
    <div className="space-y-4">
      {/* Header row */}
      <div className="flex flex-wrap items-center justify-between gap-2">
        <div className="flex items-center space-x-3">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            System Configuration
          </h2>
          {config.is_local_dev && (
            <span
              className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium
                         bg-yellow-100 text-yellow-800 dark:bg-yellow-900/40 dark:text-yellow-300"
              data-testid="local-dev-badge"
            >
              Local Development Mode
            </span>
          )}
        </div>

        <div className="flex items-center space-x-2">
          {/* Export dropdown */}
          <div className="relative">
            <button
              onClick={() => setExportOpen((o) => !o)}
              className="flex items-center px-3 py-2 text-sm border border-gray-300 dark:border-gray-600
                         rounded-lg bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200
                         hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
              aria-label="Export configuration"
            >
              <ArrowDownTrayIcon className="h-4 w-4 mr-1.5" />
              Export
            </button>
            {exportOpen && (
              <div className="absolute right-0 mt-1 w-48 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-lg z-10">
                {EXPORT_OPTIONS.map((opt) => (
                  <button
                    key={opt.format}
                    onClick={() => handleExport(opt.format)}
                    className="w-full text-left px-4 py-2 text-sm text-gray-700 dark:text-gray-200
                               hover:bg-gray-100 dark:hover:bg-gray-700 first:rounded-t-lg last:rounded-b-lg"
                  >
                    {opt.label}
                  </button>
                ))}
              </div>
            )}
          </div>

          {/* Expand / Collapse */}
          <button
            onClick={expandAll}
            className="px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg
                       bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200
                       hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            Expand All
          </button>
          <button
            onClick={collapseAll}
            className="px-3 py-2 text-sm border border-gray-300 dark:border-gray-600 rounded-lg
                       bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200
                       hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
          >
            Collapse All
          </button>

          {/* Refresh */}
          <button
            onClick={fetchConfig}
            className="p-2 border border-gray-300 dark:border-gray-600 rounded-lg
                       bg-white dark:bg-gray-800 text-gray-700 dark:text-gray-200
                       hover:bg-gray-50 dark:hover:bg-gray-700 transition-colors"
            aria-label="Refresh configuration"
            title="Refresh"
          >
            <ArrowPathIcon className="h-4 w-4" />
          </button>
        </div>
      </div>

      {/* Search */}
      <div className="relative">
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-5 w-5 text-gray-400" />
        <input
          type="text"
          value={searchTerm}
          onChange={(e) => setSearchTerm(e.target.value)}
          placeholder="Search configuration..."
          aria-label="Search configuration"
          className="w-full pl-10 pr-10 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                     bg-white dark:bg-gray-800 text-gray-900 dark:text-white
                     focus:ring-2 focus:ring-purple-500 focus:border-transparent"
        />
        {searchTerm && (
          <button
            onClick={() => setSearchTerm('')}
            className="absolute right-3 top-1/2 -translate-y-1/2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
            aria-label="Clear search"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        )}
      </div>

      {/* Search results count */}
      {searchTerm.trim() && filteredGroups.length > 0 && (
        <p className="text-sm text-gray-500 dark:text-gray-400" data-testid="search-count">
          {totalMatchingFields} {totalMatchingFields === 1 ? 'field' : 'fields'} in{' '}
          {filteredGroups.length} {filteredGroups.length === 1 ? 'group' : 'groups'}
        </p>
      )}

      {/* No results */}
      {searchTerm.trim() && filteredGroups.length === 0 && (
        <div className="text-center py-8" data-testid="no-results">
          <p className="text-gray-500 dark:text-gray-400">
            No configuration fields match "<span className="font-medium">{searchTerm}</span>"
          </p>
        </div>
      )}

      {/* Config groups */}
      <div className="space-y-3">
        {filteredGroups.map((group) => (
          <ConfigGroupPanel
            key={group.id}
            group={group}
            expanded={expandedGroups.has(group.id)}
            onToggle={() => toggleGroup(group.id)}
            searchTerm={searchTerm}
            copiedKey={copiedKey}
            onCopy={copyToClipboard}
          />
        ))}
      </div>

      {/* Legend */}
      <div className="flex items-center space-x-4 text-xs text-gray-400 dark:text-gray-500 pt-2 border-t border-gray-200 dark:border-gray-700">
        <span>
          <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">****</code> = masked sensitive value
        </span>
        <span>
          <code className="bg-gray-100 dark:bg-gray-800 px-1 rounded">(not set)</code> = not configured
        </span>
      </div>
    </div>
  );
};

export default ConfigPanel;
