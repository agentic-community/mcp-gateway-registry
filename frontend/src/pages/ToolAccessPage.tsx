import React, { useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  ArrowLeftIcon,
  ExclamationTriangleIcon,
  MagnifyingGlassIcon,
  ShieldCheckIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';

import Pagination from '../components/Pagination';
import {
  useToolAccess,
  type ToolAccessGroup,
  type ToolAccessQuery,
  type ToolAccessRow,
  type ToolAccessServerType,
  type ToolAccessSummary,
} from '../hooks/useToolAccess';

export const PAGE_SIZE = 20;
// Short enough to feel like filtering as you type; the server caches the
// listing, so each keystroke's request is cheap.
const SEARCH_DEBOUNCE_MS = 150;
// Chips shown per cell before "+N more".
const VISIBLE_CHIPS = 3;
const SERVER_TYPE_LABELS: Record<ToolAccessServerType, string> = {
  mcp: 'MCP',
  openapi: 'OpenAPI',
  virtual: 'Virtual',
};

type GroupLookup = Record<string, ToolAccessGroup>;

/** Read the listing's query from the URL so filters survive reloads and the back button. */
function queryFromParams(params: URLSearchParams): ToolAccessQuery {
  const page = Math.max(0, Number(params.get('page') || '0') || 0);
  const type = params.get('server_type') || '';
  return {
    q: params.get('q') || '',
    serverPath: params.get('server_path') || '',
    serverType: (['mcp', 'openapi', 'virtual'].includes(type) ? type : '') as ToolAccessQuery['serverType'],
    group: params.get('group') || '',
    includeDisabled: params.get('include_disabled') !== 'false',
    limit: PAGE_SIZE,
    offset: page * PAGE_SIZE,
  };
}

/** Tooltip text: the identity provider groups mapped to a registry group. */
function chipTitle(name: string, groups: GroupLookup): string {
  const idp = (groups[name]?.idp_groups || []).filter((g) => g !== name);
  return idp.length > 0 ? `${name}\nIdP groups: ${idp.join(', ')}` : name;
}

const GroupChip: React.FC<{ name: string; groups: GroupLookup; onPick: (name: string) => void }> = ({
  name,
  groups,
  onPick,
}) => (
  <button
    type="button"
    onClick={() => onPick(name)}
    title={chipTitle(name, groups)}
    className="max-w-[11rem] truncate whitespace-nowrap rounded-full border border-line px-2 py-0.5 text-xs text-fg hover:bg-surface-hover"
  >
    {name}
  </button>
);

/** One line of chips: non-admin groups first, admin groups as one "Admins" chip, extras behind "+N". */
const GroupChips: React.FC<{ names: string[]; groups: GroupLookup; onPick: (name: string) => void }> = ({
  names,
  groups,
  onPick,
}) => {
  const [expanded, setExpanded] = useState(false);
  const admins = names.filter((n) => groups[n]?.is_admin);
  const others = names.filter((n) => !groups[n]?.is_admin);
  const shown = expanded ? others : others.slice(0, VISIBLE_CHIPS);
  const hidden = others.length - shown.length;
  return (
    <div className="flex flex-wrap items-center gap-1">
      {shown.map((name) => (
        <GroupChip key={name} name={name} groups={groups} onPick={onPick} />
      ))}
      {hidden > 0 && (
        <button type="button" onClick={() => setExpanded(true)} className="text-xs text-accent-text hover:underline">
          +{hidden} more
        </button>
      )}
      {admins.length > 0 && (
        <span
          className="whitespace-nowrap rounded-full border border-dashed border-line px-2 py-0.5 text-xs text-fg-subtle"
          title={`Admin groups: ${admins.join(', ')}`}
        >
          Admins
        </span>
      )}
    </div>
  );
};

/** Groups that grant access alone, with the combinations behind a toggle. */
export const AccessCell: React.FC<{
  summary: ToolAccessSummary;
  groups: GroupLookup;
  onPick: (name: string) => void;
}> = ({ summary, groups, onPick }) => {
  const [showTogether, setShowTogether] = useState(false);
  const hasTogether = summary.together.length > 0;
  if (summary.alone.length === 0 && !hasTogether) {
    return <span className="text-xs text-fg-subtle">No group</span>;
  }
  return (
    <div className="space-y-1">
      {summary.alone.length > 0 ? (
        <GroupChips names={summary.alone} groups={groups} onPick={onPick} />
      ) : (
        <span className="text-xs text-fg-subtle">No single group</span>
      )}
      {hasTogether && (
        <button
          type="button"
          onClick={() => setShowTogether(!showTogether)}
          aria-expanded={showTogether}
          className="text-xs text-accent-text hover:underline"
        >
          {showTogether ? 'Hide combined groups' : 'Combined groups'}
        </button>
      )}
      {hasTogether && showTogether && (
        <div className="space-y-1 rounded-md bg-surface-hover/50 p-2">
          <p className="text-xs text-fg-muted">A member of one group from each line has access:</p>
          {summary.together.map((condition) => (
            <div key={condition.label}>
              <p className="text-[11px] text-fg-subtle">{condition.label}</p>
              <GroupChips names={condition.groups} groups={groups} onPick={onPick} />
            </div>
          ))}
        </div>
      )}
    </div>
  );
};

const ToolCell: React.FC<{ row: ToolAccessRow }> = ({ row }) => (
  <div className="min-w-0">
    <div className="flex items-center gap-1.5">
      <code className="truncate text-sm text-fg" title={row.tool_name}>{row.tool_name}</code>
      {row.blocked && (
        <span
          className="whitespace-nowrap rounded bg-danger-50 px-1.5 text-[11px] text-danger-700 dark:bg-danger-900/30 dark:text-danger-300"
          title={row.blocked_source === 'security_scan' ? 'Blocked by a security scan' : 'Blocked by an admin'}
        >
          Blocked
        </span>
      )}
    </div>
    {row.backend_tool_name && row.backend_tool_name !== row.tool_name && (
      <p className="truncate text-xs text-fg-subtle">backend: {row.backend_tool_name}</p>
    )}
  </div>
);

const ServerCell: React.FC<{ row: ToolAccessRow; onPick: (path: string) => void }> = ({ row, onPick }) => (
  <div className="min-w-0">
    <button
      type="button"
      onClick={() => onPick(row.server_path)}
      title={row.server_path}
      className="block max-w-full truncate text-left text-sm text-fg hover:underline"
    >
      {row.server_name}
    </button>
    <p className="text-xs text-fg-subtle">
      {SERVER_TYPE_LABELS[row.server_type]}
      {!row.enabled && ' · disabled'}
    </p>
  </div>
);

const FilterChip: React.FC<{ label: string; onClear: () => void }> = ({ label, onClear }) => (
  <span className="inline-flex items-center gap-1 rounded-full bg-accent-soft px-2.5 py-1 text-xs text-accent-soft-fg">
    {label}
    <button type="button" onClick={onClear} aria-label={`Clear ${label}`}>
      <XMarkIcon className="h-3 w-3" />
    </button>
  </span>
);

const ToolAccessPage: React.FC = () => {
  const navigate = useNavigate();
  const [params, setParams] = useSearchParams();
  const query = queryFromParams(params);
  const [searchText, setSearchText] = useState(query.q);
  const { data, loading, error } = useToolAccess(query);

  const update = (changes: Record<string, string>) => {
    const next = new URLSearchParams(params);
    Object.entries(changes).forEach(([key, value]) => (value ? next.set(key, value) : next.delete(key)));
    if (!('page' in changes)) next.delete('page');
    setParams(next, { replace: 'q' in changes });
  };

  useEffect(() => {
    if (searchText === query.q) return undefined;
    const timer = setTimeout(() => update({ q: searchText }), SEARCH_DEBOUNCE_MS);
    return () => clearTimeout(timer);
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [searchText]);

  const total = data?.total_count ?? 0;
  const groups = data?.groups ?? {};
  const page = Math.floor(query.offset / PAGE_SIZE);
  const pickGroup = (group: string) => update({ group });

  return (
    <div className="mx-auto max-w-6xl p-6">
      <button
        type="button"
        onClick={() => navigate('/')}
        className="mb-4 flex items-center space-x-1 text-sm text-fg-muted hover:text-fg"
      >
        <ArrowLeftIcon className="h-4 w-4" />
        <span>Back to Dashboard</span>
      </button>
      <div className="mb-1 flex items-center gap-2">
        <ShieldCheckIcon className="h-6 w-6 text-accent-text" />
        <h1 className="text-2xl font-semibold text-fg">Tool access</h1>
      </div>
      <p className="mb-5 text-sm text-fg-muted">Which groups can see and call each tool. Hover a group for its IdP groups.</p>

      <div className="mb-3 flex flex-wrap items-center gap-3">
        <div className="relative">
          <MagnifyingGlassIcon className="pointer-events-none absolute left-2.5 top-1/2 h-4 w-4 -translate-y-1/2 text-fg-subtle" />
          <input
            type="search"
            value={searchText}
            onChange={(e) => setSearchText(e.target.value)}
            placeholder="Filter by tool, server or group"
            aria-label="Filter by tool, server or group"
            className="input w-72 pl-8"
          />
        </div>
        <select
          value={query.serverType}
          onChange={(e) => update({ server_type: e.target.value })}
          aria-label="Server type"
          className="input w-40"
        >
          <option value="">All server types</option>
          <option value="mcp">MCP</option>
          <option value="openapi">OpenAPI</option>
          <option value="virtual">Virtual</option>
        </select>
        <label className="flex items-center gap-2 text-sm text-fg-muted">
          <input
            type="checkbox"
            checked={query.includeDisabled}
            onChange={(e) => update({ include_disabled: e.target.checked ? '' : 'false' })}
          />
          Include disabled
        </label>
        {query.serverPath && <FilterChip label={`Server ${query.serverPath}`} onClear={() => update({ server_path: '' })} />}
        {query.group && <FilterChip label={`Group ${query.group}`} onClear={() => update({ group: '' })} />}
        <span className="ml-auto text-xs text-fg-subtle" aria-live="polite">
          {loading ? 'Loading...' : data ? `${total} tools` : ''}
        </span>
      </div>

      {error && (
        <div className="mb-4 flex items-center space-x-2 rounded-lg bg-danger-50 p-3 text-danger-700 dark:bg-danger-900/30 dark:text-danger-300">
          <ExclamationTriangleIcon className="h-5 w-5 flex-shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {!error && data && total === 0 && (
        <div className="card p-6 text-sm text-fg-muted">
          No tools to show. This page lists the tools on servers you own, or every tool if you are an admin or a
          read-only admin.
        </div>
      )}

      {total > 0 && data && (
        <div className={`card overflow-x-auto ${loading ? 'opacity-70' : ''}`}>
          <table className="w-full table-fixed text-left">
            <thead className="border-b border-line text-xs uppercase tracking-wide text-fg-subtle">
              <tr>
                <th className="w-[24%] px-3 py-2">Tool</th>
                <th className="w-[18%] px-3 py-2">MCP server</th>
                <th className="w-[29%] px-3 py-2">Can see</th>
                <th className="w-[29%] px-3 py-2">Can call</th>
              </tr>
            </thead>
            <tbody>
              {data.rows.map((row) => (
                <tr key={`${row.server_path}|${row.tool_name}`} className="border-b border-line align-top last:border-0">
                  <td className="px-3 py-2"><ToolCell row={row} /></td>
                  <td className="px-3 py-2">
                    <ServerCell row={row} onPick={(path) => update({ server_path: path })} />
                  </td>
                  <td className="px-3 py-2">
                    <AccessCell summary={row.can_see} groups={groups} onPick={pickGroup} />
                  </td>
                  <td className={`px-3 py-2 ${row.blocked ? 'opacity-50' : ''}`}>
                    <AccessCell summary={row.can_call} groups={groups} onPick={pickGroup} />
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      <div className="mt-3 flex items-center justify-between">
        <span className="text-xs text-fg-subtle">
          {data ? `As of ${new Date(data.generated_at).toLocaleTimeString()}` : ''}
        </span>
        <Pagination
          currentPage={page}
          totalPages={Math.ceil(total / PAGE_SIZE)}
          totalItems={total}
          pageSize={PAGE_SIZE}
          onPageChange={(next) => update({ page: next > 0 ? String(next) : '' })}
        />
      </div>
    </div>
  );
};

export default ToolAccessPage;
