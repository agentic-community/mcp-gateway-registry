import React, { useState, useMemo, useCallback } from 'react';
import {
  PlusIcon,
  MagnifyingGlassIcon,
  TrashIcon,
  ArrowLeftIcon,
  PencilIcon,
} from '@heroicons/react/24/outline';
import {
  useRateLimitDefinitions,
  useRateLimitMemberships,
  setRateLimitDefinition,
  deleteRateLimitDefinition,
  setRateLimitEnabled,
  definitionId,
  RateLimitDefinition,
  RateLimitAxis,
  AXIS_LABELS,
  TARGET_ENTITY_TYPES,
  TARGET_ENTITY_TYPE_LABELS,
  SERVER_GROUP_ENTITY_TYPE,
} from '../hooks/useRateLimits';
import DeleteConfirmation from './DeleteConfirmation';
import ListStateBoundary from './iam/ListStateBoundary';
import SearchableSelect from './SearchableSelect';
import QuarantinePanel from './iam/QuarantinePanel';
import { useAuth } from '../contexts/AuthContext';
import { useServerList } from '../hooks/useToolCatalog';
import { useAgentList } from '../hooks/useAgentList';

// Group axes (caller + caller_target) share the same "group with per-caller-type
// limits" form shape. The quarantine axis is not creatable here (seeded only).
const GROUP_AXES: RateLimitAxis[] = ['caller', 'caller_target'];

interface IAMRateLimitsProps {
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
}

type View = 'list' | 'form';

// Extract a human-readable API error.
function apiError(err: any, fallback: string): string {
  return err?.response?.data?.detail || err?.message || fallback;
}

const inputBase =
  'w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-800 text-gray-900 ' +
  'dark:text-white border-gray-300 dark:border-gray-600 focus:ring-2 focus:ring-blue-500';

// Small at-a-glance stat tile for the summary row.
const StatTile: React.FC<{ label: string; value: number; hint?: string }> = ({
  label,
  value,
  hint,
}) => (
  <div className="rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 px-4 py-3">
    <div className="text-2xl font-semibold text-gray-900 dark:text-white">{value}</div>
    <div className="text-xs text-gray-500 dark:text-gray-400" title={hint}>{label}</div>
  </div>
);

const IAMRateLimits: React.FC<IAMRateLimitsProps> = ({ onShowToast }) => {
  const { user: currentUser } = useAuth();
  const isAdmin = currentUser?.is_admin ?? false;
  const { definitions, isLoading, error, refetch } = useRateLimitDefinitions();
  // Memberships feed the "callers governed" summary tile (users + agents mapped
  // into rate-limit groups). Read-only here; the editor lives on Users/M2M.
  const { memberships } = useRateLimitMemberships();
  // Registered server + agent paths, for the target-name typeahead.
  const { servers, isLoading: serversLoading } = useServerList();
  const { agents, isLoading: agentsLoading } = useAgentList();
  const [searchQuery, setSearchQuery] = useState('');
  const [view, setView] = useState<View>('list');

  // Form state. axis drives which limit fields show.
  const [editingId, setEditingId] = useState<string | null>(null); // non-null => edit mode
  const [formAxis, setFormAxis] = useState<RateLimitAxis>('caller');
  const [formEntityType, setFormEntityType] = useState('group');
  const [formName, setFormName] = useState('');
  const [formWindow, setFormWindow] = useState(60);
  const [formUserMax, setFormUserMax] = useState('');
  const [formAgentMax, setFormAgentMax] = useState('');
  const [formTargetMax, setFormTargetMax] = useState('');
  // server_group members: the server paths this group covers (each limited
  // individually to formTargetMax). Only used when isServerGroup.
  const [formMembers, setFormMembers] = useState<string[]>([]);
  const [formFailClosed, setFormFailClosed] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // A server_group is a target-axis definition naming a set of servers; each
  // member gets its own bucket. Drives the member multi-select in the form.
  const isServerGroup =
    formAxis === 'target' && formEntityType === SERVER_GROUP_ENTITY_TYPE;

  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  const filtered = useMemo(() => {
    if (!searchQuery) return definitions;
    const q = searchQuery.toLowerCase();
    return definitions.filter(
      (d) => d.name.toLowerCase().includes(q) || d.entity_type.toLowerCase().includes(q),
    );
  }, [definitions, searchQuery]);

  // High-level, at-a-glance config summary computed from the loaded definitions +
  // memberships (no backend call). These are configuration counts, not live
  // throttle counts (runtime throttle metrics live in Prometheus:
  // mcpgw_rate_limit_throttled_total). See the Observability guide.
  const summary = useMemo(() => {
    const enabled = definitions.filter((d) => d.enabled);
    const callerSubjects = new Set<string>();
    for (const m of memberships) {
      if (m.groups && m.groups.length > 0) callerSubjects.add(`${m.subject_type}:${m.subject}`);
    }
    return {
      total: definitions.length,
      enabled: enabled.length,
      caller: definitions.filter((d) => d.axis === 'caller').length,
      callerTarget: definitions.filter((d) => d.axis === 'caller_target').length,
      target: definitions.filter((d) => d.axis === 'target').length,
      failClosed: definitions.filter((d) => d.fail_closed).length,
      governed: callerSubjects.size,
    };
  }, [definitions, memberships]);

  // Typeahead options for the target name: registered server paths for
  // mcp_server, agent paths for a2a_agent. name shown as the label's description.
  const targetOptions = useMemo(() => {
    if (formEntityType === 'a2a_agent') {
      return agents.map((a) => ({ value: a.path, label: a.path, description: a.name }));
    }
    return servers.map((s) => ({ value: s.path, label: s.path, description: s.name }));
  }, [formEntityType, servers, agents]);
  const targetOptionsLoading = formEntityType === 'a2a_agent' ? agentsLoading : serversLoading;

  const resetForm = useCallback(() => {
    setEditingId(null);
    setFormAxis('caller');
    setFormEntityType('group');
    setFormName('');
    setFormWindow(60);
    setFormUserMax('');
    setFormAgentMax('');
    setFormTargetMax('');
    setFormMembers([]);
    setFormFailClosed(false);
  }, []);

  const openCreate = () => {
    resetForm();
    setView('form');
  };

  const openEdit = (d: RateLimitDefinition) => {
    setEditingId(definitionId(d));
    setFormAxis(d.axis);
    setFormEntityType(d.entity_type);
    setFormName(d.name);
    setFormWindow(d.window_seconds);
    setFormUserMax(d.user_max_requests != null ? String(d.user_max_requests) : '');
    setFormAgentMax(d.agent_max_requests != null ? String(d.agent_max_requests) : '');
    setFormTargetMax(d.max_requests != null ? String(d.max_requests) : '');
    setFormMembers(d.members ?? []);
    setFormFailClosed(d.fail_closed);
    setView('form');
  };

  const handleSave = async () => {
    if (!formName.trim()) {
      onShowToast('Name is required', 'error');
      return;
    }
    const isGroup = GROUP_AXES.includes(formAxis);
    if (isServerGroup && formMembers.length === 0) {
      onShowToast('Add at least one server to the group', 'error');
      return;
    }
    const def: RateLimitDefinition = {
      axis: formAxis,
      entity_type: isGroup ? 'group' : formEntityType,
      name: formName.trim(),
      window_seconds: formWindow,
      fail_closed: formFailClosed,
      enabled: true,
      max_requests: formAxis === 'target' && formTargetMax ? Number(formTargetMax) : null,
      user_max_requests: isGroup && formUserMax ? Number(formUserMax) : null,
      agent_max_requests: isGroup && formAgentMax ? Number(formAgentMax) : null,
      members: isServerGroup ? formMembers : null,
    };
    setIsSaving(true);
    try {
      await setRateLimitDefinition(def);
      onShowToast(`Rate limit ${editingId ? 'updated' : 'created'}`, 'success');
      resetForm();
      setView('list');
      refetch();
    } catch (err) {
      onShowToast(apiError(err, 'Failed to save rate limit'), 'error');
    } finally {
      setIsSaving(false);
    }
  };

  const handleToggle = async (d: RateLimitDefinition) => {
    try {
      await setRateLimitEnabled(definitionId(d), !d.enabled);
      onShowToast(`Rate limit ${d.enabled ? 'disabled' : 'enabled'}`, 'success');
      refetch();
    } catch (err) {
      onShowToast(apiError(err, 'Failed to toggle rate limit'), 'error');
    }
  };

  // DeleteConfirmation passes the entityPath and closes itself (via onCancel) on
  // success; we refetch + toast here and let a throw surface its error inline.
  const handleDelete = async (id: string) => {
    await deleteRateLimitDefinition(id);
    onShowToast('Rate limit deleted', 'success');
    refetch();
  };

  // Compact "limit per window" summary for a row.
  const limitSummary = (d: RateLimitDefinition): string => {
    const w = `${d.window_seconds}s`;
    if (d.axis === 'quarantine') {
      return 'kill switch (drops all traffic)';
    }
    if (GROUP_AXES.includes(d.axis)) {
      const parts: string[] = [];
      if (d.user_max_requests != null) parts.push(`user ${d.user_max_requests}`);
      if (d.agent_max_requests != null) parts.push(`agent ${d.agent_max_requests}`);
      return `${parts.join(', ')} / ${w}`;
    }
    if (d.entity_type === SERVER_GROUP_ENTITY_TYPE) {
      const n = d.members?.length ?? 0;
      return `${d.max_requests} / ${w} · each of ${n} server${n === 1 ? '' : 's'}`;
    }
    return `${d.max_requests} / ${w}`;
  };

  if (view === 'form') {
    const isGroup = GROUP_AXES.includes(formAxis);
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            IAM &gt; Rate Limits &gt; {editingId ? 'Edit' : 'Create'}
          </h2>
          <button
            onClick={() => { resetForm(); setView('list'); }}
            className="flex items-center text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
          >
            <ArrowLeftIcon className="h-4 w-4 mr-1" />
            Back to List
          </button>
        </div>

        <div className="space-y-4 max-w-lg">
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Axis</label>
            <select
              value={formAxis}
              onChange={(e) => {
                const axis = e.target.value as RateLimitAxis;
                setFormAxis(axis);
                // Reset entity_type to a valid value for the chosen axis so a
                // target definition never submits the group's 'group' type.
                setFormEntityType(GROUP_AXES.includes(axis) ? 'group' : TARGET_ENTITY_TYPES[0]);
                setFormName('');
              }}
              disabled={!!editingId}
              className={inputBase}
            >
              <option value="caller">{AXIS_LABELS.caller} (a group of users/agents, across all targets)</option>
              <option value="caller_target">{AXIS_LABELS.caller_target} (each member gets an independent quota per target)</option>
              <option value="target">{AXIS_LABELS.target} (an MCP server, A2A agent, or server group, across all callers)</option>
            </select>
          </div>

          {!isGroup && (
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Target type</label>
              <select
                value={formEntityType}
                onChange={(e) => setFormEntityType(e.target.value)}
                disabled={!!editingId}
                className={inputBase}
              >
                {TARGET_ENTITY_TYPES.map((t) => (
                  <option key={t} value={t}>{TARGET_ENTITY_TYPE_LABELS[t] ?? t}</option>
                ))}
              </select>
            </div>
          )}

          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">
              {isGroup || isServerGroup
                ? 'Group name'
                : 'Target name (server path / agent path)'}
            </label>
            {isGroup || isServerGroup ? (
              // Group name is an arbitrary label (not a registered path).
              <input
                type="text"
                value={formName}
                onChange={(e) => setFormName(e.target.value)}
                disabled={!!editingId}
                placeholder={isServerGroup ? 'e.g. fragile-backends' : 'e.g. rate-limited-testers'}
                className={inputBase}
              />
            ) : editingId ? (
              // On edit the name is immutable (part of the id); show it read-only.
              <input type="text" value={formName} disabled className={inputBase} />
            ) : (
              // Typeahead of registered server/agent paths; allowCustom lets an
              // admin still enter a path not in the list.
              <SearchableSelect
                options={targetOptions}
                value={formName}
                onChange={setFormName}
                isLoading={targetOptionsLoading}
                allowCustom
                placeholder={
                  formEntityType === 'a2a_agent'
                    ? 'Search agent paths… (e.g. /booking-agent)'
                    : 'Search server paths… (e.g. mcpgw)'
                }
              />
            )}
          </div>

          {isServerGroup && (
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">
                Servers in this group
              </label>
              {/* Add-a-server typeahead: each pick is appended as a removable chip. */}
              <SearchableSelect
                options={targetOptions}
                value=""
                onChange={(path) =>
                  setFormMembers((prev) => (prev.includes(path) ? prev : [...prev, path]))
                }
                isLoading={serversLoading}
                allowCustom
                placeholder="Add a server path… (e.g. mcpgw)"
              />
              {formMembers.length > 0 && (
                <div className="mt-2 flex flex-wrap gap-2">
                  {formMembers.map((m) => (
                    <span
                      key={m}
                      className="inline-flex items-center gap-1 rounded-full bg-blue-100 dark:bg-blue-900/40 px-2.5 py-1 text-xs font-mono text-blue-800 dark:text-blue-200"
                    >
                      {m}
                      <button
                        type="button"
                        onClick={() => setFormMembers((prev) => prev.filter((x) => x !== m))}
                        className="text-blue-500 hover:text-blue-700 dark:hover:text-blue-100"
                        aria-label={`Remove ${m}`}
                      >
                        ×
                      </button>
                    </span>
                  ))}
                </div>
              )}
              <p className="mt-2 text-xs text-gray-400">
                Each server in this group is limited to the Max requests/window below{' '}
                <strong>individually</strong> (a separate bucket per server). This is NOT a
                shared or pooled limit across the group.
              </p>
            </div>
          )}

          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">
              Window (seconds) — 1 to 86400
            </label>
            <input
              type="number"
              value={formWindow}
              onChange={(e) => setFormWindow(Number(e.target.value))}
              disabled={!!editingId}
              className={inputBase}
            />
            <p className="mt-1 text-xs text-gray-400">
              Editing a definition keeps its axis/type/name/window (they form its id). Change those by
              deleting and recreating.
            </p>
          </div>

          {isGroup ? (
            <div className="grid grid-cols-2 gap-4">
              <div>
                <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">
                  User max/window
                </label>
                <input
                  type="number"
                  value={formUserMax}
                  onChange={(e) => setFormUserMax(e.target.value)}
                  placeholder="e.g. 25"
                  className={inputBase}
                />
              </div>
              <div>
                <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">
                  Agent max/window
                </label>
                <input
                  type="number"
                  value={formAgentMax}
                  onChange={(e) => setFormAgentMax(e.target.value)}
                  placeholder="e.g. 15"
                  className={inputBase}
                />
              </div>
              <p className="col-span-2 text-xs text-gray-400">
                Set at least one. On windows ≤ 60s a floor applies (user ≥ 20/min, agent ≥ 10/min by
                default) and values below it are rejected.
              </p>
            </div>
          ) : (
            <div>
              <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Max requests/window</label>
              <input
                type="number"
                value={formTargetMax}
                onChange={(e) => setFormTargetMax(e.target.value)}
                placeholder="e.g. 500"
                className={inputBase}
              />
            </div>
          )}

          <label className="flex items-center gap-2 text-sm text-gray-600 dark:text-gray-400">
            <input
              type="checkbox"
              checked={formFailClosed}
              onChange={(e) => setFormFailClosed(e.target.checked)}
            />
            Fail closed (deny on backend error — security-critical limits only)
          </label>

          <div className="flex gap-3 pt-2">
            <button
              onClick={handleSave}
              disabled={isSaving}
              className="px-4 py-2 bg-blue-600 text-white rounded-md hover:bg-blue-700 disabled:opacity-50"
            >
              {isSaving ? 'Saving…' : editingId ? 'Update' : 'Create'}
            </button>
            <button
              onClick={() => { resetForm(); setView('list'); }}
              className="px-4 py-2 border border-gray-300 dark:border-gray-600 rounded-md text-gray-700 dark:text-gray-200"
            >
              Cancel
            </button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">IAM &gt; Rate Limits</h2>
        <button
          onClick={openCreate}
          className="flex items-center px-3 py-2 bg-blue-600 text-white text-sm rounded-md hover:bg-blue-700"
        >
          <PlusIcon className="h-4 w-4 mr-1" />
          New Rate Limit
        </button>
      </div>

      {/* High-level config summary (counts from definitions + memberships). These
          are configuration counts, not live throttle counts. */}
      <div className="grid grid-cols-2 sm:grid-cols-3 lg:grid-cols-6 gap-3">
        <StatTile label="Total limits" value={summary.total} hint="All rate-limit definitions" />
        <StatTile label="Enabled" value={summary.enabled} hint="Definitions currently enforced" />
        <StatTile label="Per caller" value={summary.caller} hint="Per-caller group limits (across all targets)" />
        <StatTile label="Per caller, per target" value={summary.callerTarget} hint="Independent quota per caller per target" />
        <StatTile label="Per target" value={summary.target} hint="Per MCP-server / A2A-agent limits (across all callers)" />
        <StatTile label="Callers governed" value={summary.governed} hint="Users/agents mapped into a rate-limit group" />
      </div>

      <div className="relative max-w-md">
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search by name or type"
          className={`${inputBase} pl-9`}
        />
      </div>

      <ListStateBoundary
        isLoading={isLoading}
        error={error}
        isEmpty={filtered.length === 0}
        emptyMessage="No rate-limit definitions. Create one to cap tool/agent usage."
      >
        <div className="overflow-x-auto">
          <table className="min-w-full text-sm">
            <thead>
              <tr className="text-left text-gray-500 dark:text-gray-400 border-b border-gray-200 dark:border-gray-700">
                <th className="py-2 pr-4">Axis</th>
                <th className="py-2 pr-4">Type</th>
                <th className="py-2 pr-4">Name</th>
                <th className="py-2 pr-4">Limit</th>
                <th className="py-2 pr-4">State</th>
                <th className="py-2 pr-4">Actions</th>
              </tr>
            </thead>
            <tbody>
              {filtered.map((d) => {
                const id = definitionId(d);
                return (
                  <tr key={id} className="border-b border-gray-100 dark:border-gray-800">
                    <td className="py-2 pr-4 text-gray-700 dark:text-gray-300">{AXIS_LABELS[d.axis] ?? d.axis}</td>
                    <td className="py-2 pr-4 text-gray-700 dark:text-gray-300">{d.entity_type}</td>
                    <td className="py-2 pr-4 font-mono text-xs text-gray-900 dark:text-white">{d.name}</td>
                    <td className="py-2 pr-4 text-gray-700 dark:text-gray-300">{limitSummary(d)}</td>
                    <td className="py-2 pr-4">
                      <span className={d.enabled ? 'text-green-600 dark:text-green-400' : 'text-gray-400'}>
                        {d.enabled ? 'enabled' : 'disabled'}
                      </span>
                      {d.fail_closed && (
                        <span className="ml-2 text-xs text-amber-600 dark:text-amber-400">fail-closed</span>
                      )}
                    </td>
                    <td className="py-2 pr-4">
                      <div className="flex items-center gap-3">
                        <button
                          onClick={() => openEdit(d)}
                          className="text-gray-500 hover:text-blue-600"
                          title="Edit"
                        >
                          <PencilIcon className="h-4 w-4" />
                        </button>
                        <button
                          onClick={() => handleToggle(d)}
                          className="text-xs text-gray-500 hover:text-blue-600"
                          title={d.enabled ? 'Disable' : 'Enable'}
                        >
                          {d.enabled ? 'Disable' : 'Enable'}
                        </button>
                        <button
                          onClick={() => setDeleteTarget(id)}
                          className="text-gray-500 hover:text-red-600"
                          title="Delete"
                        >
                          <TrashIcon className="h-4 w-4" />
                        </button>
                      </div>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </ListStateBoundary>

      {/* Quarantine (kill switch): the two seeded reserved groups with member
          counts, a global on/off toggle, and per-member remove. */}
      <QuarantinePanel onShowToast={onShowToast} isAdmin={isAdmin} />

      {deleteTarget && (
        <DeleteConfirmation
          entityType="rate-limit"
          entityName={deleteTarget}
          entityPath={deleteTarget}
          onConfirm={handleDelete}
          onCancel={() => setDeleteTarget(null)}
        />
      )}
    </div>
  );
};

export default IAMRateLimits;
