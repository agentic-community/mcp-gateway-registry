import React, { useMemo, useState } from 'react';
import { ExclamationTriangleIcon, TrashIcon, NoSymbolIcon } from '@heroicons/react/24/outline';
import {
  useQuarantineList,
  quarantineAdd,
  quarantineRemove,
  setRateLimitEnabled,
  useRateLimitDefinitions,
  QUARANTINE_CALLER_GROUP,
  QUARANTINE_TARGET_GROUP,
  type RateLimitMembership,
  type QuarantineSubjectType,
} from '../../hooks/useRateLimits';
import SearchableSelect from '../SearchableSelect';
import { useServerList } from '../../hooks/useToolCatalog';
import { useAgentList } from '../../hooks/useAgentList';

interface QuarantinePanelProps {
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
  // Adding a target to quarantine is admin-only (the backend also enforces admin).
  isAdmin?: boolean;
}

// The seeded sentinel id is 'quarantine:group:<name>:1' (window 1s is a
// placeholder; the sentinel carries no rate).
function sentinelId(group: string): string {
  return `quarantine:group:${group}:1`;
}

function apiError(err: any, fallback: string): string {
  return err?.response?.data?.detail || err?.message || fallback;
}

/**
 * The Quarantine (kill-switch) section of the Rate Limits panel. Shows the two
 * seeded reserved groups (callers + targets) with a live member count, a global
 * enable/disable master switch per group (destructive styling + confirm), and a
 * per-member "remove from quarantine" action. Quarantine drops ALL data-plane
 * traffic from/to a subject -- distinct from a rate limit.
 */
const QuarantinePanel: React.FC<QuarantinePanelProps> = ({ onShowToast, isAdmin = false }) => {
  const { quarantined, isLoading, refetch } = useQuarantineList();
  const { definitions, refetch: refetchDefs } = useRateLimitDefinitions();
  const { servers, isLoading: serversLoading } = useServerList();
  const { agents, isLoading: agentsLoading } = useAgentList();
  const [confirmToggle, setConfirmToggle] = useState<{ group: string; enable: boolean } | null>(
    null,
  );
  // Target-add control state: which kind of target, and the add-in-flight flag.
  const [targetKind, setTargetKind] = useState<'server' | 'agent'>('server');
  const [isAddingTarget, setIsAddingTarget] = useState(false);

  // Options for the target typeahead: registered server paths or agent paths.
  const targetOptions = useMemo(() => {
    if (targetKind === 'agent') {
      return agents.map((a) => ({ value: a.path, label: a.path, description: a.name }));
    }
    return servers.map((s) => ({ value: s.path, label: s.path, description: s.name }));
  }, [targetKind, servers, agents]);

  const handleAddTarget = async (subject: string) => {
    const name = subject.trim();
    if (!name) return;
    setIsAddingTarget(true);
    try {
      await quarantineAdd(targetKind, name);
      onShowToast(`Quarantined ${targetKind}:${name} (all traffic blocked)`, 'success');
      refetch();
    } catch (err) {
      onShowToast(apiError(err, 'Failed to quarantine target'), 'error');
    } finally {
      setIsAddingTarget(false);
    }
  };

  const callers = useMemo(
    () => quarantined.filter((m) => m.subject_type === 'user' || m.subject_type === 'client'),
    [quarantined],
  );
  const targets = useMemo(
    () => quarantined.filter((m) => m.subject_type === 'server' || m.subject_type === 'agent'),
    [quarantined],
  );

  // Whether each reserved group's sentinel is enabled (the global kill switch).
  const enabledByGroup = useMemo(() => {
    const map: Record<string, boolean> = {
      [QUARANTINE_CALLER_GROUP]: true,
      [QUARANTINE_TARGET_GROUP]: true,
    };
    for (const d of definitions) {
      if (d.axis === 'quarantine') map[d.name] = d.enabled;
    }
    return map;
  }, [definitions]);

  const handleRemove = async (m: RateLimitMembership) => {
    try {
      await quarantineRemove(m.subject_type as QuarantineSubjectType, m.subject);
      onShowToast(`Removed ${m.subject_type}:${m.subject} from quarantine`, 'success');
      refetch();
    } catch (err) {
      onShowToast(apiError(err, 'Failed to remove from quarantine'), 'error');
    }
  };

  const applyToggle = async () => {
    if (!confirmToggle) return;
    const { group, enable } = confirmToggle;
    setConfirmToggle(null);
    try {
      await setRateLimitEnabled(sentinelId(group), enable);
      onShowToast(
        `Quarantine group ${group} ${enable ? 'enabled' : 'disabled (kill switch OFF)'}`,
        'success',
      );
      refetchDefs();
    } catch (err) {
      onShowToast(apiError(err, 'Failed to toggle quarantine group'), 'error');
    }
  };

  const renderGroup = (
    title: string,
    group: string,
    members: RateLimitMembership[],
    addControl?: React.ReactNode,
  ) => {
    const enabled = enabledByGroup[group];
    return (
      <div className="rounded-lg border border-red-200 dark:border-red-900/50 bg-red-50/40 dark:bg-red-900/10 p-4">
        <div className="flex items-center justify-between mb-2">
          <div className="flex items-center gap-2">
            <ExclamationTriangleIcon className="h-5 w-5 text-red-600 dark:text-red-400" />
            <span className="font-medium text-gray-900 dark:text-white">{title}</span>
            <span className="text-xs px-2 py-0.5 rounded-full bg-red-100 dark:bg-red-900/40 text-red-800 dark:text-red-300">
              {members.length} member{members.length === 1 ? '' : 's'}
            </span>
          </div>
          <button
            onClick={() => setConfirmToggle({ group, enable: !enabled })}
            className={
              enabled
                ? 'text-xs font-medium text-red-600 dark:text-red-400 hover:underline'
                : 'text-xs font-medium text-green-600 dark:text-green-400 hover:underline'
            }
            title={enabled ? 'Turn the kill switch OFF globally' : 'Turn the kill switch ON'}
          >
            {enabled ? 'Enabled — disable globally' : 'Disabled — enable'}
          </button>
        </div>
        {members.length === 0 ? (
          <p className="text-xs text-gray-500 dark:text-gray-400">Nothing quarantined.</p>
        ) : (
          <ul className="divide-y divide-red-100 dark:divide-red-900/40">
            {members.map((m) => (
              <li key={`${m.subject_type}:${m.subject}`} className="flex items-center justify-between py-1.5">
                <span className="font-mono text-xs text-gray-800 dark:text-gray-200">
                  {m.subject_type}:{m.subject}
                </span>
                <button
                  onClick={() => handleRemove(m)}
                  className="flex items-center gap-1 text-xs text-gray-500 hover:text-red-600"
                  title="Remove from quarantine"
                >
                  <TrashIcon className="h-4 w-4" />
                  Remove
                </button>
              </li>
            ))}
          </ul>
        )}
        {addControl}
      </div>
    );
  };

  // Admin-only "add a target to quarantine" control, rendered inside the targets group.
  const targetAddControl = isAdmin ? (
    <div className="mt-3 border-t border-red-100 dark:border-red-900/40 pt-3 space-y-2">
      <div className="flex items-center gap-2">
        <span className="text-xs text-gray-500 dark:text-gray-400">Quarantine a:</span>
        <select
          value={targetKind}
          onChange={(e) => setTargetKind(e.target.value as 'server' | 'agent')}
          className="text-xs rounded border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-900 text-gray-900 dark:text-white px-2 py-1"
        >
          <option value="server">server</option>
          <option value="agent">agent</option>
        </select>
      </div>
      <SearchableSelect
        options={targetOptions}
        value=""
        onChange={handleAddTarget}
        isLoading={targetKind === 'agent' ? agentsLoading : serversLoading}
        disabled={isAddingTarget}
        allowCustom
        focusColor="focus:ring-red-500"
        placeholder={
          targetKind === 'agent'
            ? 'Add an agent to quarantine… (e.g. /booking-agent)'
            : 'Add a server to quarantine… (e.g. mcpgw)'
        }
      />
      <p className="text-[11px] text-gray-400">
        Blocks ALL traffic to this target for every caller (admins included). Use the bare
        name/path segment the request routes to (e.g. <code>mcpgw</code>).
      </p>
    </div>
  ) : null;

  return (
    <div className="space-y-3">
      <div>
        <h3 className="text-md font-semibold text-gray-900 dark:text-white">Quarantine (kill switch)</h3>
        <p className="text-xs text-gray-500 dark:text-gray-400">
          A quarantined caller or target has ALL of its data-plane traffic dropped immediately (a
          hard block, not a rate). Quarantine a caller (admin only) from the Users or M2M pages via
          the block icon on each row; quarantine a target (server/agent) using the control in the
          Quarantined targets box below. Remove a subject with the trash icon. Admin-group users
          cannot be quarantined. Changes take effect within ~30s.
        </p>
      </div>
      {isLoading ? (
        <p className="text-sm text-gray-500 dark:text-gray-400">Loading…</p>
      ) : (
        <div className="grid gap-3 md:grid-cols-2">
          {renderGroup('Quarantined callers', QUARANTINE_CALLER_GROUP, callers)}
          {renderGroup('Quarantined targets', QUARANTINE_TARGET_GROUP, targets, targetAddControl)}
        </div>
      )}

      {confirmToggle && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="max-w-md rounded-lg bg-white dark:bg-gray-800 p-6 shadow-xl">
            <div className="flex items-center gap-2 mb-3">
              <ExclamationTriangleIcon className="h-6 w-6 text-red-600 dark:text-red-400" />
              <h4 className="font-semibold text-gray-900 dark:text-white">
                {confirmToggle.enable ? 'Enable' : 'Disable'} kill switch
              </h4>
            </div>
            <p className="text-sm text-gray-600 dark:text-gray-300 mb-4">
              {confirmToggle.enable
                ? `Enabling ${confirmToggle.group} re-activates the kill switch: all of its members' traffic will be dropped again.`
                : `Disabling ${confirmToggle.group} turns the kill switch OFF for ALL its members — their traffic will flow again even though they remain listed.`}
            </p>
            <div className="flex justify-end gap-3">
              <button
                onClick={() => setConfirmToggle(null)}
                className="px-3 py-1.5 rounded-md border border-gray-300 dark:border-gray-600 text-sm text-gray-700 dark:text-gray-200"
              >
                Cancel
              </button>
              <button
                onClick={applyToggle}
                className="px-3 py-1.5 rounded-md bg-red-600 text-white text-sm hover:bg-red-700"
              >
                {confirmToggle.enable ? 'Enable' : 'Disable'}
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default QuarantinePanel;
