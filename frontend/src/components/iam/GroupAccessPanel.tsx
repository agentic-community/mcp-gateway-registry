import React, { useState, useRef, useEffect } from 'react';
import { XMarkIcon, ChevronDownIcon, ChevronUpIcon } from '@heroicons/react/24/outline';
import SearchableSelect from '../SearchableSelect';
import UiPermissionEditor from './UiPermissionEditor';

// Mirrors ServerAccessEntry in IAMGroups.tsx. Re-declared here so the panel is
// self-contained; IAMGroups imports this type from here after the refactor.
export interface ServerAccessEntry {
  server: string;
  methods: string[];
  tools: string[];
}

interface EntityScopeGroup {
  typeName: string;
  displayName: string;
  listKey: string;
  mutationKeys: { key: string; label: string }[];
}

type AccessTab = 'servers' | 'agents' | 'permissions';

// ─── Methods popover ─────────────────────────────────────────────
// Collapses the 11-checkbox method grid into a chip summary that opens a
// checkbox popover (wireframe 1). Chip shows e.g. "GET, POST" or "all".
interface MethodsPopoverProps {
  methods: string[];
  allMethods: string[];
  onToggle: (method: string) => void;
}

const MethodsPopover: React.FC<MethodsPopoverProps> = ({ methods, allMethods, onToggle }) => {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (!open) return;
    const onClickOutside = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', onClickOutside);
    return () => document.removeEventListener('mousedown', onClickOutside);
  }, [open]);

  const summary =
    methods.length === 0
      ? 'none'
      : methods.length === allMethods.length
        ? 'all'
        : methods.length <= 2
          ? methods.join(', ')
          : `${methods.slice(0, 2).join(', ')} +${methods.length - 2}`;

  return (
    <div className="relative inline-block" ref={ref}>
      <button
        type="button"
        onClick={() => setOpen((o) => !o)}
        className={`inline-flex items-center gap-1 px-2.5 py-1 text-xs rounded-full border transition-colors
          ${open
            ? 'border-purple-300 dark:border-purple-700 bg-purple-50 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300'
            : 'border-gray-300 dark:border-gray-600 text-gray-600 dark:text-gray-400 hover:border-purple-400'}`}
      >
        {summary}
        {open ? <ChevronUpIcon className="h-3 w-3" /> : <ChevronDownIcon className="h-3 w-3" />}
      </button>
      {open && (
        <div className="absolute z-20 mt-1 left-0 w-56 max-h-64 overflow-y-auto rounded-lg border border-gray-200
                        dark:border-gray-700 bg-white dark:bg-gray-800 shadow-lg p-3 space-y-1">
          <p className="text-[10px] font-semibold uppercase tracking-wide text-gray-400 mb-1">Methods</p>
          {allMethods.map((method) => (
            <label key={method} className="flex items-center space-x-2 cursor-pointer py-0.5">
              <input
                type="checkbox"
                checked={methods.includes(method)}
                onChange={() => onToggle(method)}
                className="rounded border-gray-300 dark:border-gray-600 text-purple-600 focus:ring-purple-500 h-3 w-3"
              />
              <span className="text-xs text-gray-600 dark:text-gray-400">{method}</span>
            </label>
          ))}
        </div>
      )}
    </div>
  );
};

// ─── Panel props ─────────────────────────────────────────────────
// All state stays in IAMGroups (single source of truth for _buildScopeJson and
// the JSON upload path); this component is purely presentational + callbacks.
interface GroupAccessPanelProps {
  // servers tab
  serverAccess: ServerAccessEntry[];
  availableServers: { path: string; name: string; type?: string; description?: string }[];
  serversLoading: boolean;
  commonMethods: string[];
  onAddServerEntry: () => void;
  onRemoveServerEntry: (idx: number) => void;
  onUpdateServerEntry: (idx: number, field: keyof ServerAccessEntry, value: string | string[]) => void;
  onToggleMethod: (idx: number, method: string) => void;
  renderToolsSelector: (entry: ServerAccessEntry, idx: number) => React.ReactNode;
  // agents tab
  selectedAgents: string[];
  availableAgents: { path: string; name: string; description?: string }[];
  agentsLoading: boolean;
  onAddAgent: (path: string) => void;
  onRemoveAgent: (path: string) => void;
  // permissions tab (skills + custom types live inside UiPermissionEditor)
  uiPermissions: Record<string, string>;
  setPermValue: (key: string, value: string) => void;
  entityScopeGroups: EntityScopeGroup[];
  skillOptions: { value: string; label: string; description?: string }[];
  skillsLoading: boolean;
}

// Keys auto-synced from server/agent selections by IAMGroups' save handlers —
// not user-granted, so excluded from the Permissions tab count.
const AUTO_SYNCED_PERM_KEYS = new Set([
  'list_service', 'health_check_service', 'get_service', 'list_tools',
  'call_tool', 'list_virtual_server', 'list_agents', 'get_agent',
]);

const TAB_LABELS: { id: AccessTab; label: string }[] = [
  { id: 'servers', label: 'Servers' },
  { id: 'agents', label: 'Agents' },
  { id: 'permissions', label: 'Permissions' },
];

const GroupAccessPanel: React.FC<GroupAccessPanelProps> = (props) => {
  const [activeTab, setActiveTab] = useState<AccessTab>('servers');

const grantedPermCount = Object.entries(props.uiPermissions)
  .filter(([key, val]) => !AUTO_SYNCED_PERM_KEYS.has(key) && Boolean(val)).length;  const counts: Record<AccessTab, number> = {
    servers: props.serverAccess.filter((e) => e.server).length,
    agents: props.selectedAgents.length,
    permissions: grantedPermCount,
  };

  return (
    <div className="flex gap-4 items-start">
      {/* ── Tab rail ── */}
      <div className="w-44 shrink-0 border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
        <p className="px-3 py-2 text-[10px] font-semibold uppercase tracking-wide text-gray-400
                      border-b border-gray-200 dark:border-gray-700">
          Access
        </p>
        {TAB_LABELS.map(({ id, label }) => (
          <button
            key={id}
            type="button"
            onClick={() => setActiveTab(id)}
            className={`w-full flex items-center justify-between px-3 py-2 text-sm border-l-2 transition-colors
              ${activeTab === id
                ? 'border-purple-600 bg-purple-50 dark:bg-purple-900/20 text-purple-800 dark:text-purple-300 font-medium'
                : 'border-transparent text-gray-600 dark:text-gray-400 hover:bg-gray-50 dark:hover:bg-gray-800'}`}
          >
            {label}
            <span
              className={`text-[10px] px-2 py-0.5 rounded-full
                ${activeTab === id
                  ? 'bg-purple-100 dark:bg-purple-900/40 text-purple-700 dark:text-purple-300'
                  : 'bg-gray-100 dark:bg-gray-800 text-gray-500'}`}
            >
              {counts[id]}
            </span>
          </button>
        ))}
      </div>

      {/* ── Active tab content ── */}
      <div className="flex-1 min-w-0 border border-gray-200 dark:border-gray-700 rounded-lg">
        {activeTab === 'servers' && (
          <div>
            <div className="flex items-center justify-between px-4 py-2.5 border-b border-gray-200 dark:border-gray-700">
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300">Server access</p>
              <button
                type="button"
                onClick={props.onAddServerEntry}
                className="text-xs px-2.5 py-1 rounded-lg bg-purple-600 text-white hover:bg-purple-700"
              >
                + Add row
              </button>
            </div>
            <table className="w-full text-sm">
              <thead>
                <tr className="text-left text-[10px] uppercase tracking-wide text-gray-400 border-b border-gray-200 dark:border-gray-700">
                  <th className="px-4 py-2 w-[38%] font-semibold">Server</th>
                  <th className="px-2 py-2 w-[22%] font-semibold">Methods</th>
                  <th className="px-2 py-2 font-semibold">Tools</th>
                  <th className="px-2 py-2 w-10" />
                </tr>
              </thead>
              <tbody>
                {props.serverAccess.map((entry, idx) => (
                  <tr key={idx} className="border-b border-gray-100 dark:border-gray-800 align-top">
                    <td className="px-4 py-2">
                      <SearchableSelect
                        options={props.availableServers.map((s) => ({
                          value: s.path,
                          label: `${s.type === 'virtual' ? '[Virtual] ' : ''}${s.name} (${s.path})`,
                          description: s.description,
                        }))}
                        value={entry.server}
                        onChange={(val) => {
                          props.onUpdateServerEntry(idx, 'server', val);
                          props.onUpdateServerEntry(idx, 'tools', []); // reset tools on server change
                        }}
                        placeholder="Search servers..."
                        isLoading={props.serversLoading}
                        maxDescriptionWords={8}
                        specialOptions={[
                          { value: '*', label: '* (All servers)', description: 'Grant access to all servers' },
                        ]}
                      />
                    </td>
                    <td className="px-2 py-2">
                      <MethodsPopover
                        methods={entry.methods}
                        allMethods={props.commonMethods}
                        onToggle={(m) => props.onToggleMethod(idx, m)}
                      />
                    </td>
                    <td className="px-2 py-2">{props.renderToolsSelector(entry, idx)}</td>
                    <td className="px-2 py-2 text-right">
                      {props.serverAccess.length > 1 && (
                        <button
                          type="button"
                          onClick={() => props.onRemoveServerEntry(idx)}
                          className="text-gray-400 hover:text-red-500"
                          aria-label={`Remove server row ${idx + 1}`}
                        >
                          <XMarkIcon className="h-4 w-4" />
                        </button>
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
            <p className="px-4 py-2 text-[11px] text-gray-400">
              Methods open a checkbox popover, collapsed to a chip summary.
            </p>
          </div>
        )}

        {activeTab === 'agents' && (
          <div className="p-4 space-y-3">
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
              Agent access <span className="text-xs text-gray-400">(optional)</span>
            </p>
            {props.selectedAgents.length > 0 && (
              <div className="flex flex-wrap gap-2">
                {props.selectedAgents.map((agentName) => (
                  <span
                    key={agentName}
                    className="inline-flex items-center px-2 py-1 text-xs bg-purple-100 dark:bg-purple-900/30
                               text-purple-700 dark:text-purple-300 rounded-full"
                  >
                    {agentName}
                    <button
                      type="button"
                      onClick={() => props.onRemoveAgent(agentName)}
                      className="ml-1 hover:text-purple-900 dark:hover:text-purple-100"
                    >
                      <XMarkIcon className="h-3 w-3" />
                    </button>
                  </span>
                ))}
              </div>
            )}
            <SearchableSelect
              options={props.availableAgents
                .filter((a) => !props.selectedAgents.includes(a.path))
                .map((a) => ({
                  value: a.path,
                  label: `${a.name} (${a.path})`,
                  description: a.description,
                }))}
              value=""
              onChange={(val) => {
                if (val) props.onAddAgent(val);
              }}
              placeholder="Search and add agents..."
              isLoading={props.agentsLoading}
              maxDescriptionWords={8}
              specialOptions={[
                { value: 'all', label: '* (All agents)', description: 'Grant access to all agents' },
              ]}
            />
          </div>
        )}

        {activeTab === 'permissions' && (
          <div className="p-4 space-y-3">
            <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
              UI permissions{' '}
              <span className="text-xs text-gray-400">
                (mutation permissions and skill/custom-type discovery)
              </span>
            </p>
            <UiPermissionEditor
              uiPermissions={props.uiPermissions}
              setPermValue={props.setPermValue}
              entityScopeGroups={props.entityScopeGroups}
              skillOptions={props.skillOptions}
              skillsLoading={props.skillsLoading}
            />
          </div>
        )}
      </div>
    </div>
  );
};

export default GroupAccessPanel;
