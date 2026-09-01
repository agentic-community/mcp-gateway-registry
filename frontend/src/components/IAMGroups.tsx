import React, { useState, useMemo, useCallback } from 'react';
import {
  PlusIcon,
  MagnifyingGlassIcon,
  TrashIcon,
  ArrowLeftIcon,
  ArrowPathIcon,
  ArrowDownTrayIcon,
  DocumentArrowUpIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  XMarkIcon,
  PencilIcon,
} from '@heroicons/react/24/outline';
import {
  useIAMGroups,
  createGroup,
  deleteGroup,
  getGroup,
  updateGroup,
  CreateGroupPayload,
  GroupDetail,
  UpdateGroupPayload,
} from '../hooks/useIAM';
import { useServerList, useServerTools } from '../hooks/useToolCatalog';
import type { ServerInfo } from '../hooks/useToolCatalog';
import {
  buildScopeJson,
  normalizeServerKey as normalizeScopeServerKey,
  applyUiPermSync as applyScopeUiPermSync,
  type ServerAccessEntry,
} from './iam/scopeJson';
import { useAgentList } from '../hooks/useAgentList';
import { useSkills } from '../hooks/useSkills';
import { useRegistryConfig } from '../hooks/useRegistryConfig';
import DeleteConfirmation from './DeleteConfirmation';
import SearchableSelect from './SearchableSelect';
import GroupAccessPanel from './iam/GroupAccessPanel';
import ListStateBoundary from './iam/ListStateBoundary';
import UiPermissionEditor from './iam/UiPermissionEditor';

interface IAMGroupsProps {
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
}

type View = 'list' | 'create' | 'edit';

// ServerAccessEntry is imported from ./iam/scopeJson (shared with the build logic).

// ─── Per-type custom-entity ui_permissions ────────
// Each admin-defined custom type mints list/create/modify/delete_<type>_entity
// scopes, edited in UiPermissionEditor. Enumerated from the current type set (via
// /api/config) so an admin can grant them proactively, before any record exists.
// The keys mirror registry/services/custom_entity_scopes.entity_scope() exactly.
// Mutation actions render as free-text grants; the read action (`list`) renders
// as a record picker (CustomTypeListPicker) since its grant supports specific
// record paths, not just "all".
const ENTITY_MUTATION_ACTIONS: { action: string; verb: string }[] = [
  { action: 'create', verb: 'Create' },
  { action: 'modify', verb: 'Modify' },
  { action: 'delete', verb: 'Delete' },
];

interface EntityScopeGroup {
  typeName: string;
  displayName: string;
  listKey: string;
  mutationKeys: { key: string; label: string }[];
}

function buildEntityScopeGroups(
  customTypes: { name: string; display_name: string }[],
): EntityScopeGroup[] {
  return customTypes.map((t) => ({
    typeName: t.name,
    displayName: t.display_name || t.name,
    listKey: `list_${t.name}_entity`,
    mutationKeys: ENTITY_MUTATION_ACTIONS.map(({ action, verb }) => ({
      key: `${action}_${t.name}_entity`,
      label: `${verb} ${t.display_name || t.name}`,
    })),
  }));
}

// Method value spaces are SEPARATE by entity kind (mirrors the auth-server's
// wildcard split): MCP method tokens for mcp/virtual servers, HTTP verbs for
// proxied non-MCP entities. The legacy MCP "*"/"all" wildcard does NOT authorize
// an HTTP verb server-side, so proxied entities use explicit verbs or "http:*".
const MCP_METHODS = [
  'initialize',
  'notifications/initialized',
  'ping',
  'tools/list',
  'tools/call',
  'resources/list',
  'resources/templates/list',
];

const HTTP_VERBS = ['GET', 'HEAD', 'OPTIONS', 'POST', 'PUT', 'PATCH', 'DELETE', 'http:*'];

// State-changing HTTP verbs — selecting these (or http:*) grants destructive
// access to the proxied backend; the editor warns when any is chosen.
const DESTRUCTIVE_HTTP_VERBS = new Set(['POST', 'PUT', 'PATCH', 'DELETE', 'http:*']);

// Back-compat alias: the create form's default entry still seeds MCP methods
// (an MCP server is the default new entry).
const COMMON_METHODS = MCP_METHODS;

// Example scope JSON matching the format from scripts/registry-admins.json
const EXAMPLE_SCOPE_JSON = {
  scope_name: 'currenttime-users',
  description: 'Users with access to currenttime server',
  server_access: [
    {
      server: 'currenttime',
      methods: ['initialize', 'tools/list', 'tools/call'],
      tools: ['current_time_by_timezone'],
    },
  ],
  group_mappings: ['currenttime-users'],
  ui_permissions: {
    list_service: ['currenttime'],
    health_check_service: ['currenttime'],
  },
  create_in_idp: false,
};

// Default entry has all methods selected
const EMPTY_SERVER_ENTRY: ServerAccessEntry = { server: '', methods: [...COMMON_METHODS], tools: [] };

// Human label for a server/entity option in the scope-editor picker, tagged by
// kind so an admin can tell an MCP server from a proxied skill/agent/custom.
function _serverOptionLabel(s: ServerInfo): string {
  const tag =
    s.type === 'virtual'
      ? '[Virtual] '
      : s.type === 'skill'
        ? '[Skill] '
        : s.type === 'a2a_agent'
          ? '[Agent] '
          : s.type === 'custom'
            ? `[${s.entityType || 'Custom'}] `
            : '';
  return `${tag}${s.name} (${s.path})`;
}


/**
 * Sub-component for selecting tools for a specific server.
 * Uses useServerTools hook to fetch available tools and SearchableSelect for UI.
 */
interface ServerToolsSelectorProps {
  serverPath: string;
  selectedTools: string[];
  onChange: (tools: string[]) => void;
}

const ServerToolsSelector: React.FC<ServerToolsSelectorProps> = ({
  serverPath,
  selectedTools,
  onChange,
}) => {
  const { tools, isLoading } = useServerTools(serverPath);

  // Handle adding a tool
  const handleAddTool = (toolName: string) => {
    if (!toolName) return;

    // If selecting wildcard, replace all with just wildcard
    if (toolName === '*') {
      onChange(['*']);
      return;
    }

    // If wildcard is already selected, don't add specific tools
    if (selectedTools.includes('*')) {
      return;
    }

    // Add tool if not already selected
    if (!selectedTools.includes(toolName)) {
      onChange([...selectedTools, toolName]);
    }
  };

  // Handle removing a tool
  const handleRemoveTool = (toolName: string) => {
    onChange(selectedTools.filter((t) => t !== toolName));
  };

  // If server is wildcard, show message
  if (serverPath === '*') {
    return (
      <div>
        <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Tools</label>
        <p className="text-xs text-gray-400 italic">All tools on all servers</p>
      </div>
    );
  }

  // If no server selected, show disabled state
  if (!serverPath) {
    return (
      <div>
        <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Tools</label>
        <p className="text-xs text-gray-400 italic">Select a server first</p>
      </div>
    );
  }

  // If wildcard is selected, show that with remove option
  if (selectedTools.includes('*')) {
    return (
      <div>
        <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Tools</label>
        <div className="flex items-center gap-2">
          <span className="inline-flex items-center px-2 py-1 text-xs bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full">
            * (All tools)
            <button
              type="button"
              onClick={() => handleRemoveTool('*')}
              className="ml-1 hover:text-purple-900 dark:hover:text-purple-100"
            >
              <XMarkIcon className="h-3 w-3" />
            </button>
          </span>
        </div>
      </div>
    );
  }

  // Build options from available tools, excluding already selected
  const availableOptions = tools
    .filter((t) => !selectedTools.includes(t.name))
    .map((t) => ({
      value: t.name,
      label: t.name,
      description: t.description,
    }));

  return (
    <div>
      <label className="block text-xs text-gray-500 dark:text-gray-400 mb-1">Tools</label>
      <div className="space-y-2">
        {/* Selected tools as removable tags */}
        {selectedTools.length > 0 && (
          <div className="flex flex-wrap gap-2">
            {selectedTools.map((toolName) => (
              <span
                key={toolName}
                className="inline-flex items-center px-2 py-1 text-xs bg-purple-100 dark:bg-purple-900/30 text-purple-700 dark:text-purple-300 rounded-full"
              >
                {toolName}
                <button
                  type="button"
                  onClick={() => handleRemoveTool(toolName)}
                  className="ml-1 hover:text-purple-900 dark:hover:text-purple-100"
                >
                  <XMarkIcon className="h-3 w-3" />
                </button>
              </span>
            ))}
          </div>
        )}

        {/* Searchable tool selector */}
        <SearchableSelect
          options={availableOptions}
          value=""
          onChange={handleAddTool}
          placeholder="Search and add tools..."
          isLoading={isLoading}
          maxDescriptionWords={8}
          specialOptions={[
            { value: '*', label: '* (All tools)', description: 'Grant access to all tools on this server' },
          ]}
        />
      </div>
    </div>
  );
};


// Scope-JSON build logic (canonical-key normalization + perm-sync) lives in
// ./iam/scopeJson so the create and edit paths share one implementation and the
// authz-sensitive rules are unit-tested. Aliased to the historical local names.
const _normalizeServerKey = normalizeScopeServerKey;
const _applyUiPermSync = applyScopeUiPermSync;
const _buildScopeJson = buildScopeJson;


const IAMGroups: React.FC<IAMGroupsProps> = ({ onShowToast }) => {
  const { groups, isLoading, error, refetch } = useIAMGroups();
  const { servers: availableServers, isLoading: serversLoading } = useServerList();
  const { agents: availableAgents, isLoading: agentsLoading } = useAgentList();
  const { skills: availableSkills, loading: skillsLoading } = useSkills();
  const { config } = useRegistryConfig();

  // Skill options for the UiPermissionEditor list_skills multi-select. Keyed by
  // skill name (the resource identifier list_skills is matched against).
  const skillOptions = useMemo(
    () =>
      (availableSkills ?? []).map((s) => ({
        value: s.name,
        label: s.name,
        description: s.description || undefined,
      })),
    [availableSkills],
  );

  // Dynamic per-type entity scope keys, enabled only when the custom-types
  // feature is on. Enumerated from the current type set so admins can grant
  // before any record exists. Memoized so the render sites are stable.
  const customTypesEnabled = config?.features?.custom_types ?? false;
  const entityScopeGroups = useMemo(
    () =>
      customTypesEnabled ? buildEntityScopeGroups(config?.custom_types ?? []) : [],
    [customTypesEnabled, config?.custom_types],
  );
  const [searchQuery, setSearchQuery] = useState('');
  const [view, setView] = useState<View>('list');

  // Canonical authz keys the picker offered for PROXIED non-MCP entities. Used to
  // classify server_access entries so their scope key is written verbatim (not
  // slash-stripped) and they are NOT synced into MCP UI permissions.
  const proxiedKeys = useMemo(
    () =>
      new Set(
        availableServers
          .filter((s) => s.type === 'skill' || s.type === 'a2a_agent' || s.type === 'custom')
          .map((s) => s.path),
      ),
    [availableServers],
  );

  // ─── Create form state ──────────────────────────────────────
  const [formName, setFormName] = useState('');
  const [formDescription, setFormDescription] = useState('');
  const [serverAccess, setServerAccess] = useState<ServerAccessEntry[]>([{ ...EMPTY_SERVER_ENTRY }]);
  const [groupMappings, setGroupMappings] = useState('');
  const [selectedAgents, setSelectedAgents] = useState<string[]>([]);
  const [uiPermissions, setUiPermissions] = useState<Record<string, string>>({});
  const [createInIdp, setCreateInIdp] = useState(false);
  const [isCreating, setIsCreating] = useState(false);
  const [showUiPermissions, setShowUiPermissions] = useState(false);

  // ─── Edit state ────────────────────────────────────────────
  const [editingGroup, setEditingGroup] = useState<string | null>(null);
  const [groupDetail, setGroupDetail] = useState<GroupDetail | null>(null);
  const [isLoadingGroup, setIsLoadingGroup] = useState(false);
  const [isSaving, setIsSaving] = useState(false);

  // Delete state
  const [deleteTarget, setDeleteTarget] = useState<string | null>(null);

  // Derived: read-only JSON preview
  const jsonPreview = useMemo(() => {
    if (!formName.trim()) return null;
    return JSON.stringify(
      _buildScopeJson(formName.trim(), formDescription.trim(), serverAccess, groupMappings, selectedAgents, uiPermissions, createInIdp, proxiedKeys),
      null,
      2,
    );
  }, [formName, formDescription, serverAccess, groupMappings, selectedAgents, uiPermissions, createInIdp, proxiedKeys]);

  const filteredGroups = useMemo(() => {
    if (!searchQuery) return groups;
    const q = searchQuery.toLowerCase();
    return groups.filter(
      (g) =>
        g.name.toLowerCase().includes(q) ||
        (g.description || '').toLowerCase().includes(q)
    );
  }, [groups, searchQuery]);

  const resetForm = useCallback(() => {
    setFormName('');
    setFormDescription('');
    setServerAccess([{ ...EMPTY_SERVER_ENTRY }]);
    setGroupMappings('');
    setSelectedAgents([]);
    setUiPermissions({});
    setCreateInIdp(true);
  }, []);


  // ─── Handlers ─────────────────────────────────────────────────

  const handleCreate = async () => {
    if (!formName.trim()) return;
    setIsCreating(true);
    try {
      // Build scope_config from form state.
      // The management API validates scope_config (422 on malformed input),
      // fully applies it via scope_service.import_group, and triggers an
      // auth-server reload so the scope takes effect immediately.
      const scopeJson = _buildScopeJson(
        formName.trim(), formDescription.trim(),
        serverAccess, groupMappings, selectedAgents, uiPermissions, createInIdp,
        proxiedKeys,
      );
      const { scope_name, description, ...scopeConfig } = scopeJson;

      const payload: CreateGroupPayload = {
        name: formName.trim(),
        description: formDescription.trim() || undefined,
        scope_config: Object.keys(scopeConfig).length > 0 ? scopeConfig : undefined,
      };
      await createGroup(payload);
      onShowToast(`Group "${formName}" created successfully`, 'success');
      resetForm();
      setView('list');
      await refetch();
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      const message = Array.isArray(detail)
        ? detail.map((d: any) => d.msg).join(', ')
        : detail || 'Failed to create group';
      onShowToast(message, 'error');
    } finally {
      setIsCreating(false);
    }
  };

  const handleDelete = async (name: string) => {
    await deleteGroup(name);
    onShowToast(`Group "${name}" deleted`, 'success');
    setDeleteTarget(null);
    await refetch();
  };

  const handleEditClick = async (groupName: string) => {
    setIsLoadingGroup(true);
    setEditingGroup(groupName);
    try {
      const detail = await getGroup(groupName);
      setGroupDetail(detail);

      // Populate form fields with existing group data
      setFormName(detail.name);
      setFormDescription(detail.description || '');

      // Server access - only include entries with actual server values
      if (detail.server_access && detail.server_access.length > 0) {
        const entries: ServerAccessEntry[] = detail.server_access
          .filter((sa) => sa.server && sa.server.trim())
          .map((sa) => ({
            server: sa.server || '',
            methods: sa.methods || [],
            tools: sa.tools || [],
          }));
        setServerAccess(entries.length > 0 ? entries : [{ ...EMPTY_SERVER_ENTRY }]);
      } else {
        setServerAccess([{ ...EMPTY_SERVER_ENTRY }]);
      }

      // Group mappings
      if (detail.group_mappings && detail.group_mappings.length > 0) {
        setGroupMappings(detail.group_mappings.join(', '));
      } else {
        setGroupMappings('');
      }

      // Agent access
      if (detail.agent_access && detail.agent_access.length > 0) {
        setSelectedAgents(detail.agent_access);
      } else {
        setSelectedAgents([]);
      }

      // UI permissions
      if (detail.ui_permissions) {
        const perms: Record<string, string> = {};
        for (const [key, val] of Object.entries(detail.ui_permissions)) {
          perms[key] = Array.isArray(val) ? val.join(', ') : String(val);
        }
        setUiPermissions(perms);
      } else {
        setUiPermissions({});
      }

      // Reflect whether the group is IdP-managed or local-only (issue #946).
      // Null/undefined (legacy records) defaults to true to match pre-#946 behavior.
      setCreateInIdp(detail.is_idp_managed !== false);
      setView('edit');
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      const message = typeof detail === 'string' ? detail : 'Failed to load group details';
      onShowToast(message, 'error');
      setEditingGroup(null);
    } finally {
      setIsLoadingGroup(false);
    }
  };

  const handleUpdate = async () => {
    if (!editingGroup) return;
    setIsSaving(true);
    try {
      // Build scope_config from form state. Proxied non-MCP entities keep their
      // canonical authz key (entity_type/registered_path); MCP/virtual are
      // slash-normalized. Uses the SAME shared normalizer + perm-sync as the
      // create path (_buildScopeJson) so both actions write identical output.
      const serverAccessPayload = serverAccess
        .filter((e) => e.server.trim())
        .map((e) => {
          const entry: {server: string; methods: string[]; tools?: string[]} = {
            server: _normalizeServerKey(e.server, proxiedKeys),
            methods: e.methods.length > 0 ? e.methods : ['all'],
          };
          if (e.tools.length > 0) {
            entry.tools = e.tools;
          }
          return entry;
        });

      // Build UI permissions, then auto-sync (shared helper; proxied excluded).
      const perms: Record<string, string[]> = {};
      for (const [key, val] of Object.entries(uiPermissions)) {
        const items = val.split(',').map((v) => v.trim()).filter(Boolean);
        if (items.length > 0) perms[key] = items;
      }
      _applyUiPermSync(perms, serverAccess, selectedAgents, proxiedKeys);

      const payload: UpdateGroupPayload = {
        description: formDescription.trim() || undefined,
        scope_config: {
          server_access: serverAccessPayload,
          ui_permissions: perms,
          agent_access: selectedAgents,
        },
      };

      await updateGroup(editingGroup, payload);
      onShowToast(`Group "${editingGroup}" updated successfully`, 'success');
      resetForm();
      setEditingGroup(null);
      setGroupDetail(null);
      setView('list');
      await refetch();
    } catch (err: any) {
      const detail = err.response?.data?.detail;
      const message = Array.isArray(detail)
        ? detail.map((d: any) => d.msg).join(', ')
        : detail || 'Failed to update group';
      onShowToast(message, 'error');
    } finally {
      setIsSaving(false);
    }
  };

  // ─── JSON upload sync ─────────────────────────────────────────

  const parseJsonContent = (content: string) => {
    try {
      const parsed = JSON.parse(content);

      // Sync all form fields from uploaded JSON
      if (parsed.scope_name) setFormName(parsed.scope_name);
      if (parsed.description) setFormDescription(parsed.description);
      if (parsed.create_in_idp !== undefined) setCreateInIdp(parsed.create_in_idp);

      // Group mappings (optional)
      if (Array.isArray(parsed.group_mappings)) {
        setGroupMappings(parsed.group_mappings.join(', '));
      }

      // Server access
      if (Array.isArray(parsed.server_access)) {
        const entries: ServerAccessEntry[] = parsed.server_access
          .filter((e: any) => e.server)
          .map((e: any) => ({
            server: e.server || '',
            methods: Array.isArray(e.methods) ? e.methods : [],
            tools: Array.isArray(e.tools) ? e.tools : (e.tools === '*' ? ['*'] : []),
          }));
        if (entries.length > 0) setServerAccess(entries);
      }

      // Agent access (optional)
      if (Array.isArray(parsed.agent_access)) {
        setSelectedAgents(parsed.agent_access);
      }

      // UI permissions
      if (parsed.ui_permissions && typeof parsed.ui_permissions === 'object') {
        const perms: Record<string, string> = {};
        for (const [key, val] of Object.entries(parsed.ui_permissions)) {
          perms[key] = Array.isArray(val) ? (val as string[]).join(', ') : String(val);
        }
        setUiPermissions(perms);
      }

      onShowToast('JSON loaded', 'success');
    } catch {
      onShowToast('Invalid JSON file', 'error');
    }
  };

  const handleFileUpload = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => parseJsonContent(ev.target?.result as string);
    reader.readAsText(file);
  };

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault();
    const file = e.dataTransfer.files[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (ev) => parseJsonContent(ev.target?.result as string);
    reader.readAsText(file);
  };

  const downloadExampleJson = () => {
    const blob = new Blob([JSON.stringify(EXAMPLE_SCOPE_JSON, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'example-group-scope.json';
    a.click();
    URL.revokeObjectURL(url);
  };

  // ─── Server access helpers ────────────────────────────────────

  const updateServerEntry = (idx: number, field: keyof ServerAccessEntry, value: any) => {
    setServerAccess((prev) => prev.map((e, i) => (i === idx ? { ...e, [field]: value } : e)));
  };

  // Changing the selected server can flip its value space (MCP methods <-> HTTP
  // verbs). Reset methods AND tools so stale tokens from the previous kind aren't
  // silently written into the new scope (e.g. MCP "tools/call" persisting into a
  // proxied-skill scope, which the auth-server would never match). New proxied
  // entries start with no methods (admin picks explicit verbs); MCP/virtual keep
  // the historical all-MCP-methods default.
  const handleServerChange = (idx: number, val: string) => {
    const nowProxied = proxiedKeys.has(val.trim());
    setServerAccess((prev) =>
      prev.map((e, i) =>
        i === idx
          ? { ...e, server: val, tools: [], methods: nowProxied ? [] : [...MCP_METHODS] }
          : e,
      ),
    );
  };

  const toggleMethod = (idx: number, method: string) => {
    setServerAccess((prev) =>
      prev.map((e, i) => {
        if (i !== idx) return e;
        const methods = e.methods.includes(method)
          ? e.methods.filter((m) => m !== method)
          : [...e.methods, method];
        return { ...e, methods };
      }),
    );
  };

  const addServerEntry = () => setServerAccess((prev) => [...prev, { ...EMPTY_SERVER_ENTRY }]);
  const removeServerEntry = (idx: number) => setServerAccess((prev) => prev.filter((_, i) => i !== idx));

  // ─── UI permission helpers ────────────────────────────────────

  const setPermValue = (key: string, value: string) => {
    setUiPermissions((prev) => {
      if (!value.trim()) {
        const next = { ...prev };
        delete next[key];
        return next;
      }
      return { ...prev, [key]: value };
    });
  };


  // ─── Create View ──────────────────────────────────────────────
  if (view === 'create') {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            IAM &gt; Groups &gt; Create
          </h2>
          <button
            onClick={() => { resetForm(); setView('list'); }}
            className="flex items-center text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
          >
            <ArrowLeftIcon className="h-4 w-4 mr-1" />
            Back to List
          </button>
        </div>

        {/* ── Basic Info ─────────────────────────────────────── */}
        <div className="space-y-4">
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Group Name *</label>
            <input
              type="text"
              value={formName}
              onChange={(e) => setFormName(e.target.value)}
              placeholder="e.g. currenttime-users"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                         bg-white dark:bg-gray-900 text-gray-900 dark:text-white
                         focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Description</label>
            <input
              type="text"
              value={formDescription}
              onChange={(e) => setFormDescription(e.target.value)}
              placeholder="Optional description"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                         bg-white dark:bg-gray-900 text-gray-900 dark:text-white
                         focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>
          <div>
            <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">
              Group Mappings
              <span className="text-xs text-gray-400 ml-1">(optional, comma-separated)</span>
            </label>
            <input
              type="text"
              value={groupMappings}
              onChange={(e) => setGroupMappings(e.target.value)}
              placeholder="e.g. currenttime-users, other-group"
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                         bg-white dark:bg-gray-900 text-gray-900 dark:text-white
                         focus:ring-2 focus:ring-purple-500 focus:border-transparent"
            />
          </div>
          <div className="flex items-center space-x-2">
            <input
              type="checkbox"
              checked={createInIdp}
              onChange={(e) => setCreateInIdp(e.target.checked)}
              className="rounded border-gray-300 dark:border-gray-600 text-purple-600 focus:ring-purple-500"
            />
            <label className="text-sm text-gray-600 dark:text-gray-400">
              Create in Identity Provider (Keycloak / Entra ID)
            </label>
          </div>
        </div>

  <GroupAccessPanel
    serverAccess={serverAccess}
    availableServers={availableServers}
    serversLoading={serversLoading}
    commonMethods={COMMON_METHODS}
    isProxiedServer={(s) => proxiedKeys.has(s.trim())}
    httpVerbs={HTTP_VERBS}
    destructiveHttpVerbs={DESTRUCTIVE_HTTP_VERBS}
    onAddServerEntry={addServerEntry}
    onRemoveServerEntry={removeServerEntry}
    onUpdateServerEntry={updateServerEntry}
    onServerChange={handleServerChange}
    onToggleMethod={toggleMethod}
    renderToolsSelector={(entry, idx) => (
    <ServerToolsSelector
      serverPath={entry.server}
      selectedTools={entry.tools}
      onChange={(tools) => updateServerEntry(idx, 'tools', tools)}
    />
    )}
    selectedAgents={selectedAgents}
    availableAgents={availableAgents}
    agentsLoading={agentsLoading}
    onAddAgent={(p) => setSelectedAgents((prev) => [...prev, p])}
    onRemoveAgent={(p) => setSelectedAgents((prev) => prev.filter((a) => a !== p))}
    uiPermissions={uiPermissions}
    setPermValue={setPermValue}
    entityScopeGroups={entityScopeGroups}
    skillOptions={skillOptions}
    skillsLoading={skillsLoading}
/>

        {/* ── JSON Upload / Preview ──────────────────────────── */}
        <div className="space-y-4">
          <p className="text-sm font-medium text-gray-700 dark:text-gray-300">
            Or Upload JSON Configuration
          </p>
          <div
            onDragOver={(e) => e.preventDefault()}
            onDrop={handleDrop}
            className="border-2 border-dashed border-gray-300 dark:border-gray-600 rounded-lg p-6
                       text-center hover:border-purple-400 dark:hover:border-purple-500 transition-colors"
          >
            <DocumentArrowUpIcon className="h-8 w-8 mx-auto text-gray-400 dark:text-gray-500 mb-2" />
            <p className="text-sm text-gray-500 dark:text-gray-400 mb-1">
              Drag &amp; drop a scope JSON file here
            </p>
            <label className="cursor-pointer text-sm text-purple-600 dark:text-purple-400 hover:underline">
              or click to browse
              <input type="file" accept=".json" onChange={handleFileUpload} className="hidden" />
            </label>
          </div>

          {jsonPreview && (
            <div>
              <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                JSON Preview (auto-generated from form):
              </p>
              <pre className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700
                              rounded-lg p-4 text-xs font-mono text-gray-800 dark:text-gray-200
                              overflow-auto max-h-64">
                {jsonPreview}
              </pre>
            </div>
          )}

          <button
            onClick={downloadExampleJson}
            className="flex items-center text-sm text-purple-600 dark:text-purple-400 hover:underline"
          >
            <ArrowDownTrayIcon className="h-4 w-4 mr-1" />
            Download Example JSON
          </button>
        </div>

        {/* ── Actions ────────────────────────────────────────── */}
        <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200 dark:border-gray-700">
          <button
            onClick={() => { resetForm(); setView('list'); }}
            className="px-4 py-2 text-sm text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700
                       rounded-lg hover:bg-gray-200 dark:hover:bg-gray-800"
          >
            Cancel
          </button>
          <button
            onClick={handleCreate}
            disabled={!formName.trim() || isCreating}
            className="px-4 py-2 text-sm text-white bg-purple-600 rounded-lg hover:bg-purple-700
                       disabled:opacity-50 disabled:cursor-not-allowed"
          >
            {isCreating ? 'Creating...' : 'Create Group'}
          </button>
        </div>
      </div>
    );
  }


  // ─── Edit View ───────────────────────────────────────────────
  if (view === 'edit') {
    return (
      <div className="space-y-6">
        <div className="flex items-center justify-between">
          <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
            IAM &gt; Groups &gt; Edit: {editingGroup}
          </h2>
          <button
            onClick={() => { resetForm(); setEditingGroup(null); setGroupDetail(null); setView('list'); }}
            className="flex items-center text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200"
          >
            <ArrowLeftIcon className="h-4 w-4 mr-1" />
            Back to List
          </button>
        </div>

        {isLoadingGroup && (
          <div className="flex justify-center py-12">
            <ArrowPathIcon className="h-6 w-6 text-gray-400 animate-spin" />
          </div>
        )}

        {!isLoadingGroup && (
          <>
            {/* ── Basic Info ─────────────────────────────────────── */}
            <div className="space-y-4">
              <div>
                <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Group Name</label>
                <input
                  type="text"
                  value={formName}
                  disabled
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                             bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-400
                             cursor-not-allowed"
                />
                <p className="text-xs text-gray-400 mt-1">Group name cannot be changed</p>
              </div>
              <div>
                <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">Description</label>
                <input
                  type="text"
                  value={formDescription}
                  onChange={(e) => setFormDescription(e.target.value)}
                  placeholder="Optional description"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                             bg-white dark:bg-gray-900 text-gray-900 dark:text-white
                             focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
              <div>
                <label className="block text-sm text-gray-600 dark:text-gray-400 mb-1">
                  Group Mappings
                  <span className="text-xs text-gray-400 ml-1">(optional, comma-separated)</span>
                </label>
                <input
                  type="text"
                  value={groupMappings}
                  onChange={(e) => setGroupMappings(e.target.value)}
                  placeholder="e.g. currenttime-users, other-group"
                  className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                             bg-white dark:bg-gray-900 text-gray-900 dark:text-white
                             focus:ring-2 focus:ring-purple-500 focus:border-transparent"
                />
              </div>
              {/* Create-in-IdP toggle, locked in edit mode (issue #946). */}
              <div className="flex flex-col space-y-1">
                <label className="flex items-center space-x-2">
                  <input
                    type="checkbox"
                    checked={createInIdp}
                    disabled
                    aria-describedby="create-in-idp-edit-help"
                    className="rounded border-gray-300 dark:border-gray-600 text-purple-600
                               focus:ring-purple-500 disabled:opacity-50 disabled:cursor-not-allowed"
                  />
                  <span className="text-sm text-gray-600 dark:text-gray-400">
                    Create in Identity Provider (Keycloak / Entra ID)
                  </span>
                </label>
                <p id="create-in-idp-edit-help" className="text-xs text-gray-500 dark:text-gray-400 pl-6">
                  This setting cannot be changed after creation. To convert a group between
                  local-only and IdP-managed, delete and recreate it.
                </p>
              </div>
            </div>

  <GroupAccessPanel
    serverAccess={serverAccess}
    availableServers={availableServers}
    serversLoading={serversLoading}
    commonMethods={COMMON_METHODS}
    isProxiedServer={(s) => proxiedKeys.has(s.trim())}
    httpVerbs={HTTP_VERBS}
    destructiveHttpVerbs={DESTRUCTIVE_HTTP_VERBS}
    onAddServerEntry={addServerEntry}
    onRemoveServerEntry={removeServerEntry}
    onUpdateServerEntry={updateServerEntry}
    onServerChange={handleServerChange}
    onToggleMethod={toggleMethod}
    renderToolsSelector={(entry, idx) => (
      <ServerToolsSelector
        serverPath={entry.server}
        selectedTools={entry.tools}
        onChange={(tools) => updateServerEntry(idx, 'tools', tools)}
      />
    )}
    selectedAgents={selectedAgents}
    availableAgents={availableAgents}
    agentsLoading={agentsLoading}
    onAddAgent={(p) => setSelectedAgents((prev) => [...prev, p])}
    onRemoveAgent={(p) => setSelectedAgents((prev) => prev.filter((a) => a !== p))}
    uiPermissions={uiPermissions}
    setPermValue={setPermValue}
    entityScopeGroups={entityScopeGroups}
    skillOptions={skillOptions}
    skillsLoading={skillsLoading}
/>

            {/* ── JSON Preview ──────────────────────────────────────── */}
            {jsonPreview && (
              <div className="space-y-4">
                <div>
                  <p className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                    JSON Preview (auto-generated from form):
                  </p>
                  <pre className="bg-gray-50 dark:bg-gray-900 border border-gray-200 dark:border-gray-700
                                  rounded-lg p-4 text-xs font-mono text-gray-800 dark:text-gray-200
                                  overflow-auto max-h-64">
                    {jsonPreview}
                  </pre>
                </div>
              </div>
            )}

            {/* ── Actions ────────────────────────────────────────── */}
            <div className="flex justify-end space-x-3 pt-4 border-t border-gray-200 dark:border-gray-700">
              <button
                onClick={() => { resetForm(); setEditingGroup(null); setGroupDetail(null); setView('list'); }}
                className="px-4 py-2 text-sm text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700
                           rounded-lg hover:bg-gray-200 dark:hover:bg-gray-800"
              >
                Cancel
              </button>
              <button
                onClick={handleUpdate}
                disabled={isSaving}
                className="px-4 py-2 text-sm text-white bg-purple-600 rounded-lg hover:bg-purple-700
                           disabled:opacity-50 disabled:cursor-not-allowed"
              >
                {isSaving ? 'Saving...' : 'Save Changes'}
              </button>
            </div>
          </>
        )}
      </div>
    );
  }


  // ─── List View ────────────────────────────────────────────────
  return (
    <div className="space-y-4">
      <div className="flex items-center justify-between">
        <h2 className="text-lg font-semibold text-gray-900 dark:text-white">
          IAM &gt; Groups
        </h2>
        <div className="flex items-center space-x-2">
          <button onClick={refetch} className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200" title="Refresh">
            <ArrowPathIcon className="h-5 w-5" />
          </button>
          <button
            onClick={() => setView('create')}
            className="flex items-center px-3 py-2 text-sm text-white bg-purple-600 rounded-lg hover:bg-purple-700"
          >
            <PlusIcon className="h-4 w-4 mr-1" />
            Create Group
          </button>
        </div>
      </div>

      {/* Search */}
      <div className="relative">
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search groups..."
          className="w-full pl-10 pr-4 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                     bg-white dark:bg-gray-900 text-gray-900 dark:text-white text-sm
                     focus:ring-2 focus:ring-purple-500 focus:border-transparent"
        />
      </div>

      <ListStateBoundary
        isLoading={isLoading}
        error={error}
        isEmpty={filteredGroups.length === 0}
        emptyMessage={
          searchQuery ? 'No groups match your search.' : 'No groups yet. Create your first group.'
        }
      >
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="border-b border-gray-200 dark:border-gray-700">
                <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Name</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Description</th>
                <th className="text-left py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Path</th>
                <th className="text-right py-3 px-4 font-medium text-gray-500 dark:text-gray-400">Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredGroups.map((group) => (
                <React.Fragment key={group.name}>
                  <tr className="table-row border-b border-gray-100 dark:border-gray-800">
                    <td className="py-3 px-4 text-gray-900 dark:text-white font-medium">
                      <span>{group.name}</span>
                      {group.is_idp_managed === false ? (
                        <span
                          role="status"
                          aria-label="Local-only group: PATCH and DELETE will not call the upstream identity provider. Recommended for tenants without Group.ReadWrite.All."
                          title="Local-only: PATCH and DELETE will not call the upstream IdP. Recommended for tenants without Group.ReadWrite.All."
                          className="ml-2 inline-flex items-center px-2 py-0.5 text-xs rounded-full bg-amber-100 text-amber-800 dark:bg-amber-900/30 dark:text-amber-300"
                        >
                          Local-only
                        </span>
                      ) : group.is_idp_managed === true ? (
                        <span
                          role="status"
                          aria-label="IdP-managed group: PATCH and DELETE will call the upstream identity provider."
                          title="IdP-managed: this group is managed in the upstream identity provider."
                          className="ml-2 inline-flex items-center px-2 py-0.5 text-xs rounded-full bg-blue-100 text-blue-800 dark:bg-blue-900/30 dark:text-blue-300"
                        >
                          IdP-managed
                        </span>
                      ) : null}
                    </td>
                    <td className="py-3 px-4 text-gray-600 dark:text-gray-400">{group.description || '\u2014'}</td>
                    <td className="py-3 px-4 text-gray-500 dark:text-gray-500 font-mono text-xs">{group.path || '\u2014'}</td>
                    <td className="py-3 px-4 text-right">
                      <button
                        onClick={() => handleEditClick(group.name)}
                        className="p-1 text-gray-400 hover:text-purple-500 dark:hover:text-purple-400 mr-1"
                        title="Edit group"
                        disabled={isLoadingGroup && editingGroup === group.name}
                      >
                        {isLoadingGroup && editingGroup === group.name ? (
                          <ArrowPathIcon className="h-4 w-4 animate-spin" />
                        ) : (
                          <PencilIcon className="h-4 w-4" />
                        )}
                      </button>
                      <button
                        onClick={() => setDeleteTarget(group.name)}
                        className="p-1 text-gray-400 hover:text-red-500 dark:hover:text-red-400"
                        title="Delete group"
                      >
                        <TrashIcon className="h-4 w-4" />
                      </button>
                    </td>
                  </tr>
                  {deleteTarget === group.name && (
                    <tr>
                      <td colSpan={4} className="p-2">
                        <DeleteConfirmation
                          entityType="group"
                          entityName={group.name}
                          entityPath={group.name}
                          onConfirm={handleDelete}
                          onCancel={() => setDeleteTarget(null)}
                        />
                      </td>
                    </tr>
                  )}
                </React.Fragment>
              ))}
            </tbody>
          </table>
        </div>
      </ListStateBoundary>
    </div>
  );
};

export default IAMGroups;
