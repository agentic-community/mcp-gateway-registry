import React from 'react';
import { XMarkIcon } from '@heroicons/react/24/outline';
import SearchableSelect, { SelectOption } from '../SearchableSelect';
import CustomTypeListPicker from './CustomTypeListPicker';

/**
 * Shared UI-permission editor for the IAM group create + edit forms.
 *
 * Replaces the previous free-text comma-list inputs with structured controls so
 * every resource family is edited consistently (mirroring the server/agent
 * pickers) and the "all" vs "*" wildcard and typo footguns are eliminated:
 *
 * - MUTATION scopes (register/toggle/modify/delete_*, publish_*) are
 *   grant-all-or-nothing in practice, so they render as a CHECKBOX that writes
 *   "all" when ticked and clears the key when not.
 * - DISCOVERY scopes (list_skills, list_<type>_entity) render as an "All"
 *   toggle plus a multi-select of specific names (chips + searchable add),
 *   with an explicit All option so access can be granted proactively before any
 *   record exists.
 *
 * The component operates on the existing `uiPermissions: Record<string,string>`
 * (comma-separated values) state via `setPermValue(key, csv)`, so the surrounding
 * form's build/save/load logic is unchanged. `list_service` / `list_agents` /
 * `health_check_service` / `get_agent` are intentionally NOT edited here — they
 * are derived from the Server Access / Agents pickers.
 */

export interface EntityScopeGroup {
  typeName: string;
  displayName: string;
}

interface UiPermissionEditorProps {
  uiPermissions: Record<string, string>;
  setPermValue: (key: string, csv: string) => void;
  entityScopeGroups: EntityScopeGroup[];
  skillOptions: SelectOption[];
  skillsLoading?: boolean;
  focusColor?: string;
}

// Mutation scopes for families whose LIST/read access is derived from the
// Server Access / Agents pickers (so only mutations are edited here). Actions
// are ordered consistently across every family: create-verb, Modify, Delete,
// Toggle.
const MUTATION_GROUPS: { family: string; scopes: { key: string; label: string }[] }[] = [
  {
    family: 'MCP Servers',
    scopes: [
      { key: 'register_service', label: 'Register' },
      { key: 'modify_service', label: 'Modify' },
      { key: 'delete_service', label: 'Delete' },
      { key: 'toggle_service', label: 'Toggle' },
    ],
  },
  {
    family: 'Agents',
    scopes: [
      { key: 'publish_agent', label: 'Publish' },
      { key: 'modify_agent', label: 'Modify' },
      { key: 'delete_agent', label: 'Delete' },
      { key: 'toggle_agent', label: 'Toggle' },
    ],
  },
];

// Skill mutation scopes, rendered UNDER the Skill list-discovery control (same
// list-then-mutations layout as custom types). Same action ordering.
const SKILL_MUTATION_SCOPES: { key: string; label: string }[] = [
  { key: 'publish_skill', label: 'Publish' },
  { key: 'modify_skill', label: 'Modify' },
  { key: 'delete_skill', label: 'Delete' },
  { key: 'toggle_skill', label: 'Toggle' },
];

const ENTITY_MUTATION_ACTIONS: { action: string; verb: string }[] = [
  { action: 'create', verb: 'Create' },
  { action: 'modify', verb: 'Modify' },
  { action: 'delete', verb: 'Delete' },
];


function _csvToList(csv: string | undefined): string[] {
  return (csv || '')
    .split(',')
    .map((v) => v.trim())
    .filter(Boolean);
}


function _isGranted(csv: string | undefined): boolean {
  return _csvToList(csv).length > 0;
}


/** A grant-for-"all" checkbox for a mutation scope. */
const MutationCheckbox: React.FC<{
  scopeKey: string;
  label: string;
  checked: boolean;
  onToggle: (checked: boolean) => void;
}> = ({ scopeKey, label, checked, onToggle }) => (
  <label
    htmlFor={`perm-${scopeKey}`}
    className="flex items-center space-x-2 text-sm text-gray-700 dark:text-gray-300 cursor-pointer"
  >
    <input
      id={`perm-${scopeKey}`}
      type="checkbox"
      checked={checked}
      onChange={(e) => onToggle(e.target.checked)}
      className="h-4 w-4 rounded border-gray-300 dark:border-gray-600 text-purple-600 focus:ring-purple-500"
    />
    <span>{label}</span>
  </label>
);


/** An "All" toggle + multi-select (chips + searchable add) for a discovery scope. */
const DiscoveryMultiSelect: React.FC<{
  scopeKey: string;
  label: string;
  value: string;
  options: SelectOption[];
  isLoading?: boolean;
  focusColor?: string;
  onChange: (csv: string) => void;
}> = ({ scopeKey, label, value, options, isLoading, focusColor, onChange }) => {
  const items = _csvToList(value);
  const grantsAll = items.includes('all');
  const named = items.filter((v) => v !== 'all');

  const setAll = (checked: boolean) => {
    // "All" is exclusive: granting it supersedes any named entries.
    onChange(checked ? 'all' : named.join(', '));
  };
  const addNamed = (name: string) => {
    if (!name || named.includes(name)) return;
    onChange([...named, name].join(', '));
  };
  const removeNamed = (name: string) => {
    onChange(named.filter((n) => n !== name).join(', '));
  };

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="block text-xs font-medium text-gray-600 dark:text-gray-300">{label}</label>
        <label
          htmlFor={`all-${scopeKey}`}
          className="flex items-center space-x-1.5 text-xs text-gray-500 dark:text-gray-400 cursor-pointer"
        >
          <input
            id={`all-${scopeKey}`}
            type="checkbox"
            checked={grantsAll}
            onChange={(e) => setAll(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-gray-300 dark:border-gray-600 text-purple-600 focus:ring-purple-500"
          />
          <span>All</span>
        </label>
      </div>
      {!grantsAll && (
        <>
          {named.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {named.map((name) => (
                <span
                  key={name}
                  className="inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded-full
                             bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-200"
                >
                  {name}
                  <button
                    type="button"
                    onClick={() => removeNamed(name)}
                    className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
                    aria-label={`Remove ${name}`}
                  >
                    <XMarkIcon className="h-3 w-3" />
                  </button>
                </span>
              ))}
            </div>
          )}
          <SearchableSelect
            options={options.filter((o) => !named.includes(o.value))}
            value=""
            onChange={addNamed}
            placeholder={
              options.length > 0 ? 'Add specific…' : 'Type a name and press Enter…'
            }
            isLoading={isLoading}
            allowCustom
            focusColor={focusColor}
          />
          {options.length === 0 && (
            <p className="text-[11px] text-gray-400">
              No records to pick from yet — type a specific name and press Enter to grant it,
              or use “All”.
            </p>
          )}
        </>
      )}
    </div>
  );
};


const UiPermissionEditor: React.FC<UiPermissionEditorProps> = ({
  uiPermissions,
  setPermValue,
  entityScopeGroups,
  skillOptions,
  skillsLoading,
  focusColor = 'purple',
}) => {
  const toggleMutation = (key: string, checked: boolean) =>
    setPermValue(key, checked ? 'all' : '');

  return (
    <div className="space-y-5 pl-6">
      <p className="text-xs text-gray-500 dark:text-gray-400">
        List/read access for MCP Servers and Agents is set from the Server Access and
        Agents pickers above. Grant the remaining permissions here.
      </p>

      {/* Mutation scopes — grant-for-all checkboxes, grouped by family. */}
      {MUTATION_GROUPS.map((group) => (
        <div key={group.family} className="space-y-2">
          <p className="text-xs font-semibold text-gray-600 dark:text-gray-300 uppercase tracking-wide">
            {group.family}
          </p>
          <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
            {group.scopes.map(({ key, label }) => (
              <MutationCheckbox
                key={key}
                scopeKey={key}
                label={label}
                checked={_isGranted(uiPermissions[key])}
                onToggle={(checked) => toggleMutation(key, checked)}
              />
            ))}
          </div>
        </div>
      ))}

      {/* Skills — list-discovery dropdown ABOVE the mutation checkboxes, same
          layout as the custom types below. */}
      <div className="space-y-2">
        <p className="text-xs font-semibold text-gray-600 dark:text-gray-300 uppercase tracking-wide">
          Skills
        </p>
        <DiscoveryMultiSelect
          scopeKey="list_skills"
          label="List Skills"
          value={uiPermissions['list_skills'] || ''}
          options={skillOptions}
          isLoading={skillsLoading}
          focusColor={focusColor}
          onChange={(csv) => setPermValue('list_skills', csv)}
        />
        <div className="grid grid-cols-2 md:grid-cols-4 gap-2">
          {SKILL_MUTATION_SCOPES.map(({ key, label }) => (
            <MutationCheckbox
              key={key}
              scopeKey={key}
              label={label}
              checked={_isGranted(uiPermissions[key])}
              onToggle={(checked) => toggleMutation(key, checked)}
            />
          ))}
        </div>
      </div>

      {/* Custom entity types — list discovery (All + names) + mutation checkboxes. */}
      {entityScopeGroups.map((group) => {
        const listKey = `list_${group.typeName}_entity`;
        return (
          <div key={group.typeName} className="space-y-2">
            <p className="text-xs font-semibold text-gray-600 dark:text-gray-300 uppercase tracking-wide">
              Custom Type: {group.displayName}
            </p>
            {/* list_<type>_entity is per-record aware: an "All" toggle plus a
                picker of the type's actual records (name shown, path stored),
                so an admin can grant the whole type or specific records. */}
            <CustomTypeListPicker
              typeName={group.typeName}
              displayName={group.displayName}
              value={uiPermissions[listKey] || ''}
              onChange={(csv) => setPermValue(listKey, csv)}
            />
            <div className="grid grid-cols-3 gap-2">
              {ENTITY_MUTATION_ACTIONS.map(({ action, verb }) => {
                const key = `${action}_${group.typeName}_entity`;
                return (
                  <MutationCheckbox
                    key={key}
                    scopeKey={key}
                    label={verb}
                    checked={_isGranted(uiPermissions[key])}
                    onToggle={(checked) => toggleMutation(key, checked)}
                  />
                );
              })}
            </div>
          </div>
        );
      })}
    </div>
  );
};

export default UiPermissionEditor;
