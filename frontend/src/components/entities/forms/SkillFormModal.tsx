import React from 'react';
import {
  StatusField,
  MetadataField,
  ProxyField,
  FIELD_BASE,
  FIELD_FOCUS,
  LABEL,
} from '../../formFields';
import Button from '../../Button';

// Skill forms use an amber focus accent.
const FIELD = `${FIELD_BASE} ${FIELD_FOCUS.amber}`;

/**
 * Shape of the skill form state, owned by the Dashboard. This modal handles
 * BOTH create and edit (mode is derived from `editing`), so it's named
 * SkillFormModal rather than SkillEditModal.
 */
export interface SkillForm {
  name: string;
  description: string;
  skill_md_url: string;
  repository_url: string;
  version: string;
  visibility: 'public' | 'private' | 'group';
  tags: string;
  target_agents: string;
  metadata: string;
  status: 'active' | 'draft' | 'deprecated' | 'beta';
  auth_scheme: 'none' | 'global_credentials' | 'bearer' | 'api_key';
  auth_credential: string;
  auth_header_name: string;
  // Gateway-proxy opt-in. A skill has no native backend URL, so proxy_target_url
  // is required when is_proxied is true.
  is_proxied: boolean;
  proxy_target_url: string;
}

interface SkillFormModalProps {
  /** When set, the modal is in edit mode (name locked, path shown, labels change). */
  editing: { name: string; path: string } | null;
  form: SkillForm;
  setForm: React.Dispatch<React.SetStateAction<SkillForm>>;
  loading: boolean;
  /** Auto-fill-from-SKILL.md toggle (create mode only). */
  autoFill: boolean;
  setAutoFill: (next: boolean) => void;
  /** True while a SKILL.md parse request is in flight. */
  parseLoading: boolean;
  onParse: () => void;
  onSubmit: (e: React.FormEvent) => void;
  onClose: () => void;
}

/**
 * Create/edit form for an Agent Skill. A single modal covers both modes — create
 * (with SKILL.md auto-parse) and edit (name locked, path shown) — gated on the
 * `editing` prop. Extracted from the inline Dashboard block; the Dashboard keeps
 * the form state, parse/save handlers, and duplicate-check pre-flight.
 */
const SkillFormModal: React.FC<SkillFormModalProps> = ({
  editing,
  form,
  setForm,
  loading,
  autoFill,
  setAutoFill,
  parseLoading,
  onParse,
  onSubmit,
  onClose,
}) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-md max-h-[90vh] overflow-y-auto">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          {editing ? `Edit Skill: ${editing.name}` : 'Register New Skill'}
        </h3>

        <form onSubmit={onSubmit} className="space-y-4">
          {/* Auto-fill toggle - only for new skills */}
          {!editing && (
            <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
              <div>
                <span className="text-sm font-medium text-gray-700 dark:text-gray-200">
                  Auto-fill from SKILL.md
                </span>
                <p className="text-xs text-gray-500 dark:text-gray-400">
                  Parse name and description from the SKILL.md file
                </p>
              </div>
              <button
                type="button"
                onClick={() => setAutoFill(!autoFill)}
                className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                  autoFill ? 'bg-amber-600' : 'bg-gray-300 dark:bg-gray-600'
                }`}
              >
                <span
                  className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                    autoFill ? 'translate-x-6' : 'translate-x-1'
                  }`}
                />
              </button>
            </div>
          )}

          {/* SKILL.md URL with Parse button */}
          <div>
            <label className={LABEL}>SKILL.md URL *</label>
            <div className="flex space-x-2">
              <input
                type="url"
                value={form.skill_md_url}
                onChange={(e) => setForm((prev) => ({ ...prev, skill_md_url: e.target.value }))}
                className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-amber-500 focus:border-amber-500"
                placeholder="https://raw.githubusercontent.com/org/repo/main/SKILL.md"
                required
              />
              {autoFill && !editing && (
                <button
                  type="button"
                  onClick={onParse}
                  disabled={!form.skill_md_url || parseLoading}
                  className="px-3 py-2 text-sm font-medium text-white bg-amber-600 hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-md transition-colors whitespace-nowrap"
                >
                  {parseLoading ? 'Parsing...' : 'Parse'}
                </button>
              )}
            </div>
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Use raw content URL (e.g., raw.githubusercontent.com)
            </p>
          </div>

          {/* Name field */}
          <div>
            <label className={LABEL}>Skill Name *</label>
            <input
              type="text"
              value={form.name}
              onChange={(e) => {
                const formatted = e.target.value
                  .toLowerCase()
                  .replace(/[^a-z0-9-]/g, '-')
                  .replace(/-+/g, '-')
                  .replace(/^-|-$/g, '');
                setForm((prev) => ({ ...prev, name: formatted }));
              }}
              className={FIELD}
              placeholder="my-skill-name"
              pattern="^[a-z0-9]+(-[a-z0-9]+)*$"
              title="Lowercase alphanumeric with hyphens (e.g., my-skill-name)"
              required
              disabled={!!editing}
            />
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Lowercase letters, numbers, and hyphens only
            </p>
          </div>

          {/* Description field */}
          <div>
            <label className={LABEL}>Description *</label>
            <textarea
              value={form.description}
              onChange={(e) => setForm((prev) => ({ ...prev, description: e.target.value }))}
              className={FIELD}
              rows={3}
              placeholder="Describe what this skill does and when to use it"
              required
            />
          </div>

          {/* Repository URL */}
          <div>
            <label className={LABEL}>Repository URL (optional)</label>
            <input
              type="url"
              value={form.repository_url}
              onChange={(e) => setForm((prev) => ({ ...prev, repository_url: e.target.value }))}
              className={FIELD}
              placeholder="https://github.com/org/repo"
            />
          </div>

          {/* Version field */}
          <div>
            <label className={LABEL}>Version (optional)</label>
            <input
              type="text"
              value={form.version}
              onChange={(e) => setForm((prev) => ({ ...prev, version: e.target.value }))}
              className={FIELD}
              placeholder="1.0.0"
            />
          </div>

          <div>
            <label className={LABEL}>Visibility</label>
            <select
              value={form.visibility}
              onChange={(e) =>
                setForm((prev) => ({
                  ...prev,
                  visibility: e.target.value as 'public' | 'private' | 'group',
                }))
              }
              className={FIELD}
            >
              <option value="public">Public</option>
              <option value="private">Private</option>
              <option value="group">Group</option>
            </select>
          </div>

          <StatusField
            value={form.status}
            accent="amber"
            onChange={(status) => setForm((prev) => ({ ...prev, status }))}
          />

          <ProxyField
            isProxied={form.is_proxied}
            onIsProxiedChange={(v) => setForm((prev) => ({ ...prev, is_proxied: v }))}
            proxyTargetUrl={form.proxy_target_url}
            onProxyTargetUrlChange={(v) => setForm((prev) => ({ ...prev, proxy_target_url: v }))}
            targetRequired
            accent="amber"
          />

          <div>
            <label className={LABEL}>Tags</label>
            <input
              type="text"
              value={form.tags}
              onChange={(e) => setForm((prev) => ({ ...prev, tags: e.target.value }))}
              className={FIELD}
              placeholder="automation, productivity, code-review"
            />
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Comma-separated tags for categorization
            </p>
          </div>

          <div>
            <label className={LABEL}>Target Agents</label>
            <input
              type="text"
              value={form.target_agents}
              onChange={(e) => setForm((prev) => ({ ...prev, target_agents: e.target.value }))}
              className={FIELD}
              placeholder="claude-code, cursor, windsurf"
            />
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Comma-separated list of compatible coding assistants
            </p>
          </div>

          {/* Source Authentication */}
          <div className="border-t border-gray-200 dark:border-gray-600 pt-4 mt-4">
            <label className={LABEL}>Source Authentication</label>
            <select
              value={form.auth_scheme}
              onChange={(e) => {
                const newScheme = e.target.value as
                  | 'none'
                  | 'global_credentials'
                  | 'bearer'
                  | 'api_key';
                setForm((prev) => ({
                  ...prev,
                  auth_scheme: newScheme,
                  auth_credential:
                    newScheme === 'none' || newScheme === 'global_credentials'
                      ? ''
                      : prev.auth_credential,
                  auth_header_name: newScheme === 'api_key' ? prev.auth_header_name : '',
                }));
              }}
              className={FIELD}
            >
              <option value="none">None (public repo, no auth)</option>
              <option value="global_credentials">Use global credentials (registry PAT)</option>
              <option value="bearer">Bearer token (per-skill)</option>
              <option value="api_key">API key (per-skill, custom header)</option>
            </select>
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              How the registry authenticates when fetching SKILL.md from the source
            </p>
            {autoFill && form.auth_scheme === 'global_credentials' && form.skill_md_url && (
              <button
                type="button"
                onClick={onParse}
                disabled={parseLoading}
                className="mt-2 px-3 py-1.5 text-sm font-medium text-white bg-amber-600 hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-md transition-colors"
              >
                {parseLoading ? 'Parsing...' : 'Re-parse with global credentials'}
              </button>
            )}
          </div>

          {(form.auth_scheme === 'bearer' || form.auth_scheme === 'api_key') && (
            <div>
              <label className={LABEL}>
                {form.auth_scheme === 'bearer' ? 'Bearer Token' : 'API Key'} *
              </label>
              <div className="flex space-x-2">
                <input
                  type="password"
                  value={form.auth_credential}
                  onChange={(e) =>
                    setForm((prev) => ({ ...prev, auth_credential: e.target.value }))
                  }
                  className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-amber-500 focus:border-amber-500"
                  placeholder={
                    editing
                      ? 'Leave blank to keep existing credential'
                      : form.auth_scheme === 'bearer'
                        ? 'Enter bearer token (e.g., ghp_...)'
                        : 'Enter API key'
                  }
                />
                {autoFill && form.skill_md_url && form.auth_credential && (
                  <button
                    type="button"
                    onClick={onParse}
                    disabled={parseLoading}
                    className="px-3 py-2 text-sm font-medium text-white bg-amber-600 hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-md transition-colors whitespace-nowrap"
                  >
                    {parseLoading ? 'Parsing...' : 'Re-parse'}
                  </button>
                )}
              </div>
              <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                Encrypted before storage. Never displayed after saving.
              </p>
            </div>
          )}

          {form.auth_scheme === 'api_key' && (
            <div>
              <label className={LABEL}>Header Name</label>
              <input
                type="text"
                value={form.auth_header_name}
                onChange={(e) =>
                  setForm((prev) => ({ ...prev, auth_header_name: e.target.value }))
                }
                className={FIELD}
                placeholder="PRIVATE-TOKEN"
              />
              <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                HTTP header name for the API key (default: PRIVATE-TOKEN)
              </p>
            </div>
          )}

          <MetadataField
            value={form.metadata}
            accent="amber"
            hint="Key-value pairs in JSON format for searchable custom metadata"
            placeholder='{"category": "data-processing", "framework": "langchain"}'
            onChange={(metadata) => setForm((prev) => ({ ...prev, metadata }))}
          />

          {editing && (
            <div>
              <label className={LABEL}>Path (read-only)</label>
              <input
                type="text"
                value={editing.path}
                className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-300"
                disabled
              />
            </div>
          )}

          <div className="flex space-x-3 pt-4">
            <button
              type="submit"
              disabled={loading}
              className="flex-1 px-4 py-2 text-sm font-medium text-white bg-amber-600 hover:bg-amber-700 disabled:opacity-50 rounded-md transition-colors"
            >
              {loading
                ? editing
                  ? 'Saving...'
                  : 'Registering & Scanning...'
                : editing
                  ? 'Save Changes'
                  : 'Register Skill'}
            </button>
            <Button variant="secondary" onClick={onClose} fullWidth>
              Cancel
            </Button>
          </div>
          {!editing && (
            <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 text-center">
              Registration includes a security scan and may take a few seconds
            </p>
          )}
        </form>
      </div>
    </div>
  );
};

export default SkillFormModal;
