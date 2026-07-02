import React from 'react';
import {
  FormField,
  TagsField,
  StatusField,
  VisibilityField,
  MetadataField,
  ProxyField,
  FIELD_BASE,
  FIELD_FOCUS,
  LABEL,
} from '../../formFields';

import Button from '../../Button';

// Agent forms use a cyan focus accent.
const FIELD = `${FIELD_BASE} ${FIELD_FOCUS.cyan}`;

/**
 * Shape of the agent edit form state, owned by the Dashboard. This modal is a
 * controlled, presentational view over it.
 */
export interface AgentEditForm {
  name: string;
  path: string;
  url: string;
  description: string;
  version: string;
  visibility: 'public' | 'private' | 'group-restricted';
  allowed_groups: string;
  trust_level: 'community' | 'verified' | 'trusted' | 'unverified';
  supported_protocol: 'a2a' | 'other';
  tags: string[];
  skillsJson: string;
  metadata: string;
  status: 'active' | 'draft' | 'deprecated' | 'beta';
  // Gateway-proxy opt-in. An agent has a native url, so proxy_target_url is
  // optional (falls back to the agent's url when blank).
  is_proxied: boolean;
  proxy_target_url: string;
}

interface AgentEditModalProps {
  agentName: string;
  form: AgentEditForm;
  setForm: React.Dispatch<React.SetStateAction<AgentEditForm>>;
  loading: boolean;
  /** Validation message for the skills JSON textarea (null when valid). */
  skillsJsonError: string | null;
  /** Clears the skills JSON error (called as the field is edited). */
  onSkillsJsonChange: () => void;
  onSave: () => Promise<void> | void;
  onClose: () => void;
}

/**
 * Edit form for an A2A agent. Controlled by the Dashboard's editAgentForm state.
 * Agents are created via AgentCore import, so there is no create mode here.
 */
const AgentEditModal: React.FC<AgentEditModalProps> = ({
  agentName,
  form,
  setForm,
  loading,
  skillsJsonError,
  onSkillsJsonChange,
  onSave,
  onClose,
}) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-md max-h-[90vh] overflow-y-auto">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Edit Agent: {agentName}
        </h3>

        <form
          onSubmit={async (e) => {
            e.preventDefault();
            await onSave();
          }}
          className="space-y-4"
        >
          <div>
            <label className={LABEL}>Agent Name *</label>
            <input
              type="text"
              value={form.name}
              onChange={(e) => setForm((prev) => ({ ...prev, name: e.target.value }))}
              className={FIELD}
              required
            />
          </div>

          <div>
            <label className={LABEL}>Description</label>
            <textarea
              value={form.description}
              onChange={(e) => setForm((prev) => ({ ...prev, description: e.target.value }))}
              className={FIELD}
              rows={3}
              placeholder="Brief description of the agent"
            />
          </div>

          <StatusField
            value={form.status}
            accent="cyan"
            onChange={(status) => setForm((prev) => ({ ...prev, status }))}
          />

          <ProxyField
            isProxied={form.is_proxied}
            onIsProxiedChange={(v) => setForm((prev) => ({ ...prev, is_proxied: v }))}
            proxyTargetUrl={form.proxy_target_url}
            onProxyTargetUrlChange={(v) => setForm((prev) => ({ ...prev, proxy_target_url: v }))}
            accent="cyan"
          />

          <FormField label="Version">
            <input
              type="text"
              value={form.version}
              onChange={(e) => setForm((prev) => ({ ...prev, version: e.target.value }))}
              className={FIELD}
              placeholder="1.0.0"
            />
          </FormField>

          <VisibilityField
            value={form.visibility}
            accent="cyan"
            onChange={(visibility) => setForm((prev) => ({ ...prev, visibility }))}
            allowedGroups={form.allowed_groups}
            onAllowedGroupsChange={(allowed_groups) =>
              setForm((prev) => ({ ...prev, allowed_groups }))
            }
          />

          <div>
            <label className={LABEL}>Trust Level</label>
            <select
              value={form.trust_level}
              onChange={(e) =>
                setForm((prev) => ({
                  ...prev,
                  trust_level: e.target.value as
                    | 'community'
                    | 'verified'
                    | 'trusted'
                    | 'unverified',
                }))
              }
              className={FIELD}
            >
              <option value="unverified">Unverified</option>
              <option value="community">Community</option>
              <option value="verified">Verified</option>
              <option value="trusted">Trusted</option>
            </select>
          </div>

          <div>
            <label className={LABEL}>Supported Protocol</label>
            <select
              value={form.supported_protocol}
              onChange={(e) =>
                setForm((prev) => ({
                  ...prev,
                  supported_protocol: e.target.value as 'a2a' | 'other',
                }))
              }
              className={FIELD}
            >
              <option value="a2a">A2A</option>
              <option value="other">Other</option>
            </select>
          </div>

          <TagsField
            value={form.tags}
            accent="cyan"
            onChange={(tags) => setForm((prev) => ({ ...prev, tags }))}
          />

          <MetadataField
            value={form.metadata}
            accent="cyan"
            hint="Custom key-value pairs for organization, compliance, or integration purposes"
            placeholder='{"team": "platform", "owner": "alice@example.com", "cost_center": "CC-1001"}'
            onChange={(metadata) => setForm((prev) => ({ ...prev, metadata }))}
          />

          <div>
            <label className={LABEL}>Skills (JSON array)</label>
            <textarea
              value={form.skillsJson}
              onChange={(e) => {
                setForm((prev) => ({ ...prev, skillsJson: e.target.value }));
                onSkillsJsonChange();
              }}
              className={`block w-full px-3 py-2 border rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white font-mono text-xs focus:ring-cyan-500 focus:border-cyan-500 ${
                skillsJsonError
                  ? 'border-red-500 dark:border-red-400'
                  : 'border-gray-300 dark:border-gray-600'
              }`}
              rows={8}
              placeholder='[{"id": "skill-1", "name": "My Skill", "description": "What this skill does"}]'
            />
            {skillsJsonError && (
              <p className="mt-1 text-xs text-red-600 dark:text-red-400">{skillsJsonError}</p>
            )}
            <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
              Each skill needs at least: id, name, description. Saving triggers a security rescan.
            </p>
          </div>

          <div>
            <label className={LABEL}>Path (read-only)</label>
            <input
              type="text"
              value={form.path}
              className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-300"
              disabled
            />
          </div>

          <div className="flex space-x-3 pt-4">
            <button
              type="submit"
              disabled={loading}
              className="flex-1 px-4 py-2 text-sm font-medium text-white bg-cyan-600 hover:bg-cyan-700 disabled:opacity-50 rounded-md transition-colors"
            >
              {loading ? 'Saving...' : 'Save Changes'}
            </button>
            <Button variant="secondary" onClick={onClose} fullWidth>
              Cancel
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default AgentEditModal;
