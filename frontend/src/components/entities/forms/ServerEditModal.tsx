import React from 'react';
import LocalRuntimeFormPanel from '../../LocalRuntimeFormPanel';
import type { LocalRuntimeFormData } from '../../../utils/localRuntime';
import {
  FormField,
  TagsField,
  StatusField,
  MetadataField,
  AuthSchemeFields,
  type AuthScheme,
  FIELD,
  LABEL,
} from '../../formFields';
import Button from '../../Button';

/**
 * Shape of the server edit form state. Owned by the Dashboard (kept as state
 * there); this modal is a controlled, presentational view over it.
 */
export interface ServerEditForm {
  name: string;
  path: string;
  proxyPass: string;
  description: string;
  tags: string[];
  license: string;
  num_tools: number;
  mcp_endpoint: string;
  metadata: string;
  auth_scheme: string;
  auth_credential: string;
  auth_header_name: string;
  status: 'active' | 'draft' | 'deprecated' | 'beta';
  deployment: 'remote' | 'local';
  local_runtime: LocalRuntimeFormData;
  custom_headers: Array<{ name: string; value: string }>;
  // Per-user egress credential vault (admin config). egress_provider empty == off.
  egress_provider: string;
  egress_client_id: string;
  egress_client_secret: string; // write-only; blank on edit keeps the stored one
  egress_scopes: string; // comma/space separated
  egress_custom_authorize_url: string;
  egress_custom_token_url: string;
}

interface ServerEditModalProps {
  /** Server display name shown in the header. */
  serverName: string;
  form: ServerEditForm;
  setForm: React.Dispatch<React.SetStateAction<ServerEditForm>>;
  loading: boolean;
  /** Whether the per-user egress credential vault feature is enabled (gates the egress section). */
  egressEnabled: boolean;
  onSave: () => Promise<void> | void;
  onClose: () => void;
}

/**
 * Edit form for a registered MCP server. Controlled by the Dashboard's editForm
 * state via setForm; all server-edit fields (name, deployment, auth, custom
 * headers, local-runtime panel) live here instead of inline in Dashboard.
 */
const ServerEditModal: React.FC<ServerEditModalProps> = ({
  serverName,
  form,
  setForm,
  loading,
  egressEnabled,
  onSave,
  onClose,
}) => {
  return (
    <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
      <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-2xl max-h-[90vh] overflow-y-auto">
        <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
          Edit Server: {serverName}
        </h3>

        <form
          onSubmit={async (e) => {
            e.preventDefault();
            await onSave();
          }}
          className="space-y-4"
        >
          <div>
            <label className={LABEL}>Server Name *</label>
            <input
              type="text"
              value={form.name}
              onChange={(e) => setForm((prev) => ({ ...prev, name: e.target.value }))}
              className={FIELD}
              required
            />
          </div>

          {/* Deployment type indicator (read-only — switching types is unusual
              enough to require re-registration). */}
          <div>
            <label className={LABEL}>Deployment</label>
            <div className="px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-50 dark:bg-gray-900 text-gray-700 dark:text-gray-300">
              {form.deployment === 'local' ? 'Local (stdio)' : 'Remote (HTTP)'}
            </div>
          </div>

          {form.deployment === 'remote' && (
            <div>
              <label className={LABEL}>Proxy Pass URL *</label>
              <input
                type="url"
                value={form.proxyPass}
                onChange={(e) => setForm((prev) => ({ ...prev, proxyPass: e.target.value }))}
                className={FIELD}
                placeholder="http://localhost:8080"
                required
              />
            </div>
          )}

          {form.deployment === 'local' && (
            <LocalRuntimeFormPanel
              runtime={form.local_runtime}
              onChange={(next) => setForm((prev) => ({ ...prev, local_runtime: next }))}
            />
          )}

          <div>
            <label className={LABEL}>Description</label>
            <textarea
              value={form.description}
              onChange={(e) => setForm((prev) => ({ ...prev, description: e.target.value }))}
              className={FIELD}
              rows={3}
              placeholder="Brief description of the server"
            />
          </div>

          <StatusField
            value={form.status}
            onChange={(status) => setForm((prev) => ({ ...prev, status }))}
          />

          <TagsField
            value={form.tags}
            onChange={(tags) => setForm((prev) => ({ ...prev, tags }))}
          />

          <div className="grid grid-cols-2 gap-4">
            <FormField label="Number of Tools">
              <input
                type="number"
                value={form.num_tools}
                onChange={(e) =>
                  setForm((prev) => ({ ...prev, num_tools: parseInt(e.target.value) || 0 }))
                }
                className={FIELD}
                min="0"
              />
            </FormField>
          </div>

          <FormField label="License">
            <input
              type="text"
              value={form.license}
              onChange={(e) => setForm((prev) => ({ ...prev, license: e.target.value }))}
              className={FIELD}
              placeholder="MIT, Apache-2.0, etc."
            />
          </FormField>

          {form.deployment === 'remote' && (
            <FormField label="MCP Endpoint (optional)">
              <input
                type="url"
                value={form.mcp_endpoint}
                onChange={(e) => setForm((prev) => ({ ...prev, mcp_endpoint: e.target.value }))}
                className={FIELD}
                placeholder="Custom MCP endpoint URL (overrides default)"
              />
            </FormField>
          )}

          <MetadataField
            value={form.metadata}
            onChange={(metadata) => setForm((prev) => ({ ...prev, metadata }))}
          />

          {/* Backend Authentication — only meaningful for remote servers.
              Local servers handle auth via env vars on the user's machine. */}
          {form.deployment === 'remote' && (
            <AuthSchemeFields
              scheme={form.auth_scheme as AuthScheme}
              credential={form.auth_credential}
              headerName={form.auth_header_name}
              editing
              onSchemeChange={(newScheme) =>
                setForm((prev) => ({
                  ...prev,
                  auth_scheme: newScheme,
                  auth_credential: newScheme === 'none' ? '' : prev.auth_credential,
                  auth_header_name:
                    newScheme === 'api_key' ? prev.auth_header_name : 'X-API-Key',
                }))
              }
              onCredentialChange={(v) =>
                setForm((prev) => ({ ...prev, auth_credential: v }))
              }
              onHeaderNameChange={(v) =>
                setForm((prev) => ({ ...prev, auth_header_name: v }))
              }
            />
          )}

          {/* Per-user egress credential vault (admin config) */}
          {egressEnabled && form.deployment !== 'local' && (
            <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
              <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
                Per-User Egress Auth (OAuth)
              </h4>
              <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
                Let each user connect their own third-party account (GitHub, Slack, …). The
                gateway injects the user&apos;s token on egress. Leave provider blank to disable.
                Register this callback URL in your OAuth app:{' '}
                <code className="text-purple-600 dark:text-purple-400">
                  {`${window.location.origin}/oauth2/egress/callback`}
                </code>
              </p>
              <div className="space-y-3">
                <div>
                  <label className={LABEL}>Provider</label>
                  <select
                    value={form.egress_provider}
                    onChange={(e) =>
                      setForm((prev) => ({ ...prev, egress_provider: e.target.value }))
                    }
                    className={FIELD}
                  >
                    <option value="">Disabled</option>
                    <option value="github">GitHub</option>
                    <option value="google">Google</option>
                    <option value="atlassian">Atlassian</option>
                    <option value="microsoft">Microsoft</option>
                    <option value="slack">Slack</option>
                    <option value="custom">Custom OIDC</option>
                  </select>
                </div>
                {form.egress_provider && (
                  <>
                    <div>
                      <label className={LABEL}>Client ID</label>
                      <input
                        type="text"
                        value={form.egress_client_id}
                        onChange={(e) =>
                          setForm((prev) => ({ ...prev, egress_client_id: e.target.value }))
                        }
                        className={FIELD}
                      />
                    </div>
                    <div>
                      <label className={LABEL}>Client Secret</label>
                      <input
                        type="password"
                        value={form.egress_client_secret}
                        onChange={(e) =>
                          setForm((prev) => ({ ...prev, egress_client_secret: e.target.value }))
                        }
                        placeholder="leave blank to keep current"
                        autoComplete="new-password"
                        className={FIELD}
                      />
                    </div>
                    <div>
                      <label className={LABEL}>Scopes</label>
                      <input
                        type="text"
                        value={form.egress_scopes}
                        onChange={(e) =>
                          setForm((prev) => ({ ...prev, egress_scopes: e.target.value }))
                        }
                        placeholder="repo, read:user"
                        className={FIELD}
                      />
                    </div>
                    {form.egress_provider === 'custom' && (
                      <>
                        <div>
                          <label className={LABEL}>Authorize URL</label>
                          <input
                            type="text"
                            value={form.egress_custom_authorize_url}
                            onChange={(e) =>
                              setForm((prev) => ({
                                ...prev,
                                egress_custom_authorize_url: e.target.value,
                              }))
                            }
                            placeholder="https://idp.example/authorize"
                            className={FIELD}
                          />
                        </div>
                        <div>
                          <label className={LABEL}>Token URL</label>
                          <input
                            type="text"
                            value={form.egress_custom_token_url}
                            onChange={(e) =>
                              setForm((prev) => ({
                                ...prev,
                                egress_custom_token_url: e.target.value,
                              }))
                            }
                            placeholder="https://idp.example/token"
                            className={FIELD}
                          />
                        </div>
                      </>
                    )}
                  </>
                )}
              </div>
            </div>
          )}

          {/* Custom Headers */}
          <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
            <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
              Additional Headers
            </h4>
            <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
              Fixed HTTP headers your MCP server requires beyond authentication. Leave value
              blank to keep existing encrypted value.
            </p>
            {form.custom_headers.map((h, idx) => (
              <div key={idx} className="flex gap-2 mb-2">
                <input
                  type="text"
                  placeholder="X-My-Header"
                  value={h.name}
                  onChange={(e) => {
                    const updated = [...form.custom_headers];
                    updated[idx] = { ...updated[idx], name: e.target.value };
                    setForm((prev) => ({ ...prev, custom_headers: updated }));
                  }}
                  className={`flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500 text-sm`}
                />
                <input
                  type="text"
                  placeholder="header value (blank = keep existing)"
                  value={h.value}
                  onChange={(e) => {
                    const updated = [...form.custom_headers];
                    updated[idx] = { ...updated[idx], value: e.target.value };
                    setForm((prev) => ({ ...prev, custom_headers: updated }));
                  }}
                  className={`flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500 text-sm`}
                />
                <button
                  type="button"
                  onClick={() => {
                    const updated = form.custom_headers.filter((_, i) => i !== idx);
                    setForm((prev) => ({ ...prev, custom_headers: updated }));
                  }}
                  className="px-3 py-2 text-sm text-red-600 hover:text-red-800 dark:text-red-400"
                >
                  Remove
                </button>
              </div>
            ))}
            {form.custom_headers.length < 10 && (
              <button
                type="button"
                onClick={() => {
                  setForm((prev) => ({
                    ...prev,
                    custom_headers: [...prev.custom_headers, { name: '', value: '' }],
                  }));
                }}
                className="text-sm text-purple-600 hover:text-purple-800 dark:text-purple-400"
              >
                + Add header
              </button>
            )}
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
            <Button variant="primary" type="submit" disabled={loading} fullWidth>
              {loading ? 'Saving...' : 'Save Changes'}
            </Button>
            <Button variant="secondary" onClick={onClose} fullWidth>
              Cancel
            </Button>
          </div>
        </form>
      </div>
    </div>
  );
};

export default ServerEditModal;
