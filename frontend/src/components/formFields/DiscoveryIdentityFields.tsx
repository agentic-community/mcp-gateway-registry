import React from 'react';
import { FIELD, LABEL } from './formClasses';

/**
 * The subset of server-form state this block reads/writes. Both the register
 * form (`ServerFormData`) and the edit modal (`ServerEditForm`) contain these
 * keys, so either can be passed as `values` (structural typing).
 */
export interface DiscoveryIdentityValues {
  oauth_discovery_provider: string;
  oauth_discovery_client_id: string;
  oauth_discovery_client_secret: string;
  oauth_discovery_scopes: string;
  oauth_discovery_custom_authorize_url: string;
  oauth_discovery_custom_token_url: string;
  oauth_discovery_custom_scope_separator: string;
  oauth_discovery_custom_token_auth_style: string;
  oauth_discovery_custom_resource: string;
}

interface Props {
  values: DiscoveryIdentityValues;
  onChange: (patch: Partial<DiscoveryIdentityValues>) => void;
  /** Edit mode: the secret field shows a "leave blank to keep current" hint. */
  editing?: boolean;
  /**
   * Optional trailing content rendered inside the block — the edit modal uses
   * this for the guarded "Connect account for discovery" button (the consent
   * flow needs a persisted server, so the register form omits it and the user
   * connects afterwards from Edit).
   */
  footer?: React.ReactNode;
}

/**
 * Discovery Identity (OAuth 2.1) — a backend-auth identity for OAuth 2.1 servers
 * with no machine grant: the registry borrows a designated user's connected
 * account for its OWN headless health checks / tool discovery. Shown when the
 * independent `oauth_discovery_enabled` flag is on (orthogonal to the Backend
 * Authentication scheme). Requires the egress-auth feature, which the caller
 * gates on. Persisted via `PUT/DELETE /servers/{path}/oauth-discovery`. Shared
 * by the register form and the server edit modal so the two never drift.
 */
const DiscoveryIdentityFields: React.FC<Props> = ({ values, onChange, editing, footer }) => (
  <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
    <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
      Discovery Identity (OAuth 2.1)
    </h4>
    <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
      For OAuth 2.1 servers with no machine grant, the registry borrows THIS
      admin&apos;s connected account for its own headless health checks and tool
      discovery. Discovery reflects that account&apos;s visibility and pauses if the
      token expires. Saving designates you.
    </p>
    <div className="space-y-3">
      <div>
        <label className={LABEL}>Provider</label>
        <select
          value={values.oauth_discovery_provider}
          onChange={(e) => onChange({ oauth_discovery_provider: e.target.value })}
          className={FIELD}
        >
          <option value="">Select…</option>
          <option value="github">GitHub</option>
          <option value="google">Google</option>
          <option value="atlassian">Atlassian</option>
          <option value="microsoft">Microsoft</option>
          <option value="slack">Slack</option>
          <option value="custom">Custom (OAuth 2.1)</option>
        </select>
      </div>
      {values.oauth_discovery_provider && (
        <>
          <div>
            <label className={LABEL}>Client ID</label>
            <input
              type="text"
              value={values.oauth_discovery_client_id}
              onChange={(e) => onChange({ oauth_discovery_client_id: e.target.value })}
              className={FIELD}
            />
          </div>
          {/* A public client (custom provider, token auth 'none') has no secret
              by design — hide the field so nothing stale is sent. */}
          {!(
            values.oauth_discovery_provider === 'custom' &&
            values.oauth_discovery_custom_token_auth_style === 'none'
          ) && (
            <div>
              <label className={LABEL}>Client Secret</label>
              <input
                type="password"
                value={values.oauth_discovery_client_secret}
                onChange={(e) => onChange({ oauth_discovery_client_secret: e.target.value })}
                placeholder={editing ? 'leave blank to keep current' : ''}
                autoComplete="new-password"
                className={FIELD}
              />
            </div>
          )}
          <div>
            <label className={LABEL}>Scopes</label>
            <input
              type="text"
              value={values.oauth_discovery_scopes}
              onChange={(e) => onChange({ oauth_discovery_scopes: e.target.value })}
              placeholder="repo, read:user"
              className={FIELD}
            />
          </div>
          {values.oauth_discovery_provider === 'custom' && (
            <>
              <div>
                <label className={LABEL}>Authorize URL</label>
                <input
                  type="text"
                  value={values.oauth_discovery_custom_authorize_url}
                  onChange={(e) =>
                    onChange({ oauth_discovery_custom_authorize_url: e.target.value })
                  }
                  placeholder="https://idp.example/authorize"
                  className={FIELD}
                />
              </div>
              <div>
                <label className={LABEL}>Token URL</label>
                <input
                  type="text"
                  value={values.oauth_discovery_custom_token_url}
                  onChange={(e) =>
                    onChange({ oauth_discovery_custom_token_url: e.target.value })
                  }
                  placeholder="https://idp.example/token"
                  className={FIELD}
                />
              </div>
              <div>
                <label className={LABEL}>Scope Separator</label>
                <input
                  type="text"
                  value={values.oauth_discovery_custom_scope_separator}
                  onChange={(e) =>
                    onChange({ oauth_discovery_custom_scope_separator: e.target.value })
                  }
                  placeholder="space (default) or comma"
                  className={FIELD}
                />
              </div>
              <div>
                <label className={LABEL}>Token Endpoint Authentication</label>
                <select
                  value={values.oauth_discovery_custom_token_auth_style || 'post_body'}
                  onChange={(e) =>
                    onChange({ oauth_discovery_custom_token_auth_style: e.target.value })
                  }
                  className={FIELD}
                >
                  <option value="post_body">Client secret in POST body</option>
                  <option value="basic_header">Client secret in Basic header</option>
                  <option value="none">None — public client (PKCE only)</option>
                </select>
              </div>
              <div>
                <label className={LABEL}>Resource (RFC 8707, optional)</label>
                <input
                  type="text"
                  value={values.oauth_discovery_custom_resource}
                  onChange={(e) =>
                    onChange({ oauth_discovery_custom_resource: e.target.value })
                  }
                  placeholder="https://mcp.example.com/mcp"
                  className={FIELD}
                />
              </div>
            </>
          )}
        </>
      )}
      {footer}
    </div>
  </div>
);

export default DiscoveryIdentityFields;
