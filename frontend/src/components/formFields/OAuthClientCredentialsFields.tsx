import React from 'react';
import { FIELD, LABEL } from './formClasses';

/**
 * The subset of server-form state this block reads/writes. Both the register
 * form (`ServerFormData`) and the edit modal (`ServerEditForm`) contain these
 * keys, so either can be passed as `values` (structural typing).
 */
export interface OAuthClientCredentialsValues {
  oauth_token_url: string;
  oauth_client_id: string;
  oauth_client_secret: string;
  oauth_token_auth_style: string;
  oauth_scopes: string;
  oauth_resource: string;
}

interface Props {
  values: OAuthClientCredentialsValues;
  onChange: (patch: Partial<OAuthClientCredentialsValues>) => void;
  /** Edit mode: the secret field shows a "leave blank to keep current" hint. */
  editing?: boolean;
}

/**
 * Backend OAuth 2.0 (client_credentials) config, shown when the Backend
 * Authentication scheme is `oauth`. The registry acquires a token from these
 * and injects it as `Authorization: Bearer` on its OWN health checks and
 * tool-list fetches — works in registry-only mode (no gateway). Persisted via
 * `PUT /servers/{path}/oauth-config`. Shared by the register form and the
 * server edit modal so the two never drift.
 */
const OAuthClientCredentialsFields: React.FC<Props> = ({ values, onChange, editing }) => (
  <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
    <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-2">
      OAuth Client Credentials
    </h4>
    <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
      The registry authenticates to this server as an OAuth2 client
      (client_credentials grant) for health checks and tool discovery.
    </p>
    <div className="space-y-3">
      <div>
        <label className={LABEL}>Token URL</label>
        <input
          type="text"
          value={values.oauth_token_url}
          onChange={(e) => onChange({ oauth_token_url: e.target.value })}
          placeholder="https://idp.example/oauth2/token"
          className={FIELD}
        />
      </div>
      <div>
        <label className={LABEL}>Client ID</label>
        <input
          type="text"
          value={values.oauth_client_id}
          onChange={(e) => onChange({ oauth_client_id: e.target.value })}
          className={FIELD}
        />
      </div>
      <div>
        <label className={LABEL}>Token Endpoint Authentication</label>
        <select
          value={values.oauth_token_auth_style || 'post_body'}
          onChange={(e) => onChange({ oauth_token_auth_style: e.target.value })}
          className={FIELD}
        >
          <option value="post_body">Client secret in POST body</option>
          <option value="basic_header">Client secret in Basic header</option>
          <option value="none">None — public client (no secret)</option>
        </select>
      </div>
      {values.oauth_token_auth_style !== 'none' && (
        <div>
          <label className={LABEL}>Client Secret</label>
          <input
            type="password"
            value={values.oauth_client_secret}
            onChange={(e) => onChange({ oauth_client_secret: e.target.value })}
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
          value={values.oauth_scopes}
          onChange={(e) => onChange({ oauth_scopes: e.target.value })}
          placeholder="api:read, api:write"
          className={FIELD}
        />
      </div>
      <div>
        <label className={LABEL}>Resource (RFC 8707, optional)</label>
        <input
          type="text"
          value={values.oauth_resource}
          onChange={(e) => onChange({ oauth_resource: e.target.value })}
          placeholder="https://api.example.com"
          className={FIELD}
        />
      </div>
    </div>
  </div>
);

export default OAuthClientCredentialsFields;
