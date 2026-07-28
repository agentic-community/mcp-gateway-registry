import React from 'react';
import FormField from './FormField';
import { fieldClass, FIELD_FOCUS } from './formClasses';

export type AuthScheme = 'none' | 'bearer' | 'api_key';

interface AuthSchemeFieldsProps {
  scheme: AuthScheme;
  credential: string;
  headerName: string;
  /** Called with the new scheme; the parent applies the reset cascade. */
  onSchemeChange: (scheme: AuthScheme) => void;
  onCredentialChange: (value: string) => void;
  onHeaderNameChange: (value: string) => void;
  /** When true, the credential placeholder reflects "keep existing" (edit mode). */
  editing?: boolean;
  accent?: keyof typeof FIELD_FOCUS;
}

/**
 * The backend-authentication cascade (scheme select -> credential -> header
 * name) shared by the server form's "Backend Authentication" block. The
 * credential field shows for bearer/api_key; the header-name field shows only
 * for api_key. The parent owns the reset semantics (clearing the credential
 * when switching to none, etc.) via onSchemeChange.
 *
 * The skill form's auth has an extra 'global_credentials' option and inline
 * re-parse buttons, so it keeps its own richer controls.
 */
const AuthSchemeFields: React.FC<AuthSchemeFieldsProps> = ({
  scheme,
  credential,
  headerName,
  onSchemeChange,
  onCredentialChange,
  onHeaderNameChange,
  editing = false,
  accent = 'purple',
}) => {
  return (
    <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
      <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-1">
        Backend Authentication
      </h4>
      <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
        The credential the registry uses itself to reach this server for health
        checks and tool discovery. Per-user egress reuses this header definition.
      </p>

      <div className="space-y-4">
        <FormField label="Authentication Scheme">
          <select
            value={scheme}
            onChange={(e) => onSchemeChange(e.target.value as AuthScheme)}
            className={fieldClass(accent)}
          >
            <option value="none">None</option>
            <option value="bearer">Bearer Token</option>
            <option value="api_key">API Key</option>
          </select>
        </FormField>

        {scheme !== 'none' && (
          <FormField
            label={scheme === 'bearer' ? 'Bearer Token' : 'API Key'}
            hint="Leave blank to keep the existing credential unchanged."
          >
            <input
              type="password"
              value={credential}
              onChange={(e) => onCredentialChange(e.target.value)}
              className={fieldClass(accent)}
              placeholder={
                editing ? 'Leave blank to keep current credential' : ''
              }
            />
          </FormField>
        )}

        {scheme === 'api_key' && (
          <FormField label="Header Name">
            <input
              type="text"
              value={headerName}
              onChange={(e) => onHeaderNameChange(e.target.value)}
              className={fieldClass(accent)}
              placeholder="X-API-Key"
            />
          </FormField>
        )}
      </div>
    </div>
  );
};

export default AuthSchemeFields;
