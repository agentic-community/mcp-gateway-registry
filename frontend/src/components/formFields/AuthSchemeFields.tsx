import React from 'react';
import FormField from './FormField';
import { fieldClass, FIELD_FOCUS } from './formClasses';

export type AuthScheme = 'none' | 'bearer' | 'api_key' | 'oauth';

const AUTH_SCHEMES: readonly AuthScheme[] = ['none', 'bearer', 'api_key', 'oauth'];

/**
 * Narrow an untrusted `auth_scheme` from the API to the union, falling back to 'none'.
 *
 * Server records are not guaranteed to hold a member of this union: the field is a
 * plain string server-side, and a record written by an older build can carry a value
 * this UI no longer knows (the retired synthetic `oauth2_1` being the case in point).
 * Casting instead of narrowing would let such a value reach the scheme dropdown, slip
 * past `!== 'none'` checks, and be POSTed straight back as an invalid scheme.
 */
export function toAuthScheme(value: unknown): AuthScheme {
  return AUTH_SCHEMES.includes(value as AuthScheme) ? (value as AuthScheme) : 'none';
}

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
  /**
   * True when this server uses On-Behalf-Of egress (egress_auth_mode ===
   * 'obo_exchange'). Backend discovery for such a server has no per-server
   * credential -- it is derived (a gateway machine token). When set and the
   * scheme is 'none', an informational panel explains the derived behavior and
   * the Entra prerequisite.
   */
  oboDiscoveryActive?: boolean;
  /** The obo target audience, shown in the derived-discovery panel. */
  oboTargetAudience?: string;
  /**
   * When true, render the "Discovery Identity (OAuth 2.1)" checkbox. The parent
   * owns the gate (discovery needs the egress feature, a gateway and a remote
   * server), so this component only draws the control.
   */
  showDiscoveryToggle?: boolean;
  /** Current value of the independent discovery-identity flag. */
  discoveryEnabled?: boolean;
  onDiscoveryEnabledChange?: (enabled: boolean) => void;
}

/**
 * The backend-authentication cascade (scheme select -> credential -> header
 * name) shared by the server form's "Backend Authentication" block. The
 * credential field shows for bearer/api_key; the header-name field shows only
 * for api_key. The parent owns the reset semantics (clearing the credential
 * when switching to none, etc.) via onSchemeChange.
 *
 * The discovery-identity checkbox is a SIBLING of the scheme, not a value in
 * it: the backend treats `auth_scheme` and `oauth_discovery` as orthogonal, so
 * a server may carry both a real scheme and a discovery identity.
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
  oboDiscoveryActive = false,
  oboTargetAudience = '',
  showDiscoveryToggle = false,
  discoveryEnabled = false,
  onDiscoveryEnabledChange,
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
            <option value="oauth">OAuth 2.0 (client credentials)</option>
          </select>
        </FormField>

        {(scheme === 'bearer' || scheme === 'api_key') && (
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

        {showDiscoveryToggle && (
          <div className="rounded-md border border-gray-200 dark:border-gray-700 p-3">
            <label className="flex items-start gap-2 cursor-pointer">
              <input
                type="checkbox"
                checked={discoveryEnabled}
                onChange={(e) => onDiscoveryEnabledChange?.(e.target.checked)}
                className="mt-0.5 h-4 w-4 rounded border-gray-300 dark:border-gray-600"
              />
              <span>
                <span className="block text-sm font-medium text-gray-900 dark:text-white">
                  Discovery Identity (OAuth 2.1)
                </span>
                <span className="block text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                  When enabled, the registry borrows the connected admin&apos;s own
                  account for its headless health checks and tool discovery against
                  this server. Requires the authentication scheme above to be{' '}
                  <span className="font-medium">None</span>.
                </span>
              </span>
            </label>
            {/*
              The resolver chain bows out of the borrow for ANY explicit auth_scheme,
              because a resolved OAuth bearer short-circuits the header builders and
              would otherwise silently DROP the operator's static credential. So a
              discovery identity configured alongside bearer/api_key/oauth is inert.
              Say so here rather than letting an operator configure it, complete an
              interactive OAuth consent, and vault a token nothing ever reads.
            */}
            {discoveryEnabled && scheme !== 'none' && (
              <p className="mt-2 rounded border border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/20 p-2 text-xs text-amber-800 dark:text-amber-300">
                <span className="font-semibold">Not in effect.</span> The{' '}
                <span className="font-mono">{scheme}</span> credential above is this
                server&apos;s discovery credential and takes precedence. Set the
                scheme to <span className="font-medium">None</span> for the discovery
                identity to be used.
              </p>
            )}
          </div>
        )}

        {oboDiscoveryActive && scheme === 'none' && (
          <div className="rounded-md border border-blue-200 dark:border-blue-800 bg-blue-50 dark:bg-blue-900/20 p-3 text-xs text-blue-800 dark:text-blue-300">
            <p className="font-semibold mb-1">
              On-Behalf-Of server — discovery uses a gateway machine token
            </p>
            <p>
              Health checks and tool discovery authenticate as the gateway&apos;s
              own IdP app (client_credentials) audienced to{' '}
              <code className="font-mono break-all">
                {oboTargetAudience || 'this server\u2019s target audience'}
              </code>
              . No per-server credential is needed here.
            </p>
            <p className="mt-1">
              On Entra: grant the gateway app an application permission (app role)
              on the target server&apos;s app and admin-consent it; the internal
              server must accept app-only tokens for discovery. Selecting a scheme
              above overrides discovery with an explicit credential.
            </p>
          </div>
        )}
      </div>
    </div>
  );
};

export default AuthSchemeFields;
