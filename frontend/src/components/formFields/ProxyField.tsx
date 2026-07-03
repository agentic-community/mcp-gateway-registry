import React from 'react';
import FormField from './FormField';
import { fieldClass, FIELD_FOCUS } from './formClasses';

interface ProxyFieldProps {
  /** Whether the entity is served through the gateway generic proxy. */
  isProxied: boolean;
  onIsProxiedChange: (value: boolean) => void;
  /** Backend URL the gateway forwards to (shown only when isProxied). */
  proxyTargetUrl: string;
  onProxyTargetUrlChange: (value: string) => void;
  /**
   * When true, proxy_target_url is REQUIRED while proxied (skills, custom
   * entities). When false the entity has a native backend URL to fall back to
   * (MCP servers, agents), so the target field is optional.
   */
  targetRequired?: boolean;
  accent?: keyof typeof FIELD_FOCUS;
  /** Validation error for the target URL. */
  error?: string | null;
  /**
   * The read-only, auto-derived client-facing gateway path
   * ({prefix}/{type}/{name}) the registry generated. Shown (not editable) when
   * present so the user sees where clients connect. Undefined on create (the
   * path is assigned server-side) — the helper text explains it will be
   * generated on save.
   */
  clientUrl?: string | null;
}

/**
 * Gateway-proxy opt-in: an is_proxied checkbox plus the conditional
 * proxy_target_url input that appears only when proxying is on. Shared across
 * the skill/agent/custom forms so the toggle looks and behaves identically.
 *
 * NOTE: the backend still validates the target (SSRF resolve-and-validate) at
 * registration; this control is a convenience, not the security boundary.
 */
const ProxyField: React.FC<ProxyFieldProps> = ({
  isProxied,
  onIsProxiedChange,
  proxyTargetUrl,
  onProxyTargetUrlChange,
  targetRequired = false,
  accent = 'purple',
  error,
  clientUrl,
}) => {
  const showTargetWarning =
    isProxied && targetRequired && proxyTargetUrl.trim() === '';
  return (
    <>
      <FormField
        label="Serve through the gateway proxy"
        hint="Route authenticated traffic to this resource through the gateway."
      >
        <label className="inline-flex items-center gap-2 cursor-pointer">
          <input
            type="checkbox"
            checked={isProxied}
            onChange={(e) => onIsProxiedChange(e.target.checked)}
            className="h-4 w-4 rounded border-gray-300 dark:border-gray-600"
          />
          <span className="text-sm text-gray-600 dark:text-gray-400">
            {isProxied ? 'Proxied' : 'Not proxied'}
          </span>
        </label>
      </FormField>

      {isProxied && (
        <div className="rounded-md border border-cyan-200 bg-cyan-50 px-3 py-2 text-sm dark:border-cyan-800 dark:bg-cyan-900/20">
          {clientUrl ? (
            <p className="text-cyan-800 dark:text-cyan-200">
              Clients connect at{' '}
              <code className="rounded bg-cyan-100 px-1 py-0.5 font-mono text-xs text-cyan-900 dark:bg-cyan-800/50 dark:text-cyan-100">
                {clientUrl}
              </code>
              . The registry forwards this to the backend URL below.
            </p>
          ) : (
            <p className="text-cyan-800 dark:text-cyan-200">
              A client URL (<code className="font-mono text-xs">/{'{prefix}'}/{'{type}'}/{'{name}'}</code>)
              is generated automatically when you save. Clients connect there; the registry forwards to
              the backend URL below.
            </p>
          )}
        </div>
      )}

      {isProxied && (
        <FormField
          label="Backend URL"
          required={targetRequired}
          error={error}
          hint={
            targetRequired
              ? 'The http(s) origin the gateway forwards to (required).'
              : 'The http(s) origin the gateway forwards to. Leave blank to use the entity’s own URL.'
          }
        >
          <input
            type="url"
            value={proxyTargetUrl}
            onChange={(e) => onProxyTargetUrlChange(e.target.value)}
            className={fieldClass(accent)}
            placeholder="https://backend.example.com/"
          />
          {showTargetWarning && (
            <p className="mt-1 text-xs text-amber-600 dark:text-amber-400">
              A proxy target URL is required when proxying is enabled.
            </p>
          )}
        </FormField>
      )}
    </>
  );
};

export default ProxyField;
