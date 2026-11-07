import React, { useCallback, useEffect, useState } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import {
  LinkIcon,
  TrashIcon,
  ExclamationTriangleIcon,
  ArrowTopRightOnSquareIcon,
  ArrowLeftIcon,
  ClipboardDocumentIcon,
  CheckIcon,
} from '@heroicons/react/24/outline';

import {
  listConnections,
  listAvailableServers,
  initiateConsent,
  disconnect,
  setEgressPat,
  type EgressConnection,
  type AvailableEgressServer,
} from '../utils/egressAuth';
import { isSafeUrl } from '../utils/safeUrl';
import { FIELD, LABEL } from '../components/formFields/formClasses';

/**
 * Connected Accounts: the end-user surface for the per-user egress credential
 * vault. Lists linked third-party accounts and lets the user connect a new one
 * (opens the provider consent in a new tab) or disconnect. Discoverable BEFORE
 * the first-use tool-call error so users can self-serve.
 */
const ConnectedAccountsPage: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const [connections, setConnections] = useState<EgressConnection[]>([]);
  const [available, setAvailable] = useState<AvailableEgressServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [serverPath, setServerPath] = useState('');
  const [connecting, setConnecting] = useState(false);
  const [copied, setCopied] = useState(false);
  // PAT submit form state (used when the selected server is in `pat` mode).
  const [patSecret, setPatSecret] = useState('');
  const [patTtlValue, setPatTtlValue] = useState(30);
  const [patTtlUnit, setPatTtlUnit] = useState('days');
  const [patExpiresAt, setPatExpiresAt] = useState<string | null>(null);
  const [submittingPat, setSubmittingPat] = useState(false);

  // The server currently selected in the dropdown, if any.
  const selectedServer = available.find(s => s.server_path === serverPath);
  const isPatServer = selectedServer?.egress_auth_mode === 'pat';

  // The gateway callback (redirect) URL that must be registered in each
  // third-party OAuth app. Served by this same host through nginx, so it is
  // derived from the current origin.
  const callbackUrl = `${window.location.origin}/oauth2/egress/callback`;

  const handleCopyCallback = async () => {
    try {
      await navigator.clipboard.writeText(callbackUrl);
    } catch {
      // Clipboard API unavailable (e.g. non-HTTPS/older browser); fall back.
      const ta = document.createElement('textarea');
      ta.value = callbackUrl;
      document.body.appendChild(ta);
      ta.select();
      document.execCommand('copy');
      document.body.removeChild(ta);
    }
    setCopied(true);
    // "Copied" confirmation clears after 5 seconds.
    window.setTimeout(() => setCopied(false), 5000);
  };

  const refresh = useCallback(async () => {
    setLoading(true);
    setError('');
    try {
      const [conns, avail] = await Promise.all([
        listConnections(),
        listAvailableServers(),
      ]);
      setConnections(conns);
      setAvailable(avail);
    } catch {
      setError('Could not load connections.');
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  // Preselect the dropdown when a server is passed via ?server= (from the card
  // icon / connect-modal callout), so the user lands ready to connect/submit.
  useEffect(() => {
    const s = searchParams.get('server');
    if (s) setServerPath(s);
  }, [searchParams]);

  const handleConnect = async (e: React.FormEvent) => {
    e.preventDefault();
    const path = serverPath.trim();
    if (!path) return;
    setConnecting(true);
    setError('');
    try {
      const authorizeUrl = await initiateConsent(path);
      // The authorize URL derives from per-server egress OAuth config, so treat
      // it as untrusted: only open http/https, never a javascript:/data: scheme
      // that would execute in the user's session on connect.
      if (!isSafeUrl(authorizeUrl)) {
        setError(`Could not start a connection for "${path}": the provider returned an unsafe authorization URL.`);
        return;
      }
      // Open the provider consent in a new tab; the callback stores the token.
      window.open(authorizeUrl, '_blank', 'noopener,noreferrer');
    } catch {
      setError(`Could not start a connection for "${path}". Check the server path.`);
    } finally {
      setConnecting(false);
    }
  };

  const handleSubmitPat = async (e: React.FormEvent) => {
    e.preventDefault();
    const path = serverPath.trim();
    const secret = patSecret.trim();
    if (!path || !secret) return;
    setSubmittingPat(true);
    setError('');
    try {
      const result = await setEgressPat(path, secret, patTtlValue, patTtlUnit);
      // Never keep the secret in state after submit; it is write-only.
      setPatSecret('');
      setPatExpiresAt(result.expires_at);
      await refresh();
    } catch {
      setError(`Could not submit the token for "${path}". Check the value and try again.`);
    } finally {
      setSubmittingPat(false);
    }
  };

  const handleDisconnect = async (conn: EgressConnection) => {
    setError('');
    try {
      await disconnect(conn.provider, conn.server_path);
      await refresh();
    } catch {
      setError(`Could not disconnect ${conn.provider} for ${conn.server_path}.`);
    }
  };

  return (
    <div className="max-w-3xl mx-auto p-6">
      <button
        type="button"
        onClick={() => navigate('/')}
        className="flex items-center space-x-1 mb-4 text-sm text-fg-muted hover:text-fg"
      >
        <ArrowLeftIcon className="h-4 w-4" />
        <span>Back to Dashboard</span>
      </button>
      <div className="flex items-center space-x-3 mb-2">
        <LinkIcon className="h-6 w-6 text-purple-600 dark:text-purple-400" />
        <h1 className="text-2xl font-semibold text-fg">Connected Accounts</h1>
      </div>
      <p className="text-sm text-fg-muted mb-6">
        Link your third-party accounts (GitHub, Slack, Google, …) so MCP servers can act on your
        behalf. Connect an account here before using a server that requires it.
      </p>

      {/* Callback URL: must be registered as the redirect/callback URL in each
          third-party OAuth app (GitHub, Slack, Atlassian, …). Kept compact. */}
      <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-fg-muted mb-6">
        <span>OAuth redirect/callback URL (register in each provider's OAuth app):</span>
        <code className="break-all text-accent-text">{callbackUrl}</code>
        <button
          type="button"
          onClick={handleCopyCallback}
          className="inline-flex items-center rounded text-fg-subtle hover:text-accent-text"
          aria-label="Copy callback URL"
          title={copied ? 'Copied' : 'Copy'}
        >
          {copied ? (
            <CheckIcon className="h-4 w-4 text-success-600 dark:text-success-400" />
          ) : (
            <ClipboardDocumentIcon className="h-4 w-4" />
          )}
        </button>
        {copied && <span className="text-success-600 dark:text-success-400">Copied</span>}
      </p>

      {error && (
        <div className="flex items-center space-x-2 mb-4 p-3 rounded-lg bg-danger-50 dark:bg-danger-900/30 text-danger-700 dark:text-danger-300">
          <ExclamationTriangleIcon className="h-5 w-5 flex-shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {/* Connect a new account: pick from the egress-enabled servers the user
          can access (no need to know/type a raw server path). A `pat`-mode
          server shows a submit-token form instead of the OAuth Connect button. */}
      <div className="card mb-8 p-4">
        <div className="mb-3">
          <label
            htmlFor="egress-server"
            className={LABEL}
          >
            Server requiring per-user authentication
          </label>
          <select
            id="egress-server"
            value={serverPath}
            onChange={e => {
              setServerPath(e.target.value);
              // Reset the PAT form when the selected server changes.
              setPatSecret('');
              setPatExpiresAt(null);
            }}
            disabled={available.length === 0}
            className={FIELD}
          >
            <option value="">
              {available.length === 0
                ? 'No servers require per-user authentication'
                : 'Select a server…'}
            </option>
            {available.map(s => (
              <option key={s.server_path} value={s.server_path}>
                {s.server_name} ({s.provider}) — {s.server_path}
              </option>
            ))}
          </select>
        </div>

        {isPatServer ? (
          <form onSubmit={handleSubmitPat} className="space-y-3">
            <div>
              <label
                htmlFor="egress-pat-secret"
                className={LABEL}
              >
                Personal access token / API key
              </label>
              <input
                id="egress-pat-secret"
                type="password"
                value={patSecret}
                onChange={e => setPatSecret(e.target.value)}
                autoComplete="off"
                placeholder="Paste your token"
                className={FIELD}
              />
            </div>
            <div className="flex items-end gap-3">
              <div>
                <label
                  htmlFor="egress-pat-ttl-value"
                  className={LABEL}
                >
                  Valid for
                </label>
                <input
                  id="egress-pat-ttl-value"
                  type="number"
                  min={1}
                  value={patTtlValue}
                  onChange={e => setPatTtlValue(parseInt(e.target.value, 10) || 0)}
                  className={`${FIELD} w-24`}
                />
              </div>
              <div>
                <label
                  htmlFor="egress-pat-ttl-unit"
                  className={LABEL}
                >
                  Unit
                </label>
                <select
                  id="egress-pat-ttl-unit"
                  value={patTtlUnit}
                  onChange={e => setPatTtlUnit(e.target.value)}
                  className={`${FIELD} w-auto`}
                >
                  <option value="minutes">Minutes</option>
                  <option value="hours">Hours</option>
                  <option value="days">Days</option>
                </select>
              </div>
              <button
                type="submit"
                disabled={submittingPat || !patSecret.trim()}
                className="btn-primary"
              >
                <span>{submittingPat ? 'Submitting…' : 'Submit token'}</span>
              </button>
            </div>
            <p className="text-xs text-fg-muted">
              The token is stored write-only and never shown again. Maximum lifetime is 30 days.
            </p>
            {patExpiresAt && (
              <p className="text-xs text-success-600 dark:text-success-400">
                Token stored. Expires at {patExpiresAt}.
              </p>
            )}
          </form>
        ) : (
          <form onSubmit={handleConnect} className="flex items-end gap-3">
            <button
              type="submit"
              disabled={connecting || !serverPath.trim()}
              className="btn-primary"
            >
              <ArrowTopRightOnSquareIcon className="h-4 w-4" />
              <span>{connecting ? 'Opening…' : 'Connect'}</span>
            </button>
          </form>
        )}
      </div>

      {/* Existing connections */}
      {loading ? (
        <p className="text-sm text-fg-muted">Loading…</p>
      ) : connections.length === 0 ? (
        <p className="text-sm text-fg-muted">No connected accounts yet.</p>
      ) : (
        <ul className="card divide-y divide-line">
          {connections.map(conn => (
            <li
              key={`${conn.provider}:${conn.server_path}`}
              className="flex items-start justify-between gap-4 p-4"
            >
              <div className="min-w-0 flex-1">
                <div className="flex items-center space-x-2">
                  <span className="font-medium text-fg capitalize">
                    {conn.provider}
                  </span>
                  <span className="text-xs text-fg-muted">
                    {conn.server_path}
                  </span>
                  {conn.status !== 'active' && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-warning-100 dark:bg-warning-900/40 text-warning-800 dark:text-warning-300">
                      {conn.status}
                    </span>
                  )}
                </div>
                {conn.scopes.length > 0 && (
                  <div className="text-xs text-fg-muted mt-1 break-words">
                    {conn.scopes.join(', ')}
                  </div>
                )}
              </div>
              <button
                onClick={() => handleDisconnect(conn)}
                className="btn-base flex-shrink-0 px-3 py-1.5 text-danger-600 dark:text-danger-400 hover:bg-danger-50 hover:text-danger-700 dark:hover:bg-danger-500/10 dark:hover:text-danger-300"
                aria-label={`Disconnect ${conn.provider} for ${conn.server_path}`}
              >
                <TrashIcon className="h-4 w-4" />
                <span>Disconnect</span>
              </button>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
};

export default ConnectedAccountsPage;
