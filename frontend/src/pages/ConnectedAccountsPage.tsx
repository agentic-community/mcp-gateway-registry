import React, { useCallback, useEffect, useState } from 'react';
import { useNavigate } from 'react-router-dom';
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
  type EgressConnection,
  type AvailableEgressServer,
} from '../utils/egressAuth';

/**
 * Connected Accounts: the end-user surface for the per-user egress credential
 * vault. Lists linked third-party accounts and lets the user connect a new one
 * (opens the provider consent in a new tab) or disconnect. Discoverable BEFORE
 * the first-use tool-call error so users can self-serve.
 */
const ConnectedAccountsPage: React.FC = () => {
  const navigate = useNavigate();
  const [connections, setConnections] = useState<EgressConnection[]>([]);
  const [available, setAvailable] = useState<AvailableEgressServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string>('');
  const [serverPath, setServerPath] = useState('');
  const [connecting, setConnecting] = useState(false);
  const [copied, setCopied] = useState(false);

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

  const handleConnect = async (e: React.FormEvent) => {
    e.preventDefault();
    const path = serverPath.trim();
    if (!path) return;
    setConnecting(true);
    setError('');
    try {
      const authorizeUrl = await initiateConsent(path);
      // Open the provider consent in a new tab; the callback stores the token.
      window.open(authorizeUrl, '_blank', 'noopener,noreferrer');
    } catch {
      setError(`Could not start a connection for "${path}". Check the server path.`);
    } finally {
      setConnecting(false);
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
        className="flex items-center space-x-1 mb-4 text-sm text-gray-500 dark:text-gray-400 hover:text-gray-700 dark:hover:text-gray-200 focus:outline-none"
      >
        <ArrowLeftIcon className="h-4 w-4" />
        <span>Back to Dashboard</span>
      </button>
      <div className="flex items-center space-x-3 mb-2">
        <LinkIcon className="h-6 w-6 text-purple-600 dark:text-purple-400" />
        <h1 className="text-2xl font-semibold text-gray-900 dark:text-white">Connected Accounts</h1>
      </div>
      <p className="text-sm text-gray-600 dark:text-gray-300 mb-6">
        Link your third-party accounts (GitHub, Slack, Google, …) so MCP servers can act on your
        behalf. Connect an account here before using a server that requires it.
      </p>

      {/* Callback URL: must be registered as the redirect/callback URL in each
          third-party OAuth app (GitHub, Slack, Atlassian, …). Kept compact. */}
      <p className="flex flex-wrap items-center gap-x-2 gap-y-1 text-xs text-gray-500 dark:text-gray-400 mb-6">
        <span>OAuth redirect/callback URL (register in each provider's OAuth app):</span>
        <code className="break-all text-purple-700 dark:text-purple-300">{callbackUrl}</code>
        <button
          type="button"
          onClick={handleCopyCallback}
          className="inline-flex items-center text-gray-400 hover:text-purple-600 dark:hover:text-purple-400 focus:outline-none"
          aria-label="Copy callback URL"
          title={copied ? 'Copied' : 'Copy'}
        >
          {copied ? (
            <CheckIcon className="h-4 w-4 text-green-600 dark:text-green-400" />
          ) : (
            <ClipboardDocumentIcon className="h-4 w-4" />
          )}
        </button>
        {copied && <span className="text-green-600 dark:text-green-400">Copied</span>}
      </p>

      {error && (
        <div className="flex items-center space-x-2 mb-4 p-3 rounded-lg bg-red-50 dark:bg-red-900/30 text-red-700 dark:text-red-300">
          <ExclamationTriangleIcon className="h-5 w-5 flex-shrink-0" />
          <span className="text-sm">{error}</span>
        </div>
      )}

      {/* Connect a new account: pick from the egress-enabled servers the user
          can access (no need to know/type a raw server path). */}
      <form
        onSubmit={handleConnect}
        className="flex items-end gap-3 mb-8 p-4 rounded-lg bg-gray-50 dark:bg-gray-800"
      >
        <div className="flex-1">
          <label
            htmlFor="egress-server"
            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
          >
            Server requiring per-user authentication
          </label>
          <select
            id="egress-server"
            value={serverPath}
            onChange={e => setServerPath(e.target.value)}
            disabled={available.length === 0}
            className="w-full px-3 py-2 rounded-md border border-gray-300 dark:border-gray-600 bg-white dark:bg-gray-700 text-gray-900 dark:text-white text-sm focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50"
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
        <button
          type="submit"
          disabled={connecting || !serverPath.trim()}
          className="flex items-center space-x-2 px-4 py-2 rounded-md bg-purple-600 text-white text-sm font-medium hover:bg-purple-700 disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-purple-500"
        >
          <ArrowTopRightOnSquareIcon className="h-4 w-4" />
          <span>{connecting ? 'Opening…' : 'Connect'}</span>
        </button>
      </form>

      {/* Existing connections */}
      {loading ? (
        <p className="text-sm text-gray-500 dark:text-gray-400">Loading…</p>
      ) : connections.length === 0 ? (
        <p className="text-sm text-gray-500 dark:text-gray-400">No connected accounts yet.</p>
      ) : (
        <ul className="divide-y divide-gray-200 dark:divide-gray-700 rounded-lg border border-gray-200 dark:border-gray-700">
          {connections.map(conn => (
            <li
              key={`${conn.provider}:${conn.server_path}`}
              className="flex items-start justify-between gap-4 p-4"
            >
              <div className="min-w-0 flex-1">
                <div className="flex items-center space-x-2">
                  <span className="font-medium text-gray-900 dark:text-white capitalize">
                    {conn.provider}
                  </span>
                  <span className="text-xs text-gray-500 dark:text-gray-400">
                    {conn.server_path}
                  </span>
                  {conn.status !== 'active' && (
                    <span className="text-xs px-2 py-0.5 rounded-full bg-yellow-100 dark:bg-yellow-900/40 text-yellow-800 dark:text-yellow-300">
                      {conn.status}
                    </span>
                  )}
                </div>
                {conn.scopes.length > 0 && (
                  <div className="text-xs text-gray-500 dark:text-gray-400 mt-1 break-words">
                    {conn.scopes.join(', ')}
                  </div>
                )}
              </div>
              <button
                onClick={() => handleDisconnect(conn)}
                className="flex flex-shrink-0 items-center space-x-1 px-3 py-1.5 rounded-md text-sm text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 focus:outline-none focus:ring-2 focus:ring-red-500"
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
