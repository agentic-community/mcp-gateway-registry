import React, { useCallback, useState, useEffect } from 'react';
import { ClipboardDocumentIcon, KeyIcon } from '@heroicons/react/24/outline';
import PlugIcon from './icons/PlugIcon';
import axios from 'axios';
import type { Server } from './ServerCard';
import { useRegistryConfig } from '../hooks/useRegistryConfig';
import useEscapeKey from '../hooks/useEscapeKey';
import { getBaseURL } from '../utils/basePath';
import { isSafeUrl } from '../utils/safeUrl';
import {
  initiateConsent,
  disconnect as disconnectEgress,
  type EgressCardState,
} from '../utils/egressAuth';

const IDE_LABELS = {
  'cursor': 'Cursor',
  'roo-code': 'Roo Code',
  'claude-code': 'Claude Code',
  'kiro': 'Kiro',
  'goose': 'Goose',
  'codex': 'Codex',
  'cli': 'CLI (curl)',
} as const;

type IDE = keyof typeof IDE_LABELS;

interface ServerConfigModalProps {
  server: Server;
  isOpen: boolean;
  onClose: () => void;
  onShowToast?: (message: string, type: 'success' | 'error') => void;
  /**
   * Resource type for the bound-token mint. Callers pass
   * 'virtual_server' when opening this modal for a virtual server
   * or 'server' (the default) for a regular MCP
   * server. Used to build the `resource` field on /api/tokens/generate.
   */
  resourceType?: 'server' | 'virtual_server';
  // Per-user egress state for this server (undefined => no callout). Shared with
  // the card icon so the two surfaces never disagree.
  egressConnect?: EgressCardState;
  // Refresh the dashboard's egress state after a connect/disconnect here.
  onEgressChanged?: () => void;
}


/**
 * The issue #1495 "Connect your account" callout, shown at the top of the
 * connect modal for a per-user-egress server. Roomier than the card icon, so it
 * carries the full treatment: connect / connected+disconnect / reconnect on a
 * dead token. Reuses the same initiate()/disconnect() client the Connected
 * Accounts page uses.
 */
function EgressConnectCallout({
  serverPath,
  state,
  onEgressChanged,
  onShowToast,
}: {
  serverPath: string;
  state: EgressCardState;
  onEgressChanged?: () => void;
  onShowToast?: (message: string, type: 'success' | 'error') => void;
}) {
  const [busy, setBusy] = useState(false);
  const providerLabel = state.provider
    ? state.provider.charAt(0).toUpperCase() + state.provider.slice(1)
    : 'account';

  const startConnect = async () => {
    setBusy(true);
    try {
      // Prefer the same initiate() the Connected Accounts page uses; fall back
      // to the server-built connect_url front door. Both are opened, never fetched.
      let url = '';
      try {
        url = await initiateConsent(serverPath);
      } catch {
        url = state.connectUrl;
      }
      // MANDATORY isSafeUrl guard before window.open (fail closed on unsafe/empty).
      if (!isSafeUrl(url)) {
        onShowToast?.('Cannot open the connect URL for this server.', 'error');
        return;
      }
      window.open(url, '_blank', 'noopener,noreferrer');
      // The callback tab vaults the token; refresh so the card/callout flip to
      // "Connected" when the user returns.
      onEgressChanged?.();
    } finally {
      setBusy(false);
    }
  };

  const handleDisconnect = async () => {
    setBusy(true);
    try {
      await disconnectEgress(state.provider, serverPath);
      onShowToast?.(`Disconnected ${providerLabel}.`, 'success');
      onEgressChanged?.();
    } catch {
      onShowToast?.(`Could not disconnect ${providerLabel}.`, 'error');
    } finally {
      setBusy(false);
    }
  };

  // PAT server: no OAuth redirect — point at the token form on Connected Accounts.
  if (state.mode === 'pat') {
    return (
      <div className="flex items-center gap-3 rounded-lg border border-purple-200 dark:border-purple-800 bg-purple-50 dark:bg-purple-900/20 p-4">
        <PlugIcon className="h-5 w-5 flex-shrink-0 text-purple-600 dark:text-purple-400" />
        <span className="text-sm text-purple-900 dark:text-purple-100">
          This server needs a personal access token before its tools appear.
        </span>
        <a
          href={`/connected-accounts?server=${encodeURIComponent(serverPath)}`}
          className="ml-auto text-sm font-medium text-purple-700 dark:text-purple-300 hover:underline"
        >
          Submit token
        </a>
      </div>
    );
  }

  // 3LO: not connected / needs reconnect / connected.
  if (!state.connected || state.needsReconnect) {
    const warn = state.needsReconnect;
    return (
      <div
        className={`flex items-center gap-3 rounded-lg border p-4 ${
          warn
            ? 'border-amber-300 dark:border-amber-700 bg-amber-50 dark:bg-amber-900/20'
            : 'border-purple-200 dark:border-purple-800 bg-purple-50 dark:bg-purple-900/20'
        }`}
      >
        <PlugIcon
          className={`h-5 w-5 flex-shrink-0 ${
            warn ? 'text-amber-500' : 'text-purple-600 dark:text-purple-400'
          }`}
        />
        <span className="text-sm text-gray-900 dark:text-gray-100">
          {warn
            ? `Your ${providerLabel} connection expired or failed — reconnect to restore this server's tools.`
            : `This server acts on your behalf. Connect your ${providerLabel} account before copying the config.`}
        </span>
        <button
          type="button"
          disabled={busy}
          onClick={startConnect}
          className="ml-auto flex-shrink-0 rounded-md bg-purple-600 px-3 py-1.5 text-sm font-medium text-white hover:bg-purple-700 disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-purple-500"
        >
          {busy ? 'Opening…' : warn ? `Reconnect ${providerLabel}` : `Connect ${providerLabel} account`}
        </button>
      </div>
    );
  }

  return (
    <div className="flex items-center gap-3 rounded-lg border border-green-200 dark:border-green-800 bg-green-50 dark:bg-green-900/20 p-4">
      <PlugIcon className="h-5 w-5 flex-shrink-0 text-green-600 dark:text-green-400" />
      <span className="text-sm text-green-900 dark:text-green-100">
        Connected to {providerLabel}. This server can act on your behalf.
      </span>
      <button
        type="button"
        disabled={busy}
        onClick={handleDisconnect}
        className="ml-auto flex-shrink-0 rounded-md px-3 py-1.5 text-sm font-medium text-red-600 dark:text-red-400 hover:bg-red-50 dark:hover:bg-red-900/30 disabled:opacity-50 focus:outline-none focus:ring-2 focus:ring-red-500"
      >
        {busy ? 'Working…' : 'Disconnect'}
      </button>
    </div>
  );
}

const ServerConfigModal: React.FC<ServerConfigModalProps> = ({
  server,
  isOpen,
  onClose,
  onShowToast,
  resourceType = 'server',
  egressConnect,
  onEgressChanged,
}) => {
  const [jwtToken, setJwtToken] = useState<string | null>(null);
  const [tokenLoading, setTokenLoading] = useState(false);
  const [tokenError, setTokenError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const { config: registryConfig, loading: configLoading } = useRegistryConfig();

  const enabledIDEs: IDE[] = React.useMemo(() => {
    let allIDEs = Object.keys(IDE_LABELS) as IDE[];
    // CLI (curl) and Codex URL mode don't apply to local stdio servers
    if (server.deployment === 'local') {
      allIDEs = allIDEs.filter((ide) => ide !== 'cli' && ide !== 'codex');
    }
    const allowlist = registryConfig?.coding_assistants ?? [];
    if (allowlist.length === 0) return allIDEs;
    const filtered = allIDEs.filter((ide) => allowlist.includes(ide));
    return filtered.length > 0 ? filtered : allIDEs;
  }, [registryConfig?.coding_assistants, server.deployment]);

  const [selectedIDE, setSelectedIDE] = useState<IDE>(enabledIDEs[0] ?? 'cursor');

  useEffect(() => {
    if (!enabledIDEs.includes(selectedIDE)) {
      setSelectedIDE(enabledIDEs[0]);
    }
  }, [enabledIDEs, selectedIDE]);

  useEscapeKey(onClose, isOpen);

  // Determine if we're in registry-only mode
  // While config is loading, default to with-gateway behavior (safer default)
  const isRegistryOnly = !configLoading && registryConfig?.deployment_mode === 'registry-only';

  // Detect if DCR (Dynamic Client Registration) is available.
  // Keycloak supports DCR, so clients like Codex and Claude Code can handle auth
  // automatically without needing pre-configured tokens in the config.
  const isDCR = !configLoading && registryConfig?.auth_provider === 'keycloak';

  // Custom headers + OAuth login config from the connect-config endpoint
  const [customHeaders, setCustomHeaders] = useState<Array<{name: string; value: string}>>([]);
  const [connectConfigError, setConnectConfigError] = useState<string | null>(null);
  // Pre-registered public OAuth client_id (per-server, resolved server-side with
  // the registry-wide IDE_OAUTH_CLIENT_ID default). When set, the Connect config
  // advertises it and drops the static gateway token so the IDE shows a login
  // button and runs the OAuth/PKCE flow instead of pasting a token.
  const [oauthClientId, setOauthClientId] = useState<string>('');
  // Fixed loopback callback port for OAuth login. 0/null = let the IDE pick one
  // (fine for Keycloak's wildcard redirect). For IdPs that match redirect_uri
  // literally (Okta/Entra/Cognito), the operator sets a fixed port and we emit
  // --callback-port so the IDE does not use a random port the IdP rejects.
  const [oauthCallbackPort, setOauthCallbackPort] = useState<number | null>(null);
  // Optional scope for the Claude Code Connect snippet (local|project|user).
  // null/'' => omit --scope, keeping Claude Code's default (local). Operators set
  // ide_connect_scope=user so the copied command installs the server for every
  // project, not just the current directory.
  const [connectScope, setConnectScope] = useState<string | null>(null);
  // Per-server override for the trailing '/mcp' transport segment on the gateway
  // URL. null = auto-detect from proxy_pass_url.
  const [appendMcpPath, setAppendMcpPath] = useState<boolean | null>(null);
  // Per-user egress credential vault mode ('none' | 'oauth_user'). When
  // 'oauth_user' the gateway injects the user's vaulted upstream token on
  // egress, so the Connect config must emit NO server Authorization/API-key
  // header (the client sends none; a placeholder would be forwarded verbatim
  // and break the connection).
  const [egressAuthMode, setEgressAuthMode] = useState<string>('none');

  const useOAuthLogin = !!oauthClientId && !isRegistryOnly;
  // True when the gateway supplies the upstream credential itself (egress 3LO),
  // so the client must not send any server-auth header.
  const egressManaged = egressAuthMode === 'oauth_user';

  // Fetch JWT token when modal opens (only in gateway mode, and only for remote servers).
  // Local stdio servers don't go through the gateway — no token needed.
  useEffect(() => {
    if (isOpen && !isRegistryOnly && server.deployment !== 'local') {
      // Reset token state when modal opens
      setJwtToken(null);
      setTokenError(null);
      fetchJwtToken();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, isRegistryOnly, server.deployment]);

  // Fetch custom headers when modal opens
  useEffect(() => {
    if (!isOpen) return;
    setConnectConfigError(null);
    setCustomHeaders([]);
    setOauthClientId('');
    setOauthCallbackPort(null);
    setConnectScope(null);
    setAppendMcpPath(null);
    const serverPath = server.path.replace(/^\/+/, '');
    // Fetch CSRF token first, then include it as header for the GET request
    // (required by verify_csrf_token_header_only for cookie-authenticated sessions)
    axios
      .get('/api/auth/csrf-token')
      .then(csrfResp => {
        const csrfToken = csrfResp.data?.csrf_token;
        const headers: Record<string, string> = {};
        if (csrfToken) {
          headers['X-CSRF-Token'] = csrfToken;
        }
        return axios.get(`/api/servers/${serverPath}/connect-config`, { headers });
      })
      .then(resp => {
        setCustomHeaders(resp.data.custom_headers ?? []);
        setOauthClientId(resp.data.oauth_client_id ?? '');
        setOauthCallbackPort(
          typeof resp.data.oauth_callback_port === 'number' ? resp.data.oauth_callback_port : null
        );
        setConnectScope(
          typeof resp.data.connect_scope === 'string' && resp.data.connect_scope
            ? resp.data.connect_scope
            : null
        );
        setAppendMcpPath(
          typeof resp.data.append_mcp_path === 'boolean' ? resp.data.append_mcp_path : null
        );
        setEgressAuthMode(resp.data.egress_auth_mode ?? 'none');
        if (resp.data.decrypt_failures > 0) {
          setConnectConfigError(
            `${resp.data.decrypt_failures} custom header(s) could not be decrypted.`
          );
        }
      })
      .catch((err) => {
        console.error("Failed to fetch connect config", err);
        // A 403 here is almost always a stale CSRF/session after a redeploy.
        // Surface an actionable message rather than silently falling back to
        // the static-token config (which hides that OAuth login was configured).
        const status = err?.response?.status;
        if (status === 403) {
          setConnectConfigError(
            "Could not load connect configuration (403): your session may be stale. " +
            "Log out and back in, then reopen this dialog. The configuration shown " +
            "below is a fallback and may omit OAuth login or required headers."
          );
        } else {
          setConnectConfigError(
            "Could not load connect configuration for this server. " +
            "The configuration shown below is a fallback and may omit OAuth login " +
            "or headers your server requires."
          );
        }
      });
  }, [isOpen, server.path]);

  const fetchJwtToken = async () => {
    setTokenLoading(true);
    setTokenError(null);
    try {
      const body: Record<string, unknown> = {
        description: `Generated for MCP configuration (${server.name})`,
        expires_in_hours: 8,
      };
      // ai-registry-tools (mcpgw) needs a user token because it calls registry
      // APIs internally (/api/search/semantic, /api/servers, etc.) which require
      // broader scopes than a resource-bound token provides.
      const isRegistryTools = server.path?.replace(/^\//, '').startsWith('airegistry-tools');
      if (server.path && !isRegistryTools) {
        body.resource = { type: resourceType, id: server.path };
      }
      const response = await axios.post('/api/tokens/generate', body, {
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (response.data.success) {
        // Token can be in response.data.tokens.access_token or response.data.access_token
        const accessToken = response.data.tokens?.access_token || response.data.access_token;
        if (accessToken) {
          setJwtToken(accessToken);
        } else {
          setTokenError('Token not found in response');
        }
      } else {
        setTokenError('Token generation failed');
      }
    } catch (err: any) {
      const status = err.response?.status;
      const errorMessage = err.response?.data?.detail || err.message || 'Failed to generate token';

      // Provide more helpful error messages based on status
      if (status === 401 || status === 403) {
        setTokenError('Authentication required. Please log in first.');
      } else {
        setTokenError(errorMessage);
      }
      console.error('Failed to fetch JWT token:', err);
    } finally {
      setTokenLoading(false);
    }
  };

  const isLocal = server.deployment === 'local';

  const buildLocalLaunchSpec = useCallback(() => {
    const rt = server.local_runtime;
    if (!rt) return null;

    const env: Record<string, string> = { ...(rt.env ?? {}) };
    // Show literal placeholders for required_env keys the user hasn't filled in.
    for (const k of rt.required_env ?? []) {
      if (!(k in env)) env[k] = '<your-value>';
    }

    switch (rt.type) {
      case 'docker': {
        // For docker, env must be passed into the container with -e flags.
        // The top-level `env` map only sets vars on the host docker CLI process —
        // it does NOT propagate inside the container. So we expand both literal
        // env entries and required_env into -e flags on the docker run command.
        const args = ['run', '-i', '--rm'];
        for (const [k, v] of Object.entries(rt.env ?? {})) {
          args.push('-e', `${k}=${v}`);
        }
        for (const k of rt.required_env ?? []) {
          // -e KEY (no value) tells docker to inherit the host env var of that
          // name — letting the IDE pass the user-supplied secret through.
          args.push('-e', k);
        }
        const imageRef = rt.image_digest ? `${rt.package}@${rt.image_digest}` : rt.package;
        args.push(imageRef, ...(rt.args ?? []));
        // The IDE-visible `env` block carries placeholders for required keys so
        // users know what to fill in; literal values are already in the args.
        const ideEnv: Record<string, string> = {};
        for (const k of rt.required_env ?? []) {
          ideEnv[k] = '<your-value>';
        }
        return { command: 'docker', args, env: ideEnv };
      }
      case 'npx': {
        const pkg = rt.version ? `${rt.package}@${rt.version}` : rt.package;
        return { command: 'npx', args: ['-y', pkg, ...(rt.args ?? [])], env };
      }
      case 'uvx': {
        const pkg = rt.version ? `${rt.package}@${rt.version}` : rt.package;
        return { command: 'uvx', args: [pkg, ...(rt.args ?? [])], env };
      }
      case 'command':
      default:
        return { command: rt.package, args: rt.args ?? [], env };
    }
  }, [server.local_runtime]);

  // Build the remote connect URL with a single fallback chain:
  //   1. mcp_endpoint (explicit full-URL override) - always wins
  //   2. proxy_pass_url (registry-only mode - client reaches the server directly)
  //   3. gateway URL = origin + base + server.path, optionally + "/mcp"
  // The trailing "/mcp" transport segment is auto-detected from proxy_pass_url
  // but can be forced on/off per-server via append_mcp_path (e.g. root-endpoint
  // servers like AWS Knowledge that serve MCP at the server path itself).
  const buildConnectUrl = useCallback(() => {
    if (server.mcp_endpoint) return server.mcp_endpoint;
    if (isRegistryOnly && server.proxy_pass_url) return server.proxy_pass_url;

    const baseUrl = `${window.location.origin}${getBaseURL()}`;
    const cleanPath = server.path.replace(/\/+$/, '').replace(/^\/+/, '/');
    const proxyUrl = server.proxy_pass_url || '';
    const hasMcpPath = /\/(mcp|sse|v1)(\/.*)?$/.test(proxyUrl);
    const shouldAppend = appendMcpPath ?? !hasMcpPath;
    return shouldAppend ? `${baseUrl}${cleanPath}/mcp` : `${baseUrl}${cleanPath}`;
  }, [server.mcp_endpoint, server.proxy_pass_url, server.path, isRegistryOnly, appendMcpPath]);

  const generateMCPConfig = useCallback(() => {
    const serverName = server.name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');

    // Local (stdio) servers: emit a launch recipe shaped per IDE.
    if (isLocal) {
      const spec = buildLocalLaunchSpec();
      if (!spec) {
        return { mcpServers: { [serverName]: { error: 'No local_runtime configured' } } };
      }
      switch (selectedIDE) {
        case 'roo-code':
          return {
            mcpServers: {
              [serverName]: { type: 'stdio', ...spec, disabled: false },
            },
          };
        case 'kiro':
          return {
            mcpServers: {
              [serverName]: { ...spec, disabled: false, autoApprove: [] },
            },
          };
        default:
          // Cursor, Claude Code: identical command/args/env shape.
          return { mcpServers: { [serverName]: spec } };
      }
    }

    const url = buildConnectUrl();

    // In registry-only mode, don't include gateway auth headers
    const includeAuthHeaders = !isRegistryOnly;

    // Use actual JWT token if available, otherwise show placeholder
    const authToken = jwtToken || '[YOUR_GATEWAY_AUTH_TOKEN]';

    // Build headers object: custom first, then auth_scheme, then gateway auth.
    // When the IDE handles login via OAuth (useOAuthLogin), the static gateway
    // token is omitted - the IDE obtains it through the OAuth/PKCE flow.
    // The server-auth header is dropped (default includeServerAuth) in two
    // cases, for every IDE that emits it: (1) OAuth-login mode, and (2)
    // egress-managed servers (egress_auth_mode == 'oauth_user'). In both the
    // gateway injects the upstream credential itself, so the client must NOT
    // send a server Authorization/API-key header (the [YOUR_SERVER_AUTH_TOKEN]
    // placeholder would be forwarded as-is and break the connection). IDEs that
    // can't run OAuth login (Roo Code / Kiro / VS Code default) still keep the
    // static gateway token, just not server auth.
    const buildHeaders = (
      includeGatewayToken = true,
      includeServerAuth = !useOAuthLogin && !egressManaged,
    ) => {
      const headers: Record<string, string> = {};

      // Custom headers go first so auth_scheme and gateway auth overwrite collisions
      for (const h of customHeaders) {
        headers[h.name] = h.value;
      }

      // Add server authentication headers if server requires auth
      if (includeServerAuth && server.auth_scheme && server.auth_scheme !== 'none') {
        if (server.auth_scheme === 'bearer') {
          headers['Authorization'] = 'Bearer [YOUR_SERVER_AUTH_TOKEN]';
        } else if (server.auth_scheme === 'api_key') {
          const headerName = server.auth_header_name || 'X-API-Key';
          headers[headerName] = '[YOUR_API_KEY]';
        }
      }

      // Add gateway authentication header last - cannot be overridden
      if (includeGatewayToken) {
        headers['X-Authorization'] = `Bearer ${authToken}`;
      }

      return headers;
    };

    switch (selectedIDE) {
      case 'cursor': {
        // OAuth login mode: advertise the pre-registered client_id and omit the
        // gateway token so Cursor renders a login button. The `auth.CLIENT_ID`
        // key (upper-snake) is Cursor's documented MCP OAuth config shape; it
        // intentionally differs from the lowercase keys elsewhere in this file.
        if (useOAuthLogin) {
          const oauthHeaders = buildHeaders(false, false);
          return {
            mcpServers: {
              [serverName]: {
                url,
                ...(Object.keys(oauthHeaders).length > 0 && {
                  headers: oauthHeaders,
                }),
                auth: { CLIENT_ID: oauthClientId },
              },
            },
          };
        }
        return {
          mcpServers: {
            [serverName]: {
              url,
              ...(includeAuthHeaders && {
                headers: buildHeaders(),
              }),
            },
          },
        };
      }
      // The IDEs below intentionally do NOT emit the OAuth-login (client_id)
      // config even when useOAuthLogin is true: they have no verified
      // fixed-public-client OAuth config syntax, so they keep the static-token
      // behavior. Only Cursor / Claude Code / Codex support the login config.
      case 'roo-code':
        return {
          mcpServers: {
            [serverName]: {
              type: 'streamable-http',
              url,
              disabled: false,
              ...(includeAuthHeaders && {
                headers: buildHeaders(),
              }),
            },
          },
        };
      case 'claude-code':
        return {
          mcpServers: {
            [serverName]: {
              type: 'http',
              url,
              ...(includeAuthHeaders && !isDCR && {
                headers: buildHeaders(),
              }),
            },
          },
        };
      case 'kiro':
        // Kiro supports Dynamic Client Registration (DCR): it discovers and
        // registers its own OAuth client against the gateway, so the config
        // only needs the server URL. No static bearer token, and no
        // disabled/autoApprove blocks.
        return {
          mcpServers: {
            [serverName]: {
              url,
            },
          },
        };
      case 'codex':
      case 'cli':
        return null;
      default:
        return {
          mcpServers: {
            [serverName]: {
              url,
              ...(includeAuthHeaders && {
                headers: buildHeaders(),
              }),
            },
          },
        };
    }
  }, [server.name, server.auth_scheme, server.auth_header_name, selectedIDE, isRegistryOnly, isDCR, useOAuthLogin, egressManaged, oauthClientId, jwtToken, customHeaders, buildConnectUrl]);

  const generateCodexCommand = useCallback(() => {
    const url = buildConnectUrl();

    const serverName = server.name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');

    let cmd = `codex mcp add "${serverName}" --url "${url}"`;

    if (useOAuthLogin) {
      // Codex runs OAuth/PKCE with the pre-registered public client id.
      cmd += ` --oauth-client-id "${oauthClientId}"`;
    } else if (!isDCR && !isRegistryOnly) {
      cmd += ' --bearer-token-env-var "MCP_AUTH_TOKEN"';
    }

    return cmd;
  }, [server.name, isRegistryOnly, isDCR, useOAuthLogin, oauthClientId, buildConnectUrl]);

  const generateCurlCommands = useCallback(() => {
    const url = buildConnectUrl();

    const headerLines: string[] = [
      '-H "Content-Type: application/json"',
      '-H "Accept: application/json, text/event-stream"',
    ];

    if (!isRegistryOnly) {
      const authToken = jwtToken || '[YOUR_TOKEN]';
      headerLines.push(`-H "X-Authorization: Bearer ${authToken}"`);
    }

    for (const h of customHeaders) {
      headerLines.push(`-H "${h.name}: ${h.value}"`);
    }

    const headers = headerLines.join(' \\\n  ');

    const initCmd = `curl -X POST "${url}" \\\n  ${headers} \\\n  -d '{"jsonrpc":"2.0","id":1,"method":"initialize","params":{"protocolVersion":"2025-03-26","capabilities":{},"clientInfo":{"name":"curl-client","version":"1.0.0"}}}'`;

    const toolsCmd = `curl -X POST "${url}" \\\n  ${headers} \\\n  -H "Mcp-Session-Id: <session-id-from-step-1>" \\\n  -d '{"jsonrpc":"2.0","id":2,"method":"tools/list","params":{}}'`;

    return { initCmd, toolsCmd };
  }, [isRegistryOnly, jwtToken, customHeaders, buildConnectUrl]);

  const generateGooseConfig = useCallback(() => {
    const serverName = server.name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');

    // Local (stdio) servers: emit Goose's stdio extension form. Build the
    // env block separately and concat — index-based splice into the lines
    // array silently breaks if the surrounding lines are reordered.
    if (isLocal) {
      const spec = buildLocalLaunchSpec();
      if (!spec) {
        return `# No local_runtime configured for ${serverName}`;
      }
      const envBlock: string[] = [];
      if (Object.keys(spec.env).length > 0) {
        envBlock.push('    envs:');
        for (const [k, v] of Object.entries(spec.env)) {
          envBlock.push(`      ${k}: ${JSON.stringify(v)}`);
        }
      }
      const lines = [
        'extensions:',
        `  ${serverName}:`,
        `    name: ${serverName}`,
        `    description: ${server.description}`,
        `    type: stdio`,
        `    cmd: ${spec.command}`,
        `    args: [${spec.args.map(a => JSON.stringify(a)).join(', ')}]`,
        ...envBlock,
        `    enabled: true`,
        `    timeout: 300`,
      ];
      return lines.join('\n');
    }

    const url = buildConnectUrl();

    const includeAuthHeaders = !isRegistryOnly;
    const authToken = jwtToken || '[YOUR_GATEWAY_AUTH_TOKEN]';

    const headerLines: string[] = [];
    // Custom headers first
    for (const h of customHeaders) {
      headerLines.push(`      ${h.name}: ${h.value}`);
    }
    // Server auth header. Skipped in OAuth-login mode: the gateway injects the
    // server's stored egress credential upstream, so the client must not send a
    // server Authorization/API-key header (the placeholder would break it).
    if (!useOAuthLogin && !egressManaged && server.auth_scheme && server.auth_scheme !== 'none') {
      if (server.auth_scheme === 'bearer') {
        headerLines.push(`      Authorization: Bearer [YOUR_SERVER_AUTH_TOKEN]`);
      } else if (server.auth_scheme === 'api_key') {
        const headerName = server.auth_header_name || 'X-API-Key';
        headerLines.push(`      ${headerName}: [YOUR_API_KEY]`);
      }
    }
    if (includeAuthHeaders) {
      headerLines.push(`      X-Authorization: Bearer ${authToken}`);
    }

    const lines = [
      'extensions:',
      `  ${serverName}:`,
      `    name: ${serverName}`,
      `    description: ${server.description}`,
      `    type: streamable_http`,
      `    uri: ${url}`,
      `    enabled: true`,
    ];
    if (headerLines.length > 0) {
      lines.push('    headers:');
      lines.push(...headerLines);
    }
    lines.push('    timeout: 300');

    return lines.join('\n');
  }, [server.name, server.auth_scheme, server.description, server.auth_header_name, isRegistryOnly, useOAuthLogin, egressManaged, jwtToken, customHeaders, buildConnectUrl]);

  const generateClaudeCodeCommand = useCallback(() => {
    const serverName = server.name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');

    // Optional install scope (local|project|user). When the operator sets
    // ide_connect_scope, emit `--scope <value>` so the copied command installs
    // the server at the intended scope (e.g. `user` = every project) instead of
    // Claude Code's default (local = current directory only). Empty => omitted.
    const scopeFlag = connectScope ? ` --scope ${connectScope}` : '';

    // Local (stdio) servers: emit `claude mcp add` with stdio transport.
    if (isLocal) {
      const spec = buildLocalLaunchSpec();
      if (!spec) {
        return `# No local_runtime configured for ${serverName}`;
      }
      const envFlags = Object.entries(spec.env)
        .map(([k, v]) => `-e ${k}=${JSON.stringify(v)}`)
        .join(' ');
      const argsStr = spec.args.map(a => JSON.stringify(a)).join(' ');
      let command = `claude mcp add${scopeFlag} ${serverName}`;
      if (envFlags) command += ` ${envFlags}`;
      command += ` -- ${spec.command}`;
      if (argsStr) command += ` ${argsStr}`;
      return command;
    }

    const url = buildConnectUrl();

    const includeAuthHeaders = !isRegistryOnly;
    const authToken = jwtToken || '[YOUR_GATEWAY_AUTH_TOKEN]';

    // Build command with headers. OAuth login mode passes the pre-registered
    // public client id (--client-id) so Claude Code runs OAuth/PKCE; no token.
    // A configured callback port is emitted as --callback-port so the IDE uses a
    // fixed loopback port (required for Okta/Entra/Cognito, which match the
    // redirect_uri literally including the port).
    let command = `claude mcp add${scopeFlag} --transport http`;
    if (useOAuthLogin) {
      command += ` --client-id ${oauthClientId}`;
      if (oauthCallbackPort) {
        command += ` --callback-port ${oauthCallbackPort}`;
      }
    }
    command += ` ${serverName} ${url}`;

    // Custom headers first
    for (const h of customHeaders) {
      command += ` \\\n  --header "${h.name}: ${h.value}"`;
    }

    // Server auth header. Skipped in OAuth-login mode: the gateway injects the
    // server's stored egress credential upstream, so the client must not send a
    // server Authorization/API-key header (the placeholder would break it).
    if (!useOAuthLogin && !egressManaged && server.auth_scheme && server.auth_scheme !== 'none') {
      if (server.auth_scheme === 'bearer') {
        command += ` \\\n  --header "Authorization: Bearer [YOUR_SERVER_AUTH_TOKEN]"`;
      } else if (server.auth_scheme === 'api_key') {
        const headerName = server.auth_header_name || 'X-API-Key';
        command += ` \\\n  --header "${headerName}: [YOUR_API_KEY]"`;
      }
    }

    // Gateway auth header last (skip when DCR or OAuth login handles it)
    if (includeAuthHeaders && !isDCR && !useOAuthLogin) {
      command += ` \\\n  --header "X-Authorization: Bearer ${authToken}"`;
    }

    return command;
  }, [server.name, server.auth_scheme, server.auth_header_name, isRegistryOnly, isDCR, useOAuthLogin, egressManaged, oauthClientId, oauthCallbackPort, connectScope, jwtToken, customHeaders, buildConnectUrl]);


  const copyConfigToClipboard = useCallback(async () => {
    try {
      const config = generateMCPConfig();
      const configText = JSON.stringify(config, null, 2);
      await navigator.clipboard.writeText(configText);

      // Show visual feedback
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);

      onShowToast?.('Configuration copied to clipboard!', 'success');
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      onShowToast?.('Failed to copy configuration', 'error');
    }
  }, [generateMCPConfig, onShowToast]);

  const copyGooseConfigToClipboard = useCallback(async () => {
    try {
      const configText = generateGooseConfig();
      await navigator.clipboard.writeText(configText);

      setCopied(true);
      setTimeout(() => setCopied(false), 2000);

      onShowToast?.('Configuration copied to clipboard!', 'success');
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      onShowToast?.('Failed to copy configuration', 'error');
    }
  }, [generateGooseConfig, onShowToast]);

  const copyCurlToClipboard = useCallback(async (text: string) => {
    try {
      await navigator.clipboard.writeText(text);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
      onShowToast?.('Command copied to clipboard!', 'success');
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      onShowToast?.('Failed to copy command', 'error');
    }
  }, [onShowToast]);

  const copyCommandToClipboard = useCallback(async () => {
    try {
      const command = generateClaudeCodeCommand();
      await navigator.clipboard.writeText(command);

      // Show visual feedback
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);

      onShowToast?.('Command copied to clipboard!', 'success');
    } catch (error) {
      console.error('Failed to copy to clipboard:', error);
      onShowToast?.('Failed to copy command', 'error');
    }
  }, [generateClaudeCodeCommand, onShowToast]);

  if (!isOpen) {
    return null;
  }

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-3xl w-full mx-4 max-h-[80vh] overflow-auto">
        <div className="flex items-center justify-between mb-4">
          <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
            MCP Configuration for {server.name}
          </h3>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          >
            ✕
          </button>
        </div>

        <div className="space-y-4">
          {/* Egress "Connect your account" callout (#1495): shown before the
              config the user copies, so per-user-auth servers stop silently
              returning "0 tools". Only rendered when eligible. */}
          {egressConnect && (
            <EgressConnectCallout
              serverPath={server.path}
              state={egressConnect}
              onEgressChanged={onEgressChanged}
              onShowToast={onShowToast}
            />
          )}

          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
            <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">
              How to use this configuration:
            </h4>
            <ol className="text-sm text-blue-800 dark:text-blue-200 space-y-1 list-decimal list-inside">
              <li>Copy the configuration below</li>
              <li>
                Paste it into your <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">mcp.json</code> file
              </li>
              {!isRegistryOnly && !jwtToken && (
                <li>
                  Replace <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">[YOUR_AUTH_TOKEN]</code> with your
                  gateway authentication token (or wait for auto-generation)
                </li>
              )}
              <li>Restart your AI coding assistant to load the new configuration</li>
            </ol>
          </div>

          {connectConfigError && (
            <div
              role="alert"
              className="bg-amber-50 dark:bg-amber-900/20 border border-amber-300 dark:border-amber-800 rounded-lg p-4"
            >
              <p className="text-sm text-amber-800 dark:text-amber-200">{connectConfigError}</p>
            </div>
          )}

          {isLocal ? (
            <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-4">
              <h4 className="font-medium text-purple-900 dark:text-purple-100 mb-2">Local Server</h4>
              <p className="text-sm text-purple-800 dark:text-purple-200">
                This server runs on your machine via stdio. The configuration below
                is a launch recipe — no gateway authentication needed.
              </p>
              {server.local_runtime?.required_env && server.local_runtime.required_env.length > 0 && (
                <div className="mt-3 p-3 rounded bg-yellow-50 dark:bg-yellow-900/30 border border-yellow-200 dark:border-yellow-800">
                  <p className="text-sm text-yellow-900 dark:text-yellow-100">
                    <strong>Action required:</strong> replace{' '}
                    <code className="bg-yellow-100 dark:bg-yellow-800 px-1 rounded">&lt;your-value&gt;</code>{' '}
                    in the <code className="bg-yellow-100 dark:bg-yellow-800 px-1 rounded">env</code> block
                    for these keys before pasting into your IDE config:{' '}
                    <code className="bg-yellow-100 dark:bg-yellow-800 px-1 rounded">
                      {server.local_runtime.required_env.join(', ')}
                    </code>
                  </p>
                </div>
              )}
            </div>
          ) : !isRegistryOnly ? (
            <div className={`border rounded-lg p-4 ${
              jwtToken
                ? 'bg-green-50 dark:bg-green-900/20 border-green-200 dark:border-green-800'
                : tokenError
                ? 'bg-red-50 dark:bg-red-900/20 border-red-200 dark:border-red-800'
                : 'bg-amber-50 dark:bg-amber-900/20 border-amber-200 dark:border-amber-800'
            }`}>
              <div className="flex items-center justify-between mb-2">
                <h4 className={`font-medium ${
                  jwtToken
                    ? 'text-green-900 dark:text-green-100'
                    : tokenError
                    ? 'text-red-900 dark:text-red-100'
                    : 'text-amber-900 dark:text-amber-100'
                }`}>
                  {tokenLoading
                    ? 'Fetching Token...'
                    : jwtToken
                    ? 'Token Ready - Copy and Paste!'
                    : tokenError
                    ? 'Token Generation Failed'
                    : 'Authentication Required'}
                </h4>
                {!tokenLoading && (
                  <button
                    onClick={fetchJwtToken}
                    className="flex items-center gap-1 px-2 py-1 text-xs bg-blue-600 hover:bg-blue-700 text-white rounded transition-colors"
                    title="Generate new token"
                  >
                    <KeyIcon className="h-3 w-3" />
                    {jwtToken ? 'Refresh' : 'Get Token'}
                  </button>
                )}
              </div>
              {tokenLoading ? (
                <p className="text-sm text-amber-800 dark:text-amber-200">
                  Generating JWT token for your configuration...
                </p>
              ) : jwtToken ? (
                <p className="text-sm text-green-800 dark:text-green-200">
                  JWT token has been automatically added to the configuration below. You can copy and paste it directly into your mcp.json file. Token expires in 8 hours.
                </p>
              ) : tokenError ? (
                <p className="text-sm text-red-800 dark:text-red-200">
                  {tokenError}. Click &quot;Get Token&quot; to retry, or manually replace [YOUR_AUTH_TOKEN] with your gateway token.
                </p>
              ) : (
                <p className="text-sm text-amber-800 dark:text-amber-200">
                  This configuration requires gateway authentication tokens. The tokens authenticate your AI assistant with
                  the MCP Gateway, not the individual server.
                </p>
              )}
            </div>
          ) : (
            <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
              <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">Direct Connection Mode</h4>
              <p className="text-sm text-blue-800 dark:text-blue-200">
                This registry operates in catalog-only mode. The configuration connects directly to the MCP server
                endpoint without going through a gateway proxy.
              </p>
              <p className="text-sm text-blue-800 dark:text-blue-200 mt-2">
                <strong>Note:</strong> The MCP server may still require authentication (API key, auth header, etc.).
                Check the server's documentation to determine if any credentials are needed.
              </p>
            </div>
          )}

          {server.mcp_endpoint && (
            <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-4">
              <h4 className="font-medium text-purple-900 dark:text-purple-100 mb-2">Custom Endpoint Configured</h4>
              <p className="text-sm text-purple-800 dark:text-purple-200">
                This server uses a custom MCP endpoint:{' '}
                <code className="bg-purple-100 dark:bg-purple-800 px-1 rounded break-all">{server.mcp_endpoint}</code>
              </p>
            </div>
          )}

          <div className="bg-gray-50 dark:bg-gray-900 border dark:border-gray-700 rounded-lg p-4">
            <h4 className="font-medium text-gray-900 dark:text-white mb-3">Select your IDE/Tool:</h4>
            <div className="flex flex-wrap gap-2">
              {enabledIDEs.map((ide) => (
                <button
                  key={ide}
                  onClick={() => setSelectedIDE(ide)}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    selectedIDE === ide
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-800'
                  }`}
                >
                  {IDE_LABELS[ide]}
                </button>
              ))}
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              Configuration format optimized for {IDE_LABELS[selectedIDE]} integration
            </p>
          </div>

          {selectedIDE === 'claude-code' ? (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="font-medium text-gray-900 dark:text-white">CLI Command:</h4>
                <button
                  onClick={copyCommandToClipboard}
                  className={`flex items-center gap-2 px-3 py-2 text-white rounded-lg transition-colors duration-200 ${
                    copied
                      ? 'bg-green-700'
                      : 'bg-green-600 hover:bg-green-700'
                  }`}
                >
                  <ClipboardDocumentIcon className="h-4 w-4" />
                  {copied ? 'Copied!' : 'Copy Command'}
                </button>
              </div>
              <pre className="bg-gray-900 text-green-100 p-4 rounded-lg text-sm overflow-x-auto whitespace-pre-wrap break-all">
                {generateClaudeCodeCommand()}
              </pre>
              <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                Run this command in your terminal to add the MCP server to Claude Code.
                {useOAuthLogin && ' It passes the pre-registered public client id (--client-id) ' +
                  'so Claude Code runs the OAuth login flow — no gateway token is embedded.'}
                {useOAuthLogin && oauthCallbackPort ? ' --callback-port pins the OAuth redirect ' +
                  'port so it matches what the identity provider has registered.' : ''}
                {connectScope ? ` --scope ${connectScope} installs the server at the "${connectScope}" ` +
                  'scope so it is available beyond the current directory.' : ''}
              </p>
            </div>
          ) : selectedIDE === 'codex' ? (
            <div className="space-y-2">
              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-3">
                <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">Codex CLI:</h4>
                <p className="text-sm text-blue-800 dark:text-blue-200">
                  Run this command in your terminal to add the MCP server to Codex.
                  {useOAuthLogin
                    ? ' It registers the pre-registered public client id (--oauth-client-id) ' +
                      'so Codex runs the OAuth login flow — no gateway token is embedded.'
                    : isDCR && ' OAuth authentication is handled automatically via DCR.'}
                </p>
                {useOAuthLogin && oauthCallbackPort ? (
                  <p className="text-xs text-amber-800 dark:text-amber-300 mt-2">
                    Note: Codex does not support pinning the OAuth callback port, so it uses a
                    random loopback port. If your identity provider requires the redirect URI
                    (including the port) to be pre-registered (e.g. Okta, Entra, Cognito), Codex
                    login may fail. Claude Code supports a fixed port via --callback-port.
                  </p>
                ) : null}
              </div>
              <div className="flex items-center justify-between">
                <h4 className="font-medium text-gray-900 dark:text-white">Command:</h4>
                <button
                  onClick={() => copyCurlToClipboard(generateCodexCommand())}
                  className={`flex items-center gap-2 px-3 py-2 text-white rounded-lg transition-colors duration-200 ${
                    copied ? 'bg-green-700' : 'bg-green-600 hover:bg-green-700'
                  }`}
                >
                  <ClipboardDocumentIcon className="h-4 w-4" />
                  {copied ? 'Copied!' : 'Copy Command'}
                </button>
              </div>
              <pre className="bg-gray-900 text-green-100 p-4 rounded-lg text-sm overflow-x-auto whitespace-pre-wrap break-all">
                {generateCodexCommand()}
              </pre>
              {!isDCR && !isRegistryOnly && !useOAuthLogin && (
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                  Set the <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">MCP_AUTH_TOKEN</code> environment variable to your JWT token before running Codex.
                </p>
              )}
            </div>
          ) : selectedIDE === 'cli' ? (
            <div className="space-y-4">
              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-3">
                <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">MCP Protocol via curl:</h4>
                <p className="text-sm text-blue-800 dark:text-blue-200">
                  Use these commands to interact with the MCP server directly. Copy the{' '}
                  <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">Mcp-Session-Id</code>{' '}
                  response header from Step 1 into Step 2.
                </p>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-gray-900 dark:text-white">Step 1: Initialize session</h4>
                  <button
                    onClick={() => copyCurlToClipboard(generateCurlCommands().initCmd)}
                    className={`flex items-center gap-2 px-3 py-1.5 text-sm text-white rounded-lg transition-colors duration-200 ${
                      copied ? 'bg-green-700' : 'bg-green-600 hover:bg-green-700'
                    }`}
                  >
                    <ClipboardDocumentIcon className="h-3 w-3" />
                    {copied ? 'Copied!' : 'Copy'}
                  </button>
                </div>
                <pre className="bg-gray-900 text-green-100 p-4 rounded-lg text-sm overflow-x-auto whitespace-pre-wrap">
                  {generateCurlCommands().initCmd}
                </pre>
              </div>
              <div>
                <div className="flex items-center justify-between mb-2">
                  <h4 className="font-medium text-gray-900 dark:text-white">Step 2: List tools</h4>
                  <button
                    onClick={() => copyCurlToClipboard(generateCurlCommands().toolsCmd)}
                    className={`flex items-center gap-2 px-3 py-1.5 text-sm text-white rounded-lg transition-colors duration-200 ${
                      copied ? 'bg-green-700' : 'bg-green-600 hover:bg-green-700'
                    }`}
                  >
                    <ClipboardDocumentIcon className="h-3 w-3" />
                    {copied ? 'Copied!' : 'Copy'}
                  </button>
                </div>
                <pre className="bg-gray-900 text-green-100 p-4 rounded-lg text-sm overflow-x-auto whitespace-pre-wrap">
                  {generateCurlCommands().toolsCmd}
                </pre>
              </div>
            </div>
          ) : selectedIDE === 'goose' ? (
            <div className="space-y-2">
              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-3">
                <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">Goose Configuration:</h4>
                <p className="text-sm text-blue-800 dark:text-blue-200">
                  Copy the YAML below and merge it into{' '}
                  <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">~/.config/goose/config.yaml</code>{' '}
                  under the <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">extensions:</code> key. If an{' '}
                  <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">extensions:</code> block already exists, add this entry underneath it.
                </p>
              </div>
              <div className="flex items-center justify-between">
                <h4 className="font-medium text-gray-900 dark:text-white">Configuration YAML:</h4>
                <button
                  onClick={copyGooseConfigToClipboard}
                  className={`flex items-center gap-2 px-3 py-2 text-white rounded-lg transition-colors duration-200 ${
                    copied
                      ? 'bg-green-700'
                      : 'bg-green-600 hover:bg-green-700'
                  }`}
                >
                  <ClipboardDocumentIcon className="h-4 w-4" />
                  {copied ? 'Copied!' : 'Copy to Clipboard'}
                </button>
              </div>
              <pre className="bg-gray-900 text-green-100 p-4 rounded-lg text-sm overflow-x-auto">
                {generateGooseConfig()}
              </pre>
            </div>
          ) : selectedIDE === 'kiro' ? (
            <div className="space-y-2">
              <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4 mb-3">
                <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">Kiro Configuration:</h4>
                <p className="text-sm text-blue-800 dark:text-blue-200">
                  Copy the JSON below and paste it into{' '}
                  <code className="bg-blue-100 dark:bg-blue-800 px-1 rounded">~/.kiro/settings/mcp.json</code>
                </p>
              </div>
              <div className="flex items-center justify-between">
                <h4 className="font-medium text-gray-900 dark:text-white">Configuration JSON:</h4>
                <button
                  onClick={copyConfigToClipboard}
                  className={`flex items-center gap-2 px-3 py-2 text-white rounded-lg transition-colors duration-200 ${
                    copied
                      ? 'bg-green-700'
                      : 'bg-green-600 hover:bg-green-700'
                  }`}
                >
                  <ClipboardDocumentIcon className="h-4 w-4" />
                  {copied ? 'Copied!' : 'Copy to Clipboard'}
                </button>
              </div>
              <pre className="bg-gray-900 text-green-100 p-4 rounded-lg text-sm overflow-x-auto">
                {JSON.stringify(generateMCPConfig(), null, 2)}
              </pre>
            </div>
          ) : (
            <div className="space-y-2">
              <div className="flex items-center justify-between">
                <h4 className="font-medium text-gray-900 dark:text-white">Configuration JSON:</h4>
                <button
                  onClick={copyConfigToClipboard}
                  className={`flex items-center gap-2 px-3 py-2 text-white rounded-lg transition-colors duration-200 ${
                    copied
                      ? 'bg-green-700'
                      : 'bg-green-600 hover:bg-green-700'
                  }`}
                >
                  <ClipboardDocumentIcon className="h-4 w-4" />
                  {copied ? 'Copied!' : 'Copy to Clipboard'}
                </button>
              </div>
              <pre className="bg-gray-900 text-green-100 p-4 rounded-lg text-sm overflow-x-auto">
                {JSON.stringify(generateMCPConfig(), null, 2)}
              </pre>
              {useOAuthLogin && selectedIDE === 'cursor' && (
                <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
                  This config uses OAuth login: no gateway token is embedded.
                  Your IDE shows a login button and authenticates via the
                  gateway's OAuth flow using the pre-registered client.
                </p>
              )}
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ServerConfigModal;
