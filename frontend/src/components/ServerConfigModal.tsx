import React, { useCallback, useState, useEffect } from 'react';
import { ClipboardDocumentIcon, KeyIcon } from '@heroicons/react/24/outline';
import axios from 'axios';
import type { Server } from './ServerCard';
import { useRegistryConfig } from '../hooks/useRegistryConfig';
import useEscapeKey from '../hooks/useEscapeKey';
import { getBaseURL } from '../utils/basePath';

const IDE_LABELS = {
  'cursor': 'Cursor',
  'roo-code': 'Roo Code',
  'claude-code': 'Claude Code',
  'kiro': 'Kiro',
  'goose': 'Goose',
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
}

const ServerConfigModal: React.FC<ServerConfigModalProps> = ({
  server,
  isOpen,
  onClose,
  onShowToast,
  resourceType = 'server',
}) => {
  const [jwtToken, setJwtToken] = useState<string | null>(null);
  const [tokenLoading, setTokenLoading] = useState(false);
  const [tokenError, setTokenError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const { config: registryConfig, loading: configLoading } = useRegistryConfig();

  const enabledIDEs: IDE[] = React.useMemo(() => {
    const allIDEs = Object.keys(IDE_LABELS) as IDE[];
    const allowlist = registryConfig?.coding_assistants ?? [];
    if (allowlist.length === 0) return allIDEs;
    const filtered = allIDEs.filter((ide) => allowlist.includes(ide));
    return filtered.length > 0 ? filtered : allIDEs;
  }, [registryConfig?.coding_assistants]);

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

  // Custom headers from connect-config endpoint
  const [customHeaders, setCustomHeaders] = useState<Array<{name: string; value: string}>>([]);
  const [connectConfigError, setConnectConfigError] = useState<string | null>(null);

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
        if (resp.data.decrypt_failures > 0) {
          setConnectConfigError(
            `${resp.data.decrypt_failures} custom header(s) could not be decrypted.`
          );
        }
      })
      .catch((err) => {
        console.error("Failed to fetch connect config", err);
        setConnectConfigError(
          "Could not load custom headers for this server. " +
          "The copied configuration may be missing headers your server requires."
        );
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
      if (server.path) {
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

    // URL determination with fallback chain:
    // 1. mcp_endpoint (custom override) - always takes precedence
    // 2. proxy_pass_url (in registry-only mode)
    // 3. Constructed gateway URL (default/fallback)
    let url: string;

    if (server.mcp_endpoint) {
      url = server.mcp_endpoint;
    } else if (isRegistryOnly && server.proxy_pass_url) {
      url = server.proxy_pass_url;
    } else {
      // Gateway URL = origin + ROOT_PATH + server.path + "/mcp".
      // getBaseURL() returns the registry's ROOT_PATH (e.g. "/registry"
      // in path routing mode, "" in subdomain mode), read from the
      // <base> tag the server injected into index.html.
      const baseUrl = `${window.location.origin}${getBaseURL()}`;
      const cleanPath = server.path.replace(/\/+$/, '').replace(/^\/+/, '/');
      url = `${baseUrl}${cleanPath}/mcp`;
    }

    // In registry-only mode, don't include gateway auth headers
    const includeAuthHeaders = !isRegistryOnly;

    // Use actual JWT token if available, otherwise show placeholder
    const authToken = jwtToken || '[YOUR_GATEWAY_AUTH_TOKEN]';

    // Build headers object: custom first, then auth_scheme, then gateway auth
    const buildHeaders = () => {
      const headers: Record<string, string> = {};

      // Custom headers go first so auth_scheme and gateway auth overwrite collisions
      for (const h of customHeaders) {
        headers[h.name] = h.value;
      }

      // Add server authentication headers if server requires auth
      if (server.auth_scheme && server.auth_scheme !== 'none') {
        if (server.auth_scheme === 'bearer') {
          headers['Authorization'] = 'Bearer [YOUR_SERVER_AUTH_TOKEN]';
        } else if (server.auth_scheme === 'api_key') {
          const headerName = server.auth_header_name || 'X-API-Key';
          headers[headerName] = '[YOUR_API_KEY]';
        }
      }

      // Add gateway authentication header last - cannot be overridden
      headers['X-Authorization'] = `Bearer ${authToken}`;

      return headers;
    };

    switch (selectedIDE) {
      case 'cursor':
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
              ...(includeAuthHeaders && {
                headers: buildHeaders(),
              }),
            },
          },
        };
      case 'kiro':
        return {
          mcpServers: {
            [serverName]: {
              url,
              ...(includeAuthHeaders && {
                headers: buildHeaders(),
              }),
              disabled: false,
              autoApprove: [],
            },
          },
        };
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
  }, [server.name, server.path, server.proxy_pass_url, server.mcp_endpoint, server.auth_scheme, server.auth_header_name, selectedIDE, isRegistryOnly, jwtToken, customHeaders]);

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

    let url: string;
    if (server.mcp_endpoint) {
      url = server.mcp_endpoint;
    } else if (isRegistryOnly && server.proxy_pass_url) {
      url = server.proxy_pass_url;
    } else {
      const baseUrl = `${window.location.origin}${getBaseURL()}`;
      const cleanPath = server.path.replace(/\/+$/, '').replace(/^\/+/, '/');
      url = `${baseUrl}${cleanPath}/mcp`;
    }

    const includeAuthHeaders = !isRegistryOnly;
    const authToken = jwtToken || '[YOUR_GATEWAY_AUTH_TOKEN]';

    const headerLines: string[] = [];
    // Custom headers first
    for (const h of customHeaders) {
      headerLines.push(`      ${h.name}: ${h.value}`);
    }
    if (server.auth_scheme && server.auth_scheme !== 'none') {
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
  }, [server.name, server.mcp_endpoint, server.proxy_pass_url, server.auth_scheme, server.description, server.path, server.auth_header_name, isRegistryOnly, jwtToken, customHeaders]);

  const generateClaudeCodeCommand = useCallback(() => {
    const serverName = server.name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');

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
      let command = `claude mcp add ${serverName}`;
      if (envFlags) command += ` ${envFlags}`;
      command += ` -- ${spec.command}`;
      if (argsStr) command += ` ${argsStr}`;
      return command;
    }

    // URL determination (same logic as generateMCPConfig)
    let url: string;
    if (server.mcp_endpoint) {
      url = server.mcp_endpoint;
    } else if (isRegistryOnly && server.proxy_pass_url) {
      url = server.proxy_pass_url;
    } else {
      const baseUrl = `${window.location.origin}${getBaseURL()}`;
      const cleanPath = server.path.replace(/\/+$/, '').replace(/^\/+/, '/');
      url = `${baseUrl}${cleanPath}/mcp`;
    }

    const includeAuthHeaders = !isRegistryOnly;
    const authToken = jwtToken || '[YOUR_GATEWAY_AUTH_TOKEN]';

    // Build command with headers
    let command = `claude mcp add --transport http ${serverName} ${url}`;

    // Custom headers first
    for (const h of customHeaders) {
      command += ` \\\n  --header "${h.name}: ${h.value}"`;
    }

    // Server auth header
    if (server.auth_scheme && server.auth_scheme !== 'none') {
      if (server.auth_scheme === 'bearer') {
        command += ` \\\n  --header "Authorization: Bearer [YOUR_SERVER_AUTH_TOKEN]"`;
      } else if (server.auth_scheme === 'api_key') {
        const headerName = server.auth_header_name || 'X-API-Key';
        command += ` \\\n  --header "${headerName}: [YOUR_API_KEY]"`;
      }
    }

    // Gateway auth header last
    if (includeAuthHeaders) {
      command += ` \\\n  --header "X-Authorization: Bearer ${authToken}"`;
    }

    return command;
  }, [server.name, server.path, server.proxy_pass_url, server.mcp_endpoint, server.auth_scheme, server.auth_header_name, isRegistryOnly, jwtToken, customHeaders]);


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
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
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
              </p>
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
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default ServerConfigModal;
