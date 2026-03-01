import React, { useCallback, useState, useEffect } from 'react';
import { ClipboardDocumentIcon, KeyIcon } from '@heroicons/react/24/outline';
import axios from 'axios';
import type { Server } from './ServerCard';
import { useRegistryConfig } from '../hooks/useRegistryConfig';

type IDE = 'cursor' | 'roo-code' | 'claude-code' | 'kiro';

interface ServerConfigModalProps {
  server: Server;
  isOpen: boolean;
  onClose: () => void;
  onShowToast?: (message: string, type: 'success' | 'error') => void;
}

const ServerConfigModal: React.FC<ServerConfigModalProps> = ({
  server,
  isOpen,
  onClose,
  onShowToast,
}) => {
  const [selectedIDE, setSelectedIDE] = useState<IDE>('cursor');
  const [jwtToken, setJwtToken] = useState<string | null>(null);
  const [tokenLoading, setTokenLoading] = useState(false);
  const [tokenError, setTokenError] = useState<string | null>(null);
  const [copied, setCopied] = useState(false);
  const { config: registryConfig, loading: configLoading } = useRegistryConfig();

  // Determine if we're in registry-only mode
  // While config is loading, default to with-gateway behavior (safer default)
  const isRegistryOnly = !configLoading && registryConfig?.deployment_mode === 'registry-only';

  // Fetch JWT token when modal opens (only in gateway mode)
  // We intentionally only depend on isOpen and isRegistryOnly to fetch once per modal open
  useEffect(() => {
    if (isOpen && !isRegistryOnly) {
      // Reset token state when modal opens
      setJwtToken(null);
      setTokenError(null);
      fetchJwtToken();
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [isOpen, isRegistryOnly]);

  const fetchJwtToken = async () => {
    setTokenLoading(true);
    setTokenError(null);
    try {
      const response = await axios.post('/api/tokens/generate', {
        description: 'Generated for MCP configuration',
        expires_in_hours: 8,
      }, {
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

  const generateMCPConfig = useCallback(() => {
    const serverName = server.name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');

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
      const currentUrl = new URL(window.location.origin);
      const baseUrl = `${currentUrl.protocol}//${currentUrl.hostname}`;
      const cleanPath = server.path.replace(/\/+$/, '').replace(/^\/+/, '/');
      url = `${baseUrl}${cleanPath}/mcp`;
    }

    // In registry-only mode, don't include gateway auth headers
    const includeAuthHeaders = !isRegistryOnly;

    // Use actual JWT token if available, otherwise show placeholder
    const authToken = jwtToken || '[YOUR_GATEWAY_AUTH_TOKEN]';

    // Build headers object with both gateway auth and server auth (if applicable)
    const buildHeaders = () => {
      const headers: Record<string, string> = {};

      // Add gateway authentication header
      headers['X-Authorization'] = `Bearer ${authToken}`;

      // Add server authentication headers if server requires auth
      if (server.auth_scheme && server.auth_scheme !== 'none') {
        if (server.auth_scheme === 'bearer') {
          headers['Authorization'] = 'Bearer [YOUR_SERVER_AUTH_TOKEN]';
        } else if (server.auth_scheme === 'api_key') {
          const headerName = server.auth_header_name || 'X-API-Key';
          headers[headerName] = '[YOUR_API_KEY]';
        }
      }

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
  }, [server.name, server.path, server.proxy_pass_url, server.mcp_endpoint, server.auth_scheme, server.auth_header_name, selectedIDE, isRegistryOnly, jwtToken]);

  const generateClaudeCodeCommand = useCallback(() => {
    const serverName = server.name.toLowerCase().replace(/\s+/g, '-').replace(/[^a-z0-9-]/g, '');

    // URL determination (same logic as generateMCPConfig)
    let url: string;
    if (server.mcp_endpoint) {
      url = server.mcp_endpoint;
    } else if (isRegistryOnly && server.proxy_pass_url) {
      url = server.proxy_pass_url;
    } else {
      const currentUrl = new URL(window.location.origin);
      const baseUrl = `${currentUrl.protocol}//${currentUrl.hostname}`;
      const cleanPath = server.path.replace(/\/+$/, '').replace(/^\/+/, '/');
      url = `${baseUrl}${cleanPath}/mcp`;
    }

    const includeAuthHeaders = !isRegistryOnly;
    const authToken = jwtToken || '[YOUR_GATEWAY_AUTH_TOKEN]';

    // Build command with headers
    let command = `claude mcp add --transport http ${serverName} ${url}`;

    if (includeAuthHeaders) {
      // Add gateway auth header
      command += ` \\\n  --header "X-Authorization: Bearer ${authToken}"`;

      // Add server auth header if applicable
      if (server.auth_scheme && server.auth_scheme !== 'none') {
        if (server.auth_scheme === 'bearer') {
          command += ` \\\n  --header "Authorization: Bearer [YOUR_SERVER_AUTH_TOKEN]"`;
        } else if (server.auth_scheme === 'api_key') {
          const headerName = server.auth_header_name || 'X-API-Key';
          command += ` \\\n  --header "${headerName}: [YOUR_API_KEY]"`;
        }
      }
    }

    return command;
  }, [server.name, server.path, server.proxy_pass_url, server.mcp_endpoint, server.auth_scheme, server.auth_header_name, isRegistryOnly, jwtToken]);


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

          {!isRegistryOnly ? (
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
              {(['cursor', 'roo-code', 'claude-code', 'kiro'] as IDE[]).map((ide) => (
                <button
                  key={ide}
                  onClick={() => setSelectedIDE(ide)}
                  className={`px-3 py-2 rounded-lg text-sm font-medium transition-colors ${
                    selectedIDE === ide
                      ? 'bg-blue-600 text-white'
                      : 'bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 hover:bg-gray-300 dark:hover:bg-gray-600'
                  }`}
                >
                  {ide === 'cursor'
                    ? 'Cursor'
                    : ide === 'roo-code'
                    ? 'Roo Code'
                    : ide === 'claude-code'
                    ? 'Claude Code'
                    : 'Kiro'}
                </button>
              ))}
            </div>
            <p className="text-xs text-gray-600 dark:text-gray-400 mt-2">
              Configuration format optimized for{' '}
              {selectedIDE === 'cursor'
                ? 'Cursor'
                : selectedIDE === 'roo-code'
                ? 'Roo Code'
                : selectedIDE === 'claude-code'
                ? 'Claude Code'
                : 'Kiro'}{' '}
              integration
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
