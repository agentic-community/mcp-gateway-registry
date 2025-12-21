/**
 * UpstreamCredentialModal - Modal for creating/managing upstream credentials
 *
 * Supports:
 * - API key and static JWT credential types (secrets shown once after creation)
 * - OAuth/OIDC/Provider OAuth types (connect/disconnect flows)
 */

import { useState, useEffect } from 'react';
import { Modal, ModalFooter } from '@/components/ui/Modal';
import { Button, Input, Checkbox } from '@/components/ui';
import { CopyButton } from '@/components/ui/CopyButton';
import { Badge } from '@/components/ui/Badge';
import { Textarea } from '@/components/ui/Textarea';
import { useToast } from '@/components/ui/Toast';
import {
  KeyIcon,
  ExclamationTriangleIcon,
  CheckCircleIcon,
  ArrowDownTrayIcon,
  ClipboardDocumentIcon,
  ServerIcon,
  TrashIcon,
  ArrowTopRightOnSquareIcon,
  LinkIcon,
  LinkSlashIcon,
} from '@heroicons/react/24/outline';
import {
  createUpstreamCredential,
  getUpstreamCredential,
  revokeUpstreamCredential,
  getUpstreamOAuthCredential,
  startUpstreamOAuth,
  disconnectUpstreamOAuth,
} from '@/api/enforceai';
import type {
  Server,
  UpstreamCredential,
  CreateUpstreamCredentialResponse,
  UpstreamAuthType,
} from '@/api/types';

// ============================================================================
// Types
// ============================================================================

export interface UpstreamCredentialModalProps {
  open: boolean;
  server: Server;
  onClose: () => void;
  onSuccess: () => void;
}

type ModalView =
  | 'loading'
  | 'form'
  | 'secret'
  | 'view'
  | 'confirm-revoke'
  | 'oauth-connect'
  | 'oauth-view'
  | 'oauth-disconnect';

// ============================================================================
// Helper Functions
// ============================================================================

function getAuthTypeLabel(type: UpstreamAuthType): string {
  switch (type) {
    case 'api-key':
      return 'API Key';
    case 'jwt':
      return 'JWT Token';
    case 'oauth2':
      return 'OAuth 2.0';
    case 'oidc':
      return 'OIDC';
    case 'provider-oauth':
      return 'Provider OAuth';
    default:
      return type;
  }
}

function isOAuthType(type: UpstreamAuthType): boolean {
  return type === 'oauth2' || type === 'oidc' || type === 'provider-oauth';
}

function getSecretPlaceholder(type: UpstreamAuthType): string {
  switch (type) {
    case 'api-key':
      return 'Enter your API key (e.g., sk-xxx...)';
    case 'jwt':
      return 'Enter your JWT token (e.g., eyJhbGc...)';
    default:
      return 'Enter secret value';
  }
}

function getSecretLabel(type: UpstreamAuthType): string {
  switch (type) {
    case 'api-key':
      return 'API Key';
    case 'jwt':
      return 'JWT Token';
    default:
      return 'Secret';
  }
}

// ============================================================================
// Component
// ============================================================================

export function UpstreamCredentialModal({
  open,
  server,
  onClose,
  onSuccess,
}: UpstreamCredentialModalProps) {
  const { addToast } = useToast();

  // State
  const [view, setView] = useState<ModalView>('loading');
  const [existingCredential, setExistingCredential] = useState<UpstreamCredential | null>(null);
  const [oauthCredential, setOAuthCredential] = useState<UpstreamCredential | null>(null);
  const [createdCredential, setCreatedCredential] = useState<CreateUpstreamCredentialResponse | null>(null);
  const [acknowledged, setAcknowledged] = useState(false);
  const [isSubmitting, setIsSubmitting] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);

  // Form state
  const [secretValue, setSecretValue] = useState('');
  const [expiresAt, setExpiresAt] = useState('');
  const [errors, setErrors] = useState<{ secret?: string; expiresAt?: string }>({});

  const authType = server.upstream_auth?.type || 'api-key';
  const isApiKeyOrJwt = authType === 'api-key' || authType === 'jwt';
  const isOAuth = isOAuthType(authType);
  const provider = server.upstream_auth?.provider;
  const credentialBinding = server.upstream_auth?.credential_binding || 'user';

  const extractCreatedSecret = (): string => {
    const secretPayload = createdCredential?.secret_payload;
    if (!secretPayload) {
      return '';
    }

    if (authType === 'api-key') {
      const apiKey = (secretPayload as Record<string, unknown>)['api_key'];
      return typeof apiKey === 'string' ? apiKey : '';
    }

    if (authType === 'jwt') {
      const token =
        (secretPayload as Record<string, unknown>)['token'] ??
        (secretPayload as Record<string, unknown>)['jwt'];
      return typeof token === 'string' ? token : '';
    }

    return '';
  };

  // Load existing credential when modal opens
  useEffect(() => {
    if (open && server.path) {
      loadCredential();
    }
  }, [open, server.path]);

  // Reset state when modal closes
  useEffect(() => {
    if (!open) {
      setView('loading');
      setExistingCredential(null);
      setOAuthCredential(null);
      setCreatedCredential(null);
      setAcknowledged(false);
      setIsConnecting(false);
      setSecretValue('');
      setExpiresAt('');
      setErrors({});
    }
  }, [open]);

  const loadCredential = async () => {
    setView('loading');
    try {
      if (isOAuth) {
        // Load OAuth credential
        const credential = await getUpstreamOAuthCredential(server.path, {
          credential_type: authType as 'oauth2' | 'oidc' | 'provider-oauth',
          provider: provider || undefined,
        });
        setOAuthCredential(credential);
        setView(credential ? 'oauth-view' : 'oauth-connect');
      } else {
        // Load API key or JWT credential
        const credential = await getUpstreamCredential(server.path);
        setExistingCredential(credential);
        setView(credential ? 'view' : 'form');
      }
    } catch (err: any) {
      console.error('Failed to load credential:', err);
      // If we fail to load, show connect/form (assume no credential)
      setView(isOAuth ? 'oauth-connect' : 'form');
    }
  };

  // Validation
  const validateForm = (): boolean => {
    const newErrors: { secret?: string; expiresAt?: string } = {};

    if (!secretValue.trim()) {
      newErrors.secret = `${getSecretLabel(authType as UpstreamAuthType)} is required`;
    }

    if (expiresAt) {
      const selectedDate = new Date(expiresAt);
      if (selectedDate <= new Date()) {
        newErrors.expiresAt = 'Expiration date must be in the future';
      }
    }

    setErrors(newErrors);
    return Object.keys(newErrors).length === 0;
  };

  // Handlers
  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!validateForm()) {
      return;
    }

    setIsSubmitting(true);

    try {
      const secretPayload =
        authType === 'api-key'
          ? { api_key: secretValue.trim() }
          : { token: secretValue.trim() };

      const result = await createUpstreamCredential(server.path, {
        credential_type: authType as 'api-key' | 'jwt',
        credential_binding: credentialBinding,
        expires_at: expiresAt ? new Date(expiresAt).toISOString() : null,
        secret_payload: secretPayload,
      });

      setCreatedCredential(result);
      setView('secret');

      addToast({
        type: 'success',
        title: 'Credential configured',
        message: `Upstream ${getAuthTypeLabel(authType as UpstreamAuthType)} has been saved`,
      });
    } catch (err: any) {
      console.error('Failed to create credential:', err);
      addToast({
        type: 'error',
        title: 'Failed to save credential',
        message: err.message || 'An error occurred',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleRevoke = async () => {
    if (!existingCredential) return;

    setIsSubmitting(true);

    try {
      await revokeUpstreamCredential(existingCredential.credential_id);

      addToast({
        type: 'success',
        title: 'Credential revoked',
        message: 'The upstream credential has been revoked',
      });

      onSuccess();
      onClose();
    } catch (err: any) {
      console.error('Failed to revoke credential:', err);
      addToast({
        type: 'error',
        title: 'Failed to revoke credential',
        message: err.message || 'An error occurred',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const handleClose = () => {
    // Don't allow closing without acknowledgment if showing secret
    if (view === 'secret' && !acknowledged) {
      return;
    }
    onClose();
  };

  const handleDone = () => {
    onSuccess();
    onClose();
  };

  const handleConfigureNew = () => {
    setExistingCredential(null);
    setSecretValue('');
    setExpiresAt('');
    setErrors({});
    setView('form');
  };

  // OAuth handlers
  const handleOAuthConnect = async () => {
    setIsConnecting(true);

    try {
      if (!provider) {
        addToast({
          type: 'error',
          title: 'Missing provider',
          message: 'This server requires a provider identifier for OAuth.',
        });
        setIsConnecting(false);
        return;
      }

      if (credentialBinding !== 'user') {
        addToast({
          type: 'error',
          title: 'Unsupported binding',
          message: `OAuth binding '${credentialBinding}' is not supported in the UI yet.`,
        });
        setIsConnecting(false);
        return;
      }

      const uiReturnUrl = `/credentials/upstream/oauth/callback?server_path=${encodeURIComponent(
        server.path
      )}`;

      const result = await startUpstreamOAuth(server.path, {
        credential_type: authType as 'oauth2' | 'oidc' | 'provider-oauth',
        credential_binding: 'user',
        provider,
        ui_return_url: uiReturnUrl,
      });

      // Redirect to provider authorization URL
      window.location.href = result.authorization_url;
    } catch (err: any) {
      console.error('Failed to start OAuth flow:', err);
      addToast({
        type: 'error',
        title: 'Failed to start OAuth flow',
        message: err.message || 'An error occurred',
      });
      setIsConnecting(false);
    }
  };

  const handleOAuthDisconnect = async () => {
    setIsSubmitting(true);

    try {
      if (!provider) {
        addToast({
          type: 'error',
          title: 'Missing provider',
          message: 'This server requires a provider identifier for OAuth.',
        });
        return;
      }

      if (credentialBinding !== 'user') {
        addToast({
          type: 'error',
          title: 'Unsupported binding',
          message: `OAuth binding '${credentialBinding}' is not supported in the UI yet.`,
        });
        return;
      }

      await disconnectUpstreamOAuth(server.path, {
        credential_type: authType as 'oauth2' | 'oidc' | 'provider-oauth',
        credential_binding: 'user',
        provider,
      });

      addToast({
        type: 'success',
        title: 'Disconnected',
        message: `${provider || 'OAuth provider'} has been disconnected`,
      });

      onSuccess();
      onClose();
    } catch (err: any) {
      console.error('Failed to disconnect OAuth:', err);
      addToast({
        type: 'error',
        title: 'Failed to disconnect',
        message: err.message || 'An error occurred',
      });
    } finally {
      setIsSubmitting(false);
    }
  };

  const downloadAsFile = () => {
    if (!createdCredential) return;

    const secret = extractCreatedSecret();
    if (!secret) return;

    const content = `# Upstream Credential
# Server: ${server.display_name || server.path}
# Type: ${getAuthTypeLabel(authType as UpstreamAuthType)}
# Created: ${new Date().toISOString()}
#
# IMPORTANT: This is a copy of the secret you provided.
# The gateway has stored this securely for upstream authentication.

${getSecretLabel(authType as UpstreamAuthType)}: ${secret}
`;

    const blob = new Blob([content], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `upstream-credential-${server.path.replace(/\//g, '-')}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  // OAuth Connect view
  if (view === 'oauth-connect') {
    return (
      <Modal
        open={open}
        onClose={onClose}
        title={`Connect ${provider ? provider.charAt(0).toUpperCase() + provider.slice(1) : getAuthTypeLabel(authType as UpstreamAuthType)}`}
        size="md"
      >
        <div className="space-y-4">
          {/* Server info */}
          <div className="flex items-center space-x-3 p-3 bg-gray-50 dark:bg-gray-900 rounded-md">
            <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
              <LinkIcon className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                {server.display_name || server.path}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {getAuthTypeLabel(authType as UpstreamAuthType)}
                {provider && ` via ${provider}`}
              </p>
            </div>
          </div>

          {/* Explanation */}
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-4">
            <div className="flex">
              <LinkIcon className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0" />
              <div className="ml-3">
                <h3 className="text-sm font-medium text-blue-800 dark:text-blue-300">
                  OAuth Connection Required
                </h3>
                <p className="mt-1 text-sm text-blue-700 dark:text-blue-400">
                  This server requires OAuth authentication. Click "Connect" to authorize
                  the gateway to access this service on your behalf. You will be redirected
                  to {provider || 'the provider'} to complete the authorization.
                </p>
              </div>
            </div>
          </div>

          {/* Gateway-terminated info */}
          <div className="bg-gray-50 dark:bg-gray-800 rounded-md p-3">
            <p className="text-sm text-gray-600 dark:text-gray-400">
              Once connected, the gateway will manage tokens automatically. Your clients
              authenticate to the gateway only - they never need {provider || 'provider'} credentials.
            </p>
          </div>
        </div>

        <ModalFooter>
          <Button variant="ghost" onClick={onClose}>
            Cancel
          </Button>
          <Button
            variant="primary"
            onClick={handleOAuthConnect}
            loading={isConnecting}
            leftIcon={<ArrowTopRightOnSquareIcon className="w-4 h-4" />}
          >
            Connect {provider ? provider.charAt(0).toUpperCase() + provider.slice(1) : ''}
          </Button>
        </ModalFooter>
      </Modal>
    );
  }

  // OAuth View (connected) view
  if (view === 'oauth-view' && oauthCredential) {
    return (
      <Modal
        open={open}
        onClose={onClose}
        title={`${provider ? provider.charAt(0).toUpperCase() + provider.slice(1) : 'OAuth'} Connection`}
        size="md"
      >
        <div className="space-y-4">
          {/* Server info */}
          <div className="flex items-center space-x-3 p-3 bg-gray-50 dark:bg-gray-900 rounded-md">
            <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
              <CheckCircleIcon className="w-5 h-5 text-green-600 dark:text-green-400" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                {server.display_name || server.path}
              </p>
              <p className="text-xs text-green-600 dark:text-green-400">
                Connected
              </p>
            </div>
          </div>

          {/* Credential info */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500 dark:text-gray-400">Type</span>
              <Badge variant="info">
                {getAuthTypeLabel(oauthCredential.credential_type)}
              </Badge>
            </div>
            {oauthCredential.provider && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500 dark:text-gray-400">Provider</span>
                <span className="text-sm text-gray-900 dark:text-gray-100 capitalize">
                  {oauthCredential.provider}
                </span>
              </div>
            )}
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500 dark:text-gray-400">Status</span>
              <Badge
                variant={
                  oauthCredential.revoked_at
                    ? 'error'
                    : oauthCredential.expires_at &&
                        new Date(oauthCredential.expires_at) <= new Date()
                      ? 'error'
                      : 'success'
                }
              >
                {oauthCredential.revoked_at
                  ? 'revoked'
                  : oauthCredential.expires_at &&
                      new Date(oauthCredential.expires_at) <= new Date()
                    ? 'expired'
                    : 'configured'}
              </Badge>
            </div>
            {oauthCredential.scopes && oauthCredential.scopes.length > 0 && (
              <div className="flex items-start justify-between">
                <span className="text-sm text-gray-500 dark:text-gray-400">Scopes</span>
                <div className="flex flex-wrap gap-1 max-w-[200px] justify-end">
                  {oauthCredential.scopes.map((scope) => (
                    <Badge key={scope} variant="neutral" size="sm">
                      {scope}
                    </Badge>
                  ))}
                </div>
              </div>
            )}
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500 dark:text-gray-400">Connected</span>
              <span className="text-sm text-gray-900 dark:text-gray-100">
                {new Date(oauthCredential.created_at).toLocaleDateString()}
              </span>
            </div>
            {oauthCredential.expires_at && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500 dark:text-gray-400">Token Expires</span>
                <span className="text-sm text-gray-900 dark:text-gray-100">
                  {new Date(oauthCredential.expires_at).toLocaleString()}
                </span>
              </div>
            )}
            {oauthCredential.last_used_at && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500 dark:text-gray-400">Last Used</span>
                <span className="text-sm text-gray-900 dark:text-gray-100">
                  {new Date(oauthCredential.last_used_at).toLocaleDateString()}
                </span>
              </div>
            )}
          </div>

          {/* Info about token management */}
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-3">
            <p className="text-sm text-blue-700 dark:text-blue-400">
              The gateway manages token refresh automatically. Your clients only authenticate
              to the gateway - they never see {provider || 'provider'} tokens.
            </p>
          </div>
        </div>

        <ModalFooter>
          <Button
            variant="danger"
            onClick={() => setView('oauth-disconnect')}
            leftIcon={<LinkSlashIcon className="w-4 h-4" />}
          >
            Disconnect
          </Button>
          <Button variant="ghost" onClick={onClose}>
            Close
          </Button>
        </ModalFooter>
      </Modal>
    );
  }

  // OAuth Disconnect confirmation view
  if (view === 'oauth-disconnect') {
    return (
      <Modal
        open={open}
        onClose={() => setView('oauth-view')}
        title="Disconnect OAuth"
        size="sm"
      >
        <div className="space-y-4">
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4">
            <div className="flex">
              <ExclamationTriangleIcon className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0" />
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800 dark:text-red-300">
                  Are you sure?
                </h3>
                <p className="mt-1 text-sm text-red-700 dark:text-red-400">
                  Disconnecting will revoke the gateway's access to {provider || 'this provider'}.
                  Requests to this server will fail until you reconnect.
                </p>
              </div>
            </div>
          </div>
        </div>

        <ModalFooter>
          <Button variant="ghost" onClick={() => setView('oauth-view')} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button
            variant="danger"
            onClick={handleOAuthDisconnect}
            loading={isSubmitting}
          >
            Disconnect
          </Button>
        </ModalFooter>
      </Modal>
    );
  }

  // Check for unsupported types (mTLS, header-trust)
  if (!isApiKeyOrJwt && !isOAuth) {
    return (
      <Modal
        open={open}
        onClose={onClose}
        title="Unsupported Credential Type"
        size="md"
      >
        <div className="p-4 text-center">
          <p className="text-gray-600 dark:text-gray-400">
            This server uses {getAuthTypeLabel(authType as UpstreamAuthType)} authentication,
            which is not yet supported in the UI.
          </p>
        </div>
        <ModalFooter>
          <Button variant="secondary" onClick={onClose}>
            Close
          </Button>
        </ModalFooter>
      </Modal>
    );
  }

  // Loading state
  if (view === 'loading') {
    return (
      <Modal
        open={open}
        onClose={handleClose}
        title="Loading..."
        size="md"
      >
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary-600" />
        </div>
      </Modal>
    );
  }

  // Secret display view (after creation)
  if (view === 'secret' && createdCredential) {
    return (
      <Modal
        open={open}
        onClose={handleClose}
        title="Credential Configured"
        size="lg"
      >
        <div className="space-y-4">
          {/* Success message */}
          <div className="flex items-start space-x-3">
            <div className="flex-shrink-0">
              <div className="w-10 h-10 rounded-full bg-green-100 dark:bg-green-900/30 flex items-center justify-center">
                <CheckCircleIcon className="w-5 h-5 text-green-600 dark:text-green-400" />
              </div>
            </div>
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                Upstream credential saved successfully
              </p>
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                Server: {server.display_name || server.path}
              </p>
            </div>
          </div>

          {/* Info notice */}
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-4">
            <div className="flex">
              <KeyIcon className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0" />
              <div className="ml-3">
                <h3 className="text-sm font-medium text-blue-800 dark:text-blue-300">
                  Credential stored securely
                </h3>
                <p className="mt-1 text-sm text-blue-700 dark:text-blue-400">
                  The gateway will use this credential to authenticate with the upstream server.
                  You can download a copy for your records if needed.
                </p>
              </div>
            </div>
          </div>

          {/* Secret display */}
          <div className="space-y-2">
            <label className="block text-sm font-medium text-gray-700 dark:text-gray-300">
              {getSecretLabel(authType as UpstreamAuthType)} (your input)
            </label>
            <div className="relative">
              <code className="block w-full p-3 pr-24 bg-gray-100 dark:bg-gray-800 rounded-md text-sm font-mono text-gray-900 dark:text-gray-100 break-all border border-gray-200 dark:border-gray-700">
                {extractCreatedSecret()}
              </code>
              <div className="absolute right-2 top-1/2 -translate-y-1/2">
                <CopyButton text={extractCreatedSecret()} />
              </div>
            </div>
          </div>

          {/* Action buttons */}
          <div className="flex space-x-2">
            <Button
              variant="secondary"
              size="sm"
              onClick={downloadAsFile}
              leftIcon={<ArrowDownTrayIcon className="w-4 h-4" />}
            >
              Download as .txt
            </Button>
            <Button
              variant="secondary"
              size="sm"
              onClick={() => {
                const secret = extractCreatedSecret();
                if (!secret) {
                  return;
                }

                const headerValue = authType === 'api-key'
                  ? secret
                  : `Bearer ${secret}`;
                const headerName = authType === 'api-key' ? 'X-API-Key' : 'Authorization';
                navigator.clipboard.writeText(`${headerName}: ${headerValue}`);
              }}
              leftIcon={<ClipboardDocumentIcon className="w-4 h-4" />}
            >
              Copy as Header
            </Button>
          </div>

          {/* Acknowledgment checkbox */}
          <div className="border-t border-gray-200 dark:border-gray-700 pt-4">
            <Checkbox
              label="I understand that the gateway will use this credential for upstream authentication"
              checked={acknowledged}
              onChange={(e) => setAcknowledged(e.target.checked)}
            />
          </div>
        </div>

        <ModalFooter>
          <Button
            type="button"
            variant="primary"
            onClick={handleDone}
            disabled={!acknowledged}
          >
            Done
          </Button>
        </ModalFooter>
      </Modal>
    );
  }

  // View existing credential
  if (view === 'view' && existingCredential) {
    return (
      <Modal
        open={open}
        onClose={handleClose}
        title="Upstream Credential"
        size="md"
      >
        <div className="space-y-4">
          {/* Server info */}
          <div className="flex items-center space-x-3 p-3 bg-gray-50 dark:bg-gray-900 rounded-md">
            <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
              <ServerIcon className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
            </div>
            <div>
              <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                {server.display_name || server.path}
              </p>
              <p className="text-xs text-gray-500 dark:text-gray-400 font-mono">
                {server.path}
              </p>
            </div>
          </div>

          {/* Credential info */}
          <div className="space-y-3">
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500 dark:text-gray-400">Type</span>
              <Badge variant="info">
                {getAuthTypeLabel(existingCredential.credential_type)}
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500 dark:text-gray-400">Status</span>
              <Badge
                variant={
                  existingCredential.revoked_at
                    ? 'error'
                    : existingCredential.expires_at &&
                        new Date(existingCredential.expires_at) <= new Date()
                      ? 'error'
                      : 'success'
                }
              >
                {existingCredential.revoked_at
                  ? 'revoked'
                  : existingCredential.expires_at &&
                      new Date(existingCredential.expires_at) <= new Date()
                    ? 'expired'
                    : 'configured'}
              </Badge>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500 dark:text-gray-400">Binding</span>
              <span className="text-sm text-gray-900 dark:text-gray-100">
                {existingCredential.credential_binding}
              </span>
            </div>
            <div className="flex items-center justify-between">
              <span className="text-sm text-gray-500 dark:text-gray-400">Created</span>
              <span className="text-sm text-gray-900 dark:text-gray-100">
                {new Date(existingCredential.created_at).toLocaleDateString()}
              </span>
            </div>
            {existingCredential.expires_at && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500 dark:text-gray-400">Expires</span>
                <span className="text-sm text-gray-900 dark:text-gray-100">
                  {new Date(existingCredential.expires_at).toLocaleDateString()}
                </span>
              </div>
            )}
            {existingCredential.last_used_at && (
              <div className="flex items-center justify-between">
                <span className="text-sm text-gray-500 dark:text-gray-400">Last Used</span>
                <span className="text-sm text-gray-900 dark:text-gray-100">
                  {new Date(existingCredential.last_used_at).toLocaleDateString()}
                </span>
              </div>
            )}
          </div>

          {/* Warning about replacing */}
          <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md p-3">
            <p className="text-sm text-yellow-700 dark:text-yellow-400">
              The secret value cannot be retrieved. To change it, configure a new credential.
            </p>
          </div>
        </div>

        <ModalFooter>
          <Button
            variant="danger"
            onClick={() => setView('confirm-revoke')}
            leftIcon={<TrashIcon className="w-4 h-4" />}
          >
            Revoke
          </Button>
          <Button variant="secondary" onClick={handleConfigureNew}>
            Configure New
          </Button>
          <Button variant="ghost" onClick={handleClose}>
            Close
          </Button>
        </ModalFooter>
      </Modal>
    );
  }

  // Confirm revoke dialog
  if (view === 'confirm-revoke' && existingCredential) {
    return (
      <Modal
        open={open}
        onClose={() => setView('view')}
        title="Revoke Credential"
        size="sm"
      >
        <div className="space-y-4">
          <div className="bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md p-4">
            <div className="flex">
              <ExclamationTriangleIcon className="w-5 h-5 text-red-600 dark:text-red-400 flex-shrink-0" />
              <div className="ml-3">
                <h3 className="text-sm font-medium text-red-800 dark:text-red-300">
                  Are you sure?
                </h3>
                <p className="mt-1 text-sm text-red-700 dark:text-red-400">
                  Revoking this credential will prevent the gateway from authenticating
                  with the upstream server. Requests to this server will fail until a
                  new credential is configured.
                </p>
              </div>
            </div>
          </div>
        </div>

        <ModalFooter>
          <Button variant="ghost" onClick={() => setView('view')} disabled={isSubmitting}>
            Cancel
          </Button>
          <Button
            variant="danger"
            onClick={handleRevoke}
            loading={isSubmitting}
          >
            Revoke Credential
          </Button>
        </ModalFooter>
      </Modal>
    );
  }

  // Form to create/configure credential
  return (
    <Modal
      open={open}
      onClose={handleClose}
      title={`Configure ${getAuthTypeLabel(authType as UpstreamAuthType)}`}
      size="md"
    >
      <form onSubmit={handleSubmit} className="space-y-4">
        {/* Server info */}
        <div className="flex items-center space-x-3 p-3 bg-gray-50 dark:bg-gray-900 rounded-md">
          <div className="flex-shrink-0 w-10 h-10 rounded-lg bg-indigo-100 dark:bg-indigo-900/30 flex items-center justify-center">
            <KeyIcon className="w-5 h-5 text-indigo-600 dark:text-indigo-400" />
          </div>
          <div>
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
              {server.display_name || server.path}
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Upstream authentication: {getAuthTypeLabel(authType as UpstreamAuthType)}
            </p>
          </div>
        </div>

        {/* Info about gateway-terminated auth */}
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-3">
          <p className="text-sm text-blue-700 dark:text-blue-400">
            The gateway will inject this credential when proxying requests to the upstream server.
            Your clients authenticate to the gateway only - they never need this credential.
          </p>
        </div>

        {/* Secret input */}
        <div>
          <label
            htmlFor="secret"
            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
          >
            {getSecretLabel(authType as UpstreamAuthType)} <span className="text-red-500">*</span>
          </label>
          {authType === 'jwt' ? (
            <Textarea
              id="secret"
              value={secretValue}
              onChange={(e) => setSecretValue(e.target.value)}
              placeholder={getSecretPlaceholder(authType as UpstreamAuthType)}
              rows={4}
              disabled={isSubmitting}
              className="font-mono text-sm"
            />
          ) : (
            <Input
              id="secret"
              type="password"
              value={secretValue}
              onChange={(e) => setSecretValue(e.target.value)}
              placeholder={getSecretPlaceholder(authType as UpstreamAuthType)}
              error={errors.secret}
              disabled={isSubmitting}
            />
          )}
          {errors.secret && (
            <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.secret}</p>
          )}
        </div>

        {/* Expiration */}
        <div>
          <label
            htmlFor="expiresAt"
            className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1"
          >
            Expiration Date (optional)
          </label>
          <Input
            id="expiresAt"
            type="datetime-local"
            value={expiresAt}
            onChange={(e) => setExpiresAt(e.target.value)}
            error={errors.expiresAt}
            disabled={isSubmitting}
          />
          {errors.expiresAt && (
            <p className="mt-1 text-sm text-red-600 dark:text-red-400">{errors.expiresAt}</p>
          )}
          <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
            Leave empty if the credential does not expire.
          </p>
        </div>

        {/* Security notice */}
        <div className="bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md p-3">
          <div className="flex">
            <ExclamationTriangleIcon className="w-5 h-5 text-yellow-600 dark:text-yellow-400 flex-shrink-0" />
            <div className="ml-3">
              <p className="text-sm text-yellow-700 dark:text-yellow-400">
                This credential will be stored securely and encrypted. You will not be able to
                view it again after saving.
              </p>
            </div>
          </div>
        </div>

        <ModalFooter>
          <Button
            type="button"
            variant="ghost"
            onClick={handleClose}
            disabled={isSubmitting}
          >
            Cancel
          </Button>
          <Button type="submit" variant="primary" loading={isSubmitting}>
            Save Credential
          </Button>
        </ModalFooter>
      </form>
    </Modal>
  );
}
