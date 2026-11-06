import React, { useState } from 'react';
import { KeyIcon, ClipboardIcon, CheckIcon } from '@heroicons/react/24/outline';
import axios from 'axios';
import Button from './Button';

export type ResourceType = 'server' | 'virtual_server' | 'agent' | 'skill';

interface ResourceBoundTokenButtonProps {
  /** The resource type the token should be bound to. */
  resourceType: ResourceType;
  /** The resource id (path/slug) the token should be bound to. */
  resourceId: string;
  /**
   * Human-readable name for the resource; used in the default token
   * description if the caller does not override it.
   */
  resourceName?: string;
  expiresInHours?: number;
}

/**
 * Requests a resource-bound JWT from the registry for the given resource
 * (Issue #944). The resulting token can only be used to access this single
 * resource; any attempt to use it against a different server/agent/skill
 * is rejected at the edge with 403.
 *
 * Displayed inline in resource detail modals. The user-token flow (sidebar
 * "Get JWT Token" and TokenGeneration page) is unchanged.
 */
const ResourceBoundTokenButton: React.FC<ResourceBoundTokenButtonProps> = ({
  resourceType,
  resourceId,
  resourceName,
  expiresInHours = 8,
}) => {
  const [loading, setLoading] = useState(false);
  const [token, setToken] = useState<string>('');
  const [error, setError] = useState<string>('');
  const [copied, setCopied] = useState(false);

  const handleGenerate = async () => {
    setLoading(true);
    setError('');
    setToken('');
    try {
      const response = await axios.post(
        '/api/tokens/generate',
        {
          description: `Resource-bound token for ${resourceType}:${resourceName || resourceId}`,
          expires_in_hours: expiresInHours,
          resource: { type: resourceType, id: resourceId },
        },
        { headers: { 'Content-Type': 'application/json' } },
      );
      const accessToken =
        response.data?.tokens?.access_token || response.data?.token_data?.access_token;
      if (!accessToken) {
        throw new Error('No access_token in response');
      }
      setToken(accessToken);
    } catch (err: unknown) {
      if (axios.isAxiosError(err)) {
        setError(err.response?.data?.detail || err.message || 'Failed to generate token');
      } else if (err instanceof Error) {
        setError(err.message);
      } else {
        setError('Failed to generate token');
      }
    } finally {
      setLoading(false);
    }
  };

  const handleCopy = async () => {
    try {
      await navigator.clipboard.writeText(token);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard failure is non-fatal; user can still select and copy.
    }
  };

  return (
    <div className="bg-purple-50 dark:bg-purple-900/20 border border-purple-200 dark:border-purple-800 rounded-lg p-4">
      <div className="flex items-start justify-between gap-3">
        <div>
          <h4 className="font-medium text-purple-900 dark:text-purple-100 mb-1">
            Resource-bound access token
          </h4>
          <p className="text-xs text-purple-800 dark:text-purple-200">
            Generates a JWT scoped to this {resourceType.replace('_', ' ')} only. The token cannot
            be used to reach any other resource.
          </p>
        </div>
        <Button
          variant="primary"
          size="sm"
          onClick={handleGenerate}
          disabled={loading}
          className="whitespace-nowrap"
        >
          {loading ? (
            <>
              <div className="animate-spin rounded-full h-4 w-4 border-b-2 border-white" />
              <span>Generating...</span>
            </>
          ) : (
            <>
              <KeyIcon className="h-4 w-4" />
              <span>Get bound token</span>
            </>
          )}
        </Button>
      </div>

      {error && (
        <p className="mt-3 text-xs text-red-700 dark:text-red-300 break-words">{error}</p>
      )}

      {token && (
        <div className="mt-3">
          <div className="relative">
            <pre className="p-3 pr-12 bg-surface-muted border border-purple-200 dark:border-purple-700 rounded text-xs font-mono break-all whitespace-pre-wrap text-fg max-h-32 overflow-y-auto">
              {token}
            </pre>
            <button
              onClick={handleCopy}
              title={copied ? 'Copied!' : 'Copy token'}
              className="btn-icon absolute top-1 right-1"
            >
              {copied ? (
                <CheckIcon className="h-4 w-4 text-green-600 dark:text-green-400" />
              ) : (
                <ClipboardIcon className="h-4 w-4" />
              )}
            </button>
          </div>
          <p className="mt-2 text-xs text-purple-700 dark:text-purple-300">
            This token expires in {expiresInHours} hours. Save it securely — it will not be shown
            again.
          </p>
        </div>
      )}
    </div>
  );
};

export default ResourceBoundTokenButton;
