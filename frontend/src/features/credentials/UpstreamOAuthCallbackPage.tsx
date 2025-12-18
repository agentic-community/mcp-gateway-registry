/**
 * UpstreamOAuthCallbackPage - Handles OAuth callback for upstream credentials
 *
 * This page is distinct from the user auth callback. It handles the return
 * from OAuth providers when connecting upstream credentials for servers.
 */

import { useEffect, useState, useRef } from 'react';
import { useNavigate, useSearchParams, Link } from 'react-router-dom';
import { Spinner, Button } from '@/components/ui';
import {
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowLeftIcon,
} from '@heroicons/react/24/outline';

type CallbackStatus = 'loading' | 'success' | 'error';

interface StateData {
  server_path?: string;
  return_url?: string;
}

export default function UpstreamOAuthCallbackPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const [status, setStatus] = useState<CallbackStatus>('loading');
  const [errorMessage, setErrorMessage] = useState<string | null>(null);
  const [serverPath, setServerPath] = useState<string | null>(null);
  const [provider, setProvider] = useState<string | null>(null);
  const hasProcessed = useRef(false);

  useEffect(() => {
    // Only process once
    if (hasProcessed.current) return;
    hasProcessed.current = true;

    // Check for OAuth error in URL params
    const errorParam = searchParams.get('error');
    const errorDescription = searchParams.get('error_description');

    if (errorParam) {
      setStatus('error');
      setErrorMessage(
        errorDescription || errorParam || 'OAuth authorization failed'
      );
      // Try to extract server path from state for error recovery
      parseState();
      return;
    }

    // Parse the state parameter
    parseState();

    // If we got here without error, the backend has processed the callback
    // and stored the tokens. Show success.
    setStatus('success');
  }, [searchParams]);

  const parseState = () => {
    const stateParam = searchParams.get('state');
    if (stateParam) {
      try {
        const stateData: StateData = JSON.parse(atob(stateParam));
        if (stateData.server_path) {
          setServerPath(stateData.server_path);
        }
      } catch {
        // Invalid state, continue without server path
      }
    }

    // Try to get provider from URL or state
    const providerParam = searchParams.get('provider');
    if (providerParam) {
      setProvider(providerParam);
    }
  };

  // Show timeout error if taking too long
  useEffect(() => {
    const timeout = setTimeout(() => {
      if (status === 'loading') {
        setStatus('error');
        setErrorMessage('Connection timed out. Please try again.');
      }
    }, 10000); // 10 second timeout

    return () => clearTimeout(timeout);
  }, [status]);

  const handleReturnToCredentials = () => {
    navigate('/credentials/upstream', { replace: true });
  };

  const handleReturnToServer = () => {
    if (serverPath) {
      navigate(`/servers/${serverPath}`, { replace: true });
    } else {
      handleReturnToCredentials();
    }
  };

  const handleTryAgain = () => {
    if (serverPath) {
      navigate(`/credentials/upstream?configure=${serverPath}`, { replace: true });
    } else {
      navigate('/credentials/upstream', { replace: true });
    }
  };

  // Error state
  if (status === 'error') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-md w-full space-y-8 text-center">
          <div className="mx-auto h-16 w-16 bg-red-100 dark:bg-red-900/30 rounded-full flex items-center justify-center">
            <ExclamationTriangleIcon className="w-8 h-8 text-red-600 dark:text-red-400" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              Connection Failed
            </h2>
            <p className="mt-2 text-gray-600 dark:text-gray-400">
              {errorMessage || 'Failed to connect to the OAuth provider.'}
            </p>
            {serverPath && (
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-500">
                Server: <span className="font-mono">{serverPath}</span>
              </p>
            )}
          </div>

          <div className="bg-gray-100 dark:bg-gray-800 rounded-lg p-4 text-left">
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-2">
              What you can try:
            </h3>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1">
              <li>1. Check that you authorized the correct permissions</li>
              <li>2. Verify your provider account is in good standing</li>
              <li>3. Try connecting again</li>
            </ul>
          </div>

          <div className="flex flex-col space-y-3 pt-4">
            <Button variant="primary" onClick={handleTryAgain}>
              Try Again
            </Button>
            <Button variant="secondary" onClick={handleReturnToCredentials}>
              <ArrowLeftIcon className="w-4 h-4 mr-2" />
              Return to Upstream Credentials
            </Button>
          </div>
        </div>
      </div>
    );
  }

  // Success state
  if (status === 'success') {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-md w-full space-y-8 text-center">
          <div className="mx-auto h-16 w-16 bg-green-100 dark:bg-green-900/30 rounded-full flex items-center justify-center">
            <CheckCircleIcon className="w-8 h-8 text-green-600 dark:text-green-400" />
          </div>
          <div>
            <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
              Successfully Connected
            </h2>
            <p className="mt-2 text-gray-600 dark:text-gray-400">
              {provider
                ? `Your ${provider} account has been connected.`
                : 'Your OAuth provider has been connected.'}
            </p>
            {serverPath && (
              <p className="mt-1 text-sm text-gray-500 dark:text-gray-500">
                Server: <span className="font-mono">{serverPath}</span>
              </p>
            )}
          </div>

          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
            <p className="text-sm text-blue-700 dark:text-blue-400">
              The gateway will now use this connection to authenticate with the
              upstream server on your behalf. Your clients only need to
              authenticate to the gateway.
            </p>
          </div>

          <div className="flex flex-col space-y-3 pt-4">
            {serverPath ? (
              <Button variant="primary" onClick={handleReturnToServer}>
                View Server Details
              </Button>
            ) : null}
            <Button
              variant={serverPath ? 'secondary' : 'primary'}
              onClick={handleReturnToCredentials}
            >
              <ArrowLeftIcon className="w-4 h-4 mr-2" />
              Return to Upstream Credentials
            </Button>
          </div>
        </div>
      </div>
    );
  }

  // Loading state
  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
      <div className="text-center">
        <Spinner size="lg" />
        <p className="mt-4 text-gray-600 dark:text-gray-400">
          Completing connection...
        </p>
      </div>
    </div>
  );
}
