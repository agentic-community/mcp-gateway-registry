import { useEffect, useState, useRef } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../../contexts/AuthContext';
import { Spinner, Button } from '../../components/ui';

export default function OAuthCallbackPage() {
  const [searchParams] = useSearchParams();
  const navigate = useNavigate();
  const { user, loading, checkAuth } = useAuth();
  const [error, setError] = useState<string | null>(null);
  const [checking, setChecking] = useState(true);
  const hasChecked = useRef(false);

  useEffect(() => {
    // Only run once
    if (hasChecked.current) return;
    hasChecked.current = true;

    // Check for OAuth error in URL params
    const errorParam = searchParams.get('error');
    const errorDescription = searchParams.get('error_description');

    if (errorParam) {
      setError(errorDescription || errorParam);
      setChecking(false);
      return;
    }

    // After OAuth redirect, the server should have set session cookies
    // We need to check auth status to confirm authentication succeeded
    const verifyAuth = async () => {
      try {
        await checkAuth();
        // If checkAuth succeeds, the user state will be updated
        // and the useEffect below will handle the redirect
      } catch {
        setError('Failed to verify authentication. Please try again.');
      } finally {
        setChecking(false);
      }
    };

    verifyAuth();
  }, [searchParams, checkAuth]);

  // Redirect when user is authenticated
  useEffect(() => {
    if (!checking && !loading && user) {
      // Get redirect URL from state or default to home
      const state = searchParams.get('state');
      let redirectTo = '/';

      if (state) {
        try {
          const stateData = JSON.parse(atob(state));
          if (stateData.redirect) {
            redirectTo = stateData.redirect;
          }
        } catch {
          // Invalid state, use default redirect
        }
      }

      navigate(redirectTo, { replace: true });
    }
  }, [checking, loading, user, navigate, searchParams]);

  // Show timeout error if taking too long
  useEffect(() => {
    const timeout = setTimeout(() => {
      if (checking && !error) {
        setError('Authentication timed out. Please try again.');
        setChecking(false);
      }
    }, 30000); // 30 second timeout

    return () => clearTimeout(timeout);
  }, [checking, error]);

  if (error) {
    return (
      <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900 py-12 px-4 sm:px-6 lg:px-8">
        <div className="max-w-md w-full space-y-8 text-center">
          <div className="mx-auto h-12 w-12 bg-red-600 rounded-lg flex items-center justify-center">
            <svg
              className="w-6 h-6 text-white"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={2}
                d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-3L13.732 4c-.77-1.333-2.694-1.333-3.464 0L3.34 16c-.77 1.333.192 3 1.732 3z"
              />
            </svg>
          </div>
          <h2 className="text-2xl font-bold text-gray-900 dark:text-white">
            Authentication Failed
          </h2>
          <p className="text-gray-600 dark:text-gray-400">{error}</p>
          <div className="pt-4">
            <Button
              variant="primary"
              onClick={() => navigate('/login', { replace: true })}
            >
              Return to login
            </Button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen flex items-center justify-center bg-gray-50 dark:bg-gray-900">
      <div className="text-center">
        <Spinner size="lg" />
        <p className="mt-4 text-gray-600 dark:text-gray-400">
          Completing authentication...
        </p>
      </div>
    </div>
  );
}
