import { useAuth } from '../../contexts/AuthContext';
import { Button } from '../ui';

interface SessionExpiryWarningProps {
  className?: string;
}

export function SessionExpiryWarning({ className = '' }: SessionExpiryWarningProps) {
  const { sessionExpiring, refreshSession, logout } = useAuth();

  if (!sessionExpiring) {
    return null;
  }

  const handleRefresh = async () => {
    try {
      await refreshSession();
    } catch {
      // Refresh failed, user will be logged out
    }
  };

  return (
    <div
      className={`fixed bottom-4 right-4 max-w-sm bg-yellow-50 dark:bg-yellow-900/50 border border-yellow-200 dark:border-yellow-800 rounded-lg shadow-lg p-4 z-50 ${className}`}
      role="alert"
      aria-live="polite"
    >
      <div className="flex items-start gap-3">
        <div className="flex-shrink-0">
          <svg
            className="w-5 h-5 text-yellow-600 dark:text-yellow-400"
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
        <div className="flex-1">
          <h3 className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
            Session Expiring Soon
          </h3>
          <p className="mt-1 text-sm text-yellow-700 dark:text-yellow-300">
            Your session will expire in a few minutes. Would you like to stay signed in?
          </p>
          <div className="mt-3 flex gap-2">
            <Button
              size="sm"
              variant="primary"
              onClick={handleRefresh}
            >
              Stay Signed In
            </Button>
            <Button
              size="sm"
              variant="ghost"
              onClick={logout}
            >
              Sign Out
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}

export default SessionExpiryWarning;
