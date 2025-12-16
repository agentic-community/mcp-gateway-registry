import { useCallback, useEffect, useState } from 'react';
import { useAuth } from '../contexts/AuthContext';

interface SessionInfo {
  isActive: boolean;
  expiresAt: Date | null;
  timeRemaining: number | null;
  isExpiringSoon: boolean;
}

const EXPIRY_WARNING_MS = 5 * 60 * 1000; // 5 minutes

export function useSession(): SessionInfo & {
  refresh: () => Promise<void>;
  logout: () => Promise<void>;
} {
  const { user, sessionExpiring, refreshSession, logout } = useAuth();
  const [timeRemaining, setTimeRemaining] = useState<number | null>(null);

  const expiresAt = user?.session_expires_at
    ? new Date(user.session_expires_at)
    : null;

  // Update time remaining periodically
  useEffect(() => {
    if (!expiresAt) {
      setTimeRemaining(null);
      return;
    }

    const updateRemaining = () => {
      const remaining = expiresAt.getTime() - Date.now();
      setTimeRemaining(remaining > 0 ? remaining : 0);
    };

    updateRemaining();
    const interval = setInterval(updateRemaining, 1000);

    return () => clearInterval(interval);
  }, [expiresAt]);

  const refresh = useCallback(async () => {
    await refreshSession();
  }, [refreshSession]);

  return {
    isActive: !!user,
    expiresAt,
    timeRemaining,
    isExpiringSoon:
      sessionExpiring ||
      (timeRemaining !== null && timeRemaining < EXPIRY_WARNING_MS),
    refresh,
    logout,
  };
}

export function useRequireAuth(redirectTo: string = '/login'): boolean {
  const { user, loading } = useAuth();
  const [shouldRedirect, setShouldRedirect] = useState(false);

  useEffect(() => {
    if (!loading && !user) {
      setShouldRedirect(true);
      // Use window.location for a full redirect that preserves the return URL
      const returnUrl = encodeURIComponent(window.location.pathname + window.location.search);
      window.location.href = `${redirectTo}?returnUrl=${returnUrl}`;
    }
  }, [user, loading, redirectTo]);

  return !loading && !!user && !shouldRedirect;
}

export function useRequireAdmin(redirectTo: string = '/'): boolean {
  const { user, loading } = useAuth();
  const [shouldRedirect, setShouldRedirect] = useState(false);

  useEffect(() => {
    if (!loading) {
      if (!user) {
        setShouldRedirect(true);
        window.location.href = '/login';
      } else if (!user.is_admin) {
        setShouldRedirect(true);
        window.location.href = redirectTo;
      }
    }
  }, [user, loading, redirectTo]);

  return !loading && !!user?.is_admin && !shouldRedirect;
}

export default useSession;
