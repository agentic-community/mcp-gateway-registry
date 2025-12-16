import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
  ReactNode,
} from 'react';
import { apiClient, clearCsrfToken } from '../api/client';
import type { AuthProviders } from '../api/types';

// Session configuration
const SESSION_CHECK_INTERVAL = 5 * 60 * 1000; // Check session every 5 minutes
const SESSION_WARNING_THRESHOLD = 5 * 60 * 1000; // Warn 5 minutes before expiry

export interface User {
  user_id?: string;
  username: string;
  email?: string;
  scopes?: string[];
  groups?: string[];
  roles?: string[];
  auth_method?: 'oidc' | 'password' | 'basic';
  provider?: string;
  can_modify_servers?: boolean;
  is_admin?: boolean;
  session_expires_at?: string;
}

export interface OAuthProvider {
  name: string;
  display_name: string;
}

export interface AuthContextType {
  user: User | null;
  loading: boolean;
  sessionExpiring: boolean;
  oauthProviders: OAuthProvider[];
  oauthProvidersLoading: boolean;
  login: (username: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  loginWithOAuth: (providerName: string) => void;
  refreshSession: () => Promise<void>;
  checkAuth: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<User | null>(null);
  const [loading, setLoading] = useState(true);
  const [sessionExpiring, setSessionExpiring] = useState(false);
  const [oauthProviders, setOauthProviders] = useState<OAuthProvider[]>([]);
  const [oauthProvidersLoading, setOauthProvidersLoading] = useState(true);

  const sessionCheckInterval = useRef<NodeJS.Timeout | null>(null);
  const sessionWarningTimeout = useRef<NodeJS.Timeout | null>(null);

  // Check if user is authenticated
  const checkAuth = useCallback(async () => {
    try {
      const response = await apiClient.get<{
        user_id?: string;
        username: string;
        email?: string;
        scopes?: string[];
        groups?: string[];
        roles?: string[];
        auth_method?: string;
        provider?: string;
        can_modify_servers?: boolean;
        is_admin?: boolean;
        session_expires_at?: string;
      }>('/api/auth/me');
      const userData = response.data;

      const newUser: User = {
        user_id: userData.user_id,
        username: userData.username,
        email: userData.email,
        scopes: userData.scopes || [],
        groups: userData.groups || [],
        roles: userData.roles || [],
        auth_method: (userData.auth_method as User['auth_method']) || 'basic',
        provider: userData.provider,
        can_modify_servers: userData.can_modify_servers || false,
        is_admin: userData.is_admin || false,
        session_expires_at: userData.session_expires_at,
      };

      setUser(newUser);
      setSessionExpiring(false);

      // Set up session expiry warning if we have expiry time
      if (userData.session_expires_at) {
        setupSessionExpiryWarning(userData.session_expires_at);
      }
    } catch (error) {
      // User not authenticated
      console.log('[AuthContext] Auth check failed (expected for non-authenticated users)');
      setUser(null);
      clearSessionTimers();
    } finally {
      console.log('[AuthContext] Setting loading to false');
      setLoading(false);
    }
  }, []);

  // Set up session expiry warning timer
  const setupSessionExpiryWarning = useCallback((expiresAt: string) => {
    // Clear existing timer
    if (sessionWarningTimeout.current) {
      clearTimeout(sessionWarningTimeout.current);
    }

    const expiryTime = new Date(expiresAt).getTime();
    const now = Date.now();
    const timeUntilExpiry = expiryTime - now;
    const timeUntilWarning = timeUntilExpiry - SESSION_WARNING_THRESHOLD;

    if (timeUntilWarning > 0) {
      sessionWarningTimeout.current = setTimeout(() => {
        setSessionExpiring(true);
      }, timeUntilWarning);
    } else if (timeUntilExpiry > 0) {
      // Already within warning threshold
      setSessionExpiring(true);
    }
  }, []);

  // Clear session timers
  const clearSessionTimers = useCallback(() => {
    if (sessionCheckInterval.current) {
      clearInterval(sessionCheckInterval.current);
      sessionCheckInterval.current = null;
    }
    if (sessionWarningTimeout.current) {
      clearTimeout(sessionWarningTimeout.current);
      sessionWarningTimeout.current = null;
    }
    setSessionExpiring(false);
  }, []);

  // Fetch available OAuth providers
  const fetchOAuthProviders = useCallback(async () => {
    try {
      const response = await apiClient.get<AuthProviders>('/api/auth/providers');
      setOauthProviders(
        response.data.providers.map((p) => ({
          name: p.name,
          display_name: p.display_name,
        }))
      );
    } catch {
      // OAuth providers not available
      setOauthProviders([]);
    } finally {
      setOauthProvidersLoading(false);
    }
  }, []);

  // Initialize auth state
  useEffect(() => {
    checkAuth();
    fetchOAuthProviders();

    // Set up periodic session check
    sessionCheckInterval.current = setInterval(() => {
      checkAuth();
    }, SESSION_CHECK_INTERVAL);

    return () => {
      clearSessionTimers();
    };
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []); // Only run once on mount

  // Password login
  const login = async (username: string, password: string): Promise<void> => {
    const formData = new URLSearchParams();
    formData.append('username', username);
    formData.append('password', password);

    const response = await apiClient.post('/api/auth/login', formData, {
      headers: {
        'Content-Type': 'application/x-www-form-urlencoded',
      },
    });

    if (response.status === 200) {
      await checkAuth();
    }
  };

  // OAuth login - redirect to provider
  const loginWithOAuth = (providerName: string): void => {
    // Get current URL for redirect after auth
    const returnUrl = window.location.origin + '/auth/callback';
    // Backend route is /api/auth/auth/{provider}
    const authUrl = `/api/auth/auth/${providerName}?redirect_uri=${encodeURIComponent(returnUrl)}`;
    window.location.href = authUrl;
  };

  // Logout
  const logout = async (): Promise<void> => {
    try {
      await apiClient.post('/api/auth/logout');
    } catch {
      // Ignore errors during logout
    } finally {
      setUser(null);
      clearCsrfToken();
      clearSessionTimers();
    }
  };

  // Refresh session
  const refreshSession = async (): Promise<void> => {
    try {
      await apiClient.post('/api/auth/refresh');
      await checkAuth();
    } catch {
      // Session refresh failed - user needs to re-login
      setUser(null);
      clearSessionTimers();
    }
  };

  const value: AuthContextType = {
    user,
    loading,
    sessionExpiring,
    oauthProviders,
    oauthProvidersLoading,
    login,
    logout,
    loginWithOAuth,
    refreshSession,
    checkAuth,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}; 