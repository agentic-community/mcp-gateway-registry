import React, { createContext, useContext, useState, useEffect, useRef, ReactNode } from 'react';
import axios from 'axios';
import { getBaseURL } from '../utils/basePath';

// Configure axios to include credentials (cookies) with all requests
axios.defaults.withCredentials = true;

// Configure axios to prepend the registry's ROOT_PATH (e.g. "/registry"
// in path routing mode, "" in subdomain mode) to every request. This
// runs synchronously at module-import time so bare `axios.get('/api/...')`
// calls anywhere in the app resolve correctly — without this, calls
// that fire before AuthProvider's first useEffect run would hit
// origin/api/... and 404 in path mode.
axios.defaults.baseURL = getBaseURL();

// UIPermissions keys match exactly what scopes.yml defines.
// These control server/agent access
interface UIPermissions {
  list_service?: string[];
  register_service?: string[];
  health_check_service?: string[];
  toggle_service?: string[];
  modify_service?: string[];
  list_agents?: string[];
  get_agent?: string[];
  publish_agent?: string[];
  modify_agent?: string[];
  delete_agent?: string[];
  toggle_agent?: string[];
  list_skills?: string[];
  publish_skill?: string[];
  modify_skill?: string[];
  delete_skill?: string[];
  toggle_skill?: string[];
  [key: string]: string[] | undefined;
}

interface User {
  username: string;
  email?: string;
  scopes?: string[];
  groups?: string[];
  auth_method?: string;
  provider?: string;
  can_modify_servers?: boolean;
  is_admin?: boolean;
  ui_permissions?: UIPermissions;
}

interface AuthContextType {
  user: User | null;
  logout: () => Promise<void>;
  loading: boolean;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

export const useAuth = () => {
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
  // Hold CSRF token in a ref so updating it doesn't re-run the mount effect
  // (a re-run would call checkAuth() again, which fetches /api/auth/me and
  // /api/auth/csrf-token, sets the token, and triggers another re-run — a
  // tight loop that surfaces transient 401s as spurious logouts).
  const csrfTokenRef = useRef<string | null>(null);

  useEffect(() => {
    // Setup axios interceptor to include CSRF token in requests
    // (baseURL is already set at module-import time above)
    const interceptor = axios.interceptors.request.use((config) => {
      const token = csrfTokenRef.current;
      if (token && config.method && ['post', 'put', 'delete', 'patch'].includes(config.method.toLowerCase())) {
        config.headers['X-CSRF-Token'] = token;
      }
      return config;
    });

    checkAuth();

    // Cleanup interceptor on unmount
    return () => {
      axios.interceptors.request.eject(interceptor);
    };
  }, []);

  const checkAuth = async () => {
    try {
      const response = await axios.get('/api/auth/me');
      const userData = response.data;
      setUser({
        username: userData.username,
        email: userData.email,
        scopes: userData.scopes || [],
        groups: userData.groups || [],
        auth_method: userData.auth_method || 'oauth2',
        provider: userData.provider,
        can_modify_servers: userData.can_modify_servers || false,
        is_admin: userData.is_admin || false,
        ui_permissions: userData.ui_permissions || {},
      });

      // Fetch CSRF token after successful authentication
      try {
        const csrfResponse = await axios.get('/api/auth/csrf-token');
        if (csrfResponse.data.csrf_token) {
          csrfTokenRef.current = csrfResponse.data.csrf_token;
        }
      } catch (csrfError) {
        console.warn('Failed to fetch CSRF token:', csrfError);
      }
    } catch (error) {
      // Only clear the user on a definite 401. Network errors, 5xx, or
      // transient blips would otherwise flip an authenticated user to
      // logged-out and bounce them to /login.
      const status = axios.isAxiosError(error) ? error.response?.status : undefined;
      if (status === 401) {
        setUser(null);
        csrfTokenRef.current = null;
      } else {
        console.warn('checkAuth failed (non-401), preserving user state:', error);
      }
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    // Clear user state and CSRF token immediately for responsive UI
    setUser(null);
    csrfTokenRef.current = null;
    // Perform full-page redirect to logout endpoint
    // This allows the browser to follow the redirect chain: Registry → Auth-server → IdP → Registry
    // Using window.location.href avoids CORS issues with cross-origin redirects
    window.location.href = `${getBaseURL()}/api/auth/logout`;
  };

  const value = {
    user,
    logout,
    loading,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
