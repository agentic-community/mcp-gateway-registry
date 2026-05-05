import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import axios from 'axios';
import { getBaseURL } from '../utils/basePath';

// Configure axios to include credentials (cookies) with all requests
axios.defaults.withCredentials = true;

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
  const [csrfToken, setCsrfToken] = useState<string | null>(null);

  useEffect(() => {
    // Set axios baseURL from <base> tag when component mounts
    axios.defaults.baseURL = getBaseURL();

    // Setup axios interceptor to include CSRF token in requests
    const interceptor = axios.interceptors.request.use((config) => {
      if (csrfToken && config.method && ['post', 'put', 'delete', 'patch'].includes(config.method.toLowerCase())) {
        config.headers['X-CSRF-Token'] = csrfToken;
      }
      return config;
    });

    checkAuth();

    // Cleanup interceptor on unmount
    return () => {
      axios.interceptors.request.eject(interceptor);
    };
  }, [csrfToken]);

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
          setCsrfToken(csrfResponse.data.csrf_token);
        }
      } catch (csrfError) {
        console.warn('Failed to fetch CSRF token:', csrfError);
      }
    } catch (error) {
      // User not authenticated
      setUser(null);
      setCsrfToken(null);
    } finally {
      setLoading(false);
    }
  };

  const logout = async () => {
    // Clear user state and CSRF token immediately for responsive UI
    setUser(null);
    setCsrfToken(null);
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