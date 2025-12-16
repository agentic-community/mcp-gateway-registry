import { lazy, Suspense, type ReactNode } from 'react';
import {
  createBrowserRouter,
  Navigate,
  Outlet,
  useLocation,
} from 'react-router-dom';
import { Spinner } from './components/ui';
import { AppShell } from './components/layout/index';

// Lazy load all pages for code splitting
const OverviewPage = lazy(() => import('./features/overview/OverviewPage'));
const ServersPage = lazy(() => import('./features/servers/ServersPage'));
const ServerDetailsPage = lazy(() => import('./features/servers/ServerDetailsPage'));
const A2AAgentsPage = lazy(() => import('./features/a2a-agents/A2AAgentsPage'));
const A2AAgentDetailsPage = lazy(() => import('./features/a2a-agents/A2AAgentDetailsPage'));
const EnforceAIAgentsPage = lazy(() => import('./features/enforceai-agents/EnforceAIAgentsPage'));
const EnforceAIAgentDetailsPage = lazy(() => import('./features/enforceai-agents/EnforceAIAgentDetailsPage'));
const CredentialsPage = lazy(() => import('./features/credentials/CredentialsPage'));
const ApiKeysPage = lazy(() => import('./features/credentials/ApiKeysPage'));
const TokensPage = lazy(() => import('./features/credentials/TokensPage'));
const ScopesPage = lazy(() => import('./features/scopes/ScopesPage'));
const ToolsPage = lazy(() => import('./features/tools/ToolsPage'));
const AuditPage = lazy(() => import('./features/audit/AuditPage'));
const AdminPage = lazy(() => import('./features/admin/AdminPage'));
const AdminUsersPage = lazy(() => import('./features/admin/AdminUsersPage'));
const AdminUserDetailsPage = lazy(() => import('./features/admin/AdminUserDetailsPage'));
const SettingsPage = lazy(() => import('./features/settings/SettingsPage'));
const HelpPage = lazy(() => import('./features/help/HelpPage'));
const LoginPage = lazy(() => import('./features/auth/LoginPage'));
const OAuthCallbackPage = lazy(() => import('./features/auth/OAuthCallbackPage'));

// Loading spinner wrapper
function PageLoader() {
  return (
    <div className="flex items-center justify-center min-h-[400px]">
      <Spinner size="lg" />
    </div>
  );
}

// Suspense wrapper for lazy components
function SuspenseWrapper({ children }: { children: ReactNode }) {
  return <Suspense fallback={<PageLoader />}>{children}</Suspense>;
}

// Auth context interface (to be provided by AuthProvider)
interface AuthContextValue {
  user: {
    username?: string;
    email?: string;
    is_admin?: boolean;
  } | null;
  loading: boolean;
  logout: () => Promise<void>;
}

// These will be set by the RouterProvider wrapper
let authContext: AuthContextValue | null = null;

export function setAuthContext(ctx: AuthContextValue) {
  authContext = ctx;
}

// Protected route wrapper
function ProtectedRoute({ children }: { children: ReactNode }) {
  const location = useLocation();

  if (!authContext) {
    return <PageLoader />;
  }

  if (authContext.loading) {
    return <PageLoader />;
  }

  if (!authContext.user) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  return <>{children}</>;
}

// Admin route wrapper
function AdminRoute({ children }: { children: ReactNode }) {
  const location = useLocation();

  if (!authContext) {
    return <PageLoader />;
  }

  if (authContext.loading) {
    return <PageLoader />;
  }

  if (!authContext.user) {
    return <Navigate to="/login" state={{ from: location }} replace />;
  }

  if (!authContext.user.is_admin) {
    return <Navigate to="/" replace />;
  }

  return <>{children}</>;
}

// App shell layout wrapper
function AppLayout() {
  const handleLogout = async () => {
    if (authContext) {
      await authContext.logout();
    }
  };

  return (
    <AppShell user={authContext?.user ?? null} onLogout={handleLogout}>
      <Outlet />
    </AppShell>
  );
}

// Public layout (no shell)
function PublicLayout() {
  return <Outlet />;
}

// Router configuration
export const router = createBrowserRouter([
  // Public routes
  {
    element: <PublicLayout />,
    children: [
      {
        path: '/login',
        element: (
          <SuspenseWrapper>
            <LoginPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: '/auth/callback',
        element: (
          <SuspenseWrapper>
            <OAuthCallbackPage />
          </SuspenseWrapper>
        ),
      },
    ],
  },
  // Protected routes with app shell
  {
    element: (
      <ProtectedRoute>
        <AppLayout />
      </ProtectedRoute>
    ),
    children: [
      // Overview
      {
        path: '/',
        element: (
          <SuspenseWrapper>
            <OverviewPage />
          </SuspenseWrapper>
        ),
      },
      // Registry: Servers
      {
        path: '/servers',
        element: (
          <SuspenseWrapper>
            <ServersPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: '/servers/:path',
        element: (
          <SuspenseWrapper>
            <ServerDetailsPage />
          </SuspenseWrapper>
        ),
      },
      // Registry: A2A Agents
      {
        path: '/a2a-agents',
        element: (
          <SuspenseWrapper>
            <A2AAgentsPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: '/a2a-agents/:path',
        element: (
          <SuspenseWrapper>
            <A2AAgentDetailsPage />
          </SuspenseWrapper>
        ),
      },
      // EnforceAI: Agents
      {
        path: '/agents',
        element: (
          <SuspenseWrapper>
            <EnforceAIAgentsPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: '/agents/:agentId',
        element: (
          <SuspenseWrapper>
            <EnforceAIAgentDetailsPage />
          </SuspenseWrapper>
        ),
      },
      // EnforceAI: Credentials
      {
        path: '/credentials',
        element: (
          <SuspenseWrapper>
            <CredentialsPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: '/credentials/api-keys',
        element: (
          <SuspenseWrapper>
            <ApiKeysPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: '/credentials/tokens',
        element: (
          <SuspenseWrapper>
            <TokensPage />
          </SuspenseWrapper>
        ),
      },
      // Scopes and Policy
      {
        path: '/scopes',
        element: (
          <SuspenseWrapper>
            <ScopesPage />
          </SuspenseWrapper>
        ),
      },
      // Tools
      {
        path: '/tools',
        element: (
          <SuspenseWrapper>
            <ToolsPage />
          </SuspenseWrapper>
        ),
      },
      // Audit
      {
        path: '/audit',
        element: (
          <SuspenseWrapper>
            <AuditPage />
          </SuspenseWrapper>
        ),
      },
      // Settings
      {
        path: '/settings',
        element: (
          <SuspenseWrapper>
            <SettingsPage />
          </SuspenseWrapper>
        ),
      },
      // Help
      {
        path: '/help',
        element: (
          <SuspenseWrapper>
            <HelpPage />
          </SuspenseWrapper>
        ),
      },
    ],
  },
  // Admin routes
  {
    element: (
      <AdminRoute>
        <AppLayout />
      </AdminRoute>
    ),
    children: [
      {
        path: '/admin',
        element: (
          <SuspenseWrapper>
            <AdminPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: '/admin/users',
        element: (
          <SuspenseWrapper>
            <AdminUsersPage />
          </SuspenseWrapper>
        ),
      },
      {
        path: '/admin/users/:userId',
        element: (
          <SuspenseWrapper>
            <AdminUserDetailsPage />
          </SuspenseWrapper>
        ),
      },
    ],
  },
  // Catch-all redirect
  {
    path: '*',
    element: <Navigate to="/" replace />,
  },
]);
