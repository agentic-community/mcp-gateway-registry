import React, { Suspense, lazy } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import { AuthProvider } from './contexts/AuthContext';
import { BrandingProvider } from './contexts/BrandingContext';
import { ThemeProvider } from './contexts/ThemeContext';
import Layout from './components/Layout';
import ProtectedRoute from './components/ProtectedRoute';
import PageFallback from './components/PageFallback';
import RouteErrorBoundary from './components/RouteErrorBoundary';
import { getBasename } from './utils/basePath';

// Route-level code splitting: each page (and heavy dependencies such as the
// audit grid under Settings) downloads only when its route is first visited.
const Dashboard = lazy(() => import('./pages/Dashboard'));
const TokenGeneration = lazy(() => import('./pages/TokenGeneration'));
const RegisterPage = lazy(() => import('./pages/RegisterPage'));
const Login = lazy(() => import('./pages/Login'));
const Logout = lazy(() => import('./pages/Logout'));
const OAuthCallback = lazy(() => import('./pages/OAuthCallback'));
const SettingsPage = lazy(() => import('./pages/SettingsPage'));
const ConnectedAccountsPage = lazy(() => import('./pages/ConnectedAccountsPage'));

function App() {
  return (
    <BrandingProvider>
    <ThemeProvider>
      <AuthProvider>
        <Router basename={getBasename()}>
          <RouteErrorBoundary>
          <Suspense fallback={<PageFallback fullScreen />}>
            <Routes>
              <Route path="/login" element={<Login />} />
              <Route path="/logout" element={<Logout />} />
              <Route path="/auth/callback" element={<OAuthCallback />} />
              <Route path="/" element={
                <ProtectedRoute>
                  <Layout>
                    <Dashboard />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/generate-token" element={
                <ProtectedRoute>
                  <Layout>
                    <TokenGeneration />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/servers/register" element={
                <ProtectedRoute>
                  <Layout>
                    <RegisterPage />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/settings/*" element={
                <ProtectedRoute>
                  <Layout>
                    <SettingsPage />
                  </Layout>
                </ProtectedRoute>
              } />
              <Route path="/connected-accounts" element={
                <ProtectedRoute>
                  <Layout>
                    <ConnectedAccountsPage />
                  </Layout>
                </ProtectedRoute>
              } />
            </Routes>
          </Suspense>
          </RouteErrorBoundary>
        </Router>
      </AuthProvider>
    </ThemeProvider>
    </BrandingProvider>
  );
}

export default App;
