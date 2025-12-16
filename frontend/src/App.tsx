import { useEffect } from 'react';
import { RouterProvider } from 'react-router-dom';
import { AuthProvider, useAuth } from './contexts/AuthContext';
import { ThemeProvider } from './contexts/ThemeContext';
import { ToastProvider } from './components/ui/Toast';
import { router, setAuthContext } from './router';

// Component that syncs auth context to router
function AuthSync() {
  const auth = useAuth();

  useEffect(() => {
    setAuthContext(auth);
  }, [auth]);

  return null;
}

// Inner app with auth sync
function AppWithAuth() {
  return (
    <>
      <AuthSync />
      <RouterProvider router={router} />
    </>
  );
}

function App() {
  return (
    <ThemeProvider>
      <ToastProvider>
        <AuthProvider>
          <AppWithAuth />
        </AuthProvider>
      </ToastProvider>
    </ThemeProvider>
  );
}

export default App;
