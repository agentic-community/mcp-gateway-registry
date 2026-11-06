import React, { useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';

const OAuthCallback: React.FC = () => {
  const navigate = useNavigate();
  const [searchParams] = useSearchParams();
  const { user, loading } = useAuth();

  useEffect(() => {
    // Check if there's an error parameter from the auth server
    const error = searchParams.get('error');
    const errorDetails = searchParams.get('details');

    if (error) {
      // Redirect to login with error message
      const errorMessage = errorDetails ? `${error}: ${errorDetails}` : error;
      navigate(`/login?error=${encodeURIComponent(errorMessage)}`, { replace: true });
      return;
    }

    // If no error and not loading, check authentication status
    if (!loading) {
      if (user) {
        // User is authenticated, redirect to dashboard
        navigate('/', { replace: true });
      } else {
        // User is not authenticated, redirect to login
        navigate('/login?error=oauth2_session_invalid', { replace: true });
      }
    }
  }, [user, loading, navigate, searchParams]);

  // Show loading spinner while checking authentication
  return (
    <div className="min-h-screen bg-canvas flex flex-col justify-center items-center">
      <div className="animate-spin rounded-full h-12 w-12 border-2 border-line border-t-accent mb-4"></div>
      <p className="text-fg-muted">
        Completing authentication...
      </p>
    </div>
  );
};

export default OAuthCallback;
