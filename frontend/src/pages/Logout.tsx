import React, { useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { CheckCircleIcon } from '@heroicons/react/24/outline';
import { useUiTitle } from '../hooks/useUiTitle';
import Button from '../components/Button';

const Logout: React.FC = () => {
  const navigate = useNavigate();
  const uiTitle = useUiTitle();

  useEffect(() => {
    document.title = uiTitle;
  }, [uiTitle]);

  useEffect(() => {
    // Auto redirect to login after 5 seconds
    const timer = setTimeout(() => {
      navigate('/login');
    }, 5000);

    return () => clearTimeout(timer);
  }, [navigate]);

  return (
    <div className="min-h-screen bg-canvas flex flex-col justify-center py-12 sm:px-6 lg:px-8">
      <div className="sm:mx-auto sm:w-full sm:max-w-md">
        <div className="flex justify-center mb-6">
          <CheckCircleIcon className="h-16 w-16 text-green-500 dark:text-green-400" />
        </div>
        <h2 className="text-center text-3xl font-bold text-gray-900 dark:text-white">
          Successfully Logged Out
        </h2>
        <p className="mt-2 text-center text-sm text-gray-600 dark:text-gray-400">
          You have been logged out from all sessions
        </p>
      </div>

      <div className="mt-8 sm:mx-auto sm:w-full sm:max-w-md">
        <div className="card p-8">
          <div className="text-center space-y-6">
            <p className="text-gray-700 dark:text-gray-300">
              Your session has been terminated and you've been logged out from the identity provider.
            </p>

            <div className="pt-4">
              <Button
                variant="primary"
                fullWidth
                onClick={() => navigate('/login')}
                className="py-3"
              >
                Return to Login
              </Button>
            </div>

            <p className="text-xs text-gray-500 dark:text-gray-400">
              Redirecting to login in 5 seconds...
            </p>
          </div>
        </div>

        <div className="mt-6 text-center">
          <p className="text-xs text-gray-500 dark:text-gray-400">
            {uiTitle} - Secure Access Management
          </p>
        </div>
      </div>
    </div>
  );
};

export default Logout;
