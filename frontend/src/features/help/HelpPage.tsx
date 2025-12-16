import { PageContent, PageHeader } from '../../components/layout/index';

export default function HelpPage() {
  return (
    <PageContent>
      <PageHeader
        title="Help"
        description="Documentation and troubleshooting"
        breadcrumbs={[{ name: 'Help' }]}
      />
      <div className="space-y-6">
        {/* Documentation links */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Documentation
          </h3>
          <ul className="space-y-2 text-gray-600 dark:text-gray-400">
            <li>
              <a href="#" className="text-primary-600 dark:text-primary-400 hover:underline">
                Setup Guide
              </a>
            </li>
            <li>
              <a href="#" className="text-primary-600 dark:text-primary-400 hover:underline">
                Management Instructions
              </a>
            </li>
            <li>
              <a href="#" className="text-primary-600 dark:text-primary-400 hover:underline">
                Audit and Retention
              </a>
            </li>
          </ul>
        </div>

        {/* Troubleshooting */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Troubleshooting
          </h3>
          <div className="space-y-4">
            <div>
              <h4 className="font-medium text-gray-900 dark:text-white">401 - Authentication Failed</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Check credentials and token expiry. Ensure the token has not been revoked.
              </p>
            </div>
            <div>
              <h4 className="font-medium text-gray-900 dark:text-white">403 - Forbidden</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Check ownership, agent binding, and required scopes. Verify the agent has not been revoked.
              </p>
            </div>
            <div>
              <h4 className="font-medium text-gray-900 dark:text-white">503 - Enforcement Unavailable</h4>
              <p className="text-sm text-gray-600 dark:text-gray-400">
                Check database connectivity, keyring configuration, and contact the operator.
              </p>
            </div>
          </div>
        </div>
      </div>
    </PageContent>
  );
}
