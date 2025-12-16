import { PageContent, PageHeader } from '../../components/layout/index';
import { useTheme } from '../../contexts/ThemeContext';

export default function SettingsPage() {
  const { theme, setTheme } = useTheme();

  return (
    <PageContent>
      <PageHeader
        title="Settings"
        description="Configure your preferences"
        breadcrumbs={[{ name: 'Settings' }]}
      />
      <div className="space-y-6">
        {/* Theme settings */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Appearance
          </h3>
          <div className="space-y-4">
            <div>
              <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Theme
              </label>
              <select
                value={theme}
                onChange={(e) => setTheme(e.target.value as 'light' | 'dark' | 'system')}
                className="block w-full max-w-xs rounded-md border-gray-300 dark:border-gray-600 dark:bg-gray-700 dark:text-white shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm"
              >
                <option value="system">System</option>
                <option value="light">Light</option>
                <option value="dark">Dark</option>
              </select>
            </div>
          </div>
        </div>

        {/* Session info placeholder */}
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
          <h3 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
            Session Information
          </h3>
          <p className="text-gray-600 dark:text-gray-400">
            Session details and operator configuration will be displayed here.
          </p>
        </div>
      </div>
    </PageContent>
  );
}
