import { Link } from 'react-router-dom';
import { PageContent, PageHeader } from '../../components/layout/index';
import { Badge } from '../../components/ui';

export default function AdminPage() {
  return (
    <PageContent>
      <PageHeader
        title="Administration"
        description="Administrative tools and user management"
        breadcrumbs={[{ name: 'Admin' }]}
        actions={<Badge variant="warning">Admin Only</Badge>}
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <div className="space-y-4">
          <p className="text-gray-600 dark:text-gray-400">
            Administrative functions for managing users and performing cross-user operations.
          </p>
          <div className="pt-4">
            <Link
              to="/admin/users"
              className="inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-md shadow-sm text-white bg-primary-600 hover:bg-primary-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-primary-500"
            >
              Users Directory
            </Link>
          </div>
        </div>
      </div>
    </PageContent>
  );
}
