import { useParams } from 'react-router-dom';
import { PageContent, PageHeader } from '../../components/layout/index';
import { Badge } from '../../components/ui';

export default function AdminUserDetailsPage() {
  const { userId } = useParams<{ userId: string }>();

  return (
    <PageContent>
      <PageHeader
        title={`User: ${userId?.slice(0, 20) || 'Unknown'}...`}
        description="User details and cross-user operations"
        breadcrumbs={[
          { name: 'Admin', href: '/admin' },
          { name: 'Users', href: '/admin/users' },
          { name: userId?.slice(0, 20) || 'Details' },
        ]}
        actions={<Badge variant="warning">Admin Only</Badge>}
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <p className="text-gray-600 dark:text-gray-400">
          User details coming soon. This page will display user information and allow
          managing their agents and credentials on their behalf.
        </p>
      </div>
    </PageContent>
  );
}
