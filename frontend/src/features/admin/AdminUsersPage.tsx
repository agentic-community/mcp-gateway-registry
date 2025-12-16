import { PageContent, PageHeader } from '../../components/layout/index';
import { Badge } from '../../components/ui';

export default function AdminUsersPage() {
  return (
    <PageContent>
      <PageHeader
        title="Users Directory"
        description="Search and manage user accounts"
        breadcrumbs={[
          { name: 'Admin', href: '/admin' },
          { name: 'Users' },
        ]}
        actions={<Badge variant="warning">Admin Only</Badge>}
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <p className="text-gray-600 dark:text-gray-400">
          User directory coming soon. This page will allow searching users by email or username
          and viewing their agents and credentials.
        </p>
      </div>
    </PageContent>
  );
}
