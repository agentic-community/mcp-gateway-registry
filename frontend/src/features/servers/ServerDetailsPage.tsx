import { useParams } from 'react-router-dom';
import { PageContent, PageHeader } from '../../components/layout/index';

export default function ServerDetailsPage() {
  const { path } = useParams<{ path: string }>();

  return (
    <PageContent>
      <PageHeader
        title={`Server: ${path || 'Unknown'}`}
        description="Server details and configuration"
        breadcrumbs={[
          { name: 'Servers', href: '/servers' },
          { name: path || 'Details' },
        ]}
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <p className="text-gray-600 dark:text-gray-400">
          Server details coming soon. This page will display server information,
          health status, tools list, and configuration options.
        </p>
      </div>
    </PageContent>
  );
}
