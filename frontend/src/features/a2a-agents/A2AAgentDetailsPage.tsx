import { useParams } from 'react-router-dom';
import { PageContent, PageHeader } from '../../components/layout/index';

export default function A2AAgentDetailsPage() {
  const { path } = useParams<{ path: string }>();

  return (
    <PageContent>
      <PageHeader
        title={`A2A Agent: ${path || 'Unknown'}`}
        description="Agent details and configuration"
        breadcrumbs={[
          { name: 'A2A Agents', href: '/a2a-agents' },
          { name: path || 'Details' },
        ]}
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <p className="text-gray-600 dark:text-gray-400">
          A2A agent details coming soon. This page will display agent card information,
          skills, and configuration options.
        </p>
      </div>
    </PageContent>
  );
}
