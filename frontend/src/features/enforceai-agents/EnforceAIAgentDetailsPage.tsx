import { useParams } from 'react-router-dom';
import { PageContent, PageHeader } from '../../components/layout/index';

export default function EnforceAIAgentDetailsPage() {
  const { agentId } = useParams<{ agentId: string }>();

  return (
    <PageContent>
      <PageHeader
        title={`Agent: ${agentId?.slice(0, 8) || 'Unknown'}...`}
        description="Agent identity details and credentials"
        breadcrumbs={[
          { name: 'EnforceAI Agents', href: '/agents' },
          { name: agentId?.slice(0, 8) || 'Details' },
        ]}
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <p className="text-gray-600 dark:text-gray-400">
          Agent details coming soon. This page will display full agent information,
          scopes, allowed tools, associated credentials, and management actions.
        </p>
      </div>
    </PageContent>
  );
}
