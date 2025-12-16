import { PageContent, PageHeader } from '../../components/layout/index';

export default function EnforceAIAgentsPage() {
  return (
    <PageContent>
      <PageHeader
        title="EnforceAI Agents"
        description="Manage agent identities and authorization"
        breadcrumbs={[{ name: 'EnforceAI Agents' }]}
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <p className="text-gray-600 dark:text-gray-400">
          EnforceAI agent management coming soon. This page will display agent identities
          with their scopes, allowed tools, and lifecycle management options.
        </p>
      </div>
    </PageContent>
  );
}
