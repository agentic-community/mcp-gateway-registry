import { PageContent, PageHeader } from '../../components/layout/index';

export default function A2AAgentsPage() {
  return (
    <PageContent>
      <PageHeader
        title="A2A Agents"
        description="Manage registered Agent-to-Agent agents"
        breadcrumbs={[{ name: 'A2A Agents' }]}
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <p className="text-gray-600 dark:text-gray-400">
          A2A agent management coming soon. This page will display registered agents
          with options to register, edit, toggle, and discover new agents.
        </p>
      </div>
    </PageContent>
  );
}
