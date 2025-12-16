import { PageContent, PageHeader } from '../../components/layout/index';

export default function ToolsPage() {
  return (
    <PageContent>
      <PageHeader
        title="Tools Discovery"
        description="Explore available tools across MCP servers"
        breadcrumbs={[{ name: 'Tools' }]}
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <p className="text-gray-600 dark:text-gray-400">
          Tools discovery coming soon. This page will list all tools from registered servers
          and help configure allowed_tools for agents.
        </p>
      </div>
    </PageContent>
  );
}
