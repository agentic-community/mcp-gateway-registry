import { PageContent, PageHeader } from '../../components/layout/index';

export default function ServersPage() {
  return (
    <PageContent>
      <PageHeader
        title="MCP Servers"
        description="Manage registered MCP servers"
        breadcrumbs={[{ name: 'Servers' }]}
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <p className="text-gray-600 dark:text-gray-400">
          Server management coming soon. This page will display the list of
          registered MCP servers with options to register, edit, toggle, and refresh.
        </p>
      </div>
    </PageContent>
  );
}
