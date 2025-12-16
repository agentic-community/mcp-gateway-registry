import { PageContent, PageHeader } from '../../components/layout/index';

export default function ScopesPage() {
  return (
    <PageContent>
      <PageHeader
        title="Scopes and Policy"
        description="View the scope catalog and policy definitions"
        breadcrumbs={[{ name: 'Scopes' }]}
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <p className="text-gray-600 dark:text-gray-400">
          Scope catalog viewer coming soon. This page will display all available scopes
          with their allowed servers, methods, and tool restrictions.
        </p>
      </div>
    </PageContent>
  );
}
