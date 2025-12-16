import { PageContent, PageHeader } from '../../components/layout/index';

export default function ApiKeysPage() {
  return (
    <PageContent>
      <PageHeader
        title="API Keys"
        description="Create and manage API keys for agent authentication"
        breadcrumbs={[
          { name: 'Credentials', href: '/credentials' },
          { name: 'API Keys' },
        ]}
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <p className="text-gray-600 dark:text-gray-400">
          API key management coming soon. This page will allow creating, viewing,
          and revoking API keys for agent authentication.
        </p>
      </div>
    </PageContent>
  );
}
