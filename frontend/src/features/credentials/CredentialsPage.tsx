import { PageContent, PageHeader } from '../../components/layout/index';

export default function CredentialsPage() {
  return (
    <PageContent>
      <PageHeader
        title="Credentials"
        description="Manage API keys and gateway tokens"
        breadcrumbs={[{ name: 'Credentials' }]}
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <p className="text-gray-600 dark:text-gray-400">
          Credentials management coming soon. This page will provide access to API keys
          and gateway token management sections.
        </p>
      </div>
    </PageContent>
  );
}
