import { PageContent, PageHeader } from '../../components/layout/index';

export default function TokensPage() {
  return (
    <PageContent>
      <PageHeader
        title="Gateway Tokens"
        description="Mint and manage gateway tokens"
        breadcrumbs={[
          { name: 'Credentials', href: '/credentials' },
          { name: 'Gateway Tokens' },
        ]}
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <p className="text-gray-600 dark:text-gray-400">
          Gateway token management coming soon. This page will allow minting new tokens,
          viewing active tokens, and revoking tokens by JTI.
        </p>
      </div>
    </PageContent>
  );
}
