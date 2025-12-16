import { PageContent, PageHeader } from '../../components/layout/index';

export default function OverviewPage() {
  return (
    <PageContent>
      <PageHeader
        title="Overview"
        description="Welcome to the Enforce Gateway management console"
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <p className="text-gray-600 dark:text-gray-400">
          Dashboard overview coming soon. This page will display connection status,
          summary counts, and quick actions.
        </p>
      </div>
    </PageContent>
  );
}
