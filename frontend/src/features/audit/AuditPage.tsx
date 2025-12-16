import { PageContent, PageHeader } from '../../components/layout/index';

export default function AuditPage() {
  return (
    <PageContent>
      <PageHeader
        title="Audit"
        description="View audit events and access logs"
        breadcrumbs={[{ name: 'Audit' }]}
      />
      <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-6">
        <p className="text-gray-600 dark:text-gray-400">
          Audit viewer coming soon. This page will provide guidance on finding audit events
          and tools for correlating requests by X-Request-Id.
        </p>
      </div>
    </PageContent>
  );
}
