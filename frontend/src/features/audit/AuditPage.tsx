/**
 * AuditPage - Audit explorer and guidance
 * Provides an explorer for viewing audit events and documentation on audit logs
 */

import { useState } from 'react';
import { Tab, TabGroup, TabList, TabPanel, TabPanels } from '@headlessui/react';
import { PageContent } from '@/components/layout/PageContent';
import { PageHeader } from '@/components/layout/PageHeader';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { CopyButton } from '@/components/ui/CopyButton';
import { AuditExplorer } from './AuditExplorer';
import { cn } from '@/lib/cn';
import {
  TableCellsIcon,
  DocumentTextIcon,
  MagnifyingGlassIcon,
  ClipboardDocumentIcon,
} from '@heroicons/react/24/outline';

/**
 * Common audit action types with descriptions
 */
const AUDIT_ACTIONS = [
  {
    category: 'MCP Operations',
    actions: [
      { name: 'tools/list', description: 'List available tools from a server', severity: 'info' },
      { name: 'tools/call', description: 'Execute a tool', severity: 'warning' },
    ],
  },
  {
    category: 'Agent Management',
    actions: [
      { name: 'management/agents/create', description: 'Create a new EnforceAI agent', severity: 'warning' },
      { name: 'management/agents/update', description: 'Update agent configuration', severity: 'warning' },
      { name: 'management/agents/revoke', description: 'Revoke an agent', severity: 'error' },
      { name: 'management/agents/list', description: 'List agents', severity: 'info' },
      { name: 'management/agents/get', description: 'Get agent details', severity: 'info' },
    ],
  },
  {
    category: 'API Key Management',
    actions: [
      { name: 'management/api-keys/create', description: 'Create a new API key', severity: 'warning' },
      { name: 'management/api-keys/revoke', description: 'Revoke an API key', severity: 'error' },
      { name: 'management/api-keys/list', description: 'List API keys', severity: 'info' },
    ],
  },
  {
    category: 'Token Management',
    actions: [
      { name: 'management/tokens/mint', description: 'Mint a new gateway token', severity: 'warning' },
      { name: 'management/tokens/revoke', description: 'Revoke a gateway token', severity: 'error' },
      { name: 'management/tokens/revoke-all', description: 'Revoke all tokens for an agent', severity: 'error' },
    ],
  },
];

const getSeverityVariant = (severity: string) => {
  switch (severity) {
    case 'error':
      return 'error';
    case 'warning':
      return 'warning';
    case 'info':
    default:
      return 'neutral';
  }
};

/**
 * Tab button styling helper
 */
function TabButton({
  children,
  icon: Icon,
}: {
  children: React.ReactNode;
  icon: React.ComponentType<{ className?: string }>;
}) {
  return (
    <Tab
      className={({ selected }) =>
        cn(
          'flex items-center gap-2 px-4 py-2 text-sm font-medium rounded-lg outline-none transition-colors',
          selected
            ? 'bg-primary-100 dark:bg-primary-900/50 text-primary-700 dark:text-primary-300'
            : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800'
        )
      }
    >
      <Icon className="h-5 w-5" />
      {children}
    </Tab>
  );
}

/**
 * AuditPage component
 */
export default function AuditPage() {
  const [sampleRequestId] = useState(
    () => `req-${Date.now()}-${Math.random().toString(36).substr(2, 9)}`
  );

  return (
    <>
      <PageHeader
        title="Audit"
        description="Explore audit events and access enforcement logs"
      />

      <PageContent>
        <TabGroup>
          <TabList className="flex gap-2 mb-6">
            <TabButton icon={TableCellsIcon}>Explorer</TabButton>
            <TabButton icon={DocumentTextIcon}>Guidance</TabButton>
          </TabList>

          <TabPanels>
            {/* Explorer Tab */}
            <TabPanel>
              <AuditExplorer />
            </TabPanel>

            {/* Guidance Tab */}
            <TabPanel>
              <div className="space-y-6">
                {/* How to access audit logs */}
          <Card>
            <div className="p-6">
              <div className="flex items-center gap-2 mb-4">
                <DocumentTextIcon className="h-6 w-6 text-gray-700 dark:text-gray-300" />
                <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
                  How to Access Audit Events
                </h3>
              </div>

              <div className="space-y-4">
                <div>
                  <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-2">
                    Docker Compose Logs
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    Audit events are written to stdout and can be viewed using Docker Compose:
                  </p>
                  <div className="bg-gray-100 dark:bg-gray-900 rounded p-3 font-mono text-xs">
                    <code>docker-compose logs -f auth_server</code>
                  </div>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-2">
                    SQLite Database
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    If SQLite audit storage is configured, audit events are persisted to:
                  </p>
                  <div className="bg-gray-100 dark:bg-gray-900 rounded p-3 font-mono text-xs">
                    <code>$ENFORCEAI_STATE_DIR/enforceai_audit.db</code>
                  </div>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mt-2">
                    Query the database using sqlite3 CLI or other SQLite tools.
                  </p>
                </div>

                <div>
                  <h4 className="text-sm font-semibold text-gray-900 dark:text-gray-100 mb-2">
                    Filtering by Request ID
                  </h4>
                  <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                    Each request includes an <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">X-Request-Id</code> header.
                    Filter logs to correlate events:
                  </p>
                  <div className="bg-gray-100 dark:bg-gray-900 rounded p-3 font-mono text-xs">
                    <code>docker-compose logs auth_server | grep "{sampleRequestId}"</code>
                  </div>
                </div>
              </div>
            </div>
          </Card>

          {/* Request ID display */}
          <Card>
            <div className="p-6">
              <div className="flex items-center gap-2 mb-4">
                <ClipboardDocumentIcon className="h-6 w-6 text-gray-700 dark:text-gray-300" />
                <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
                  Current Session Request ID
                </h3>
              </div>

              <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                Use this request ID to correlate audit events from your current session:
              </p>

              <div className="flex items-center gap-3 bg-gray-50 dark:bg-gray-900 rounded p-3">
                <code className="flex-1 font-mono text-sm text-gray-900 dark:text-gray-100">
                  {sampleRequestId}
                </code>
                <CopyButton text={sampleRequestId} />
              </div>

              <p className="text-xs text-gray-500 dark:text-gray-400 mt-2">
                Note: In a real implementation, this would show the actual X-Request-Id from API calls.
              </p>
            </div>
          </Card>

          {/* Audit actions glossary */}
          <Card>
            <div className="p-6">
              <div className="flex items-center gap-2 mb-4">
                <MagnifyingGlassIcon className="h-6 w-6 text-gray-700 dark:text-gray-300" />
                <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
                  Common Audit Actions
                </h3>
              </div>

              <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                Audit events are categorized by action type. Use these to filter and understand
                enforcement logs:
              </p>

              <div className="space-y-6">
                {AUDIT_ACTIONS.map((category) => (
                  <div key={category.category}>
                    <h4 className="text-sm font-semibold text-gray-700 dark:text-gray-300 mb-3">
                      {category.category}
                    </h4>
                    <div className="space-y-2">
                      {category.actions.map((action) => (
                        <div
                          key={action.name}
                          className="flex items-start gap-3 p-3 bg-gray-50 dark:bg-gray-900 rounded"
                        >
                          <Badge
                            variant={getSeverityVariant(action.severity) as any}
                            size="sm"
                            className="mt-0.5"
                          >
                            {action.severity}
                          </Badge>
                          <div className="flex-1 min-w-0">
                            <code className="text-sm font-mono text-gray-900 dark:text-gray-100">
                              {action.name}
                            </code>
                            <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                              {action.description}
                            </p>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>
                ))}
              </div>
            </div>
          </Card>

          {/* Additional resources */}
          <Card>
            <div className="p-6">
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
                Additional Resources
              </h3>
              <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                <li className="flex items-start gap-2">
                  <span className="text-gray-400 mt-1">•</span>
                  <span>
                    <strong>Audit Retention:</strong> See{' '}
                    <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded text-xs">
                      enforceai/instructions/ENFORCEAI_AUDIT_RETENTION.md
                    </code>{' '}
                    for retention policy configuration
                  </span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-gray-400 mt-1">•</span>
                  <span>
                    <strong>Management Guide:</strong> See{' '}
                    <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded text-xs">
                      enforceai/instructions/ENFORCEAI_MANAGEMENT.md
                    </code>{' '}
                    for operational procedures
                  </span>
                </li>
                <li className="flex items-start gap-2">
                  <span className="text-gray-400 mt-1">•</span>
                  <span>
                    <strong>Setup Guide:</strong> See{' '}
                    <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded text-xs">
                      docs/enforceai-setup-guide.md
                    </code>{' '}
                    for initial configuration
                  </span>
                </li>
              </ul>
            </div>
          </Card>
              </div>
            </TabPanel>
          </TabPanels>
        </TabGroup>
      </PageContent>
    </>
  );
}
