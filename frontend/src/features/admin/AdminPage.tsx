/**
 * AdminPage - Admin landing page with overview and quick actions
 */

import { Link } from 'react-router-dom';
import { PageHeader } from '@/components/layout/PageHeader';
import { PageContent } from '@/components/layout/PageContent';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { AdminLayout } from './AdminLayout';
import {
  UsersIcon,
  ShieldCheckIcon,
  DocumentTextIcon,
  Cog6ToothIcon,
  LinkIcon,
} from '@heroicons/react/24/outline';

/**
 * Admin landing page component
 */
export default function AdminPage() {
  const adminSections = [
    {
      icon: <UsersIcon className="h-8 w-8" />,
      title: 'User Directory',
      description: 'Search and manage users, view user details and agent assignments',
      link: '/admin/users',
      primary: true,
    },
    {
      icon: <ShieldCheckIcon className="h-8 w-8" />,
      title: 'Access Control',
      description: 'Manage cross-user operations, revoke agents and credentials',
      link: '/admin/users',
      badge: 'Phase 14',
    },
    {
      icon: <DocumentTextIcon className="h-8 w-8" />,
      title: 'Audit & Logs',
      description: 'View admin actions and system audit events',
      link: '/audit',
    },
    {
      icon: <Cog6ToothIcon className="h-8 w-8" />,
      title: 'System Configuration',
      description: 'View operator configuration and system settings',
      link: '/settings',
    },
    {
      icon: <LinkIcon className="h-8 w-8" />,
      title: 'Upstream OAuth Providers',
      description: 'Manage OAuth client configurations used for gateway-terminated upstream OAuth',
      link: '/admin/upstream-oauth-providers',
      badge: 'Phase 4',
    },
  ];

  return (
    <>
      <PageHeader
        title="Admin Dashboard"
        description="Manage users, permissions, and system configuration"
      />

      <PageContent>
        <AdminLayout>
          <div className="space-y-6">
            {/* Quick Stats */}
            <div className="grid grid-cols-1 gap-4 sm:grid-cols-2 lg:grid-cols-4">
              <Card>
                <div className="p-6">
                  <div className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Total Users
                  </div>
                  <div className="mt-2 text-3xl font-semibold text-gray-900 dark:text-gray-100">
                    —
                  </div>
                  <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                    User search available
                  </div>
                </div>
              </Card>

              <Card>
                <div className="p-6">
                  <div className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Active Agents
                  </div>
                  <div className="mt-2 text-3xl font-semibold text-gray-900 dark:text-gray-100">
                    —
                  </div>
                  <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                    Across all users
                  </div>
                </div>
              </Card>

              <Card>
                <div className="p-6">
                  <div className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Revoked Agents
                  </div>
                  <div className="mt-2 text-3xl font-semibold text-gray-900 dark:text-gray-100">
                    —
                  </div>
                  <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                    Requires cleanup
                  </div>
                </div>
              </Card>

              <Card>
                <div className="p-6">
                  <div className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Recent Actions
                  </div>
                  <div className="mt-2 text-3xl font-semibold text-gray-900 dark:text-gray-100">
                    —
                  </div>
                  <div className="mt-2 text-xs text-gray-500 dark:text-gray-400">
                    Last 24 hours
                  </div>
                </div>
              </Card>
            </div>

            {/* Admin Sections */}
            <div>
              <h2 className="text-lg font-semibold text-gray-900 dark:text-gray-100 mb-4">
                Admin Tools
              </h2>
              <div className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                {adminSections.map((section) => (
                  <Card key={section.title}>
                    <div className="p-6">
                      <div className="flex items-start gap-4">
                        <div className="flex-shrink-0 text-gray-700 dark:text-gray-300">
                          {section.icon}
                        </div>
                        <div className="flex-1 min-w-0">
                          <div className="flex items-center gap-2 mb-2">
                            <h3 className="text-base font-semibold text-gray-900 dark:text-gray-100">
                              {section.title}
                            </h3>
                            {section.badge && (
                              <span className="inline-flex items-center rounded-full bg-gray-100 dark:bg-gray-700 px-2 py-0.5 text-xs font-medium text-gray-600 dark:text-gray-300">
                                {section.badge}
                              </span>
                            )}
                          </div>
                          <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                            {section.description}
                          </p>
                          <Link to={section.link}>
                            <Button
                              variant={section.primary ? 'primary' : 'secondary'}
                              size="sm"
                            >
                              Open
                            </Button>
                          </Link>
                        </div>
                      </div>
                    </div>
                  </Card>
                ))}
              </div>
            </div>

            {/* Important Notes */}
            <Card>
              <div className="p-6">
                <h3 className="text-base font-semibold text-gray-900 dark:text-gray-100 mb-3">
                  Important Notes
                </h3>
                <ul className="space-y-2 text-sm text-gray-600 dark:text-gray-400">
                  <li className="flex items-start gap-2">
                    <span className="text-gray-400 mt-0.5">•</span>
                    <span>
                      All admin actions are logged and auditable. Review audit logs regularly.
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-gray-400 mt-0.5">•</span>
                    <span>
                      Cross-user operations require confirmation and should include a reason.
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-gray-400 mt-0.5">•</span>
                    <span>
                      Agent revocations are permanent. Use "Revoke All Tokens" for temporary
                      suspension.
                    </span>
                  </li>
                  <li className="flex items-start gap-2">
                    <span className="text-gray-400 mt-0.5">•</span>
                    <span>
                      See{' '}
                      <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded text-xs">
                        enforceai/instructions/ENFORCEAI_MANAGEMENT.md
                      </code>{' '}
                      for operational procedures.
                    </span>
                  </li>
                </ul>
              </div>
            </Card>
          </div>
        </AdminLayout>
      </PageContent>
    </>
  );
}
