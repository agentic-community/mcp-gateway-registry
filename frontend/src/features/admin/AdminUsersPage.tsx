/**
 * AdminUsersPage - User directory with search functionality
 */

import { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { PageHeader } from '@/components/layout/PageHeader';
import { PageContent } from '@/components/layout/PageContent';
import { Card } from '@/components/ui/Card';
import { Input } from '@/components/ui/Input';
import { Badge } from '@/components/ui/Badge';
import { Spinner } from '@/components/ui/Spinner';
import { EmptyState } from '@/components/ui/EmptyState';
import { CopyButton } from '@/components/ui/CopyButton';
import { AdminLayout } from './AdminLayout';
import { useAdminUsers } from './hooks';
import { MagnifyingGlassIcon, UsersIcon } from '@heroicons/react/24/outline';
import { formatDistanceToNow } from 'date-fns';

/**
 * Admin Users Directory page component
 */
export default function AdminUsersPage() {
  const [searchQuery, setSearchQuery] = useState('');
  const [debouncedQuery, setDebouncedQuery] = useState('');
  const navigate = useNavigate();

  // Debounce search query
  useEffect(() => {
    const timeoutId = setTimeout(() => {
      setDebouncedQuery(searchQuery);
    }, 300);
    return () => clearTimeout(timeoutId);
  }, [searchQuery]);

  const { users, isLoading, isError, error } = useAdminUsers(debouncedQuery);

  const handleUserClick = (userId: string) => {
    navigate(`/admin/users/${encodeURIComponent(userId)}`);
  };

  const showResults = debouncedQuery.length > 0;

  return (
    <>
      <PageHeader
        title="User Directory"
        description="Search for users and manage their accounts"
      />

      <PageContent>
        <AdminLayout>
          <div className="space-y-6">
            {/* Search Input */}
            <Card>
              <div className="p-6">
                <label htmlFor="user-search" className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                  Search Users
                </label>
                <Input
                  id="user-search"
                  type="text"
                  placeholder="Search by email or username..."
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  leftAddon={<MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />}
                />
                <p className="mt-2 text-sm text-gray-500 dark:text-gray-400">
                  Enter at least 1 character to start searching. Partial matches supported.
                </p>
              </div>
            </Card>

            {/* Search Results */}
            {showResults && (
              <Card>
                <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
                  <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                    Search Results
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                    {isLoading ? 'Searching...' : `Found ${users.length} user${users.length !== 1 ? 's' : ''}`}
                  </p>
                </div>

                <div className="p-6">
                  {isLoading && (
                    <div className="flex items-center justify-center py-12">
                      <Spinner size="lg" />
                    </div>
                  )}

                  {isError && (
                    <EmptyState
                      icon={<UsersIcon className="h-12 w-12" />}
                      title="Search failed"
                      description={error?.message || 'Failed to search users. Please try again.'}
                    />
                  )}

                  {!isLoading && !isError && users.length === 0 && (
                    <EmptyState
                      icon={<UsersIcon className="h-12 w-12" />}
                      title="No users found"
                      description="Try a different search query."
                    />
                  )}

                  {!isLoading && !isError && users.length > 0 && (
                    <div className="overflow-x-auto">
                      <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                        <thead>
                          <tr>
                            <th
                              scope="col"
                              className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                            >
                              Email
                            </th>
                            <th
                              scope="col"
                              className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                            >
                              Username
                            </th>
                            <th
                              scope="col"
                              className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                            >
                              User ID
                            </th>
                            <th
                              scope="col"
                              className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                            >
                              Auth Method
                            </th>
                            <th
                              scope="col"
                              className="px-3 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
                            >
                              Last Seen
                            </th>
                          </tr>
                        </thead>
                        <tbody className="divide-y divide-gray-200 dark:divide-gray-700">
                          {users.map((user) => (
                            <tr
                              key={user.user_id}
                              onClick={() => handleUserClick(user.user_id)}
                              className="hover:bg-gray-50 dark:hover:bg-gray-800 cursor-pointer transition-colors"
                            >
                              <td className="px-3 py-4 whitespace-nowrap">
                                <div className="text-sm font-medium text-gray-900 dark:text-gray-100">
                                  {user.email || '—'}
                                </div>
                              </td>
                              <td className="px-3 py-4 whitespace-nowrap">
                                <div className="text-sm text-gray-600 dark:text-gray-400">
                                  {user.username || '—'}
                                </div>
                              </td>
                              <td className="px-3 py-4 whitespace-nowrap">
                                <div className="flex items-center gap-2">
                                  <code className="text-xs text-gray-600 dark:text-gray-400 bg-gray-100 dark:bg-gray-800 px-2 py-1 rounded">
                                    {user.user_id}
                                  </code>
                                  <CopyButton text={user.user_id} />
                                </div>
                              </td>
                              <td className="px-3 py-4 whitespace-nowrap">
                                <Badge variant="neutral" size="sm">
                                  {user.auth_method}
                                </Badge>
                              </td>
                              <td className="px-3 py-4 whitespace-nowrap text-sm text-gray-600 dark:text-gray-400">
                                {user.last_seen_at
                                  ? formatDistanceToNow(new Date(user.last_seen_at), {
                                      addSuffix: true,
                                    })
                                  : '—'}
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    </div>
                  )}
                </div>
              </Card>
            )}

            {/* Initial Instructions */}
            {!showResults && (
              <Card>
                <div className="p-6">
                  <div className="flex items-start gap-3">
                    <UsersIcon className="h-6 w-6 text-gray-400 flex-shrink-0 mt-0.5" />
                    <div>
                      <h3 className="text-base font-semibold text-gray-900 dark:text-gray-100 mb-2">
                        Search for Users
                      </h3>
                      <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
                        Use the search box above to find users by email or username. Search
                        supports partial matches.
                      </p>
                      <ul className="space-y-1 text-sm text-gray-600 dark:text-gray-400">
                        <li className="flex items-start gap-2">
                          <span className="text-gray-400 mt-0.5">•</span>
                          <span>
                            Click on a user row to view details and manage their agents
                          </span>
                        </li>
                        <li className="flex items-start gap-2">
                          <span className="text-gray-400 mt-0.5">•</span>
                          <span>The canonical user_id is displayed and copyable</span>
                        </li>
                        <li className="flex items-start gap-2">
                          <span className="text-gray-400 mt-0.5">•</span>
                          <span>Cross-user operations available from user details page</span>
                        </li>
                      </ul>
                    </div>
                  </div>
                </div>
              </Card>
            )}
          </div>
        </AdminLayout>
      </PageContent>
    </>
  );
}
