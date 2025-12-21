/**
 * UpstreamOAuthProvidersPage - Admin page for managing upstream OAuth providers
 */

import { useEffect, useMemo, useState } from 'react';
import { PageHeader } from '@/components/layout/PageHeader';
import { PageContent } from '@/components/layout/PageContent';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Spinner } from '@/components/ui/Spinner';
import { EmptyState } from '@/components/ui/EmptyState';
import { useToast } from '@/components/ui/Toast';
import { ConfirmDialog } from '@/components/ui/ConfirmDialog';
import { AdminLayout } from './AdminLayout';
import { UpstreamOAuthProviderModal } from './UpstreamOAuthProviderModal';
import {
  KeyIcon,
  PlusIcon,
  PencilIcon,
  TrashIcon,
} from '@heroicons/react/24/outline';
import {
  deleteUpstreamOAuthProvider,
  listUpstreamOAuthProviders,
} from '@/api/admin';
import type { UpstreamOAuthProviderPublic } from '@/api/types';
import { formatDistanceToNow, parseISO } from 'date-fns';

export default function UpstreamOAuthProvidersPage() {
  const { addToast } = useToast();
  const [providers, setProviders] = useState<UpstreamOAuthProviderPublic[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const [modalOpen, setModalOpen] = useState(false);
  const [providerToEdit, setProviderToEdit] = useState<UpstreamOAuthProviderPublic | null>(null);

  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [providerToDelete, setProviderToDelete] = useState<UpstreamOAuthProviderPublic | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  const fetchProviders = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await listUpstreamOAuthProviders();
      setProviders(data);
    } catch (err: any) {
      setError(err?.message || 'Failed to load upstream OAuth providers');
      addToast({
        type: 'error',
        title: 'Failed to load providers',
        message: err?.message || 'An error occurred',
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchProviders();
  }, []);

  const providerCountLabel = useMemo(() => {
    const count = providers.length;
    return `${count} provider${count === 1 ? '' : 's'}`;
  }, [providers.length]);

  const handleCreate = () => {
    setProviderToEdit(null);
    setModalOpen(true);
  };

  const handleEdit = (provider: UpstreamOAuthProviderPublic) => {
    setProviderToEdit(provider);
    setModalOpen(true);
  };

  const handleModalClose = () => {
    setModalOpen(false);
    setProviderToEdit(null);
  };

  const handleDeleteClick = (provider: UpstreamOAuthProviderPublic) => {
    setProviderToDelete(provider);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!providerToDelete) {
      return;
    }

    setIsDeleting(true);
    try {
      await deleteUpstreamOAuthProvider(providerToDelete.provider.provider_id);
      addToast({
        type: 'success',
        title: 'Provider deleted',
        message: `Provider "${providerToDelete.provider.provider_id}" has been deleted`,
      });
      await fetchProviders();
    } catch (err: any) {
      addToast({
        type: 'error',
        title: 'Failed to delete provider',
        message: err?.message || 'An error occurred',
      });
    } finally {
      setIsDeleting(false);
      setDeleteDialogOpen(false);
      setProviderToDelete(null);
    }
  };

  return (
    <>
      <PageHeader
        title="Upstream OAuth Providers"
        description="Manage OAuth client configurations used for gateway-terminated upstream OAuth"
      />

      <PageContent>
        <AdminLayout>
          <div className="space-y-6">
            <Card>
              <div className="p-6 space-y-4">
                <div className="flex items-start gap-3">
                  <div className="flex-shrink-0 mt-0.5">
                    <KeyIcon className="h-6 w-6 text-gray-700 dark:text-gray-300" />
                  </div>
                  <div className="flex-1">
                    <h3 className="text-base font-semibold text-gray-900 dark:text-gray-100">
                      Provider Registry
                    </h3>
                    <p className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                      Providers define authorization/token endpoints and client credentials used to
                      complete OAuth flows. Client secrets are write-only and never shown after
                      creation.
                    </p>
                  </div>
                </div>

                <div className="flex items-center justify-between">
                  <div className="flex items-center gap-2">
                    <Badge variant="info">{providerCountLabel}</Badge>
                    {error ? (
                      <span className="text-sm text-red-600 dark:text-red-400">{error}</span>
                    ) : null}
                  </div>
                  <Button variant="primary" onClick={handleCreate}>
                    <PlusIcon className="h-4 w-4 mr-2" />
                    Add Provider
                  </Button>
                </div>
              </div>
            </Card>

            {loading ? (
              <div className="flex items-center justify-center py-12">
                <Spinner size="lg" />
              </div>
            ) : providers.length === 0 ? (
              <EmptyState
                title="No providers configured"
                description="Create an upstream OAuth provider to enable OAuth connect flows for upstream servers."
                action={
                  <Button variant="primary" onClick={handleCreate}>
                    <PlusIcon className="h-4 w-4 mr-2" />
                    Add Provider
                  </Button>
                }
              />
            ) : (
              <Card>
                <div className="p-6 overflow-x-auto">
                  <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
                    <thead>
                      <tr>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Provider
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Client ID
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Endpoints
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Secret
                        </th>
                        <th className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Updated
                        </th>
                        <th className="px-4 py-3 text-right text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                          Actions
                        </th>
                      </tr>
                    </thead>
                    <tbody className="divide-y divide-gray-100 dark:divide-gray-800">
                      {providers.map((item) => (
                        <tr key={item.provider.provider_id}>
                          <td className="px-4 py-3">
                            <div className="text-sm font-mono text-gray-900 dark:text-gray-100">
                              {item.provider.provider_id}
                            </div>
                            <div className="text-xs text-gray-500 dark:text-gray-400">
                              {item.provider.default_scopes.length > 0
                                ? `${item.provider.default_scopes.length} scope${item.provider.default_scopes.length === 1 ? '' : 's'}`
                                : 'no default scopes'}
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <div className="text-sm font-mono text-gray-900 dark:text-gray-100">
                              {item.provider.client_id}
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            <div className="text-xs text-gray-600 dark:text-gray-400 space-y-1">
                              <div className="truncate max-w-[420px]">
                                <span className="font-medium">Auth:</span>{' '}
                                <span className="font-mono">{item.provider.authorization_endpoint}</span>
                              </div>
                              <div className="truncate max-w-[420px]">
                                <span className="font-medium">Token:</span>{' '}
                                <span className="font-mono">{item.provider.token_endpoint}</span>
                              </div>
                            </div>
                          </td>
                          <td className="px-4 py-3">
                            {item.secret_present ? (
                              <Badge variant="success">Set</Badge>
                            ) : (
                              <Badge variant="warning">Missing</Badge>
                            )}
                          </td>
                          <td className="px-4 py-3">
                            <div className="text-sm text-gray-600 dark:text-gray-400">
                              {formatDistanceToNow(parseISO(item.provider.updated_at), { addSuffix: true })}
                            </div>
                          </td>
                          <td className="px-4 py-3 text-right space-x-2">
                            <Button
                              variant="secondary"
                              size="sm"
                              onClick={() => handleEdit(item)}
                              aria-label={`Edit ${item.provider.provider_id}`}
                            >
                              <PencilIcon className="h-4 w-4" />
                            </Button>
                            <Button
                              variant="danger"
                              size="sm"
                              onClick={() => handleDeleteClick(item)}
                              aria-label={`Delete ${item.provider.provider_id}`}
                            >
                              <TrashIcon className="h-4 w-4" />
                            </Button>
                          </td>
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </Card>
            )}

            <UpstreamOAuthProviderModal
              open={modalOpen}
              onClose={handleModalClose}
              onSuccess={fetchProviders}
              provider={providerToEdit}
            />

            <ConfirmDialog
              open={deleteDialogOpen}
              onClose={() => {
                if (isDeleting) {
                  return;
                }
                setDeleteDialogOpen(false);
                setProviderToDelete(null);
              }}
              onConfirm={handleDeleteConfirm}
              title="Delete Provider"
              message={
                providerToDelete
                  ? `Delete provider "${providerToDelete.provider.provider_id}"? This does not revoke existing user tokens, but it may break new OAuth connections for servers that reference this provider.`
                  : 'Delete this provider?'
              }
              confirmLabel="Delete"
              variant="danger"
              loading={isDeleting}
            />
          </div>
        </AdminLayout>
      </PageContent>
    </>
  );
}

