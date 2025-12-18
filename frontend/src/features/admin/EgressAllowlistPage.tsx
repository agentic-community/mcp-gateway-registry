/**
 * EgressAllowlistPage - Admin page for managing egress allowlist entries
 * Used to prevent SSRF attacks by controlling which upstream servers can be proxied
 */

import { useState, useEffect } from 'react';
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
import { EgressAllowlistModal } from './EgressAllowlistModal';
import {
  ShieldCheckIcon,
  PlusIcon,
  PencilIcon,
  TrashIcon,
  ExclamationTriangleIcon,
  ClockIcon,
} from '@heroicons/react/24/outline';
import {
  getEgressAllowlistEntries,
  deleteEgressAllowlistEntry,
} from '@/api/admin';
import type { EgressAllowlistEntry } from '@/api/types';
import { formatDistanceToNow, isPast, parseISO } from 'date-fns';

/**
 * Main egress allowlist page component
 */
export default function EgressAllowlistPage() {
  const { addToast } = useToast();
  const [entries, setEntries] = useState<EgressAllowlistEntry[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [deleteDialogOpen, setDeleteDialogOpen] = useState(false);
  const [entryToDelete, setEntryToDelete] = useState<EgressAllowlistEntry | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);
  const [modalOpen, setModalOpen] = useState(false);
  const [entryToEdit, setEntryToEdit] = useState<EgressAllowlistEntry | null>(null);

  // Fetch entries
  const fetchEntries = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getEgressAllowlistEntries();
      setEntries(data);
    } catch (err: any) {
      console.error('Failed to fetch egress allowlist:', err);
      setError(err.message || 'Failed to load egress allowlist');
      addToast({
        type: 'error',
        title: 'Failed to load allowlist',
        message: err.message || 'An error occurred',
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchEntries();
  }, []);

  // Handle create
  const handleCreateClick = () => {
    setEntryToEdit(null);
    setModalOpen(true);
  };

  // Handle edit
  const handleEditClick = (entry: EgressAllowlistEntry) => {
    setEntryToEdit(entry);
    setModalOpen(true);
  };

  // Handle modal close
  const handleModalClose = () => {
    setModalOpen(false);
    setEntryToEdit(null);
  };

  // Handle modal success
  const handleModalSuccess = () => {
    fetchEntries();
  };

  // Handle delete
  const handleDeleteClick = (entry: EgressAllowlistEntry) => {
    setEntryToDelete(entry);
    setDeleteDialogOpen(true);
  };

  const handleDeleteConfirm = async () => {
    if (!entryToDelete) return;

    try {
      setIsDeleting(true);
      await deleteEgressAllowlistEntry(entryToDelete.entry_id);
      addToast({
        type: 'success',
        title: 'Entry deleted',
        message: `Pattern "${entryToDelete.pattern}" has been removed from the allowlist`,
      });
      fetchEntries();
    } catch (err: any) {
      console.error('Failed to delete entry:', err);
      addToast({
        type: 'error',
        title: 'Failed to delete entry',
        message: err.message || 'An error occurred',
      });
    } finally {
      setIsDeleting(false);
      setDeleteDialogOpen(false);
      setEntryToDelete(null);
    }
  };

  // Check if entry is expired
  const isExpired = (entry: EgressAllowlistEntry): boolean => {
    if (!entry.expires_at) return false;
    return isPast(parseISO(entry.expires_at));
  };

  // Get expiration status
  const getExpirationStatus = (entry: EgressAllowlistEntry): {
    expired: boolean;
    text: string | null;
  } => {
    if (!entry.expires_at) {
      return { expired: false, text: null };
    }

    const expiresDate = parseISO(entry.expires_at);
    const expired = isPast(expiresDate);

    if (expired) {
      return {
        expired: true,
        text: `Expired ${formatDistanceToNow(expiresDate, { addSuffix: true })}`,
      };
    }

    return {
      expired: false,
      text: `Expires ${formatDistanceToNow(expiresDate, { addSuffix: true })}`,
    };
  };

  return (
    <>
      <PageHeader
        title="Egress Allowlist"
        description="Control which upstream servers can be proxied to prevent SSRF attacks"
      />

      <PageContent>
        <AdminLayout>
          <div className="space-y-6">
            {/* SSRF Warning Banner */}
            <Card className="border-yellow-200 dark:border-yellow-800 bg-yellow-50 dark:bg-yellow-900/20">
              <div className="flex items-start space-x-3 p-4">
                <ExclamationTriangleIcon className="h-6 w-6 text-yellow-600 dark:text-yellow-400 flex-shrink-0 mt-0.5" />
                <div className="flex-1">
                  <h3 className="text-sm font-semibold text-yellow-800 dark:text-yellow-200">
                    SSRF Protection
                  </h3>
                  <p className="mt-1 text-sm text-yellow-700 dark:text-yellow-300">
                    This allowlist prevents Server-Side Request Forgery (SSRF) attacks by restricting
                    which upstream URLs can be registered as MCP servers. Only patterns listed here
                    will be accepted during server registration.
                  </p>
                  <p className="mt-2 text-sm text-yellow-700 dark:text-yellow-300 font-medium">
                    Exercise caution when adding patterns. Wildcards should be used sparingly.
                  </p>
                </div>
              </div>
            </Card>

            {/* Actions Bar */}
            <div className="flex items-center justify-between">
              <div className="flex items-center space-x-2">
                <ShieldCheckIcon className="h-5 w-5 text-gray-400" />
                <h2 className="text-lg font-medium text-gray-900 dark:text-gray-100">
                  Allowed Patterns
                </h2>
                <Badge variant="neutral" size="sm">
                  {entries.length} {entries.length === 1 ? 'entry' : 'entries'}
                </Badge>
              </div>
              <Button
                variant="primary"
                size="sm"
                leftIcon={<PlusIcon className="h-4 w-4" />}
                onClick={handleCreateClick}
              >
                Add Pattern
              </Button>
            </div>

            {/* Entries List */}
            {loading ? (
              <div className="flex items-center justify-center py-12">
                <Spinner size="lg" />
              </div>
            ) : error ? (
              <Card>
                <EmptyState
                  icon={<ExclamationTriangleIcon className="w-12 h-12" />}
                  title="Failed to load allowlist"
                  description={error}
                  action={
                    <Button variant="primary" onClick={fetchEntries}>
                      Retry
                    </Button>
                  }
                />
              </Card>
            ) : entries.length === 0 ? (
              <Card>
                <EmptyState
                  icon={<ShieldCheckIcon className="w-12 h-12" />}
                  title="No allowlist entries"
                  description="Add patterns to control which upstream servers can be proxied"
                  action={
                    <Button
                      variant="primary"
                      leftIcon={<PlusIcon className="h-4 w-4" />}
                      onClick={handleCreateClick}
                    >
                      Add First Pattern
                    </Button>
                  }
                />
              </Card>
            ) : (
              <div className="space-y-3">
                {entries.map((entry) => {
                  const expirationStatus = getExpirationStatus(entry);
                  const expired = expirationStatus.expired;

                  return (
                    <Card key={entry.entry_id}>
                      <div className="p-4">
                        <div className="flex items-start justify-between">
                          <div className="flex-1 min-w-0">
                            <div className="flex items-center space-x-2 mb-2">
                              <code className="text-sm font-mono font-semibold text-gray-900 dark:text-gray-100">
                                {entry.pattern}
                              </code>
                              {expired && (
                                <Badge variant="error" size="sm">
                                  Expired
                                </Badge>
                              )}
                              {!expired && expirationStatus.text && (
                                <Badge variant="warning" size="sm">
                                  {expirationStatus.text}
                                </Badge>
                              )}
                            </div>
                            {entry.description && (
                              <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                                {entry.description}
                              </p>
                            )}
                            <div className="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
                              <span>
                                Added {formatDistanceToNow(parseISO(entry.created_at), { addSuffix: true })}
                              </span>
                              {entry.updated_at !== entry.created_at && (
                                <span>
                                  Updated {formatDistanceToNow(parseISO(entry.updated_at), { addSuffix: true })}
                                </span>
                              )}
                            </div>
                          </div>
                          <div className="flex items-center space-x-2 ml-4 flex-shrink-0">
                            <Button
                              variant="ghost"
                              size="sm"
                              leftIcon={<PencilIcon className="h-4 w-4" />}
                              onClick={() => handleEditClick(entry)}
                            >
                              Edit
                            </Button>
                            <Button
                              variant="danger"
                              size="sm"
                              leftIcon={<TrashIcon className="h-4 w-4" />}
                              onClick={() => handleDeleteClick(entry)}
                            >
                              Delete
                            </Button>
                          </div>
                        </div>
                      </div>
                    </Card>
                  );
                })}
              </div>
            )}
          </div>
        </AdminLayout>
      </PageContent>

      {/* Delete Confirmation Dialog */}
      <ConfirmDialog
        open={deleteDialogOpen}
        onClose={() => {
          setDeleteDialogOpen(false);
          setEntryToDelete(null);
        }}
        onConfirm={handleDeleteConfirm}
        title="Delete Allowlist Entry"
        message={
          entryToDelete
            ? `Are you sure you want to delete the pattern "${entryToDelete.pattern}"? This may prevent servers using this pattern from being registered.`
            : ''
        }
        confirmLabel="Delete"
        variant="danger"
        loading={isDeleting}
      />

      {/* Create/Edit Modal */}
      <EgressAllowlistModal
        open={modalOpen}
        onClose={handleModalClose}
        onSuccess={handleModalSuccess}
        entry={entryToEdit}
      />
    </>
  );
}
