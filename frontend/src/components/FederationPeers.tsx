import React, { useState, useMemo, useEffect, useRef, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { useNavigate } from 'react-router-dom';
import {
  PlusIcon,
  MagnifyingGlassIcon,
  ArrowPathIcon,
  EllipsisVerticalIcon,
  PencilIcon,
  TrashIcon,
  PlayIcon,
  ExclamationCircleIcon,
} from '@heroicons/react/24/outline';
import { Menu, Transition } from '@headlessui/react';
import {
  useFederationPeers,
  PeerRegistry,
  PeerWithStatus,
  deletePeer,
  syncPeer,
} from '../hooks/useFederationPeers';
import useEscapeKey from '../hooks/useEscapeKey';


/**
 * Props for the FederationPeers component.
 */
interface FederationPeersProps {
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
}


/**
 * Health status type for peers.
 */
type PeerHealthStatus = 'healthy' | 'warning' | 'error' | 'unknown';


/**
 * Get health status color classes.
 */
function getHealthColorClasses(health: PeerHealthStatus): string {
  switch (health) {
    case 'healthy':
      return 'bg-success-500';
    case 'warning':
      return 'bg-warning-500';
    case 'error':
      return 'bg-danger-500';
    default:
      return 'bg-gray-400';
  }
}


/**
 * Format last sync time for display.
 */
function formatLastSync(dateString: string | null | undefined): string {
  if (!dateString) return 'Never';

  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;

  return date.toLocaleDateString();
}


/**
 * Props for PeerActionMenu component.
 */
interface PeerActionMenuProps {
  peer: PeerRegistry;
  isSyncing: boolean;
  onSync: () => void;
  onEdit: () => void;
  onDelete: () => void;
}


/**
 * PeerActionMenu renders the action dropdown for a peer row.
 * Uses portal to escape overflow containers.
 */
const PeerActionMenu: React.FC<PeerActionMenuProps> = ({
  peer,
  isSyncing,
  onSync,
  onEdit,
  onDelete,
}) => {
  const buttonRef = useRef<HTMLButtonElement>(null);
  const [menuPosition, setMenuPosition] = useState({ top: 0, left: 0 });

  const updatePosition = useCallback(() => {
    if (buttonRef.current) {
      const rect = buttonRef.current.getBoundingClientRect();
      setMenuPosition({
        top: rect.bottom + 4,
        left: rect.right - 192, // 192px = w-48 width
      });
    }
  }, []);

  return (
    <Menu as="div" className="relative inline-block text-left">
      {({ open }) => {
        // Update position when menu opens
        if (open) {
          // Use setTimeout to ensure DOM is ready
          setTimeout(updatePosition, 0);
        }

        return (
          <>
            <Menu.Button
              ref={buttonRef}
              className="btn-icon"
            >
              <EllipsisVerticalIcon className="h-5 w-5" />
            </Menu.Button>
            {open &&
              createPortal(
                <Transition
                  show={open}
                  enter="transition ease-out duration-100"
                  enterFrom="transform opacity-0 scale-95"
                  enterTo="transform opacity-100 scale-100"
                  leave="transition ease-in duration-75"
                  leaveFrom="transform opacity-100 scale-100"
                  leaveTo="transform opacity-0 scale-95"
                >
                  <Menu.Items
                    static
                    className="fixed z-[9999] w-48 rounded-lg bg-surface-overlay border border-line shadow-lg focus:outline-none"
                    style={{
                      top: menuPosition.top,
                      left: menuPosition.left,
                    }}
                  >
                    <div className="py-1">
                      <Menu.Item>
                        {({ active }) => (
                          <button
                            onClick={onSync}
                            disabled={isSyncing || !peer.enabled}
                            className={`${
                              active ? 'bg-surface-hover' : ''
                            } flex items-center w-full px-4 py-2 text-sm text-fg-muted disabled:opacity-50`}
                          >
                            {isSyncing ? (
                              <ArrowPathIcon className="h-4 w-4 mr-3 animate-spin" />
                            ) : (
                              <PlayIcon className="h-4 w-4 mr-3" />
                            )}
                            {isSyncing ? 'Syncing...' : 'Sync Now'}
                          </button>
                        )}
                      </Menu.Item>
                      <Menu.Item>
                        {({ active }) => (
                          <button
                            onClick={onEdit}
                            className={`${
                              active ? 'bg-surface-hover' : ''
                            } flex items-center w-full px-4 py-2 text-sm text-fg-muted`}
                          >
                            <PencilIcon className="h-4 w-4 mr-3" />
                            Edit
                          </button>
                        )}
                      </Menu.Item>
                      <div className="border-t border-line my-1" />
                      <Menu.Item>
                        {({ active }) => (
                          <button
                            onClick={onDelete}
                            className={`${
                              active ? 'bg-surface-hover' : ''
                            } flex items-center w-full px-4 py-2 text-sm text-danger-600 dark:text-danger-400`}
                          >
                            <TrashIcon className="h-4 w-4 mr-3" />
                            Delete
                          </button>
                        )}
                      </Menu.Item>
                    </div>
                  </Menu.Items>
                </Transition>,
                document.body
              )}
          </>
        );
      }}
    </Menu>
  );
};


/**
 * FederationPeers component displays a list of configured peer registries.
 *
 * Provides functionality to view, search, sync, and delete peers.
 */
const FederationPeers: React.FC<FederationPeersProps> = ({ onShowToast }) => {
  const navigate = useNavigate();
  const { peers, isLoading, error, refetch } = useFederationPeers();

  const [searchQuery, setSearchQuery] = useState('');
  const [syncingPeers, setSyncingPeers] = useState<Set<string>>(new Set());
  const [deleteTarget, setDeleteTarget] = useState<PeerWithStatus | null>(null);
  const [typedName, setTypedName] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);

  useEscapeKey(() => { setDeleteTarget(null); setTypedName(''); }, !!deleteTarget);

  // Auto-refresh every 30 seconds for sync status updates
  useEffect(() => {
    const interval = setInterval(refetch, 30000);
    return () => clearInterval(interval);
  }, [refetch]);

  // Filter peers by search query
  const filteredPeers = useMemo(() => {
    if (!searchQuery) return peers;
    const query = searchQuery.toLowerCase();
    return peers.filter(
      (peer) =>
        peer.peer_id.toLowerCase().includes(query) ||
        peer.name.toLowerCase().includes(query) ||
        peer.endpoint.toLowerCase().includes(query)
    );
  }, [peers, searchQuery]);

  /**
   * Calculate health status for a peer based on sync status.
   */
  const getPeerHealth = (peer: PeerWithStatus): PeerHealthStatus => {
    if (!peer.enabled) return 'unknown';
    if (!peer.syncStatus) return 'unknown';
    if (peer.syncStatus.consecutive_failures > 2) return 'error';
    if (peer.syncStatus.consecutive_failures > 0) return 'warning';
    if (peer.syncStatus.is_healthy) return 'healthy';
    return 'unknown';
  };

  /**
   * Handle manual sync for a peer.
   */
  const handleSync = async (peer: PeerRegistry) => {
    setSyncingPeers((prev) => new Set(prev).add(peer.peer_id));
    try {
      const result = await syncPeer(peer.peer_id);
      if (result.success) {
        onShowToast(
          `Synced ${result.servers_synced} servers and ${result.agents_synced} agents from "${peer.name}"`,
          'success'
        );
      } else {
        onShowToast(
          result.error_message || `Sync failed for "${peer.name}"`,
          'error'
        );
      }
      await refetch();
    } catch (err: any) {
      onShowToast(
        err.response?.data?.detail || `Failed to sync "${peer.name}"`,
        'error'
      );
    } finally {
      setSyncingPeers((prev) => {
        const next = new Set(prev);
        next.delete(peer.peer_id);
        return next;
      });
    }
  };

  /**
   * Handle peer deletion.
   */
  const handleDelete = async () => {
    if (!deleteTarget || typedName !== deleteTarget.name) return;

    setIsDeleting(true);
    try {
      await deletePeer(deleteTarget.peer_id);
      onShowToast(`Peer "${deleteTarget.name}" has been deleted`, 'success');
      setDeleteTarget(null);
      setTypedName('');
      await refetch();
    } catch (err: any) {
      onShowToast(
        err.response?.data?.detail || `Failed to delete peer`,
        'error'
      );
    } finally {
      setIsDeleting(false);
    }
  };

  /**
   * Get sync mode display text.
   */
  const getSyncModeLabel = (peer: PeerRegistry): string => {
    switch (peer.sync_mode) {
      case 'all':
        return 'All Public';
      case 'whitelist':
        return 'Whitelist';
      case 'tag_filter':
        return `Tags: ${peer.tag_filters?.join(', ') || 'None'}`;
      default:
        return peer.sync_mode;
    }
  };

  // Loading state
  if (isLoading) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="h-8 w-48 bg-surface-hover rounded animate-pulse" />
          <div className="h-10 w-32 bg-surface-hover rounded animate-pulse" />
        </div>
        <div className="h-10 w-64 bg-surface-hover rounded animate-pulse" />
        <div className="space-y-2">
          {[1, 2, 3].map((i) => (
            <div
              key={i}
              className="h-16 bg-surface-hover rounded animate-pulse"
            />
          ))}
        </div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="text-center py-12">
        <ExclamationCircleIcon className="h-12 w-12 mx-auto text-danger-500 mb-4" />
        <h3 className="text-lg font-medium text-fg mb-2">
          Failed to Load Peers
        </h3>
        <p className="text-fg-subtle mb-4">{error}</p>
        <button
          onClick={refetch}
          className="btn-primary"
        >
          Retry
        </button>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-fg">
            Federation Peers
          </h2>
          <p className="text-sm text-fg-subtle">
            Manage peer registries for cross-registry synchronization
          </p>
        </div>
        <button
          onClick={() => navigate('/settings/federation/peers/add')}
          className="btn-primary"
        >
          <PlusIcon className="h-5 w-5" />
          Add Peer
        </button>
      </div>

      {/* Search */}
      <div className="relative">
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search peers..."
          className="input pl-10 pr-4"
        />
      </div>

      {/* Peers table */}
      {filteredPeers.length === 0 ? (
        <div className="text-center py-12 bg-surface-muted rounded-lg">
          <svg
            className="h-12 w-12 mx-auto text-fg-subtle mb-4"
            fill="none"
            viewBox="0 0 24 24"
            stroke="currentColor"
          >
            <path
              strokeLinecap="round"
              strokeLinejoin="round"
              strokeWidth={1.5}
              d="M21 12a9 9 0 01-9 9m9-9a9 9 0 00-9-9m9 9H3m9 9a9 9 0 01-9-9m9 9c1.657 0 3-4.03 3-9s-1.343-9-3-9m0 18c-1.657 0-3-4.03-3-9s1.343-9 3-9m-9 9a9 9 0 019-9"
            />
          </svg>
          <h3 className="text-lg font-medium text-fg mb-2">
            {searchQuery ? 'No matching peers' : 'No peers configured'}
          </h3>
          <p className="text-fg-subtle mb-4">
            {searchQuery
              ? 'Try a different search term'
              : 'Add a peer registry to enable federation'}
          </p>
          {!searchQuery && (
            <button
              onClick={() => navigate('/settings/federation/peers/add')}
              className="btn-primary"
            >
              Add First Peer
            </button>
          )}
        </div>
      ) : (
        <div className="overflow-x-auto">
          <table className="min-w-full divide-y divide-line">
            <thead className="bg-surface-muted">
              <tr>
                <th className="px-4 py-3 text-left text-xs font-medium text-fg-subtle uppercase tracking-wider">
                  Name
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-fg-subtle uppercase tracking-wider">
                  Endpoint
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-fg-subtle uppercase tracking-wider">
                  Status
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-fg-subtle uppercase tracking-wider">
                  Sync Mode
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-fg-subtle uppercase tracking-wider">
                  Interval
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-fg-subtle uppercase tracking-wider">
                  Last Sync
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-fg-subtle uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-surface divide-y divide-line">
              {filteredPeers.map((peer) => {
                const health = getPeerHealth(peer);
                const isSyncing = syncingPeers.has(peer.peer_id);

                return (
                  <tr key={peer.peer_id} className="table-row">
                    <td className="px-4 py-4 whitespace-nowrap">
                      <div className="flex flex-col">
                        <span className="text-sm font-medium text-fg">
                          {peer.name}
                        </span>
                        <span className="text-xs text-fg-subtle">
                          {peer.peer_id}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      <span
                        className="text-sm text-fg-muted truncate block max-w-[200px]"
                        title={peer.endpoint}
                      >
                        {peer.endpoint}
                      </span>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      <div className="flex items-center space-x-2">
                        <span
                          className={`h-2 w-2 rounded-full ${getHealthColorClasses(health)}`}
                        />
                        <span className="text-sm text-fg-muted capitalize">
                          {peer.enabled ? 'Enabled' : 'Disabled'}
                        </span>
                      </div>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap">
                      <span className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-surface-hover text-fg-muted">
                        {getSyncModeLabel(peer)}
                      </span>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-sm text-fg-muted">
                      {peer.sync_interval_minutes}m
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-sm text-fg-muted">
                      <span title={peer.syncStatus?.last_successful_sync || 'Never synced'}>
                        {formatLastSync(peer.syncStatus?.last_successful_sync)}
                      </span>
                    </td>
                    <td className="px-4 py-4 whitespace-nowrap text-right">
                      <PeerActionMenu
                        peer={peer}
                        isSyncing={isSyncing}
                        onSync={() => handleSync(peer)}
                        onEdit={() => navigate(`/settings/federation/peers/${peer.peer_id}/edit`)}
                        onDelete={() => setDeleteTarget(peer)}
                      />
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      )}

      {/* Delete confirmation modal */}
      {deleteTarget && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-surface-overlay border border-line rounded-lg shadow-xl max-w-md w-full mx-4 p-6">
            <h3 className="text-lg font-semibold text-fg mb-2">
              Delete Peer
            </h3>
            <p className="text-sm text-fg-muted mb-4">
              This action is irreversible. All servers and agents synced from this
              peer will be removed.
            </p>
            <p className="text-sm text-fg-muted mb-3">
              Type <strong>{deleteTarget.name}</strong> to confirm:
            </p>
            <input
              type="text"
              value={typedName}
              onChange={(e) => setTypedName(e.target.value)}
              placeholder={deleteTarget.name}
              disabled={isDeleting}
              className="input mb-4"
            />
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => {
                  setDeleteTarget(null);
                  setTypedName('');
                }}
                disabled={isDeleting}
                className="btn-secondary"
              >
                Cancel
              </button>
              <button
                onClick={handleDelete}
                disabled={typedName !== deleteTarget.name || isDeleting}
                className="btn-danger"
              >
                {isDeleting && <ArrowPathIcon className="h-4 w-4 animate-spin" />}
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default FederationPeers;
