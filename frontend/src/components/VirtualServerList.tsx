import React, { useState, useEffect } from 'react';
import { useSearchParams } from 'react-router-dom';
import {
  PlusIcon,
  MagnifyingGlassIcon,
  ExclamationCircleIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';
import { useAuth } from '../contexts/AuthContext';
import {
  useVirtualServers,
  useVirtualServer,
} from '../hooks/useVirtualServers';
import {
  VirtualServerInfo,
  CreateVirtualServerRequest,
  UpdateVirtualServerRequest,
} from '../types/virtualServer';
import VirtualServerForm from './VirtualServerForm';
import useEscapeKey from '../hooks/useEscapeKey';


/**
 * Props for VirtualServerList component.
 */
interface VirtualServerListProps {
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
}


/**
 * VirtualServerList displays a table of all virtual MCP servers
 * with search, create, edit, delete, and toggle functionality.
 */
const VirtualServerList: React.FC<VirtualServerListProps> = ({ onShowToast }) => {
  const { user } = useAuth();
  const [searchParams, setSearchParams] = useSearchParams();
  const {
    virtualServers,
    loading,
    error,
    refreshData,
    createVirtualServer,
    updateVirtualServer,
    deleteVirtualServer,
    toggleVirtualServer,
  } = useVirtualServers();

  const [searchQuery, setSearchQuery] = useState('');
  const [showForm, setShowForm] = useState(false);
  const [editingPath, setEditingPath] = useState<string | undefined>(undefined);
  const [deleteTarget, setDeleteTarget] = useState<VirtualServerInfo | null>(null);
  const [typedName, setTypedName] = useState('');
  const [isDeleting, setIsDeleting] = useState(false);

  useEscapeKey(() => { setShowForm(false); setEditingPath(undefined); }, showForm);
  useEscapeKey(() => { setDeleteTarget(null); setTypedName(''); }, !!deleteTarget);

  const canModify = user?.can_modify_servers || user?.is_admin || false;

  // Fetch full config when editing
  const { virtualServer: editingServer, loading: editingServerLoading } = useVirtualServer(editingPath);

  // Handle ?edit=<path> query parameter from Dashboard navigation
  useEffect(() => {
    const editParam = searchParams.get('edit');
    if (editParam && !loading && virtualServers.length > 0) {
      const decodedPath = decodeURIComponent(editParam);
      const serverExists = virtualServers.some((s) => s.path === decodedPath);
      if (serverExists) {
        setEditingPath(decodedPath);
        setShowForm(true);
      }
      // Clear the query param so it doesn't re-trigger
      searchParams.delete('edit');
      setSearchParams(searchParams, { replace: true });
    }
  }, [searchParams, loading, virtualServers]);

  // Filter servers by search
  const filteredServers = searchQuery
    ? virtualServers.filter(
        (s) =>
          s.server_name.toLowerCase().includes(searchQuery.toLowerCase()) ||
          s.path.toLowerCase().includes(searchQuery.toLowerCase()) ||
          s.description?.toLowerCase().includes(searchQuery.toLowerCase()) ||
          s.tags?.some((t) => t.toLowerCase().includes(searchQuery.toLowerCase()))
      )
    : virtualServers;

  const handleCreate = () => {
    setEditingPath(undefined);
    setShowForm(true);
  };

  const handleEdit = (server: VirtualServerInfo) => {
    setEditingPath(server.path);
    setShowForm(true);
  };

  const handleSave = async (
    data: CreateVirtualServerRequest | UpdateVirtualServerRequest,
  ) => {
    try {
      if (editingPath) {
        await updateVirtualServer(editingPath, data as UpdateVirtualServerRequest);
        onShowToast('Virtual server updated successfully', 'success');
      } else {
        await createVirtualServer(data as CreateVirtualServerRequest);
        onShowToast('Virtual server created successfully', 'success');
      }
      setShowForm(false);
      setEditingPath(undefined);
    } catch (err: unknown) {
      const message =
        err instanceof Error ? err.message : 'An unexpected error occurred';
      onShowToast(`Failed to save virtual server: ${message}`, 'error');
    }
  };

  const handleDelete = async () => {
    if (!deleteTarget || typedName !== deleteTarget.server_name) return;

    setIsDeleting(true);
    try {
      await deleteVirtualServer(deleteTarget.path);
      onShowToast(`Virtual server "${deleteTarget.server_name}" deleted`, 'success');
      setDeleteTarget(null);
      setTypedName('');
    } catch (err: unknown) {
      const axiosErr = err as { response?: { data?: { detail?: string } }; message?: string };
      onShowToast(
        axiosErr.response?.data?.detail || 'Failed to delete virtual server',
        'error',
      );
    } finally {
      setIsDeleting(false);
    }
  };

  const handleToggle = async (path: string, enabled: boolean) => {
    try {
      await toggleVirtualServer(path, enabled);
      onShowToast(
        `Virtual server ${enabled ? 'enabled' : 'disabled'}`,
        'success',
      );
    } catch {
      onShowToast('Failed to toggle virtual server', 'error');
    }
  };

  // Loading state
  if (loading) {
    return (
      <div className="space-y-4">
        <div className="flex items-center justify-between">
          <div className="h-8 w-48 bg-surface-hover rounded animate-pulse" />
          <div className="h-10 w-40 bg-surface-hover rounded animate-pulse" />
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
          Failed to Load Virtual Servers
        </h3>
        <p className="text-fg-subtle mb-4">{error}</p>
        <button
          onClick={refreshData}
          className="px-4 py-2 bg-virtual-600 text-white rounded-lg hover:bg-virtual-700"
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
            Virtual MCP Servers
          </h2>
          <p className="text-sm text-fg-subtle">
            Manage virtual servers that aggregate tools from multiple backends
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button
            onClick={refreshData}
            className="btn-icon"
            title="Refresh"
          >
            <ArrowPathIcon className="h-5 w-5" />
          </button>
          {canModify && (
            <button
              onClick={handleCreate}
              className="flex items-center px-4 py-2 bg-virtual-600 text-white rounded-lg
                         hover:bg-virtual-700 transition-colors"
            >
              <PlusIcon className="h-5 w-5 mr-2" />
              Create Virtual Server
            </button>
          )}
        </div>
      </div>

      {/* Search */}
      <div className="relative">
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-5 w-5 text-gray-400" />
        <input
          type="text"
          value={searchQuery}
          onChange={(e) => setSearchQuery(e.target.value)}
          placeholder="Search virtual servers..."
          className="input pl-10 pr-4"
        />
      </div>

      {/* Table */}
      {filteredServers.length === 0 ? (
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
              d="M5 12h14M5 12a2 2 0 01-2-2V6a2 2 0 012-2h14a2 2 0 012 2v4a2 2 0 01-2 2M5 12a2 2 0 00-2 2v4a2 2 0 002 2h14a2 2 0 002-2v-4a2 2 0 00-2-2"
            />
          </svg>
          <h3 className="text-lg font-medium text-fg mb-2">
            {searchQuery ? 'No matching virtual servers' : 'No virtual servers configured'}
          </h3>
          <p className="text-fg-subtle mb-4">
            {searchQuery
              ? 'Try a different search term'
              : 'Create a virtual server to aggregate tools from multiple backends'}
          </p>
          {!searchQuery && canModify && (
            <button
              onClick={handleCreate}
              className="px-4 py-2 bg-virtual-600 text-white rounded-lg hover:bg-virtual-700"
            >
              Create First Virtual Server
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
                  Path
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-fg-subtle uppercase tracking-wider">
                  Tools
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-fg-subtle uppercase tracking-wider">
                  Backends
                </th>
                <th className="px-4 py-3 text-left text-xs font-medium text-fg-subtle uppercase tracking-wider">
                  Status
                </th>
                <th className="px-4 py-3 text-right text-xs font-medium text-fg-subtle uppercase tracking-wider">
                  Actions
                </th>
              </tr>
            </thead>
            <tbody className="bg-surface divide-y divide-line">
              {filteredServers.map((server) => (
                <tr
                  key={server.path}
                  className="table-row"
                >
                  <td className="px-4 py-4 whitespace-nowrap">
                    <div className="flex flex-col">
                      <span className="text-sm font-medium text-fg">
                        {server.server_name}
                      </span>
                      {server.description && (
                        <span className="text-xs text-fg-subtle truncate max-w-[200px]">
                          {server.description}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <code className="text-sm text-fg-muted font-mono">
                      {server.path}
                    </code>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap text-sm text-fg-muted">
                    {server.tool_count}
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <div className="flex flex-wrap gap-1">
                      {server.backend_paths.slice(0, 2).map((bp) => (
                        <span
                          key={bp}
                          className="px-2 py-0.5 text-xs font-mono bg-surface-hover text-fg-muted rounded"
                        >
                          {bp}
                        </span>
                      ))}
                      {server.backend_paths.length > 2 && (
                        <span className="px-2 py-0.5 text-xs bg-surface-hover text-fg-subtle rounded">
                          +{server.backend_paths.length - 2}
                        </span>
                      )}
                    </div>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap">
                    <span
                      className={`inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium ${
                        server.is_enabled
                          ? 'bg-success-100 dark:bg-success-900/30 text-success-800 dark:text-success-400'
                          : 'bg-surface-hover text-fg-muted'
                      }`}
                    >
                      {server.is_enabled ? 'Enabled' : 'Disabled'}
                    </span>
                  </td>
                  <td className="px-4 py-4 whitespace-nowrap text-right">
                    <div className="flex items-center justify-end gap-2">
                      {canModify && (
                        <>
                          <label className="relative inline-flex items-center cursor-pointer">
                            <input
                              type="checkbox"
                              checked={server.is_enabled}
                              onChange={(e) =>
                                handleToggle(server.path, e.target.checked)
                              }
                              className="sr-only peer"
                              aria-label={`Enable ${server.server_name}`}
                            />
                            <div
                              className={`relative w-9 h-5 rounded-full transition-colors duration-200 ${
                                server.is_enabled
                                  ? 'bg-virtual-600'
                                  : 'bg-gray-300 dark:bg-gray-600'
                              }`}
                            >
                              <div
                                className={`absolute top-0.5 left-0.5 w-4 h-4 bg-white rounded-full transition-transform duration-200 ${
                                  server.is_enabled
                                    ? 'translate-x-4'
                                    : 'translate-x-0'
                                }`}
                              />
                            </div>
                          </label>
                          <button
                            onClick={() => handleEdit(server)}
                            className="px-3 py-1 text-xs font-medium text-virtual-700 dark:text-virtual-300
                                       bg-virtual-50 dark:bg-virtual-900/20 rounded hover:bg-virtual-100
                                       dark:hover:bg-virtual-900/40 transition-colors"
                          >
                            Edit
                          </button>
                          <button
                            onClick={() => setDeleteTarget(server)}
                            className="px-3 py-1 text-xs font-medium text-danger-700 dark:text-danger-300
                                       bg-danger-50 dark:bg-danger-900/20 rounded hover:bg-danger-100
                                       dark:hover:bg-danger-900/40 transition-colors"
                          >
                            Delete
                          </button>
                        </>
                      )}
                    </div>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {/* Form modal */}
      {showForm && editingPath && editingServerLoading && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/50">
          <div className="bg-surface-overlay border border-line rounded-lg shadow-xl p-8 flex flex-col items-center">
            <ArrowPathIcon className="h-8 w-8 text-virtual-500 animate-spin mb-3" />
            <p className="text-sm text-fg-muted">Loading server data...</p>
          </div>
        </div>
      )}
      {showForm && (!editingPath || (editingPath && !editingServerLoading)) && (
        <VirtualServerForm
          virtualServer={editingPath ? editingServer : null}
          onSave={handleSave}
          onCancel={() => {
            setShowForm(false);
            setEditingPath(undefined);
          }}
        />
      )}

      {/* Delete confirmation modal */}
      {deleteTarget && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
          role="dialog"
          aria-modal="true"
          aria-label="Delete virtual server confirmation"
        >
          <div className="bg-surface-overlay border border-line rounded-lg shadow-xl max-w-md w-full mx-4 p-6">
            <h3 className="text-lg font-semibold text-fg mb-2">
              Delete Virtual Server
            </h3>
            <p className="text-sm text-fg-muted mb-4">
              This action is irreversible. The virtual server and all its tool
              mappings will be permanently removed.
            </p>
            <p className="text-sm text-fg-muted mb-3">
              Type <strong>{deleteTarget.server_name}</strong> to confirm:
            </p>
            <input
              type="text"
              value={typedName}
              onChange={(e) => setTypedName(e.target.value)}
              placeholder={deleteTarget.server_name}
              disabled={isDeleting}
              className="input mb-4"
              onKeyDown={(e) => {
                if (e.key === 'Escape') {
                  setDeleteTarget(null);
                  setTypedName('');
                }
              }}
              autoFocus
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
                disabled={typedName !== deleteTarget.server_name || isDeleting}
                className="btn-danger"
              >
                {isDeleting && (
                  <ArrowPathIcon className="h-4 w-4 animate-spin" />
                )}
                Delete
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default VirtualServerList;
