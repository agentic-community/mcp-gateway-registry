import React, { useState } from 'react';
import {
  PlusIcon,
  ArrowPathIcon,
  ExclamationCircleIcon,
} from '@heroicons/react/24/outline';
import { useOpenApiServers, parseOpenApiError } from '../hooks/useOpenApiServers';
import {
  OpenApiServer,
  CreateOpenApiServerRequest,
  UpdateOpenApiServerRequest,
} from '../types/openApiServer';
import OpenApiServerCard, { openApiToolCount } from './OpenApiServerCard';
import OpenApiServerForm from './OpenApiServerForm';
import OpenApiServerDetailsModal from './OpenApiServerDetailsModal';
import ConfirmModal from './ConfirmModal';


interface OpenApiServerListProps {
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
}


/**
 * OpenApiServerList is the admin OpenAPI view on the dashboard. It lists every
 * OpenAPI MCP server as a card and owns the create, edit, details and delete
 * dialogs. The Dashboard renders it only for admins with the feature on.
 */
const OpenApiServerList: React.FC<OpenApiServerListProps> = ({ onShowToast }) => {
  const {
    servers,
    loading,
    error,
    refreshData,
    createServer,
    updateServer,
    deleteServer,
  } = useOpenApiServers();

  // undefined: form closed. null: create. A server: edit.
  const [formServer, setFormServer] = useState<OpenApiServer | null | undefined>(undefined);
  const [detailsSlug, setDetailsSlug] = useState<string | null>(null);
  const [deleteCandidate, setDeleteCandidate] = useState<OpenApiServer | null>(null);
  const [isDeleting, setIsDeleting] = useState(false);

  // The form shows API errors itself, so this lets a failed save throw back to it.
  const handleSave = async (
    data: CreateOpenApiServerRequest | UpdateOpenApiServerRequest,
  ) => {
    if (formServer) {
      await updateServer(formServer.slug, data as UpdateOpenApiServerRequest);
      onShowToast(`Server "${formServer.name}" updated`, 'success');
      setFormServer(undefined);
      return;
    }
    const created = await createServer(data as CreateOpenApiServerRequest);
    onShowToast(`Server "${created.name}" created at ${created.path}`, 'success');
    setFormServer(undefined);
    setDetailsSlug(created.slug);
  };

  const handleDelete = async () => {
    if (!deleteCandidate) return;
    setIsDeleting(true);
    try {
      await deleteServer(deleteCandidate.slug);
      onShowToast(`Server "${deleteCandidate.name}" deleted`, 'success');
      setDeleteCandidate(null);
    } catch (err: unknown) {
      onShowToast(parseOpenApiError(err, 'Failed to delete the server').message, 'error');
    } finally {
      setIsDeleting(false);
    }
  };

  const deleteMessage = deleteCandidate
    ? `Delete "${deleteCandidate.name}"? Its path ${deleteCandidate.path} and all ` +
      `${openApiToolCount(deleteCandidate)} of its tools from ` +
      `${deleteCandidate.targets?.length || 0} APIs are removed. Agents using it lose access at once.`
    : '';

  if (error && !loading) {
    return (
      <div className="text-center py-12">
        <ExclamationCircleIcon className="h-12 w-12 mx-auto text-danger-500 mb-4" />
        <h3 className="text-lg font-medium text-fg mb-2">Failed to load OpenAPI MCP servers</h3>
        <p className="text-fg-subtle mb-4">{error}</p>
        <button onClick={refreshData} className="btn-primary">Retry</button>
      </div>
    );
  }

  return (
    <div className="space-y-4" data-testid="openapi-server-list">
      <div className="flex items-center justify-between">
        <div>
          <h2 className="text-lg font-semibold text-fg">OpenAPI MCP Servers</h2>
          <p className="text-sm text-fg-subtle">
            MCP servers the registry builds from REST APIs described by OpenAPI specs
          </p>
        </div>
        <div className="flex items-center gap-2">
          <button onClick={refreshData} className="btn-icon" title="Refresh" aria-label="Refresh">
            <ArrowPathIcon className={`h-5 w-5 ${loading ? 'animate-spin' : ''}`} />
          </button>
          <button onClick={() => setFormServer(null)} className="btn-primary flex items-center gap-2">
            <PlusIcon className="h-5 w-5" />
            New OpenAPI server
          </button>
        </div>
      </div>

      {loading && servers.length === 0 ? (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {[1, 2, 3].map((i) => (
            <div key={i} className="h-48 bg-surface-hover rounded-2xl animate-pulse" />
          ))}
        </div>
      ) : servers.length === 0 ? (
        <div className="text-center py-12 bg-surface-muted rounded-lg">
          <h3 className="text-lg font-medium text-fg mb-2">No OpenAPI MCP servers yet</h3>
          <p className="text-fg-subtle">
            Create a server, then add REST APIs to it. Their operations become tools.
          </p>
        </div>
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 xl:grid-cols-3 gap-4">
          {servers.map((server) => (
            <OpenApiServerCard
              key={server.slug}
              server={server}
              onManage={(s) => setDetailsSlug(s.slug)}
              onEdit={(s) => setFormServer(s)}
              onDelete={(s) => setDeleteCandidate(s)}
            />
          ))}
        </div>
      )}

      {formServer !== undefined && (
        <OpenApiServerForm
          server={formServer}
          onSave={handleSave}
          onCancel={() => setFormServer(undefined)}
        />
      )}

      {detailsSlug && (
        <OpenApiServerDetailsModal
          slug={detailsSlug}
          isOpen={!!detailsSlug}
          onClose={() => setDetailsSlug(null)}
          onShowToast={onShowToast}
          onChanged={refreshData}
        />
      )}

      <ConfirmModal
        isOpen={!!deleteCandidate}
        onClose={() => setDeleteCandidate(null)}
        onConfirm={handleDelete}
        title="Delete OpenAPI MCP server"
        message={deleteMessage}
        confirmLabel="Delete"
        loadingLabel="Deleting..."
        isDestructive
        isLoading={isDeleting}
      />
    </div>
  );
};


export default OpenApiServerList;
