import React from 'react';
import { PlusIcon } from '@heroicons/react/24/outline';
import VirtualServerCard from '../../VirtualServerCard';
import type { VirtualServerInfo } from '../../../types/virtualServer';
import EntityGrid from '../EntityGrid';
import EmptyState from '../EmptyState';

interface VirtualServersSectionProps {
  /** Filtered virtual servers to render. */
  servers: VirtualServerInfo[];
  loading: boolean;
  error: string | null;
  /** True when a search term or non-default lifecycle filter is active. */
  isFiltered: boolean;
  canModify: boolean;
  authToken?: string | null;
  onAdd: () => void;
  onToggle: (path: string, enabled: boolean) => void;
  onEdit: (server: VirtualServerInfo) => void;
  onDelete: (path: string) => void;
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
}

/**
 * The "Virtual MCP Servers" Dashboard collection: header + add button and a
 * teal-accented grid of VirtualServerCards with shared empty/error states.
 */
const VirtualServersSection: React.FC<VirtualServersSectionProps> = ({
  servers,
  loading,
  error,
  isFiltered,
  canModify,
  authToken,
  onAdd,
  onToggle,
  onEdit,
  onDelete,
  onShowToast,
}) => {
  return (
    <div className="mb-8">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-xl font-bold text-fg">
          Virtual MCP Servers
        </h2>
        {canModify && (
          <button
            onClick={onAdd}
            className="btn-primary"
          >
            <PlusIcon className="h-4 w-4" />
            Add Virtual Server
          </button>
        )}
      </div>

      {error ? (
        <EmptyState tone="error" title="Failed to load virtual servers" subtitle={error} />
      ) : loading ? (
        <div className="flex items-center justify-center py-12">
          <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-virtual-600"></div>
        </div>
      ) : servers.length === 0 ? (
        <EmptyState
          tone="teal"
          title="No virtual servers found"
          subtitle={
            isFiltered
              ? 'Try adjusting your search or filter'
              : 'No virtual servers are configured yet'
          }
        />
      ) : (
        <EntityGrid>
          {servers.map((vs) => (
            <VirtualServerCard
              key={vs.path}
              virtualServer={vs}
              canModify={canModify}
              onToggle={onToggle}
              onEdit={onEdit}
              onDelete={onDelete}
              onShowToast={onShowToast}
              authToken={authToken}
            />
          ))}
        </EntityGrid>
      )}
    </div>
  );
};

export default VirtualServersSection;
