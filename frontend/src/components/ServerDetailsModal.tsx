import React, { useEffect, useState } from 'react';
import axios from 'axios';
import DetailsModal from './DetailsModal';
import ResourceBoundTokenButton from './ResourceBoundTokenButton';
import { CopyButton, FieldReferenceGrid } from './modals';
import { CHECKBOX } from './formFields';

interface ServerDetailsModalProps {
  server: any;
  isOpen: boolean;
  onClose: () => void;
  loading?: boolean;
  error?: string | null;
  fullDetails?: any;
  onCopy?: (data: any) => Promise<void> | void;
  authToken?: string | null;
}

/**
 * ServerDetailsModal displays the complete server JSON schema.
 *
 * Features:
 * - Uses shared DetailsModal component
 * - Copy to clipboard functionality
 * - Field reference documentation
 * - Loading and error states
 */
const ServerDetailsModal: React.FC<ServerDetailsModalProps> = ({
  server,
  isOpen,
  onClose,
  loading = false,
  error = null,
  fullDetails,
  onCopy,
  authToken,
}) => {
  const storedDetails = fullDetails || server;

  // When checked, show and copy the canonical server.json projection
  // (GET /api/servers/{path}/server.json) instead of the stored document.
  const [useCanonical, setUseCanonical] = useState(false);
  const [canonicalDetails, setCanonicalDetails] = useState<any>(null);
  const [canonicalLoading, setCanonicalLoading] = useState(false);
  const [canonicalError, setCanonicalError] = useState<string | null>(null);

  // The doc shown in the preview and placed on the clipboard. Falls back to the
  // stored document until the canonical fetch completes.
  const dataToShow = useCanonical && canonicalDetails ? canonicalDetails : storedDetails;

  // Reset canonical state when the modal switches to a different server, so the
  // previous server's canonical doc and checkbox state never leak across opens.
  useEffect(() => {
    setUseCanonical(false);
    setCanonicalDetails(null);
    setCanonicalError(null);
    setCanonicalLoading(false);
  }, [server?.path]);

  const _fetchCanonical = async () => {
    if (canonicalDetails || !server?.path) {
      return;
    }
    setCanonicalLoading(true);
    setCanonicalError(null);
    try {
      const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
      const response = await axios.get(
        `/api/servers${server.path}/server.json`,
        headers ? { headers } : undefined
      );
      setCanonicalDetails(response.data);
    } catch (err) {
      console.error('Failed to fetch canonical server.json:', err);
      setCanonicalError('Could not load canonical server.json.');
    } finally {
      setCanonicalLoading(false);
    }
  };

  const handleToggleCanonical = async (checked: boolean) => {
    setUseCanonical(checked);
    if (checked) {
      await _fetchCanonical();
    }
  };

  return (
    <DetailsModal
      title={`${server?.name || 'Server'} - Full Details (JSON)`}
      isOpen={isOpen}
      onClose={onClose}
      loading={loading}
      error={error}
      maxWidth="4xl"
    >
      <div className="space-y-4">
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">
            Complete Server Schema
          </h4>
          <p className="text-sm text-blue-800 dark:text-blue-200">
            This is the complete MCP server definition stored in the registry. It includes all
            metadata, tools, authentication configuration, and runtime details.
          </p>
        </div>

        {server?.path && (
          <ResourceBoundTokenButton
            resourceType="server"
            resourceId={server.path}
            resourceName={server?.name}
          />
        )}

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-fg">
              {useCanonical ? 'Canonical server.json:' : 'Server JSON Schema:'}
            </h4>
            <div className="flex items-center gap-4">
              <label
                className="flex items-center gap-2 text-sm text-fg-muted cursor-pointer select-none"
                title="Show and copy the canonical server.json, compliant with the official Model Context Protocol (MCP) server schema"
              >
                <input
                  type="checkbox"
                  checked={useCanonical}
                  onChange={(e) => handleToggleCanonical(e.target.checked)}
                  className={CHECKBOX}
                />
                Canonical
              </label>
              <CopyButton
                getText={() => JSON.stringify(dataToShow, null, 2)}
                onCopy={onCopy && !useCanonical ? () => onCopy(dataToShow) : undefined}
                disabled={useCanonical && canonicalLoading}
                label="Copy JSON"
              />
            </div>
          </div>

          {useCanonical && !canonicalError && (
            <p className="text-sm text-gray-500 dark:text-gray-400">
              Canonical server.json, compliant with the official Model Context Protocol (MCP) server
              schema.
            </p>
          )}

          {useCanonical && canonicalError && (
            <p className="text-sm text-red-600 dark:text-red-400">{canonicalError}</p>
          )}

          <pre className="p-4 bg-surface-muted border border-line rounded-lg overflow-x-auto text-xs text-fg max-h-[30vh] overflow-y-auto">
            {useCanonical && canonicalLoading
              ? 'Loading canonical server.json...'
              : JSON.stringify(dataToShow, null, 2)}
          </pre>
        </div>

        <FieldReferenceGrid
          columns={[
            {
              heading: 'Core Fields',
              fields: [
                { name: 'name', description: 'Server display name' },
                { name: 'path', description: 'Registry path' },
                { name: 'description', description: 'Server purpose' },
                { name: 'mcp_endpoint', description: 'MCP endpoint URL' },
                { name: 'status', description: 'Lifecycle status (active/deprecated/draft/beta)' },
              ],
            },
            {
              heading: 'Metadata Fields',
              fields: [
                { name: 'enabled', description: 'Server enabled state' },
                { name: 'tags', description: 'Categorization tags' },
                { name: 'num_tools', description: 'Number of tools' },
                { name: 'provider', description: 'Source registry information' },
                { name: 'source_created_at', description: 'Creation timestamp' },
              ],
            },
          ]}
        />
      </div>
    </DetailsModal>
  );
};

export default ServerDetailsModal;
