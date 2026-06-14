import React, { useEffect, useState } from 'react';
import axios from 'axios';
import { ClipboardDocumentIcon, CheckIcon } from '@heroicons/react/24/outline';
import DetailsModal from './DetailsModal';
import ResourceBoundTokenButton from './ResourceBoundTokenButton';

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
  const [copied, setCopied] = useState(false);

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

  const handleCopy = async () => {
    try {
      if (onCopy && !useCanonical) {
        await onCopy(dataToShow);
      } else {
        await navigator.clipboard.writeText(JSON.stringify(dataToShow, null, 2));
      }
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch (err) {
      console.error('Failed to copy server JSON:', err);
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
            <h4 className="font-medium text-gray-900 dark:text-white">
              {useCanonical ? 'Canonical server.json:' : 'Server JSON Schema:'}
            </h4>
            <div className="flex items-center gap-4">
              <label
                className="flex items-center gap-2 text-sm text-gray-700 dark:text-gray-300 cursor-pointer select-none"
                title="Show and copy the canonical server.json, compliant with the official Model Context Protocol (MCP) server schema"
              >
                <input
                  type="checkbox"
                  checked={useCanonical}
                  onChange={(e) => handleToggleCanonical(e.target.checked)}
                  className="h-4 w-4 rounded border-gray-300 text-blue-600 focus:ring-blue-500"
                />
                Canonical
              </label>
              <button
                onClick={handleCopy}
                disabled={useCanonical && canonicalLoading}
                className={`flex items-center gap-2 px-3 py-2 text-white rounded-lg transition-colors duration-200 disabled:opacity-50 disabled:cursor-not-allowed ${
                  copied
                    ? 'bg-green-600'
                    : 'bg-blue-600 hover:bg-blue-700'
                }`}
              >
                {copied ? (
                  <>
                    <CheckIcon className="h-4 w-4" />
                    Copied
                  </>
                ) : (
                  <>
                    <ClipboardDocumentIcon className="h-4 w-4" />
                    Copy JSON
                  </>
                )}
              </button>
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

          <pre className="p-4 bg-gray-50 dark:bg-gray-900 border dark:border-gray-700 rounded-lg overflow-x-auto text-xs text-gray-900 dark:text-gray-100 max-h-[30vh] overflow-y-auto">
            {useCanonical && canonicalLoading
              ? 'Loading canonical server.json...'
              : JSON.stringify(dataToShow, null, 2)}
          </pre>
        </div>

        <div className="bg-gray-50 dark:bg-gray-900 border dark:border-gray-700 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 dark:text-white mb-3">Field Reference</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <h5 className="font-medium text-gray-700 dark:text-gray-300 mb-2">Core Fields</h5>
              <ul className="space-y-1 text-gray-600 dark:text-gray-400">
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">name</code> - Server
                  display name
                </li>
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">path</code> - Registry
                  path
                </li>
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">description</code> -
                  Server purpose
                </li>
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">mcp_endpoint</code> -
                  MCP endpoint URL
                </li>
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">status</code> -
                  Lifecycle status (active/deprecated/draft/beta)
                </li>
              </ul>
            </div>
            <div>
              <h5 className="font-medium text-gray-700 dark:text-gray-300 mb-2">Metadata Fields</h5>
              <ul className="space-y-1 text-gray-600 dark:text-gray-400">
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">enabled</code> -
                  Server enabled state
                </li>
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">tags</code> -
                  Categorization tags
                </li>
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">num_tools</code> -
                  Number of tools
                </li>
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">provider</code> -
                  Source registry information
                </li>
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">source_created_at</code>{' '}
                  - Creation timestamp
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </DetailsModal>
  );
};

export default ServerDetailsModal;
