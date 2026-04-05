import React from 'react';
import { ClipboardDocumentIcon } from '@heroicons/react/24/outline';
import DetailsModal from './DetailsModal';

interface AgentLike {
  name: string;
  path: string;
  description?: string;
  version?: string;
  visibility?: string;
  trust_level?: string;
  enabled: boolean;
  tags?: string[];
}

interface AgentDetailsModalProps {
  agent: AgentLike & { [key: string]: any };
  isOpen: boolean;
  onClose: () => void;
  loading: boolean;
  fullDetails?: any;
  onCopy?: (data: any) => Promise<void> | void;
}

/**
 * AgentDetailsModal displays the complete agent JSON schema.
 *
 * Features:
 * - Uses shared DetailsModal component
 * - Copy to clipboard functionality
 * - Field reference documentation
 * - Loading states handled by parent DetailsModal
 */
const getAgentCardUrl = (agentUrl: string): string | null => {
  try {
    const origin = new URL(agentUrl).origin;
    return `${origin}/.well-known/agent-card.json`;
  } catch {
    return null;
  }
};

const AgentDetailsModal: React.FC<AgentDetailsModalProps> = ({
  agent,
  isOpen,
  onClose,
  loading,
  fullDetails,
  onCopy,
}) => {
  const dataToCopy = fullDetails || agent;

  const handleCopy = async () => {
    try {
      if (onCopy) {
        await onCopy(dataToCopy);
      } else {
        await navigator.clipboard.writeText(JSON.stringify(dataToCopy, null, 2));
      }
    } catch (error) {
      console.error('Failed to copy agent JSON:', error);
    }
  };

  return (
    <DetailsModal
      title={`${agent.name} - Full Details (JSON)`}
      isOpen={isOpen}
      onClose={onClose}
      loading={loading}
      maxWidth="4xl"
    >
      <div className="space-y-4">
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-lg p-4">
          <h4 className="font-medium text-blue-900 dark:text-blue-100 mb-2">Complete Agent Schema</h4>
          <p className="text-sm text-blue-800 dark:text-blue-200">
            This is the complete A2A agent definition stored in the registry. It includes all metadata, skills,
            security schemes, and configuration details.
          </p>
        </div>

        {/* A2A Agent Card URL for A2A agents */}
        {fullDetails?.supported_protocol === 'a2a' && fullDetails?.url && (() => {
          const cardUrl = getAgentCardUrl(fullDetails.url);
          return cardUrl ? (
            <div className="bg-cyan-50 dark:bg-cyan-900/20 border border-cyan-200 dark:border-cyan-800 rounded-lg p-3 mt-2">
              <p className="text-sm text-cyan-800 dark:text-cyan-200">
                <span className="font-medium">A2A Agent Card:</span>{' '}
                <a
                  href={cardUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-cyan-600 dark:text-cyan-400 hover:underline break-all"
                >
                  {cardUrl}
                </a>
              </p>
            </div>
          ) : null;
        })()}

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-gray-900 dark:text-white">Agent JSON Schema:</h4>
            <button
              onClick={handleCopy}
              className="flex items-center gap-2 px-3 py-2 bg-blue-600 hover:bg-blue-700 text-white rounded-lg transition-colors duration-200"
            >
              <ClipboardDocumentIcon className="h-4 w-4" />
              Copy JSON
            </button>
          </div>

          <pre className="p-4 bg-gray-50 dark:bg-gray-900 border dark:border-gray-700 rounded-lg overflow-x-auto text-xs text-gray-900 dark:text-gray-100 max-h-[30vh] overflow-y-auto">
            {JSON.stringify(dataToCopy, null, 2)}
          </pre>
        </div>

        <div className="bg-gray-50 dark:bg-gray-900 border dark:border-gray-700 rounded-lg p-4">
          <h4 className="font-medium text-gray-900 dark:text-white mb-3">Field Reference</h4>
          <div className="grid grid-cols-1 md:grid-cols-2 gap-4 text-sm">
            <div>
              <h5 className="font-medium text-gray-700 dark:text-gray-300 mb-2">Core Fields</h5>
              <ul className="space-y-1 text-gray-600 dark:text-gray-400">
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">protocol_version</code> - A2A protocol
                  version
                </li>
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">name</code> - Agent display name
                </li>
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">description</code> - Agent purpose
                </li>
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">url</code> - Agent endpoint URL
                </li>
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">path</code> - Registry path
                </li>
              </ul>
            </div>
            <div>
              <h5 className="font-medium text-gray-700 dark:text-gray-300 mb-2">Metadata Fields</h5>
              <ul className="space-y-1 text-gray-600 dark:text-gray-400">
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">skills</code> - Agent capabilities
                </li>
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">security_schemes</code> - Auth methods
                </li>
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">tags</code> - Categorization
                </li>
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">trust_level</code> - Verification status
                </li>
                <li>
                  <code className="bg-gray-200 dark:bg-gray-700 px-1 rounded">status</code> - Lifecycle status
                </li>
              </ul>
            </div>
          </div>
        </div>
      </div>
    </DetailsModal>
  );
};

export default AgentDetailsModal;
