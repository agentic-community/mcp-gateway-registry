import React from 'react';
import DetailsModal from './DetailsModal';
import ResourceBoundTokenButton from './ResourceBoundTokenButton';
import { SafeLink } from './SafeLink';
import { CopyButton, FieldReferenceGrid } from './modals';

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
    // The agent card is served relative to the agent's advertised url, which in
    // reverse-proxy mode includes the gateway path (e.g.
    // https://host/agent/flight-booking-agent/). Append the well-known suffix to
    // that full base, do NOT collapse to the origin (that would drop the
    // /agent/<path>/ prefix and 404).
    const parsed = new URL(agentUrl);
    const basePath = parsed.pathname.endsWith('/')
      ? parsed.pathname
      : `${parsed.pathname}/`;
    return `${parsed.origin}${basePath}.well-known/agent-card.json`;
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

  return (
    <DetailsModal
      title={`${agent.name} - Full Details (JSON)`}
      isOpen={isOpen}
      onClose={onClose}
      loading={loading}
      maxWidth="4xl"
    >
      <div className="space-y-4">
        <div className="bg-info-50 dark:bg-info-900/20 border border-info-200 dark:border-info-800 rounded-lg p-4">
          <h4 className="font-medium text-info-900 dark:text-info-100 mb-2">Complete Agent Schema</h4>
          <p className="text-sm text-info-800 dark:text-info-200">
            This is the complete A2A agent definition stored in the registry. It includes all metadata, skills,
            security schemes, and configuration details.
          </p>
        </div>

        {agent?.path && (
          <ResourceBoundTokenButton
            resourceType="agent"
            resourceId={agent.path}
            resourceName={agent.name}
          />
        )}

        {/* A2A Agent Card URL for A2A agents */}
        {fullDetails?.supported_protocol === 'a2a' && fullDetails?.url && (() => {
          const cardUrl = getAgentCardUrl(fullDetails.url);
          return cardUrl ? (
            <div className="bg-agent-50 dark:bg-agent-900/20 border border-agent-200 dark:border-agent-800 rounded-lg p-3 mt-2">
              <p className="text-sm text-agent-800 dark:text-agent-200">
                <span className="font-medium">A2A Agent Card:</span>{' '}
                <SafeLink
                  href={cardUrl}
                  target="_blank"
                  rel="noopener noreferrer"
                  className="text-agent-600 dark:text-agent-400 hover:underline break-all"
                >
                  {cardUrl}
                </SafeLink>
              </p>
            </div>
          ) : null;
        })()}

        <div className="space-y-2">
          <div className="flex items-center justify-between">
            <h4 className="font-medium text-fg">Agent JSON Schema:</h4>
            <CopyButton
              getText={() => JSON.stringify(dataToCopy, null, 2)}
              onCopy={onCopy ? () => onCopy(dataToCopy) : undefined}
              label="Copy JSON"
            />
          </div>

          <pre className="p-4 bg-surface-muted border border-line rounded-lg overflow-x-auto text-xs text-fg max-h-[30vh] overflow-y-auto">
            {JSON.stringify(dataToCopy, null, 2)}
          </pre>
        </div>

        <FieldReferenceGrid
          columns={[
            {
              heading: 'Core Fields',
              fields: [
                { name: 'protocol_version', description: 'A2A protocol version' },
                { name: 'name', description: 'Agent display name' },
                { name: 'description', description: 'Agent purpose' },
                { name: 'url', description: 'Agent endpoint URL' },
                { name: 'path', description: 'Registry path' },
              ],
            },
            {
              heading: 'Metadata Fields',
              fields: [
                { name: 'skills', description: 'Agent capabilities' },
                { name: 'security_schemes', description: 'Auth methods' },
                { name: 'tags', description: 'Categorization' },
                { name: 'trust_level', description: 'Verification status' },
                { name: 'status', description: 'Lifecycle status' },
              ],
            },
          ]}
        />
      </div>
    </DetailsModal>
  );
};

export default AgentDetailsModal;
