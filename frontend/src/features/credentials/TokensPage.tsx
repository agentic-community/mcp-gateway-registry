import { useState } from 'react';
import {
  TicketIcon,
  PlusIcon,
  NoSymbolIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';
import { PageContent, PageHeader } from '@/components/layout/index';
import { Button, Spinner, EmptyState } from '@/components/ui';
import { useToast } from '@/components/ui/Toast';
import { useEnforceAIAgents } from '@/features/enforceai-agents/hooks';
import { MintTokenModal } from './MintTokenModal';
import { RevokeTokenModal } from './RevokeTokenModal';
import type { EnforceAIAgent } from '@/api/types';

// ============================================================================
// Main Page Component
// ============================================================================

export default function TokensPage() {
  const { addToast } = useToast();
  const { agents, isLoading: agentsLoading } = useEnforceAIAgents();
  const [selectedAgent, setSelectedAgent] = useState<EnforceAIAgent | null>(null);

  // Modal state
  const [isMintModalOpen, setIsMintModalOpen] = useState(false);
  const [isRevokeModalOpen, setIsRevokeModalOpen] = useState(false);

  // Filter to active (non-revoked) agents
  const activeAgents = agents.filter((a) => !a.revoked_at);

  // Handlers
  const handleMintSuccess = () => {
    setIsMintModalOpen(false);
    addToast({
      type: 'success',
      title: 'Token minted',
      message: 'Gateway token has been minted successfully',
    });
  };

  const handleRevokeSuccess = () => {
    setIsRevokeModalOpen(false);
    addToast({
      type: 'success',
      title: 'Token revoked',
      message: 'Gateway token has been revoked',
    });
  };

  // Loading state for agents
  if (agentsLoading) {
    return (
      <PageContent>
        <PageHeader
          title="Gateway Tokens"
          description="Mint and manage gateway tokens"
          breadcrumbs={[
            { name: 'Credentials', href: '/credentials' },
            { name: 'Gateway Tokens' },
          ]}
        />
        <div className="flex items-center justify-center py-12">
          <Spinner size="lg" />
        </div>
      </PageContent>
    );
  }

  // No agents state
  if (activeAgents.length === 0) {
    return (
      <PageContent>
        <PageHeader
          title="Gateway Tokens"
          description="Mint and manage gateway tokens"
          breadcrumbs={[
            { name: 'Credentials', href: '/credentials' },
            { name: 'Gateway Tokens' },
          ]}
        />
        <EmptyState
          icon={<TicketIcon className="w-12 h-12" />}
          title="No agents available"
          description="You need to create an EnforceAI agent before you can mint gateway tokens."
          action={
            <Button
              variant="primary"
              onClick={() => window.location.href = '/agents'}
            >
              Go to Agents
            </Button>
          }
        />
      </PageContent>
    );
  }

  return (
    <PageContent>
      <PageHeader
        title="Gateway Tokens"
        description="Mint and manage gateway tokens"
        breadcrumbs={[
          { name: 'Credentials', href: '/credentials' },
          { name: 'Gateway Tokens' },
        ]}
        actions={
          <div className="flex space-x-2">
            <Button
              variant="secondary"
              leftIcon={<NoSymbolIcon className="w-4 h-4" />}
              onClick={() => setIsRevokeModalOpen(true)}
            >
              Revoke Token
            </Button>
            {selectedAgent && (
              <Button
                variant="primary"
                leftIcon={<PlusIcon className="w-4 h-4" />}
                onClick={() => setIsMintModalOpen(true)}
              >
                Mint Token
              </Button>
            )}
          </div>
        }
      />

      {/* Agent selector */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Select Agent for Minting
        </label>
        <select
          value={selectedAgent?.agent_id || ''}
          onChange={(e) => {
            const agent = activeAgents.find((a) => a.agent_id === e.target.value);
            setSelectedAgent(agent || null);
          }}
          className="block w-full max-w-md rounded-md border-gray-300 dark:border-gray-600 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm dark:bg-gray-700 dark:text-gray-100"
        >
          <option value="">Select an agent...</option>
          {activeAgents.map((agent) => (
            <option key={agent.agent_id} value={agent.agent_id}>
              {agent.alias || agent.agent_id.slice(0, 8)} ({agent.scopes.length} scopes)
            </option>
          ))}
        </select>
      </div>

      {/* Info panel */}
      <div className="space-y-4">
        {!selectedAgent ? (
          <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-4">
            <div className="flex">
              <ExclamationTriangleIcon className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0" />
              <div className="ml-3">
                <p className="text-sm text-blue-700 dark:text-blue-400">
                  Select an agent above to mint a new gateway token.
                </p>
              </div>
            </div>
          </div>
        ) : (
          <EmptyState
            icon={<TicketIcon className="w-8 h-8" />}
            title="Ready to mint"
            description={`Selected agent: ${selectedAgent.alias || selectedAgent.agent_id.slice(0, 8)}. Gateway tokens are stateless and not stored - only the revocation status is tracked.`}
            action={
              <Button
                variant="primary"
                leftIcon={<PlusIcon className="w-4 h-4" />}
                onClick={() => setIsMintModalOpen(true)}
              >
                Mint Token
              </Button>
            }
          />
        )}

        {/* Token information cards */}
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4 mt-6">
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-2">
              About Gateway Tokens
            </h3>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1 list-disc list-inside">
              <li>JWT tokens signed by the gateway</li>
              <li>Contain agent_id, scopes, and expiration</li>
              <li>Stateless - no server storage required</li>
              <li>Can be revoked by JTI or bulk-revoked via agent</li>
            </ul>
          </div>

          <div className="bg-white dark:bg-gray-800 rounded-lg shadow p-4">
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-2">
              Token Revocation
            </h3>
            <ul className="text-sm text-gray-600 dark:text-gray-400 space-y-1 list-disc list-inside">
              <li>Revoke by JTI: invalidate a specific token</li>
              <li>Revoke by token: paste the full JWT to revoke</li>
              <li>Bulk revoke: use agent's "Revoke All Tokens" action</li>
              <li>Revocation is immediate and permanent</li>
            </ul>
          </div>
        </div>
      </div>

      {/* Modals */}
      {isMintModalOpen && selectedAgent && (
        <MintTokenModal
          open={isMintModalOpen}
          agent={selectedAgent}
          onClose={() => setIsMintModalOpen(false)}
          onSuccess={handleMintSuccess}
        />
      )}

      {isRevokeModalOpen && (
        <RevokeTokenModal
          open={isRevokeModalOpen}
          onClose={() => setIsRevokeModalOpen(false)}
          onSuccess={handleRevokeSuccess}
          defaultAgentId={selectedAgent?.agent_id}
        />
      )}
    </PageContent>
  );
}
