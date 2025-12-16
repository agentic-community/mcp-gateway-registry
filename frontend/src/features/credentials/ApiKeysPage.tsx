import { useState } from 'react';
import {
  KeyIcon,
  PlusIcon,
  NoSymbolIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';
import { PageContent, PageHeader } from '@/components/layout/index';
import { Button, Badge, Card, Spinner, EmptyState } from '@/components/ui';
import { useToast } from '@/components/ui/Toast';
import { useEnforceAIAgents } from '@/features/enforceai-agents/hooks';
import { useApiKeys } from './hooks';
import { CreateApiKeyModal } from './CreateApiKeyModal';
import { RevokeApiKeyModal } from './RevokeApiKeyModal';
import { formatRelativeTime } from '@/lib/format';
import { cn } from '@/lib/cn';
import type { EnforceAIAgent, ApiKeySummary } from '@/api/types';

// ============================================================================
// API Key Card Component
// ============================================================================

interface ApiKeyCardProps {
  apiKey: ApiKeySummary;
  onRevoke: () => void;
}

function ApiKeyCard({ apiKey, onRevoke }: ApiKeyCardProps) {
  const isRevoked = apiKey.revoked_at !== null;
  const isExpired = apiKey.expires_at && new Date(apiKey.expires_at) < new Date();

  return (
    <Card className={cn((isRevoked || isExpired) && 'opacity-60')}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center space-x-3">
          <div
            className={cn(
              'flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center',
              isRevoked
                ? 'bg-gray-100 dark:bg-gray-700'
                : 'bg-amber-100 dark:bg-amber-900/30'
            )}
          >
            <KeyIcon
              className={cn(
                'w-5 h-5',
                isRevoked
                  ? 'text-gray-400 dark:text-gray-500'
                  : 'text-amber-600 dark:text-amber-400'
              )}
            />
          </div>
          <div>
            <p className="text-sm font-mono text-gray-900 dark:text-gray-100">
              {apiKey.key_id.slice(0, 16)}...
            </p>
            <p className="text-xs text-gray-500 dark:text-gray-400">
              Created {formatRelativeTime(apiKey.created_at)}
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          {isRevoked ? (
            <Badge variant="error">Revoked</Badge>
          ) : isExpired ? (
            <Badge variant="warning">Expired</Badge>
          ) : (
            <Badge variant="success">Active</Badge>
          )}
        </div>
      </div>

      {/* Scopes */}
      {apiKey.scopes && apiKey.scopes.length > 0 && (
        <div className="mb-3">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
            Scopes ({apiKey.scopes.length})
          </p>
          <div className="flex flex-wrap gap-1">
            {apiKey.scopes.slice(0, 3).map((scope) => (
              <Badge key={scope} variant="info" size="sm">
                {scope}
              </Badge>
            ))}
            {apiKey.scopes.length > 3 && (
              <Badge variant="neutral" size="sm">
                +{apiKey.scopes.length - 3} more
              </Badge>
            )}
          </div>
        </div>
      )}

      {/* Metadata */}
      <div className="text-xs text-gray-500 dark:text-gray-400 space-y-1 mb-3">
        {apiKey.expires_at && (
          <p>
            Expires: {formatRelativeTime(apiKey.expires_at)}
          </p>
        )}
        {apiKey.last_used_at && (
          <p>
            Last used: {formatRelativeTime(apiKey.last_used_at)}
          </p>
        )}
        {isRevoked && apiKey.revoked_at && (
          <p>
            Revoked: {formatRelativeTime(apiKey.revoked_at)}
          </p>
        )}
      </div>

      {/* Actions */}
      {!isRevoked && (
        <div className="flex flex-wrap gap-2 border-t border-gray-200 dark:border-gray-700 pt-3">
          <Button
            variant="danger"
            size="sm"
            onClick={onRevoke}
            leftIcon={<NoSymbolIcon className="w-4 h-4" />}
          >
            Revoke
          </Button>
        </div>
      )}
    </Card>
  );
}

// ============================================================================
// Main Page Component
// ============================================================================

export default function ApiKeysPage() {
  const { addToast } = useToast();
  const { agents, isLoading: agentsLoading } = useEnforceAIAgents();
  const [selectedAgent, setSelectedAgent] = useState<EnforceAIAgent | null>(null);
  const { apiKeys, isLoading: keysLoading, refetch } = useApiKeys(selectedAgent?.agent_id || '');

  // Modal state
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [revokingKey, setRevokingKey] = useState<ApiKeySummary | null>(null);

  // Filter to active (non-revoked) agents
  const activeAgents = agents.filter((a) => !a.revoked_at);

  // Handlers
  const handleCreateSuccess = () => {
    setIsCreateModalOpen(false);
    refetch();
    addToast({
      type: 'success',
      title: 'API key created',
      message: 'New API key has been created successfully',
    });
  };

  const handleRevokeSuccess = () => {
    setRevokingKey(null);
    refetch();
    addToast({
      type: 'success',
      title: 'API key revoked',
      message: 'API key has been permanently revoked',
    });
  };

  // Loading state for agents
  if (agentsLoading) {
    return (
      <PageContent>
        <PageHeader
          title="API Keys"
          description="Create and manage API keys for agent authentication"
          breadcrumbs={[
            { name: 'Credentials', href: '/credentials' },
            { name: 'API Keys' },
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
          title="API Keys"
          description="Create and manage API keys for agent authentication"
          breadcrumbs={[
            { name: 'Credentials', href: '/credentials' },
            { name: 'API Keys' },
          ]}
        />
        <EmptyState
          icon={<KeyIcon className="w-12 h-12" />}
          title="No agents available"
          description="You need to create an EnforceAI agent before you can create API keys."
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
        title="API Keys"
        description="Create and manage API keys for agent authentication"
        breadcrumbs={[
          { name: 'Credentials', href: '/credentials' },
          { name: 'API Keys' },
        ]}
        actions={
          selectedAgent && (
            <Button
              variant="primary"
              leftIcon={<PlusIcon className="w-4 h-4" />}
              onClick={() => setIsCreateModalOpen(true)}
            >
              Create API Key
            </Button>
          )
        }
      />

      {/* Agent selector */}
      <div className="mb-6">
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
          Select Agent
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

      {/* Content based on selection */}
      {!selectedAgent ? (
        <div className="bg-blue-50 dark:bg-blue-900/20 border border-blue-200 dark:border-blue-800 rounded-md p-4">
          <div className="flex">
            <ExclamationTriangleIcon className="w-5 h-5 text-blue-600 dark:text-blue-400 flex-shrink-0" />
            <div className="ml-3">
              <p className="text-sm text-blue-700 dark:text-blue-400">
                Select an agent above to view and manage its API keys.
              </p>
            </div>
          </div>
        </div>
      ) : keysLoading ? (
        <div className="flex items-center justify-center py-12">
          <Spinner size="lg" />
        </div>
      ) : apiKeys.length === 0 ? (
        <EmptyState
          icon={<KeyIcon className="w-8 h-8" />}
          title="No API keys"
          description={`No API keys have been created for ${selectedAgent.alias || 'this agent'} yet.`}
          action={
            <Button
              variant="primary"
              leftIcon={<PlusIcon className="w-4 h-4" />}
              onClick={() => setIsCreateModalOpen(true)}
            >
              Create API Key
            </Button>
          }
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {apiKeys.map((apiKey) => (
            <ApiKeyCard
              key={apiKey.key_id}
              apiKey={apiKey}
              onRevoke={() => setRevokingKey(apiKey)}
            />
          ))}
        </div>
      )}

      {/* Modals */}
      {isCreateModalOpen && selectedAgent && (
        <CreateApiKeyModal
          open={isCreateModalOpen}
          agent={selectedAgent}
          onClose={() => setIsCreateModalOpen(false)}
          onSuccess={handleCreateSuccess}
        />
      )}

      {revokingKey && (
        <RevokeApiKeyModal
          open={!!revokingKey}
          apiKey={revokingKey}
          onClose={() => setRevokingKey(null)}
          onSuccess={handleRevokeSuccess}
        />
      )}
    </PageContent>
  );
}
