import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ShieldCheckIcon,
  PlusIcon,
  PencilIcon,
  EyeIcon,
  NoSymbolIcon,
  KeyIcon,
  ExclamationTriangleIcon,
  ClipboardDocumentIcon,
} from '@heroicons/react/24/outline';
import { PageContent, PageHeader } from '@/components/layout/index';
import { Button, Badge, Card, Spinner, EmptyState, Input } from '@/components/ui';
import { useToast } from '@/components/ui/Toast';
import {
  useEnforceAIAgents,
  filterEnforceAIAgents,
  type EnforceAIAgentFilter,
  type AgentStatusFilter,
} from './hooks';
import { CreateAgentModal } from './CreateAgentModal';
import { EditAgentModal } from './EditAgentModal';
import { RevokeAgentModal } from './RevokeAgentModal';
import { RevokeAllTokensModal } from './RevokeAllTokensModal';
import { formatRelativeTime } from '@/lib/format';
import { cn } from '@/lib/cn';
import type { EnforceAIAgent } from '@/api/types';

// ============================================================================
// Agent Card Component
// ============================================================================

interface AgentCardProps {
  agent: EnforceAIAgent;
  onView: () => void;
  onEdit: () => void;
  onRevoke: () => void;
  onRevokeTokens: () => void;
  onCopyId: () => void;
}

function AgentCard({
  agent,
  onView,
  onEdit,
  onRevoke,
  onRevokeTokens,
  onCopyId,
}: AgentCardProps) {
  const isRevoked = agent.revoked_at !== null;
  const hasEmptyScopes = agent.scopes.length === 0;
  const hasAllowedTools = agent.allowed_tools && agent.allowed_tools.length > 0;

  return (
    <Card className={cn(isRevoked && 'opacity-60')}>
      <div className="flex items-start justify-between mb-3">
        <div className="flex items-center space-x-3">
          <div
            className={cn(
              'flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center',
              isRevoked
                ? 'bg-gray-100 dark:bg-gray-700'
                : 'bg-indigo-100 dark:bg-indigo-900/30'
            )}
          >
            <ShieldCheckIcon
              className={cn(
                'w-5 h-5',
                isRevoked
                  ? 'text-gray-400 dark:text-gray-500'
                  : 'text-indigo-600 dark:text-indigo-400'
              )}
            />
          </div>
          <div>
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
              {agent.alias || 'Unnamed Agent'}
            </h3>
            <p className="text-xs text-gray-500 dark:text-gray-400 font-mono">
              {agent.agent_id.slice(0, 8)}...
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant={isRevoked ? 'error' : 'success'}>
            {isRevoked ? 'Revoked' : 'Active'}
          </Badge>
        </div>
      </div>

      {/* Warnings */}
      {(hasEmptyScopes || (isRevoked === false && !hasAllowedTools)) && (
        <div className="mb-3 space-y-1">
          {hasEmptyScopes && (
            <div className="flex items-center space-x-1 text-yellow-600 dark:text-yellow-400">
              <ExclamationTriangleIcon className="w-4 h-4" />
              <span className="text-xs">No scopes defined</span>
            </div>
          )}
        </div>
      )}

      {/* Scopes */}
      <div className="mb-3">
        <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
          Scopes ({agent.scopes.length})
        </p>
        {agent.scopes.length > 0 ? (
          <div className="flex flex-wrap gap-1">
            {agent.scopes.slice(0, 3).map((scope) => (
              <Badge key={scope} variant="info" size="sm">
                {scope}
              </Badge>
            ))}
            {agent.scopes.length > 3 && (
              <Badge variant="neutral" size="sm">
                +{agent.scopes.length - 3} more
              </Badge>
            )}
          </div>
        ) : (
          <span className="text-xs text-gray-400 dark:text-gray-500">None</span>
        )}
      </div>

      {/* Allowed Tools */}
      {hasAllowedTools && (
        <div className="mb-3">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-1">
            Allowed Tools ({agent.allowed_tools!.length})
          </p>
          <div className="flex flex-wrap gap-1">
            {agent.allowed_tools!.slice(0, 3).map((tool) => (
              <Badge key={tool} variant="neutral" size="sm">
                {tool}
              </Badge>
            ))}
            {agent.allowed_tools!.length > 3 && (
              <Badge variant="neutral" size="sm">
                +{agent.allowed_tools!.length - 3} more
              </Badge>
            )}
          </div>
        </div>
      )}

      {/* Timestamps */}
      <div className="text-xs text-gray-500 dark:text-gray-400 mb-4">
        {isRevoked ? (
          <span>Revoked {formatRelativeTime(agent.revoked_at!)}</span>
        ) : (
          <span>Updated {formatRelativeTime(agent.updated_at)}</span>
        )}
      </div>

      {/* Actions */}
      <div className="flex flex-wrap gap-2 border-t border-gray-200 dark:border-gray-700 pt-3">
        <Button
          variant="ghost"
          size="sm"
          onClick={onView}
          leftIcon={<EyeIcon className="w-4 h-4" />}
        >
          View
        </Button>
        <Button
          variant="ghost"
          size="sm"
          onClick={onCopyId}
          leftIcon={<ClipboardDocumentIcon className="w-4 h-4" />}
        >
          Copy ID
        </Button>
        {!isRevoked && (
          <>
            <Button
              variant="ghost"
              size="sm"
              onClick={onEdit}
              leftIcon={<PencilIcon className="w-4 h-4" />}
            >
              Edit
            </Button>
            <Button
              variant="ghost"
              size="sm"
              onClick={onRevokeTokens}
              leftIcon={<KeyIcon className="w-4 h-4" />}
            >
              Revoke Tokens
            </Button>
            <Button
              variant="danger"
              size="sm"
              onClick={onRevoke}
              leftIcon={<NoSymbolIcon className="w-4 h-4" />}
            >
              Revoke
            </Button>
          </>
        )}
      </div>
    </Card>
  );
}

// ============================================================================
// Filter Bar Component
// ============================================================================

interface FilterBarProps {
  filter: EnforceAIAgentFilter;
  onFilterChange: (filter: EnforceAIAgentFilter) => void;
  onClearFilters: () => void;
  resultCount: number;
  totalCount: number;
}

function FilterBar({
  filter,
  onFilterChange,
  onClearFilters,
  resultCount,
  totalCount,
}: FilterBarProps) {
  const hasFilters = filter.search || filter.status !== 'all';

  return (
    <div className="flex flex-col sm:flex-row gap-4 mb-6">
      <div className="flex-1">
        <Input
          placeholder="Search agents..."
          value={filter.search}
          onChange={(e) => onFilterChange({ ...filter, search: e.target.value })}
        />
      </div>
      <div className="flex gap-2">
        <select
          value={filter.status}
          onChange={(e) =>
            onFilterChange({
              ...filter,
              status: e.target.value as AgentStatusFilter,
            })
          }
          className="block rounded-md border-gray-300 dark:border-gray-600 shadow-sm focus:border-primary-500 focus:ring-primary-500 sm:text-sm dark:bg-gray-700 dark:text-gray-100"
        >
          <option value="all">All Status</option>
          <option value="active">Active</option>
          <option value="revoked">Revoked</option>
        </select>
      </div>
      <div className="flex items-center gap-2">
        {hasFilters && (
          <Button variant="ghost" size="sm" onClick={onClearFilters}>
            Clear Filters
          </Button>
        )}
        <span className="text-sm text-gray-500 dark:text-gray-400">
          Showing {resultCount} of {totalCount} agents
        </span>
      </div>
    </div>
  );
}

// ============================================================================
// Main Page Component
// ============================================================================

export default function EnforceAIAgentsPage() {
  const navigate = useNavigate();
  const { addToast } = useToast();
  const { agents, isLoading, isError, refetch } = useEnforceAIAgents();

  // Filter state
  const [filter, setFilter] = useState<EnforceAIAgentFilter>({
    search: '',
    status: 'all',
  });

  // Modal state
  const [isCreateModalOpen, setIsCreateModalOpen] = useState(false);
  const [editingAgent, setEditingAgent] = useState<EnforceAIAgent | null>(null);
  const [revokingAgent, setRevokingAgent] = useState<EnforceAIAgent | null>(null);
  const [revokingTokensAgent, setRevokingTokensAgent] = useState<EnforceAIAgent | null>(null);

  // Filter agents
  const filteredAgents = useMemo(
    () => filterEnforceAIAgents(agents, filter),
    [agents, filter]
  );

  // Handlers
  const handleClearFilters = () => {
    setFilter({ search: '', status: 'all' });
  };

  const handleCopyId = (agent: EnforceAIAgent) => {
    navigator.clipboard.writeText(agent.agent_id);
    addToast({
      type: 'success',
      title: 'Copied',
      message: 'Agent ID copied to clipboard',
    });
  };

  const handleCreateSuccess = () => {
    setIsCreateModalOpen(false);
    refetch();
    addToast({
      type: 'success',
      title: 'Agent created',
      message: 'New EnforceAI agent has been created',
    });
  };

  const handleEditSuccess = () => {
    setEditingAgent(null);
    refetch();
    addToast({
      type: 'success',
      title: 'Agent updated',
      message: 'Agent has been updated successfully',
    });
  };

  const handleRevokeSuccess = () => {
    setRevokingAgent(null);
    refetch();
    addToast({
      type: 'success',
      title: 'Agent revoked',
      message: 'Agent has been permanently revoked',
    });
  };

  const handleRevokeTokensSuccess = () => {
    setRevokingTokensAgent(null);
    refetch();
    addToast({
      type: 'success',
      title: 'Tokens revoked',
      message: 'All existing tokens have been invalidated',
    });
  };

  // Loading state
  if (isLoading) {
    return (
      <PageContent>
        <PageHeader
          title="EnforceAI Agents"
          description="Manage agent identities and authorization"
          breadcrumbs={[{ name: 'EnforceAI Agents' }]}
        />
        <div className="flex items-center justify-center py-12">
          <Spinner size="lg" />
        </div>
      </PageContent>
    );
  }

  // Error state
  if (isError) {
    return (
      <PageContent>
        <PageHeader
          title="EnforceAI Agents"
          description="Manage agent identities and authorization"
          breadcrumbs={[{ name: 'EnforceAI Agents' }]}
        />
        <EmptyState
          icon={<ShieldCheckIcon className="w-12 h-12" />}
          title="Failed to load agents"
          description="Unable to fetch EnforceAI agents. This may be because EnforceAI management is not enabled."
          action={
            <Button variant="primary" onClick={() => refetch()}>
              Retry
            </Button>
          }
        />
      </PageContent>
    );
  }

  // Empty state (no agents at all)
  if (agents.length === 0) {
    return (
      <PageContent>
        <PageHeader
          title="EnforceAI Agents"
          description="Manage agent identities and authorization"
          breadcrumbs={[{ name: 'EnforceAI Agents' }]}
          actions={
            <Button
              variant="primary"
              leftIcon={<PlusIcon className="w-4 h-4" />}
              onClick={() => setIsCreateModalOpen(true)}
            >
              Create Agent
            </Button>
          }
        />
        <EmptyState
          icon={<ShieldCheckIcon className="w-12 h-12" />}
          title="No agents created"
          description="Create your first EnforceAI agent to start managing tool access authorization"
          action={
            <Button
              variant="primary"
              leftIcon={<PlusIcon className="w-4 h-4" />}
              onClick={() => setIsCreateModalOpen(true)}
            >
              Create Agent
            </Button>
          }
        />

        {isCreateModalOpen && (
          <CreateAgentModal
            open={isCreateModalOpen}
            onClose={() => setIsCreateModalOpen(false)}
            onSuccess={handleCreateSuccess}
          />
        )}
      </PageContent>
    );
  }

  return (
    <PageContent>
      <PageHeader
        title="EnforceAI Agents"
        description="Manage agent identities and authorization"
        breadcrumbs={[{ name: 'EnforceAI Agents' }]}
        actions={
          <Button
            variant="primary"
            leftIcon={<PlusIcon className="w-4 h-4" />}
            onClick={() => setIsCreateModalOpen(true)}
          >
            Create Agent
          </Button>
        }
      />

      <FilterBar
        filter={filter}
        onFilterChange={setFilter}
        onClearFilters={handleClearFilters}
        resultCount={filteredAgents.length}
        totalCount={agents.length}
      />

      {filteredAgents.length === 0 ? (
        <EmptyState
          icon={<ShieldCheckIcon className="w-8 h-8" />}
          title="No agents match your filters"
          description="Try adjusting your search or filter criteria"
          action={
            <Button variant="secondary" onClick={handleClearFilters}>
              Clear Filters
            </Button>
          }
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredAgents.map((agent) => (
            <AgentCard
              key={agent.agent_id}
              agent={agent}
              onView={() => navigate(`/agents/${agent.agent_id}`)}
              onEdit={() => setEditingAgent(agent)}
              onRevoke={() => setRevokingAgent(agent)}
              onRevokeTokens={() => setRevokingTokensAgent(agent)}
              onCopyId={() => handleCopyId(agent)}
            />
          ))}
        </div>
      )}

      {/* Modals */}
      {isCreateModalOpen && (
        <CreateAgentModal
          open={isCreateModalOpen}
          onClose={() => setIsCreateModalOpen(false)}
          onSuccess={handleCreateSuccess}
        />
      )}

      {editingAgent && (
        <EditAgentModal
          open={!!editingAgent}
          agent={editingAgent}
          onClose={() => setEditingAgent(null)}
          onSuccess={handleEditSuccess}
        />
      )}

      {revokingAgent && (
        <RevokeAgentModal
          open={!!revokingAgent}
          agent={revokingAgent}
          onClose={() => setRevokingAgent(null)}
          onSuccess={handleRevokeSuccess}
        />
      )}

      {revokingTokensAgent && (
        <RevokeAllTokensModal
          open={!!revokingTokensAgent}
          agent={revokingTokensAgent}
          onClose={() => setRevokingTokensAgent(null)}
          onSuccess={handleRevokeTokensSuccess}
        />
      )}
    </PageContent>
  );
}
