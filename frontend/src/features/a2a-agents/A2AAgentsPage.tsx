import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  CpuChipIcon,
  MagnifyingGlassIcon,
  PlusIcon,
  EyeIcon,
  PencilIcon,
  StarIcon,
  GlobeAltIcon,
  LockClosedIcon,
} from '@heroicons/react/24/outline';
import { PageContent, PageHeader } from '@/components/layout/index';
import { Button, Badge, Card, EmptyState, Input, Spinner } from '@/components/ui';
import { useToast } from '@/components/ui/Toast';
import { useAgents, useToggleAgent, filterAgents, type AgentFilter } from './hooks';
import { AgentRegisterModal } from './AgentRegisterModal';
import { AgentEditModal } from './AgentEditModal';
import type { A2AAgent } from '@/api/types';
import { cn } from '@/lib/cn';

// ============================================================================
// Agent Card Component
// ============================================================================

interface AgentCardProps {
  agent: A2AAgent;
  onView: () => void;
  onEdit: () => void;
  onToggle: () => void;
  isToggling: boolean;
}

function AgentCard({
  agent,
  onView,
  onEdit,
  onToggle,
  isToggling,
}: AgentCardProps) {
  const healthBadgeVariant =
    agent.health_status === 'healthy'
      ? 'success'
      : agent.health_status === 'unhealthy'
        ? 'error'
        : 'neutral';

  const statusBadgeVariant = agent.is_enabled ? 'success' : 'neutral';
  const visibilityBadgeVariant = agent.visibility === 'public' ? 'info' : 'neutral';

  return (
    <Card className="hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-3">
          <div
            className={cn(
              'flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center',
              agent.is_enabled
                ? 'bg-purple-100 dark:bg-purple-900/30'
                : 'bg-gray-100 dark:bg-gray-700'
            )}
          >
            <CpuChipIcon
              className={cn(
                'w-5 h-5',
                agent.is_enabled
                  ? 'text-purple-600 dark:text-purple-400'
                  : 'text-gray-400 dark:text-gray-500'
              )}
            />
          </div>
          <div className="min-w-0 flex-1">
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
              {agent.name}
            </h3>
            <p className="text-xs text-gray-500 dark:text-gray-400 font-mono">
              {agent.path}
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant={statusBadgeVariant} size="sm">
            {agent.is_enabled ? 'Enabled' : 'Disabled'}
          </Badge>
          <Badge variant={healthBadgeVariant} size="sm">
            {agent.health_status || 'Unknown'}
          </Badge>
        </div>
      </div>

      {agent.description && (
        <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
          {agent.description}
        </p>
      )}

      <div className="mt-4 flex flex-wrap gap-2">
        {/* Visibility Badge */}
        <Badge variant={visibilityBadgeVariant} size="sm">
          {agent.visibility === 'public' ? (
            <span className="flex items-center">
              <GlobeAltIcon className="w-3 h-3 mr-1" />
              Public
            </span>
          ) : (
            <span className="flex items-center">
              <LockClosedIcon className="w-3 h-3 mr-1" />
              Private
            </span>
          )}
        </Badge>
        {/* Tags */}
        {agent.tags?.map((tag) => (
          <Badge key={tag} variant="info" size="sm">
            {tag}
          </Badge>
        ))}
        {/* Skills count */}
        {(agent.num_skills ?? 0) > 0 && (
          <Badge variant="neutral" size="sm">
            {agent.num_skills} skill{agent.num_skills !== 1 ? 's' : ''}
          </Badge>
        )}
        {/* Stars */}
        {(agent.num_stars ?? 0) > 0 && (
          <Badge variant="warning" size="sm">
            <span className="flex items-center">
              <StarIcon className="w-3 h-3 mr-1" />
              {agent.num_stars}
            </span>
          </Badge>
        )}
      </div>

      {/* Skills Preview */}
      {agent.skills && agent.skills.length > 0 && (
        <div className="mt-3 flex flex-wrap gap-1">
          {agent.skills.slice(0, 3).map((skill) => (
            <span
              key={skill}
              className="text-xs px-2 py-0.5 rounded bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400"
            >
              {skill}
            </span>
          ))}
          {agent.skills.length > 3 && (
            <span className="text-xs text-gray-400 dark:text-gray-500">
              +{agent.skills.length - 3} more
            </span>
          )}
        </div>
      )}

      <div className="mt-4 pt-4 border-t border-gray-200 dark:border-gray-700 flex items-center justify-between">
        <div className="flex items-center space-x-2">
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
            onClick={onEdit}
            leftIcon={<PencilIcon className="w-4 h-4" />}
          >
            Edit
          </Button>
        </div>
        <Button
          variant={agent.is_enabled ? 'outline' : 'primary'}
          size="sm"
          onClick={onToggle}
          loading={isToggling}
        >
          {agent.is_enabled ? 'Disable' : 'Enable'}
        </Button>
      </div>
    </Card>
  );
}

// ============================================================================
// Filter Bar Component
// ============================================================================

interface FilterBarProps {
  filter: AgentFilter;
  onFilterChange: (filter: AgentFilter) => void;
  onRegister: () => void;
}

function FilterBar({ filter, onFilterChange, onRegister }: FilterBarProps) {
  return (
    <div className="flex flex-col sm:flex-row gap-4">
      <div className="flex-1">
        <Input
          placeholder="Search agents..."
          value={filter.search}
          onChange={(e) => onFilterChange({ ...filter, search: e.target.value })}
          leftAddon={<MagnifyingGlassIcon className="w-4 h-4" />}
        />
      </div>
      <div className="flex items-center gap-2 flex-wrap">
        <select
          className={cn(
            'rounded-md border-gray-300 text-sm',
            'focus:border-primary-500 focus:ring-primary-500',
            'dark:bg-gray-800 dark:border-gray-600 dark:text-gray-200'
          )}
          value={filter.status}
          onChange={(e) =>
            onFilterChange({ ...filter, status: e.target.value as AgentFilter['status'] })
          }
        >
          <option value="all">All Status</option>
          <option value="enabled">Enabled</option>
          <option value="disabled">Disabled</option>
        </select>
        <select
          className={cn(
            'rounded-md border-gray-300 text-sm',
            'focus:border-primary-500 focus:ring-primary-500',
            'dark:bg-gray-800 dark:border-gray-600 dark:text-gray-200'
          )}
          value={filter.healthStatus}
          onChange={(e) =>
            onFilterChange({
              ...filter,
              healthStatus: e.target.value as AgentFilter['healthStatus'],
            })
          }
        >
          <option value="all">All Health</option>
          <option value="healthy">Healthy</option>
          <option value="unhealthy">Unhealthy</option>
          <option value="unknown">Unknown</option>
        </select>
        <select
          className={cn(
            'rounded-md border-gray-300 text-sm',
            'focus:border-primary-500 focus:ring-primary-500',
            'dark:bg-gray-800 dark:border-gray-600 dark:text-gray-200'
          )}
          value={filter.visibility}
          onChange={(e) =>
            onFilterChange({
              ...filter,
              visibility: e.target.value as AgentFilter['visibility'],
            })
          }
        >
          <option value="all">All Visibility</option>
          <option value="public">Public</option>
          <option value="private">Private</option>
        </select>
        <Button
          variant="primary"
          size="md"
          onClick={onRegister}
          leftIcon={<PlusIcon className="w-4 h-4" />}
        >
          Register Agent
        </Button>
      </div>
    </div>
  );
}

// ============================================================================
// Main A2A Agents Page
// ============================================================================

export default function A2AAgentsPage() {
  const navigate = useNavigate();
  const { addToast } = useToast();

  // Data fetching
  const { agents, isLoading, isError, refetch } = useAgents();
  const { toggleAgent, isToggling } = useToggleAgent();

  // Filter state
  const [filter, setFilter] = useState<AgentFilter>({
    search: '',
    status: 'all',
    healthStatus: 'all',
    visibility: 'all',
  });

  // Modal state
  const [isRegisterModalOpen, setIsRegisterModalOpen] = useState(false);
  const [editingAgent, setEditingAgent] = useState<A2AAgent | null>(null);

  // Filtered agents
  const filteredAgents = useMemo(
    () => filterAgents(agents, filter),
    [agents, filter]
  );

  // Handlers
  const handleView = (agent: A2AAgent) => {
    navigate(`/a2a-agents/${encodeURIComponent(agent.path)}`);
  };

  const handleEdit = (agent: A2AAgent) => {
    setEditingAgent(agent);
  };

  const handleToggle = async (agent: A2AAgent) => {
    try {
      await toggleAgent(agent.path, !agent.is_enabled);
      addToast({
        type: 'success',
        title: 'Agent updated',
        message: `${agent.name} has been ${agent.is_enabled ? 'disabled' : 'enabled'}`,
      });
    } catch {
      addToast({
        type: 'error',
        title: 'Failed to toggle agent',
        message: 'An error occurred while updating the agent status',
      });
    }
  };

  const handleRegisterSuccess = () => {
    setIsRegisterModalOpen(false);
    addToast({
      type: 'success',
      title: 'Agent registered',
      message: 'The new agent has been successfully registered',
    });
  };

  const handleEditSuccess = () => {
    setEditingAgent(null);
    addToast({
      type: 'success',
      title: 'Agent updated',
      message: 'The agent has been successfully updated',
    });
  };

  // Loading state
  if (isLoading) {
    return (
      <PageContent>
        <PageHeader
          title="A2A Agents"
          description="Manage registered Agent-to-Agent agents"
          breadcrumbs={[{ name: 'A2A Agents' }]}
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
          title="A2A Agents"
          description="Manage registered Agent-to-Agent agents"
          breadcrumbs={[{ name: 'A2A Agents' }]}
        />
        <EmptyState
          icon={<CpuChipIcon className="w-12 h-12" />}
          title="Failed to load agents"
          description="An error occurred while fetching the agent list. Please try again."
          action={
            <Button variant="primary" onClick={() => refetch()}>
              Retry
            </Button>
          }
        />
      </PageContent>
    );
  }

  return (
    <PageContent>
      <PageHeader
        title="A2A Agents"
        description="Manage registered Agent-to-Agent agents"
        breadcrumbs={[{ name: 'A2A Agents' }]}
      />

      {/* Filter Bar */}
      <div className="mb-6">
        <FilterBar
          filter={filter}
          onFilterChange={setFilter}
          onRegister={() => setIsRegisterModalOpen(true)}
        />
      </div>

      {/* Agent Grid */}
      {filteredAgents.length === 0 ? (
        <EmptyState
          icon={<CpuChipIcon className="w-12 h-12" />}
          title={agents.length === 0 ? 'No agents registered' : 'No agents match your filters'}
          description={
            agents.length === 0
              ? 'Register your first A2A agent to get started'
              : 'Try adjusting your search or filter criteria'
          }
          action={
            agents.length === 0 ? (
              <Button
                variant="primary"
                onClick={() => setIsRegisterModalOpen(true)}
                leftIcon={<PlusIcon className="w-4 h-4" />}
              >
                Register Agent
              </Button>
            ) : (
              <Button
                variant="secondary"
                onClick={() =>
                  setFilter({ search: '', status: 'all', healthStatus: 'all', visibility: 'all' })
                }
              >
                Clear Filters
              </Button>
            )
          }
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredAgents.map((agent) => (
            <AgentCard
              key={agent.path}
              agent={agent}
              onView={() => handleView(agent)}
              onEdit={() => handleEdit(agent)}
              onToggle={() => handleToggle(agent)}
              isToggling={isToggling}
            />
          ))}
        </div>
      )}

      {/* Results count */}
      {filteredAgents.length > 0 && (
        <p className="mt-4 text-sm text-gray-500 dark:text-gray-400">
          Showing {filteredAgents.length} of {agents.length} agent
          {agents.length !== 1 ? 's' : ''}
        </p>
      )}

      {/* Register Modal */}
      <AgentRegisterModal
        open={isRegisterModalOpen}
        onClose={() => setIsRegisterModalOpen(false)}
        onSuccess={handleRegisterSuccess}
      />

      {/* Edit Modal */}
      {editingAgent && (
        <AgentEditModal
          open={true}
          agent={editingAgent}
          onClose={() => setEditingAgent(null)}
          onSuccess={handleEditSuccess}
        />
      )}
    </PageContent>
  );
}
