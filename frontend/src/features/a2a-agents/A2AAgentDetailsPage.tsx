import { useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import {
  CpuChipIcon,
  PencilIcon,
  TrashIcon,
  ArrowLeftIcon,
  SparklesIcon,
  GlobeAltIcon,
  LockClosedIcon,
  StarIcon,
} from '@heroicons/react/24/outline';
import { PageContent, PageHeader } from '@/components/layout/index';
import { Button, Badge, Card, Spinner, EmptyState, ConfirmDialog } from '@/components/ui';
import { useToast } from '@/components/ui/Toast';
import { useAgent, useToggleAgent, useDeleteAgent } from './hooks';
import { AgentEditModal } from './AgentEditModal';
import { cn } from '@/lib/cn';

// ============================================================================
// Skill Item Component
// ============================================================================

interface SkillItemProps {
  skill: string;
}

function SkillItem({ skill }: SkillItemProps) {
  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-md p-3">
      <div className="flex items-center space-x-3">
        <SparklesIcon className="w-5 h-5 text-purple-500" />
        <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
          {skill}
        </p>
      </div>
    </div>
  );
}

// ============================================================================
// Main Agent Details Page
// ============================================================================

export default function A2AAgentDetailsPage() {
  const { path } = useParams<{ path: string }>();
  const navigate = useNavigate();
  const { addToast } = useToast();

  // Data fetching
  const { agent, isLoading, isError, refetch } = useAgent(path || '');
  const { toggleAgent, isToggling } = useToggleAgent();
  const { deleteAgent, isDeleting } = useDeleteAgent();

  // Modal state
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);

  // Handlers
  const handleToggle = async () => {
    if (!agent) return;
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

  const handleDelete = async () => {
    if (!agent) return;
    try {
      await deleteAgent(agent.path);
      addToast({
        type: 'success',
        title: 'Agent deleted',
        message: `${agent.name} has been removed from the registry`,
      });
      navigate('/a2a-agents');
    } catch {
      addToast({
        type: 'error',
        title: 'Failed to delete agent',
        message: 'An error occurred while deleting the agent',
      });
    }
    setIsDeleteDialogOpen(false);
  };

  const handleEditSuccess = () => {
    setIsEditModalOpen(false);
    refetch();
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
          title="Agent Details"
          breadcrumbs={[
            { name: 'A2A Agents', href: '/a2a-agents' },
            { name: 'Loading...' },
          ]}
        />
        <div className="flex items-center justify-center py-12">
          <Spinner size="lg" />
        </div>
      </PageContent>
    );
  }

  // Error state
  if (isError || !agent) {
    return (
      <PageContent>
        <PageHeader
          title="Agent Not Found"
          breadcrumbs={[
            { name: 'A2A Agents', href: '/a2a-agents' },
            { name: 'Not Found' },
          ]}
        />
        <EmptyState
          icon={<CpuChipIcon className="w-12 h-12" />}
          title="Agent not found"
          description="The agent you're looking for doesn't exist or may have been deleted."
          action={
            <Link to="/a2a-agents">
              <Button variant="primary" leftIcon={<ArrowLeftIcon className="w-4 h-4" />}>
                Back to Agents
              </Button>
            </Link>
          }
        />
      </PageContent>
    );
  }

  const healthBadgeVariant =
    agent.health_status === 'healthy'
      ? 'success'
      : agent.health_status === 'unhealthy'
        ? 'error'
        : 'neutral';

  const statusBadgeVariant = agent.is_enabled ? 'success' : 'neutral';
  const visibilityBadgeVariant = agent.visibility === 'public' ? 'info' : 'neutral';

  return (
    <PageContent>
      <PageHeader
        title={agent.name}
        description={agent.description}
        breadcrumbs={[
          { name: 'A2A Agents', href: '/a2a-agents' },
          { name: agent.name },
        ]}
        actions={
          <div className="flex items-center space-x-2">
            <Button
              variant="ghost"
              size="sm"
              onClick={() => setIsEditModalOpen(true)}
              leftIcon={<PencilIcon className="w-4 h-4" />}
            >
              Edit
            </Button>
            <Button
              variant="danger"
              size="sm"
              onClick={() => setIsDeleteDialogOpen(true)}
              leftIcon={<TrashIcon className="w-4 h-4" />}
            >
              Delete
            </Button>
          </div>
        }
      />

      <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
        {/* Main Content */}
        <div className="lg:col-span-2 space-y-6">
          {/* Agent Info Card */}
          <Card>
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div
                  className={cn(
                    'flex-shrink-0 w-12 h-12 rounded-lg flex items-center justify-center',
                    agent.is_enabled
                      ? 'bg-purple-100 dark:bg-purple-900/30'
                      : 'bg-gray-100 dark:bg-gray-700'
                  )}
                >
                  <CpuChipIcon
                    className={cn(
                      'w-6 h-6',
                      agent.is_enabled
                        ? 'text-purple-600 dark:text-purple-400'
                        : 'text-gray-400 dark:text-gray-500'
                    )}
                  />
                </div>
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
                    {agent.name}
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400 font-mono">
                    {agent.path}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Badge variant={statusBadgeVariant}>
                  {agent.is_enabled ? 'Enabled' : 'Disabled'}
                </Badge>
                <Badge variant={healthBadgeVariant}>
                  {agent.health_status || 'Unknown'}
                </Badge>
              </div>
            </div>

            <dl className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              {agent.url && (
                <div>
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Agent URL
                  </dt>
                  <dd className="mt-1 text-sm text-gray-900 dark:text-gray-100 font-mono break-all">
                    {agent.url}
                  </dd>
                </div>
              )}
              <div>
                <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  Visibility
                </dt>
                <dd className="mt-1">
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
                </dd>
              </div>
              {agent.tags && agent.tags.length > 0 && (
                <div>
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Tags
                  </dt>
                  <dd className="mt-1 flex flex-wrap gap-1">
                    {agent.tags.map((tag) => (
                      <Badge key={tag} variant="info" size="sm">
                        {tag}
                      </Badge>
                    ))}
                  </dd>
                </div>
              )}
              {agent.provider && (
                <div>
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Provider
                  </dt>
                  <dd className="mt-1 text-sm text-gray-900 dark:text-gray-100">
                    {agent.provider}
                  </dd>
                </div>
              )}
            </dl>

            <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700 flex items-center justify-end space-x-2">
              <Button
                variant={agent.is_enabled ? 'outline' : 'primary'}
                onClick={handleToggle}
                loading={isToggling}
              >
                {agent.is_enabled ? 'Disable Agent' : 'Enable Agent'}
              </Button>
            </div>
          </Card>

          {/* Skills Card */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
                Skills
              </h3>
              <Badge variant="neutral">
                {agent.skills?.length || 0} skill{(agent.skills?.length || 0) !== 1 ? 's' : ''}
              </Badge>
            </div>
            {agent.skills && agent.skills.length > 0 ? (
              <div className="space-y-2">
                {agent.skills.map((skill) => (
                  <SkillItem key={skill} skill={skill} />
                ))}
              </div>
            ) : (
              <EmptyState
                icon={<SparklesIcon className="w-8 h-8" />}
                title="No skills defined"
                description="This agent hasn't defined any skills yet"
              />
            )}
          </Card>
        </div>

        {/* Sidebar */}
        <div className="space-y-6">
          {/* Quick Info */}
          <Card>
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
              Quick Info
            </h3>
            <dl className="space-y-3">
              <div className="flex items-center justify-between">
                <dt className="text-sm text-gray-500 dark:text-gray-400">Status</dt>
                <dd>
                  <Badge variant={statusBadgeVariant} size="sm">
                    {agent.is_enabled ? 'Enabled' : 'Disabled'}
                  </Badge>
                </dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-sm text-gray-500 dark:text-gray-400">Health</dt>
                <dd>
                  <Badge variant={healthBadgeVariant} size="sm">
                    {agent.health_status || 'Unknown'}
                  </Badge>
                </dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-sm text-gray-500 dark:text-gray-400">Skills</dt>
                <dd className="text-sm text-gray-900 dark:text-gray-100">
                  {agent.skills?.length || agent.num_skills || 0}
                </dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-sm text-gray-500 dark:text-gray-400">Stars</dt>
                <dd className="text-sm text-gray-900 dark:text-gray-100 flex items-center">
                  <StarIcon className="w-4 h-4 text-yellow-500 mr-1" />
                  {agent.num_stars || 0}
                </dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-sm text-gray-500 dark:text-gray-400">Visibility</dt>
                <dd>
                  <Badge variant={visibilityBadgeVariant} size="sm">
                    {agent.visibility || 'public'}
                  </Badge>
                </dd>
              </div>
              {agent.streaming !== undefined && (
                <div className="flex items-center justify-between">
                  <dt className="text-sm text-gray-500 dark:text-gray-400">Streaming</dt>
                  <dd className="text-sm text-gray-900 dark:text-gray-100">
                    {agent.streaming ? 'Yes' : 'No'}
                  </dd>
                </div>
              )}
              {agent.trust_level && (
                <div className="flex items-center justify-between">
                  <dt className="text-sm text-gray-500 dark:text-gray-400">Trust Level</dt>
                  <dd className="text-sm text-gray-900 dark:text-gray-100">
                    {agent.trust_level}
                  </dd>
                </div>
              )}
            </dl>
          </Card>

          {/* Actions */}
          <Card>
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
              Actions
            </h3>
            <div className="space-y-2">
              <Button
                variant="secondary"
                className="w-full justify-start"
                onClick={() => setIsEditModalOpen(true)}
                leftIcon={<PencilIcon className="w-4 h-4" />}
              >
                Edit Agent
              </Button>
              <Button
                variant="secondary"
                className="w-full justify-start"
                onClick={handleToggle}
                loading={isToggling}
              >
                {agent.is_enabled ? 'Disable Agent' : 'Enable Agent'}
              </Button>
              <Button
                variant="danger"
                className="w-full justify-start"
                onClick={() => setIsDeleteDialogOpen(true)}
                leftIcon={<TrashIcon className="w-4 h-4" />}
              >
                Delete Agent
              </Button>
            </div>
          </Card>
        </div>
      </div>

      {/* Edit Modal */}
      {isEditModalOpen && (
        <AgentEditModal
          open={isEditModalOpen}
          agent={agent}
          onClose={() => setIsEditModalOpen(false)}
          onSuccess={handleEditSuccess}
        />
      )}

      {/* Delete Confirmation Dialog */}
      <ConfirmDialog
        open={isDeleteDialogOpen}
        onClose={() => setIsDeleteDialogOpen(false)}
        onConfirm={handleDelete}
        title="Delete Agent"
        message={`Are you sure you want to delete "${agent.name}"? This action cannot be undone.`}
        confirmLabel="Delete"
        variant="danger"
        loading={isDeleting}
      />
    </PageContent>
  );
}
