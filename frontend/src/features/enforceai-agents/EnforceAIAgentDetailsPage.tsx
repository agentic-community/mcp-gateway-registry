import { useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import {
  ShieldCheckIcon,
  PencilIcon,
  ArrowLeftIcon,
  NoSymbolIcon,
  KeyIcon,
  ClipboardDocumentIcon,
  ExclamationTriangleIcon,
  ClockIcon,
  WrenchScrewdriverIcon,
} from '@heroicons/react/24/outline';
import { PageContent, PageHeader } from '@/components/layout/index';
import { Button, Badge, Card, Spinner, EmptyState } from '@/components/ui';
import { useToast } from '@/components/ui/Toast';
import { useEnforceAIAgent } from './hooks';
import { EditAgentModal } from './EditAgentModal';
import { RevokeAgentModal } from './RevokeAgentModal';
import { RevokeAllTokensModal } from './RevokeAllTokensModal';
import { formatRelativeTime, formatDateTime } from '@/lib/format';
import { cn } from '@/lib/cn';

// ============================================================================
// Main Agent Details Page
// ============================================================================

export default function EnforceAIAgentDetailsPage() {
  const { agentId } = useParams<{ agentId: string }>();
  const navigate = useNavigate();
  const { addToast } = useToast();

  // Data fetching
  const { agent, isLoading, isError, refetch } = useEnforceAIAgent(agentId || '');

  // Modal state
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isRevokeModalOpen, setIsRevokeModalOpen] = useState(false);
  const [isRevokeTokensModalOpen, setIsRevokeTokensModalOpen] = useState(false);

  // Handlers
  const handleCopyId = () => {
    if (agent) {
      navigator.clipboard.writeText(agent.agent_id);
      addToast({
        type: 'success',
        title: 'Copied',
        message: 'Agent ID copied to clipboard',
      });
    }
  };

  const handleEditSuccess = () => {
    setIsEditModalOpen(false);
    refetch();
    addToast({
      type: 'success',
      title: 'Agent updated',
      message: 'Agent has been updated successfully',
    });
  };

  const handleRevokeSuccess = () => {
    setIsRevokeModalOpen(false);
    addToast({
      type: 'success',
      title: 'Agent revoked',
      message: 'Agent has been permanently revoked',
    });
    navigate('/agents');
  };

  const handleRevokeTokensSuccess = () => {
    setIsRevokeTokensModalOpen(false);
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
          title="Agent Details"
          breadcrumbs={[
            { name: 'EnforceAI Agents', href: '/agents' },
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
            { name: 'EnforceAI Agents', href: '/agents' },
            { name: 'Not Found' },
          ]}
        />
        <EmptyState
          icon={<ShieldCheckIcon className="w-12 h-12" />}
          title="Agent not found"
          description="The agent you're looking for doesn't exist or may have been deleted."
          action={
            <Link to="/agents">
              <Button variant="primary" leftIcon={<ArrowLeftIcon className="w-4 h-4" />}>
                Back to Agents
              </Button>
            </Link>
          }
        />
      </PageContent>
    );
  }

  const isRevoked = agent.revoked_at !== null;
  const hasAllowedTools = agent.allowed_tools && agent.allowed_tools.length > 0;
  const hasEmptyScopes = agent.scopes.length === 0;

  return (
    <PageContent>
      <PageHeader
        title={agent.alias || 'Unnamed Agent'}
        description={`Agent ID: ${agent.agent_id}`}
        breadcrumbs={[
          { name: 'EnforceAI Agents', href: '/agents' },
          { name: agent.alias || agent.agent_id.slice(0, 8) },
        ]}
        actions={
          !isRevoked && (
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
                onClick={() => setIsRevokeModalOpen(true)}
                leftIcon={<NoSymbolIcon className="w-4 h-4" />}
              >
                Revoke
              </Button>
            </div>
          )
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
                    isRevoked
                      ? 'bg-gray-100 dark:bg-gray-700'
                      : 'bg-indigo-100 dark:bg-indigo-900/30'
                  )}
                >
                  <ShieldCheckIcon
                    className={cn(
                      'w-6 h-6',
                      isRevoked
                        ? 'text-gray-400 dark:text-gray-500'
                        : 'text-indigo-600 dark:text-indigo-400'
                    )}
                  />
                </div>
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
                    {agent.alias || 'Unnamed Agent'}
                  </h3>
                  <div className="flex items-center space-x-2">
                    <code className="text-xs text-gray-500 dark:text-gray-400 font-mono">
                      {agent.agent_id}
                    </code>
                    <button
                      onClick={handleCopyId}
                      className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                      title="Copy agent ID"
                    >
                      <ClipboardDocumentIcon className="w-4 h-4" />
                    </button>
                  </div>
                </div>
              </div>
              <Badge variant={isRevoked ? 'error' : 'success'}>
                {isRevoked ? 'Revoked' : 'Active'}
              </Badge>
            </div>

            {/* Warnings */}
            {hasEmptyScopes && !isRevoked && (
              <div className="mb-4 p-3 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-800 rounded-md">
                <div className="flex items-center space-x-2 text-yellow-700 dark:text-yellow-400">
                  <ExclamationTriangleIcon className="w-5 h-5" />
                  <span className="text-sm font-medium">No scopes defined</span>
                </div>
                <p className="text-xs text-yellow-600 dark:text-yellow-500 mt-1">
                  This agent has no scopes and will not have access to any tools.
                </p>
              </div>
            )}

            {isRevoked && (
              <div className="mb-4 p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-md">
                <div className="flex items-center space-x-2 text-red-700 dark:text-red-400">
                  <NoSymbolIcon className="w-5 h-5" />
                  <span className="text-sm font-medium">Agent Revoked</span>
                </div>
                <p className="text-xs text-red-600 dark:text-red-500 mt-1">
                  This agent was revoked on {formatDateTime(agent.revoked_at!)}. All tokens and API keys are invalid.
                </p>
              </div>
            )}

            <dl className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  User ID
                </dt>
                <dd className="mt-1 text-sm text-gray-900 dark:text-gray-100 font-mono break-all">
                  {agent.user_id}
                </dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  Created
                </dt>
                <dd className="mt-1 text-sm text-gray-900 dark:text-gray-100">
                  {formatDateTime(agent.created_at)}
                </dd>
              </div>
              <div>
                <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  Last Updated
                </dt>
                <dd className="mt-1 text-sm text-gray-900 dark:text-gray-100">
                  {formatDateTime(agent.updated_at)}
                </dd>
              </div>
              {agent.tokens_valid_after && (
                <div>
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 flex items-center space-x-1">
                    <ClockIcon className="w-4 h-4" />
                    <span>Tokens Valid After</span>
                  </dt>
                  <dd className="mt-1 text-sm text-gray-900 dark:text-gray-100">
                    {formatDateTime(agent.tokens_valid_after)}
                    <span className="text-xs text-gray-500 ml-1">
                      ({formatRelativeTime(agent.tokens_valid_after)})
                    </span>
                  </dd>
                </div>
              )}
            </dl>

            {!isRevoked && (
              <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700 flex items-center justify-end space-x-2">
                <Button
                  variant="outline"
                  onClick={() => setIsRevokeTokensModalOpen(true)}
                  leftIcon={<KeyIcon className="w-4 h-4" />}
                >
                  Revoke All Tokens
                </Button>
              </div>
            )}
          </Card>

          {/* Scopes Card */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
                Scopes
              </h3>
              <Badge variant="neutral">
                {agent.scopes.length} scope{agent.scopes.length !== 1 ? 's' : ''}
              </Badge>
            </div>
            {agent.scopes.length > 0 ? (
              <div className="space-y-2">
                {agent.scopes.map((scope) => (
                  <div
                    key={scope}
                    className="border border-gray-200 dark:border-gray-700 rounded-md p-3"
                  >
                    <div className="flex items-center space-x-3">
                      <ShieldCheckIcon className="w-5 h-5 text-indigo-500" />
                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100 font-mono">
                        {scope}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <EmptyState
                icon={<ShieldCheckIcon className="w-8 h-8" />}
                title="No scopes defined"
                description="Add scopes to grant this agent access to tools"
              />
            )}
          </Card>

          {/* Allowed Tools Card */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
                Allowed Tools
              </h3>
              <Badge variant="neutral">
                {hasAllowedTools
                  ? `${agent.allowed_tools!.length} tool${agent.allowed_tools!.length !== 1 ? 's' : ''}`
                  : 'All tools'}
              </Badge>
            </div>
            {hasAllowedTools ? (
              <div className="space-y-2">
                {agent.allowed_tools!.map((tool) => (
                  <div
                    key={tool}
                    className="border border-gray-200 dark:border-gray-700 rounded-md p-3"
                  >
                    <div className="flex items-center space-x-3">
                      <WrenchScrewdriverIcon className="w-5 h-5 text-blue-500" />
                      <p className="text-sm font-medium text-gray-900 dark:text-gray-100 font-mono">
                        {tool}
                      </p>
                    </div>
                  </div>
                ))}
              </div>
            ) : (
              <div className="p-4 bg-gray-50 dark:bg-gray-800 rounded-md">
                <p className="text-sm text-gray-600 dark:text-gray-400">
                  No tool restrictions. This agent can access all tools permitted by its scopes.
                </p>
              </div>
            )}
          </Card>

          {/* Metadata Card */}
          {agent.metadata && Object.keys(agent.metadata).length > 0 && (
            <Card>
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100 mb-4">
                Metadata
              </h3>
              <pre className="text-sm bg-gray-50 dark:bg-gray-800 rounded-md p-4 overflow-auto">
                <code className="text-gray-700 dark:text-gray-300">
                  {JSON.stringify(agent.metadata, null, 2)}
                </code>
              </pre>
            </Card>
          )}
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
                  <Badge variant={isRevoked ? 'error' : 'success'} size="sm">
                    {isRevoked ? 'Revoked' : 'Active'}
                  </Badge>
                </dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-sm text-gray-500 dark:text-gray-400">Scopes</dt>
                <dd className="text-sm text-gray-900 dark:text-gray-100">
                  {agent.scopes.length}
                </dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-sm text-gray-500 dark:text-gray-400">Tools</dt>
                <dd className="text-sm text-gray-900 dark:text-gray-100">
                  {hasAllowedTools ? agent.allowed_tools!.length : 'All'}
                </dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-sm text-gray-500 dark:text-gray-400">Created</dt>
                <dd className="text-sm text-gray-900 dark:text-gray-100">
                  {formatRelativeTime(agent.created_at)}
                </dd>
              </div>
            </dl>
          </Card>

          {/* Actions */}
          {!isRevoked && (
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
                  onClick={() => setIsRevokeTokensModalOpen(true)}
                  leftIcon={<KeyIcon className="w-4 h-4" />}
                >
                  Revoke All Tokens
                </Button>
                <Button
                  variant="danger"
                  className="w-full justify-start"
                  onClick={() => setIsRevokeModalOpen(true)}
                  leftIcon={<NoSymbolIcon className="w-4 h-4" />}
                >
                  Revoke Agent
                </Button>
              </div>
            </Card>
          )}

          {/* Links */}
          <Card>
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 mb-4">
              Related
            </h3>
            <div className="space-y-2">
              <Link
                to={`/credentials?agent=${agent.agent_id}`}
                className="block text-sm text-indigo-600 dark:text-indigo-400 hover:underline"
              >
                View API Keys &rarr;
              </Link>
              <Link
                to="/scopes"
                className="block text-sm text-indigo-600 dark:text-indigo-400 hover:underline"
              >
                View Scope Catalog &rarr;
              </Link>
            </div>
          </Card>
        </div>
      </div>

      {/* Modals */}
      {isEditModalOpen && (
        <EditAgentModal
          open={isEditModalOpen}
          agent={agent}
          onClose={() => setIsEditModalOpen(false)}
          onSuccess={handleEditSuccess}
        />
      )}

      {isRevokeModalOpen && (
        <RevokeAgentModal
          open={isRevokeModalOpen}
          agent={agent}
          onClose={() => setIsRevokeModalOpen(false)}
          onSuccess={handleRevokeSuccess}
        />
      )}

      {isRevokeTokensModalOpen && (
        <RevokeAllTokensModal
          open={isRevokeTokensModalOpen}
          agent={agent}
          onClose={() => setIsRevokeTokensModalOpen(false)}
          onSuccess={handleRevokeTokensSuccess}
        />
      )}
    </PageContent>
  );
}
