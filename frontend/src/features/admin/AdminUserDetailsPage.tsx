/**
 * AdminUserDetailsPage - Detailed view of a specific user with cross-user operations
 */

import { useState } from 'react';
import { useParams, Link } from 'react-router-dom';
import { PageHeader } from '@/components/layout/PageHeader';
import { PageContent } from '@/components/layout/PageContent';
import { Card } from '@/components/ui/Card';
import { Badge } from '@/components/ui/Badge';
import { Spinner } from '@/components/ui/Spinner';
import { EmptyState } from '@/components/ui/EmptyState';
import { CopyButton } from '@/components/ui/CopyButton';
import { Button } from '@/components/ui/Button';
import { useToast } from '@/components/ui/Toast';
import { AdminLayout } from './AdminLayout';
import { TargetUserBanner } from './TargetUserBanner';
import {
  AdminCreateAgentModal,
  AdminRevokeAgentModal,
  AdminRevokeAllTokensModal,
  AdminRevokeApiKeyModal,
  AdminRevokeTokenModal,
} from './AdminModals';
import {
  useAdminUser,
  useAdminUserAgents,
  useAdminUserAgentApiKeys,
  useAdminCreateAgent,
  useAdminRevokeAgent,
  useAdminRevokeAllTokens,
  useAdminRevokeApiKey,
  useAdminRevokeToken,
} from './hooks';
import {
  UserCircleIcon,
  ShieldCheckIcon,
  KeyIcon,
  PlusIcon,
  NoSymbolIcon,
  ArrowLeftIcon,
  ChevronDownIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';
import { formatDistanceToNow } from 'date-fns';
import type { EnforceAIAgent, ApiKeySummary } from '@/api/types';

/**
 * Agent Card Component with expandable API keys
 */
interface AgentCardProps {
  agent: EnforceAIAgent;
  userId: string;
  onRevokeAgent: (agent: EnforceAIAgent) => void;
  onRevokeAllTokens: (agent: EnforceAIAgent) => void;
  onRevokeApiKey: (keyId: string) => void;
}

function AgentCard({
  agent,
  userId,
  onRevokeAgent,
  onRevokeAllTokens,
  onRevokeApiKey,
}: AgentCardProps) {
  const [expanded, setExpanded] = useState(false);
  const { apiKeys, isLoading: keysLoading } = useAdminUserAgentApiKeys(
    userId,
    expanded ? agent.agent_id : ''
  );

  const isRevoked = Boolean(agent.revoked_at);
  const activeKeys = apiKeys.filter((k) => !k.revoked_at);
  const revokedKeys = apiKeys.filter((k) => k.revoked_at);

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-lg overflow-hidden">
      {/* Agent Header */}
      <div
        className={`flex items-center justify-between p-4 cursor-pointer hover:bg-gray-50 dark:hover:bg-gray-900 ${
          isRevoked ? 'bg-red-50/50 dark:bg-red-900/10' : ''
        }`}
        onClick={() => setExpanded(!expanded)}
      >
        <div className="flex items-center gap-3 min-w-0 flex-1">
          {expanded ? (
            <ChevronDownIcon className="h-5 w-5 text-gray-400 flex-shrink-0" />
          ) : (
            <ChevronRightIcon className="h-5 w-5 text-gray-400 flex-shrink-0" />
          )}
          <ShieldCheckIcon
            className={`h-5 w-5 flex-shrink-0 ${
              isRevoked ? 'text-red-400' : 'text-green-500'
            }`}
          />
          <div className="min-w-0">
            <div className="flex items-center gap-2 flex-wrap">
              <span className="text-sm font-medium text-gray-900 dark:text-gray-100">
                {agent.alias || 'Unnamed Agent'}
              </span>
              {isRevoked && (
                <Badge variant="error" size="sm">
                  Revoked
                </Badge>
              )}
            </div>
            <div className="flex items-center gap-2">
              <code className="text-xs text-gray-500 dark:text-gray-400 truncate">
                {agent.agent_id}
              </code>
              <CopyButton text={agent.agent_id} />
            </div>
          </div>
        </div>
        <div className="flex items-center gap-2 ml-4">
          <Badge variant="neutral" size="sm">
            {agent.scopes.length} scopes
          </Badge>
          {!isRevoked && (
            <div className="flex gap-1" onClick={(e) => e.stopPropagation()}>
              <Button
                variant="secondary"
                size="sm"
                onClick={() => onRevokeAllTokens(agent)}
                title="Revoke all tokens"
              >
                <NoSymbolIcon className="h-4 w-4" />
              </Button>
              <Button
                variant="danger"
                size="sm"
                onClick={() => onRevokeAgent(agent)}
                title="Revoke agent"
              >
                Revoke
              </Button>
            </div>
          )}
        </div>
      </div>

      {/* Expanded Content */}
      {expanded && (
        <div className="border-t border-gray-200 dark:border-gray-700 p-4 bg-gray-50 dark:bg-gray-900/50">
          {/* Scopes */}
          <div className="mb-4">
            <h5 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase mb-2">
              Scopes
            </h5>
            <div className="flex flex-wrap gap-1">
              {agent.scopes.map((scope) => (
                <Badge key={scope} variant="info" size="sm">
                  {scope}
                </Badge>
              ))}
            </div>
          </div>

          {/* Timestamps */}
          <div className="mb-4 grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-500 dark:text-gray-400">Created:</span>{' '}
              <span className="text-gray-900 dark:text-gray-100">
                {formatDistanceToNow(new Date(agent.created_at), { addSuffix: true })}
              </span>
            </div>
            {agent.tokens_valid_after && (
              <div>
                <span className="text-gray-500 dark:text-gray-400">Tokens valid after:</span>{' '}
                <span className="text-gray-900 dark:text-gray-100">
                  {formatDistanceToNow(new Date(agent.tokens_valid_after), { addSuffix: true })}
                </span>
              </div>
            )}
            {agent.revoked_at && (
              <div>
                <span className="text-gray-500 dark:text-gray-400">Revoked:</span>{' '}
                <span className="text-red-600 dark:text-red-400">
                  {formatDistanceToNow(new Date(agent.revoked_at), { addSuffix: true })}
                </span>
              </div>
            )}
          </div>

          {/* API Keys */}
          <div>
            <h5 className="text-xs font-semibold text-gray-500 dark:text-gray-400 uppercase mb-2">
              API Keys
            </h5>
            {keysLoading ? (
              <div className="flex justify-center py-4">
                <Spinner size="sm" />
              </div>
            ) : apiKeys.length === 0 ? (
              <p className="text-sm text-gray-500 dark:text-gray-400">No API keys</p>
            ) : (
              <div className="space-y-2">
                {activeKeys.map((key) => (
                  <ApiKeyRow key={key.key_id} apiKey={key} onRevoke={onRevokeApiKey} />
                ))}
                {revokedKeys.length > 0 && (
                  <details className="mt-2">
                    <summary className="text-xs text-gray-500 cursor-pointer">
                      {revokedKeys.length} revoked key(s)
                    </summary>
                    <div className="mt-2 space-y-2">
                      {revokedKeys.map((key) => (
                        <ApiKeyRow key={key.key_id} apiKey={key} onRevoke={onRevokeApiKey} />
                      ))}
                    </div>
                  </details>
                )}
              </div>
            )}
          </div>
        </div>
      )}
    </div>
  );
}

/**
 * API Key Row Component
 */
interface ApiKeyRowProps {
  apiKey: ApiKeySummary;
  onRevoke: (keyId: string) => void;
}

function ApiKeyRow({ apiKey, onRevoke }: ApiKeyRowProps) {
  const isRevoked = Boolean(apiKey.revoked_at);

  return (
    <div
      className={`flex items-center justify-between p-2 rounded ${
        isRevoked
          ? 'bg-gray-100 dark:bg-gray-800 opacity-60'
          : 'bg-white dark:bg-gray-800'
      }`}
    >
      <div className="flex items-center gap-2 min-w-0">
        <KeyIcon className="h-4 w-4 text-gray-400 flex-shrink-0" />
        <code className="text-xs text-gray-600 dark:text-gray-400 truncate">
          {apiKey.key_id}
        </code>
        <CopyButton text={apiKey.key_id} />
        {isRevoked && (
          <Badge variant="error" size="sm">
            Revoked
          </Badge>
        )}
      </div>
      {!isRevoked && (
        <Button variant="danger" size="sm" onClick={() => onRevoke(apiKey.key_id)}>
          Revoke
        </Button>
      )}
    </div>
  );
}

/**
 * Admin User Details page component
 */
export default function AdminUserDetailsPage() {
  const { userId } = useParams<{ userId: string }>();
  const { showToast } = useToast();
  const { user, isLoading: userLoading, isError: userError } = useAdminUser(userId || '');
  const {
    agents,
    isLoading: agentsLoading,
    isError: agentsError,
    refetch: refetchAgents,
  } = useAdminUserAgents(userId || '');

  // Mutation hooks
  const { createAgent, isLoading: createAgentLoading } = useAdminCreateAgent(userId || '');
  const { revokeAgent, isLoading: revokeAgentLoading } = useAdminRevokeAgent(userId || '');
  const { revokeAllTokens, isLoading: revokeAllTokensLoading } = useAdminRevokeAllTokens(
    userId || ''
  );
  const { revokeApiKey, isLoading: revokeApiKeyLoading } = useAdminRevokeApiKey(userId || '');
  const { revokeToken, isLoading: revokeTokenLoading } = useAdminRevokeToken(userId || '');

  // Modal state
  const [createAgentModalOpen, setCreateAgentModalOpen] = useState(false);
  const [revokeAgentModalOpen, setRevokeAgentModalOpen] = useState(false);
  const [revokeAllTokensModalOpen, setRevokeAllTokensModalOpen] = useState(false);
  const [revokeApiKeyModalOpen, setRevokeApiKeyModalOpen] = useState(false);
  const [revokeTokenModalOpen, setRevokeTokenModalOpen] = useState(false);
  const [selectedAgent, setSelectedAgent] = useState<EnforceAIAgent | null>(null);
  const [selectedKeyId, setSelectedKeyId] = useState<string | null>(null);

  // Handlers
  const handleCreateAgent = async (data: Parameters<typeof createAgent>[0]) => {
    await createAgent(data);
    showToast({ type: 'success', message: 'Agent created successfully' });
    refetchAgents();
  };

  const handleRevokeAgent = async (reason?: string) => {
    if (!selectedAgent) return;
    await revokeAgent(selectedAgent.agent_id);
    showToast({ type: 'success', message: 'Agent revoked successfully' });
    refetchAgents();
  };

  const handleRevokeAllTokens = async () => {
    if (!selectedAgent) return;
    await revokeAllTokens(selectedAgent.agent_id);
    showToast({ type: 'success', message: 'All tokens revoked successfully' });
    refetchAgents();
  };

  const handleRevokeApiKey = async () => {
    if (!selectedKeyId) return;
    await revokeApiKey(selectedKeyId);
    showToast({ type: 'success', message: 'API key revoked successfully' });
    refetchAgents();
  };

  const handleRevokeToken = async (agentId: string, jti: string, reason?: string) => {
    await revokeToken({ agent_id: agentId, jti, reason });
    showToast({ type: 'success', message: 'Token revoked successfully' });
  };

  const openRevokeAgentModal = (agent: EnforceAIAgent) => {
    setSelectedAgent(agent);
    setRevokeAgentModalOpen(true);
  };

  const openRevokeAllTokensModal = (agent: EnforceAIAgent) => {
    setSelectedAgent(agent);
    setRevokeAllTokensModalOpen(true);
  };

  const openRevokeApiKeyModal = (keyId: string) => {
    setSelectedKeyId(keyId);
    setRevokeApiKeyModalOpen(true);
  };

  if (userLoading) {
    return (
      <>
        <PageHeader title="User Details" description="Loading user information..." />
        <PageContent>
          <div className="flex items-center justify-center py-12">
            <Spinner size="lg" />
          </div>
        </PageContent>
      </>
    );
  }

  if (userError || !user) {
    return (
      <>
        <PageHeader title="User Details" description="User not found" />
        <PageContent>
          <EmptyState
            icon={<UserCircleIcon className="h-12 w-12" />}
            title="User not found"
            description="The requested user could not be found or you don't have permission to view it."
            action={
              <Link to="/admin/users">
                <Button>Back to User Directory</Button>
              </Link>
            }
          />
        </PageContent>
      </>
    );
  }

  const activeAgents = agents.filter((a) => !a.revoked_at);
  const revokedAgents = agents.filter((a) => a.revoked_at);
  const targetUserEmail = user.email || user.username || user.user_id;

  return (
    <>
      <PageHeader
        title="User Details"
        description={`Managing user: ${targetUserEmail}`}
        actions={
          <Link to="/admin/users">
            <Button variant="secondary" size="sm">
              <ArrowLeftIcon className="h-4 w-4 mr-1" />
              Back to Directory
            </Button>
          </Link>
        }
      />

      <PageContent>
        <AdminLayout>
          {/* Target User Banner */}
          <TargetUserBanner targetUserEmail={targetUserEmail} targetUserId={user.user_id} />

          <div className="space-y-6">
            {/* User Info Card */}
            <Card>
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  User Information
                </h3>
              </div>
              <div className="p-6">
                <dl className="grid grid-cols-1 gap-4 sm:grid-cols-2">
                  <div>
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                      Email
                    </dt>
                    <dd className="mt-1 text-sm text-gray-900 dark:text-gray-100">
                      {user.email || '—'}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                      Username
                    </dt>
                    <dd className="mt-1 text-sm text-gray-900 dark:text-gray-100">
                      {user.username || '—'}
                    </dd>
                  </div>
                  <div className="sm:col-span-2">
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400 mb-2">
                      User ID (Canonical)
                    </dt>
                    <dd className="flex items-center gap-2">
                      <code className="text-sm text-gray-900 dark:text-gray-100 bg-gray-100 dark:bg-gray-800 px-3 py-1 rounded">
                        {user.user_id}
                      </code>
                      <CopyButton text={user.user_id} />
                    </dd>
                  </div>
                  <div>
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                      Auth Method
                    </dt>
                    <dd className="mt-1">
                      <Badge variant="neutral">{user.auth_method}</Badge>
                    </dd>
                  </div>
                  <div>
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                      Role
                    </dt>
                    <dd className="mt-1 text-sm text-gray-900 dark:text-gray-100">
                      {user.role || '—'}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                      Last Seen
                    </dt>
                    <dd className="mt-1 text-sm text-gray-900 dark:text-gray-100">
                      {user.last_seen_at
                        ? formatDistanceToNow(new Date(user.last_seen_at), {
                            addSuffix: true,
                          })
                        : '—'}
                    </dd>
                  </div>
                  <div>
                    <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                      Created At
                    </dt>
                    <dd className="mt-1 text-sm text-gray-900 dark:text-gray-100">
                      {user.created_at
                        ? formatDistanceToNow(new Date(user.created_at), {
                            addSuffix: true,
                          })
                        : '—'}
                    </dd>
                  </div>
                </dl>
              </div>
            </Card>

            {/* Agent Summary Stats */}
            <Card>
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700 flex items-center justify-between">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  EnforceAI Agents
                </h3>
                <Button
                  variant="primary"
                  size="sm"
                  onClick={() => setCreateAgentModalOpen(true)}
                >
                  <PlusIcon className="h-4 w-4 mr-1" />
                  Create Agent
                </Button>
              </div>
              <div className="p-6">
                {agentsLoading && (
                  <div className="flex items-center justify-center py-8">
                    <Spinner />
                  </div>
                )}

                {agentsError && (
                  <EmptyState
                    icon={<ShieldCheckIcon className="h-12 w-12" />}
                    title="Failed to load agents"
                    description="Could not retrieve agents for this user."
                  />
                )}

                {!agentsLoading && !agentsError && (
                  <>
                    {/* Stats */}
                    <div className="grid grid-cols-1 gap-4 sm:grid-cols-3 mb-6">
                      <div className="bg-gray-50 dark:bg-gray-900 rounded-lg p-4">
                        <div className="text-2xl font-semibold text-gray-900 dark:text-gray-100">
                          {agents.length}
                        </div>
                        <div className="text-sm text-gray-600 dark:text-gray-400 mt-1">
                          Total Agents
                        </div>
                      </div>
                      <div className="bg-green-50 dark:bg-green-900/20 rounded-lg p-4">
                        <div className="text-2xl font-semibold text-green-700 dark:text-green-400">
                          {activeAgents.length}
                        </div>
                        <div className="text-sm text-green-600 dark:text-green-400 mt-1">
                          Active Agents
                        </div>
                      </div>
                      <div className="bg-red-50 dark:bg-red-900/20 rounded-lg p-4">
                        <div className="text-2xl font-semibold text-red-700 dark:text-red-400">
                          {revokedAgents.length}
                        </div>
                        <div className="text-sm text-red-600 dark:text-red-400 mt-1">
                          Revoked Agents
                        </div>
                      </div>
                    </div>

                    {/* Agents List */}
                    {agents.length === 0 ? (
                      <EmptyState
                        icon={<ShieldCheckIcon className="h-12 w-12" />}
                        title="No agents"
                        description="This user has no EnforceAI agents yet."
                        action={
                          <Button
                            variant="primary"
                            onClick={() => setCreateAgentModalOpen(true)}
                          >
                            <PlusIcon className="h-4 w-4 mr-1" />
                            Create First Agent
                          </Button>
                        }
                      />
                    ) : (
                      <div className="space-y-3">
                        {activeAgents.map((agent) => (
                          <AgentCard
                            key={agent.agent_id}
                            agent={agent}
                            userId={user.user_id}
                            onRevokeAgent={openRevokeAgentModal}
                            onRevokeAllTokens={openRevokeAllTokensModal}
                            onRevokeApiKey={openRevokeApiKeyModal}
                          />
                        ))}
                        {revokedAgents.length > 0 && (
                          <details className="mt-4">
                            <summary className="text-sm text-gray-500 cursor-pointer font-medium">
                              {revokedAgents.length} revoked agent(s)
                            </summary>
                            <div className="mt-3 space-y-3">
                              {revokedAgents.map((agent) => (
                                <AgentCard
                                  key={agent.agent_id}
                                  agent={agent}
                                  userId={user.user_id}
                                  onRevokeAgent={openRevokeAgentModal}
                                  onRevokeAllTokens={openRevokeAllTokensModal}
                                  onRevokeApiKey={openRevokeApiKeyModal}
                                />
                              ))}
                            </div>
                          </details>
                        )}
                      </div>
                    )}
                  </>
                )}
              </div>
            </Card>

            {/* Revoke Token Section */}
            <Card>
              <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                  Revoke Gateway Token
                </h3>
              </div>
              <div className="p-6">
                <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
                  Revoke a specific gateway token by its JTI (JWT ID). You'll need the agent ID
                  and JTI from the token's claims.
                </p>
                <Button variant="secondary" onClick={() => setRevokeTokenModalOpen(true)}>
                  <NoSymbolIcon className="h-4 w-4 mr-1" />
                  Revoke Token by JTI
                </Button>
              </div>
            </Card>
          </div>
        </AdminLayout>
      </PageContent>

      {/* Modals */}
      <AdminCreateAgentModal
        open={createAgentModalOpen}
        onClose={() => setCreateAgentModalOpen(false)}
        onSubmit={handleCreateAgent}
        targetUserEmail={targetUserEmail}
        targetUserId={user.user_id}
        loading={createAgentLoading}
      />

      <AdminRevokeAgentModal
        open={revokeAgentModalOpen}
        onClose={() => {
          setRevokeAgentModalOpen(false);
          setSelectedAgent(null);
        }}
        onConfirm={handleRevokeAgent}
        agent={selectedAgent}
        targetUserEmail={targetUserEmail}
        loading={revokeAgentLoading}
      />

      <AdminRevokeAllTokensModal
        open={revokeAllTokensModalOpen}
        onClose={() => {
          setRevokeAllTokensModalOpen(false);
          setSelectedAgent(null);
        }}
        onConfirm={handleRevokeAllTokens}
        agent={selectedAgent}
        targetUserEmail={targetUserEmail}
        loading={revokeAllTokensLoading}
      />

      <AdminRevokeApiKeyModal
        open={revokeApiKeyModalOpen}
        onClose={() => {
          setRevokeApiKeyModalOpen(false);
          setSelectedKeyId(null);
        }}
        onConfirm={handleRevokeApiKey}
        keyId={selectedKeyId}
        targetUserEmail={targetUserEmail}
        loading={revokeApiKeyLoading}
      />

      <AdminRevokeTokenModal
        open={revokeTokenModalOpen}
        onClose={() => setRevokeTokenModalOpen(false)}
        onConfirm={handleRevokeToken}
        targetUserEmail={targetUserEmail}
        targetUserId={user.user_id}
        loading={revokeTokenLoading}
      />
    </>
  );
}
