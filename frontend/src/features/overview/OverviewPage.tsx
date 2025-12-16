import { useNavigate } from 'react-router-dom';
import {
  ServerIcon,
  UserGroupIcon,
  ShieldCheckIcon,
  PlusIcon,
  KeyIcon,
  ArrowPathIcon,
  CheckCircleIcon,
  XCircleIcon,
  ExclamationTriangleIcon,
} from '@heroicons/react/24/outline';
import { PageContent, PageHeader } from '@/components/layout/index';
import { Button, Card, Badge, Spinner } from '@/components/ui/index';
import {
  useServerStats,
  useA2AAgentStats,
  useEnforceAIAgentStats,
  useConnectionTest,
} from './hooks';

// ============================================================================
// Connection Status Card Component
// ============================================================================

interface ConnectionStatusCardProps {
  title: string;
  description: string;
  reachable: boolean;
  tested: boolean;
  elapsedMs: number;
  enabled?: boolean;
  icon: React.ReactNode;
}

function ConnectionStatusCard({
  title,
  description,
  reachable,
  tested,
  elapsedMs,
  enabled = true,
  icon,
}: ConnectionStatusCardProps) {
  const getStatusIcon = () => {
    if (!enabled) {
      return (
        <ExclamationTriangleIcon className="h-5 w-5 text-yellow-500" aria-hidden="true" />
      );
    }
    if (!tested) {
      return (
        <div className="h-5 w-5 rounded-full bg-gray-200 dark:bg-gray-700" />
      );
    }
    if (reachable) {
      return (
        <CheckCircleIcon className="h-5 w-5 text-green-500" aria-hidden="true" />
      );
    }
    return <XCircleIcon className="h-5 w-5 text-red-500" aria-hidden="true" />;
  };

  const getStatusText = () => {
    if (!enabled) {
      return 'Not Enabled';
    }
    if (!tested) {
      return 'Not Tested';
    }
    return reachable ? 'Connected' : 'Unreachable';
  };

  const getStatusBadge = () => {
    if (!enabled) {
      return <Badge variant="warning">Disabled</Badge>;
    }
    if (!tested) {
      return <Badge variant="neutral">Pending</Badge>;
    }
    return reachable ? (
      <Badge variant="success">Online</Badge>
    ) : (
      <Badge variant="error">Offline</Badge>
    );
  };

  return (
    <Card padding="none">
      <div className="p-6">
        <div className="flex items-start justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex-shrink-0 p-2 bg-gray-100 dark:bg-gray-700 rounded-lg">
              {icon}
            </div>
            <div>
              <h3 className="text-sm font-medium text-gray-900 dark:text-white">
                {title}
              </h3>
              <p className="text-xs text-gray-500 dark:text-gray-400">
                {description}
              </p>
            </div>
          </div>
          {getStatusBadge()}
        </div>

        <div className="mt-4 flex items-center justify-between">
          <div className="flex items-center space-x-2">
            {getStatusIcon()}
            <span className="text-sm text-gray-600 dark:text-gray-300">
              {getStatusText()}
            </span>
          </div>
          {tested && elapsedMs > 0 && (
            <span className="text-xs text-gray-500 dark:text-gray-400">
              {elapsedMs}ms
            </span>
          )}
        </div>
      </div>
    </Card>
  );
}

// ============================================================================
// Stats Card Component
// ============================================================================

interface StatsCardProps {
  title: string;
  total: number;
  breakdown: Array<{
    label: string;
    value: number;
    color: 'green' | 'gray' | 'red';
  }>;
  icon: React.ReactNode;
  isLoading?: boolean;
  linkTo?: string;
}

function StatsCard({
  title,
  total,
  breakdown,
  icon,
  isLoading,
  linkTo,
}: StatsCardProps) {
  const navigate = useNavigate();

  const colorClasses = {
    green: 'bg-green-100 text-green-800 dark:bg-green-900 dark:text-green-200',
    gray: 'bg-gray-100 text-gray-800 dark:bg-gray-700 dark:text-gray-200',
    red: 'bg-red-100 text-red-800 dark:bg-red-900 dark:text-red-200',
  };

  const handleClick = linkTo ? () => navigate(linkTo) : undefined;

  return (
    <div
      className={`bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-lg shadow-sm ${linkTo ? 'cursor-pointer hover:shadow-md transition-shadow' : ''}`}
      onClick={handleClick}
      role={linkTo ? 'button' : undefined}
      tabIndex={linkTo ? 0 : undefined}
      onKeyDown={
        linkTo
          ? (e) => {
              if (e.key === 'Enter' || e.key === ' ') {
                navigate(linkTo);
              }
            }
          : undefined
      }
    >
      <div className="p-6">
        <div className="flex items-center justify-between">
          <div className="flex items-center space-x-3">
            <div className="flex-shrink-0 p-2 bg-blue-100 dark:bg-blue-900 rounded-lg">
              {icon}
            </div>
            <h3 className="text-sm font-medium text-gray-900 dark:text-white">
              {title}
            </h3>
          </div>
          {isLoading ? (
            <Spinner size="sm" />
          ) : (
            <span className="text-2xl font-semibold text-gray-900 dark:text-white">
              {total}
            </span>
          )}
        </div>

        {!isLoading && breakdown.length > 0 && (
          <div className="mt-4 flex flex-wrap gap-2">
            {breakdown.map((item) => (
              <span
                key={item.label}
                className={`inline-flex items-center px-2 py-1 rounded text-xs font-medium ${colorClasses[item.color]}`}
              >
                {item.value} {item.label}
              </span>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}

// ============================================================================
// Quick Action Button Component
// ============================================================================

interface QuickActionProps {
  label: string;
  description: string;
  icon: React.ReactNode;
  onClick: () => void;
  disabled?: boolean;
}

function QuickAction({
  label,
  description,
  icon,
  onClick,
  disabled,
}: QuickActionProps) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="flex items-start p-4 w-full text-left rounded-lg border border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800 hover:bg-gray-50 dark:hover:bg-gray-750 hover:border-blue-300 dark:hover:border-blue-600 transition-colors disabled:opacity-50 disabled:cursor-not-allowed"
    >
      <div className="flex-shrink-0 p-2 bg-blue-100 dark:bg-blue-900 rounded-lg mr-4">
        {icon}
      </div>
      <div>
        <h4 className="text-sm font-medium text-gray-900 dark:text-white">
          {label}
        </h4>
        <p className="text-xs text-gray-500 dark:text-gray-400 mt-1">
          {description}
        </p>
      </div>
    </button>
  );
}

// ============================================================================
// Main Overview Page
// ============================================================================

export default function OverviewPage() {
  const navigate = useNavigate();

  // Fetch stats from all sources
  const serverStats = useServerStats();
  const a2aStats = useA2AAgentStats();
  const enforceStats = useEnforceAIAgentStats();

  // Connection test
  const { status, isTesting, testConnections } = useConnectionTest();

  return (
    <PageContent>
      <PageHeader
        title="Overview"
        description="Welcome to the Enforce Gateway management console"
        actions={
          <Button
            variant="secondary"
            size="sm"
            onClick={testConnections}
            loading={isTesting}
          >
            <ArrowPathIcon className="h-4 w-4 mr-2" aria-hidden="true" />
            Test Connections
          </Button>
        }
      />

      {/* Connection Status Section */}
      <section className="mb-8">
        <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Connection Status
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 gap-4">
          <ConnectionStatusCard
            title="Registry"
            description="MCP Server & Agent Registry"
            reachable={status.registry.reachable}
            tested={status.registry.tested}
            elapsedMs={status.registry.elapsed_ms}
            icon={
              <ServerIcon
                className="h-6 w-6 text-blue-600 dark:text-blue-400"
                aria-hidden="true"
              />
            }
          />
          <ConnectionStatusCard
            title="EnforceAI"
            description="Identity & Authorization Gateway"
            reachable={status.enforceai.reachable}
            tested={status.enforceai.tested}
            elapsedMs={status.enforceai.elapsed_ms}
            enabled={enforceStats.enforceAIEnabled}
            icon={
              <ShieldCheckIcon
                className="h-6 w-6 text-blue-600 dark:text-blue-400"
                aria-hidden="true"
              />
            }
          />
        </div>
      </section>

      {/* Summary Counts Section */}
      <section className="mb-8">
        <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Summary
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
          <StatsCard
            title="MCP Servers"
            total={serverStats.total}
            isLoading={serverStats.isLoading}
            linkTo="/servers"
            icon={
              <ServerIcon
                className="h-5 w-5 text-blue-600 dark:text-blue-400"
                aria-hidden="true"
              />
            }
            breakdown={[
              { label: 'enabled', value: serverStats.enabled, color: 'green' },
              { label: 'disabled', value: serverStats.disabled, color: 'gray' },
            ]}
          />
          <StatsCard
            title="A2A Agents"
            total={a2aStats.total}
            isLoading={a2aStats.isLoading}
            linkTo="/a2a-agents"
            icon={
              <UserGroupIcon
                className="h-5 w-5 text-blue-600 dark:text-blue-400"
                aria-hidden="true"
              />
            }
            breakdown={[
              { label: 'enabled', value: a2aStats.enabled, color: 'green' },
              { label: 'disabled', value: a2aStats.disabled, color: 'gray' },
            ]}
          />
          <StatsCard
            title="EnforceAI Agents"
            total={enforceStats.total}
            isLoading={enforceStats.isLoading}
            linkTo="/agents"
            icon={
              <ShieldCheckIcon
                className="h-5 w-5 text-blue-600 dark:text-blue-400"
                aria-hidden="true"
              />
            }
            breakdown={
              enforceStats.enforceAIEnabled
                ? [
                    { label: 'active', value: enforceStats.active, color: 'green' },
                    { label: 'revoked', value: enforceStats.revoked, color: 'red' },
                  ]
                : []
            }
          />
        </div>

        {/* EnforceAI not enabled warning */}
        {!enforceStats.enforceAIEnabled && (
          <div className="mt-4 p-4 bg-yellow-50 dark:bg-yellow-900/20 border border-yellow-200 dark:border-yellow-700 rounded-lg">
            <div className="flex">
              <ExclamationTriangleIcon
                className="h-5 w-5 text-yellow-400 mr-3 flex-shrink-0"
                aria-hidden="true"
              />
              <div>
                <h3 className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
                  EnforceAI Not Enabled
                </h3>
                <p className="mt-1 text-sm text-yellow-700 dark:text-yellow-300">
                  EnforceAI management features are not available. Contact your
                  administrator to enable the EnforceAI security gateway.
                </p>
              </div>
            </div>
          </div>
        )}
      </section>

      {/* Quick Actions Section */}
      <section>
        <h2 className="text-lg font-medium text-gray-900 dark:text-white mb-4">
          Quick Actions
        </h2>
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-4 gap-4">
          <QuickAction
            label="Create Agent"
            description="Register a new EnforceAI agent"
            icon={
              <PlusIcon
                className="h-5 w-5 text-blue-600 dark:text-blue-400"
                aria-hidden="true"
              />
            }
            onClick={() => navigate('/agents/new')}
            disabled={!enforceStats.enforceAIEnabled}
          />
          <QuickAction
            label="Mint Token"
            description="Generate a new gateway token"
            icon={
              <KeyIcon
                className="h-5 w-5 text-blue-600 dark:text-blue-400"
                aria-hidden="true"
              />
            }
            onClick={() => navigate('/credentials/tokens/mint')}
            disabled={!enforceStats.enforceAIEnabled}
          />
          <QuickAction
            label="Create API Key"
            description="Generate a new API key"
            icon={
              <KeyIcon
                className="h-5 w-5 text-blue-600 dark:text-blue-400"
                aria-hidden="true"
              />
            }
            onClick={() => navigate('/credentials/api-keys/new')}
            disabled={!enforceStats.enforceAIEnabled}
          />
          <QuickAction
            label="Register Server"
            description="Add a new MCP server"
            icon={
              <ServerIcon
                className="h-5 w-5 text-blue-600 dark:text-blue-400"
                aria-hidden="true"
              />
            }
            onClick={() => navigate('/servers/new')}
          />
        </div>
      </section>
    </PageContent>
  );
}
