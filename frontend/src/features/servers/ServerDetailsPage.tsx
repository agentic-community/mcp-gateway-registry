import { useState } from 'react';
import { useParams, useNavigate, Link } from 'react-router-dom';
import {
  ServerIcon,
  ArrowPathIcon,
  PencilIcon,
  TrashIcon,
  ChevronDownIcon,
  ChevronRightIcon,
  ArrowLeftIcon,
  WrenchScrewdriverIcon,
} from '@heroicons/react/24/outline';
import { PageContent, PageHeader } from '@/components/layout/index';
import { Button, Badge, Card, Spinner, EmptyState, ConfirmDialog } from '@/components/ui';
import { useToast } from '@/components/ui/Toast';
import { useServer, useToggleServer, useRefreshServer, useDeleteServer } from './hooks';
import { ServerEditModal } from './ServerEditModal';
import { cn } from '@/lib/cn';
import { formatDistanceToNow, format } from 'date-fns';
import type { Tool } from '@/api/types';
import UpstreamAuthBadge from '@/components/UpstreamAuthBadge';

// ============================================================================
// Tool Item Component
// ============================================================================

interface ToolItemProps {
  tool: Tool;
}

function ToolItem({ tool }: ToolItemProps) {
  const [isExpanded, setIsExpanded] = useState(false);
  const hasSchema = tool.input_schema && Object.keys(tool.input_schema).length > 0;

  return (
    <div className="border border-gray-200 dark:border-gray-700 rounded-md">
      <button
        type="button"
        onClick={() => hasSchema && setIsExpanded(!isExpanded)}
        className={cn(
          'w-full flex items-center justify-between p-3 text-left',
          hasSchema && 'hover:bg-gray-50 dark:hover:bg-gray-700/50'
        )}
        disabled={!hasSchema}
      >
        <div className="flex items-center space-x-3">
          <WrenchScrewdriverIcon className="w-5 h-5 text-gray-400" />
          <div>
            <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
              {tool.name}
            </p>
            {tool.description && (
              <p className="text-xs text-gray-500 dark:text-gray-400 mt-0.5">
                {tool.description}
              </p>
            )}
          </div>
        </div>
        {hasSchema && (
          isExpanded ? (
            <ChevronDownIcon className="w-4 h-4 text-gray-400" />
          ) : (
            <ChevronRightIcon className="w-4 h-4 text-gray-400" />
          )
        )}
      </button>
      {isExpanded && hasSchema && (
        <div className="px-3 pb-3 pt-1 border-t border-gray-200 dark:border-gray-700">
          <p className="text-xs font-medium text-gray-500 dark:text-gray-400 mb-2">
            Input Schema
          </p>
          <pre className="text-xs bg-gray-100 dark:bg-gray-700 p-2 rounded overflow-x-auto">
            {JSON.stringify(tool.input_schema, null, 2)}
          </pre>
        </div>
      )}
    </div>
  );
}

// ============================================================================
// Main Server Details Page
// ============================================================================

export default function ServerDetailsPage() {
  const { path } = useParams<{ path: string }>();
  const navigate = useNavigate();
  const { addToast } = useToast();

  // Data fetching
  const { server, isLoading, isError, refetch } = useServer(path || '');
  const { toggleServer, isToggling } = useToggleServer();
  const { refreshServer, isRefreshing } = useRefreshServer();
  const { deleteServer, isDeleting } = useDeleteServer();

  // Modal state
  const [isEditModalOpen, setIsEditModalOpen] = useState(false);
  const [isDeleteDialogOpen, setIsDeleteDialogOpen] = useState(false);

  // Handlers
  const handleToggle = async () => {
    if (!server) return;
    try {
      await toggleServer(server.path, !server.is_enabled);
      addToast({
        type: 'success',
        title: 'Server updated',
        message: `${server.display_name} has been ${server.is_enabled ? 'disabled' : 'enabled'}`,
      });
    } catch {
      addToast({
        type: 'error',
        title: 'Failed to toggle server',
        message: 'An error occurred while updating the server status',
      });
    }
  };

  const handleRefresh = async () => {
    if (!server) return;
    try {
      await refreshServer(server.path);
      addToast({
        type: 'success',
        title: 'Health check complete',
        message: `${server.display_name} health status has been refreshed`,
      });
    } catch {
      addToast({
        type: 'error',
        title: 'Health check failed',
        message: 'Unable to refresh server health status',
      });
    }
  };

  const handleDelete = async () => {
    if (!server) return;
    try {
      await deleteServer(server.path);
      addToast({
        type: 'success',
        title: 'Server deleted',
        message: `${server.display_name} has been removed from the registry`,
      });
      navigate('/servers');
    } catch {
      addToast({
        type: 'error',
        title: 'Failed to delete server',
        message: 'An error occurred while deleting the server',
      });
    }
    setIsDeleteDialogOpen(false);
  };

  const handleEditSuccess = () => {
    setIsEditModalOpen(false);
    refetch();
    addToast({
      type: 'success',
      title: 'Server updated',
      message: 'The server has been successfully updated',
    });
  };

  // Loading state
  if (isLoading) {
    return (
      <PageContent>
        <PageHeader
          title="Server Details"
          breadcrumbs={[
            { name: 'Servers', href: '/servers' },
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
  if (isError || !server) {
    return (
      <PageContent>
        <PageHeader
          title="Server Not Found"
          breadcrumbs={[
            { name: 'Servers', href: '/servers' },
            { name: 'Not Found' },
          ]}
        />
        <EmptyState
          icon={<ServerIcon className="w-12 h-12" />}
          title="Server not found"
          description="The server you're looking for doesn't exist or may have been deleted."
          action={
            <Link to="/servers">
              <Button variant="primary" leftIcon={<ArrowLeftIcon className="w-4 h-4" />}>
                Back to Servers
              </Button>
            </Link>
          }
        />
      </PageContent>
    );
  }

  const healthBadgeVariant =
    server.health_status === 'healthy'
      ? 'success'
      : server.health_status === 'unhealthy'
        ? 'error'
        : 'neutral';

  const statusBadgeVariant = server.is_enabled ? 'success' : 'neutral';

  return (
    <PageContent>
      <PageHeader
        title={server.display_name}
        description={server.description}
        breadcrumbs={[
          { name: 'Servers', href: '/servers' },
          { name: server.display_name },
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
          {/* Server Info Card */}
          <Card>
            <div className="flex items-start justify-between mb-4">
              <div className="flex items-center space-x-3">
                <div
                  className={cn(
                    'flex-shrink-0 w-12 h-12 rounded-lg flex items-center justify-center',
                    server.is_enabled
                      ? 'bg-primary-100 dark:bg-primary-900/30'
                      : 'bg-gray-100 dark:bg-gray-700'
                  )}
                >
                  <ServerIcon
                    className={cn(
                      'w-6 h-6',
                      server.is_enabled
                        ? 'text-primary-600 dark:text-primary-400'
                        : 'text-gray-400 dark:text-gray-500'
                    )}
                  />
                </div>
                <div>
                  <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
                    {server.display_name}
                  </h3>
                  <p className="text-sm text-gray-500 dark:text-gray-400 font-mono">
                    {server.path}
                  </p>
                </div>
              </div>
              <div className="flex items-center space-x-2">
                <Badge variant={statusBadgeVariant}>
                  {server.is_enabled ? 'Enabled' : 'Disabled'}
                </Badge>
                <Badge variant={healthBadgeVariant}>
                  {server.health_status || 'Unknown'}
                </Badge>
              </div>
            </div>

            <dl className="grid grid-cols-1 sm:grid-cols-2 gap-4">
              <div>
                <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                  Proxy URL
                </dt>
                <dd className="mt-1 text-sm text-gray-900 dark:text-gray-100 font-mono break-all">
                  {server.proxy_pass_url}
                </dd>
              </div>
              {server.tags && server.tags.length > 0 && (
                <div>
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Tags
                  </dt>
                  <dd className="mt-1 flex flex-wrap gap-1">
                    {server.tags.map((tag) => (
                      <Badge key={tag} variant="info" size="sm">
                        {tag}
                      </Badge>
                    ))}
                  </dd>
                </div>
              )}
              {server.upstream_auth && (
                <div>
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Upstream Authentication
                  </dt>
                  <dd className="mt-1">
                    <UpstreamAuthBadge
                      authType={server.upstream_auth.type}
                      credentialStatus={server.upstream_credential_status}
                      provider={server.upstream_auth.provider}
                      compact={false}
                    />
                  </dd>
                </div>
              )}
              {server.last_checked_iso && (
                <div>
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Last Health Check
                  </dt>
                  <dd className="mt-1 text-sm text-gray-900 dark:text-gray-100">
                    {formatDistanceToNow(new Date(server.last_checked_iso), {
                      addSuffix: true,
                    })}
                    <span className="text-gray-500 dark:text-gray-400 ml-2">
                      ({format(new Date(server.last_checked_iso), 'PPpp')})
                    </span>
                  </dd>
                </div>
              )}
              {server.supported_transports && server.supported_transports.length > 0 && (
                <div>
                  <dt className="text-sm font-medium text-gray-500 dark:text-gray-400">
                    Supported Transports
                  </dt>
                  <dd className="mt-1 flex flex-wrap gap-1">
                    {server.supported_transports.map((transport) => (
                      <Badge key={transport} variant="neutral" size="sm">
                        {transport}
                      </Badge>
                    ))}
                  </dd>
                </div>
              )}
            </dl>

            <div className="mt-6 pt-4 border-t border-gray-200 dark:border-gray-700 flex items-center justify-end space-x-2">
              <Button
                variant="secondary"
                onClick={handleRefresh}
                loading={isRefreshing}
                leftIcon={<ArrowPathIcon className="w-4 h-4" />}
              >
                Refresh Health
              </Button>
              <Button
                variant={server.is_enabled ? 'outline' : 'primary'}
                onClick={handleToggle}
                loading={isToggling}
              >
                {server.is_enabled ? 'Disable Server' : 'Enable Server'}
              </Button>
            </div>
          </Card>

          {/* Tools Card */}
          <Card>
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
                Tools
              </h3>
              <Badge variant="neutral">
                {server.tools?.length || 0} tool{(server.tools?.length || 0) !== 1 ? 's' : ''}
              </Badge>
            </div>
            {server.tools && server.tools.length > 0 ? (
              <div className="space-y-2">
                {server.tools.map((tool) => (
                  <ToolItem key={tool.name} tool={tool} />
                ))}
              </div>
            ) : (
              <EmptyState
                icon={<WrenchScrewdriverIcon className="w-8 h-8" />}
                title="No tools available"
                description="This server hasn't registered any tools yet"
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
                    {server.is_enabled ? 'Enabled' : 'Disabled'}
                  </Badge>
                </dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-sm text-gray-500 dark:text-gray-400">Health</dt>
                <dd>
                  <Badge variant={healthBadgeVariant} size="sm">
                    {server.health_status || 'Unknown'}
                  </Badge>
                </dd>
              </div>
              <div className="flex items-center justify-between">
                <dt className="text-sm text-gray-500 dark:text-gray-400">Tools</dt>
                <dd className="text-sm text-gray-900 dark:text-gray-100">
                  {server.tools?.length || server.num_tools || 0}
                </dd>
              </div>
              {server.license && (
                <div className="flex items-center justify-between">
                  <dt className="text-sm text-gray-500 dark:text-gray-400">License</dt>
                  <dd className="text-sm text-gray-900 dark:text-gray-100">
                    {server.license}
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
                Edit Server
              </Button>
              <Button
                variant="secondary"
                className="w-full justify-start"
                onClick={handleRefresh}
                loading={isRefreshing}
                leftIcon={<ArrowPathIcon className="w-4 h-4" />}
              >
                Refresh Health
              </Button>
              <Button
                variant="secondary"
                className="w-full justify-start"
                onClick={handleToggle}
                loading={isToggling}
              >
                {server.is_enabled ? 'Disable Server' : 'Enable Server'}
              </Button>
              <Button
                variant="danger"
                className="w-full justify-start"
                onClick={() => setIsDeleteDialogOpen(true)}
                leftIcon={<TrashIcon className="w-4 h-4" />}
              >
                Delete Server
              </Button>
            </div>
          </Card>
        </div>
      </div>

      {/* Edit Modal */}
      {isEditModalOpen && (
        <ServerEditModal
          open={isEditModalOpen}
          server={server}
          onClose={() => setIsEditModalOpen(false)}
          onSuccess={handleEditSuccess}
        />
      )}

      {/* Delete Confirmation Dialog */}
      <ConfirmDialog
        open={isDeleteDialogOpen}
        onClose={() => setIsDeleteDialogOpen(false)}
        onConfirm={handleDelete}
        title="Delete Server"
        message={`Are you sure you want to delete "${server.display_name}"? This action cannot be undone.`}
        confirmLabel="Delete"
        variant="danger"
        loading={isDeleting}
      />
    </PageContent>
  );
}
