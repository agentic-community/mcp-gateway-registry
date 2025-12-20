import { useState, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import {
  ServerIcon,
  MagnifyingGlassIcon,
  PlusIcon,
  ArrowPathIcon,
  EyeIcon,
  PencilIcon,
} from '@heroicons/react/24/outline';
import { PageContent, PageHeader } from '@/components/layout/index';
import { Button, Badge, Card, EmptyState, Input, Spinner } from '@/components/ui';
import { useToast } from '@/components/ui/Toast';
import { useServers, useToggleServer, useRefreshServer, filterServers, type ServerFilter } from './hooks';
import { ServerRegisterModal } from './ServerRegisterModal';
import { ServerEditModal } from './ServerEditModal';
import type { Server } from '@/api/types';
import { cn } from '@/lib/cn';
import { formatDistanceToNow } from 'date-fns';
import UpstreamAuthBadge from '@/components/UpstreamAuthBadge';

// ============================================================================
// Server Card Component
// ============================================================================

interface ServerCardProps {
  server: Server;
  onView: () => void;
  onEdit: () => void;
  onToggle: () => void;
  onRefresh: () => void;
  isToggling: boolean;
  isRefreshing: boolean;
}

function ServerCard({
  server,
  onView,
  onEdit,
  onToggle,
  onRefresh,
  isToggling,
  isRefreshing,
}: ServerCardProps) {
  const healthBadgeVariant =
    server.health_status === 'healthy'
      ? 'success'
      : server.health_status === 'unhealthy'
        ? 'error'
        : 'neutral';

  const statusBadgeVariant = server.is_enabled ? 'success' : 'neutral';

  return (
    <Card className="hover:shadow-md transition-shadow">
      <div className="flex items-start justify-between">
        <div className="flex items-start space-x-3">
          <div
            className={cn(
              'flex-shrink-0 w-10 h-10 rounded-lg flex items-center justify-center',
              server.is_enabled
                ? 'bg-primary-100 dark:bg-primary-900/30'
                : 'bg-gray-100 dark:bg-gray-700'
            )}
          >
            <ServerIcon
              className={cn(
                'w-5 h-5',
                server.is_enabled
                  ? 'text-primary-600 dark:text-primary-400'
                  : 'text-gray-400 dark:text-gray-500'
              )}
            />
          </div>
          <div className="min-w-0 flex-1">
            <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
              {server.display_name}
            </h3>
            <p className="text-xs text-gray-500 dark:text-gray-400 font-mono">
              {server.path}
            </p>
          </div>
        </div>
        <div className="flex items-center space-x-2">
          <Badge variant={statusBadgeVariant} size="sm">
            {server.is_enabled ? 'Enabled' : 'Disabled'}
          </Badge>
          <Badge variant={healthBadgeVariant} size="sm">
            {server.health_status || 'Unknown'}
          </Badge>
          {server.upstream_auth && server.upstream_auth.type !== 'none' && (
            <UpstreamAuthBadge
              authType={server.upstream_auth.type}
              credentialStatus={server.upstream_credential_status}
              provider={server.upstream_auth.provider}
              compact={true}
            />
          )}
        </div>
      </div>

      {server.description && (
        <p className="mt-3 text-sm text-gray-600 dark:text-gray-400 line-clamp-2">
          {server.description}
        </p>
      )}

      <div className="mt-4 flex flex-wrap gap-2">
        {server.tags?.map((tag) => (
          <Badge key={tag} variant="info" size="sm">
            {tag}
          </Badge>
        ))}
        {server.num_tools !== undefined && server.num_tools > 0 && (
          <Badge variant="neutral" size="sm">
            {server.num_tools} tool{server.num_tools !== 1 ? 's' : ''}
          </Badge>
        )}
      </div>

      {server.last_checked_iso && (
        <p className="mt-3 text-xs text-gray-400 dark:text-gray-500">
          Last checked {formatDistanceToNow(new Date(server.last_checked_iso), { addSuffix: true })}
        </p>
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
        <div className="flex items-center space-x-2">
          <Button
            variant="ghost"
            size="sm"
            onClick={onRefresh}
            loading={isRefreshing}
            leftIcon={<ArrowPathIcon className="w-4 h-4" />}
          >
            Refresh
          </Button>
          <Button
            variant={server.is_enabled ? 'outline' : 'primary'}
            size="sm"
            onClick={onToggle}
            loading={isToggling}
          >
            {server.is_enabled ? 'Disable' : 'Enable'}
          </Button>
        </div>
      </div>
    </Card>
  );
}

// ============================================================================
// Filter Bar Component
// ============================================================================

interface FilterBarProps {
  filter: ServerFilter;
  onFilterChange: (filter: ServerFilter) => void;
  onRegister: () => void;
}

function FilterBar({ filter, onFilterChange, onRegister }: FilterBarProps) {
  return (
    <div className="flex flex-col sm:flex-row gap-4">
      <div className="flex-1">
        <Input
          placeholder="Search servers..."
          value={filter.search}
          onChange={(e) => onFilterChange({ ...filter, search: e.target.value })}
          leftAddon={<MagnifyingGlassIcon className="w-4 h-4" />}
        />
      </div>
      <div className="flex items-center gap-2">
        <select
          className={cn(
            'rounded-md border-gray-300 text-sm',
            'focus:border-primary-500 focus:ring-primary-500',
            'dark:bg-gray-800 dark:border-gray-600 dark:text-gray-200'
          )}
          value={filter.status}
          onChange={(e) =>
            onFilterChange({ ...filter, status: e.target.value as ServerFilter['status'] })
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
              healthStatus: e.target.value as ServerFilter['healthStatus'],
            })
          }
        >
          <option value="all">All Health</option>
          <option value="healthy">Healthy</option>
          <option value="unhealthy">Unhealthy</option>
          <option value="unknown">Unknown</option>
        </select>
        <Button
          variant="primary"
          size="md"
          onClick={onRegister}
          leftIcon={<PlusIcon className="w-4 h-4" />}
        >
          Register Server
        </Button>
      </div>
    </div>
  );
}

// ============================================================================
// Main Servers Page
// ============================================================================

export default function ServersPage() {
  const navigate = useNavigate();
  const { addToast } = useToast();

  // Data fetching
  const { servers, isLoading, isError, refetch } = useServers();
  const { toggleServer, isToggling } = useToggleServer();
  const { refreshServer, isRefreshing, refreshingPath } = useRefreshServer();

  // Filter state
  const [filter, setFilter] = useState<ServerFilter>({
    search: '',
    status: 'all',
    healthStatus: 'all',
  });

  // Modal state
  const [isRegisterModalOpen, setIsRegisterModalOpen] = useState(false);
  const [editingServer, setEditingServer] = useState<Server | null>(null);

  // Filtered servers
  const filteredServers = useMemo(
    () => filterServers(servers, filter),
    [servers, filter]
  );

  // Handlers
  const handleView = (server: Server) => {
    navigate(`/servers/${encodeURIComponent(server.path)}`);
  };

  const handleEdit = (server: Server) => {
    setEditingServer(server);
  };

  const handleToggle = async (server: Server) => {
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

  const handleRefresh = async (server: Server) => {
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

  const handleRegisterSuccess = () => {
    setIsRegisterModalOpen(false);
    addToast({
      type: 'success',
      title: 'Server registered',
      message: 'The new server has been successfully registered',
    });
  };

  const handleEditSuccess = () => {
    setEditingServer(null);
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
          title="MCP Servers"
          description="Manage registered MCP servers"
          breadcrumbs={[{ name: 'Servers' }]}
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
          title="MCP Servers"
          description="Manage registered MCP servers"
          breadcrumbs={[{ name: 'Servers' }]}
        />
        <EmptyState
          icon={<ServerIcon className="w-12 h-12" />}
          title="Failed to load servers"
          description="An error occurred while fetching the server list. Please try again."
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
        title="MCP Servers"
        description="Manage registered MCP servers"
        breadcrumbs={[{ name: 'Servers' }]}
      />

      {/* Filter Bar */}
      <div className="mb-6">
        <FilterBar
          filter={filter}
          onFilterChange={setFilter}
          onRegister={() => setIsRegisterModalOpen(true)}
        />
      </div>

      {/* Server Grid */}
      {filteredServers.length === 0 ? (
        <EmptyState
          icon={<ServerIcon className="w-12 h-12" />}
          title={servers.length === 0 ? 'No servers registered' : 'No servers match your filters'}
          description={
            servers.length === 0
              ? 'Register your first MCP server to get started'
              : 'Try adjusting your search or filter criteria'
          }
          action={
            servers.length === 0 ? (
              <Button
                variant="primary"
                onClick={() => setIsRegisterModalOpen(true)}
                leftIcon={<PlusIcon className="w-4 h-4" />}
              >
                Register Server
              </Button>
            ) : (
              <Button
                variant="secondary"
                onClick={() =>
                  setFilter({ search: '', status: 'all', healthStatus: 'all' })
                }
              >
                Clear Filters
              </Button>
            )
          }
        />
      ) : (
        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
          {filteredServers.map((server) => (
            <ServerCard
              key={server.path}
              server={server}
              onView={() => handleView(server)}
              onEdit={() => handleEdit(server)}
              onToggle={() => handleToggle(server)}
              onRefresh={() => handleRefresh(server)}
              isToggling={isToggling}
              isRefreshing={isRefreshing && refreshingPath === server.path}
            />
          ))}
        </div>
      )}

      {/* Results count */}
      {filteredServers.length > 0 && (
        <p className="mt-4 text-sm text-gray-500 dark:text-gray-400">
          Showing {filteredServers.length} of {servers.length} server
          {servers.length !== 1 ? 's' : ''}
        </p>
      )}

      {/* Register Modal */}
      <ServerRegisterModal
        open={isRegisterModalOpen}
        onClose={() => setIsRegisterModalOpen(false)}
        onSuccess={handleRegisterSuccess}
      />

      {/* Edit Modal */}
      {editingServer && (
        <ServerEditModal
          open={true}
          server={editingServer}
          onClose={() => setEditingServer(null)}
          onSuccess={handleEditSuccess}
        />
      )}
    </PageContent>
  );
}
