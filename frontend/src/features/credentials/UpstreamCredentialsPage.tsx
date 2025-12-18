/**
 * UpstreamCredentialsPage - Manage upstream authentication credentials
 * Shows all servers requiring upstream credentials with status filtering
 */

import { useState, useEffect, useMemo } from 'react';
import { useNavigate } from 'react-router-dom';
import { PageHeader } from '@/components/layout/PageHeader';
import { PageContent } from '@/components/layout/PageContent';
import { Card } from '@/components/ui/Card';
import { Button } from '@/components/ui/Button';
import { Badge } from '@/components/ui/Badge';
import { Spinner } from '@/components/ui/Spinner';
import { EmptyState } from '@/components/ui/EmptyState';
import { useToast } from '@/components/ui/Toast';
import { UpstreamAuthBadge } from '@/components/UpstreamAuthBadge';
import { UpstreamCredentialModal } from './UpstreamCredentialModal';
import {
  KeyIcon,
  FunnelIcon,
  ServerIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  ClockIcon,
  XCircleIcon,
} from '@heroicons/react/24/outline';
import { getServers } from '@/api/registry';
import type { Server, UpstreamCredentialStatus } from '@/api/types';

type StatusFilter = 'all' | UpstreamCredentialStatus;

/**
 * Main upstream credentials page component
 */
export default function UpstreamCredentialsPage() {
  const { addToast } = useToast();
  const navigate = useNavigate();
  const [servers, setServers] = useState<Server[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [statusFilter, setStatusFilter] = useState<StatusFilter>('all');

  // Modal state
  const [selectedServer, setSelectedServer] = useState<Server | null>(null);
  const [isModalOpen, setIsModalOpen] = useState(false);

  // Fetch servers
  const fetchServers = async () => {
    try {
      setLoading(true);
      setError(null);
      const data = await getServers();
      setServers(data);
    } catch (err: any) {
      console.error('Failed to fetch servers:', err);
      setError(err.message || 'Failed to load servers');
      addToast({
        type: 'error',
        title: 'Failed to load servers',
        message: err.message || 'An error occurred',
      });
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => {
    fetchServers();
  }, []);

  // Filter servers that require upstream credentials
  const serversWithUpstreamAuth = useMemo(() => {
    return servers.filter(
      (server) =>
        server.upstream_auth &&
        server.upstream_auth.mode === 'gateway-managed' &&
        server.upstream_auth.type !== 'none'
    );
  }, [servers]);

  // Apply status filter
  const filteredServers = useMemo(() => {
    if (statusFilter === 'all') {
      return serversWithUpstreamAuth;
    }
    return serversWithUpstreamAuth.filter(
      (server) => server.upstream_credential_status === statusFilter
    );
  }, [serversWithUpstreamAuth, statusFilter]);

  // Calculate status counts
  const statusCounts = useMemo(() => {
    const counts = {
      all: serversWithUpstreamAuth.length,
      configured: 0,
      missing: 0,
      expired: 0,
      revoked: 0,
    };

    serversWithUpstreamAuth.forEach((server) => {
      const status = server.upstream_credential_status;
      if (status) {
        counts[status]++;
      }
    });

    return counts;
  }, [serversWithUpstreamAuth]);

  // Get icon for status filter
  const getStatusIcon = (status: StatusFilter) => {
    switch (status) {
      case 'configured':
        return <CheckCircleIcon className="h-4 w-4" />;
      case 'missing':
        return <ExclamationCircleIcon className="h-4 w-4" />;
      case 'expired':
        return <ClockIcon className="h-4 w-4" />;
      case 'revoked':
        return <XCircleIcon className="h-4 w-4" />;
      default:
        return <FunnelIcon className="h-4 w-4" />;
    }
  };

  // Get badge variant for status
  const getStatusBadgeVariant = (
    status?: UpstreamCredentialStatus
  ): 'success' | 'warning' | 'error' | 'neutral' => {
    switch (status) {
      case 'configured':
        return 'success';
      case 'missing':
        return 'warning';
      case 'expired':
      case 'revoked':
        return 'error';
      default:
        return 'neutral';
    }
  };

  // Handle server configuration click
  const handleServerClick = (server: Server) => {
    const authType = server.upstream_auth?.type;

    // For API key and JWT types, open the modal
    if (authType === 'api-key' || authType === 'jwt') {
      setSelectedServer(server);
      setIsModalOpen(true);
    } else {
      // For other types (OAuth, OIDC, etc.), navigate to server details
      navigate(`/servers/${encodeURIComponent(server.path)}`);
    }
  };

  // Handle modal close
  const handleModalClose = () => {
    setIsModalOpen(false);
    setSelectedServer(null);
  };

  // Handle modal success (credential created/updated)
  const handleModalSuccess = () => {
    setIsModalOpen(false);
    setSelectedServer(null);
    // Refresh servers to get updated status
    fetchServers();
  };

  return (
    <>
      <PageHeader
        title="Upstream Credentials"
        description="Manage authentication credentials for upstream servers"
      />

      <PageContent>
        <div className="space-y-6">
          {/* Status Filter Bar */}
          <Card>
            <div className="p-4">
              <div className="flex items-center space-x-2 mb-3">
                <FunnelIcon className="h-5 w-5 text-gray-400" />
                <h3 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                  Filter by Status
                </h3>
              </div>
              <div className="flex flex-wrap gap-2">
                <Button
                  variant={statusFilter === 'all' ? 'primary' : 'ghost'}
                  size="sm"
                  leftIcon={getStatusIcon('all')}
                  onClick={() => setStatusFilter('all')}
                >
                  All
                  <Badge variant="neutral" size="sm" className="ml-2">
                    {statusCounts.all}
                  </Badge>
                </Button>
                <Button
                  variant={statusFilter === 'configured' ? 'primary' : 'ghost'}
                  size="sm"
                  leftIcon={getStatusIcon('configured')}
                  onClick={() => setStatusFilter('configured')}
                >
                  Configured
                  <Badge variant="success" size="sm" className="ml-2">
                    {statusCounts.configured}
                  </Badge>
                </Button>
                <Button
                  variant={statusFilter === 'missing' ? 'primary' : 'ghost'}
                  size="sm"
                  leftIcon={getStatusIcon('missing')}
                  onClick={() => setStatusFilter('missing')}
                >
                  Missing
                  <Badge variant="warning" size="sm" className="ml-2">
                    {statusCounts.missing}
                  </Badge>
                </Button>
                <Button
                  variant={statusFilter === 'expired' ? 'primary' : 'ghost'}
                  size="sm"
                  leftIcon={getStatusIcon('expired')}
                  onClick={() => setStatusFilter('expired')}
                >
                  Expired
                  <Badge variant="error" size="sm" className="ml-2">
                    {statusCounts.expired}
                  </Badge>
                </Button>
                <Button
                  variant={statusFilter === 'revoked' ? 'primary' : 'ghost'}
                  size="sm"
                  leftIcon={getStatusIcon('revoked')}
                  onClick={() => setStatusFilter('revoked')}
                >
                  Revoked
                  <Badge variant="error" size="sm" className="ml-2">
                    {statusCounts.revoked}
                  </Badge>
                </Button>
              </div>
            </div>
          </Card>

          {/* Servers List */}
          {loading ? (
            <div className="flex items-center justify-center py-12">
              <Spinner size="lg" />
            </div>
          ) : error ? (
            <Card>
              <EmptyState
                icon={<ExclamationCircleIcon className="w-12 h-12" />}
                title="Failed to load servers"
                description={error}
                action={
                  <Button variant="primary" onClick={fetchServers}>
                    Retry
                  </Button>
                }
              />
            </Card>
          ) : filteredServers.length === 0 ? (
            <Card>
              <EmptyState
                icon={<KeyIcon className="w-12 h-12" />}
                title={
                  statusFilter === 'all'
                    ? 'No servers with upstream authentication'
                    : `No ${statusFilter} credentials`
                }
                description={
                  statusFilter === 'all'
                    ? 'Register servers that require upstream authentication to manage credentials here'
                    : `There are no servers with ${statusFilter} upstream credentials`
                }
                action={
                  statusFilter !== 'all' ? (
                    <Button variant="ghost" onClick={() => setStatusFilter('all')}>
                      Clear Filter
                    </Button>
                  ) : undefined
                }
              />
            </Card>
          ) : (
            <div className="space-y-3">
              {filteredServers.map((server) => (
                <Card key={server.path} className="hover:shadow-md transition-shadow">
                  <div className="p-4">
                    <div className="flex items-start justify-between">
                      <div className="flex-1 min-w-0">
                        <div className="flex items-center space-x-3 mb-2">
                          <ServerIcon className="h-5 w-5 text-gray-400 flex-shrink-0" />
                          <h3 className="text-base font-semibold text-gray-900 dark:text-gray-100 truncate">
                            {server.display_name || server.path}
                          </h3>
                          {server.upstream_credential_status && (
                            <Badge
                              variant={getStatusBadgeVariant(server.upstream_credential_status)}
                              size="sm"
                            >
                              {server.upstream_credential_status}
                            </Badge>
                          )}
                        </div>
                        <div className="flex items-center space-x-4 mb-3">
                          <code className="text-sm text-gray-600 dark:text-gray-400">
                            {server.path}
                          </code>
                          {server.upstream_auth && (
                            <UpstreamAuthBadge
                              authType={server.upstream_auth.type}
                              credentialStatus={server.upstream_credential_status}
                              provider={server.upstream_auth.provider}
                              compact={true}
                            />
                          )}
                        </div>
                        {server.description && (
                          <p className="text-sm text-gray-600 dark:text-gray-400 mb-2">
                            {server.description}
                          </p>
                        )}
                        {server.upstream_auth && (
                          <div className="flex items-center space-x-4 text-xs text-gray-500 dark:text-gray-400">
                            <span>
                              Binding: {server.upstream_auth.credential_binding}
                            </span>
                            <span>Mode: {server.upstream_auth.mode}</span>
                          </div>
                        )}
                      </div>
                      <div className="ml-4 flex-shrink-0">
                        <Button
                          variant="primary"
                          size="sm"
                          onClick={() => handleServerClick(server)}
                        >
                          Configure
                        </Button>
                      </div>
                    </div>
                  </div>
                </Card>
              ))}
            </div>
          )}
        </div>
      </PageContent>

      {/* Upstream Credential Modal */}
      {selectedServer && (
        <UpstreamCredentialModal
          open={isModalOpen}
          server={selectedServer}
          onClose={handleModalClose}
          onSuccess={handleModalSuccess}
        />
      )}
    </>
  );
}
