import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  ArrowPathIcon,
  CheckCircleIcon,
  ExclamationCircleIcon,
  CloudIcon,
  ServerStackIcon,
  CpuChipIcon,
  SparklesIcon,
  PlusIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import AddRegistryEntryModal, { RegistrySourceType } from './AddRegistryEntryModal';
import ConfirmModal from './ConfirmModal';


/**
 * Props for the ExternalRegistries component.
 */
interface ExternalRegistriesProps {
  onShowToast: (message: string, type: 'success' | 'error' | 'info') => void;
}


/**
 * Anthropic server config shape.
 */
interface AnthropicServerConfig {
  name: string;
}


/**
 * Anthropic federation config shape.
 */
interface AnthropicConfig {
  enabled: boolean;
  endpoint: string;
  sync_on_startup: boolean;
  servers: AnthropicServerConfig[];
}


/**
 * ASOR agent config shape.
 */
interface AsorAgentConfig {
  id: string;
}


/**
 * ASOR federation config shape.
 */
interface AsorConfig {
  enabled: boolean;
  endpoint: string;
  auth_env_var: string | null;
  sync_on_startup: boolean;
  agents: AsorAgentConfig[];
}


/**
 * AgentCore registry config shape.
 */
interface AgentCoreRegistryConfig {
  registry_id: string;
  aws_account_id: string | null;
  aws_region: string | null;
  assume_role_arn: string | null;
  descriptor_types: string[];
  sync_status_filter: string;
}


/**
 * AgentCore federation config shape.
 */
interface AgentCoreConfig {
  enabled: boolean;
  aws_region: string;
  sync_on_startup: boolean;
  sync_interval_minutes: number;
  sync_timeout_seconds: number;
  max_concurrent_fetches: number;
  registries: AgentCoreRegistryConfig[];
}


/**
 * Root federation config shape.
 */
interface FederationConfig {
  anthropic: AnthropicConfig;
  asor: AsorConfig;
  aws_registry: AgentCoreConfig;
}


/**
 * Sync result shape from /api/federation/sync.
 */
interface SyncResults {
  anthropic: { count: number; servers: string[] };
  asor: { count: number; agents: string[] };
  aws_registry: { count: number; servers: string[]; agents: string[]; skills: string[] };
}


/**
 * Format a relative time string from an ISO timestamp.
 */
function _formatRelativeTime(dateString: string | null | undefined): string {
  if (!dateString) return 'Never';

  const date = new Date(dateString);
  const now = new Date();
  const diffMs = now.getTime() - date.getTime();
  const diffMins = Math.floor(diffMs / (1000 * 60));
  const diffHours = Math.floor(diffMs / (1000 * 60 * 60));
  const diffDays = Math.floor(diffMs / (1000 * 60 * 60 * 24));

  if (diffMins < 1) return 'Just now';
  if (diffMins < 60) return `${diffMins}m ago`;
  if (diffHours < 24) return `${diffHours}h ago`;
  if (diffDays < 7) return `${diffDays}d ago`;

  return date.toLocaleDateString();
}


/**
 * Truncate a string (like an ARN) for display.
 */
function _truncateArn(arn: string, maxLen: number = 60): string {
  if (arn.length <= maxLen) return arn;
  return arn.slice(0, maxLen - 3) + '...';
}


/**
 * ExternalRegistries displays the federation configuration for
 * Anthropic, AWS Agent Registry, and ASOR external registries.
 *
 * Shows config details, sync status, and provides a Sync Now button.
 */
const ExternalRegistries: React.FC<ExternalRegistriesProps> = ({ onShowToast }) => {
  const [config, setConfig] = useState<FederationConfig | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [syncing, setSyncing] = useState<string | null>(null);
  const [lastSyncTime, setLastSyncTime] = useState<string | null>(null);
  const [lastSyncResults, setLastSyncResults] = useState<SyncResults | null>(null);
  const [addModalSource, setAddModalSource] = useState<RegistrySourceType | null>(null);
  const [deletingItem, setDeletingItem] = useState<string | null>(null);
  const [confirmDelete, setConfirmDelete] = useState<{
    source: 'aws_registry' | 'anthropic' | 'asor';
    identifier: string;
  } | null>(null);

  /**
   * Fetch federation config from API.
   */
  const fetchConfig = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await axios.get('/api/federation/config');
      setConfig(response.data);
    } catch (err: any) {
      if (err?.response?.status === 404) {
        setConfig(null);
        setError(null);
      } else {
        setError('Failed to load federation configuration');
      }
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchConfig();
  }, [fetchConfig]);

  /**
   * Trigger a federation sync for a specific source.
   */
  const handleSync = async (source: string) => {
    setSyncing(source);
    try {
      const response = await axios.post(`/api/federation/sync?source=${source}`);
      const data = response.data;
      const totalSynced = data.total_synced || 0;
      setLastSyncTime(new Date().toISOString());
      setLastSyncResults(data.results || null);
      onShowToast(`Sync completed: ${totalSynced} items synced from ${source}`, 'success');
    } catch (err: any) {
      const detail = err?.response?.data?.detail || 'Sync failed';
      onShowToast(`Sync failed for ${source}: ${detail}`, 'error');
    } finally {
      setSyncing(null);
    }
  };

  /**
   * Trigger sync for all enabled sources.
   */
  const handleSyncAll = async () => {
    setSyncing('all');
    try {
      const response = await axios.post('/api/federation/sync');
      const data = response.data;
      const totalSynced = data.total_synced || 0;
      setLastSyncTime(new Date().toISOString());
      setLastSyncResults(data.results || null);
      onShowToast(`Sync completed: ${totalSynced} total items synced`, 'success');
    } catch (err: any) {
      const detail = err?.response?.data?.detail || 'Sync failed';
      onShowToast(`Sync failed: ${detail}`, 'error');
    } finally {
      setSyncing(null);
    }
  };

  /**
   * Show the confirm modal before deleting an entry.
   */
  const handleDeleteEntry = (
    source: 'aws_registry' | 'anthropic' | 'asor',
    identifier: string,
  ) => {
    setConfirmDelete({ source, identifier });
  };

  /**
   * Execute the deletion after user confirms via modal.
   */
  const executeDelete = async () => {
    if (!confirmDelete) return;

    const { source, identifier } = confirmDelete;
    setDeletingItem(identifier);
    try {
      if (source === 'anthropic') {
        await axios.delete(
          `/api/federation/config/default/anthropic/servers/${encodeURIComponent(identifier)}`
        );
      } else if (source === 'asor') {
        await axios.delete(
          `/api/federation/config/default/asor/agents/${encodeURIComponent(identifier)}`
        );
      } else if (source === 'aws_registry') {
        await axios.delete(
          `/api/federation/config/default/aws_registry/registries/${encodeURIComponent(identifier)}`
        );
      }
      onShowToast(`Removed "${identifier}"`, 'success');
      fetchConfig();
    } catch (err: any) {
      const detail = err?.response?.data?.detail || 'Failed to remove entry';
      onShowToast(detail, 'error');
    } finally {
      setDeletingItem(null);
      setConfirmDelete(null);
    }
  };

  /**
   * Called after successfully adding a new entry via the modal.
   */
  const handleAddSuccess = () => {
    fetchConfig();
  };

  // Loading state
  if (loading) {
    return (
      <div className="flex justify-center items-center py-20">
        <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-purple-600"></div>
      </div>
    );
  }

  // Error state
  if (error) {
    return (
      <div className="text-center py-12">
        <ExclamationCircleIcon className="mx-auto h-12 w-12 text-red-400" />
        <h3 className="mt-2 text-lg font-medium text-gray-900 dark:text-white">{error}</h3>
        <button
          onClick={fetchConfig}
          className="mt-4 px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700"
        >
          Retry
        </button>
      </div>
    );
  }

  // No config state
  if (!config) {
    return (
      <div className="text-center py-12">
        <CloudIcon className="mx-auto h-12 w-12 text-gray-400" />
        <h3 className="mt-2 text-lg font-medium text-gray-900 dark:text-white">
          No Federation Configuration
        </h3>
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          Federation configuration has not been set up yet.
          Use the CLI or API to create a federation config.
        </p>
      </div>
    );
  }

  // Count enabled sources
  const enabledSources: string[] = [];
  if (config.anthropic.enabled) enabledSources.push('anthropic');
  if (config.aws_registry.enabled) enabledSources.push('aws_registry');
  if (config.asor.enabled) enabledSources.push('asor');

  return (
    <div>
      {/* Header */}
      <div className="flex items-center justify-between mb-6">
        <div>
          <h2 className="text-xl font-semibold text-gray-900 dark:text-white">
            External Registries
          </h2>
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            {enabledSources.length} source{enabledSources.length !== 1 ? 's' : ''} configured
            {lastSyncTime && (
              <span className="ml-2">
                | Last sync: {_formatRelativeTime(lastSyncTime)}
              </span>
            )}
          </p>
        </div>
        <button
          onClick={handleSyncAll}
          disabled={syncing !== null || enabledSources.length === 0}
          className="inline-flex items-center px-4 py-2 border border-transparent text-sm
                     font-medium rounded-lg shadow-sm text-white bg-purple-600
                     hover:bg-purple-700 disabled:opacity-50 disabled:cursor-not-allowed
                     transition-colors"
        >
          <ArrowPathIcon className={`h-4 w-4 mr-2 ${syncing === 'all' ? 'animate-spin' : ''}`} />
          {syncing === 'all' ? 'Syncing...' : 'Sync All'}
        </button>
      </div>

      {/* Registry cards */}
      <div className="space-y-4">
        {/* AWS Agent Registry */}
        {_renderAgentCoreCard(
          config.aws_registry, syncing, lastSyncResults, handleSync,
          () => setAddModalSource('aws_registry'),
          (id) => handleDeleteEntry('aws_registry', id),
          deletingItem,
        )}

        {/* Anthropic */}
        {_renderAnthropicCard(
          config.anthropic, syncing, lastSyncResults, handleSync,
          () => setAddModalSource('anthropic'),
          (name) => handleDeleteEntry('anthropic', name),
          deletingItem,
        )}

        {/* ASOR */}
        {_renderAsorCard(
          config.asor, syncing, lastSyncResults, handleSync,
          () => setAddModalSource('asor'),
          (id) => handleDeleteEntry('asor', id),
          deletingItem,
        )}
      </div>

      {/* Add Entry Modal */}
      {addModalSource && (
        <AddRegistryEntryModal
          isOpen={true}
          onClose={() => setAddModalSource(null)}
          sourceType={addModalSource}
          onSuccess={handleAddSuccess}
          onShowToast={onShowToast}
        />
      )}

      {/* Delete Confirmation Modal */}
      {confirmDelete && (
        <ConfirmModal
          isOpen={true}
          onClose={() => setConfirmDelete(null)}
          onConfirm={executeDelete}
          title="Remove Entry"
          message={`Are you sure you want to remove "${confirmDelete.identifier}"? Any servers, agents, and skills synced from this source will also be deregistered.`}
          confirmLabel="Remove"
          isDestructive={true}
          isLoading={deletingItem !== null}
        />
      )}
    </div>
  );
};


/**
 * Render the AWS Agent Registry card.
 */
function _renderAgentCoreCard(
  agentcore: AgentCoreConfig,
  syncing: string | null,
  lastSyncResults: SyncResults | null,
  onSync: (source: string) => void,
  onAdd: () => void,
  onRemove: (registryId: string) => void,
  deletingItem: string | null,
): React.ReactNode {
  return (
    <div className={`border rounded-lg p-5 ${
      agentcore.enabled
        ? 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800'
        : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 opacity-60'
    }`}>
      {/* Card header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="flex-shrink-0 p-2 bg-orange-100 dark:bg-orange-900/30 rounded-lg">
            <CpuChipIcon className="h-5 w-5 text-orange-600 dark:text-orange-400" />
          </div>
          <div>
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">
              AWS Agent Registry
            </h3>
            <div className="flex items-center space-x-2 mt-0.5">
              <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                agentcore.enabled
                  ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
              }`}>
                {agentcore.enabled ? 'Enabled' : 'Disabled'}
              </span>
              {agentcore.sync_on_startup && (
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium
                                 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300">
                  Sync on startup
                </span>
              )}
            </div>
          </div>
        </div>
        {agentcore.enabled && (
          <div className="flex items-center space-x-2">
            <button
              onClick={onAdd}
              className="inline-flex items-center px-3 py-1.5 text-sm font-medium rounded-lg
                         border border-gray-300 dark:border-gray-600 text-gray-700
                         dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700
                         transition-colors"
            >
              <PlusIcon className="h-4 w-4 mr-1.5" />
              Add
            </button>
            <button
              onClick={() => onSync('aws_registry')}
              disabled={syncing !== null}
              className="inline-flex items-center px-3 py-1.5 text-sm font-medium rounded-lg
                         border border-gray-300 dark:border-gray-600 text-gray-700
                         dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700
                         disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ArrowPathIcon className={`h-4 w-4 mr-1.5 ${syncing === 'aws_registry' ? 'animate-spin' : ''}`} />
              {syncing === 'aws_registry' ? 'Syncing...' : 'Sync'}
            </button>
          </div>
        )}
      </div>

      {/* Config details */}
      {agentcore.enabled && (
        <div className="space-y-3">
          <div className="grid grid-cols-2 gap-4 text-sm">
            <div>
              <span className="text-gray-500 dark:text-gray-400">Region:</span>
              <span className="ml-2 text-gray-900 dark:text-white font-mono text-xs">
                {agentcore.aws_region}
              </span>
            </div>
            <div>
              <span className="text-gray-500 dark:text-gray-400">Sync interval:</span>
              <span className="ml-2 text-gray-900 dark:text-white">
                {agentcore.sync_interval_minutes} min
              </span>
            </div>
            <div>
              <span className="text-gray-500 dark:text-gray-400">Timeout:</span>
              <span className="ml-2 text-gray-900 dark:text-white">
                {agentcore.sync_timeout_seconds}s
              </span>
            </div>
            <div>
              <span className="text-gray-500 dark:text-gray-400">Concurrency:</span>
              <span className="ml-2 text-gray-900 dark:text-white">
                {agentcore.max_concurrent_fetches}
              </span>
            </div>
          </div>

          {/* Registry list */}
          {agentcore.registries.length > 0 && (
            <div className="mt-3">
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Registries ({agentcore.registries.length})
              </h4>
              <div className="space-y-2">
                {agentcore.registries.map((reg, idx) => (
                  <div
                    key={idx}
                    className="bg-gray-50 dark:bg-gray-900 rounded-lg p-3 border border-gray-100
                               dark:border-gray-700"
                  >
                    <div className="flex items-start justify-between">
                      <div className="font-mono text-xs text-gray-700 dark:text-gray-300 break-all">
                        {reg.registry_id}
                      </div>
                      <button
                        onClick={() => onRemove(reg.registry_id)}
                        disabled={deletingItem === reg.registry_id}
                        className="ml-2 flex-shrink-0 p-0.5 text-gray-400 hover:text-red-500
                                   dark:hover:text-red-400 disabled:opacity-50 transition-colors"
                        title="Remove registry"
                      >
                        <XMarkIcon className="h-4 w-4" />
                      </button>
                    </div>
                    <div className="flex flex-wrap gap-2 mt-2">
                      {reg.aws_region && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs
                                         bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400">
                          {reg.aws_region}
                        </span>
                      )}
                      {reg.aws_account_id && (
                        <span className="inline-flex items-center px-2 py-0.5 rounded text-xs
                                         bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400">
                          Account: {reg.aws_account_id}
                        </span>
                      )}
                      <span className="inline-flex items-center px-2 py-0.5 rounded text-xs
                                       bg-gray-200 dark:bg-gray-700 text-gray-600 dark:text-gray-400">
                        Status: {reg.sync_status_filter}
                      </span>
                      {reg.descriptor_types.map((dt) => (
                        <span
                          key={dt}
                          className="inline-flex items-center px-2 py-0.5 rounded text-xs
                                     bg-purple-100 dark:bg-purple-900/30 text-purple-700
                                     dark:text-purple-300"
                        >
                          {dt}
                        </span>
                      ))}
                    </div>
                    {reg.assume_role_arn && (
                      <div className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                        Role: <span className="font-mono">{_truncateArn(reg.assume_role_arn)}</span>
                      </div>
                    )}
                  </div>
                ))}
              </div>
            </div>
          )}

          {/* Last sync results */}
          {lastSyncResults?.aws_registry && lastSyncResults.aws_registry.count > 0 && (
            <div className="mt-3 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border
                            border-green-200 dark:border-green-800">
              <div className="flex items-center space-x-2">
                <CheckCircleIcon className="h-4 w-4 text-green-600 dark:text-green-400" />
                <span className="text-sm font-medium text-green-800 dark:text-green-300">
                  Last sync: {lastSyncResults.aws_registry.count} items
                </span>
              </div>
              <div className="mt-1 text-xs text-green-700 dark:text-green-400">
                {lastSyncResults.aws_registry.servers.length > 0 && (
                  <span>Servers: {lastSyncResults.aws_registry.servers.length} </span>
                )}
                {lastSyncResults.aws_registry.agents.length > 0 && (
                  <span>Agents: {lastSyncResults.aws_registry.agents.length} </span>
                )}
                {lastSyncResults.aws_registry.skills.length > 0 && (
                  <span>Skills: {lastSyncResults.aws_registry.skills.length}</span>
                )}
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}


/**
 * Render the Anthropic registry card.
 */
function _renderAnthropicCard(
  anthropic: AnthropicConfig,
  syncing: string | null,
  lastSyncResults: SyncResults | null,
  onSync: (source: string) => void,
  onAdd: () => void,
  onRemove: (serverName: string) => void,
  deletingItem: string | null,
): React.ReactNode {
  return (
    <div className={`border rounded-lg p-5 ${
      anthropic.enabled
        ? 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800'
        : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 opacity-60'
    }`}>
      {/* Card header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="flex-shrink-0 p-2 bg-purple-100 dark:bg-purple-900/30 rounded-lg">
            <SparklesIcon className="h-5 w-5 text-purple-600 dark:text-purple-400" />
          </div>
          <div>
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">
              Anthropic
            </h3>
            <div className="flex items-center space-x-2 mt-0.5">
              <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                anthropic.enabled
                  ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
              }`}>
                {anthropic.enabled ? 'Enabled' : 'Disabled'}
              </span>
              {anthropic.sync_on_startup && (
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium
                                 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300">
                  Sync on startup
                </span>
              )}
            </div>
          </div>
        </div>
        {anthropic.enabled && (
          <div className="flex items-center space-x-2">
            <button
              onClick={onAdd}
              className="inline-flex items-center px-3 py-1.5 text-sm font-medium rounded-lg
                         border border-gray-300 dark:border-gray-600 text-gray-700
                         dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700
                         transition-colors"
            >
              <PlusIcon className="h-4 w-4 mr-1.5" />
              Add
            </button>
            <button
              onClick={() => onSync('anthropic')}
              disabled={syncing !== null}
              className="inline-flex items-center px-3 py-1.5 text-sm font-medium rounded-lg
                         border border-gray-300 dark:border-gray-600 text-gray-700
                         dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700
                         disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ArrowPathIcon className={`h-4 w-4 mr-1.5 ${syncing === 'anthropic' ? 'animate-spin' : ''}`} />
              {syncing === 'anthropic' ? 'Syncing...' : 'Sync'}
            </button>
          </div>
        )}
      </div>

      {/* Config details */}
      {anthropic.enabled && (
        <div className="space-y-3">
          <div className="text-sm">
            <span className="text-gray-500 dark:text-gray-400">Endpoint:</span>
            <span className="ml-2 text-gray-900 dark:text-white font-mono text-xs">
              {anthropic.endpoint}
            </span>
          </div>

          {/* Server list */}
          {anthropic.servers.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Servers ({anthropic.servers.length})
              </h4>
              <div className="flex flex-wrap gap-2">
                {anthropic.servers.map((srv) => (
                  <span
                    key={srv.name}
                    className="inline-flex items-center px-2.5 py-1 rounded-lg text-xs font-mono
                               bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300
                               border border-gray-200 dark:border-gray-600"
                  >
                    <ServerStackIcon className="h-3.5 w-3.5 mr-1.5 text-gray-400" />
                    {srv.name}
                    <button
                      onClick={() => onRemove(srv.name)}
                      disabled={deletingItem === srv.name}
                      className="ml-1.5 text-gray-400 hover:text-red-500 dark:hover:text-red-400
                                 disabled:opacity-50 transition-colors"
                      title="Remove server"
                    >
                      <XMarkIcon className="h-3.5 w-3.5" />
                    </button>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Last sync results */}
          {lastSyncResults?.anthropic && lastSyncResults.anthropic.count > 0 && (
            <div className="mt-3 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border
                            border-green-200 dark:border-green-800">
              <div className="flex items-center space-x-2">
                <CheckCircleIcon className="h-4 w-4 text-green-600 dark:text-green-400" />
                <span className="text-sm font-medium text-green-800 dark:text-green-300">
                  Last sync: {lastSyncResults.anthropic.count} servers
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}


/**
 * Render the ASOR registry card.
 */
function _renderAsorCard(
  asor: AsorConfig,
  syncing: string | null,
  lastSyncResults: SyncResults | null,
  onSync: (source: string) => void,
  onAdd: () => void,
  onRemove: (agentId: string) => void,
  deletingItem: string | null,
): React.ReactNode {
  return (
    <div className={`border rounded-lg p-5 ${
      asor.enabled
        ? 'border-gray-200 dark:border-gray-700 bg-white dark:bg-gray-800'
        : 'border-gray-200 dark:border-gray-700 bg-gray-50 dark:bg-gray-900 opacity-60'
    }`}>
      {/* Card header */}
      <div className="flex items-center justify-between mb-4">
        <div className="flex items-center space-x-3">
          <div className="flex-shrink-0 p-2 bg-blue-100 dark:bg-blue-900/30 rounded-lg">
            <GlobeIcon className="h-5 w-5 text-blue-600 dark:text-blue-400" />
          </div>
          <div>
            <h3 className="text-lg font-medium text-gray-900 dark:text-white">
              ASOR
            </h3>
            <div className="flex items-center space-x-2 mt-0.5">
              <span className={`inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium ${
                asor.enabled
                  ? 'bg-green-100 dark:bg-green-900/30 text-green-800 dark:text-green-300'
                  : 'bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-400'
              }`}>
                {asor.enabled ? 'Enabled' : 'Disabled'}
              </span>
              {asor.sync_on_startup && (
                <span className="inline-flex items-center px-2 py-0.5 rounded-full text-xs font-medium
                                 bg-blue-100 dark:bg-blue-900/30 text-blue-800 dark:text-blue-300">
                  Sync on startup
                </span>
              )}
            </div>
          </div>
        </div>
        {asor.enabled && (
          <div className="flex items-center space-x-2">
            <button
              onClick={onAdd}
              className="inline-flex items-center px-3 py-1.5 text-sm font-medium rounded-lg
                         border border-gray-300 dark:border-gray-600 text-gray-700
                         dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700
                         transition-colors"
            >
              <PlusIcon className="h-4 w-4 mr-1.5" />
              Add
            </button>
            <button
              onClick={() => onSync('asor')}
              disabled={syncing !== null}
              className="inline-flex items-center px-3 py-1.5 text-sm font-medium rounded-lg
                         border border-gray-300 dark:border-gray-600 text-gray-700
                         dark:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700
                         disabled:opacity-50 disabled:cursor-not-allowed transition-colors"
            >
              <ArrowPathIcon className={`h-4 w-4 mr-1.5 ${syncing === 'asor' ? 'animate-spin' : ''}`} />
              {syncing === 'asor' ? 'Syncing...' : 'Sync'}
            </button>
          </div>
        )}
      </div>

      {/* Config details */}
      {asor.enabled && (
        <div className="space-y-3">
          {asor.endpoint && (
            <div className="text-sm">
              <span className="text-gray-500 dark:text-gray-400">Endpoint:</span>
              <span className="ml-2 text-gray-900 dark:text-white font-mono text-xs">
                {asor.endpoint}
              </span>
            </div>
          )}

          {asor.agents.length > 0 && (
            <div>
              <h4 className="text-sm font-medium text-gray-700 dark:text-gray-300 mb-2">
                Agents ({asor.agents.length})
              </h4>
              <div className="flex flex-wrap gap-2">
                {asor.agents.map((agent) => (
                  <span
                    key={agent.id}
                    className="inline-flex items-center px-2.5 py-1 rounded-lg text-xs font-mono
                               bg-gray-100 dark:bg-gray-700 text-gray-700 dark:text-gray-300
                               border border-gray-200 dark:border-gray-600"
                  >
                    {agent.id}
                    <button
                      onClick={() => onRemove(agent.id)}
                      disabled={deletingItem === agent.id}
                      className="ml-1.5 text-gray-400 hover:text-red-500 dark:hover:text-red-400
                                 disabled:opacity-50 transition-colors"
                      title="Remove agent"
                    >
                      <XMarkIcon className="h-3.5 w-3.5" />
                    </button>
                  </span>
                ))}
              </div>
            </div>
          )}

          {/* Last sync results */}
          {lastSyncResults?.asor && lastSyncResults.asor.count > 0 && (
            <div className="mt-3 p-3 bg-green-50 dark:bg-green-900/20 rounded-lg border
                            border-green-200 dark:border-green-800">
              <div className="flex items-center space-x-2">
                <CheckCircleIcon className="h-4 w-4 text-green-600 dark:text-green-400" />
                <span className="text-sm font-medium text-green-800 dark:text-green-300">
                  Last sync: {lastSyncResults.asor.count} agents
                </span>
              </div>
            </div>
          )}
        </div>
      )}
    </div>
  );
}


/**
 * Simple globe icon wrapper (GlobeAltIcon from heroicons).
 */
function GlobeIcon(props: React.ComponentProps<'svg'>) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      fill="none"
      viewBox="0 0 24 24"
      strokeWidth={1.5}
      stroke="currentColor"
      {...props}
    >
      <path
        strokeLinecap="round"
        strokeLinejoin="round"
        d="M12 21a9.004 9.004 0 008.716-6.747M12 21a9.004 9.004 0 01-8.716-6.747M12 21c2.485 0
           4.5-4.03 4.5-9S14.485 3 12 3m0 18c-2.485 0-4.5-4.03-4.5-9S9.515 3 12 3m0 0a8.997
           8.997 0 017.843 4.582M12 3a8.997 8.997 0 00-7.843 4.582m15.686 0A11.953 11.953 0 0112
           10.5c-2.998 0-5.74-1.1-7.843-2.918m15.686 0A8.959 8.959 0 0121 12c0 .778-.099
           1.533-.284 2.253m0 0A17.919 17.919 0 0112 16.5c-3.162 0-6.133-.815-8.716-2.247m0
           0A9.015 9.015 0 013 12c0-1.605.42-3.113 1.157-4.418"
      />
    </svg>
  );
}


export default ExternalRegistries;
