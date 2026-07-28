import React, { useState, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import axios from 'axios';
import {
  WrenchScrewdriverIcon,
  ArrowPathIcon,
  PencilIcon,
  ClockIcon,
  LinkIcon,
  ShieldCheckIcon,
  ShieldExclamationIcon,
  TrashIcon,
  InformationCircleIcon,
} from '@heroicons/react/24/outline';
import PlugIcon from './icons/PlugIcon';
import { isSafeUrl } from '../utils/safeUrl';
import type { EgressCardState } from '../utils/egressAuth';
import { useEgressConnect } from '../contexts/EgressConnectContext';
import ServerConfigModal from './ServerConfigModal';
import SecurityScanModal from './SecurityScanModal';
import StarRatingWidget from './StarRatingWidget';
import VersionBadge from './VersionBadge';
import VersionSelectorModal from './VersionSelectorModal';
import ConfirmModal from './ConfirmModal';
import StatusBadge from './StatusBadge';
import Badge from './Badge';
import { ANSBadge } from './ANSBadge';
import ServerDetailsModal from './ServerDetailsModal';
import { SafeLink } from './SafeLink';
import useEscapeKey from '../hooks/useEscapeKey';
import { formatRelativeTime, formatTimeSince } from '../utils/dateUtils';
import { normalizeHealthStatus } from '../utils/healthStatus';
import { useAuth } from '../contexts/AuthContext';
import { toScanSummary } from '../utils/securityScan';
import type { LocalRuntime } from '../types/server';
import {
  CardShell,
  CardHeader,
  CardBody,
  CardStatsRow,
  CardFooter,
  StatusDot,
  StatusDivider,
  TagList,
  ToggleSwitch,
  InlineDeleteConfirm,
  ACCENTS,
  ENTITY_ACCENTS,
  type StatusTone,
} from './cards';

const ACCENT = ENTITY_ACCENTS.server;

/** Map a server's health/local state to a StatusDot tone + label. */
function serverHealthDot(
  status: string | undefined,
  isLocal: boolean,
): { tone: StatusTone; label: string; title?: string } {
  if (isLocal) {
    return {
      tone: 'emerald',
      label: 'Local',
      title:
        'Runs on the developer’s machine via stdio launch recipe — registry does not health-check',
    };
  }
  switch (status) {
    case 'healthy':
      return { tone: 'emerald', label: 'Healthy' };
    case 'healthy-auth-expired':
      return { tone: 'orange', label: 'Healthy (Auth Expired)' };
    case 'unhealthy':
      return { tone: 'red', label: 'Unhealthy' };
    default:
      return { tone: 'amber', label: 'Unknown' };
  }
}

interface ServerVersion {
  version: string;
  proxy_pass_url: string;
  status: string;
  is_default: boolean;
  released?: string;
  sunset_date?: string;
  description?: string;
}

interface SyncMetadata {
  is_federated?: boolean;
  source_peer_id?: string;
  upstream_path?: string;
  last_synced_at?: string;
  is_read_only?: boolean;
  is_orphaned?: boolean;
  orphaned_at?: string;
}

export interface Server {
  name: string;
  path: string;
  description?: string;
  official?: boolean;
  enabled: boolean;
  tags?: string[];
  last_checked_time?: string;
  usersCount?: number;
  rating?: number;
  rating_details?: Array<{ user: string; rating: number }>;
  // Lightweight scan summary from the list payload, used to colour the shield
  // icon without a per-card /security-scan fetch. Undefined if not yet scanned.
  security_scan?: {
    scan_failed?: boolean;
    critical_issues?: number;
    high_severity?: number;
    medium_severity?: number;
    low_severity?: number;
  } | null;
  status?: 'healthy' | 'healthy-auth-expired' | 'unhealthy' | 'unknown' | 'local';
  num_tools?: number;
  proxy_pass_url?: string;
  mcp_endpoint?: string;
  // Local-server fields
  deployment?: 'remote' | 'local';
  local_runtime?: LocalRuntime;
  registered_by?: string | null;
  // Version routing fields
  version?: string;  // Current active version
  versions?: ServerVersion[];
  default_version?: string;
  // MCP server info from initialize response
  mcp_server_version?: string;
  mcp_server_version_previous?: string;
  mcp_server_version_updated_at?: string;
  // Federation sync metadata
  sync_metadata?: SyncMetadata;
  // ARD discovery-only import: URL to the original server.json on the source registry
  ard_source_url?: string;
  // Backend authentication
  auth_scheme?: string;
  auth_header_name?: string;
  custom_header_names?: string[];
  // Lifecycle status
  lifecycle_status?: 'active' | 'deprecated' | 'draft' | 'beta';
  source_created_at?: string;
  source_updated_at?: string;
  // ANS Integration
  ans_metadata?: {
    ans_agent_id: string;
    status: 'verified' | 'expired' | 'revoked' | 'not_found' | 'pending';
    domain?: string;
    organization?: string;
    certificate?: {
      not_after?: string;
      subject_dn?: string;
      issuer_dn?: string;
    };
    last_verified?: string;
  };
}

interface ServerCardProps {
  server: Server;
  onToggle: (path: string, enabled: boolean) => void;
  onEdit?: (server: Server) => void;
  canModify?: boolean;
  canHealthCheck?: boolean;
  canToggle?: boolean;
  canDelete?: boolean;
  onRefreshSuccess?: () => void;
  onShowToast?: (message: string, type: 'success' | 'error') => void;
  onServerUpdate?: (path: string, updates: Partial<Server>) => void;
  onDelete?: (path: string) => Promise<void>;
  authToken?: string | null;
  // Per-user egress state for this server (undefined => not eligible / no icon).
  egressConnect?: EgressCardState;
  // Called after a connect/disconnect in the modal so the dashboard refreshes.
  onEgressChanged?: () => void;
}

interface Tool {
  name: string;
  description?: string;
  schema?: any;
}


/**
 * Compact per-server egress affordance in the card header. Reflects three
 * states (not-connected / connected / needs-reconnect) with a mode-aware label,
 * from the shared EgressCardState so the card and the connect-modal callout
 * never disagree.
 */
function EgressConnectButton({
  serverPath,
  serverName,
  state,
  onShowToast,
}: {
  serverPath: string;
  serverName: string;
  state: EgressCardState;
  onShowToast?: (message: string, type: 'success' | 'error') => void;
}) {
  const navigate = useNavigate();
  const connected = state.connected;

  let label: string;
  let tooltip: string;
  if (state.needsReconnect) {
    label = 'Reconnect';
    tooltip = 'Connection expired or failed — reconnect';
  } else if (connected) {
    label = 'Connected';
    tooltip = 'Account connected — manage on Connected Accounts';
  } else if (state.mode === 'pat') {
    label = 'Submit token';
    tooltip = 'Submit a personal access token for this server';
  } else {
    label = 'Link account';
    tooltip = 'Link your account for this server';
  }

  const handleClick = () => {
    // PAT cannot use an OAuth redirect, and a healthy connected 3LO just needs
    // management: route both to Connected Accounts (server preselected).
    if (state.mode === 'pat' || (connected && !state.needsReconnect)) {
      navigate(`/connected-accounts?server=${encodeURIComponent(serverPath)}`);
      return;
    }
    // Not-connected or needs-reconnect 3LO: open the gateway front door.
    // MANDATORY isSafeUrl guard before window.open (defense in depth): connect_url
    // is server-built but is technically influenceable via a crafted admin path,
    // so never open it unguarded. Fail closed on unsafe/empty.
    if (!isSafeUrl(state.connectUrl)) {
      onShowToast?.('Cannot open the connect URL for this server.', 'error');
      return;
    }
    window.open(state.connectUrl, '_blank', 'noopener,noreferrer');
  };

  const iconClass = state.needsReconnect
    ? 'h-4 w-4 text-amber-500'
    : connected
      ? 'h-4 w-4 text-green-600 dark:text-green-400'
      : 'h-4 w-4 text-gray-400';

  return (
    <button
      type="button"
      onClick={handleClick}
      className="p-2 hover:bg-gray-50 dark:hover:bg-gray-700/50 rounded-lg transition-all duration-200 flex-shrink-0"
      title={tooltip}
      aria-label={`${label} for ${serverName}`}
    >
      <PlugIcon className={iconClass} />
    </button>
  );
}

const ServerCard: React.FC<ServerCardProps> = React.memo(({ server, onToggle, onEdit, canModify, canHealthCheck = true, canToggle = true, canDelete, onRefreshSuccess, onShowToast, onServerUpdate, onDelete, authToken, egressConnect: egressConnectProp, onEgressChanged: onEgressChangedProp }) => {
  const { user } = useAuth();
  const isAdmin = user?.is_admin === true;
  // Egress connect state: prefer the explicit prop (dashboard grid passes it),
  // else fall back to the shared context so render paths that don't thread the
  // prop (discover/search rows via DiscoverListRow) still show the affordance.
  const egressCtx = useEgressConnect();
  const egressConnect = egressConnectProp ?? egressCtx.stateByPath.get(server.path);
  const onEgressChanged = onEgressChangedProp ?? egressCtx.reload;
  const [tools, setTools] = useState<Tool[]>([]);
  const [loadingTools, setLoadingTools] = useState(false);
  const [showTools, setShowTools] = useState(false);
  const [showConfig, setShowConfig] = useState(false);
  const [loadingRefresh, setLoadingRefresh] = useState(false);
  const [showSecurityScan, setShowSecurityScan] = useState(false);
  // Seed from the list payload's lightweight scan summary so the shield icon
  // colours correctly with no per-card fetch. The on-click handler upgrades this
  // to the full scan document (analysis_results, tool_results) for the modal.
  const [securityScanResult, setSecurityScanResult] = useState<any>(server.security_scan ?? null);
  const [loadingSecurityScan, setLoadingSecurityScan] = useState(false);
  const [showVersionSelector, setShowVersionSelector] = useState(false);
  const [showDeleteConfirm, setShowDeleteConfirm] = useState(false);
  const [showDetails, setShowDetails] = useState(false);
  const [fullServerDetails, setFullServerDetails] = useState<any>(null);
  const [detailsLoading, setDetailsLoading] = useState(false);
  const [expandedDescriptions, setExpandedDescriptions] = useState<Set<number>>(new Set());
  const [showClearSecurityPendingConfirm, setShowClearSecurityPendingConfirm] = useState(false);
  const [clearingSecurityPending, setClearingSecurityPending] = useState(false);

  const closeToolsModal = useCallback(() => {
    setShowTools(false);
    setExpandedDescriptions(new Set());
  }, []);
  useEscapeKey(closeToolsModal, showTools);
  useEscapeKey(() => setShowDeleteConfirm(false), showDeleteConfirm);

  // Keep the icon in sync with the list payload's scan summary. No fetch: the
  // summary (scan_failed + severity counts) arrives inline on /api/servers, so a
  // page of cards costs zero extra requests instead of one /security-scan each.
  // Skip when the user has already opened the full detail (a richer object).
  //
  // Invariant: handleRescan MUST update the parent (onServerUpdate) synchronously
  // so server.security_scan is fresh by the time the modal closes and this effect
  // re-syncs. React 18 batches the rescan's state updates, so the prop is current
  // on the next render; wrapping the modal close in setTimeout would break this.
  useEffect(() => {
    if (!showSecurityScan) {
      setSecurityScanResult(server.security_scan ?? null);
    }
  }, [server.security_scan, showSecurityScan]);

  const handleViewTools = useCallback(async () => {
    if (loadingTools) return;

    setLoadingTools(true);
    try {
      const response = await axios.get(`/api/tools${server.path}`);
      setTools(response.data.tools || []);
      setShowTools(true);
    } catch (error) {
      console.error('Failed to fetch tools:', error);
      if (onShowToast) {
        onShowToast('Failed to fetch tools', 'error');
      }
    } finally {
      setLoadingTools(false);
    }
  }, [server.path, loadingTools, onShowToast]);

  const handleRefreshHealth = useCallback(async () => {
    if (loadingRefresh) return;

    setLoadingRefresh(true);
    try {
      // Extract service name from path (remove leading slash)
      const serviceName = server.path.replace(/^\//, '');

      const response = await axios.post(`/api/refresh/${serviceName}`);

      // Update just this server instead of triggering global refresh
      if (onServerUpdate && response.data) {
        const updates: Partial<Server> = {
          status: normalizeHealthStatus(response.data.status),
          last_checked_time: response.data.last_checked_iso,
          num_tools: response.data.num_tools
        };

        onServerUpdate(server.path, updates);
      } else if (onRefreshSuccess) {
        // Fallback to global refresh if onServerUpdate is not provided
        onRefreshSuccess();
      }

      if (onShowToast) {
        onShowToast('Health status refreshed successfully', 'success');
      }
    } catch (error: any) {
      console.error('Failed to refresh health:', error);
      if (onShowToast) {
        onShowToast(error.response?.data?.detail || 'Failed to refresh health status', 'error');
      }
    } finally {
      setLoadingRefresh(false);
    }
  }, [server.path, loadingRefresh, onRefreshSuccess, onShowToast, onServerUpdate]);

  const handleViewSecurityScan = useCallback(async () => {
    if (loadingSecurityScan) return;

    setShowSecurityScan(true);
    setLoadingSecurityScan(true);
    try {
      const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
      const response = await axios.get(
        `/api/servers${server.path}/security-scan`,
        headers ? { headers } : undefined
      );
      setSecurityScanResult(response.data);
    } catch (error: any) {
      if (error.response?.status !== 404) {
        console.error('Failed to fetch security scan:', error);
        if (onShowToast) {
          onShowToast('Failed to load security scan results', 'error');
        }
      }
      setSecurityScanResult(null);
    } finally {
      setLoadingSecurityScan(false);
    }
  }, [server.path, authToken, loadingSecurityScan, onShowToast]);

  const handleRescan = useCallback(async () => {
    const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
    const response = await axios.post(
      `/api/servers${server.path}/rescan`,
      undefined,
      headers ? { headers } : undefined
    );
    // Show the full result in the open modal, and push the lightweight summary
    // up so server.security_scan (the list entry) reflects the new scan. Without
    // this the prop-sync effect would revert the badge to the stale list value
    // when the modal closes.
    setSecurityScanResult(response.data);
    onServerUpdate?.(server.path, { security_scan: toScanSummary(response.data) });
  }, [server.path, authToken, onServerUpdate]);

  const handleRefreshServerData = useCallback(async () => {
    try {
      const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
      const response = await axios.get(
        `/api/server_details${server.path}`,
        headers ? { headers } : undefined
      );

      if (onServerUpdate && response.data) {
        const serverData = response.data;
        const updates: Partial<Server> = {
          name: serverData.server_name,
          description: serverData.description,
          enabled: serverData.is_enabled,
          tags: serverData.tags,
          status: normalizeHealthStatus(serverData.health_status),
          last_checked_time: serverData.last_checked_iso,
          num_tools: serverData.num_tools,
          proxy_pass_url: serverData.proxy_pass_url,
          mcp_endpoint: serverData.mcp_endpoint,
          version: serverData.version,
          versions: serverData.versions,
          default_version: serverData.default_version,
          mcp_server_version: serverData.mcp_server_version,
          mcp_server_version_previous: serverData.mcp_server_version_previous,
          mcp_server_version_updated_at: serverData.mcp_server_version_updated_at,
        };
        onServerUpdate(server.path, updates);
      }
    } catch (error) {
      console.error('Failed to refresh server data:', error);
    }
  }, [server.path, authToken, onServerUpdate]);

  const handleClearSecurityPending = useCallback(async () => {
    if (clearingSecurityPending) return;
    setClearingSecurityPending(true);
    try {
      const cleanPath = server.path.replace(/^\/+/, '');
      await axios.post(`/api/clear-security-pending-local/${cleanPath}`);
      onShowToast?.('Marked as security-reviewed', 'success');
      if (onServerUpdate) {
        onServerUpdate(server.path, {
          tags: (server.tags || []).filter(t => t !== 'security-pending-local'),
        });
      }
      setShowClearSecurityPendingConfirm(false);
    } catch (err: unknown) {
      const e = err as { response?: { data?: { detail?: string } } };
      onShowToast?.(e.response?.data?.detail || 'Failed to clear tag', 'error');
    } finally {
      setClearingSecurityPending(false);
    }
  }, [server.path, server.tags, clearingSecurityPending, onServerUpdate, onShowToast]);

  const getSecurityIconState = () => {
    // Local (stdio) servers can't be auto-scanned. Their security state is
    // signalled by the security-pending-local tag instead:
    //   tag present  → amber (admin manual review still owed)
    //   tag absent   → green (admin has reviewed and cleared)
    if (server.deployment === 'local') {
      if (server.tags?.includes('security-pending-local')) {
        return {
          Icon: ShieldExclamationIcon,
          color: 'text-amber-500 dark:text-amber-400',
          title: 'Pending manual security review (local server)',
        };
      }
      return {
        Icon: ShieldCheckIcon,
        color: 'text-green-500 dark:text-green-400',
        title: 'Local server marked as security-reviewed',
      };
    }

    // Gray: no scan result yet
    if (!securityScanResult) {
      return { Icon: ShieldCheckIcon, color: 'text-gray-400 dark:text-gray-500', title: 'View security scan results' };
    }
    // Red: scan failed or any vulnerabilities found
    if (securityScanResult.scan_failed) {
      return { Icon: ShieldExclamationIcon, color: 'text-red-500 dark:text-red-400', title: 'Security scan failed' };
    }
    const hasVulnerabilities = securityScanResult.critical_issues > 0 ||
      securityScanResult.high_severity > 0 ||
      securityScanResult.medium_severity > 0 ||
      securityScanResult.low_severity > 0;
    if (hasVulnerabilities) {
      return { Icon: ShieldExclamationIcon, color: 'text-red-500 dark:text-red-400', title: 'Security issues found' };
    }
    // Green: scan passed with no vulnerabilities
    return { Icon: ShieldCheckIcon, color: 'text-green-500 dark:text-green-400', title: 'Security scan passed' };
  };

  // Generate MCP configuration for the server
  // Check if this is an Anthropic registry server
  const isAnthropicServer = server.tags?.includes('anthropic-registry');

  // Main renders Anthropic-registry servers with the purple "primary" accent and
  // all other servers fully neutral (white shell, gray footer, blue tags). Pick
  // the per-card accent here so every accent-driven surface follows that rule.
  const cardAccent = isAnthropicServer ? ACCENT : 'neutral';

  // Check if this server has security pending
  const isSecurityPending = server.tags?.includes('security-pending');
  const isSecurityPendingLocal = server.tags?.includes('security-pending-local');

  // Local (stdio) deployment — no HTTP endpoint, so health/scan paths skip it.
  const isLocal = server.deployment === 'local';

  // Check if this is a federated server from a peer registry using sync_metadata
  const isFederatedServer = server.sync_metadata?.is_federated === true;
  const peerRegistryId = isFederatedServer && server.sync_metadata?.source_peer_id
    ? server.sync_metadata.source_peer_id
    : null;

  // Check if this server is orphaned (no longer exists on peer registry)
  const isOrphanedServer = server.sync_metadata?.is_orphaned === true;

  // Check if this is an ARD discovery-only import. The public ai-catalog.json
  // gives metadata only (no tools, no proxy URL), so these are read-only and
  // non-connectable. Detected via the 'ard' tag or the read-only sync flag.
  const isArdDiscovery =
    (server.tags || []).includes('ard') || server.sync_metadata?.is_read_only === true;
  // Link back to the original server.json on the source registry, if published.
  const ardSourceUrl = server.ard_source_url || server.sync_metadata?.upstream_path;

  const health = serverHealthDot(server.status, isLocal);

  return (
    <>
      <CardShell accent={cardAccent}>
        {/* Render DeleteConfirmation inline when showDeleteConfirm is true */}
        {showDeleteConfirm ? (
          <InlineDeleteConfirm
            entityType="server"
            entityName={server.name || server.path.replace(/^\//, '')}
            entityPath={server.path}
            onConfirm={onDelete!}
            onCancel={() => setShowDeleteConfirm(false)}
          />
        ) : (
        <>
        <CardHeader
          title={server.name}
          path={server.path}
          wrapBadges
          minTitleWidth
          badges={
            <>
                {server.lifecycle_status && server.lifecycle_status !== 'active' && (
                  <StatusBadge status={server.lifecycle_status} />
                )}
                {server.deployment === 'local' && (
                  <span
                    className="px-2 py-0.5 text-xs font-semibold bg-gradient-to-r from-emerald-100 to-teal-100 text-emerald-700 dark:from-emerald-900/30 dark:to-teal-900/30 dark:text-emerald-300 rounded-full flex-shrink-0 border border-emerald-200 dark:border-emerald-600"
                    title="Local (stdio) — runs on your machine via launch recipe"
                  >
                    LOCAL
                  </span>
                )}
                {server.official && (
                  <Badge tone="purple">OFFICIAL</Badge>
                )}
                {isAnthropicServer && (
                  <span className="px-2 py-0.5 text-xs font-semibold bg-gradient-to-r from-purple-100 to-indigo-100 text-purple-700 dark:from-purple-900/30 dark:to-indigo-900/30 dark:text-purple-300 rounded-full flex-shrink-0 border border-purple-200 dark:border-purple-600">
                    ANTHROPIC
                  </span>
                )}
                {/* Check if this is an ASOR server */}
                {server.tags?.includes('asor') && (
                  <span className="px-2 py-0.5 text-xs font-semibold bg-gradient-to-r from-orange-100 to-red-100 text-orange-700 dark:from-orange-900/30 dark:to-red-900/30 dark:text-orange-300 rounded-full flex-shrink-0 border border-orange-200 dark:border-orange-600">
                    ASOR
                  </span>
                )}
                {isSecurityPending && (
                  <span className="px-2 py-0.5 text-xs font-semibold bg-gradient-to-r from-amber-100 to-orange-100 text-amber-700 dark:from-amber-900/30 dark:to-orange-900/30 dark:text-amber-300 rounded-full flex-shrink-0 border border-amber-200 dark:border-amber-600">
                    SECURITY PENDING
                  </span>
                )}
                {isSecurityPendingLocal && (
                  isAdmin ? (
                    <button
                      type="button"
                      onClick={() => setShowClearSecurityPendingConfirm(true)}
                      className="px-2 py-0.5 text-xs font-semibold bg-gradient-to-r from-amber-100 to-orange-100 text-amber-700 dark:from-amber-900/30 dark:to-orange-900/30 dark:text-amber-300 rounded-full flex-shrink-0 border border-amber-200 dark:border-amber-600 hover:from-amber-200 hover:to-orange-200 transition"
                      title="Click to mark as security-reviewed (admin only)"
                    >
                      SECURITY PENDING (LOCAL) ×
                    </button>
                  ) : (
                    <span
                      className="px-2 py-0.5 text-xs font-semibold bg-gradient-to-r from-amber-100 to-orange-100 text-amber-700 dark:from-amber-900/30 dark:to-orange-900/30 dark:text-amber-300 rounded-full flex-shrink-0 border border-amber-200 dark:border-amber-600"
                      title="Pending security review by an admin (local server)"
                    >
                      SECURITY PENDING (LOCAL)
                    </span>
                  )
                )}
                {/* ANS badge moved to trust bar below description */}
                {/* Registry source badge - only show for federated (peer registry) items */}
                {isFederatedServer && (
                  <span className="px-2 py-0.5 text-xs font-semibold bg-gradient-to-r from-cyan-100 to-blue-100 text-cyan-700 dark:from-cyan-900/30 dark:to-blue-900/30 dark:text-cyan-300 rounded-full flex-shrink-0 border border-cyan-200 dark:border-cyan-600" title={`Synced from ${peerRegistryId}`}>
                    {peerRegistryId?.toUpperCase().replace('PEER-REGISTRY-', '').replace('PEER-', '')}
                    {/* ARD marker — this peer is an ai-catalog (ARD) registry. */}
                    {isArdDiscovery && (
                      <span className="ml-1 text-[10px] font-bold text-cyan-500 dark:text-cyan-400" title="ARD (ai-catalog.json) registry">
                        ARD
                      </span>
                    )}
                  </span>
                )}
                {/* Orphaned badge - server no longer exists on peer registry */}
                {isOrphanedServer && (
                  <span className="px-2 py-0.5 text-xs font-semibold bg-gradient-to-r from-red-100 to-rose-100 text-red-700 dark:from-red-900/30 dark:to-rose-900/30 dark:text-red-300 rounded-full flex-shrink-0 border border-red-200 dark:border-red-600" title="No longer exists on peer registry">
                    ORPHANED
                  </span>
                )}
                {/* Discovery badge - ARD discovery-only import (metadata only) */}
                {isArdDiscovery && (
                  <span className="px-2 py-0.5 text-xs font-semibold bg-gradient-to-r from-slate-100 to-gray-100 text-slate-700 dark:from-slate-900/30 dark:to-gray-900/30 dark:text-slate-300 rounded-full flex-shrink-0 border border-slate-200 dark:border-slate-600" title="Discovery-only import — resolve and connect at the source registry">
                    Discovery
                  </span>
                )}
                {/* Backend auth scheme badge */}
                {server.auth_scheme && server.auth_scheme !== 'none' && server.auth_scheme === 'bearer' && (
                  <span className="px-2 py-0.5 text-xs font-semibold bg-gradient-to-r from-blue-100 to-indigo-100 text-blue-700 dark:from-blue-900/30 dark:to-indigo-900/30 dark:text-blue-300 rounded-full flex-shrink-0 border border-blue-200 dark:border-blue-600" title="Backend uses Bearer token authentication">
                    BEARER AUTH
                  </span>
                )}
                {server.auth_scheme && server.auth_scheme === 'api_key' && (
                  <span className="px-2 py-0.5 text-xs font-semibold bg-gradient-to-r from-yellow-100 to-amber-100 text-yellow-700 dark:from-yellow-900/30 dark:to-amber-900/30 dark:text-yellow-300 rounded-full flex-shrink-0 border border-yellow-200 dark:border-yellow-600" title={`Backend uses API Key authentication (header: ${server.auth_header_name || 'X-API-Key'})`}>
                    API KEY AUTH
                  </span>
                )}
            </>
          }
          actions={
            <>
            {canModify && !isArdDiscovery && (
              <button
                className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 hover:bg-gray-50 dark:hover:bg-gray-700/50 rounded-lg transition-all duration-200 flex-shrink-0"
                onClick={() => onEdit?.(server)}
                title="Edit server"
                aria-label={`Edit ${server.name}`}
              >
                <PencilIcon className="h-4 w-4" />
              </button>
            )}

            {/* Connect Button — hidden for ARD discovery-only imports, which have
                no endpoint to connect to (resolve and connect at the source). */}
            {!isArdDiscovery && (
              <button
                onClick={() => setShowConfig(true)}
                className="flex items-center gap-1 px-2 py-1.5 text-xs font-medium text-green-600 dark:text-green-400 hover:bg-green-50 dark:hover:bg-green-700/50 rounded-lg transition-all duration-200 flex-shrink-0 border border-green-200 dark:border-green-700"
                title="Get connection details and mcp.json configuration"
                aria-label={`Connect to ${server.name}`}
              >
                <LinkIcon className="h-3.5 w-3.5" />
                Connect
              </button>
            )}

            {/* Egress "Link account" affordance — only for servers with per-user
                egress auth the current user can reach (egressConnect defined). */}
            {egressConnect && !isArdDiscovery && (
              <EgressConnectButton
                serverPath={server.path}
                serverName={server.name}
                state={egressConnect}
                onShowToast={onShowToast}
              />
            )}

            {/* Full JSON Details + Security Scan — interactive actions, hidden for
                ARD discovery-only imports so the card is fully read-only. */}
            {!isArdDiscovery && (
              <>
              {/* Full JSON Details Button */}
              <button
                onClick={async () => {
                  setShowDetails(true);
                  setDetailsLoading(true);
                  try {
                    const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
                    const response = await axios.get(
                      `/api/server_details${server.path}`,
                      headers ? { headers } : undefined
                    );
                    setFullServerDetails(response.data);
                  } catch (err) {
                    console.error('Failed to fetch full server details:', err);
                  } finally {
                    setDetailsLoading(false);
                  }
                }}
                className="p-2 text-gray-400 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-700/50 rounded-lg transition-all duration-200 flex-shrink-0"
                title="View full server JSON from database"
                aria-label={`View full details for ${server.name}`}
              >
                <InformationCircleIcon className="h-4 w-4" />
              </button>

              {/* Security Scan Button */}
              <button
                onClick={handleViewSecurityScan}
                className={`p-2 hover:bg-gray-50 dark:hover:bg-gray-700/50 rounded-lg transition-all duration-200 flex-shrink-0 ${getSecurityIconState().color}`}
                title={getSecurityIconState().title}
                aria-label="View security scan results"
              >
                {React.createElement(getSecurityIconState().Icon, { className: "h-4 w-4" })}
              </button>
              </>
            )}

            {/* Delete Button */}
            {canDelete && !isArdDiscovery && (
              <button
                onClick={() => setShowDeleteConfirm(true)}
                className="p-2 text-gray-400 hover:text-red-600 dark:hover:text-red-400 hover:bg-red-50 dark:hover:bg-red-700/50 rounded-lg transition-all duration-200 flex-shrink-0"
                title="Delete server"
                aria-label={`Delete ${server.name}`}
              >
                <TrashIcon className="h-4 w-4" />
              </button>
            )}
            </>
          }
        >
          <CardBody description={server.description} unwrapped>
            {/* ANS Trust Bar */}
            {server.ans_metadata && (
              <div className="mb-4 p-2.5 rounded-lg bg-gray-50/80 dark:bg-gray-800/50 border border-gray-200/60 dark:border-gray-700/60 flex items-center gap-3">
                <ANSBadge ansMetadata={server.ans_metadata} compact />
                <span className="text-xs text-gray-500 dark:text-gray-400 truncate">
                  {server.ans_metadata.domain || server.ans_metadata.ans_agent_id}
                </span>
              </div>
            )}

            <TagList tags={server.tags || []} accent={cardAccent} prefix="#" className="mb-4" />

            {/* ARD discovery-only: link back to the source registry's server.json,
                if the source published it (ard_source_url / upstream_path). */}
            {isArdDiscovery && ardSourceUrl && (
              <SafeLink
                href={ardSourceUrl}
                target="_blank"
                rel="noopener noreferrer"
                className="mb-4 inline-flex items-center gap-1 text-xs text-blue-600 dark:text-blue-400 hover:underline"
              >
                <LinkIcon className="h-3.5 w-3.5" />
                View at source ↗
              </SafeLink>
            )}
          </CardBody>
        </CardHeader>

        <CardStatsRow columns={3}>
            {/* Rating block — hidden for ARD discovery-only imports (read-only).
                An empty cell keeps the 3-column grid aligned. */}
            {isArdDiscovery ? (
              <div />
            ) : (
              <StarRatingWidget
                resourceType="servers"
                path={server.path}
                initialRating={server.rating || 0}
                initialCount={server.rating_details?.length || 0}
                ratingDetails={server.rating_details}
                authToken={authToken}
                onShowToast={onShowToast}
              />
            )}
            {/* Tools stat — hidden entirely for ARD discovery-only imports
                (no tools published by source). An empty cell keeps the grid
                aligned. */}
            {isArdDiscovery ? (
              <div />
            ) : (
            <div className="flex items-center gap-2">
              {(server.num_tools || 0) > 0 ? (
                <button
                  onClick={handleViewTools}
                  disabled={loadingTools}
                  className={`flex items-center gap-2 disabled:opacity-50 px-2 py-1 -mx-2 -my-1 rounded transition-all ${ACCENTS[cardAccent].interactive}`}
                  title="View tools"
                >
                  <div className={`p-1.5 rounded ${ACCENTS[cardAccent].iconChip}`}>
                    <WrenchScrewdriverIcon className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="text-sm font-semibold">{server.num_tools}</div>
                    <div className="text-xs">Tools</div>
                  </div>
                </button>
              ) : (
                <div className="flex items-center gap-2 text-gray-400 dark:text-gray-500">
                  <div className="p-1.5 bg-gray-50 dark:bg-gray-800 rounded">
                    <WrenchScrewdriverIcon className="h-4 w-4" />
                  </div>
                  <div>
                    <div className="text-sm font-semibold">{server.num_tools || 0}</div>
                    <div className="text-xs">Tools</div>
                  </div>
                </div>
              )}
            </div>
            )}
            {/* Version display - user routing version and/or MCP server version */}
            <div className="flex flex-col items-end gap-1">
              {server.versions && server.versions.length > 1 && (
                <VersionBadge
                  versions={server.versions}
                  defaultVersion={server.default_version || server.version}
                  onClick={() => setShowVersionSelector(true)}
                />
              )}
              {server.mcp_server_version && (
                <span
                  className="inline-flex items-center px-2 py-0.5 text-xs font-medium bg-gray-50 text-gray-600 dark:bg-gray-800 dark:text-gray-400 rounded"
                  title={
                    server.mcp_server_version_previous
                      ? `MCP Server Version: ${server.mcp_server_version} (previously ${server.mcp_server_version_previous})`
                      : `MCP Server Version: ${server.mcp_server_version}`
                  }
                >
                  <span className="text-gray-400 dark:text-gray-500 mr-1">srv</span>
                  {server.mcp_server_version}
                  {server.mcp_server_version_updated_at &&
                    (Date.now() - new Date(server.mcp_server_version_updated_at).getTime()) < 24 * 60 * 60 * 1000 && (
                    <span className="ml-1 h-1.5 w-1.5 rounded-full bg-green-500 inline-block" title="Recently updated" />
                  )}
                </span>
              )}
            </div>
        </CardStatsRow>

        <CardFooter
          accent={cardAccent}
          status={
            // ARD discovery-only imports are always enabled and read-only, so the
            // "Enabled" dot serves no purpose; the health dot is also hidden (no
            // health check). Show no status indicators at all for them.
            isArdDiscovery ? null : (
            <>
              <StatusDot
                tone={server.enabled ? 'green' : 'off'}
                label={server.enabled ? 'Enabled' : 'Disabled'}
              />
              <StatusDivider accent={cardAccent} />
              <StatusDot tone={health.tone} label={health.label} title={health.title} />
            </>
            )
          }
          controls={
            <>
              {/* Last Updated (source timestamp) */}
              {server.source_updated_at && (
                <div className="text-xs text-gray-500 dark:text-gray-300 flex items-center gap-1.5">
                  <ClockIcon className="h-3.5 w-3.5" />
                  <span title={new Date(server.source_updated_at).toLocaleString()}>
                    {formatRelativeTime(server.source_updated_at)}
                  </span>
                </div>
              )}

              {/* Last Checked */}
              {(() => {
                const timeText = formatTimeSince(server.last_checked_time);
                return server.last_checked_time && timeText && !server.source_updated_at ? (
                  <div className="text-xs text-gray-500 dark:text-gray-300 flex items-center gap-1.5">
                    <ClockIcon className="h-3.5 w-3.5" />
                    <span>{timeText}</span>
                  </div>
                ) : null;
              })()}

              {/* Refresh Button — only show if the user has health_check_service
                  permission AND the server has something to refresh. Local
                  (stdio) servers have no HTTP endpoint to probe, so refresh
                  is a no-op for them — hide the button entirely. */}
              {canHealthCheck && !isLocal && !isArdDiscovery && (
                <button
                  onClick={handleRefreshHealth}
                  disabled={loadingRefresh}
                  className="p-2.5 text-gray-500 hover:text-blue-600 dark:hover:text-blue-400 hover:bg-blue-50 dark:hover:bg-blue-900/20 rounded-lg transition-all duration-200 disabled:opacity-50"
                  title="Refresh health status"
                  aria-label={`Refresh health status for ${server.name}`}
                >
                  <ArrowPathIcon className={`h-4 w-4 ${loadingRefresh ? 'animate-spin' : ''}`} />
                </button>
              )}

              {/* Toggle Switch - only show if user has toggle_service permission.
                  Hidden for ARD discovery-only imports — a read-only record
                  shouldn't be toggleable. */}
              {canToggle && !isArdDiscovery && (
                <ToggleSwitch
                  checked={server.enabled}
                  onChange={(checked) => onToggle(server.path, checked)}
                  ariaLabel={`Enable ${server.name}`}
                  accent={cardAccent}
                />
              )}
            </>
          }
        />
        </>
        )}
      </CardShell>

      {/* Tools Modal */}
      {showTools && (
        <div
          className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50"
          onClick={() => {
            setShowTools(false);
            setExpandedDescriptions(new Set());
          }}
        >
          <div
            className="bg-white dark:bg-gray-800 rounded-xl p-6 max-w-2xl w-full mx-4 max-h-[80vh] overflow-auto"
            onClick={(e) => e.stopPropagation()}
          >
            <div className="flex items-center justify-between mb-4">
              <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                Tools for {server.name}
              </h3>
              <button
                onClick={() => {
                  setShowTools(false);
                  setExpandedDescriptions(new Set());
                }}
                className="text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
              >
                ✕
              </button>
            </div>

            <div className="space-y-4">
              {tools.length > 0 ? (
                tools.map((tool, index) => {
                  const isExpanded = expandedDescriptions.has(index);
                  const toggleExpand = () => {
                    const newExpanded = new Set(expandedDescriptions);
                    if (isExpanded) {
                      newExpanded.delete(index);
                    } else {
                      newExpanded.add(index);
                    }
                    setExpandedDescriptions(newExpanded);
                  };

                  return (
                    <div key={index} className="border border-gray-200 dark:border-gray-700 rounded-lg p-4">
                      <h4 className="font-medium text-gray-900 dark:text-white mb-2">
                        {tool.name}
                      </h4>
                      {tool.description && (
                        <div className="mb-2">
                          <p className={`text-sm text-gray-600 dark:text-gray-300 ${!isExpanded ? 'line-clamp-2' : ''}`}>
                            {tool.description}
                          </p>
                          {tool.description.length > 150 && (
                            <button
                              onClick={toggleExpand}
                              className="text-xs text-blue-600 dark:text-blue-400 hover:underline mt-1"
                            >
                              {isExpanded ? 'Show less' : 'Show more'}
                            </button>
                          )}
                        </div>
                      )}
                      {tool.schema && (
                        <details className="text-xs">
                          <summary className="cursor-pointer text-gray-500 dark:text-gray-300">
                            View Schema
                          </summary>
                          <pre className="mt-2 p-3 bg-gray-50 dark:bg-gray-900 border dark:border-gray-700 rounded overflow-x-auto text-gray-900 dark:text-gray-100">
                            {JSON.stringify(tool.schema, null, 2)}
                          </pre>
                        </details>
                      )}
                    </div>
                  );
                })
              ) : server.enabled && (server.status === 'healthy' || server.status === 'healthy-auth-expired') ? (
                <p className="text-gray-500 dark:text-gray-300">
                  No tools available to you on this server. Ask your administrator if you need access.
                </p>
              ) : (
                <p className="text-gray-500 dark:text-gray-300">No tools available for this server.</p>
              )}
            </div>
          </div>
        </div>
      )}

      <ServerConfigModal
        server={server}
        isOpen={showConfig}
        onClose={() => setShowConfig(false)}
        onShowToast={onShowToast}
        egressConnect={egressConnect}
        onEgressChanged={onEgressChanged}
      />

      <SecurityScanModal
        resourceName={server.name}
        resourceType="server"
        isOpen={showSecurityScan}
        onClose={() => setShowSecurityScan(false)}
        loading={loadingSecurityScan}
        scanResult={securityScanResult}
        // Local (stdio) servers can't be scanned — registry has no HTTP
        // endpoint to probe. They carry the security-pending-local tag for
        // manual review instead.
        onRescan={canModify && !isLocal ? handleRescan : undefined}
        canRescan={canModify && !isLocal}
        unscannableReason={
          isLocal
            ? 'Local (stdio) servers cannot be scanned automatically — the registry has no HTTP endpoint to probe. These servers carry the “security-pending-local” tag and require manual review of the launch recipe (package, args, env) before being marked as reviewed.'
            : undefined
        }
        onShowToast={onShowToast}
      />

      <ConfirmModal
        isOpen={showClearSecurityPendingConfirm}
        onClose={() => setShowClearSecurityPendingConfirm(false)}
        onConfirm={handleClearSecurityPending}
        title={`Mark "${server.name}" as security-reviewed?`}
        message={
          'Local servers are not auto-scanned because the registry has no HTTP ' +
          'endpoint to probe. Clearing this tag asserts that you have manually ' +
          'reviewed the launch recipe (package, args, env) and consider it safe ' +
          'for distribution to developers.'
        }
        confirmLabel="Mark as reviewed"
        cancelLabel="Cancel"
        loadingLabel="Marking..."
        isLoading={clearingSecurityPending}
      />

      <VersionSelectorModal
        isOpen={showVersionSelector}
        onClose={() => setShowVersionSelector(false)}
        serverName={server.name}
        serverPath={server.path}
        versions={server.versions || []}
        defaultVersion={server.default_version || null}
        onVersionChange={(newDefaultVersion) => {
          if (onServerUpdate) {
            // Update both default_version and versions array to reflect the change
            const updatedVersions = server.versions?.map(v => ({
              ...v,
              is_default: v.version === newDefaultVersion
            }));
            onServerUpdate(server.path, {
              default_version: newDefaultVersion,
              versions: updatedVersions
            });
          }
        }}
        onRefreshServer={handleRefreshServerData}
        onShowToast={onShowToast}
        authToken={authToken}
        canModify={canModify}
      />

      <ServerDetailsModal
        server={server}
        isOpen={showDetails}
        onClose={() => { setShowDetails(false); setFullServerDetails(null); }}
        loading={detailsLoading}
        fullDetails={fullServerDetails}
        authToken={authToken}
      />

    </>
  );
});

ServerCard.displayName = 'ServerCard';

export default ServerCard;
