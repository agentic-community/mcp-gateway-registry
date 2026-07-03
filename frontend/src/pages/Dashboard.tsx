import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { useNavigate, useSearchParams } from 'react-router-dom';
import { MagnifyingGlassIcon, PlusIcon, XMarkIcon, ArrowPathIcon, CheckCircleIcon, ExclamationCircleIcon } from '@heroicons/react/24/outline';
import { useServerStats } from '../hooks/useServerStats';
import { useSkills, Skill } from '../hooks/useSkills';
import { useAuth } from '../contexts/AuthContext';
import { useRegistryConfig } from '../hooks/useRegistryConfig';
import ServerCard from '../components/ServerCard';
import AgentCard from '../components/AgentCard';
import SkillCard from '../components/SkillCard';
import VirtualServerCard from '../components/VirtualServerCard';
import SemanticSearchResults from '../components/SemanticSearchResults';
import { useSemanticSearch } from '../hooks/useSemanticSearch';
import { useVirtualServers, useVirtualServer } from '../hooks/useVirtualServers';
import {
  VirtualServerInfo,
  CreateVirtualServerRequest,
  UpdateVirtualServerRequest,
} from '../types/virtualServer';
import VirtualServerForm from '../components/VirtualServerForm';
import DiscoverTab from '../components/DiscoverTab';
import type { CustomEntitySection } from '../components/DiscoverTab';
import CustomEntityTab from '../components/CustomEntityTab';
import CustomEntityForm from '../components/CustomEntityForm';
import CustomEntityDetail from '../components/CustomEntityDetail';
import ConfirmModal from '../components/ConfirmModal';
import SkillsSection from '../components/entities/sections/SkillsSection';
import VirtualServersSection from '../components/entities/sections/VirtualServersSection';
import RegistrySection, {
  type RegistryAccent,
} from '../components/entities/sections/RegistrySection';
import ExternalRegistriesSection from '../components/entities/sections/ExternalRegistriesSection';
import ServerEditModal, {
  type ServerEditForm,
} from '../components/entities/forms/ServerEditModal';
import AgentEditModal, {
  type AgentEditForm,
} from '../components/entities/forms/AgentEditModal';
import SkillFormModal, {
  type SkillForm,
} from '../components/entities/forms/SkillFormModal';
import ServerRegisterModal, {
  type ServerRegisterForm,
} from '../components/entities/forms/ServerRegisterModal';
import { useEntityToggle } from '../hooks/useEntityToggle';
import { filterEntities } from '../utils/entityFilters';

// Federated-registry header accents (local groups are always green/emerald).
const SERVER_REGISTRY_ACCENT: RegistryAccent = {
  headerBg:
    'bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 hover:from-cyan-100 hover:to-blue-100 dark:hover:from-cyan-900/30 dark:hover:to-blue-900/30',
  title: 'text-cyan-700 dark:text-cyan-300',
  resyncButton:
    'text-cyan-600 dark:text-cyan-400 hover:text-cyan-800 dark:hover:text-cyan-200 hover:bg-cyan-100 dark:hover:bg-cyan-900/30',
  border: 'border-gray-200 dark:border-gray-700',
};
const AGENT_REGISTRY_ACCENT: RegistryAccent = {
  headerBg:
    'bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 hover:from-violet-100 hover:to-purple-100 dark:hover:from-violet-900/30 dark:hover:to-purple-900/30',
  title: 'text-violet-700 dark:text-violet-300',
  resyncButton:
    'text-violet-600 dark:text-violet-400 hover:text-violet-800 dark:hover:text-violet-200 hover:bg-violet-100 dark:hover:bg-violet-900/30',
  border: 'border-cyan-200 dark:border-cyan-700',
};
import { uuidFromPath } from '../hooks/useCustomEntities';
import type {
  CustomEntityRecord,
  CustomEntityCreate,
  CustomEntityUpdate,
  CustomTypeDescriptor,
} from '../types/customEntity';
import axios from 'axios';
import { getBaseURL } from '../utils/basePath';
import { isEgressAuthEnabled } from '../utils/egressAuth';
import {
  buildLocalRuntimeForm,
  buildLocalRuntimeJson,
} from '../utils/localRuntime';
import type { LocalRuntime } from '../types/server';
import Pagination from '../components/Pagination';
import DuplicateCheckModal from '../components/DuplicateCheckModal';
import { useDuplicateCheck } from '../hooks/useDuplicateCheck';
import type { ExistingEntity } from '../types/duplicateCheck';


interface SyncMetadata {
  is_federated?: boolean;
  source_peer_id?: string;
  upstream_path?: string;
  last_synced_at?: string;
  is_read_only?: boolean;
  is_orphaned?: boolean;
  orphaned_at?: string;
}

interface Server {
  name: string;
  path: string;
  description?: string;
  official?: boolean;
  enabled: boolean;
  tags?: string[];
  last_checked_time?: string;
  usersCount?: number;
  rating?: number;
  status?: 'healthy' | 'healthy-auth-expired' | 'unhealthy' | 'unknown' | 'local';
  num_tools?: number;
  proxy_pass_url?: string;
  license?: string;
  mcp_endpoint?: string;
  metadata?: Record<string, unknown>;
  sync_metadata?: SyncMetadata;
  auth_scheme?: string;
  auth_header_name?: string;
  // Local-server fields
  deployment?: 'remote' | 'local';
  local_runtime?: LocalRuntime;
  registered_by?: string | null;
}

interface Agent {
  name: string;
  path: string;
  url?: string;
  description?: string;
  version?: string;
  visibility?: 'public' | 'private' | 'group-restricted';
  trust_level?: 'community' | 'verified' | 'trusted' | 'unverified';
  supported_protocol?: string | null;
  enabled: boolean;
  tags?: string[];
  last_checked_time?: string;
  usersCount?: number;
  rating?: number;
  status?: 'healthy' | 'healthy-auth-expired' | 'unhealthy' | 'unknown';
  lifecycle_status?: 'active' | 'draft' | 'deprecated' | 'beta';
  sync_metadata?: SyncMetadata;
  // ARD discovery imports: URL to the source registry's descriptor.
  ard_source_url?: string;
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
  registered_by?: string | null;
}

// Toast notification component
interface ToastProps {
  message: string;
  type: 'success' | 'error';
  onClose: () => void;
}

const Toast: React.FC<ToastProps> = ({ message, type, onClose }) => {
  useEffect(() => {
    const timer = setTimeout(() => {
      onClose();
    }, 4000);
    return () => clearTimeout(timer);
  }, [onClose]);

  return (
    <div className="fixed top-4 right-4 z-50 animate-slide-in-top">
      <div className={`flex items-center p-4 rounded-lg shadow-lg border ${
        type === 'success'
          ? 'bg-green-50 border-green-200 text-green-800 dark:bg-green-900/50 dark:border-green-700 dark:text-green-200'
          : 'bg-red-50 border-red-200 text-red-800 dark:bg-red-900/50 dark:border-red-700 dark:text-red-200'
      }`}>
        {type === 'success' ? (
          <CheckCircleIcon className="h-5 w-5 mr-3 flex-shrink-0" />
        ) : (
          <ExclamationCircleIcon className="h-5 w-5 mr-3 flex-shrink-0" />
        )}
        <p className="text-sm font-medium">{message}</p>
        <button
          onClick={onClose}
          className="ml-3 flex-shrink-0 text-current opacity-70 hover:opacity-100"
        >
          <XMarkIcon className="h-4 w-4" />
        </button>
      </div>
    </div>
  );
};

const normalizeAgentStatus = (status?: string | null): Agent['status'] => {
  if (status === 'healthy' || status === 'healthy-auth-expired') {
    return status;
  }
  if (status === 'unhealthy') {
    return 'unhealthy';
  }
  return 'unknown';
};

const buildAgentAuthHeaders = (token?: string | null) =>
  token ? { Authorization: `Bearer ${token}` } : undefined;

interface DashboardProps {
  activeFilter?: string;
  setActiveFilter?: (filter: string) => void;
  selectedTags?: string[];
}

const Dashboard: React.FC<DashboardProps> = ({ activeFilter = 'all', setActiveFilter, selectedTags = [] }) => {
  const navigate = useNavigate();
  const { servers, agents: agentsFromStats, customRecordsByType: customEntityRecordsByType, loading, error, refreshData, setServers, setAgents } = useServerStats();
  const { skills, setSkills, loading: skillsLoading, error: skillsError, refreshData: refreshSkills } = useSkills();
  const {
    virtualServers,
    loading: virtualServersLoading,
    error: virtualServersError,
    toggleVirtualServer,
    deleteVirtualServer,
    updateVirtualServer,
    refreshData: refreshVirtualServers,
  } = useVirtualServers();

  // Virtual server edit modal state
  const [editingVirtualServerPath, setEditingVirtualServerPath] = useState<string | undefined>(undefined);
  const [showVirtualServerForm, setShowVirtualServerForm] = useState(false);
  const { virtualServer: editingVirtualServer, loading: editingVirtualServerLoading } = useVirtualServer(editingVirtualServerPath);
  const { user } = useAuth();
  const { config: registryConfig } = useRegistryConfig();
  const [searchTerm, setSearchTerm] = useState('');
  const [committedQuery, setCommittedQuery] = useState('');

  // When navigated here from the dedup-suggestion modal
  // (Register page) with `?highlight=<path>`, pre-fill the search box so
  // the user lands directly on the existing entry they were considering.
  // The optional `tab=<servers|agents|skills>` param switches the
  // dashboard view filter so the user lands on the correct list — the
  // dedup advisory list is cross-entity, so a skill registration can
  // surface a similar MCP server, and the tab needs to match.
  // Query params are consumed once and stripped to avoid re-applying on
  // back/forward navigation.
  const [searchParams, setSearchParams] = useSearchParams();
  // Depend on the param string (not the searchParams object identity)
  // so this re-runs when navigate() updates the URL while we're
  // already mounted — which is the path the dedup modal takes when
  // the user picks "View" on a Dashboard-hosted duplicate check
  // (skill flow). Without this, the highlight/tab params would be
  // applied only on mount, and same-page navigations would silently
  // do nothing.
  const paramString = searchParams.toString();
  useEffect(() => {
    const highlight = searchParams.get('highlight');
    const tab = searchParams.get('tab');
    if (!highlight && !tab) return;
    if (highlight) {
      setSearchTerm(highlight);
    }
    if (tab === 'servers' || tab === 'agents' || tab === 'skills') {
      setViewFilter(tab);
    }
    const next = new URLSearchParams(searchParams);
    next.delete('highlight');
    next.delete('tab');
    setSearchParams(next, { replace: true });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [paramString]);
  const [showRegisterModal, setShowRegisterModal] = useState(false);
  const [registerForm, setRegisterForm] = useState<ServerRegisterForm>({
    name: '',
    path: '',
    proxyPass: '',
    description: '',
    official: false,
    tags: [],
  });
  const [registerLoading, setRegisterLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [editingServer, setEditingServer] = useState<Server | null>(null);
  const [editForm, setEditForm] = useState<ServerEditForm>({
    name: '',
    path: '',
    proxyPass: '',
    description: '',
    tags: [] as string[],
    license: 'N/A',
    num_tools: 0,
    mcp_endpoint: '',
    metadata: '',
    auth_scheme: 'none',
    auth_credential: '',
    auth_header_name: 'X-API-Key',
    status: 'active' as 'active' | 'draft' | 'deprecated' | 'beta',
    // Local-server fields
    deployment: 'remote' as 'remote' | 'local',
    local_runtime: {
      type: 'npx' as 'npx' | 'docker' | 'uvx' | 'command',
      package: '',
      version: '',
      image_digest: '',
      argList: [] as string[],
      envRows: [] as { key: string; value: string; required: boolean }[],
    },
    custom_headers: [] as Array<{ name: string; value: string }>,
    // Per-user egress credential vault (admin config). egress_provider empty == off.
    egress_provider: '',
    egress_client_id: '',
    egress_client_secret: '',  // write-only; blank on edit keeps the stored one
    egress_scopes: '',  // comma/space separated
    egress_custom_authorize_url: '',
    egress_custom_token_url: '',
  });
  const [egressEnabled, setEgressEnabled] = useState(false);
  const [editLoading, setEditLoading] = useState(false);
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);

  // Agent state management - using agents from useServerStats hook instead of separate fetch
  // Agents loading state is now handled by the useServerStats hook's 'loading' state
  const [agentsError, setAgentsError] = useState<string | null>(null);
  const [editingAgent, setEditingAgent] = useState<Agent | null>(null);
  const [agentApiToken, setAgentApiToken] = useState<string | null>(null);

  // View filter state. Custom entity types use a dynamic `custom:{name}` value.
  type ViewFilter =
    | 'discover'
    | 'servers'
    | 'agents'
    | 'skills'
    | 'virtual'
    | 'external'
    | `custom:${string}`;
  const [viewFilter, setViewFilter] = useState<ViewFilter>('discover');

  // Pagination state (per entity type)
  const PAGE_SIZE = 50;
  const [serverPage, setServerPage] = useState(0);
  const [agentPage, setAgentPage] = useState(0);
  const [skillPage, setSkillPage] = useState(0);

  // Reset pagination when filters or search change
  useEffect(() => {
    setServerPage(0);
    setAgentPage(0);
    setSkillPage(0);
  }, [activeFilter, selectedTags, viewFilter]);

  // Probe whether the per-user egress vault feature is enabled (gates the
  // egress section in the server-edit modal).
  useEffect(() => {
    let active = true;
    void isEgressAuthEnabled().then(enabled => {
      if (active) setEgressEnabled(enabled);
    });
    return () => {
      active = false;
    };
  }, []);

  // Reset viewFilter to 'discover' when the active tab is hidden by config
  useEffect(() => {
    if (viewFilter === 'virtual' && registryConfig?.features.virtual_servers === false) {
      setViewFilter('discover');
    }
    if (viewFilter === 'agents' && registryConfig?.features.agents === false) {
      setViewFilter('discover');
    }
    if (viewFilter === 'skills' && registryConfig?.features.skills === false) {
      setViewFilter('discover');
    }
    if (viewFilter === 'servers' && registryConfig?.features.mcp_servers === false) {
      setViewFilter('discover');
    }
    // A custom-type tab whose type is no longer in config (admin deleted it).
    if (viewFilter.startsWith('custom:')) {
      const typeName = viewFilter.slice('custom:'.length);
      const exists = (registryConfig?.custom_types ?? []).some((t) => t.name === typeName);
      if (registryConfig && !exists) {
        setViewFilter('discover');
      }
    }
  }, [viewFilter, registryConfig]);

  // Collapsible state for registry groups (tracks which groups are expanded)
  // Key is registry name: 'local' or peer registry ID like 'peer-registry-lob-1'
  const [expandedRegistries, setExpandedRegistries] = useState<Record<string, boolean>>({
    'local': true  // Local registry expanded by default
  });

  // Toggle a registry group's expanded state
  const toggleRegistryGroup = useCallback((registryId: string) => {
    setExpandedRegistries(prev => ({
      ...prev,
      [registryId]: !prev[registryId]
    }));
  }, []);

  // Store peer registry endpoints for display
  // Maps peer_id to endpoint URL: { 'peer-registry-lob-1': 'https://mcpregistry.ddns.net', ... }
  const [peerRegistryEndpoints, setPeerRegistryEndpoints] = useState<Record<string, string>>({});

  // Track which peer is currently being synced
  const [syncingPeer, setSyncingPeer] = useState<string | null>(null);

  // Active source tab within External Registries (null = show all)
  const [externalSourceTab, setExternalSourceTab] = useState<string | null>(null);

  // Fetch peer registry configs to get their endpoints
  useEffect(() => {
    const fetchPeerEndpoints = async () => {
      try {
        const response = await axios.get('/api/peers');
        const peers = response.data?.peers || response.data || [];
        const endpoints: Record<string, string> = {};
        peers.forEach((peer: { peer_id: string; endpoint: string }) => {
          if (peer.peer_id && peer.endpoint) {
            endpoints[peer.peer_id] = peer.endpoint;
          }
        });
        setPeerRegistryEndpoints(endpoints);
      } catch (error) {
        // Silently fail - peer endpoints are optional display info
        console.debug('Could not fetch peer registry endpoints:', error);
      }
    };
    fetchPeerEndpoints();
  }, []);

  // Built-in federation sources (anthropic / asor / aws_registry) that are
  // actually added/configured. A source counts as "added" when it is enabled OR
  // has any entries. This gates the built-in source sub-tabs so stale tagged
  // data alone no longer surfaces a tab for a source that is no longer set up.
  const [configuredBuiltinSources, setConfiguredBuiltinSources] = useState<Set<string>>(new Set());

  useEffect(() => {
    const fetchFederationConfig = async () => {
      try {
        const response = await axios.get('/api/federation/config');
        const config = response.data || {};
        const configured = new Set<string>();
        if (config.anthropic?.enabled || (config.anthropic?.servers?.length || 0) > 0) {
          configured.add('anthropic');
        }
        if (config.asor?.enabled || (config.asor?.agents?.length || 0) > 0) {
          configured.add('asor');
        }
        if (config.aws_registry?.enabled || (config.aws_registry?.registries?.length || 0) > 0) {
          configured.add('aws_registry');
        }
        setConfiguredBuiltinSources(configured);
      } catch (error) {
        // Silently fail - the tabs simply fall back to showing none of the
        // built-in sources until the config is available.
        console.debug('Could not fetch federation config:', error);
      }
    };
    fetchFederationConfig();
  }, []);

  // Get the local registry URL. Includes the ROOT_PATH prefix
  // (e.g. "/registry" in path routing mode) so the displayed URL
  // matches what clients actually hit.
  const localRegistryUrl = useMemo(() => {
    return `${window.location.origin}${getBaseURL()}`;
  }, []);

  const [editAgentForm, setEditAgentForm] = useState<AgentEditForm>({
    name: '',
    path: '',
    url: '',
    description: '',
    version: '',
    visibility: 'private' as 'public' | 'private' | 'group-restricted',
    allowed_groups: '',
    trust_level: 'community' as 'community' | 'verified' | 'trusted' | 'unverified',
    supported_protocol: 'other' as 'a2a' | 'other',
    tags: [] as string[],
    skillsJson: '[]',
    metadata: '',
    status: 'active' as 'active' | 'draft' | 'deprecated' | 'beta',
  });
  const [editAgentLoading, setEditAgentLoading] = useState(false);
  const [skillsJsonError, setSkillsJsonError] = useState<string | null>(null);

  // Skill state management
  const [showSkillModal, setShowSkillModal] = useState(false);
  const [editingSkill, setEditingSkill] = useState<Skill | null>(null);
  const [skillForm, setSkillForm] = useState<SkillForm>({
    name: '',
    description: '',
    skill_md_url: '',
    repository_url: '',
    version: '',
    visibility: 'public' as 'public' | 'private' | 'group',
    tags: '',  // Raw string, parsed on save
    target_agents: '',  // Raw string, parsed on save
    metadata: '',  // JSON string for custom metadata
    status: 'draft' as 'active' | 'draft' | 'deprecated' | 'beta',
    auth_scheme: 'none' as 'none' | 'global_credentials' | 'bearer' | 'api_key',
    auth_credential: '',
    auth_header_name: '',
  });
  const [skillFormLoading, setSkillFormLoading] = useState(false);
  const [showDeleteSkillConfirm, setShowDeleteSkillConfirm] = useState<string | null>(null);
  const [skillAutoFill, setSkillAutoFill] = useState(true);  // Auto-fill from SKILL.md

  // Duplicate-check pre-flight for new skill registrations. Edits are
  // not eligible (the path / skill_md_url is already taken by the
  // caller's own record).
  const {
    collisionWith: skillCollisionWith,
    advisoryMatches: skillAdvisoryMatches,
    showModal: showSkillDuplicateModal,
    runCheck: runSkillDuplicateCheck,
    closeModal: closeSkillDuplicateModal,
  } = useDuplicateCheck();
  const [skillParseLoading, setSkillParseLoading] = useState(false);

  const handleAgentUpdate = useCallback((path: string, updates: Partial<Agent>) => {
    setAgents(prevAgents =>
      prevAgents.map(agent =>
        agent.path === path
          ? { ...agent, ...updates }
          : agent
      )
    );
  }, [setAgents]);

  const performAgentHealthCheck = useCallback(async (agent: Agent, token?: string | null) => {
    if (!agent?.path) return;

    const headers = buildAgentAuthHeaders(token);
    try {
      const response = await axios.post(
        `/api/agents${agent.path}/health`,
        undefined,
        headers ? { headers } : undefined
      );

      handleAgentUpdate(agent.path, {
        status: normalizeAgentStatus(response.data?.status),
        last_checked_time: response.data?.last_checked_iso || null
      });
    } catch (error) {
      console.error(`Failed to check health for agent ${agent.name}:`, error);
      handleAgentUpdate(agent.path, {
        status: 'unhealthy',
        last_checked_time: new Date().toISOString()
      });
    }
  }, [handleAgentUpdate]);

  const runInitialAgentHealthChecks = useCallback((agentsList: Agent[], token?: string | null) => {
    const candidates = agentsList.filter(agent => agent.enabled);
    if (!candidates.length) return;

    Promise.allSettled(candidates.map(agent => performAgentHealthCheck(agent, token))).catch((error) => {
      console.error('Failed to run agent health checks:', error);
    });
  }, [performAgentHealthCheck]);

  // Note: Agents data now comes from useServerStats hook
  // JWT token generation moved to after agents definition

  // Helper function to check if user has a specific UI permission for a service
  const hasUiPermission = useCallback((permission: string, servicePath: string): boolean => {
    const permissions = user?.ui_permissions?.[permission];
    if (!permissions) return false;

    // Extract service name from path (remove leading slash)
    const serviceName = servicePath.replace(/^\//, '');

    // Check if user has 'all' permission or specific service permission
    return permissions.includes('all') || permissions.includes(serviceName);
  }, [user?.ui_permissions]);

  // External registry tags - can be configured via environment or constants
  // Default tags that identify servers from external registries
  const EXTERNAL_REGISTRY_TAGS = ['anthropic-registry', 'workday-asor', 'asor', 'federated'];

  // Separate internal and external registry servers
  const internalServers = useMemo(() => {
    return servers.filter(s => {
      const serverTags = s.tags || [];
      return !EXTERNAL_REGISTRY_TAGS.some(tag => serverTags.includes(tag));
    });
  }, [servers]);

  const externalServers = useMemo(() => {
    return servers.filter(s => {
      const serverTags = s.tags || [];
      return EXTERNAL_REGISTRY_TAGS.some(tag => serverTags.includes(tag));
    });
  }, [servers]);

  // Separate internal and external registry agents
  // Transform Server[] to Agent[] for agents from useServerStats
  const agents = useMemo(() => {
    return agentsFromStats.map((a): Agent => ({
      name: a.name,
      path: a.path,
      description: a.description,
      enabled: a.enabled,
      tags: a.tags,
      rating: a.rating,
      // Agents are A2A entities (HTTP-only) — 'local' is a server-only status.
      // Map it to 'unknown' if it ever leaks through.
      status: a.status === 'local' ? 'unknown' : a.status,
      last_checked_time: a.last_checked_time,
      usersCount: a.usersCount,
      url: '',  // Will be populated if needed
      version: '',
      visibility: (a.visibility || 'public') as 'public' | 'private' | 'group-restricted',
      trust_level: (a.trust_level || 'community') as 'community' | 'verified' | 'trusted' | 'unverified',
      supported_protocol: a.supported_protocol || null,
      sync_metadata: a.sync_metadata,
      // ARD discovery imports: carry the source descriptor URL through to the
      // AgentCard so the "View at source" link can render.
      ard_source_url: a.ard_source_url,
      ans_metadata: a.ans_metadata,
      registered_by: a.registered_by,
      lifecycle_status: a.lifecycle_status,
    }));
  }, [agentsFromStats]);

  const internalAgents = useMemo(() => {
    return agents.filter(a => {
      const agentTags = a.tags || [];
      return !EXTERNAL_REGISTRY_TAGS.some(tag => agentTags.includes(tag));
    });
  }, [agents]);

  const externalAgents = useMemo(() => {
    return agents.filter(a => {
      const agentTags = a.tags || [];
      return EXTERNAL_REGISTRY_TAGS.some(tag => agentTags.includes(tag));
    });
  }, [agents]);

  // Separate internal and external skills
  const externalSkills = useMemo(() => {
    return skills.filter(s => {
      const skillTags = s.tags || [];
      return EXTERNAL_REGISTRY_TAGS.some(tag => skillTags.includes(tag));
    });
  }, [skills]);

  // Tag-to-source mapping: which tag identifies which federation source
  const SOURCE_TAG_MAP: Record<string, string> = {
    'anthropic-registry': 'anthropic',
    'agentcore': 'aws_registry',
    'asor': 'asor',
    'workday-asor': 'asor',
  };

  // Display labels for each source
  const SOURCE_LABELS: Record<string, string> = {
    'anthropic': 'Anthropic',
    'aws_registry': 'AWS Agent Registry',
    'asor': 'ASOR',
  };

  // Detect which dynamic (federated/ARD) sources exist from sync_metadata.
  // ARD discovery-only imports carry the `federated` tag (so they're external)
  // and a sync_metadata.source_peer_id (e.g. 'mcpgateway-ddns'). Each distinct
  // peer id becomes its own source tab alongside the hardcoded ones. Servers and
  // agents carry sync_metadata; skills do not (their source id rides as a tag).
  const dynamicExternalSources = useMemo(() => {
    const ids = new Set<string>();
    for (const server of externalServers) {
      const peerId = server.sync_metadata?.source_peer_id;
      if (peerId) ids.add(peerId);
    }
    for (const agent of externalAgents) {
      const peerId = agent.sync_metadata?.source_peer_id;
      if (peerId) ids.add(peerId);
    }
    return Array.from(ids).sort();
  }, [externalServers, externalAgents]);

  // Detect which external sources exist based on tags in the data, then append
  // the dynamic (federated/ARD) source ids after the hardcoded ones.
  const availableExternalSources = useMemo(() => {
    const sources = new Set<string>();
    const allExternalItems = [
      ...externalServers.map(s => s.tags || []),
      ...externalAgents.map(a => a.tags || []),
      ...externalSkills.map(s => s.tags || []),
    ];
    for (const tags of allExternalItems) {
      for (const tag of tags) {
        const source = SOURCE_TAG_MAP[tag];
        if (source) {
          sources.add(source);
        }
      }
    }
    // If AWS Registry has content, show it first; otherwise default order
    const order = sources.has('aws_registry')
      ? ['aws_registry', 'anthropic', 'asor']
      : ['anthropic', 'asor'];
    // A built-in source tab renders only when the source has tagged content AND
    // it is actually added/configured in the federation config. This prevents a
    // tab from appearing for a built-in source whose data is stale but which is
    // no longer configured.
    const hardcoded = order.filter(s => sources.has(s) && configuredBuiltinSources.has(s));
    // Append dynamic (ARD) sources, which stay purely content-driven (shown when
    // they have items). Filter out any already covered by the hardcoded list.
    const dynamic = dynamicExternalSources.filter(id => !hardcoded.includes(id));
    return [...hardcoded, ...dynamic];
  }, [externalServers, externalAgents, externalSkills, dynamicExternalSources, configuredBuiltinSources]);

  // Build a labels map covering the hardcoded sources (from SOURCE_LABELS) plus
  // each dynamic source id, which falls back to its upper-cased peer id.
  const externalSourceLabels = useMemo(() => {
    const labels: Record<string, string> = { ...SOURCE_LABELS };
    for (const id of dynamicExternalSources) {
      if (!labels[id]) {
        labels[id] = id.toUpperCase();
      }
    }
    return labels;
  }, [dynamicExternalSources]);

  // Helper: check if an item belongs to a given source.
  // - For the hardcoded sources (anthropic/asor/aws_registry), match on tags via
  //   SOURCE_TAG_MAP.
  // - For dynamic source ids, match iff the item's source_peer_id equals the tab.
  //   ARD/federated items also carry the source id as a literal tag, so a tag
  //   match is used as a fallback for items without sync_metadata (e.g. skills).
  const dynamicSourceSet = useMemo(
    () => new Set(dynamicExternalSources),
    [dynamicExternalSources]
  );
  const _itemMatchesSource = useCallback(
    (tags: string[] | undefined, source: string): boolean => {
      if (dynamicSourceSet.has(source)) {
        return (tags || []).includes(source);
      }
      if (!tags) return false;
      return tags.some(tag => SOURCE_TAG_MAP[tag] === source);
    },
    [dynamicSourceSet]
  );

  // Auto-select the first available tab when switching to external view
  // or when available sources change
  useEffect(() => {
    if (viewFilter === 'external' && availableExternalSources.length > 0) {
      if (externalSourceTab === null || !availableExternalSources.includes(externalSourceTab)) {
        setExternalSourceTab(availableExternalSources[0]);
      }
    }
  }, [viewFilter, availableExternalSources, externalSourceTab]);

  // Group servers by source registry (local vs peer registries) using sync_metadata
  // Returns a map of registry ID to servers: { 'local': [...], 'peer-registry-lob-1': [...], ... }
  const serversByRegistry = useMemo(() => {
    const groups: Record<string, Server[]> = { 'local': [] };

    internalServers.forEach(server => {
      // Check if server is from a peer registry using sync_metadata
      if (server.sync_metadata?.is_federated && server.sync_metadata?.source_peer_id) {
        const registryId = server.sync_metadata.source_peer_id;
        if (!groups[registryId]) {
          groups[registryId] = [];
        }
        groups[registryId].push(server);
      } else {
        groups['local'].push(server);
      }
    });

    return groups;
  }, [internalServers]);

  // Get sorted list of registry IDs (local first, then peer registries alphabetically)
  const registryIds = useMemo(() => {
    const ids = Object.keys(serversByRegistry);
    return ['local', ...ids.filter(id => id !== 'local').sort()];
  }, [serversByRegistry]);

  // Group agents by source registry similarly using sync_metadata
  const agentsByRegistry = useMemo(() => {
    const groups: Record<string, Agent[]> = { 'local': [] };

    internalAgents.forEach(agent => {
      // Check if agent is from a peer registry using sync_metadata
      if (agent.sync_metadata?.is_federated && agent.sync_metadata?.source_peer_id) {
        const registryId = agent.sync_metadata.source_peer_id;
        if (!groups[registryId]) {
          groups[registryId] = [];
        }
        groups[registryId].push(agent);
      } else {
        groups['local'].push(agent);
      }
    });

    return groups;
  }, [internalAgents]);

  const agentRegistryIds = useMemo(() => {
    const ids = Object.keys(agentsByRegistry);
    return ['local', ...ids.filter(id => id !== 'local').sort()];
  }, [agentsByRegistry]);

  // Semantic search
  const semanticEnabled = committedQuery.trim().length >= 2;
  const {
    results: semanticResults,
    loading: semanticLoading,
    error: semanticError
  } = useSemanticSearch(committedQuery, {
    minLength: 2,
    maxResults: 10,
    enabled: semanticEnabled,
    tags: selectedTags.length > 0 ? selectedTags : undefined,
  });

  const semanticServers = semanticResults?.servers ?? [];
  const semanticTools = semanticResults?.tools ?? [];
  const semanticAgents = semanticResults?.agents ?? [];
  const semanticSkills = semanticResults?.skills ?? [];
  const semanticVirtualServers = semanticResults?.virtual_servers ?? [];
  const semanticCustom = semanticResults?.custom ?? [];
  const semanticDisplayQuery = semanticResults?.query || committedQuery || searchTerm;
  const semanticSectionVisible = semanticEnabled;
  const shouldShowFallbackGrid =
    semanticSectionVisible &&
    (Boolean(semanticError) ||
      (!semanticLoading &&
        semanticServers.length === 0 &&
        semanticTools.length === 0 &&
        semanticAgents.length === 0 &&
        semanticSkills.length === 0 &&
        semanticVirtualServers.length === 0 &&
        semanticCustom.length === 0));

  // Helper: check if entity has all selected tags (case-insensitive)
  const matchesSelectedTags = useCallback((entityTags: string[] | undefined) => {
    if (selectedTags.length === 0) return true;
    if (!entityTags || entityTags.length === 0) return false;
    const lowerTags = entityTags.map(t => t.toLowerCase());
    return selectedTags.every(st => lowerTags.includes(st.toLowerCase()));
  }, [selectedTags]);

  // Per-type custom entity counts for the sidebar summary, respecting the
  // sidebar tag filter the same way the built-in categories do.
  const customCounts = useMemo(
    () =>
      customEntityRecordsByType.map((ct) => ({
        name: ct.name,
        displayName: ct.displayName,
        count: ct.records.filter((r) => matchesSelectedTags(r.tags)).length,
      })),
    [customEntityRecordsByType, matchesSelectedTags],
  );

  // Custom-entity modal state for the Discover view. Spans multiple types, so
  // each carries its own descriptor (the per-tab hook only knows one type).
  const [customViewing, setCustomViewing] = useState<{
    descriptor: CustomTypeDescriptor;
    record: CustomEntityRecord;
  } | null>(null);
  const [customEditing, setCustomEditing] = useState<{
    typeName: string;
    descriptor: CustomTypeDescriptor;
    record: CustomEntityRecord | null;
  } | null>(null);
  const [customDeleting, setCustomDeleting] = useState<{
    typeName: string;
    record: CustomEntityRecord;
  } | null>(null);
  const [customDeleteLoading, setCustomDeleteLoading] = useState(false);

  // Build the Discover custom sections: tag-filtered records + descriptor +
  // per-record view/edit/delete handlers. Only types with a loaded descriptor
  // can render cards, so descriptor-less types are dropped here.
  const customSections = useMemo<CustomEntitySection[]>(
    () =>
      customEntityRecordsByType
        .filter((ct) => ct.descriptor)
        .map((ct) => {
          const descriptor = ct.descriptor as CustomTypeDescriptor;
          return {
            name: ct.name,
            displayName: ct.displayName,
            descriptor,
            records: ct.records.filter((r) => matchesSelectedTags(r.tags)),
            canModify: (record: CustomEntityRecord) =>
              !!user?.is_admin || record.owner === user?.username,
            onView: (record: CustomEntityRecord) =>
              setCustomViewing({ descriptor, record }),
            onEdit: (record: CustomEntityRecord) =>
              setCustomEditing({ typeName: ct.name, descriptor, record }),
            onDelete: (record: CustomEntityRecord) =>
              setCustomDeleting({ typeName: ct.name, record }),
          };
        }),
    [customEntityRecordsByType, matchesSelectedTags, user],
  );

  // Parse #tag tokens from the search term for local filtering
  const parsedSearch = useMemo(() => {
    const hashtagPattern = /#([\w-]+)/g;
    const hashTags: string[] = [];
    let match;
    while ((match = hashtagPattern.exec(searchTerm)) !== null) {
      hashTags.push(match[1].toLowerCase());
    }
    // Remove matched #tag tokens AND any trailing/leading lone # characters
    const textQuery = searchTerm
      .replace(/#[\w-]+/g, '')
      .replace(/#/g, '')
      .replace(/\s+/g, ' ')
      .trim()
      .toLowerCase();
    return { textQuery, hashTags };
  }, [searchTerm]);

  // Helper: check if entity matches #tag tokens from search term (prefix match while typing)
  const matchesHashTags = useCallback((entityTags: string[] | undefined) => {
    if (parsedSearch.hashTags.length === 0) return true;
    if (!entityTags || entityTags.length === 0) return false;
    const lowerTags = entityTags.map(t => t.toLowerCase());
    return parsedSearch.hashTags.every(ht =>
      lowerTags.some(tag => tag.startsWith(ht))
    );
  }, [parsedSearch.hashTags]);

  // Filter servers based on activeFilter, searchTerm, and selectedTags
  const filteredServers = useMemo(() => {
    return filterEntities(internalServers, {
      activeFilter,
      enabledField: 'enabled',
      statusField: 'status',
      lifecycleField: 'lifecycle_status',
      selectedTags,
      matchesSelectedTags,
      parsedSearch,
      matchesHashTags,
      getTags: (s) => s.tags,
      getSearchText: (s) => [s.name, s.description, s.path, ...(s.tags || [])],
    });
  }, [internalServers, activeFilter, selectedTags, matchesSelectedTags, parsedSearch, matchesHashTags]);

  // Filter external servers based on source tab, searchTerm, and selectedTags
  const filteredExternalServers = useMemo(() => {
    return filterEntities(externalServers, {
      sourceTab: externalSourceTab,
      matchesSource: _itemMatchesSource,
      selectedTags,
      matchesSelectedTags,
      parsedSearch,
      matchesHashTags,
      getTags: (s) => s.tags,
      getSearchText: (s) => [s.name, s.description, s.path, ...(s.tags || [])],
    });
  }, [externalServers, externalSourceTab, _itemMatchesSource, selectedTags, matchesSelectedTags, parsedSearch, matchesHashTags]);

  // Filter external agents based on source tab, searchTerm, and selectedTags
  const filteredExternalAgents = useMemo(() => {
    return filterEntities(externalAgents, {
      sourceTab: externalSourceTab,
      matchesSource: _itemMatchesSource,
      selectedTags,
      matchesSelectedTags,
      parsedSearch,
      matchesHashTags,
      getTags: (a) => a.tags,
      getSearchText: (a) => [a.name, a.description, a.path, ...(a.tags || [])],
    });
  }, [externalAgents, externalSourceTab, _itemMatchesSource, selectedTags, matchesSelectedTags, parsedSearch, matchesHashTags]);

  // Filter external skills based on source tab, searchTerm, and selectedTags
  const filteredExternalSkills = useMemo(() => {
    return filterEntities(externalSkills, {
      sourceTab: externalSourceTab,
      matchesSource: _itemMatchesSource,
      selectedTags,
      matchesSelectedTags,
      parsedSearch,
      matchesHashTags,
      getTags: (s) => s.tags,
      getSearchText: (s) => [s.name, s.description, s.path, ...(s.tags || [])],
    });
  }, [externalSkills, externalSourceTab, _itemMatchesSource, selectedTags, matchesSelectedTags, parsedSearch, matchesHashTags]);

  // Filter agents based on activeFilter, searchTerm, and selectedTags
  const filteredAgents = useMemo(() => {
    return filterEntities(internalAgents, {
      activeFilter,
      enabledField: 'enabled',
      statusField: 'status',
      lifecycleField: 'lifecycle_status',
      selectedTags,
      matchesSelectedTags,
      parsedSearch,
      matchesHashTags,
      getTags: (a) => a.tags,
      getSearchText: (a) => [a.name, a.description, a.path, ...(a.tags || [])],
    });
  }, [internalAgents, activeFilter, selectedTags, matchesSelectedTags, parsedSearch, matchesHashTags]);

  // Filter skills based on activeFilter, searchTerm, and selectedTags
  const filteredSkills = useMemo(() => {
    return filterEntities(skills, {
      activeFilter,
      enabledField: 'is_enabled',
      // Skills have no 'unhealthy' filter; lifecycle lives in `status`.
      lifecycleField: 'status',
      selectedTags,
      matchesSelectedTags,
      parsedSearch,
      matchesHashTags,
      getTags: (s) => s.tags,
      getSearchText: (s) => [s.name, s.description, s.path, s.author, ...(s.tags || [])],
    });
  }, [skills, activeFilter, selectedTags, matchesSelectedTags, parsedSearch, matchesHashTags]);

  // Paginated slice of skills (servers/agents paginate inline in the render)
  const paginatedSkills = useMemo(() => {
    const start = skillPage * PAGE_SIZE;
    return filteredSkills.slice(start, start + PAGE_SIZE);
  }, [filteredSkills, skillPage]);

  const serverTotalPages = Math.ceil(filteredServers.length / PAGE_SIZE);
  const agentTotalPages = Math.ceil(filteredAgents.length / PAGE_SIZE);
  const skillTotalPages = Math.ceil(filteredSkills.length / PAGE_SIZE);

  // Filter virtual servers based on activeFilter, searchTerm, and selectedTags
  const filteredVirtualServers = useMemo(() => {
    return filterEntities(virtualServers, {
      activeFilter,
      enabledField: 'is_enabled',
      // Virtual servers have no lifecycle status, so deprecated isn't hidden.
      selectedTags,
      matchesSelectedTags,
      parsedSearch,
      matchesHashTags,
      getTags: (vs) => vs.tags,
      getSearchText: (vs) => [vs.server_name, vs.description, vs.path, ...(vs.tags || [])],
    });
  }, [virtualServers, activeFilter, selectedTags, matchesSelectedTags, parsedSearch, matchesHashTags]);

  // Virtual server action handlers
  const handleToggleVirtualServer = useCallback(async (path: string, enabled: boolean) => {
    try {
      await toggleVirtualServer(path, enabled);
      showToast(`Virtual server ${enabled ? 'enabled' : 'disabled'} successfully`, 'success');
    } catch (err) {
      console.error('Failed to toggle virtual server:', err);
      showToast('Failed to toggle virtual server', 'error');
    }
  }, [toggleVirtualServer]);

  // State for virtual server delete confirmation on Dashboard
  const [deleteVirtualServerTarget, setDeleteVirtualServerTarget] = useState<VirtualServerInfo | null>(null);
  const [deleteVirtualServerTypedName, setDeleteVirtualServerTypedName] = useState('');
  const [deletingVirtualServer, setDeletingVirtualServer] = useState(false);

  const handleDeleteVirtualServer = useCallback((path: string) => {
    const target = virtualServers.find((vs) => vs.path === path);
    if (target) {
      setDeleteVirtualServerTarget(target);
      setDeleteVirtualServerTypedName('');
    }
  }, [virtualServers]);

  const confirmDeleteVirtualServer = useCallback(async () => {
    if (!deleteVirtualServerTarget || deleteVirtualServerTypedName !== deleteVirtualServerTarget.server_name) return;

    setDeletingVirtualServer(true);
    try {
      await deleteVirtualServer(deleteVirtualServerTarget.path);
      showToast('Virtual server deleted successfully', 'success');
      notifyDataChanged();
      setDeleteVirtualServerTarget(null);
      setDeleteVirtualServerTypedName('');
    } catch (err) {
      console.error('Failed to delete virtual server:', err);
      showToast('Failed to delete virtual server', 'error');
    } finally {
      setDeletingVirtualServer(false);
    }
  }, [deleteVirtualServerTarget, deleteVirtualServerTypedName, deleteVirtualServer]);

  const handleEditVirtualServer = useCallback((vs: VirtualServerInfo) => {
    setEditingVirtualServerPath(vs.path);
    setShowVirtualServerForm(true);
  }, []);

  const handleSaveVirtualServer = useCallback(async (
    data: CreateVirtualServerRequest | UpdateVirtualServerRequest
  ) => {
    if (!editingVirtualServerPath) return;
    try {
      await updateVirtualServer(editingVirtualServerPath, data as UpdateVirtualServerRequest);
      showToast('Virtual server updated successfully', 'success');
      notifyDataChanged();
      setShowVirtualServerForm(false);
      setEditingVirtualServerPath(undefined);
      refreshVirtualServers();
    } catch (err: unknown) {
      const message = err instanceof Error ? err.message : 'An unexpected error occurred';
      showToast(`Failed to save virtual server: ${message}`, 'error');
    }
  }, [editingVirtualServerPath, updateVirtualServer, refreshVirtualServers]);

  const handleCancelVirtualServerEdit = useCallback(() => {
    setShowVirtualServerForm(false);
    setEditingVirtualServerPath(undefined);
  }, []);

  // Debug logging for filtering
  console.log('Dashboard filtering debug:');
  console.log(`Current user:`, user);
  console.log(`Total servers from hook: ${servers.length}`);
  console.log(`Total agents from API: ${agents.length}`);
  console.log(`Active filter: ${activeFilter}`);
  console.log(`Search term: "${searchTerm}"`);
  console.log(`Filtered servers: ${filteredServers.length}`);
  console.log(`Filtered agents: ${filteredAgents.length}`);

  useEffect(() => {
    if (searchTerm.trim().length === 0 && committedQuery.length > 0) {
      setCommittedQuery('');
    }
  }, [searchTerm, committedQuery]);

  // Close any open inline modal on Escape key
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key !== 'Escape') return;
      if (showVirtualServerForm) { handleCancelVirtualServerEdit(); return; }
      if (deleteVirtualServerTarget) { setDeleteVirtualServerTarget(null); setDeleteVirtualServerTypedName(''); return; }
      if (showDeleteSkillConfirm) { setShowDeleteSkillConfirm(null); return; }
      if (showSkillModal) { setShowSkillModal(false); return; }
      if (editingAgent) { setEditingAgent(null); return; }
      if (editingServer) { setEditingServer(null); return; }
      if (showRegisterModal) { setShowRegisterModal(false); return; }
    };
    document.addEventListener('keydown', handleKeyDown);
    return () => document.removeEventListener('keydown', handleKeyDown);
  }, [showVirtualServerForm, deleteVirtualServerTarget, showDeleteSkillConfirm, showSkillModal, editingAgent, editingServer, showRegisterModal, handleCancelVirtualServerEdit]);

  const handleSemanticSearch = useCallback(() => {
    const trimmed = searchTerm.trim();
    setCommittedQuery(trimmed);
  }, [searchTerm]);

  const handleClearSearch = useCallback(() => {
    setSearchTerm('');
    setCommittedQuery('');
  }, []);

  const handleChangeViewFilter = useCallback(
    (filter: typeof viewFilter) => {
      setViewFilter(filter);
      if (semanticSectionVisible) {
        setSearchTerm('');
        setCommittedQuery('');
      }
    },
    [semanticSectionVisible]
  );

  // Derived: the active custom type (name + display label), or null for built-in tabs.
  const currentCustomType = useMemo(() => {
    if (!viewFilter.startsWith('custom:')) return null;
    const typeName = viewFilter.slice('custom:'.length);
    return (registryConfig?.custom_types ?? []).find((t) => t.name === typeName) ?? null;
  }, [viewFilter, registryConfig]);

  // Notify Layout to refresh the sidebar tag list after data changes
  const notifyDataChanged = useCallback(() => {
    window.dispatchEvent(new Event('registry-data-changed'));
  }, []);

  const handleRefreshHealth = async () => {
    setRefreshing(true);
    try {
      await refreshData(); // Refresh both servers and agents from useServerStats
    } finally {
      setRefreshing(false);
    }
  };

  // Sync a peer registry to fetch latest servers/agents
  const handleSyncPeer = async (peerId: string, event: React.MouseEvent) => {
    event.stopPropagation(); // Prevent collapsing the section
    setSyncingPeer(peerId);
    try {
      const response = await axios.post(`/api/peers/${peerId}/sync`);
      const result = response.data;

      // Check the success field in the response body
      if (result.success) {
        setToast({
          message: `Synced ${result.servers_synced || 0} servers and ${result.agents_synced || 0} agents from ${peerId}`,
          type: 'success'
        });
      } else {
        // Sync failed - show error message from response
        setToast({
          message: result.error_message || `Failed to sync from ${peerId}`,
          type: 'error'
        });
      }

      // Refresh the server list to show updated data
      await refreshData();
      notifyDataChanged();
    } catch (error) {
      console.error('Failed to sync peer:', error);
      setToast({ message: `Failed to sync from ${peerId}`, type: 'error' });
    } finally {
      setSyncingPeer(null);
    }
  };

  const handleEditServer = useCallback(async (server: Server) => {
    // buildLocalRuntimeForm() handles the populate-from-stored-dict path
    // (returns fresh defaults for remote servers / undefined input).

    try {
      // Fetch full server details including proxy_pass_url and tags
      const response = await axios.get(`/api/server_details${server.path}`);
      const serverDetails = response.data;

      const deployment = (serverDetails.deployment || server.deployment || 'remote') as 'remote' | 'local';
      const localRuntimeRaw = serverDetails.local_runtime || server.local_runtime;

      setEditingServer(server);
      setEditForm({
        name: serverDetails.server_name || server.name,
        path: server.path,
        proxyPass: serverDetails.proxy_pass_url || '',
        description: serverDetails.description || '',
        tags: serverDetails.tags || [],
        license: serverDetails.license || 'N/A',
        num_tools: serverDetails.num_tools || 0,
        mcp_endpoint: serverDetails.mcp_endpoint || '',
        metadata: serverDetails.metadata ? JSON.stringify(serverDetails.metadata, null, 2) : '',
        auth_scheme: serverDetails.auth_scheme || 'none',
        auth_credential: '',
        auth_header_name: serverDetails.auth_header_name || 'X-API-Key',
        status: serverDetails.status || 'active',
        deployment,
        local_runtime: buildLocalRuntimeForm(localRuntimeRaw),
        custom_headers: (serverDetails.custom_header_names || []).map((name: string) => ({ name, value: '' })),
        egress_provider: serverDetails.egress_oauth?.provider || '',
        egress_client_id: serverDetails.egress_oauth?.client_id || '',
        egress_client_secret: '',  // never round-trip the secret
        egress_scopes: (serverDetails.egress_oauth?.scopes || []).join(', '),
        egress_custom_authorize_url: serverDetails.egress_oauth?.custom_authorize_url || '',
        egress_custom_token_url: serverDetails.egress_oauth?.custom_token_url || '',
      });
    } catch (error) {
      console.error('Failed to fetch server details:', error);
      // Fallback to basic server data
      const deployment = (server.deployment || 'remote') as 'remote' | 'local';
      setEditingServer(server);
      setEditForm({
        name: server.name,
        path: server.path,
        proxyPass: server.proxy_pass_url || '',
        description: server.description || '',
        tags: server.tags || [],
        license: 'N/A',
        num_tools: server.num_tools || 0,
        mcp_endpoint: server.mcp_endpoint || '',
        metadata: server.metadata ? JSON.stringify(server.metadata, null, 2) : '',
        auth_scheme: server.auth_scheme || 'none',
        auth_credential: '',
        auth_header_name: server.auth_header_name || 'X-API-Key',
        status: (server as any).status || 'active',
        deployment,
        local_runtime: buildLocalRuntimeForm(server.local_runtime),
        custom_headers: [],
        egress_provider: '',
        egress_client_id: '',
        egress_client_secret: '',
        egress_scopes: '',
        egress_custom_authorize_url: '',
        egress_custom_token_url: '',
      });
    }
  }, []);

  const handleEditAgent = useCallback(async (agent: Agent) => {
    setEditingAgent(agent);
    setSkillsJsonError(null);

    // Fetch full agent details to get skills and url
    try {
      const headers = agentApiToken ? { Authorization: `Bearer ${agentApiToken}` } : undefined;
      const response = await axios.get(
        `/api/agents${agent.path}`,
        headers ? { headers } : undefined
      );
      const fullAgent = response.data;

      setEditAgentForm({
        name: fullAgent.name || agent.name,
        path: fullAgent.path || agent.path,
        url: fullAgent.url || '',
        description: fullAgent.description || agent.description || '',
        version: fullAgent.version || agent.version || '1.0.0',
        visibility: fullAgent.visibility || agent.visibility || 'private',
        allowed_groups: (fullAgent.allowedGroups || fullAgent.allowed_groups || []).join(', '),
        trust_level: fullAgent.trust_level || agent.trust_level || 'community',
        supported_protocol: (fullAgent.supported_protocol || agent.supported_protocol || 'other') as 'a2a' | 'other',
        tags: fullAgent.tags || agent.tags || [],
        skillsJson: fullAgent.skills && fullAgent.skills.length > 0
          ? JSON.stringify(fullAgent.skills, null, 2)
          : '[]',
        metadata: fullAgent.metadata && Object.keys(fullAgent.metadata).length > 0
          ? JSON.stringify(fullAgent.metadata, null, 2)
          : '',
        status: (fullAgent.status || agent.lifecycle_status || 'active') as 'active' | 'draft' | 'deprecated' | 'beta',
      });
    } catch (error) {
      console.error('Failed to fetch agent details for editing:', error);
      // Fall back to basic data from the card
      setEditAgentForm({
        name: agent.name,
        path: agent.path,
        url: '',
        description: agent.description || '',
        version: agent.version || '1.0.0',
        visibility: agent.visibility || 'private',
        allowed_groups: '',
        trust_level: agent.trust_level || 'community',
        supported_protocol: (agent.supported_protocol || 'other') as 'a2a' | 'other',
        tags: agent.tags || [],
        skillsJson: '[]',
        metadata: '',
        status: agent.lifecycle_status || 'active',
      });
    }
  }, [agentApiToken]);

  const handleCloseEdit = () => {
    setEditingServer(null);
    setEditingAgent(null);
  };

  const showToast = useCallback((message: string, type: 'success' | 'error' | 'info') => {
    setToast({ message, type: type === 'info' ? 'success' : type });
  }, []);

  const hideToast = useCallback(() => {
    setToast(null);
  }, []);

  const handleCustomSave = useCallback(
    async (body: CustomEntityCreate | CustomEntityUpdate) => {
      if (!customEditing) return;
      const { typeName, record } = customEditing;
      if (record) {
        await axios.put(`/api/custom/${typeName}/${uuidFromPath(record.path)}`, body);
        showToast(`Updated ${body.name ?? record.name}`, 'success');
      } else {
        await axios.post(`/api/custom/${typeName}`, body);
        showToast(`Created ${body.name}`, 'success');
      }
      setCustomEditing(null);
      await refreshData();
    },
    [customEditing, refreshData, showToast],
  );

  const handleCustomDelete = useCallback(async () => {
    if (!customDeleting) return;
    setCustomDeleteLoading(true);
    try {
      await axios.delete(
        `/api/custom/${customDeleting.typeName}/${uuidFromPath(customDeleting.record.path)}`,
      );
      showToast(`Deleted ${customDeleting.record.name}`, 'success');
      setCustomDeleting(null);
      await refreshData();
    } catch (err: any) {
      showToast(err.response?.data?.detail || 'Failed to delete', 'error');
    } finally {
      setCustomDeleteLoading(false);
    }
  }, [customDeleting, refreshData, showToast]);

  const handleSaveEdit = async () => {
    if (editLoading || !editingServer) return;

    const isLocal = editForm.deployment === 'local';

    if (isLocal && !editForm.local_runtime.package.trim()) {
      const rt = editForm.local_runtime;
      showToast(
        rt.type === 'docker' ? 'Image reference is required'
        : rt.type === 'command' ? 'Command path is required'
        : 'Package name is required',
        'error',
      );
      return;
    }

    const localRuntimeJson = isLocal
      ? buildLocalRuntimeJson(editForm.local_runtime)
      : null;

    try {
      setEditLoading(true);

      const params = new URLSearchParams();
      params.append('name', editForm.name);
      params.append('description', editForm.description);
      params.append('tags', editForm.tags.join(','));
      params.append('license', editForm.license);
      params.append('num_tools', editForm.num_tools.toString());
      params.append('deployment', editForm.deployment);
      params.append('status', editForm.status);

      if (isLocal) {
        params.append('local_runtime', localRuntimeJson!);
      } else {
        params.append('proxy_pass_url', editForm.proxyPass);
        if (editForm.mcp_endpoint) {
          params.append('mcp_endpoint', editForm.mcp_endpoint);
        }
        if (editForm.auth_scheme !== 'none') {
          params.append('auth_scheme', editForm.auth_scheme);
          if (editForm.auth_credential) {
            params.append('auth_credential', editForm.auth_credential);
          }
          if (editForm.auth_scheme === 'api_key' && editForm.auth_header_name) {
            params.append('auth_header_name', editForm.auth_header_name);
          }
        } else {
          params.append('auth_scheme', 'none');
        }
      }

      if (editForm.metadata) {
        params.append('metadata', editForm.metadata);
      }

      const headersToSend = editForm.custom_headers.filter(h => h.name.trim());
      if (headersToSend.length > 0) {
        params.append('custom_headers', JSON.stringify(headersToSend));
      }

      // Use the correct edit endpoint with the server path
      await axios.post(`/api/edit${editingServer.path}`, params, {
        headers: {
          'Accept': 'application/json',
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });

      // Per-user egress credential vault (separate admin endpoint; the server
      // must exist first, which it does on edit). Only when the feature is on.
      if (egressEnabled) {
        try {
          const csrf = await axios.get('/api/auth/csrf-token');
          const csrfHeaders: Record<string, string> = {};
          if (csrf.data?.csrf_token) csrfHeaders['X-CSRF-Token'] = csrf.data.csrf_token;
          const provider = editForm.egress_provider.trim();
          if (!provider) {
            await axios.post(
              `/api/servers${editingServer.path}/egress-auth`,
              { egress_auth_mode: 'none' },
              { headers: csrfHeaders }
            );
          } else {
            await axios.post(
              `/api/servers${editingServer.path}/egress-auth`,
              {
                egress_auth_mode: 'oauth_user',
                egress_provider: provider,
                client_id: editForm.egress_client_id.trim(),
                // Blank secret on edit keeps the stored one (backend semantics).
                client_secret: editForm.egress_client_secret || undefined,
                scopes: editForm.egress_scopes
                  .split(/[,\s]+/)
                  .map(s => s.trim())
                  .filter(Boolean),
                custom_authorize_url: editForm.egress_custom_authorize_url || undefined,
                custom_token_url: editForm.egress_custom_token_url || undefined,
              },
              { headers: csrfHeaders }
            );
          }
        } catch (egressErr: any) {
          // Surface but don't lose the successful server edit.
          const d = egressErr.response?.data?.detail;
          showToast(`Server saved, but egress config failed: ${d || egressErr.message}`, 'error');
        }
      }

      // Refresh server list
      await refreshData();
      setEditingServer(null);

      showToast('Server updated successfully!', 'success');
      notifyDataChanged();
    } catch (error: any) {
      console.error('Failed to update server:', error);
      const detail = error.response?.data?.detail;
      const message =
        typeof detail === 'string' ? detail
        : detail && typeof detail === 'object' ? JSON.stringify(detail)
        : 'Failed to update server';
      showToast(message, 'error');
    } finally {
      setEditLoading(false);
    }
  };

  const handleSaveEditAgent = async () => {
    if (editAgentLoading || !editingAgent) return;

    // Validate skills JSON before sending
    let parsedSkills: any[] = [];
    try {
      parsedSkills = JSON.parse(editAgentForm.skillsJson);
      if (!Array.isArray(parsedSkills)) {
        setSkillsJsonError('Skills must be a JSON array');
        return;
      }
      setSkillsJsonError(null);
    } catch {
      setSkillsJsonError('Invalid JSON format');
      return;
    }

    try {
      setEditAgentLoading(true);

      const headers: Record<string, string> = {
        'Content-Type': 'application/json',
      };
      if (agentApiToken) {
        headers['Authorization'] = `Bearer ${agentApiToken}`;
      }

      const payload = {
        name: editAgentForm.name,
        description: editAgentForm.description,
        url: editAgentForm.url,
        version: editAgentForm.version,
        visibility: editAgentForm.visibility,
        allowedGroups: editAgentForm.visibility === 'group-restricted'
          ? editAgentForm.allowed_groups.split(',').map(g => g.trim()).filter(g => g)
          : [],
        trustLevel: editAgentForm.trust_level,
        supportedProtocol: editAgentForm.supported_protocol,
        tags: editAgentForm.tags,
        skills: parsedSkills,
        status: editAgentForm.status,
        ...(editAgentForm.metadata.trim() ? { metadata: JSON.parse(editAgentForm.metadata) } : {}),
      };

      await axios.put(
        `/api/agents${editingAgent.path}`,
        payload,
        { headers },
      );

      // Trigger security rescan after successful update
      try {
        await axios.post(
          `/api/agents${editingAgent.path}/rescan`,
          undefined,
          agentApiToken ? { headers: { Authorization: `Bearer ${agentApiToken}` } } : undefined,
        );
      } catch {
        // Rescan failure is non-blocking (may lack admin privileges)
      }

      // Refresh the agents list
      await refreshData();

      setEditingAgent(null);
      showToast('Agent updated successfully!', 'success');
    } catch (error: any) {
      console.error('Failed to update agent:', error);
      const detail = error.response?.data?.detail;
      const message = typeof detail === 'object' ? detail.message || JSON.stringify(detail) : detail || 'Failed to update agent';
      showToast(message, 'error');
    } finally {
      setEditAgentLoading(false);
    }
  };

  const handleToggleServer = useEntityToggle({
    setItems: setServers,
    enabledField: 'enabled',
    label: 'Server',
    showToast,
    apiCall: async (path, enabled) => {
      const formData = new FormData();
      formData.append('enabled', enabled ? 'on' : 'off');
      await axios.post(`/api/toggle${path}`, formData, {
        headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
      });
    },
  });

  const handleDeleteServer = useCallback(async (path: string) => {
    const formData = new FormData();
    formData.append('path', path);

    await axios.post('/api/servers/remove', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });

    // Remove from local state immediately for responsive UI
    setServers(prevServers => prevServers.filter(s => s.path !== path));
    showToast('Server deleted successfully', 'success');
    notifyDataChanged();
  }, [setServers, showToast]);

  const handleDeleteAgent = useCallback(async (path: string) => {
    await axios.delete(`/api/agents${path}`);

    // Remove from local state immediately for responsive UI
    setAgents(prevAgents => prevAgents.filter(a => a.path !== path));
    showToast('Agent deleted successfully', 'success');
    notifyDataChanged();
  }, [setAgents, showToast]);

  const handleToggleAgent = useEntityToggle({
    setItems: setAgents,
    enabledField: 'enabled',
    label: 'Agent',
    showToast,
    apiCall: async (path, enabled) => {
      await axios.post(`/api/agents${path}/toggle?enabled=${enabled}`);
    },
  });

  const handleServerUpdate = useCallback((path: string, updates: Partial<Server>) => {
    setServers(prevServers =>
      prevServers.map(server =>
        server.path === path
          ? { ...server, ...updates }
          : server
      )
    );
  }, [setServers]);

  const handleToggleSkill = useEntityToggle({
    setItems: setSkills,
    enabledField: 'is_enabled',
    label: 'Skill',
    showToast,
    apiCall: async (path, enabled) => {
      // Convert full path to API path (e.g., /skills/pdf -> /pdf)
      const apiPath = path.startsWith('/skills/') ? path.replace('/skills/', '/') : path;
      await axios.post(`/api/skills${apiPath}/toggle`, { enabled });
    },
  });

  const handleSkillUpdate = useCallback((path: string, updates: Partial<Skill>) => {
    setSkills(prevSkills =>
      prevSkills.map(skill =>
        skill.path === path
          ? { ...skill, ...updates }
          : skill
      )
    );
  }, [setSkills]);

  // Skill CRUD handlers
  const handleOpenSkillModal = useCallback((skill?: Skill) => {
    if (skill) {
      // Edit mode - populate form with existing data
      setEditingSkill(skill);
      setSkillAutoFill(false);  // Manual mode for editing
      setSkillForm({
        name: skill.name,
        description: skill.description || '',
        skill_md_url: skill.skill_md_url || '',
        repository_url: skill.repository_url || '',
        version: skill.version || '',
        visibility: skill.visibility || 'public',
        tags: (skill.tags || []).join(', '),
        target_agents: (skill.target_agents || []).join(', '),
        metadata: skill.metadata?.extra ? JSON.stringify(skill.metadata.extra, null, 2) : '',
        status: (skill.status || 'active') as 'active' | 'draft' | 'deprecated' | 'beta',
        auth_scheme: (skill.auth_scheme || 'none') as 'none' | 'global_credentials' | 'bearer' | 'api_key',
        auth_credential: '',
        auth_header_name: skill.auth_header_name || '',
      });
    } else {
      // Create mode - reset form
      setEditingSkill(null);
      setSkillAutoFill(true);  // Auto-fill enabled for new skills
      setSkillForm({
        name: '',
        description: '',
        skill_md_url: '',
        repository_url: '',
        version: '',
        visibility: 'public',
        tags: '',
        target_agents: '',
        metadata: '',
        status: 'draft',
        auth_scheme: 'none',
        auth_credential: '',
        auth_header_name: '',
      });
    }
    setShowSkillModal(true);
  }, []);

  const handleCloseSkillModal = useCallback(() => {
    setShowSkillModal(false);
    setEditingSkill(null);
  }, []);

  const handleParseSkillMd = useCallback(async () => {
    if (!skillForm.skill_md_url || skillParseLoading) return;

    try {
      setSkillParseLoading(true);
      const params = new URLSearchParams({ url: skillForm.skill_md_url });
      if (skillForm.auth_scheme !== 'none') {
        params.set('auth_scheme', skillForm.auth_scheme);
      }
      if (skillForm.auth_header_name && skillForm.auth_scheme === 'api_key') {
        params.set('auth_header_name', skillForm.auth_header_name);
      }
      // The credential is sent as a request header (never a query parameter) so
      // it is not captured in access logs, proxy logs, browser history, or the
      // audit log's recorded query parameters.
      const parseConfig: { headers?: Record<string, string> } = {};
      if (skillForm.auth_credential && skillForm.auth_scheme !== 'none' && skillForm.auth_scheme !== 'global_credentials') {
        parseConfig.headers = { 'X-Auth-Credential': skillForm.auth_credential };
      }
      const response = await axios.post(`/api/skills/parse-skill-md?${params.toString()}`, null, parseConfig);
      const data = response.data;

      if (data.success) {
        setSkillForm(prev => ({
          ...prev,
          name: data.name_slug || prev.name,
          description: data.description || prev.description,
          version: data.version || prev.version,
          tags: data.tags?.length > 0 ? data.tags.join(', ') : prev.tags,
          repository_url: data.repository_url || prev.repository_url,
        }));
        showToast('Parsed SKILL.md successfully!', 'success');
      } else {
        showToast('Failed to parse SKILL.md', 'error');
      }
    } catch (error: any) {
      console.error('Failed to parse SKILL.md:', error);
      showToast(error.response?.data?.detail || 'Failed to parse SKILL.md', 'error');
    } finally {
      setSkillParseLoading(false);
    }
  }, [skillForm.skill_md_url, skillForm.auth_scheme, skillForm.auth_credential, skillForm.auth_header_name, skillParseLoading, showToast]);

  const performSkillSave = useCallback(async (): Promise<void> => {
    try {
      setSkillFormLoading(true);

      // Parse comma-separated strings into arrays
      const parseTags = (str: string): string[] =>
        str.split(',').map(t => t.trim()).filter(t => t.length > 0);

      // Parse optional metadata JSON
      let parsedMetadata: Record<string, any> | undefined = undefined;
      if (skillForm.metadata.trim()) {
        try {
          parsedMetadata = JSON.parse(skillForm.metadata);
        } catch {
          showToast('Invalid JSON in metadata field', 'error');
          setSkillFormLoading(false);
          return;
        }
      }

      const payload: Record<string, any> = {
        name: skillForm.name,
        description: skillForm.description,
        skill_md_url: skillForm.skill_md_url,
        repository_url: skillForm.repository_url || undefined,
        version: skillForm.version || undefined,
        visibility: skillForm.visibility,
        tags: parseTags(skillForm.tags),
        target_agents: parseTags(skillForm.target_agents),
        metadata: parsedMetadata,
        status: skillForm.status,
        auth_scheme: skillForm.auth_scheme,
      };

      if (skillForm.auth_scheme !== 'none' && skillForm.auth_scheme !== 'global_credentials') {
        if (skillForm.auth_credential) {
          payload.auth_credential = skillForm.auth_credential;
        }
        if (skillForm.auth_scheme === 'api_key' && skillForm.auth_header_name) {
          payload.auth_header_name = skillForm.auth_header_name;
        }
      }

      if (editingSkill) {
        // Update existing skill
        const skillPath = editingSkill.path.replace(/^\/skills\//, '');
        await axios.put(`/api/skills/${skillPath}`, payload);
        showToast('Skill updated successfully!', 'success');
        notifyDataChanged();
      } else {
        // Create new skill
        await axios.post('/api/skills', payload);
        showToast('Skill registered successfully!', 'success');
        notifyDataChanged();
      }

      // Refresh skills list
      await refreshSkills();
      handleCloseSkillModal();
    } catch (error: any) {
      console.error('Failed to save skill:', error);
      const errorMsg = error.response?.data?.detail || 'Failed to save skill';
      showToast(errorMsg, 'error');
    } finally {
      setSkillFormLoading(false);
    }
  }, [skillForm, editingSkill, refreshSkills, showToast, notifyDataChanged, handleCloseSkillModal]);


  const handleSaveSkill = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (skillFormLoading) return;

    // Validate name format (lowercase, numbers, hyphens only)
    const nameRegex = /^[a-z0-9]+(-[a-z0-9]+)*$/;
    if (!nameRegex.test(skillForm.name)) {
      showToast('Name must be lowercase letters, numbers, and hyphens only (e.g., "my-skill-name")', 'error');
      return;
    }

    if (editingSkill) {
      await performSkillSave();
      return;
    }

    setSkillFormLoading(true);
    const outcome = await runSkillDuplicateCheck({
      entityType: 'skill',
      payload: {
        name: skillForm.name.trim(),
        description: skillForm.description.trim() || null,
        skill_md_url: skillForm.skill_md_url.trim() || null,
        self_path: null,  // new skill — no own path yet
      },
    });
    setSkillFormLoading(false);

    if (outcome.kind === 'show-modal') {
      return;
    }
    if (outcome.kind === 'cancelled') {
      return;
    }
    if (outcome.notice) {
      showToast(outcome.notice, 'error');
    }
    await performSkillSave();
  }, [
    skillForm,
    skillFormLoading,
    editingSkill,
    showToast,
    performSkillSave,
    runSkillDuplicateCheck,
  ]);

  const handleSkillDuplicateProceed = useCallback(async () => {
    closeSkillDuplicateModal();
    await performSkillSave();
  }, [closeSkillDuplicateModal, performSkillSave]);


  const handleSkillDuplicatePickExisting = useCallback(
    (entity: ExistingEntity) => {
      closeSkillDuplicateModal();
      // Also close the skill registration modal — the user has chosen
      // to view an existing entry, so leaving the form open underneath
      // would strand them on a stale registration draft when the
      // dashboard tab switches.
      setShowSkillModal(false);
      // The dedup advisory list is cross-entity: a skill registration
      // can surface a similar server or agent. Switch the dashboard
      // tab to the entity's own type so the highlighted path lands on
      // the correct list.
      const tabByEntityType: Record<string, string> = {
        mcp_server: 'servers',
        a2a_agent: 'agents',
        skill: 'skills',
      };
      const tab = tabByEntityType[entity.entity_type];
      const params = new URLSearchParams();
      params.set('highlight', entity.path);
      if (tab) {
        params.set('tab', tab);
      }
      navigate(`/?${params.toString()}`);
    },
    [navigate, closeSkillDuplicateModal],
  );


  const handleEditSkill = useCallback((skill: Skill) => {
    handleOpenSkillModal(skill);
  }, [handleOpenSkillModal]);

  const handleDeleteSkill = useCallback(async (path: string) => {
    try {
      await axios.delete(`/api/skills${path}`);

      // Remove from local state immediately for responsive UI
      // path may be shortened (e.g. "/add") while s.path is full (e.g. "/skills/add")
      const fullPath = path.startsWith('/skills/') ? path : `/skills${path}`;
      setSkills(prevSkills => prevSkills.filter(s => s.path !== path && s.path !== fullPath));
      showToast('Skill deleted successfully', 'success');
      notifyDataChanged();
      setShowDeleteSkillConfirm(null);
    } catch (error: any) {
      console.error('Failed to delete skill:', error);
      showToast(error.response?.data?.detail || 'Failed to delete skill', 'error');
    }
  }, [setSkills, showToast]);

  const handleRegisterServer = useCallback(() => {
    navigate('/servers/register');
  }, [navigate]);

  const handleRegisterSubmit = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (registerLoading) return; // Prevent double submission

    try {
      setRegisterLoading(true);

      const formData = new FormData();
      formData.append('name', registerForm.name);
      formData.append('description', registerForm.description);
      formData.append('path', registerForm.path);
      formData.append('proxy_pass_url', registerForm.proxyPass);
      formData.append('tags', registerForm.tags.join(','));
      formData.append('license', 'MIT');

      await axios.post('/api/register', formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });

      // Reset form and close modal
      setRegisterForm({
        name: '',
        path: '',
        proxyPass: '',
        description: '',
        official: false,
        tags: []
      });
      setShowRegisterModal(false);

      // Refresh server list
      await refreshData();

      showToast('Server registered successfully!', 'success');
      notifyDataChanged();
    } catch (error: any) {
      console.error('Failed to register server:', error);
      showToast(error.response?.data?.detail || 'Failed to register server', 'error');
    } finally {
      setRegisterLoading(false);
    }
  }, [registerForm, registerLoading, refreshData, showToast]);

  const renderDashboardCollections = () => (
    <>
      {/* MCP Servers Section - Grouped by Registry */}
      {registryConfig?.features.mcp_servers !== false &&
        (viewFilter === 'servers') && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                MCP Servers
              </h2>

              {/* Registry Quick Navigation - Only show if there are multiple registries */}
              {registryIds.length > 1 && filteredServers.length > 0 && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500 dark:text-gray-400 mr-1">Jump to:</span>
                  {registryIds.map(registryId => {
                    const count = (serversByRegistry[registryId] || []).length;
                    if (count === 0) return null;
                    const displayName = registryId === 'local'
                      ? 'Local'
                      : registryId.replace('peer-registry-', '').replace('peer-', '').toUpperCase();
                    const isLocal = registryId === 'local';

                    return (
                      <button
                        key={registryId}
                        onClick={() => {
                          // Expand this registry, collapse others (for both servers and agents)
                          const newExpanded: Record<string, boolean> = {};
                          // Update server registry states
                          registryIds.forEach(id => {
                            newExpanded[id] = (id === registryId);
                          });
                          // Also update agent registry states to keep them in sync
                          agentRegistryIds.forEach(id => {
                            newExpanded[`agents-${id}`] = (id === registryId);
                          });
                          setExpandedRegistries(prev => ({ ...prev, ...newExpanded }));
                          // Scroll to the section
                          const element = document.getElementById(`server-registry-${registryId}`);
                          if (element) {
                            element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                          }
                        }}
                        className={`px-3 py-1.5 text-xs font-medium rounded-full transition-all hover:scale-105 ${
                          isLocal
                            ? 'bg-green-100 text-green-700 hover:bg-green-200 dark:bg-green-900/30 dark:text-green-300 dark:hover:bg-green-900/50 border border-green-200 dark:border-green-700'
                            : 'bg-cyan-100 text-cyan-700 hover:bg-cyan-200 dark:bg-cyan-900/30 dark:text-cyan-300 dark:hover:bg-cyan-900/50 border border-cyan-200 dark:border-cyan-700'
                        }`}
                      >
                        {displayName}
                        <span className="ml-1.5 px-1.5 py-0.5 text-[10px] bg-white/50 dark:bg-black/20 rounded-full">
                          {count}
                        </span>
                      </button>
                    );
                  })}
                  {/* Expand All / Collapse All */}
                  <div className="border-l border-gray-300 dark:border-gray-600 pl-2 ml-1">
                    <button
                      onClick={() => {
                        const allExpanded = registryIds.every(id => expandedRegistries[id] !== false);
                        const newExpanded: Record<string, boolean> = {};
                        registryIds.forEach(id => {
                          newExpanded[id] = !allExpanded;
                        });
                        setExpandedRegistries(prev => ({ ...prev, ...newExpanded }));
                      }}
                      className="px-2 py-1 text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200 hover:bg-gray-100 dark:hover:bg-gray-700 rounded transition-colors"
                      title={registryIds.every(id => expandedRegistries[id] !== false) ? 'Collapse all' : 'Expand all'}
                    >
                      {registryIds.every(id => expandedRegistries[id] !== false) ? 'Collapse All' : 'Expand All'}
                    </button>
                  </div>
                </div>
              )}
            </div>

            {egressEnabled && (
              <p className="text-xs text-gray-500 dark:text-gray-400 -mt-2 mb-4">
                Some servers require per-user authentication (e.g. Slack, Atlassian).{' '}
                <button
                  type="button"
                  onClick={() => navigate('/connected-accounts')}
                  className="text-purple-600 dark:text-purple-400 underline hover:text-purple-700 dark:hover:text-purple-300 focus:outline-none"
                >
                  Click here to connect your accounts
                </button>{' '}
                before adding those servers to your coding assistant.
              </p>
            )}

            {serverTotalPages > 1 && (
              <div className="flex justify-center mb-4">
                <Pagination
                  currentPage={serverPage}
                  totalPages={serverTotalPages}
                  totalItems={filteredServers.length}
                  pageSize={PAGE_SIZE}
                  onPageChange={setServerPage}
                />
              </div>
            )}

            {filteredServers.length === 0 ? (
              <div className="text-center py-12 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-gray-400 text-lg mb-2">No servers found</div>
                <p className="text-gray-500 dark:text-gray-300 text-sm">
                  {selectedTags.length > 0
                    ? `No servers match the selected tag${selectedTags.length > 1 ? 's' : ''}`
                    : searchTerm || activeFilter !== 'all'
                      ? 'Press Enter in the search bar to search semantically'
                      : 'No servers are registered yet'}
                </p>
                {!searchTerm && activeFilter === 'all' && selectedTags.length === 0 && (
                  <button
                    onClick={handleRegisterServer}
                    className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-blue-600 hover:bg-blue-700 transition-colors"
                  >
                    <PlusIcon className="h-4 w-4 mr-2" />
                    Register Server
                  </button>
                )}
              </div>
            ) : (
              <div className="space-y-6">
                {registryIds.map(registryId => {
                  const registryServers = serversByRegistry[registryId] || [];
                  // Apply active filter to registry servers
                  let filteredRegistryServers = registryServers;
                  if (activeFilter === 'enabled') filteredRegistryServers = registryServers.filter(s => s.enabled);
                  else if (activeFilter === 'disabled') filteredRegistryServers = registryServers.filter(s => !s.enabled);
                  else if (activeFilter === 'unhealthy') filteredRegistryServers = registryServers.filter(s => s.status === 'unhealthy');

                  // Apply sidebar tag filter
                  if (selectedTags.length > 0) {
                    filteredRegistryServers = filteredRegistryServers.filter(s => matchesSelectedTags(s.tags));
                  }

                  // Apply #tag and text search from search box
                  if (parsedSearch.hashTags.length > 0) {
                    filteredRegistryServers = filteredRegistryServers.filter(s => matchesHashTags(s.tags));
                  }
                  if (parsedSearch.textQuery) {
                    const query = parsedSearch.textQuery;
                    filteredRegistryServers = filteredRegistryServers.filter(server =>
                      server.name.toLowerCase().includes(query) ||
                      (server.description || '').toLowerCase().includes(query) ||
                      server.path.toLowerCase().includes(query) ||
                      (server.tags || []).some(tag => tag.toLowerCase().includes(query))
                    );
                  }

                  if (filteredRegistryServers.length === 0) return null;

                  const isExpanded = expandedRegistries[registryId] !== false;  // Default to expanded
                  const displayName = registryId === 'local'
                    ? 'Local Registry'
                    : registryId.replace('peer-registry-', '').replace('peer-', '').toUpperCase() + ' (Federated)';

                  // When there's only one registry (local), skip the collapsible wrapper
                  const showRegistryHeader = registryIds.length > 1 || registryId !== 'local';

                  const pagedServers = filteredRegistryServers.slice(
                    serverPage * PAGE_SIZE,
                    (serverPage + 1) * PAGE_SIZE,
                  );
                  const localVirtualCount = registryId === 'local' ? filteredVirtualServers.length : 0;
                  const serverCount = filteredRegistryServers.length + localVirtualCount;

                  const renderServerCard = (server: Server) => (
                    <ServerCard
                      key={server.path}
                      server={server}
                      onToggle={handleToggleServer}
                      onEdit={handleEditServer}
                      canModify={user?.can_modify_servers || false}
                      canHealthCheck={user?.is_admin || hasUiPermission('health_check_service', server.path)}
                      canToggle={user?.is_admin || hasUiPermission('toggle_service', server.path)}
                      canDelete={(user?.is_admin || hasUiPermission('delete_service', server.path)) && !server.sync_metadata?.is_federated}
                      onDelete={handleDeleteServer}
                      onRefreshSuccess={refreshData}
                      onShowToast={showToast}
                      onServerUpdate={handleServerUpdate}
                      authToken={agentApiToken}
                    />
                  );

                  // Virtual MCP servers are interleaved into the local registry group.
                  const virtualCards = registryId === 'local'
                    ? filteredVirtualServers.map((vs) => (
                        <VirtualServerCard
                          key={vs.path}
                          virtualServer={vs}
                          canModify={user?.can_modify_servers || user?.is_admin || false}
                          onToggle={handleToggleVirtualServer}
                          onEdit={handleEditVirtualServer}
                          onDelete={handleDeleteVirtualServer}
                          onShowToast={showToast}
                          authToken={agentApiToken}
                        />
                      ))
                    : null;

                  return (
                    <RegistrySection
                      key={registryId}
                      registryId={registryId}
                      domId={`server-registry-${registryId}`}
                      items={pagedServers}
                      expanded={isExpanded}
                      onToggle={() => toggleRegistryGroup(registryId)}
                      renderCard={renderServerCard}
                      showHeader={showRegistryHeader}
                      endpointUrl={registryId === 'local' ? localRegistryUrl : peerRegistryEndpoints[registryId]}
                      countLabel={`${serverCount} server${serverCount !== 1 ? 's' : ''}`}
                      displayName={displayName}
                      accent={SERVER_REGISTRY_ACCENT}
                      onResync={(e) => handleSyncPeer(registryId, e)}
                      syncing={syncingPeer === registryId}
                      extraCards={virtualCards}
                    />
                  );
                })}
              </div>
            )}
          </div>
        )}


      {/* Agents Section - Grouped by Registry */}
      {registryConfig?.features.agents !== false &&
        (viewFilter === 'agents') && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                Agents
              </h2>

              {/* Registry Quick Navigation for Agents - Only show if there are multiple registries */}
              {agentRegistryIds.length > 1 && filteredAgents.length > 0 && (
                <div className="flex items-center gap-2">
                  <span className="text-xs text-gray-500 dark:text-gray-400 mr-1">Jump to:</span>
                  {agentRegistryIds.map(registryId => {
                    const count = (agentsByRegistry[registryId] || []).length;
                    if (count === 0) return null;
                    const displayName = registryId === 'local'
                      ? 'Local'
                      : registryId.replace('peer-registry-', '').replace('peer-', '').toUpperCase();
                    const isLocal = registryId === 'local';

                    return (
                      <button
                        key={registryId}
                        onClick={() => {
                          // Expand this registry, collapse others (for both agents and servers)
                          const newExpanded: Record<string, boolean> = {};
                          // Update agent registry states
                          agentRegistryIds.forEach(id => {
                            newExpanded[`agents-${id}`] = (id === registryId);
                          });
                          // Also update server registry states to keep them in sync
                          registryIds.forEach(id => {
                            newExpanded[id] = (id === registryId);
                          });
                          setExpandedRegistries(prev => ({ ...prev, ...newExpanded }));
                          // Scroll to the section
                          const element = document.getElementById(`agent-registry-${registryId}`);
                          if (element) {
                            element.scrollIntoView({ behavior: 'smooth', block: 'start' });
                          }
                        }}
                        className={`px-3 py-1.5 text-xs font-medium rounded-full transition-all hover:scale-105 ${
                          isLocal
                            ? 'bg-green-100 text-green-700 hover:bg-green-200 dark:bg-green-900/30 dark:text-green-300 dark:hover:bg-green-900/50 border border-green-200 dark:border-green-700'
                            : 'bg-violet-100 text-violet-700 hover:bg-violet-200 dark:bg-violet-900/30 dark:text-violet-300 dark:hover:bg-violet-900/50 border border-violet-200 dark:border-violet-700'
                        }`}
                      >
                        {displayName}
                        <span className="ml-1.5 px-1.5 py-0.5 text-[10px] bg-white/50 dark:bg-black/20 rounded-full">
                          {count}
                        </span>
                      </button>
                    );
                  })}
                </div>
              )}
            </div>

            {agentTotalPages > 1 && (
              <div className="flex justify-center mb-4">
                <Pagination
                  currentPage={agentPage}
                  totalPages={agentTotalPages}
                  totalItems={filteredAgents.length}
                  pageSize={PAGE_SIZE}
                  onPageChange={setAgentPage}
                />
              </div>
            )}

            {agentsError ? (
              <div className="text-center py-12 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                <div className="text-red-500 text-lg mb-2">Failed to load agents</div>
                <p className="text-red-600 dark:text-red-400 text-sm">{agentsError}</p>
              </div>
            ) : loading ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-600"></div>
              </div>
            ) : filteredAgents.length === 0 ? (
              <div className="text-center py-12 bg-cyan-50 dark:bg-cyan-900/20 rounded-lg border border-cyan-200 dark:border-cyan-800">
                <div className="text-gray-400 text-lg mb-2">No agents found</div>
                <p className="text-gray-500 dark:text-gray-300 text-sm">
                  {searchTerm || activeFilter !== 'all'
                    ? 'Press Enter in the search bar to search semantically'
                    : 'No agents are registered yet'}
                </p>
              </div>
            ) : (
              <div className="space-y-6">
                {agentRegistryIds.map(registryId => {
                  const registryAgents = agentsByRegistry[registryId] || [];
                  // Apply active filter to registry agents
                  let filteredRegistryAgents = registryAgents;
                  if (activeFilter === 'enabled') filteredRegistryAgents = registryAgents.filter(a => a.enabled);
                  else if (activeFilter === 'disabled') filteredRegistryAgents = registryAgents.filter(a => !a.enabled);
                  else if (activeFilter === 'unhealthy') filteredRegistryAgents = registryAgents.filter(a => a.status === 'unhealthy');

                  // Apply sidebar tag filter
                  if (selectedTags.length > 0) {
                    filteredRegistryAgents = filteredRegistryAgents.filter(a => matchesSelectedTags(a.tags));
                  }

                  // Apply #tag and text search from search box
                  if (parsedSearch.hashTags.length > 0) {
                    filteredRegistryAgents = filteredRegistryAgents.filter(a => matchesHashTags(a.tags));
                  }
                  if (parsedSearch.textQuery) {
                    const query = parsedSearch.textQuery;
                    filteredRegistryAgents = filteredRegistryAgents.filter(agent =>
                      agent.name.toLowerCase().includes(query) ||
                      (agent.description || '').toLowerCase().includes(query) ||
                      agent.path.toLowerCase().includes(query) ||
                      (agent.tags || []).some(tag => tag.toLowerCase().includes(query))
                    );
                  }

                  if (filteredRegistryAgents.length === 0) return null;

                  const isExpanded = expandedRegistries[`agents-${registryId}`] !== false;  // Default to expanded
                  const displayName = registryId === 'local'
                    ? 'Local Registry'
                    : registryId.replace('peer-registry-', '').replace('peer-', '').toUpperCase() + ' (Federated)';

                  // When there's only one registry (local), skip the collapsible wrapper
                  const showRegistryHeader = agentRegistryIds.length > 1 || registryId !== 'local';

                  const pagedAgents = filteredRegistryAgents.slice(
                    agentPage * PAGE_SIZE,
                    (agentPage + 1) * PAGE_SIZE,
                  );

                  const renderAgentCard = (agent: Agent) => (
                    <AgentCard
                      key={agent.path}
                      agent={agent}
                      onToggle={handleToggleAgent}
                      onEdit={handleEditAgent}
                      canModify={user?.can_modify_servers || false}
                      canHealthCheck={user?.is_admin || hasUiPermission('health_check_agent', agent.path)}
                      canToggle={user?.is_admin || hasUiPermission('toggle_agent', agent.path)}
                      canDelete={
                        (user?.is_admin ||
                        hasUiPermission('delete_agent', agent.path) ||
                        agent.registered_by === user?.username) &&
                        !agent.sync_metadata?.is_federated
                      }
                      onDelete={handleDeleteAgent}
                      onRefreshSuccess={refreshData}
                      onShowToast={showToast}
                      onAgentUpdate={handleAgentUpdate}
                      authToken={agentApiToken}
                    />
                  );

                  return (
                    <RegistrySection
                      key={registryId}
                      registryId={registryId}
                      domId={`agent-registry-${registryId}`}
                      items={pagedAgents}
                      expanded={isExpanded}
                      onToggle={() => toggleRegistryGroup(`agents-${registryId}`)}
                      renderCard={renderAgentCard}
                      showHeader={showRegistryHeader}
                      endpointUrl={registryId === 'local' ? localRegistryUrl : peerRegistryEndpoints[registryId]}
                      countLabel={`${filteredRegistryAgents.length} agent${filteredRegistryAgents.length !== 1 ? 's' : ''}`}
                      displayName={displayName}
                      accent={AGENT_REGISTRY_ACCENT}
                      onResync={(e) => handleSyncPeer(registryId, e)}
                      syncing={syncingPeer === registryId}
                    />
                  );
                })}
              </div>
            )}
          </div>
        )}


      {/* Agent Skills Section */}
      {registryConfig?.features.skills !== false &&
        (viewFilter === 'skills') && (
          <SkillsSection
            paginatedSkills={paginatedSkills}
            filteredCount={filteredSkills.length}
            loading={skillsLoading}
            error={skillsError}
            isFiltered={!!searchTerm || activeFilter !== 'all'}
            canModify={user?.can_modify_servers || false}
            page={skillPage}
            totalPages={skillTotalPages}
            pageSize={PAGE_SIZE}
            onPageChange={setSkillPage}
            authToken={agentApiToken}
            onAddSkill={() => handleOpenSkillModal()}
            onToggle={handleToggleSkill}
            onEdit={handleEditSkill}
            onDelete={(path: string) => setShowDeleteSkillConfirm(path)}
            onRefreshSuccess={refreshSkills}
            onShowToast={showToast}
            onSkillUpdate={handleSkillUpdate}
            canToggleSkill={(skill) =>
              user?.is_admin || hasUiPermission('toggle_skill', skill.path)
            }
          />
        )}


      {/* Virtual MCP Servers Section */}
      {registryConfig?.features.virtual_servers !== false &&
        (viewFilter === 'virtual') && (
          <VirtualServersSection
            servers={filteredVirtualServers}
            loading={virtualServersLoading}
            error={virtualServersError}
            isFiltered={!!searchTerm || activeFilter !== 'all'}
            canModify={user?.can_modify_servers || user?.is_admin || false}
            authToken={agentApiToken}
            onAdd={() => navigate('/settings/virtual-mcp/servers')}
            onToggle={handleToggleVirtualServer}
            onEdit={handleEditVirtualServer}
            onDelete={handleDeleteVirtualServer}
            onShowToast={showToast}
          />
        )}

      {/* External Registries Section */}
      {registryConfig?.features.federation !== false && viewFilter === 'external' && (
        <ExternalRegistriesSection
          availableSources={availableExternalSources}
          sourceLabels={externalSourceLabels}
          ardSources={dynamicExternalSources}
          activeSource={externalSourceTab}
          onSelectSource={setExternalSourceTab}
          servers={filteredExternalServers}
          agents={filteredExternalAgents}
          skills={filteredExternalSkills}
          hasAnyExternal={
            externalServers.length > 0 ||
            externalAgents.length > 0 ||
            externalSkills.length > 0
          }
          renderServerCard={(server) => (
            <ServerCard
              key={server.path}
              server={server}
              onToggle={handleToggleServer}
              onEdit={handleEditServer}
              canModify={user?.can_modify_servers || false}
              canDelete={(user?.is_admin || hasUiPermission('delete_service', server.path)) && !server.sync_metadata?.is_federated}
              onRefreshSuccess={refreshData}
              onShowToast={showToast}
              onServerUpdate={handleServerUpdate}
              onDelete={handleDeleteServer}
              authToken={agentApiToken}
            />
          )}
          renderAgentCard={(agent) => (
            <AgentCard
              key={agent.path}
              agent={agent}
              onToggle={handleToggleAgent}
              onEdit={handleEditAgent}
              canModify={user?.can_modify_servers || false}
              canHealthCheck={user?.is_admin || hasUiPermission('health_check_agent', agent.path)}
              canToggle={user?.is_admin || hasUiPermission('toggle_agent', agent.path)}
              canDelete={
                (user?.is_admin ||
                hasUiPermission('delete_agent', agent.path) ||
                agent.registered_by === user?.username) &&
                !agent.sync_metadata?.is_federated
              }
              onDelete={handleDeleteAgent}
              onRefreshSuccess={refreshData}
              onShowToast={showToast}
              onAgentUpdate={handleAgentUpdate}
            />
          )}
          renderSkillCard={(skill) => (
            <SkillCard
              key={skill.path}
              skill={skill}
              onToggle={handleToggleSkill}
              onEdit={handleEditSkill}
              onDelete={(path) => setShowDeleteSkillConfirm(path)}
              canModify={user?.can_modify_servers || false}
              canToggle={user?.is_admin || hasUiPermission('toggle_skill', skill.path)}
              onRefreshSuccess={refreshSkills}
              onShowToast={showToast}
              onSkillUpdate={handleSkillUpdate}
            />
          )}
        />
      )}

      {/* Empty state when all are filtered out */}
      {((viewFilter === 'servers' && filteredServers.length === 0) ||
        (viewFilter === 'agents' && filteredAgents.length === 0) ||
        (viewFilter === 'skills' && filteredSkills.length === 0) ||
        (viewFilter === 'virtual' && filteredVirtualServers.length === 0)) &&
        (searchTerm || activeFilter !== 'all' || selectedTags.length > 0) && (
          <div className="text-center py-16">
            <div className="text-gray-400 text-xl mb-4">No items found</div>
            <p className="text-gray-500 dark:text-gray-300 text-base max-w-md mx-auto">
              {selectedTags.length > 0
                ? `No items match the selected tag${selectedTags.length > 1 ? 's' : ''}: ${selectedTags.join(', ')}`
                : 'Press Enter in the search bar to search semantically'}
            </p>
          </div>
        )}
    </>
  );

  // Show error state
  if (error && agentsError) {
    return (
      <div className="flex flex-col items-center justify-center h-64 space-y-4">
        <div className="text-red-500 text-lg">Failed to load servers and agents</div>
        <p className="text-gray-500 text-center">{error}</p>
        <p className="text-gray-500 text-center">{agentsError}</p>
        <button
          onClick={handleRefreshHealth}
          className="px-4 py-2 bg-purple-600 text-white rounded-lg hover:bg-purple-700 transition-colors"
        >
          Try Again
        </button>
      </div>
    );
  }

  // Show loading state
  if (loading) {
    return (
      <div className="flex items-center justify-center h-64">
        <div className="animate-spin rounded-full h-12 w-12 border-b-2 border-purple-600"></div>
      </div>
    );
  }

  return (
    <>
      {/* Toast Notification */}
      {toast && (
        <Toast
          message={toast.message}
          type={toast.type}
          onClose={hideToast}
        />
      )}

      <div className="flex flex-col h-full">
        {/* Fixed Header Section */}
        <div className="flex-shrink-0 space-y-4 pb-4">
          {/* View Filter Tabs - conditionally show based on registry mode */}
          {/* Calculate if multiple features are enabled to determine if "All" tab is needed */}
          <div className="flex gap-2 border-b border-gray-200 dark:border-gray-700 overflow-x-auto">
            <button
              onClick={() => handleChangeViewFilter('discover')}
              className={`px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors border-b-2 ${
                viewFilter === 'discover'
                  ? 'border-indigo-500 text-indigo-600 dark:text-indigo-400'
                  : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
              }`}
            >
              Discover
            </button>
            {registryConfig?.features.mcp_servers !== false && (
              <button
                onClick={() => handleChangeViewFilter('servers')}
                className={`px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors border-b-2 ${
                  viewFilter === 'servers'
                    ? 'border-blue-500 text-blue-600 dark:text-blue-400'
                    : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                MCP Servers
              </button>
            )}
            {registryConfig?.features.virtual_servers !== false && (
              <button
                onClick={() => handleChangeViewFilter('virtual')}
                className={`px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors border-b-2 ${
                  viewFilter === 'virtual'
                    ? 'border-teal-500 text-teal-600 dark:text-teal-400'
                    : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                Virtual MCP Servers
              </button>
            )}
            {registryConfig?.features.agents !== false && (
              <button
                onClick={() => handleChangeViewFilter('agents')}
                className={`px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors border-b-2 ${
                  viewFilter === 'agents'
                    ? 'border-cyan-500 text-cyan-600 dark:text-cyan-400'
                    : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                Agents
              </button>
            )}
            {registryConfig?.features.skills !== false && (
              <button
                onClick={() => handleChangeViewFilter('skills')}
                className={`px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors border-b-2 ${
                  viewFilter === 'skills'
                    ? 'border-amber-500 text-amber-600 dark:text-amber-400'
                    : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                Agent Skills
              </button>
            )}
            {/* Custom entity type tabs render before External Registries, which
                is always the last tab. */}
            {registryConfig?.features.custom_types &&
              (registryConfig?.custom_types ?? []).map((ct) => {
                const filter = `custom:${ct.name}` as const;
                return (
                  <button
                    key={ct.name}
                    onClick={() => handleChangeViewFilter(filter)}
                    className={`px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors border-b-2 ${
                      viewFilter === filter
                        ? 'border-purple-500 text-purple-600 dark:text-purple-400'
                        : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                    }`}
                  >
                    {ct.display_name}
                  </button>
                );
              })}
            {registryConfig?.features.federation !== false && (
              <button
                onClick={() => handleChangeViewFilter('external')}
                className={`px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors border-b-2 ${
                  viewFilter === 'external'
                    ? 'border-green-500 text-green-600 dark:text-green-400'
                    : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                External Registries
              </button>
            )}
          </div>

          {viewFilter !== 'discover' && !currentCustomType && (
          <>
          {/* Search Bar and Refresh Button */}
          <div className="flex gap-4 items-center">
            <div className="relative flex-1">
              <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
                <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
              </div>
              <input
                type="text"
                placeholder="Search servers, agents, descriptions, or tags… (Press Enter to run semantic search; typing filters locally.)"
                className="input pl-10 w-full"
                value={searchTerm}
                onChange={(e) => setSearchTerm(e.target.value)}
                onKeyDown={(e) => {
                  if (e.key === 'Enter') {
                    e.preventDefault();
                    handleSemanticSearch();
                  }
                }}
              />
              {searchTerm && (
                <button
                  type="button"
                  onClick={handleClearSearch}
                  className="absolute inset-y-0 right-0 flex items-center pr-3 text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
                >
                  <XMarkIcon className="h-4 w-4" />
                </button>
              )}
            </div>

            {viewFilter !== 'skills' && viewFilter !== 'virtual' && (
              <button
                onClick={handleRegisterServer}
                className="btn-primary flex items-center space-x-2 flex-shrink-0"
              >
                <PlusIcon className="h-4 w-4" />
                <span>Register</span>
              </button>
            )}

            <button
              onClick={handleRefreshHealth}
              disabled={refreshing}
              className="btn-secondary flex items-center space-x-2 flex-shrink-0"
            >
              <ArrowPathIcon className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
              <span>Refresh Health</span>
            </button>
          </div>

          {/* Results count and lifecycle filter chips */}
          <div className="flex items-center justify-between">
            <div className="flex items-center gap-3">
              <p className="text-sm text-gray-500 dark:text-gray-300">
                {semanticSectionVisible ? (
                  <>
                    Showing {semanticServers.length} servers, {semanticAgents.length} agents
                  </>
                ) : (
                  <>
                    Showing{' '}
                    {registryConfig?.features.mcp_servers !== false && (
                      <>{filteredServers.length} servers</>
                    )}
                    {registryConfig?.features.mcp_servers !== false && registryConfig?.features.agents !== false && ', '}
                    {registryConfig?.features.agents !== false && (
                      <>{filteredAgents.length} agents</>
                    )}
                    {(registryConfig?.features.mcp_servers !== false || registryConfig?.features.agents !== false) && registryConfig?.features.skills !== false && ', '}
                    {registryConfig?.features.skills !== false && (
                      <>{filteredSkills.length} skills</>
                    )}
                    {customCounts.map((ct) => (
                      <React.Fragment key={ct.name}>
                        , {ct.count} {ct.displayName}
                      </React.Fragment>
                    ))}
                  </>
                )}
              </p>
            </div>
            <p className="text-xs text-gray-400 dark:text-gray-500">
              Press Enter to run semantic search; typing filters locally.
            </p>
          </div>
          </>
          )}
        </div>

        {/* Scrollable Content Area */}
        <div className="flex-1 overflow-y-auto min-h-0 space-y-10">
          {currentCustomType ? (
            <CustomEntityTab
              key={currentCustomType.name}
              typeName={currentCustomType.name}
              displayName={currentCustomType.display_name}
              user={user}
              selectedTags={selectedTags}
              authToken={agentApiToken}
              onShowToast={showToast}
            />
          ) : viewFilter === 'discover' ? (
            <DiscoverTab
              servers={filteredServers}
              agents={filteredAgents}
              skills={filteredSkills}
              virtualServers={filteredVirtualServers}
              externalServers={filteredExternalServers}
              externalAgents={filteredExternalAgents}
              customSections={customSections}
              loading={loading || skillsLoading || virtualServersLoading}
              onServerToggle={handleToggleServer}
              onServerEdit={handleEditServer}
              onServerDelete={handleDeleteServer}
              onAgentToggle={handleToggleAgent}
              onAgentEdit={handleEditAgent}
              onAgentDelete={handleDeleteAgent}
              onSkillToggle={handleToggleSkill}
              onSkillEdit={handleEditSkill}
              onSkillDelete={handleDeleteSkill}
              onVirtualServerToggle={handleToggleVirtualServer}
              onVirtualServerEdit={handleEditVirtualServer}
              onVirtualServerDelete={handleDeleteVirtualServer}
              onShowToast={showToast}
              authToken={agentApiToken}
            />
          ) : (
            <>
              {semanticSectionVisible ? (
                <>
                  <SemanticSearchResults
                    query={semanticDisplayQuery}
                    loading={semanticLoading}
                    error={semanticError}
                    servers={semanticServers}
                    tools={semanticTools}
                    agents={semanticAgents}
                    skills={semanticSkills}
                    virtualServers={semanticVirtualServers}
                    custom={semanticCustom}
                  />

                  {shouldShowFallbackGrid && (
                    <div className="border-t border-gray-200 dark:border-gray-700 pt-6">
                      <div className="flex items-center justify-between mb-4">
                        <h4 className="text-base font-semibold text-gray-900 dark:text-gray-200">
                          Keyword search fallback
                        </h4>
                        {semanticError && (
                          <span className="text-xs font-medium text-red-500">
                            Showing local matches because semantic search is unavailable
                          </span>
                        )}
                      </div>
                      {renderDashboardCollections()}
                    </div>
                  )}
                </>
              ) : (
                renderDashboardCollections()
              )}
            </>
          )}
        </div>
      </div>

      {/* Register Server Modal */}
      {showRegisterModal && (
        <ServerRegisterModal
          form={registerForm}
          setForm={setRegisterForm}
          loading={registerLoading}
          onSubmit={handleRegisterSubmit}
          onClose={() => setShowRegisterModal(false)}
        />
      )}

      {/* Edit Server Modal */}
      {editingServer && (
        <ServerEditModal
          serverName={editingServer.name}
          form={editForm}
          setForm={setEditForm}
          loading={editLoading}
          egressEnabled={egressEnabled}
          onSave={handleSaveEdit}
          onClose={handleCloseEdit}
        />
      )}

      {/* Edit Agent Modal */}
      {editingAgent && (
        <AgentEditModal
          agentName={editingAgent.name}
          form={editAgentForm}
          setForm={setEditAgentForm}
          loading={editAgentLoading}
          skillsJsonError={skillsJsonError}
          onSkillsJsonChange={() => setSkillsJsonError(null)}
          onSave={handleSaveEditAgent}
          onClose={handleCloseEdit}
        />
      )}

      <DuplicateCheckModal
        isOpen={showSkillDuplicateModal}
        onClose={closeSkillDuplicateModal}
        onProceed={handleSkillDuplicateProceed}
        onPickExisting={handleSkillDuplicatePickExisting}
        collisionWith={skillCollisionWith}
        advisoryMatches={skillAdvisoryMatches}
        isLoading={skillFormLoading}
      />

      {/* Register/Edit Skill Modal (create + edit share one modal) */}
      {showSkillModal && (
        <SkillFormModal
          editing={editingSkill ? { name: editingSkill.name, path: editingSkill.path } : null}
          form={skillForm}
          setForm={setSkillForm}
          loading={skillFormLoading}
          autoFill={skillAutoFill}
          setAutoFill={setSkillAutoFill}
          parseLoading={skillParseLoading}
          onParse={handleParseSkillMd}
          onSubmit={handleSaveSkill}
          onClose={handleCloseSkillModal}
        />
      )}

      {/* Delete Skill Confirmation Modal */}
      {showDeleteSkillConfirm && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-sm">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              Delete Skill
            </h3>
            <p className="text-gray-600 dark:text-gray-300 mb-4">
              Are you sure you want to delete this skill? This action cannot be undone.
            </p>
            <div className="flex space-x-3">
              <button
                onClick={() => handleDeleteSkill(showDeleteSkillConfirm)}
                className="flex-1 px-4 py-2 text-sm font-medium text-white bg-red-600 hover:bg-red-700 rounded-md transition-colors"
              >
                Delete
              </button>
              <button
                onClick={() => setShowDeleteSkillConfirm(null)}
                className="flex-1 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-800 rounded-md transition-colors"
              >
                Cancel
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Virtual Server Delete Confirmation Modal */}
      {deleteVirtualServerTarget && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
          role="dialog"
          aria-modal="true"
          aria-label="Delete virtual server confirmation"
        >
          <div className="bg-white dark:bg-gray-800 rounded-lg shadow-xl max-w-md w-full mx-4 p-6">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-2">
              Delete Virtual Server
            </h3>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-4">
              This action is irreversible. The virtual server and all its tool
              mappings will be permanently removed.
            </p>
            <p className="text-sm text-gray-600 dark:text-gray-400 mb-3">
              Type <strong>{deleteVirtualServerTarget.server_name}</strong> to confirm:
            </p>
            <input
              type="text"
              value={deleteVirtualServerTypedName}
              onChange={(e) => setDeleteVirtualServerTypedName(e.target.value)}
              placeholder={deleteVirtualServerTarget.server_name}
              disabled={deletingVirtualServer}
              className="w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-lg
                         bg-white dark:bg-gray-900 text-gray-900 dark:text-white mb-4"
              onKeyDown={(e) => {
                if (e.key === 'Escape') {
                  setDeleteVirtualServerTarget(null);
                  setDeleteVirtualServerTypedName('');
                }
              }}
              autoFocus
            />
            <div className="flex justify-end space-x-3">
              <button
                onClick={() => {
                  setDeleteVirtualServerTarget(null);
                  setDeleteVirtualServerTypedName('');
                }}
                disabled={deletingVirtualServer}
                className="px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200
                           rounded-lg hover:bg-gray-300 dark:hover:bg-gray-800 disabled:opacity-50"
              >
                Cancel
              </button>
              <button
                onClick={confirmDeleteVirtualServer}
                disabled={deleteVirtualServerTypedName !== deleteVirtualServerTarget.server_name || deletingVirtualServer}
                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700
                           disabled:opacity-50 disabled:cursor-not-allowed flex items-center"
              >
                {deletingVirtualServer && (
                  <ArrowPathIcon className="h-4 w-4 mr-2 animate-spin" />
                )}
                Delete
              </button>
            </div>
          </div>
        </div>
      )}

      {/* Virtual Server Edit Modal */}
      {showVirtualServerForm && (
        <div
          className="fixed inset-0 z-50 flex items-center justify-center bg-black/50"
          role="dialog"
          aria-modal="true"
          aria-label="Edit virtual server"
        >
          <div className="bg-white dark:bg-gray-800 rounded-xl shadow-xl max-w-4xl w-full mx-4 max-h-[90vh] overflow-auto">
            {editingVirtualServerLoading ? (
              <div className="flex items-center justify-center py-16">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-teal-600"></div>
                <span className="ml-3 text-gray-500 dark:text-gray-400">Loading virtual server...</span>
              </div>
            ) : editingVirtualServer ? (
              <VirtualServerForm
                virtualServer={editingVirtualServer}
                onSave={handleSaveVirtualServer}
                onCancel={handleCancelVirtualServerEdit}
              />
            ) : (
              <div className="p-6 text-center">
                <p className="text-gray-500 dark:text-gray-400">Failed to load virtual server</p>
                <button
                  onClick={handleCancelVirtualServerEdit}
                  className="mt-4 px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-800"
                >
                  Close
                </button>
              </div>
            )}
          </div>
        </div>
      )}

      {/* Custom entity modals for the Discover view (view / edit / delete) */}
      {customViewing && (
        <CustomEntityDetail
          descriptor={customViewing.descriptor}
          record={customViewing.record}
          onClose={() => setCustomViewing(null)}
        />
      )}

      {customEditing && (
        <CustomEntityForm
          descriptor={customEditing.descriptor}
          record={customEditing.record}
          onSave={handleCustomSave}
          onCancel={() => setCustomEditing(null)}
        />
      )}

      <ConfirmModal
        isOpen={!!customDeleting}
        onClose={() => setCustomDeleting(null)}
        onConfirm={handleCustomDelete}
        title="Delete record"
        message={`Are you sure you want to delete "${customDeleting?.record.name}"? This cannot be undone.`}
        confirmLabel="Delete"
        loadingLabel="Deleting..."
        isDestructive
        isLoading={customDeleteLoading}
      />

    </>
  );
};

export default Dashboard;
