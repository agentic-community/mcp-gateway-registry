import React, { useState, useMemo, useCallback, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { MagnifyingGlassIcon, PlusIcon, XMarkIcon, ArrowPathIcon, CheckCircleIcon, ExclamationCircleIcon, ChevronDownIcon, ChevronRightIcon } from '@heroicons/react/24/outline';
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
import axios from 'axios';


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
  status?: 'healthy' | 'healthy-auth-expired' | 'unhealthy' | 'unknown';
  num_tools?: number;
  proxy_pass_url?: string;
  license?: string;
  mcp_endpoint?: string;
  metadata?: Record<string, unknown>;
  sync_metadata?: SyncMetadata;
  auth_scheme?: string;
  auth_header_name?: string;
}

interface Agent {
  name: string;
  path: string;
  url?: string;
  description?: string;
  version?: string;
  visibility?: 'public' | 'private' | 'group-restricted';
  trust_level?: 'community' | 'verified' | 'trusted' | 'unverified';
  enabled: boolean;
  tags?: string[];
  last_checked_time?: string;
  usersCount?: number;
  rating?: number;
  status?: 'healthy' | 'healthy-auth-expired' | 'unhealthy' | 'unknown';
  sync_metadata?: SyncMetadata;
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
}

const Dashboard: React.FC<DashboardProps> = ({ activeFilter = 'all' }) => {
  const navigate = useNavigate();
  const { servers, agents: agentsFromStats, loading, error, refreshData, setServers, setAgents } = useServerStats();
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
  const [showRegisterModal, setShowRegisterModal] = useState(false);
  const [registerForm, setRegisterForm] = useState({
    name: '',
    path: '',
    proxyPass: '',
    description: '',
    official: false,
    tags: [] as string[]
  });
  const [registerLoading, setRegisterLoading] = useState(false);
  const [refreshing, setRefreshing] = useState(false);
  const [editingServer, setEditingServer] = useState<Server | null>(null);
  const [editForm, setEditForm] = useState({
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
  });
  const [editLoading, setEditLoading] = useState(false);
  const [toast, setToast] = useState<{ message: string; type: 'success' | 'error' } | null>(null);

  // Agent state management - using agents from useServerStats hook instead of separate fetch
  // Agents loading state is now handled by the useServerStats hook's 'loading' state
  const [agentsError, setAgentsError] = useState<string | null>(null);
  const [editingAgent, setEditingAgent] = useState<Agent | null>(null);
  const [agentApiToken, setAgentApiToken] = useState<string | null>(null);

  // View filter state
  const [viewFilter, setViewFilter] = useState<'all' | 'servers' | 'agents' | 'skills' | 'virtual' | 'external'>('all');

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

  // Get the local registry URL
  const localRegistryUrl = useMemo(() => {
    return window.location.origin;
  }, []);

  const [editAgentForm, setEditAgentForm] = useState({
    name: '',
    path: '',
    description: '',
    version: '',
    visibility: 'private' as 'public' | 'private' | 'group-restricted',
    trust_level: 'community' as 'community' | 'verified' | 'trusted' | 'unverified',
    tags: [] as string[]
  });
  const [editAgentLoading, setEditAgentLoading] = useState(false);

  // Skill state management
  const [showSkillModal, setShowSkillModal] = useState(false);
  const [editingSkill, setEditingSkill] = useState<Skill | null>(null);
  const [skillForm, setSkillForm] = useState({
    name: '',
    description: '',
    skill_md_url: '',
    repository_url: '',
    version: '',
    visibility: 'public' as 'public' | 'private' | 'group',
    tags: '',  // Raw string, parsed on save
    target_agents: ''  // Raw string, parsed on save
  });
  const [skillFormLoading, setSkillFormLoading] = useState(false);
  const [showDeleteSkillConfirm, setShowDeleteSkillConfirm] = useState<string | null>(null);
  const [skillAutoFill, setSkillAutoFill] = useState(true);  // Auto-fill from SKILL.md
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
      status: a.status,
      last_checked_time: a.last_checked_time,
      usersCount: a.usersCount,
      url: '',  // Will be populated if needed
      version: '',
      visibility: 'public',
      trust_level: 'community',
      sync_metadata: a.sync_metadata,
      registered_by: a.registered_by,
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
    maxResults: 12,
    enabled: semanticEnabled
  });

  const semanticServers = semanticResults?.servers ?? [];
  const semanticTools = semanticResults?.tools ?? [];
  const semanticAgents = semanticResults?.agents ?? [];
  const semanticSkills = semanticResults?.skills ?? [];
  const semanticVirtualServers = semanticResults?.virtual_servers ?? [];
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
        semanticVirtualServers.length === 0));

  // Filter servers based on activeFilter and searchTerm
  const filteredServers = useMemo(() => {
    let filtered = internalServers;

    // Apply filter first
    if (activeFilter === 'enabled') filtered = filtered.filter(s => s.enabled);
    else if (activeFilter === 'disabled') filtered = filtered.filter(s => !s.enabled);
    else if (activeFilter === 'unhealthy') filtered = filtered.filter(s => s.status === 'unhealthy');

    // Then apply search
    if (searchTerm) {
      const query = searchTerm.toLowerCase();
      filtered = filtered.filter(server =>
        server.name.toLowerCase().includes(query) ||
        (server.description || '').toLowerCase().includes(query) ||
        server.path.toLowerCase().includes(query) ||
        (server.tags || []).some(tag => tag.toLowerCase().includes(query))
      );
    }

    return filtered;
  }, [internalServers, activeFilter, searchTerm]);

  // Filter external servers based on searchTerm
  const filteredExternalServers = useMemo(() => {
    let filtered = externalServers;

    if (searchTerm) {
      const query = searchTerm.toLowerCase();
      filtered = filtered.filter(server =>
        server.name.toLowerCase().includes(query) ||
        (server.description || '').toLowerCase().includes(query) ||
        server.path.toLowerCase().includes(query) ||
        (server.tags || []).some(tag => tag.toLowerCase().includes(query))
      );
    }

    return filtered;
  }, [externalServers, searchTerm]);

  // Filter external agents based on searchTerm
  const filteredExternalAgents = useMemo(() => {
    let filtered = externalAgents;

    if (searchTerm) {
      const query = searchTerm.toLowerCase();
      filtered = filtered.filter(agent =>
        agent.name.toLowerCase().includes(query) ||
        (agent.description || '').toLowerCase().includes(query) ||
        agent.path.toLowerCase().includes(query) ||
        (agent.tags || []).some(tag => tag.toLowerCase().includes(query))
      );
    }

    return filtered;
  }, [externalAgents, searchTerm]);

  // Filter agents based on activeFilter and searchTerm
  const filteredAgents = useMemo(() => {
    let filtered = internalAgents;

    // Apply filter first
    if (activeFilter === 'enabled') filtered = filtered.filter(a => a.enabled);
    else if (activeFilter === 'disabled') filtered = filtered.filter(a => !a.enabled);
    else if (activeFilter === 'unhealthy') filtered = filtered.filter(a => a.status === 'unhealthy');

    // Then apply search
    if (searchTerm) {
      const query = searchTerm.toLowerCase();
      filtered = filtered.filter(agent =>
        agent.name.toLowerCase().includes(query) ||
        (agent.description || '').toLowerCase().includes(query) ||
        agent.path.toLowerCase().includes(query) ||
        (agent.tags || []).some(tag => tag.toLowerCase().includes(query))
      );
    }

    return filtered;
  }, [internalAgents, activeFilter, searchTerm]);

  // Filter skills based on activeFilter and searchTerm
  const filteredSkills = useMemo(() => {
    let filtered = skills;

    // Apply filter first
    if (activeFilter === 'enabled') filtered = filtered.filter(s => s.is_enabled);
    else if (activeFilter === 'disabled') filtered = filtered.filter(s => !s.is_enabled);

    // Then apply search
    if (searchTerm) {
      const query = searchTerm.toLowerCase();
      filtered = filtered.filter(skill =>
        skill.name.toLowerCase().includes(query) ||
        (skill.description || '').toLowerCase().includes(query) ||
        skill.path.toLowerCase().includes(query) ||
        (skill.tags || []).some(tag => tag.toLowerCase().includes(query)) ||
        (skill.author || '').toLowerCase().includes(query)
      );
    }

    return filtered;
  }, [skills, activeFilter, searchTerm]);

  // Filter virtual servers based on activeFilter and searchTerm
  const filteredVirtualServers = useMemo(() => {
    let filtered = virtualServers;

    // Apply filter
    if (activeFilter === 'enabled') filtered = filtered.filter(s => s.is_enabled);
    else if (activeFilter === 'disabled') filtered = filtered.filter(s => !s.is_enabled);

    // Apply search
    if (searchTerm) {
      const query = searchTerm.toLowerCase();
      filtered = filtered.filter(vs =>
        vs.server_name.toLowerCase().includes(query) ||
        (vs.description || '').toLowerCase().includes(query) ||
        vs.path.toLowerCase().includes(query) ||
        (vs.tags || []).some(tag => tag.toLowerCase().includes(query))
      );
    }

    return filtered;
  }, [virtualServers, activeFilter, searchTerm]);

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
    } catch (error) {
      console.error('Failed to sync peer:', error);
      setToast({ message: `Failed to sync from ${peerId}`, type: 'error' });
    } finally {
      setSyncingPeer(null);
    }
  };

  const handleEditServer = useCallback(async (server: Server) => {
    try {
      // Fetch full server details including proxy_pass_url and tags
      const response = await axios.get(`/api/server_details${server.path}`);
      const serverDetails = response.data;

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
      });
    } catch (error) {
      console.error('Failed to fetch server details:', error);
      // Fallback to basic server data
      setEditingServer(server);
      setEditForm({
        name: server.name,
        path: server.path,
        proxyPass: '',
        description: server.description || '',
        tags: server.tags || [],
        license: 'N/A',
        num_tools: server.num_tools || 0,
        mcp_endpoint: server.mcp_endpoint || '',
        metadata: server.metadata ? JSON.stringify(server.metadata, null, 2) : '',
        auth_scheme: server.auth_scheme || 'none',
        auth_credential: '',
        auth_header_name: server.auth_header_name || 'X-API-Key',
      });
    }
  }, []);

  const handleEditAgent = useCallback(async (agent: Agent) => {
    // For now, just populate the form with existing data
    // In the future, we might fetch additional details from an API
    setEditingAgent(agent);
    setEditAgentForm({
      name: agent.name,
      path: agent.path,
      description: agent.description || '',
      version: agent.version || '1.0.0',
      visibility: agent.visibility || 'private',
      trust_level: agent.trust_level || 'community',
      tags: agent.tags || []
    });
  }, []);

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

  const handleSaveEdit = async () => {
    if (editLoading || !editingServer) return;

    try {
      setEditLoading(true);

      const formData = new FormData();
      formData.append('name', editForm.name);
      formData.append('description', editForm.description);
      formData.append('proxy_pass_url', editForm.proxyPass);
      formData.append('tags', editForm.tags.join(','));
      formData.append('license', editForm.license);
      formData.append('num_tools', editForm.num_tools.toString());
      if (editForm.mcp_endpoint) {
        formData.append('mcp_endpoint', editForm.mcp_endpoint);
      }
      if (editForm.metadata) {
        formData.append('metadata', editForm.metadata);
      }
      if (editForm.auth_scheme !== 'none') {
        formData.append('auth_scheme', editForm.auth_scheme);
        if (editForm.auth_credential) {
          formData.append('auth_credential', editForm.auth_credential);
        }
        if (editForm.auth_scheme === 'api_key' && editForm.auth_header_name) {
          formData.append('auth_header_name', editForm.auth_header_name);
        }
      } else {
        formData.append('auth_scheme', 'none');
      }

      // Use the correct edit endpoint with the server path
      await axios.post(`/api/edit${editingServer.path}`, formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });

      // Refresh server list
      await refreshData();
      setEditingServer(null);

      showToast('Server updated successfully!', 'success');
    } catch (error: any) {
      console.error('Failed to update server:', error);
      showToast(error.response?.data?.detail || 'Failed to update server', 'error');
    } finally {
      setEditLoading(false);
    }
  };

  const handleSaveEditAgent = async () => {
    if (editAgentLoading || !editingAgent) return;

    try {
      setEditAgentLoading(true);

      // TODO: Implement agent edit endpoint when backend is ready
      // For now, just show a message
      showToast('Agent editing is not yet implemented', 'error');

      // When backend is ready, uncomment and implement:
      // const formData = new FormData();
      // formData.append('name', editAgentForm.name);
      // formData.append('description', editAgentForm.description);
      // formData.append('version', editAgentForm.version);
      // formData.append('visibility', editAgentForm.visibility);
      // formData.append('trust_level', editAgentForm.trust_level);
      // formData.append('tags', editAgentForm.tags.join(','));
      //
      // await axios.post(`/api/agents${editingAgent.path}/edit`, formData, {
      //   headers: {
      //     'Content-Type': 'application/x-www-form-urlencoded',
      //   },
      // });
      //
      // await fetchAgents();
      // setEditingAgent(null);
      // showToast('Agent updated successfully!', 'success');
    } catch (error: any) {
      console.error('Failed to update agent:', error);
      showToast(error.response?.data?.detail || 'Failed to update agent', 'error');
    } finally {
      setEditAgentLoading(false);
    }
  };

  const handleToggleServer = useCallback(async (path: string, enabled: boolean) => {
    // Optimistically update the UI first
    setServers(prevServers =>
      prevServers.map(server =>
        server.path === path
          ? { ...server, enabled }
          : server
      )
    );

    try {
      const formData = new FormData();
      formData.append('enabled', enabled ? 'on' : 'off');

      await axios.post(`/api/toggle${path}`, formData, {
        headers: {
          'Content-Type': 'application/x-www-form-urlencoded',
        },
      });

      // No need to refresh all data - the optimistic update is enough
      showToast(`Server ${enabled ? 'enabled' : 'disabled'} successfully!`, 'success');
    } catch (error: any) {
      console.error('Failed to toggle server:', error);

      // Revert the optimistic update on error
      setServers(prevServers =>
        prevServers.map(server =>
          server.path === path
            ? { ...server, enabled: !enabled }
            : server
        )
      );

      showToast(error.response?.data?.detail || 'Failed to toggle server', 'error');
    }
  }, [setServers, showToast]);

  const handleDeleteServer = useCallback(async (path: string) => {
    const formData = new FormData();
    formData.append('path', path);

    await axios.post('/api/servers/remove', formData, {
      headers: { 'Content-Type': 'application/x-www-form-urlencoded' },
    });

    // Remove from local state immediately for responsive UI
    setServers(prevServers => prevServers.filter(s => s.path !== path));
    showToast('Server deleted successfully', 'success');
  }, [setServers, showToast]);

  const handleDeleteAgent = useCallback(async (path: string) => {
    await axios.delete(`/api/agents${path}`);

    // Remove from local state immediately for responsive UI
    setAgents(prevAgents => prevAgents.filter(a => a.path !== path));
    showToast('Agent deleted successfully', 'success');
  }, [setAgents, showToast]);

  const handleToggleAgent = useCallback(async (path: string, enabled: boolean) => {
    // Optimistically update the UI first
    setAgents(prevAgents =>
      prevAgents.map(agent =>
        agent.path === path
          ? { ...agent, enabled }
          : agent
      )
    );

    try {
      await axios.post(`/api/agents${path}/toggle?enabled=${enabled}`);

      showToast(`Agent ${enabled ? 'enabled' : 'disabled'} successfully!`, 'success');
    } catch (error: any) {
      console.error('Failed to toggle agent:', error);

      // Revert the optimistic update on error
      setAgents(prevAgents =>
        prevAgents.map(agent =>
          agent.path === path
            ? { ...agent, enabled: !enabled }
            : agent
        )
      );

      showToast(error.response?.data?.detail || 'Failed to toggle agent', 'error');
    }
  }, [setAgents, showToast]);

  const handleServerUpdate = useCallback((path: string, updates: Partial<Server>) => {
    setServers(prevServers =>
      prevServers.map(server =>
        server.path === path
          ? { ...server, ...updates }
          : server
      )
    );
  }, [setServers]);

  const handleToggleSkill = useCallback(async (path: string, enabled: boolean) => {
    // Optimistically update the UI first
    setSkills(prevSkills =>
      prevSkills.map(skill =>
        skill.path === path
          ? { ...skill, is_enabled: enabled }
          : skill
      )
    );

    try {
      // Convert full path to API path (e.g., /skills/pdf -> /pdf)
      const apiPath = path.startsWith('/skills/') ? path.replace('/skills/', '/') : path;
      await axios.post(`/api/skills${apiPath}/toggle`, { enabled });

      showToast(`Skill ${enabled ? 'enabled' : 'disabled'} successfully!`, 'success');
    } catch (error: any) {
      console.error('Failed to toggle skill:', error);

      // Revert the optimistic update on error
      setSkills(prevSkills =>
        prevSkills.map(skill =>
          skill.path === path
            ? { ...skill, is_enabled: !enabled }
            : skill
        )
      );

      showToast(error.response?.data?.detail || 'Failed to toggle skill', 'error');
    }
  }, [setSkills, showToast]);

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
        repository_url: '',
        version: skill.version || '',
        visibility: skill.visibility || 'public',
        tags: (skill.tags || []).join(', '),
        target_agents: (skill.target_agents || []).join(', ')
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
        target_agents: ''
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
      const response = await axios.post(`/api/skills/parse-skill-md?url=${encodeURIComponent(skillForm.skill_md_url)}`);
      const data = response.data;

      if (data.success) {
        setSkillForm(prev => ({
          ...prev,
          name: data.name_slug || prev.name,
          description: data.description || prev.description,
          version: data.version || prev.version,
          tags: data.tags?.length > 0 ? data.tags.join(', ') : prev.tags,
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
  }, [skillForm.skill_md_url, skillParseLoading, showToast]);

  const handleSaveSkill = useCallback(async (e: React.FormEvent) => {
    e.preventDefault();
    if (skillFormLoading) return;

    // Validate name format (lowercase, numbers, hyphens only)
    const nameRegex = /^[a-z0-9]+(-[a-z0-9]+)*$/;
    if (!nameRegex.test(skillForm.name)) {
      showToast('Name must be lowercase letters, numbers, and hyphens only (e.g., "my-skill-name")', 'error');
      return;
    }

    try {
      setSkillFormLoading(true);

      // Parse comma-separated strings into arrays
      const parseTags = (str: string): string[] =>
        str.split(',').map(t => t.trim()).filter(t => t.length > 0);

      const payload = {
        name: skillForm.name,
        description: skillForm.description,
        skill_md_url: skillForm.skill_md_url,
        repository_url: skillForm.repository_url || undefined,
        version: skillForm.version || undefined,
        visibility: skillForm.visibility,
        tags: parseTags(skillForm.tags),
        target_agents: parseTags(skillForm.target_agents)
      };

      if (editingSkill) {
        // Update existing skill
        await axios.put(`/api/skills${editingSkill.path}`, payload);
        showToast('Skill updated successfully!', 'success');
      } else {
        // Create new skill
        await axios.post('/api/skills', payload);
        showToast('Skill registered successfully!', 'success');
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
  }, [skillForm, skillFormLoading, editingSkill, refreshSkills, showToast, handleCloseSkillModal]);

  const handleEditSkill = useCallback((skill: Skill) => {
    handleOpenSkillModal(skill);
  }, [handleOpenSkillModal]);

  const handleDeleteSkill = useCallback(async (path: string) => {
    try {
      await axios.delete(`/api/skills${path}`);

      // Remove from local state immediately for responsive UI
      setSkills(prevSkills => prevSkills.filter(s => s.path !== path));
      showToast('Skill deleted successfully', 'success');
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
    } catch (error: any) {
      console.error('Failed to register server:', error);
      showToast(error.response?.data?.detail || 'Failed to register server', 'error');
    } finally {
      setRegisterLoading(false);
    }
  }, [registerForm, registerLoading, refreshData, showToast]);

  const renderServerGrid = (
    list: Server[],
    options?: { emptyTitle?: string; emptySubtitle?: string; showRegisterCta?: boolean }
  ) => {
    if (list.length === 0) {
      const title = options?.emptyTitle ?? 'No servers found';
      const subtitle =
        options?.emptySubtitle ??
        (searchTerm || activeFilter !== 'all'
          ? 'Press Enter in the search bar to search semantically'
          : 'No servers are registered yet');
      const shouldShowCta =
        options?.showRegisterCta ?? (!searchTerm && activeFilter === 'all');

      return (
        <div className="text-center py-16">
          <div className="text-gray-400 text-xl mb-4">{title}</div>
          <p className="text-gray-500 dark:text-gray-300 text-base max-w-md mx-auto">{subtitle}</p>
          {shouldShowCta && (
            <button
              onClick={handleRegisterServer}
              className="mt-6 inline-flex items-center px-6 py-3 border border-transparent text-base font-medium rounded-lg text-white bg-blue-600 hover:bg-blue-700 focus:outline-none focus:ring-2 focus:ring-offset-2 focus:ring-blue-500 transition-colors"
            >
              <PlusIcon className="h-5 w-5 mr-2" />
              Register Server
            </button>
          )}
        </div>
      );
    }

    return (
      <div
        className="grid pb-12"
        style={{
          gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))',
          gap: 'clamp(1.5rem, 3vw, 2.5rem)'
        }}
      >
        {list.map((server) => (
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
        ))}
      </div>
    );
  };

  const renderDashboardCollections = () => (
    <>
      {/* MCP Servers Section - Grouped by Registry */}
      {registryConfig?.features.mcp_servers !== false &&
        (viewFilter === 'all' || viewFilter === 'servers') && (
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

            {filteredServers.length === 0 ? (
              <div className="text-center py-12 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <div className="text-gray-400 text-lg mb-2">No servers found</div>
                <p className="text-gray-500 dark:text-gray-300 text-sm">
                  {searchTerm || activeFilter !== 'all'
                    ? 'Press Enter in the search bar to search semantically'
                    : 'No servers are registered yet'}
                </p>
                {!searchTerm && activeFilter === 'all' && (
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

                  // Apply search filter
                  if (searchTerm) {
                    const query = searchTerm.toLowerCase();
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

                  // Render servers without registry header when it's the only registry
                  if (!showRegistryHeader) {
                    return (
                      <div key={registryId} className="overflow-visible">
                        <div
                          className="grid overflow-visible"
                          style={{
                            gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))',
                            gap: 'clamp(1.5rem, 3vw, 2.5rem)'
                          }}
                        >
                          {filteredRegistryServers.map((server) => (
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
                          ))}
                          {/* Virtual MCP Servers in Local Registry */}
                          {filteredVirtualServers.map((vs) => (
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
                          ))}
                        </div>
                      </div>
                    );
                  }

                  return (
                    <div key={registryId} id={`server-registry-${registryId}`} className="border border-gray-200 dark:border-gray-700 rounded-xl scroll-mt-4">
                      {/* Collapsible Header */}
                      <button
                        onClick={() => toggleRegistryGroup(registryId)}
                        className={`w-full flex items-center justify-between px-4 py-3 text-left transition-colors ${
                          registryId === 'local'
                            ? 'bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 hover:from-green-100 hover:to-emerald-100 dark:hover:from-green-900/30 dark:hover:to-emerald-900/30'
                            : 'bg-gradient-to-r from-cyan-50 to-blue-50 dark:from-cyan-900/20 dark:to-blue-900/20 hover:from-cyan-100 hover:to-blue-100 dark:hover:from-cyan-900/30 dark:hover:to-blue-900/30'
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          {isExpanded ? (
                            <ChevronDownIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
                          ) : (
                            <ChevronRightIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
                          )}
                          <span className={`font-semibold ${
                            registryId === 'local'
                              ? 'text-green-700 dark:text-green-300'
                              : 'text-cyan-700 dark:text-cyan-300'
                          }`}>
                            {displayName}
                          </span>
                          {/* Registry URL */}
                          <span className="text-xs text-gray-400 dark:text-gray-500 font-mono truncate max-w-[200px] lg:max-w-[300px]" title={registryId === 'local' ? localRegistryUrl : peerRegistryEndpoints[registryId]}>
                            | {registryId === 'local' ? localRegistryUrl : (peerRegistryEndpoints[registryId] || 'Loading...')}
                          </span>
                          <span className="px-2 py-0.5 text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded-full">
                            {registryId === 'local'
                              ? `${filteredRegistryServers.length + filteredVirtualServers.length} server${(filteredRegistryServers.length + filteredVirtualServers.length) !== 1 ? 's' : ''}`
                              : `${filteredRegistryServers.length} server${filteredRegistryServers.length !== 1 ? 's' : ''}`
                            }
                          </span>
                          {/* Resync button for federated registries */}
                          {registryId !== 'local' && (
                            <button
                              onClick={(e) => handleSyncPeer(registryId, e)}
                              disabled={syncingPeer === registryId}
                              className="ml-2 p-1 text-cyan-600 dark:text-cyan-400 hover:text-cyan-800 dark:hover:text-cyan-200 hover:bg-cyan-100 dark:hover:bg-cyan-900/30 rounded-lg transition-colors disabled:opacity-50"
                              title={`Resync from ${peerRegistryEndpoints[registryId] || registryId}`}
                            >
                              <ArrowPathIcon className={`h-4 w-4 ${syncingPeer === registryId ? 'animate-spin' : ''}`} />
                            </button>
                          )}
                        </div>
                      </button>

                      {/* Collapsible Content */}
                      {isExpanded && (
                        <div className="p-4 bg-white dark:bg-gray-800 overflow-visible">
                          <div
                            className="grid overflow-visible"
                            style={{
                              gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))',
                              gap: 'clamp(1.5rem, 3vw, 2.5rem)'
                            }}
                          >
                            {filteredRegistryServers.map((server) => (
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
                            ))}
                            {/* Virtual MCP Servers in Local Registry (collapsible view) */}
                            {registryId === 'local' && filteredVirtualServers.map((vs) => (
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
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

      {/* A2A Agents Section - Grouped by Registry */}
      {registryConfig?.features.agents !== false &&
        (viewFilter === 'all' || viewFilter === 'agents') && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                A2A Agents
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

                  // Apply search filter
                  if (searchTerm) {
                    const query = searchTerm.toLowerCase();
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

                  // Render agents without registry header when it's the only registry
                  if (!showRegistryHeader) {
                    return (
                      <div key={registryId} className="overflow-visible">
                        <div
                          className="grid overflow-visible"
                          style={{
                            gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))',
                            gap: 'clamp(1.5rem, 3vw, 2.5rem)'
                          }}
                        >
                          {filteredRegistryAgents.map((agent) => (
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
                          ))}
                        </div>
                      </div>
                    );
                  }

                  return (
                    <div key={registryId} id={`agent-registry-${registryId}`} className="border border-cyan-200 dark:border-cyan-700 rounded-xl overflow-hidden scroll-mt-4">
                      {/* Collapsible Header */}
                      <button
                        onClick={() => toggleRegistryGroup(`agents-${registryId}`)}
                        className={`w-full flex items-center justify-between px-4 py-3 text-left transition-colors ${
                          registryId === 'local'
                            ? 'bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 hover:from-green-100 hover:to-emerald-100 dark:hover:from-green-900/30 dark:hover:to-emerald-900/30'
                            : 'bg-gradient-to-r from-violet-50 to-purple-50 dark:from-violet-900/20 dark:to-purple-900/20 hover:from-violet-100 hover:to-purple-100 dark:hover:from-violet-900/30 dark:hover:to-purple-900/30'
                        }`}
                      >
                        <div className="flex items-center gap-3">
                          {isExpanded ? (
                            <ChevronDownIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
                          ) : (
                            <ChevronRightIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
                          )}
                          <span className={`font-semibold ${
                            registryId === 'local'
                              ? 'text-green-700 dark:text-green-300'
                              : 'text-violet-700 dark:text-violet-300'
                          }`}>
                            {displayName}
                          </span>
                          {/* Registry URL */}
                          <span className="text-xs text-gray-400 dark:text-gray-500 font-mono truncate max-w-[200px] lg:max-w-[300px]" title={registryId === 'local' ? localRegistryUrl : peerRegistryEndpoints[registryId]}>
                            | {registryId === 'local' ? localRegistryUrl : (peerRegistryEndpoints[registryId] || 'Loading...')}
                          </span>
                          <span className="px-2 py-0.5 text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded-full">
                            {filteredRegistryAgents.length} agent{filteredRegistryAgents.length !== 1 ? 's' : ''}
                          </span>
                          {/* Resync button for federated registries */}
                          {registryId !== 'local' && (
                            <button
                              onClick={(e) => handleSyncPeer(registryId, e)}
                              disabled={syncingPeer === registryId}
                              className="ml-2 p-1 text-violet-600 dark:text-violet-400 hover:text-violet-800 dark:hover:text-violet-200 hover:bg-violet-100 dark:hover:bg-violet-900/30 rounded-lg transition-colors disabled:opacity-50"
                              title={`Resync from ${peerRegistryEndpoints[registryId] || registryId}`}
                            >
                              <ArrowPathIcon className={`h-4 w-4 ${syncingPeer === registryId ? 'animate-spin' : ''}`} />
                            </button>
                          )}
                        </div>
                      </button>

                      {/* Collapsible Content */}
                      {isExpanded && (
                        <div className="p-4 bg-white dark:bg-gray-800 overflow-visible">
                          <div
                            className="grid overflow-visible"
                            style={{
                              gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))',
                              gap: 'clamp(1.5rem, 3vw, 2.5rem)'
                            }}
                          >
                            {filteredRegistryAgents.map((agent) => (
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
                            ))}
                          </div>
                        </div>
                      )}
                    </div>
                  );
                })}
              </div>
            )}
          </div>
        )}

      {/* Agent Skills Section */}
      {registryConfig?.features.skills !== false &&
        (viewFilter === 'all' || viewFilter === 'skills') && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                Agent Skills
              </h2>
              {user?.can_modify_servers && (
                <button
                  onClick={() => handleOpenSkillModal()}
                  className="inline-flex items-center px-3 py-1.5 text-sm font-medium text-white bg-amber-600 hover:bg-amber-700 rounded-lg transition-colors"
                >
                  <PlusIcon className="h-4 w-4 mr-1" />
                  Add Skill
                </button>
              )}
            </div>

            {skillsError ? (
              <div className="text-center py-12 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                <div className="text-red-500 text-lg mb-2">Failed to load skills</div>
                <p className="text-red-600 dark:text-red-400 text-sm">{skillsError}</p>
              </div>
            ) : skillsLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-amber-600"></div>
              </div>
            ) : filteredSkills.length === 0 ? (
              <div className="text-center py-12 bg-amber-50 dark:bg-amber-900/20 rounded-lg border border-amber-200 dark:border-amber-800">
                <div className="text-gray-400 text-lg mb-2">No skills found</div>
                <p className="text-gray-500 dark:text-gray-300 text-sm">
                  {searchTerm || activeFilter !== 'all'
                    ? 'Press Enter in the search bar to search semantically'
                    : 'No skills are registered yet'}
                </p>
                {!searchTerm && activeFilter === 'all' && user?.can_modify_servers && (
                  <button
                    onClick={() => handleOpenSkillModal()}
                    className="mt-4 inline-flex items-center px-4 py-2 border border-transparent text-sm font-medium rounded-lg text-white bg-amber-600 hover:bg-amber-700 transition-colors"
                  >
                    <PlusIcon className="h-4 w-4 mr-2" />
                    Register Skill
                  </button>
                )}
              </div>
            ) : (
              <div
                className="grid"
                style={{
                  gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))',
                  gap: 'clamp(1.5rem, 3vw, 2.5rem)'
                }}
              >
                {filteredSkills.map((skill) => (
                  <SkillCard
                    key={skill.path}
                    skill={skill}
                    onToggle={handleToggleSkill}
                    onEdit={handleEditSkill}
                    onDelete={(path: string) => setShowDeleteSkillConfirm(path)}
                    canModify={user?.can_modify_servers || false}
                    canToggle={user?.is_admin || hasUiPermission('toggle_skill', skill.path)}
                    onRefreshSuccess={refreshSkills}
                    onShowToast={showToast}
                    onSkillUpdate={handleSkillUpdate}
                    authToken={agentApiToken}
                  />
                ))}
              </div>
            )}
          </div>
        )}

      {/* Virtual MCP Servers Section */}
      {(viewFilter === 'all' || viewFilter === 'virtual') &&
        (filteredVirtualServers.length > 0 || viewFilter === 'virtual') && (
          <div className="mb-8">
            <div className="flex items-center justify-between mb-4">
              <h2 className="text-xl font-bold text-gray-900 dark:text-white">
                Virtual MCP Servers
              </h2>
              {(user?.can_modify_servers || user?.is_admin) && (
                <button
                  onClick={() => navigate('/settings/virtual-mcp/servers')}
                  className="inline-flex items-center px-4 py-2 text-sm font-medium text-white bg-teal-600 hover:bg-teal-700 rounded-lg transition-colors"
                >
                  <PlusIcon className="h-4 w-4 mr-2" />
                  Add Virtual Server
                </button>
              )}
            </div>

            {virtualServersError ? (
              <div className="text-center py-12 bg-red-50 dark:bg-red-900/20 rounded-lg border border-red-200 dark:border-red-800">
                <div className="text-red-500 text-lg mb-2">Failed to load virtual servers</div>
                <p className="text-red-600 dark:text-red-400 text-sm">{virtualServersError}</p>
              </div>
            ) : virtualServersLoading ? (
              <div className="flex items-center justify-center py-12">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-teal-600"></div>
              </div>
            ) : filteredVirtualServers.length === 0 ? (
              <div className="text-center py-12 bg-teal-50 dark:bg-teal-900/20 rounded-lg border border-teal-200 dark:border-teal-800">
                <div className="text-gray-400 text-lg mb-2">No virtual servers found</div>
                <p className="text-gray-500 dark:text-gray-300 text-sm">
                  {searchTerm || activeFilter !== 'all'
                    ? 'Try adjusting your search or filter'
                    : 'No virtual servers are configured yet'}
                </p>
              </div>
            ) : (
              <div
                className="grid"
                style={{
                  gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))',
                  gap: 'clamp(1.5rem, 3vw, 2.5rem)'
                }}
              >
                {filteredVirtualServers.map((vs) => (
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
                ))}
              </div>
            )}
          </div>
        )}

      {/* External Registries Section */}
      {registryConfig?.features.federation !== false && viewFilter === 'external' && (
        <div className="mb-8">
          <h2 className="text-xl font-bold text-gray-900 dark:text-white mb-4">
            External Registries
          </h2>

          {filteredExternalServers.length === 0 && filteredExternalAgents.length === 0 ? (
            <div className="text-center py-12 bg-gray-50 dark:bg-gray-800 rounded-lg border border-dashed border-gray-300 dark:border-gray-600">
              <div className="text-gray-400 text-lg mb-2">
                {externalServers.length === 0 && externalAgents.length === 0 ? 'No External Registries Available' : 'No Results Found'}
              </div>
              <p className="text-gray-500 dark:text-gray-300 text-sm max-w-md mx-auto">
                {externalServers.length === 0 && externalAgents.length === 0
                  ? 'External registry integrations (Anthropic, ASOR, and more) will be available soon'
                  : 'Press Enter in the search bar to search semantically'}
              </p>
            </div>
          ) : (
            <div>
              {/* External Servers */}
              {filteredExternalServers.length > 0 && (
                <div className="mb-6">
                  <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">
                    Servers
                  </h3>
                  <div
                    className="grid"
                    style={{
                      gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))',
                      gap: 'clamp(1.5rem, 3vw, 2.5rem)'
                    }}
                  >
                    {filteredExternalServers.map((server) => (
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
                    ))}
                  </div>
                </div>
              )}

              {/* External Agents */}
              {filteredExternalAgents.length > 0 && (
                <div>
                  <h3 className="text-lg font-semibold text-gray-800 dark:text-gray-200 mb-3">
                    Agents
                  </h3>
                  <div
                    className="grid"
                    style={{
                      gridTemplateColumns: 'repeat(auto-fit, minmax(380px, 1fr))',
                      gap: 'clamp(1.5rem, 3vw, 2.5rem)'
                    }}
                  >
                    {filteredExternalAgents.map((agent) => (
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
                    ))}
                  </div>
                </div>
              )}
            </div>
          )}
        </div>
      )}

      {/* Empty state when all are filtered out */}
      {((viewFilter === 'all' && filteredServers.length === 0 && filteredAgents.length === 0 && filteredSkills.length === 0 && filteredVirtualServers.length === 0) ||
        (viewFilter === 'servers' && filteredServers.length === 0) ||
        (viewFilter === 'agents' && filteredAgents.length === 0) ||
        (viewFilter === 'skills' && filteredSkills.length === 0) ||
        (viewFilter === 'virtual' && filteredVirtualServers.length === 0)) &&
        (searchTerm || activeFilter !== 'all') && (
          <div className="text-center py-16">
            <div className="text-gray-400 text-xl mb-4">No items found</div>
            <p className="text-gray-500 dark:text-gray-300 text-base max-w-md mx-auto">
              Press Enter in the search bar to search semantically
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
{/* Only show "All" tab if more than one feature is enabled */}
            {[
              registryConfig?.features.mcp_servers !== false,
              registryConfig?.features.agents !== false,
              registryConfig?.features.skills !== false,
              registryConfig?.features.federation !== false
            ].filter(Boolean).length > 1 && (
              <button
                onClick={() => handleChangeViewFilter('all')}
                className={`px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors border-b-2 ${
                  viewFilter === 'all'
                    ? 'border-purple-500 text-purple-600 dark:text-purple-400'
                    : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                All
              </button>
            )}
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
            {registryConfig?.features.agents !== false && (
              <button
                onClick={() => handleChangeViewFilter('agents')}
                className={`px-4 py-2 text-sm font-medium whitespace-nowrap transition-colors border-b-2 ${
                  viewFilter === 'agents'
                    ? 'border-cyan-500 text-cyan-600 dark:text-cyan-400'
                    : 'border-transparent text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-200'
                }`}
              >
                A2A Agents
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

            <button
              onClick={handleRegisterServer}
              className="btn-primary flex items-center space-x-2 flex-shrink-0"
            >
              <PlusIcon className="h-4 w-4" />
              <span>Register Server</span>
            </button>

            <button
              onClick={handleRefreshHealth}
              disabled={refreshing}
              className="btn-secondary flex items-center space-x-2 flex-shrink-0"
            >
              <ArrowPathIcon className={`h-4 w-4 ${refreshing ? 'animate-spin' : ''}`} />
              <span>Refresh Health</span>
            </button>
          </div>

          {/* Results count */}
          <div className="flex items-center justify-between">
            <p className="text-sm text-gray-500 dark:text-gray-300">
              {semanticSectionVisible ? (
                <>
                  Showing {semanticServers.length} servers, {semanticAgents.length} agents
                </>
              ) : (
                <>
                  {/* Dynamic count display based on enabled features */}
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
                </>
              )}
              {activeFilter !== 'all' && (
                <span className="ml-2 px-2 py-1 text-xs bg-purple-100 text-purple-800 dark:bg-purple-900 dark:text-purple-300 rounded-full">
                  {activeFilter} filter active
                </span>
              )}
            </p>
            <p className="text-xs text-gray-400 dark:text-gray-500">
              Press Enter to run semantic search; typing filters locally.
            </p>
          </div>
        </div>

        {/* Scrollable Content Area */}
        <div className="flex-1 overflow-y-auto min-h-0 space-y-10">
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
        </div>

        {/* Padding at bottom for scroll */}
        <div className="pb-12"></div>
      </div>

      {/* Register Server Modal */}
      {showRegisterModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50 p-4">
          <div className="bg-white dark:bg-gray-800 rounded-lg max-w-md w-full max-h-[90vh] overflow-y-auto">
            <form onSubmit={handleRegisterSubmit} className="p-6">
              <div className="flex items-center justify-between mb-4">
                <h3 className="text-lg font-semibold text-gray-900 dark:text-white">
                  Register New Server
                </h3>
                <button
                  type="button"
                  onClick={() => setShowRegisterModal(false)}
                  className="text-gray-400 hover:text-gray-600 dark:hover:text-gray-300"
                >
                  <XMarkIcon className="h-6 w-6" />
                </button>
              </div>

              <div className="space-y-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                    Server Name *
                  </label>
                  <input
                    type="text"
                    required
                    className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500"
                    value={registerForm.name}
                    onChange={(e) => setRegisterForm(prev => ({ ...prev, name: e.target.value }))}
                    placeholder="e.g., My Custom Server"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                    Path *
                  </label>
                  <input
                    type="text"
                    required
                    className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500"
                    value={registerForm.path}
                    onChange={(e) => setRegisterForm(prev => ({ ...prev, path: e.target.value }))}
                    placeholder="/my-server"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                    Proxy URL *
                  </label>
                  <input
                    type="url"
                    required
                    className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500"
                    value={registerForm.proxyPass}
                    onChange={(e) => setRegisterForm(prev => ({ ...prev, proxyPass: e.target.value }))}
                    placeholder="http://localhost:8080"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                    Description
                  </label>
                  <textarea
                    className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500"
                    rows={3}
                    value={registerForm.description}
                    onChange={(e) => setRegisterForm(prev => ({ ...prev, description: e.target.value }))}
                    placeholder="Brief description of the server"
                  />
                </div>

                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                    Tags
                  </label>
                  <input
                    type="text"
                    value={registerForm.tags.join(',')}
                    onChange={(e) => setRegisterForm(prev => ({ ...prev, tags: e.target.value.split(',').map(t => t.trim()).filter(t => t) }))}
                    className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500"
                    placeholder="tag1,tag2,tag3"
                  />
                </div>
              </div>

              <div className="flex justify-end space-x-3 mt-6">
                <button
                  type="button"
                  onClick={() => setShowRegisterModal(false)}
                  className="flex-1 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-200 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-md transition-colors"
                >
                  Cancel
                </button>
                <button
                  type="submit"
                  disabled={registerLoading}
                  className="px-4 py-2 text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 disabled:opacity-50 rounded-md transition-colors"
                >
                  {registerLoading ? 'Registering...' : 'Register Server'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Edit Server Modal */}
      {editingServer && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-md max-h-[90vh] overflow-y-auto">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Edit Server: {editingServer.name}
            </h3>

            <form
              onSubmit={async (e) => {
                e.preventDefault();
                await handleSaveEdit();
              }}
              className="space-y-4"
            >
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Server Name *
                </label>
                <input
                  type="text"
                  value={editForm.name}
                  onChange={(e) => setEditForm(prev => ({ ...prev, name: e.target.value }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Proxy Pass URL *
                </label>
                <input
                  type="url"
                  value={editForm.proxyPass}
                  onChange={(e) => setEditForm(prev => ({ ...prev, proxyPass: e.target.value }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500"
                  placeholder="http://localhost:8080"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Description
                </label>
                <textarea
                  value={editForm.description}
                  onChange={(e) => setEditForm(prev => ({ ...prev, description: e.target.value }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500"
                  rows={3}
                  placeholder="Brief description of the server"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Tags
                </label>
                <input
                  type="text"
                  value={editForm.tags.join(',')}
                  onChange={(e) => setEditForm(prev => ({ ...prev, tags: e.target.value.split(',').map(t => t.trim()).filter(t => t) }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500"
                  placeholder="tag1,tag2,tag3"
                />
              </div>

              <div className="grid grid-cols-2 gap-4">
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                    Number of Tools
                  </label>
                  <input
                    type="number"
                    value={editForm.num_tools}
                    onChange={(e) => setEditForm(prev => ({ ...prev, num_tools: parseInt(e.target.value) || 0 }))}
                    className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500"
                    min="0"
                  />
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  License
                </label>
                <input
                  type="text"
                  value={editForm.license}
                  onChange={(e) => setEditForm(prev => ({ ...prev, license: e.target.value }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500"
                  placeholder="MIT, Apache-2.0, etc."
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  MCP Endpoint (optional)
                </label>
                <input
                  type="url"
                  value={editForm.mcp_endpoint}
                  onChange={(e) => setEditForm(prev => ({ ...prev, mcp_endpoint: e.target.value }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500"
                  placeholder="Custom MCP endpoint URL (overrides default)"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Custom Metadata (JSON, optional)
                </label>
                <textarea
                  value={editForm.metadata}
                  onChange={(e) => setEditForm(prev => ({ ...prev, metadata: e.target.value }))}
                  rows={4}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500 font-mono text-sm"
                  placeholder='{"team": "platform", "owner": "alice@example.com"}'
                />
              </div>

              {/* Backend Authentication */}
              <div className="border-t border-gray-200 dark:border-gray-700 pt-4 mt-4">
                <h4 className="text-sm font-semibold text-gray-900 dark:text-white mb-3">
                  Backend Authentication
                </h4>

                <div className="space-y-4">
                  <div>
                    <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                      Authentication Scheme
                    </label>
                    <select
                      value={editForm.auth_scheme}
                      onChange={(e) => {
                        const newScheme = e.target.value;
                        setEditForm(prev => ({
                          ...prev,
                          auth_scheme: newScheme,
                          auth_credential: newScheme === 'none' ? '' : prev.auth_credential,
                          auth_header_name: newScheme === 'api_key' ? prev.auth_header_name : 'X-API-Key',
                        }));
                      }}
                      className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500"
                    >
                      <option value="none">None</option>
                      <option value="bearer">Bearer Token</option>
                      <option value="api_key">API Key</option>
                    </select>
                  </div>

                  {editForm.auth_scheme !== 'none' && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                        {editForm.auth_scheme === 'bearer' ? 'Bearer Token' : 'API Key'}
                      </label>
                      <input
                        type="password"
                        value={editForm.auth_credential}
                        onChange={(e) => setEditForm(prev => ({ ...prev, auth_credential: e.target.value }))}
                        className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500"
                        placeholder="Leave blank to keep current credential"
                      />
                      <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                        Leave blank to keep the existing credential unchanged.
                      </p>
                    </div>
                  )}

                  {editForm.auth_scheme === 'api_key' && (
                    <div>
                      <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                        Header Name
                      </label>
                      <input
                        type="text"
                        value={editForm.auth_header_name}
                        onChange={(e) => setEditForm(prev => ({ ...prev, auth_header_name: e.target.value }))}
                        className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-purple-500 focus:border-purple-500"
                        placeholder="X-API-Key"
                      />
                    </div>
                  )}
                </div>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Path (read-only)
                </label>
                <input
                  type="text"
                  value={editForm.path}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-300"
                  disabled
                />
              </div>

              <div className="flex space-x-3 pt-4">
                <button
                  type="submit"
                  disabled={editLoading}
                  className="flex-1 px-4 py-2 text-sm font-medium text-white bg-purple-600 hover:bg-purple-700 disabled:opacity-50 rounded-md transition-colors"
                >
                  {editLoading ? 'Saving...' : 'Save Changes'}
                </button>
                <button
                  type="button"
                  onClick={handleCloseEdit}
                  className="flex-1 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md transition-colors"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Edit Agent Modal */}
      {editingAgent && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-md max-h-[90vh] overflow-y-auto">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              Edit Agent: {editingAgent.name}
            </h3>

            <form
              onSubmit={async (e) => {
                e.preventDefault();
                await handleSaveEditAgent();
              }}
              className="space-y-4"
            >
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Agent Name *
                </label>
                <input
                  type="text"
                  value={editAgentForm.name}
                  onChange={(e) => setEditAgentForm(prev => ({ ...prev, name: e.target.value }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-cyan-500 focus:border-cyan-500"
                  required
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Description
                </label>
                <textarea
                  value={editAgentForm.description}
                  onChange={(e) => setEditAgentForm(prev => ({ ...prev, description: e.target.value }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-cyan-500 focus:border-cyan-500"
                  rows={3}
                  placeholder="Brief description of the agent"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Version
                </label>
                <input
                  type="text"
                  value={editAgentForm.version}
                  onChange={(e) => setEditAgentForm(prev => ({ ...prev, version: e.target.value }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-cyan-500 focus:border-cyan-500"
                  placeholder="1.0.0"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Visibility
                </label>
                <select
                  value={editAgentForm.visibility}
                  onChange={(e) => setEditAgentForm(prev => ({ ...prev, visibility: e.target.value as 'public' | 'private' | 'group-restricted' }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-cyan-500 focus:border-cyan-500"
                >
                  <option value="private">Private</option>
                  <option value="public">Public</option>
                  <option value="group-restricted">Group Restricted</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Trust Level
                </label>
                <select
                  value={editAgentForm.trust_level}
                  onChange={(e) => setEditAgentForm(prev => ({ ...prev, trust_level: e.target.value as 'community' | 'verified' | 'trusted' | 'unverified' }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-cyan-500 focus:border-cyan-500"
                >
                  <option value="unverified">Unverified</option>
                  <option value="community">Community</option>
                  <option value="verified">Verified</option>
                  <option value="trusted">Trusted</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Tags
                </label>
                <input
                  type="text"
                  value={editAgentForm.tags.join(',')}
                  onChange={(e) => setEditAgentForm(prev => ({ ...prev, tags: e.target.value.split(',').map(t => t.trim()).filter(t => t) }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-cyan-500 focus:border-cyan-500"
                  placeholder="tag1,tag2,tag3"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Path (read-only)
                </label>
                <input
                  type="text"
                  value={editAgentForm.path}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-300"
                  disabled
                />
              </div>

              <div className="flex space-x-3 pt-4">
                <button
                  type="submit"
                  disabled={editAgentLoading}
                  className="flex-1 px-4 py-2 text-sm font-medium text-white bg-cyan-600 hover:bg-cyan-700 disabled:opacity-50 rounded-md transition-colors"
                >
                  {editAgentLoading ? 'Saving...' : 'Save Changes'}
                </button>
                <button
                  type="button"
                  onClick={handleCloseEdit}
                  className="flex-1 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md transition-colors"
                >
                  Cancel
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* Register/Edit Skill Modal */}
      {showSkillModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center p-4 z-50">
          <div className="bg-white dark:bg-gray-800 rounded-lg p-6 w-full max-w-md max-h-[90vh] overflow-y-auto">
            <h3 className="text-lg font-semibold text-gray-900 dark:text-white mb-4">
              {editingSkill ? `Edit Skill: ${editingSkill.name}` : 'Register New Skill'}
            </h3>

            <form
              onSubmit={handleSaveSkill}
              className="space-y-4"
            >
              {/* Auto-fill toggle - only for new skills */}
              {!editingSkill && (
                <div className="flex items-center justify-between p-3 bg-gray-50 dark:bg-gray-700/50 rounded-lg">
                  <div>
                    <span className="text-sm font-medium text-gray-700 dark:text-gray-200">
                      Auto-fill from SKILL.md
                    </span>
                    <p className="text-xs text-gray-500 dark:text-gray-400">
                      Parse name and description from the SKILL.md file
                    </p>
                  </div>
                  <button
                    type="button"
                    onClick={() => setSkillAutoFill(!skillAutoFill)}
                    className={`relative inline-flex h-6 w-11 items-center rounded-full transition-colors ${
                      skillAutoFill ? 'bg-amber-600' : 'bg-gray-300 dark:bg-gray-600'
                    }`}
                  >
                    <span
                      className={`inline-block h-4 w-4 transform rounded-full bg-white transition-transform ${
                        skillAutoFill ? 'translate-x-6' : 'translate-x-1'
                      }`}
                    />
                  </button>
                </div>
              )}

              {/* SKILL.md URL with Parse button */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  SKILL.md URL *
                </label>
                <div className="flex space-x-2">
                  <input
                    type="url"
                    value={skillForm.skill_md_url}
                    onChange={(e) => setSkillForm(prev => ({ ...prev, skill_md_url: e.target.value }))}
                    className="flex-1 px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-amber-500 focus:border-amber-500"
                    placeholder="https://raw.githubusercontent.com/org/repo/main/SKILL.md"
                    required
                  />
                  {skillAutoFill && !editingSkill && (
                    <button
                      type="button"
                      onClick={handleParseSkillMd}
                      disabled={!skillForm.skill_md_url || skillParseLoading}
                      className="px-3 py-2 text-sm font-medium text-white bg-amber-600 hover:bg-amber-700 disabled:opacity-50 disabled:cursor-not-allowed rounded-md transition-colors whitespace-nowrap"
                    >
                      {skillParseLoading ? 'Parsing...' : 'Parse'}
                    </button>
                  )}
                </div>
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  Use raw content URL (e.g., raw.githubusercontent.com)
                </p>
              </div>

              {/* Name field */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Skill Name *
                </label>
                <input
                  type="text"
                  value={skillForm.name}
                  onChange={(e) => {
                    const formatted = e.target.value
                      .toLowerCase()
                      .replace(/[^a-z0-9-]/g, '-')
                      .replace(/-+/g, '-')
                      .replace(/^-|-$/g, '');
                    setSkillForm(prev => ({ ...prev, name: formatted }));
                  }}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-amber-500 focus:border-amber-500"
                  placeholder="my-skill-name"
                  pattern="^[a-z0-9]+(-[a-z0-9]+)*$"
                  title="Lowercase alphanumeric with hyphens (e.g., my-skill-name)"
                  required
                  disabled={!!editingSkill}
                />
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  Lowercase letters, numbers, and hyphens only
                </p>
              </div>

              {/* Description field */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Description *
                </label>
                <textarea
                  value={skillForm.description}
                  onChange={(e) => setSkillForm(prev => ({ ...prev, description: e.target.value }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-amber-500 focus:border-amber-500"
                  rows={3}
                  placeholder="Describe what this skill does and when to use it"
                  required
                />
              </div>

              {/* Repository URL */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Repository URL (optional)
                </label>
                <input
                  type="url"
                  value={skillForm.repository_url}
                  onChange={(e) => setSkillForm(prev => ({ ...prev, repository_url: e.target.value }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-amber-500 focus:border-amber-500"
                  placeholder="https://github.com/org/repo"
                />
              </div>

              {/* Version field */}
              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Version (optional)
                </label>
                <input
                  type="text"
                  value={skillForm.version}
                  onChange={(e) => setSkillForm(prev => ({ ...prev, version: e.target.value }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-amber-500 focus:border-amber-500"
                  placeholder="1.0.0"
                />
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Visibility
                </label>
                <select
                  value={skillForm.visibility}
                  onChange={(e) => setSkillForm(prev => ({ ...prev, visibility: e.target.value as 'public' | 'private' | 'group' }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-amber-500 focus:border-amber-500"
                >
                  <option value="public">Public</option>
                  <option value="private">Private</option>
                  <option value="group">Group</option>
                </select>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Tags
                </label>
                <input
                  type="text"
                  value={skillForm.tags}
                  onChange={(e) => setSkillForm(prev => ({ ...prev, tags: e.target.value }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-amber-500 focus:border-amber-500"
                  placeholder="automation, productivity, code-review"
                />
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  Comma-separated tags for categorization
                </p>
              </div>

              <div>
                <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                  Target Agents
                </label>
                <input
                  type="text"
                  value={skillForm.target_agents}
                  onChange={(e) => setSkillForm(prev => ({ ...prev, target_agents: e.target.value }))}
                  className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-700 text-gray-900 dark:text-white focus:ring-amber-500 focus:border-amber-500"
                  placeholder="claude-code, cursor, windsurf"
                />
                <p className="mt-1 text-xs text-gray-500 dark:text-gray-400">
                  Comma-separated list of compatible coding assistants
                </p>
              </div>

              {editingSkill && (
                <div>
                  <label className="block text-sm font-medium text-gray-700 dark:text-gray-200 mb-1">
                    Path (read-only)
                  </label>
                  <input
                    type="text"
                    value={editingSkill.path}
                    className="block w-full px-3 py-2 border border-gray-300 dark:border-gray-600 rounded-md bg-gray-100 dark:bg-gray-800 text-gray-500 dark:text-gray-300"
                    disabled
                  />
                </div>
              )}

              <div className="flex space-x-3 pt-4">
                <button
                  type="submit"
                  disabled={skillFormLoading}
                  className="flex-1 px-4 py-2 text-sm font-medium text-white bg-amber-600 hover:bg-amber-700 disabled:opacity-50 rounded-md transition-colors"
                >
                  {skillFormLoading
                    ? (editingSkill ? 'Saving...' : 'Registering & Scanning...')
                    : (editingSkill ? 'Save Changes' : 'Register Skill')}
                </button>
                <button
                  type="button"
                  onClick={handleCloseSkillModal}
                  className="flex-1 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md transition-colors"
                >
                  Cancel
                </button>
              </div>
              {!editingSkill && (
                <p className="text-xs text-gray-500 dark:text-gray-400 mt-2 text-center">
                  Registration includes a security scan and may take a few seconds
                </p>
              )}
            </form>
          </div>
        </div>
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
                className="flex-1 px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-md transition-colors"
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
                           rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600 disabled:opacity-50"
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
                  className="mt-4 px-4 py-2 bg-gray-200 dark:bg-gray-700 text-gray-800 dark:text-gray-200 rounded-lg hover:bg-gray-300 dark:hover:bg-gray-600"
                >
                  Close
                </button>
              </div>
            )}
          </div>
        </div>
      )}

    </>
  );
};

export default Dashboard;
