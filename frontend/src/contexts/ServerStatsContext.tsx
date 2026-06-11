import React, { createContext, useContext, useState, useEffect, useCallback, ReactNode } from 'react';
import axios from 'axios';
import { useRegistryConfig } from '../hooks/useRegistryConfig';
import type { LocalRuntime } from '../types/server';
import type { CustomEntityRecord, CustomTypeDescriptor } from '../types/customEntity';

interface ServerVersion {
  version: string;
  proxy_pass_url: string;
  status: string;
  is_default: boolean;
}

interface SyncMetadata {
  is_federated?: boolean;
  source_peer_id?: string;
  upstream_path?: string;
  last_synced_at?: string;
  is_read_only?: boolean;
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
  type: 'server' | 'agent';
  proxy_pass_url?: string;
  // Local-server fields
  deployment?: 'remote' | 'local';
  local_runtime?: LocalRuntime;
  version?: string;
  versions?: ServerVersion[];
  default_version?: string;
  mcp_server_version?: string;
  mcp_server_version_previous?: string;
  mcp_server_version_updated_at?: string;
  sync_metadata?: SyncMetadata;
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
  trust_level?: string;
  visibility?: string;
  supported_protocol?: string | null;
  lifecycle_status?: 'active' | 'draft' | 'deprecated' | 'beta';
}

interface ServerStats {
  total: number;
  enabled: number;
  disabled: number;
  withIssues: number;
}

/** Records for one custom type, shared by the tabs, Discover counts, and sidebar. */
export interface CustomTypeRecords {
  name: string;
  displayName: string;
  /** Schema descriptor, needed to render record cards outside the dedicated tab. */
  descriptor: CustomTypeDescriptor | null;
  records: CustomEntityRecord[];
}

interface ServerStatsContextType {
  stats: ServerStats;
  servers: Server[];
  agents: Server[];
  customRecordsByType: CustomTypeRecords[];
  setServers: React.Dispatch<React.SetStateAction<Server[]>>;
  setAgents: React.Dispatch<React.SetStateAction<Server[]>>;
  activeFilter: string;
  setActiveFilter: (filter: string) => void;
  loading: boolean;
  error: string | null;
  refreshData: () => Promise<void>;
}

const ServerStatsContext = createContext<ServerStatsContextType | undefined>(undefined);

// Helper function to map backend health status to frontend status
const mapHealthStatus = (healthStatus: string): 'healthy' | 'unhealthy' | 'unknown' | 'local' => {
  if (!healthStatus || healthStatus === 'unknown') return 'unknown';
  if (healthStatus === 'healthy') return 'healthy';
  if (healthStatus === 'local') return 'local';
  if (healthStatus.includes('unhealthy') || healthStatus.includes('error') || healthStatus.includes('timeout')) return 'unhealthy';
  return 'unknown';
};

interface ServerStatsProviderProps {
  children: ReactNode;
}

export const ServerStatsProvider: React.FC<ServerStatsProviderProps> = ({ children }) => {
  const [stats, setStats] = useState<ServerStats>({
    total: 0,
    enabled: 0,
    disabled: 0,
    withIssues: 0,
  });
  const [servers, setServers] = useState<Server[]>([]);
  const [agents, setAgents] = useState<Server[]>([]);
  const [customRecordsByType, setCustomRecordsByType] = useState<CustomTypeRecords[]>([]);
  const [activeFilter, setActiveFilter] = useState<string>('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Get registry config to determine which features are enabled
  const { config: registryConfig } = useRegistryConfig();

  const fetchData = useCallback(async () => {
    // Wait for registry config before fetching. Without it we don't yet know
    // which custom types exist, so computing stats now would show a core-only
    // total that visibly jumps when the custom counts arrive a moment later.
    if (!registryConfig) {
      return;
    }
    try {
      setLoading(true);
      setError(null);

      // Check which features are enabled based on registry mode
      const serversEnabled = registryConfig?.features.mcp_servers !== false;
      const agentsEnabled = registryConfig?.features.agents !== false;
      const skillsEnabled = registryConfig?.features.skills !== false;

      // Build fetch promises based on enabled features
      const fetchPromises: Promise<any>[] = [];

      if (serversEnabled) {
        // include_tools=false: the dashboard only needs num_tools for the card
        // badge, not the full per-server tool_list. Omitting it shrinks the
        // payload and skips per-server tool filtering on the backend.
        fetchPromises.push(axios.get('/api/servers?limit=2000&include_tools=false').catch(() => ({ data: { servers: [] } })));
      } else {
        fetchPromises.push(Promise.resolve({ data: { servers: [] } }));
      }

      if (agentsEnabled) {
        fetchPromises.push(axios.get('/api/agents?limit=2000').catch(() => ({ data: { agents: [] } })));
      } else {
        fetchPromises.push(Promise.resolve({ data: { agents: [] } }));
      }

      // Fetch skills for stats if skills are enabled
      if (skillsEnabled) {
        fetchPromises.push(axios.get('/api/skills?include_disabled=true&limit=2000').catch(() => ({ data: { skills: [] } })));
      } else {
        fetchPromises.push(Promise.resolve({ data: { skills: [] } }));
      }

      // Fetch custom entity records (one call per type) alongside the core
      // entities so every count lands in the SAME stats update — no visible
      // jump from a core-only total to the full total. Each type 404s only if
      // deleted mid-session.
      const customTypes = registryConfig?.features.custom_types
        ? (registryConfig.custom_types ?? [])
        : [];
      const customPromises = customTypes.map((t) =>
        axios.get(`/api/custom/${t.name}?limit=1000`).catch((err) => {
          // Treat as zero records so one stale type doesn't blank the sidebar.
          // Log so a real failure (auth, 422, 5xx) is visible rather than
          // silently zeroed out.
          console.error(`Failed to fetch custom records for "${t.name}":`, err);
          return { data: { records: [] } };
        }),
      );
      // One bulk fetch of every descriptor so Discover can render record cards
      // (which need the schema) without a per-type round-trip. Falls back to an
      // empty list — sections then render via the per-tab descriptor fetch.
      const descriptorsPromise = customTypes.length
        ? axios
            .get('/api/custom-types')
            .catch((err) => {
              console.error('Failed to fetch custom type descriptors:', err);
              return { data: { custom_types: [] } };
            })
        : Promise.resolve({ data: { custom_types: [] } });

      const [coreResponses, customResponses, descriptorsResponse] = await Promise.all([
        Promise.all(fetchPromises),
        Promise.all(customPromises),
        descriptorsPromise,
      ]);
      const [serversResponse, agentsResponse, skillsResponse] = coreResponses;

      const descriptorsByName = new Map<string, CustomTypeDescriptor>(
        ((descriptorsResponse.data?.custom_types ?? []) as CustomTypeDescriptor[]).map(
          (d) => [d.name, d],
        ),
      );
      const customByType: CustomTypeRecords[] = customTypes.map((t, i) => ({
        name: t.name,
        displayName: t.display_name,
        descriptor: descriptorsByName.get(t.name) ?? null,
        records: (customResponses[i].data?.records ?? []) as CustomEntityRecord[],
      }));
      setCustomRecordsByType(customByType);
      const customRecords = customByType.flatMap((ct) => ct.records);

      // The API returns {"servers": [...]}
      const responseData = serversResponse.data || {};
      const serversList = responseData.servers || [];

      // The agents API returns {"agents": [...]}
      const agentsData = agentsResponse.data || {};
      const agentsList = agentsData.agents || [];

      // The skills API returns {"skills": [...]}
      const skillsData = skillsResponse.data || {};
      const skillsList = skillsData.skills || [];

      // Transform server data from backend format to frontend format
      const transformedServers: Server[] = serversList.map((serverInfo: any) => {
        const transformed = {
          name: serverInfo.display_name || 'Unknown Server',
          path: serverInfo.path,
          description: serverInfo.description || '',
          official: serverInfo.is_official || false,
          enabled: serverInfo.is_enabled !== undefined ? serverInfo.is_enabled : false,
          tags: serverInfo.tags || [],
          last_checked_time: serverInfo.last_checked_iso,  // Fixed field mapping
          usersCount: 0, // Not available in backend
          rating: serverInfo.num_stars || 0,
          rating_details: serverInfo.rating_details || [],
          status: mapHealthStatus(serverInfo.health_status || 'unknown'),
          num_tools: serverInfo.num_tools || 0,
          type: 'server' as const,
          proxy_pass_url: serverInfo.proxy_pass_url || '',
          version: serverInfo.version,
          versions: serverInfo.versions,
          default_version: serverInfo.default_version,
          mcp_server_version: serverInfo.mcp_server_version,
          mcp_server_version_previous: serverInfo.mcp_server_version_previous,
          mcp_server_version_updated_at: serverInfo.mcp_server_version_updated_at,
          sync_metadata: serverInfo.sync_metadata,
          ans_metadata: serverInfo.ans_metadata || serverInfo.ansMetadata,
          auth_scheme: serverInfo.auth_scheme,
          auth_header_name: serverInfo.auth_header_name,
          lifecycle_status: serverInfo.status || 'active',
          // Local-server fields
          deployment: serverInfo.deployment || 'remote',
          local_runtime: serverInfo.local_runtime,
          registered_by: serverInfo.registered_by ?? null,
        };
        return transformed;
      });

      // Transform agent data from backend format to frontend format
      const transformedAgents: Server[] = agentsList.map((agentInfo: any) => {
        const transformed = {
          name: agentInfo.name || 'Unknown Agent',
          path: agentInfo.path,
          description: agentInfo.description || '',
          official: false, // Agents don't have official flag
          enabled: agentInfo.is_enabled !== undefined ? agentInfo.is_enabled : false,
          tags: agentInfo.tags || [],
          last_checked_time: agentInfo.last_health_check || agentInfo.lastHealthCheck || undefined,
          usersCount: 0,
          rating: agentInfo.num_stars || 0,
          status: mapHealthStatus(agentInfo.health_status || agentInfo.healthStatus || 'unknown'),
          num_tools: agentInfo.num_skills || 0, // Use num_skills for agents
          type: 'agent' as const,
          sync_metadata: agentInfo.sync_metadata,
          ans_metadata: agentInfo.ans_metadata || agentInfo.ansMetadata,
          registered_by: agentInfo.registered_by || agentInfo.registeredBy || null,
          trust_level: agentInfo.trust_level || agentInfo.trustLevel || 'community',
          visibility: agentInfo.visibility || 'public',
          supported_protocol: agentInfo.supported_protocol || agentInfo.supportedProtocol || null,
          lifecycle_status: agentInfo.status || 'active',
        };
        return transformed;
      });

      // Store servers and agents separately
      setServers(transformedServers);
      setAgents(transformedAgents);

      // Calculate stats based on what features are enabled
      let total = 0;
      let enabled = 0;
      let disabled = 0;
      let withIssues = 0;

      // Include servers in stats if enabled
      if (serversEnabled) {
        transformedServers.forEach((service) => {
          total++;
          if (service.enabled) {
            enabled++;
          } else {
            disabled++;
          }
          if (service.status === 'unhealthy') {
            withIssues++;
          }
        });
      }

      // Include agents in stats if enabled
      if (agentsEnabled) {
        transformedAgents.forEach((service) => {
          total++;
          if (service.enabled) {
            enabled++;
          } else {
            disabled++;
          }
          if (service.status === 'unhealthy') {
            withIssues++;
          }
        });
      }

      // Include skills in stats if enabled (and servers/agents are not)
      // This ensures skills-only mode shows skill stats
      if (skillsEnabled) {
        skillsList.forEach((skill: any) => {
          total++;
          if (skill.is_enabled !== false) {
            enabled++;
          } else {
            disabled++;
          }
          // Skills don't have health status, so no withIssues increment
        });
      }

      // Include custom entity records. They have no health status and no
      // disable toggle in the UI, so each record counts as enabled.
      customRecords.forEach((record: CustomEntityRecord) => {
        total++;
        if (record.is_enabled !== false) {
          enabled++;
        } else {
          disabled++;
        }
      });

      const newStats = {
        total,
        enabled,
        disabled,
        withIssues,
      };
      setStats(newStats);
    } catch (err: any) {
      console.error('Failed to fetch data:', err);
      setError(err.response?.data?.detail || 'Failed to fetch data');
      setServers([]);
      setAgents([]);
      setCustomRecordsByType([]);
      setStats({ total: 0, enabled: 0, disabled: 0, withIssues: 0 });
    } finally {
      setLoading(false);
    }
  }, [registryConfig]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  const value: ServerStatsContextType = {
    stats,
    servers,
    agents,
    customRecordsByType,
    setServers,
    setAgents,
    activeFilter,
    setActiveFilter,
    loading,
    error,
    refreshData: fetchData,
  };

  return <ServerStatsContext.Provider value={value}>{children}</ServerStatsContext.Provider>;
};

export const useServerStats = (): ServerStatsContextType => {
  const context = useContext(ServerStatsContext);
  if (context === undefined) {
    throw new Error('useServerStats must be used within a ServerStatsProvider');
  }
  return context;
};
