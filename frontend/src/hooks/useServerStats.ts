import React, { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { useRegistryConfig } from './useRegistryConfig';

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
  status?: 'healthy' | 'healthy-auth-expired' | 'unhealthy' | 'unknown';
  num_tools?: number;
  type: 'server' | 'agent';
  proxy_pass_url?: string;
  version?: string;
  versions?: ServerVersion[];
  default_version?: string;
  mcp_server_version?: string;
  mcp_server_version_previous?: string;
  mcp_server_version_updated_at?: string;
  sync_metadata?: SyncMetadata;
  registered_by?: string | null;
}

interface ServerStats {
  total: number;
  enabled: number;
  disabled: number;
  withIssues: number;
}

interface UseServerStatsReturn {
  stats: ServerStats;
  servers: Server[];
  agents: Server[];
  setServers: React.Dispatch<React.SetStateAction<Server[]>>;
  setAgents: React.Dispatch<React.SetStateAction<Server[]>>;
  activeFilter: string;
  setActiveFilter: (filter: string) => void;
  loading: boolean;
  error: string | null;
  refreshData: () => Promise<void>;
}

export const useServerStats = (): UseServerStatsReturn => {
  const [stats, setStats] = useState<ServerStats>({
    total: 0,
    enabled: 0,
    disabled: 0,
    withIssues: 0,
  });
  const [servers, setServers] = useState<Server[]>([]);
  const [agents, setAgents] = useState<Server[]>([]);
  const [activeFilter, setActiveFilter] = useState<string>('all');
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Get registry config to determine which features are enabled
  const { config: registryConfig } = useRegistryConfig();

  // Helper function to map backend health status to frontend status
  const mapHealthStatus = (healthStatus: string): 'healthy' | 'unhealthy' | 'unknown' => {
    if (!healthStatus || healthStatus === 'unknown') return 'unknown';
    if (healthStatus === 'healthy') return 'healthy';
    if (healthStatus.includes('unhealthy') || healthStatus.includes('error') || healthStatus.includes('timeout')) return 'unhealthy';
    return 'unknown';
  };

  const fetchData = useCallback(async () => {
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
        fetchPromises.push(axios.get('/api/servers').catch(() => ({ data: { servers: [] } })));
      } else {
        fetchPromises.push(Promise.resolve({ data: { servers: [] } }));
      }

      if (agentsEnabled) {
        fetchPromises.push(axios.get('/api/agents').catch(() => ({ data: { agents: [] } })));
      } else {
        fetchPromises.push(Promise.resolve({ data: { agents: [] } }));
      }

      // Fetch skills for stats if skills are enabled
      if (skillsEnabled) {
        fetchPromises.push(axios.get('/api/skills?include_disabled=true').catch(() => ({ data: { skills: [] } })));
      } else {
        fetchPromises.push(Promise.resolve({ data: { skills: [] } }));
      }

      const [serversResponse, agentsResponse, skillsResponse] = await Promise.all(fetchPromises);
      
      // The API returns {"servers": [...]} 
      const responseData = serversResponse.data || {};
      const serversList = responseData.servers || [];
      
      // The agents API returns {"agents": [...]}
      const agentsData = agentsResponse.data || {};
      const agentsList = agentsData.agents || [];

      // The skills API returns {"skills": [...]}
      const skillsData = skillsResponse.data || {};
      const skillsList = skillsData.skills || [];

      // Debug logging to see what servers are returned
      console.log('🔍 Server filtering debug info:');
      console.log(`📊 Total servers returned from API: ${serversList.length}`);
      console.log('📋 Server list:', serversList.map((s: any) => ({ 
        name: s.display_name, 
        path: s.path, 
        enabled: s.is_enabled 
      })));
      
      // Debug logging for agents
      console.log(`📊 Total agents returned from API: ${agentsList.length}`);
      console.log('📋 Agent list:', agentsList.map((a: any) => ({ 
        name: a.name, 
        path: a.path, 
        enabled: a.is_enabled 
      })));
      
      // Transform server data from backend format to frontend format
      const transformedServers: Server[] = serversList.map((serverInfo: any) => {
        // Debug log to see what last_checked_iso data we're getting
        console.log(`🕐 Server ${serverInfo.display_name}: last_checked_iso =`, serverInfo.last_checked_iso);
        
        const transformed = {
          name: serverInfo.display_name || 'Unknown Server',
          path: serverInfo.path,
          description: serverInfo.description || '',
          official: serverInfo.is_official || false,
          enabled: serverInfo.is_enabled !== undefined ? serverInfo.is_enabled : false,
          tags: serverInfo.tags || [],
          last_checked_time: serverInfo.last_checked_iso,  // Fixed field mapping
          usersCount: 0, // Not available in backend
          rating: 0,
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
          auth_scheme: serverInfo.auth_scheme,
          auth_header_name: serverInfo.auth_header_name,
        };
        
        // Debug log the transformed server
        console.log(`🔄 Transformed server ${transformed.name}:`, {
          last_checked_time: transformed.last_checked_time,
          status: transformed.status,
          enabled: transformed.enabled
        });
        
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
          last_checked_time: undefined, // Agents don't have health check timestamp
          usersCount: 0,
          rating: agentInfo.num_stars || 0,
          status: 'unknown' as const, // Agents don't have health status yet
          num_tools: agentInfo.num_skills || 0, // Use num_skills for agents
          type: 'agent' as const,
          sync_metadata: agentInfo.sync_metadata,
          registered_by: agentInfo.registered_by || agentInfo.registeredBy || null,
        };
        
        console.log(`🔄 Transformed agent ${transformed.name}:`, {
          enabled: transformed.enabled,
          num_skills: transformed.num_tools
        });
        
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

      const newStats = {
        total,
        enabled,
        disabled,
        withIssues,
      };
      
      console.log('Calculated stats (servers + agents + skills):', newStats);
      setStats(newStats);
    } catch (err: any) {
      console.error('Failed to fetch data:', err);
      setError(err.response?.data?.detail || 'Failed to fetch data');
      setServers([]);
      setAgents([]);
      setStats({ total: 0, enabled: 0, disabled: 0, withIssues: 0 });
    } finally {
      setLoading(false);
    }
  }, [registryConfig]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return {
    stats,
    servers,
    agents,
    setServers,
    setAgents,
    activeFilter,
    setActiveFilter,
    loading,
    error,
    refreshData: fetchData,
  };
};
