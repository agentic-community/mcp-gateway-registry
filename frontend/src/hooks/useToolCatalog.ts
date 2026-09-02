/**
 * Hooks for fetching servers and their tools.
 *
 * Fetches all servers from /api/servers with descriptions
 * for use in searchable select components.
 */

import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import { fetchAllPages } from '../utils/fetchAllPages';


// The scope editor's value space is chosen by `type`: MCP method tokens for
// 'mcp'/'virtual', HTTP verbs for proxied non-MCP entities. `entityType` carries
// the canonical type token used to build the authz key ({entityType}/{path}) for
// proxied skills/agents/custom entities; it is undefined for plain mcp/virtual.
export interface ServerInfo {
  path: string;
  name: string;
  description: string;
  type: 'mcp' | 'virtual' | 'skill' | 'a2a_agent' | 'custom';
  entityType?: string;
}

export interface ToolInfo {
  name: string;
  description: string;
  serverPath: string;
}

interface VirtualServerListResponse {
  virtual_servers: Array<{
    path: string;
    name: string;
    description?: string;
    enabled?: boolean;
    [key: string]: unknown;
  }>;
}

interface ProxiedEntitiesResponse {
  proxied_entities: Array<{
    entity_type: string;
    path: string;
    name: string;
    authz_key: string;
  }>;
  count: number;
}

// Map a proxied entity's canonical type token to the ServerInfo.type union.
function _proxiedType(entityType: string): ServerInfo['type'] {
  if (entityType === 'skill') return 'skill';
  if (entityType === 'a2a_agent') return 'a2a_agent';
  return 'custom'; // any admin-defined custom descriptor name
}

interface ToolCatalogResponse {
  tools: Array<{
    tool_name: string;
    server_path: string;
    server_name: string;
    description: string;
  }>;
  by_server: Record<string, Array<{
    tool_name: string;
    description: string;
  }>>;
}

interface UseServerListReturn {
  servers: ServerInfo[];
  isLoading: boolean;
  error: string | null;
  refetch: () => Promise<void>;
}

interface UseServerToolsReturn {
  tools: ToolInfo[];
  isLoading: boolean;
  error: string | null;
}


/**
 * Hook to fetch all available servers with descriptions.
 * Includes both regular MCP servers and virtual servers.
 */
export function useServerList(): UseServerListReturn {
  const [servers, setServers] = useState<ServerInfo[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchServers = useCallback(async () => {
    setIsLoading(true);
    setError(null);

    try {
      // Fetch regular servers, virtual servers, and proxied non-MCP entities in
      // parallel. Issue #880: page through /api/servers (not a single hard-capped
      // page). The proxied-entities endpoint is admin-only + may 403 for a
      // non-admin viewer; tolerate that (the scope editor is admin-only anyway).
      const [rawServers, virtualServersResponse, proxiedResponse] = await Promise.all([
        fetchAllPages<{
          path: string;
          server_name?: string;
          name?: string;
          description?: string;
        }>({
          url: '/api/servers',
          itemsKey: 'servers',
        }),
        axios.get<VirtualServerListResponse>('/api/virtual-servers'),
        axios
          .get<ProxiedEntitiesResponse>('/api/iam/proxied-entities')
          .catch(() => ({ data: { proxied_entities: [], count: 0 } })),
      ]);

      // Map regular MCP servers
      const mcpServers: ServerInfo[] = rawServers.map((s) => ({
        path: s.path,
        name: s.server_name || s.name || s.path,
        description: s.description || '',
        type: 'mcp' as const,
      }));

      // Map virtual servers (only enabled ones)
      const virtualServers: ServerInfo[] = (virtualServersResponse.data.virtual_servers || [])
        .filter((vs) => vs.enabled !== false)
        .map((vs) => ({
          path: vs.path,
          name: vs.name || vs.path,
          description: vs.description || '',
          type: 'virtual' as const,
        }));

      // Map proxied non-MCP entities (skills/agents/custom). The `path` here is
      // the canonical authz_key (entity_type/registered_path) so the scope editor
      // writes the SAME string the auth-server expects — writing a bare slugged
      // path would silently 403 on the generic hop.
      const proxiedEntities: ServerInfo[] = (proxiedResponse.data.proxied_entities || []).map(
        (p) => ({
          path: p.authz_key,
          name: p.name,
          description: `Proxied ${p.entity_type}`,
          type: _proxiedType(p.entity_type),
          entityType: p.entity_type,
        }),
      );

      // Combine and sort by type rank (mcp, virtual, then proxied), then by name.
      const typeRank: Record<ServerInfo['type'], number> = {
        mcp: 0,
        virtual: 1,
        skill: 2,
        a2a_agent: 3,
        custom: 4,
      };
      const allServers = [...mcpServers, ...virtualServers, ...proxiedEntities];
      allServers.sort((a, b) => {
        if (a.type !== b.type) {
          return typeRank[a.type] - typeRank[b.type];
        }
        return a.name.localeCompare(b.name);
      });

      setServers(allServers);
    } catch (err) {
      const message = err instanceof Error ? err.message : 'Failed to fetch servers';
      setError(message);
      setServers([]);
    } finally {
      setIsLoading(false);
    }
  }, []);

  useEffect(() => {
    fetchServers();
  }, [fetchServers]);

  return {
    servers,
    isLoading,
    error,
    refetch: fetchServers,
  };
}


/**
 * Hook to fetch tools for a specific server.
 * Returns empty array if serverPath is empty or '*'.
 */
export function useServerTools(serverPath: string): UseServerToolsReturn {
  const [tools, setTools] = useState<ToolInfo[]>([]);
  const [isLoading, setIsLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    // Don't fetch for empty or wildcard
    if (!serverPath || serverPath === '*') {
      setTools([]);
      setIsLoading(false);
      return;
    }

    const fetchTools = async () => {
      setIsLoading(true);
      setError(null);

      try {
        const response = await axios.get<ToolCatalogResponse>(
          `/api/tool-catalog?server_path=${encodeURIComponent(serverPath)}`
        );
        const data = response.data;

        // Extract tools from the response
        const toolList: ToolInfo[] = (data.tools || []).map((t) => ({
          name: t.tool_name,
          description: t.description || '',
          serverPath: t.server_path,
        }));

        // Sort by name
        toolList.sort((a, b) => a.name.localeCompare(b.name));

        setTools(toolList);
      } catch (err) {
        const message = err instanceof Error ? err.message : 'Failed to fetch tools';
        setError(message);
        setTools([]);
      } finally {
        setIsLoading(false);
      }
    };

    fetchTools();
  }, [serverPath]);

  return {
    tools,
    isLoading,
    error,
  };
}
