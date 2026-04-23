/**
 * Hooks for fetching servers and their tools.
 *
 * Fetches all servers from /api/servers with descriptions
 * for use in searchable select components.
 */

import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';


export interface ServerInfo {
  path: string;
  name: string;
  description: string;
  type: 'mcp' | 'virtual';
}

export interface ToolInfo {
  name: string;
  description: string;
  serverPath: string;
}

interface ServerListResponse {
  servers: Array<{
    path: string;
    server_name?: string;
    name?: string;
    description?: string;
    [key: string]: unknown;
  }>;
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
      // Fetch both regular servers and virtual servers in parallel
      const [serversResponse, virtualServersResponse] = await Promise.all([
        axios.get<ServerListResponse>('/api/servers?limit=500'),
        axios.get<VirtualServerListResponse>('/api/virtual-servers'),
      ]);

      // Map regular MCP servers
      const mcpServers: ServerInfo[] = (serversResponse.data.servers || []).map((s) => ({
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

      // Combine and sort by type (MCP first), then by name
      const allServers = [...mcpServers, ...virtualServers];
      allServers.sort((a, b) => {
        // Sort by type first (mcp before virtual)
        if (a.type !== b.type) {
          return a.type === 'mcp' ? -1 : 1;
        }
        // Then by name
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
