/**
 * Registry API functions
 */

import { apiGet, apiPost, apiPut, apiDelete } from './client';
import type {
  Server,
  ServerDetails,
  A2AAgent,
  RegisterServerRequest,
  EditServerRequest,
  RegisterA2AAgentRequest,
} from './types';

// ============================================================================
// Response Types (backend format)
// ============================================================================

interface ServersResponse {
  servers: Server[];
}

interface A2AAgentsResponse {
  agents: A2AAgent[];
  total_count: number;
}

// ============================================================================
// Servers API
// ============================================================================

/**
 * Fetch all registered MCP servers
 */
export async function getServers(): Promise<Server[]> {
  const response = await apiGet<ServersResponse>('/api/servers');
  return response.servers;
}

/**
 * Fetch a single server by path
 */
export async function getServer(path: string): Promise<ServerDetails> {
  return apiGet<ServerDetails>(`/api/servers/${encodeURIComponent(path)}`);
}

/**
 * Register a new MCP server
 */
export async function registerServer(data: RegisterServerRequest): Promise<Server> {
  return apiPost<Server>('/api/servers', data);
}

/**
 * Update an existing MCP server
 */
export async function updateServer(
  path: string,
  data: EditServerRequest
): Promise<Server> {
  return apiPut<Server>(`/api/servers/${encodeURIComponent(path)}`, data);
}

/**
 * Delete an MCP server
 */
export async function deleteServer(path: string): Promise<void> {
  return apiDelete<void>(`/api/servers/${encodeURIComponent(path)}`);
}

/**
 * Toggle server enabled state
 */
export async function toggleServer(
  path: string,
  enabled: boolean
): Promise<Server> {
  return apiPut<Server>(`/api/servers/${encodeURIComponent(path)}`, { enabled });
}

/**
 * Refresh server health status
 */
export async function refreshServerHealth(path: string): Promise<Server> {
  return apiPost<Server>(`/api/servers/${encodeURIComponent(path)}/refresh`);
}

// ============================================================================
// A2A Agents API
// ============================================================================

/**
 * Fetch all registered A2A agents
 */
export async function getA2AAgents(): Promise<A2AAgent[]> {
  const response = await apiGet<A2AAgentsResponse>('/api/agents');
  return response.agents;
}

/**
 * Fetch a single A2A agent by path
 */
export async function getA2AAgent(path: string): Promise<A2AAgent> {
  return apiGet<A2AAgent>(`/api/agents/${encodeURIComponent(path)}`);
}

/**
 * Register a new A2A agent
 */
export async function registerA2AAgent(
  data: RegisterA2AAgentRequest
): Promise<A2AAgent> {
  return apiPost<A2AAgent>('/api/agents', data);
}

/**
 * Update an existing A2A agent
 */
export async function updateA2AAgent(
  path: string,
  data: Partial<RegisterA2AAgentRequest>
): Promise<A2AAgent> {
  return apiPut<A2AAgent>(`/api/agents/${encodeURIComponent(path)}`, data);
}

/**
 * Delete an A2A agent
 */
export async function deleteA2AAgent(path: string): Promise<void> {
  return apiDelete<void>(`/api/agents/${encodeURIComponent(path)}`);
}

/**
 * Toggle A2A agent enabled state
 */
export async function toggleA2AAgent(
  path: string,
  enabled: boolean
): Promise<A2AAgent> {
  return apiPut<A2AAgent>(`/api/agents/${encodeURIComponent(path)}`, { enabled });
}
