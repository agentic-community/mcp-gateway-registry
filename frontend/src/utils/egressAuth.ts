/**
 * API helpers for the per-user egress credential vault (third-party OBO).
 *
 * Mirrors the CSRF pattern used elsewhere (ServerConfigModal): mutating calls
 * fetch /api/auth/csrf-token and send it as X-CSRF-Token. All calls rely on the
 * session cookie for auth (withCredentials is the axios default here).
 */
import axios from 'axios';

export interface EgressConnection {
  provider: string;
  server_path: string;
  scopes: string[];
  expires_at: string | null;
  status: string;
  last_refreshed_at: string | null;
}

export interface AvailableEgressServer {
  server_path: string;
  server_name: string;
  provider: string;
}

async function csrfHeaders(): Promise<Record<string, string>> {
  const headers: Record<string, string> = {};
  try {
    const resp = await axios.get('/api/auth/csrf-token');
    const token = resp.data?.csrf_token;
    if (token) headers['X-CSRF-Token'] = token;
  } catch {
    // No CSRF token (e.g. bearer auth) — the backend dependency is flexible.
  }
  return headers;
}

/** List the current user's egress connections (tokens are never returned). */
export async function listConnections(): Promise<EgressConnection[]> {
  const resp = await axios.get('/api/egress-auth/connections');
  return resp.data as EgressConnection[];
}

/** List egress-enabled servers the current user can access (for the dropdown). */
export async function listAvailableServers(): Promise<AvailableEgressServer[]> {
  const resp = await axios.get('/api/egress-auth/available-servers');
  return resp.data as AvailableEgressServer[];
}

/** Begin consent for a server; returns the provider authorize URL to open. */
export async function initiateConsent(serverPath: string): Promise<string> {
  const headers = await csrfHeaders();
  const resp = await axios.post(
    '/api/egress-auth/initiate',
    { server_path: serverPath },
    { headers }
  );
  return resp.data.authorize_url as string;
}

/** Disconnect (revoke + delete the vault entry) for a (provider, server). */
export async function disconnect(provider: string, serverPath: string): Promise<void> {
  const headers = await csrfHeaders();
  const path = serverPath.replace(/^\//, '');
  await axios.delete(`/api/egress-auth/connections/${provider}/${path}`, { headers });
}

/** Whether the egress-auth feature is enabled (drives nav/page visibility). */
export async function isEgressAuthEnabled(): Promise<boolean> {
  try {
    // The connections endpoint 404s when the feature is disabled.
    await axios.get('/api/egress-auth/connections');
    return true;
  } catch (err) {
    if (axios.isAxiosError(err) && err.response?.status === 404) return false;
    // Any other error (401/500): assume enabled so the page can surface it.
    return true;
  }
}
