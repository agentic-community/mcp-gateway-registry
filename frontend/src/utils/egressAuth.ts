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
  egress_auth_mode?: string;
  // Server-built gateway front-door URL (oauth_user only; null for pat). Built
  // from the configured registry_url so the browser never guesses the base.
  connect_url?: string | null;
}

/**
 * Merged per-server egress state consumed by BOTH the server-card icon and the
 * connect-modal callout, so the two surfaces never disagree.
 */
export interface EgressCardState {
  mode: 'oauth_user' | 'pat';
  provider: string;
  connectUrl: string; // from connect_url (oauth_user); '' for pat (never opened)
  connected: boolean; // a connection exists for this server_path
  status: string | null; // connection status: 'active' | 'refresh_failed' | ... (null if not connected)
  expiresAt: string | null; // connection expiry ISO string (null if not connected)
  // Connected but the token is dead (refresh_failed or past expiry): drives the
  // one-click "Reconnect" affordance so it stops being a silent "0 tools".
  needsReconnect: boolean;
}

export interface PatStatus {
  configured: boolean;
  expires_at: string | null;
  expired: boolean;
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

/**
 * Fetch the two per-user egress lists once and merge them into a per-server-path
 * map, consumed by BOTH the server-card icon and the connect-modal callout.
 * Returns an empty map when the feature is disabled (available-servers 404s) or
 * the caller is not a per-user principal (returns []).
 */
export async function loadEgressCardState(): Promise<Map<string, EgressCardState>> {
  const byPath = new Map<string, EgressCardState>();
  let available: AvailableEgressServer[] = [];
  try {
    available = await listAvailableServers();
  } catch {
    // 404 = feature disabled; anything else = treat as "no egress affordance".
    return byPath;
  }
  let connections: EgressConnection[] = [];
  try {
    connections = await listConnections();
  } catch {
    connections = [];
  }
  // Index connections by server_path for O(1) lookup of status/expiry.
  const connByPath = new Map(connections.map(c => [c.server_path, c]));
  const nowMs = Date.now();
  for (const s of available) {
    const mode = s.egress_auth_mode === 'pat' ? 'pat' : 'oauth_user';
    const conn = connByPath.get(s.server_path);
    const connected = !!conn;
    const status = conn?.status ?? null;
    const expiresAt = conn?.expires_at ?? null;
    // A dead token (refresh_failed or expired) is still "connected" but needs a
    // one-click reconnect so it stops being a silent "0 tools".
    const expired = !!expiresAt && Date.parse(expiresAt) < nowMs;
    const needsReconnect = connected && (status === 'refresh_failed' || expired);
    byPath.set(s.server_path, {
      mode,
      provider: s.provider,
      connectUrl: s.connect_url || '', // '' for pat (backend sends null); pat never opens it
      connected,
      status,
      expiresAt,
      needsReconnect,
    });
  }
  return byPath;
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

/**
 * Submit (or replace) the current user's static PAT / API key for a `pat`-mode
 * server. The secret is write-only: it is never returned or logged here.
 */
export async function setEgressPat(
  serverPath: string,
  secret: string,
  ttlValue: number,
  ttlUnit: string
): Promise<{ configured: boolean; expires_at: string | null }> {
  const headers = await csrfHeaders();
  const path = serverPath.replace(/^\//, '');
  const resp = await axios.put(
    `/api/servers/${path}/egress-pat`,
    { secret, ttl_value: ttlValue, ttl_unit: ttlUnit },
    { headers }
  );
  return resp.data as { configured: boolean; expires_at: string | null };
}

/** Get the PAT status for a server (presence + expiry only; never the secret). */
export async function getEgressPatStatus(serverPath: string): Promise<PatStatus> {
  const path = serverPath.replace(/^\//, '');
  const resp = await axios.get(`/api/servers/${path}/egress-pat`);
  return resp.data as PatStatus;
}

/** Delete the current user's stored PAT for a server. */
export async function deleteEgressPat(serverPath: string): Promise<void> {
  const headers = await csrfHeaders();
  const path = serverPath.replace(/^\//, '');
  await axios.delete(`/api/servers/${path}/egress-pat`, { headers });
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
