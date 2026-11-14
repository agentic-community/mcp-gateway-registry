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

/** Vault entries for OpenAPI MCP per-user keys use provider `openapi-<target>`. */
const OPENAPI_PROVIDER_PREFIX: string = 'openapi-';


/**
 * The OpenAPI target of a vault entry, or null for any other entry. Per-user
 * OpenAPI keys are stored with provider `openapi-<target>` and server_path
 * `/<slug>`, and are managed through the OpenAPI routes, not egress-pat.
 */
export function openApiTargetOf(provider: string): string | null {
  if (!provider.startsWith(OPENAPI_PROVIDER_PREFIX)) return null;
  const target = provider.slice(OPENAPI_PROVIDER_PREFIX.length);
  return target || null;
}


/** The egress mode of an OpenAPI MCP server with per-user targets. */
export const OPENAPI_USER_MODE: string = 'openapi_user';


/** The server edit form fields that make up an egress config request. */
export interface EgressConfigFormFields {
  egress_auth_mode: 'none' | 'oauth_user' | 'obo_exchange' | 'pat';
  egress_provider: string;
  egress_client_id: string;
  egress_client_secret: string;
  egress_scopes: string;
  egress_custom_authorize_url: string;
  egress_custom_token_url: string;
  egress_custom_token_auth_style: string;
  egress_custom_resource: string;
  egress_expected_issuer: string;
  egress_target_audience: string;
  egress_list_tools_before_consent: boolean;
}

/** Egress modes whose users have a pre-consent state that can list stored tools. */
export const LIST_BEFORE_CONSENT_MODES: ReadonlyArray<string> = ['oauth_user', 'pat'];

/**
 * Build the body for POST /api/servers/{path}/egress-auth from the edit form.
 *
 * The request replaces the whole egress config, so a field this function leaves
 * out is reset on the server. Every mode goes through here so that a new field
 * (such as list_tools_before_consent, issue #1986) cannot be missed in one mode.
 */
export function buildEgressConfigBody(form: EgressConfigFormFields): Record<string, unknown> {
  const mode = form.egress_auth_mode;
  const scopes = form.egress_scopes
    .split(/[,\s]+/)
    .map(s => s.trim())
    .filter(Boolean);
  if (mode === 'obo_exchange') {
    return {
      egress_auth_mode: 'obo_exchange',
      target_audience: form.egress_target_audience.trim(),
      scopes,
    };
  }
  if (mode === 'oauth_user') {
    return {
      egress_auth_mode: 'oauth_user',
      egress_provider: form.egress_provider.trim(),
      client_id: form.egress_client_id.trim(),
      // Blank secret on edit keeps the stored one (backend semantics).
      client_secret: form.egress_client_secret || undefined,
      scopes,
      custom_authorize_url: form.egress_custom_authorize_url || undefined,
      custom_token_url: form.egress_custom_token_url || undefined,
      custom_token_auth_style: form.egress_custom_token_auth_style || undefined,
      custom_resource: form.egress_custom_resource || undefined,
      expected_issuer: form.egress_expected_issuer || undefined,
      list_tools_before_consent: form.egress_list_tools_before_consent,
    };
  }
  if (mode === 'pat') {
    // The inject header is inherited from Backend Authentication at vend time,
    // so it is not sent here.
    return {
      egress_auth_mode: 'pat',
      egress_provider: form.egress_provider.trim(),
      list_tools_before_consent: form.egress_list_tools_before_consent,
    };
  }
  return { egress_auth_mode: 'none' };
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
    // OpenAPI per-user keys have their own "Your API keys" button on the card.
    if (s.egress_auth_mode === OPENAPI_USER_MODE) continue;
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
