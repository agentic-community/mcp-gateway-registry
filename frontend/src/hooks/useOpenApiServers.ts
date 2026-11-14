import { useState, useEffect, useCallback } from 'react';
import axios from 'axios';
import {
  OpenApiServer,
  OpenApiApiError,
  CreateOpenApiServerRequest,
  UpdateOpenApiServerRequest,
  CreateOpenApiTargetRequest,
  UpdateOpenApiTargetRequest,
  OpenApiTargetWriteResponse,
  OpenApiTarget,
  MyCredentialStatus,
  MyCredentialsResponse,
  SetMyCredentialRequest,
} from '../types/openApiServer';
import { extractErrorDetail } from '../utils/apiError';


const BASE_PATH = '/api/openapi-servers';


/**
 * Fetch a CSRF token and return it as a header, the same way
 * utils/egressAuth.ts does. Bearer-token callers have no CSRF token, and the
 * backend dependency accepts that, so a failed fetch sends no header.
 */
async function _csrfHeaders(): Promise<Record<string, string>> {
  const headers: Record<string, string> = {};
  try {
    const resp = await axios.get('/api/auth/csrf-token');
    const token = resp.data?.csrf_token;
    if (token) {
      headers['X-CSRF-Token'] = token;
    }
  } catch {
    // No CSRF token (for example bearer auth). The backend check is flexible.
  }
  return headers;
}


function _serverUrl(slug: string): string {
  return `${BASE_PATH}/${encodeURIComponent(slug)}`;
}


function _targetUrl(slug: string, name: string): string {
  return `${_serverUrl(slug)}/targets/${encodeURIComponent(name)}`;
}


/**
 * Find the error object in a response body. The routes return
 * `{"error": "<code>", "detail": "<message>", ...}`. FastAPI's HTTPException
 * wraps a dict detail one level down (`{"detail": {"error": ...}}`), so both
 * shapes are accepted.
 */
function _errorBody(err: unknown): Record<string, unknown> | null {
  const data = (err as { response?: { data?: unknown } })?.response?.data;
  if (!data || typeof data !== 'object') {
    return null;
  }
  const body = data as Record<string, unknown>;
  if (typeof body.error === 'string') {
    return body;
  }
  const detail = body.detail;
  if (detail && typeof detail === 'object' && !Array.isArray(detail)) {
    const inner = detail as Record<string, unknown>;
    if (typeof inner.error === 'string') {
      return inner;
    }
  }
  return null;
}


/**
 * Read the tag counts from a too_many_tools body. Accepts a map of tag to
 * count, or a list of `{tag, count}` entries.
 */
function _parseTagCounts(value: unknown): Record<string, number> | undefined {
  if (!value || typeof value !== 'object') {
    return undefined;
  }
  const result: Record<string, number> = {};
  if (Array.isArray(value)) {
    for (const entry of value) {
      const tag = entry?.tag ?? entry?.name;
      if (typeof tag === 'string') {
        result[tag] = Number(entry?.count) || 0;
      }
    }
    return result;
  }
  for (const [tag, count] of Object.entries(value as Record<string, unknown>)) {
    result[tag] = Number(count) || 0;
  }
  return result;
}


/**
 * Turn an axios error from an /api/openapi-servers route into an
 * OpenApiApiError, so a form can show each code beside the field it concerns.
 */
export function parseOpenApiError(
  err: unknown,
  fallback: string,
): OpenApiApiError {
  const status = (err as { response?: { status?: number } })?.response?.status ?? null;
  const body = _errorBody(err);
  if (!body) {
    return { status, code: null, message: extractErrorDetail(err, fallback) };
  }

  let message = fallback;
  if (typeof body.detail === 'string') {
    message = body.detail;
  } else if (typeof body.message === 'string') {
    message = body.message;
  }

  const parsed: OpenApiApiError = {
    status,
    code: body.error as string,
    message,
    tags: _parseTagCounts(body.tags),
  };
  if (typeof body.count === 'number') {
    parsed.count = body.count;
  }
  if (typeof body.limit === 'number') {
    parsed.limit = body.limit;
  }
  if (Array.isArray(body.names)) {
    parsed.names = body.names.filter((n): n is string => typeof n === 'string');
  }
  if (typeof body.field === 'string') {
    parsed.field = body.field;
  }
  return parsed;
}


/** List every OpenAPI MCP server the caller can see. */
export async function listOpenApiServers(): Promise<OpenApiServer[]> {
  const response = await axios.get(BASE_PATH);
  const data = response.data;
  if (Array.isArray(data)) {
    return data;
  }
  return data?.servers || [];
}


/** Get one server with its targets and the container's status for each. */
export async function getOpenApiServer(slug: string): Promise<OpenApiServer> {
  const response = await axios.get(_serverUrl(slug));
  return response.data;
}


export async function createOpenApiServer(
  data: CreateOpenApiServerRequest,
): Promise<OpenApiServer> {
  const headers = await _csrfHeaders();
  const response = await axios.post(BASE_PATH, data, { headers });
  return response.data;
}


export async function updateOpenApiServer(
  slug: string,
  data: UpdateOpenApiServerRequest,
): Promise<OpenApiServer> {
  const headers = await _csrfHeaders();
  const response = await axios.patch(_serverUrl(slug), data, { headers });
  return response.data;
}


export async function deleteOpenApiServer(slug: string): Promise<void> {
  const headers = await _csrfHeaders();
  await axios.delete(_serverUrl(slug), { headers });
}


export async function getOpenApiTarget(
  slug: string,
  name: string,
): Promise<OpenApiTarget> {
  const response = await axios.get(_targetUrl(slug, name));
  return response.data;
}


export async function addOpenApiTarget(
  slug: string,
  data: CreateOpenApiTargetRequest,
): Promise<OpenApiTargetWriteResponse> {
  const headers = await _csrfHeaders();
  const response = await axios.post(`${_serverUrl(slug)}/targets`, data, { headers });
  return response.data;
}


export async function updateOpenApiTarget(
  slug: string,
  name: string,
  data: UpdateOpenApiTargetRequest,
): Promise<OpenApiTargetWriteResponse> {
  const headers = await _csrfHeaders();
  const response = await axios.patch(_targetUrl(slug, name), data, { headers });
  return response.data;
}


export async function deleteOpenApiTarget(
  slug: string,
  name: string,
): Promise<void> {
  const headers = await _csrfHeaders();
  await axios.delete(_targetUrl(slug, name), { headers });
}


function _myCredentialUrl(slug: string, name: string): string {
  return `${_targetUrl(slug, name)}/my-credential`;
}


/**
 * Store or replace the caller's own key for a per-user target. The secret is
 * write-only: it is sent once and never returned or logged.
 */
export async function setMyCredential(
  slug: string,
  name: string,
  body: SetMyCredentialRequest,
): Promise<MyCredentialStatus> {
  const headers = await _csrfHeaders();
  const response = await axios.put(_myCredentialUrl(slug, name), body, { headers });
  return response.data;
}


/** The caller's key status for one per-user target (presence and expiry only). */
export async function getMyCredential(
  slug: string,
  name: string,
): Promise<MyCredentialStatus> {
  const response = await axios.get(_myCredentialUrl(slug, name));
  return response.data;
}


/** Delete the caller's own key for a per-user target. */
export async function deleteMyCredential(
  slug: string,
  name: string,
): Promise<void> {
  const headers = await _csrfHeaders();
  await axios.delete(_myCredentialUrl(slug, name), { headers });
}


/** The caller's key status for every per-user target of a server. */
export async function listMyCredentials(
  slug: string,
): Promise<MyCredentialStatus[]> {
  const response = await axios.get(`${_serverUrl(slug)}/my-credentials`);
  const data = response.data as MyCredentialsResponse | undefined;
  return Array.isArray(data?.targets) ? data.targets : [];
}


/**
 * Return type for the useOpenApiServers hook.
 */
interface UseOpenApiServersReturn {
  servers: OpenApiServer[];
  loading: boolean;
  error: string | null;
  refreshData: () => Promise<void>;
  createServer: (data: CreateOpenApiServerRequest) => Promise<OpenApiServer>;
  updateServer: (slug: string, data: UpdateOpenApiServerRequest) => Promise<OpenApiServer>;
  deleteServer: (slug: string) => Promise<void>;
}


/**
 * Hook for listing and managing OpenAPI MCP servers.
 *
 * The list refreshes after every create, update and delete. Target operations
 * are the standalone functions above, because the details modal owns them.
 */
export const useOpenApiServers = (): UseOpenApiServersReturn => {
  const [servers, setServers] = useState<OpenApiServer[]>([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  const fetchData = useCallback(async () => {
    try {
      setLoading(true);
      setError(null);
      setServers(await listOpenApiServers());
    } catch (err: unknown) {
      console.error('Failed to fetch OpenAPI MCP servers');
      setError(parseOpenApiError(err, 'Failed to fetch OpenAPI MCP servers').message);
      setServers([]);
    } finally {
      setLoading(false);
    }
  }, []);

  const createServer = useCallback(async (
    data: CreateOpenApiServerRequest,
  ): Promise<OpenApiServer> => {
    const created = await createOpenApiServer(data);
    await fetchData();
    return created;
  }, [fetchData]);

  const updateServer = useCallback(async (
    slug: string,
    data: UpdateOpenApiServerRequest,
  ): Promise<OpenApiServer> => {
    const updated = await updateOpenApiServer(slug, data);
    await fetchData();
    return updated;
  }, [fetchData]);

  const deleteServer = useCallback(async (
    slug: string,
  ): Promise<void> => {
    await deleteOpenApiServer(slug);
    await fetchData();
  }, [fetchData]);

  useEffect(() => {
    fetchData();
  }, [fetchData]);

  return {
    servers,
    loading,
    error,
    refreshData: fetchData,
    createServer,
    updateServer,
    deleteServer,
  };
};
