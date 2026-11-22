/**
 * Data for the Tool access page (issue #2102): which groups can see and call
 * each tool. Reads GET /api/tool-access, which returns every row to admins and
 * read-only admins, and only the caller's own servers to anyone else.
 */

import { useCallback, useEffect, useRef, useState } from 'react';
import axios from 'axios';

export type ToolAccessServerType = 'mcp' | 'openapi' | 'virtual';

export interface ToolAccessGroup {
  group: string;
  idp_groups: string[];
  is_admin: boolean;
}

export interface ToolAccessCondition {
  label: string;
  /** Registry group names; details are in ToolAccessResponse.groups. */
  groups: string[];
}

export interface ToolAccessSummary {
  alone: string[];
  together: ToolAccessCondition[];
}

export interface ToolAccessRow {
  tool_name: string;
  server_name: string;
  server_path: string;
  server_type: ToolAccessServerType;
  enabled: boolean;
  blocked: boolean;
  blocked_source: string | null;
  backend_server_path: string | null;
  backend_tool_name: string | null;
  can_see: ToolAccessSummary;
  can_call: ToolAccessSummary;
}

export interface ToolAccessResponse {
  rows: ToolAccessRow[];
  /** Identity provider names and admin flag of every group named on the page. */
  groups: Record<string, ToolAccessGroup>;
  total_count: number;
  limit: number;
  offset: number;
  has_next: boolean;
  generated_at: string;
}

export interface ToolAccessQuery {
  q: string;
  serverPath: string;
  serverType: '' | ToolAccessServerType;
  group: string;
  includeDisabled: boolean;
  limit: number;
  offset: number;
}

export const TOOL_ACCESS_ENDPOINT = '/api/tool-access';

/** Build the query string parameters, leaving out the ones that are not set. */
export function toolAccessParams(query: ToolAccessQuery): Record<string, string | number | boolean> {
  const params: Record<string, string | number | boolean> = {
    limit: query.limit,
    offset: query.offset,
    include_disabled: query.includeDisabled,
  };
  if (query.q.trim()) params.q = query.q.trim();
  if (query.serverPath) params.server_path = query.serverPath;
  if (query.serverType) params.server_type = query.serverType;
  if (query.group) params.group = query.group;
  return params;
}

/**
 * Fetch one page of the listing for the given query. The previous page stays
 * on screen while the next one loads, and a response for an older query is
 * dropped, so search-as-you-type never shows results for stale text.
 */
export function useToolAccess(query: ToolAccessQuery) {
  const [data, setData] = useState<ToolAccessResponse | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const latestRequest = useRef(0);
  const params = JSON.stringify(toolAccessParams(query));

  const load = useCallback(async () => {
    latestRequest.current += 1;
    const requestId = latestRequest.current;
    setLoading(true);
    try {
      const response = await axios.get<ToolAccessResponse>(TOOL_ACCESS_ENDPOINT, {
        params: JSON.parse(params),
      });
      if (requestId !== latestRequest.current) return;
      setData(response.data);
      setError(null);
    } catch (err: unknown) {
      if (requestId !== latestRequest.current) return;
      const detail = axios.isAxiosError(err) ? err.response?.data?.detail : undefined;
      setError(typeof detail === 'string' ? detail : 'Could not load tool access.');
    } finally {
      if (requestId === latestRequest.current) setLoading(false);
    }
  }, [params]);

  useEffect(() => {
    load();
  }, [load]);

  return { data, loading, error, reload: load };
}

/**
 * Return true when the caller has at least one row, so the sidebar can show
 * the entry to server owners. Admins and read-only admins skip this call.
 */
export async function hasOwnToolAccessRows(): Promise<boolean> {
  try {
    const response = await axios.get<ToolAccessResponse>(TOOL_ACCESS_ENDPOINT, {
      params: { limit: 1 },
    });
    return response.data.total_count > 0;
  } catch {
    return false;
  }
}
