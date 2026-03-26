import { useEffect, useState } from 'react';
import axios from 'axios';

type EntityType = 'mcp_server' | 'tool' | 'a2a_agent' | 'skill' | 'virtual_server';

const DEFAULT_ENTITY_TYPES: EntityType[] = ['mcp_server', 'tool', 'a2a_agent', 'skill', 'virtual_server'];
const DEFAULT_ENTITY_TYPES_KEY = DEFAULT_ENTITY_TYPES.join('|');

export interface MatchingToolHit {
  tool_name: string;
  description?: string;
  relevance_score: number;
  match_context?: string;
}

export interface SyncMetadata {
  is_federated?: boolean;
  source_peer_id?: string;
  upstream_path?: string;
  last_synced_at?: string;
  is_read_only?: boolean;
  is_orphaned?: boolean;
  orphaned_at?: string;
}

export interface SemanticServerHit {
  path: string;
  server_name: string;
  description?: string;
  tags: string[];
  num_tools: number;
  is_enabled: boolean;
  relevance_score: number;
  match_context?: string;
  matching_tools: MatchingToolHit[];
  sync_metadata?: SyncMetadata;
  // Endpoint URL for agent connectivity (computed based on deployment mode)
  endpoint_url?: string;
  // Raw endpoint fields (for advanced use cases)
  proxy_pass_url?: string;
  mcp_endpoint?: string;
  sse_endpoint?: string;
  supported_transports?: string[];
}

export interface SemanticToolHit {
  server_path: string;
  server_name: string;
  tool_name: string;
  description?: string;
  inputSchema?: Record<string, any>;
  relevance_score: number;
  match_context?: string;
  // Endpoint URL for the parent MCP server
  endpoint_url?: string;
}

export interface SemanticAgentHit {
  // Only search-specific fields at top level; all agent details in agent_card
  path: string;
  relevance_score: number;
  match_context?: string;
  agent_card: Record<string, any>;
  trust_verified?: string;
}

export interface SemanticSkillHit {
  path: string;
  skill_name: string;
  description?: string;
  tags: string[];
  skill_md_url?: string;
  skill_md_raw_url?: string;
  version?: string;
  author?: string;
  visibility?: string;
  owner?: string;
  is_enabled?: boolean;
  health_status?: 'healthy' | 'unhealthy' | 'unknown';
  last_checked_time?: string;
  relevance_score: number;
  match_context?: string;
}

export interface VirtualServerToolHit {
  tool_name: string;
  description?: string;
  relevance_score?: number;
  match_context?: string;
  inputSchema?: Record<string, any>;
}

export interface SemanticVirtualServerHit {
  path: string;
  server_name: string;
  description?: string;
  tags: string[];
  num_tools: number;
  backend_count?: number;
  backend_paths?: string[];
  is_enabled: boolean;
  relevance_score: number;
  match_context?: string;
  matching_tools?: VirtualServerToolHit[];
  // Endpoint URL for agent connectivity (computed based on deployment mode)
  endpoint_url?: string;
}

export interface SemanticSearchResponse {
  query: string;
  servers: SemanticServerHit[];
  tools: SemanticToolHit[];
  agents: SemanticAgentHit[];
  skills: SemanticSkillHit[];
  virtual_servers: SemanticVirtualServerHit[];
  total_servers: number;
  total_tools: number;
  total_agents: number;
  total_skills: number;
  total_virtual_servers: number;
}

interface UseSemanticSearchOptions {
  enabled?: boolean;
  minLength?: number;
  maxResults?: number;
  entityTypes?: EntityType[];
  tags?: string[];
}

interface UseSemanticSearchReturn {
  results: SemanticSearchResponse | null;
  loading: boolean;
  error: string | null;
  debouncedQuery: string;
}

export const useSemanticSearch = (
  query: string,
  options: UseSemanticSearchOptions = {}
): UseSemanticSearchReturn => {
  const [results, setResults] = useState<SemanticSearchResponse | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [debouncedQuery, setDebouncedQuery] = useState('');

  const enabled = options.enabled ?? true;
  const minLength = options.minLength ?? 2;
  const maxResults = options.maxResults ?? 10;
  const entityTypes = options.entityTypes ?? DEFAULT_ENTITY_TYPES;
  const entityTypesKey =
    options.entityTypes?.join('|') ?? DEFAULT_ENTITY_TYPES_KEY;
  const tags = options.tags;
  const tagsKey = tags?.join('|') ?? '';

  // Debounce user input to minimize API calls
  useEffect(() => {
    const handle = setTimeout(() => {
      setDebouncedQuery(query.trim());
    }, 350);

    return () => clearTimeout(handle);
  }, [query]);

  useEffect(() => {
    // Allow search if we have a text query or explicit tag filters
    const hasQuery = debouncedQuery.length >= minLength;
    const hasTags = tags && tags.length > 0;
    if (!enabled || (!hasQuery && !hasTags)) {
      setResults(null);
      setError(null);
      setLoading(false);
      return;
    }

    let cancelled = false;
    const controller = new AbortController();

    const runSearch = async () => {
      setLoading(true);
      setError(null);
      try {
        const body: Record<string, unknown> = {
          query: debouncedQuery || '*',
          entity_types: entityTypes,
          max_results: maxResults,
        };
        if (tags && tags.length > 0) {
          body.tags = tags;
        }
        const response = await axios.post<SemanticSearchResponse>(
          '/api/search/semantic',
          body,
          { signal: controller.signal }
        );
        if (!cancelled) {
          setResults(response.data);
        }
      } catch (err: any) {
        if (axios.isCancel(err) || cancelled) return;
        const message =
          err.response?.data?.detail ||
          err.message ||
          'Semantic search failed.';
        setError(message);
        setResults(null);
      } finally {
        if (!cancelled) {
          setLoading(false);
        }
      }
    };

    runSearch();

    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [debouncedQuery, enabled, minLength, maxResults, entityTypesKey, tagsKey]);

  return { results, loading, error, debouncedQuery };
};
