import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactNode } from 'react';
import { http, HttpResponse } from 'msw';
import { server } from '@/test/mocks/server';
import { mockScopeCatalog } from '@/test/mocks/handlers';
import {
  useScopeCatalog,
  useScope,
  useScopesByServer,
  useScopeNames,
  transformScopeToInfo,
  filterScopes,
  getServersDisplay,
  getMethodsDisplay,
  getToolsDisplay,
} from '../hooks';
import type { ScopeDefinition, ScopeInfo } from '@/api/types';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
        staleTime: 0,
      },
    },
  });

  return function Wrapper({ children }: { children: ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>
    );
  };
}

describe('Scopes Hooks', () => {
  beforeEach(() => {
    server.resetHandlers();
  });

  describe('useScopeCatalog', () => {
    it('fetches the scope catalog', async () => {
      const { result } = renderHook(() => useScopeCatalog(), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(true);

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.catalog).toBeDefined();
      expect(result.current.scopes.length).toBeGreaterThan(0);
      expect(result.current.isAvailable).toBe(true);
    });

    it('returns scope infos', async () => {
      const { result } = renderHook(() => useScopeCatalog(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.scopes.length).toBe(4);
      const scopeNames = result.current.scopes.map((s) => s.name);
      expect(scopeNames).toContain('registry-admins');
      expect(scopeNames).toContain('sqlite.manage');
    });

    it('handles API error gracefully', async () => {
      server.use(
        http.get('/api/scopes/catalog', () => {
          return HttpResponse.json({ detail: 'Not found' }, { status: 404 });
        })
      );

      const { result } = renderHook(() => useScopeCatalog(), {
        wrapper: createWrapper(),
      });

      // Allow extra time for retry to complete
      await waitFor(
        () => {
          expect(result.current.isLoading).toBe(false);
        },
        { timeout: 3000 }
      );

      expect(result.current.isError).toBe(true);
      expect(result.current.isAvailable).toBe(false);
    });
  });

  describe('useScope', () => {
    it('returns a specific scope by name', async () => {
      const { result } = renderHook(() => useScope('sqlite.manage'), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.scope).toBeDefined();
      expect(result.current.scope?.name).toBe('sqlite.manage');
      expect(result.current.scopeInfo).toBeDefined();
    });

    it('returns null for non-existent scope', async () => {
      const { result } = renderHook(() => useScope('non-existent'), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.scope).toBeNull();
      expect(result.current.scopeInfo).toBeNull();
    });
  });

  describe('useScopesByServer', () => {
    it('returns scopes with access to a specific server', async () => {
      const { result } = renderHook(() => useScopesByServer('sqlite'), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // sqlite.manage and registry-users-lob1 have sqlite access
      // registry-admins has wildcard access
      expect(result.current.scopes.length).toBeGreaterThanOrEqual(2);
    });
  });

  describe('useScopeNames', () => {
    it('returns all scope names', async () => {
      const { result } = renderHook(() => useScopeNames(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.scopeNames.length).toBe(4);
      expect(result.current.scopeNames).toContain('registry-admins');
      expect(result.current.scopeNames).toContain('sqlite.manage');
    });
  });
});

describe('Scope Utility Functions', () => {
  describe('transformScopeToInfo', () => {
    it('transforms a scope definition to scope info', () => {
      const scope: ScopeDefinition = mockScopeCatalog.scopes['sqlite.manage'];
      const info = transformScopeToInfo(scope);

      expect(info.name).toBe('sqlite.manage');
      expect(info.servers).toContain('sqlite');
      expect(info.all_tools).toBe(true);
    });

    it('handles wildcard servers', () => {
      const scope: ScopeDefinition = mockScopeCatalog.scopes['registry-admins'];
      const info = transformScopeToInfo(scope);

      expect(info.all_servers).toBe(true);
      expect(info.all_methods).toBe(true);
      expect(info.all_tools).toBe(true);
    });

    it('extracts agent actions', () => {
      const scope: ScopeDefinition = mockScopeCatalog.scopes['registry-admins'];
      const info = transformScopeToInfo(scope);

      expect(info.agent_actions.length).toBeGreaterThan(0);
      expect(info.agent_actions).toContain('list_agents');
    });
  });

  describe('filterScopes', () => {
    const mockScopes: ScopeInfo[] = [
      {
        name: 'registry-admins',
        servers: [],
        methods: [],
        tools: [],
        all_servers: true,
        all_methods: true,
        all_tools: true,
        agent_actions: ['list_agents'],
      },
      {
        name: 'sqlite.manage',
        servers: ['sqlite'],
        methods: ['tools/call'],
        tools: ['read_query'],
        all_servers: false,
        all_methods: false,
        all_tools: false,
        agent_actions: [],
      },
    ];

    it('filters by scope name', () => {
      const filtered = filterScopes(mockScopes, 'admin');
      expect(filtered.length).toBe(1);
      expect(filtered[0].name).toBe('registry-admins');
    });

    it('filters by server name', () => {
      const filtered = filterScopes(mockScopes, 'sqlite');
      expect(filtered.length).toBe(1);
      expect(filtered[0].name).toBe('sqlite.manage');
    });

    it('filters by tool name', () => {
      const filtered = filterScopes(mockScopes, 'read_query');
      expect(filtered.length).toBe(1);
      expect(filtered[0].name).toBe('sqlite.manage');
    });

    it('filters by agent action', () => {
      const filtered = filterScopes(mockScopes, 'list_agents');
      expect(filtered.length).toBe(1);
      expect(filtered[0].name).toBe('registry-admins');
    });

    it('returns all scopes for empty query', () => {
      const filtered = filterScopes(mockScopes, '');
      expect(filtered.length).toBe(2);
    });
  });

  describe('getServersDisplay', () => {
    it('shows "All servers" for wildcard', () => {
      const scope: ScopeInfo = {
        name: 'test',
        servers: [],
        methods: [],
        tools: [],
        all_servers: true,
        all_methods: false,
        all_tools: false,
        agent_actions: [],
      };
      expect(getServersDisplay(scope)).toBe('All servers');
    });

    it('shows server names for limited servers', () => {
      const scope: ScopeInfo = {
        name: 'test',
        servers: ['sqlite', 'api'],
        methods: [],
        tools: [],
        all_servers: false,
        all_methods: false,
        all_tools: false,
        agent_actions: [],
      };
      expect(getServersDisplay(scope)).toBe('sqlite, api');
    });

    it('truncates long server list', () => {
      const scope: ScopeInfo = {
        name: 'test',
        servers: ['s1', 's2', 's3', 's4', 's5'],
        methods: [],
        tools: [],
        all_servers: false,
        all_methods: false,
        all_tools: false,
        agent_actions: [],
      };
      expect(getServersDisplay(scope)).toBe('s1, s2, s3 +2 more');
    });
  });

  describe('getMethodsDisplay', () => {
    it('shows "All methods" for wildcard', () => {
      const scope: ScopeInfo = {
        name: 'test',
        servers: [],
        methods: [],
        tools: [],
        all_servers: false,
        all_methods: true,
        all_tools: false,
        agent_actions: [],
      };
      expect(getMethodsDisplay(scope)).toBe('All methods');
    });
  });

  describe('getToolsDisplay', () => {
    it('shows "All tools" for wildcard', () => {
      const scope: ScopeInfo = {
        name: 'test',
        servers: [],
        methods: [],
        tools: [],
        all_servers: false,
        all_methods: false,
        all_tools: true,
        agent_actions: [],
      };
      expect(getToolsDisplay(scope)).toBe('All tools');
    });

    it('shows "Inherits from server" when no specific tools', () => {
      const scope: ScopeInfo = {
        name: 'test',
        servers: [],
        methods: [],
        tools: [],
        all_servers: false,
        all_methods: false,
        all_tools: false,
        agent_actions: [],
      };
      expect(getToolsDisplay(scope)).toBe('Inherits from server');
    });
  });
});
