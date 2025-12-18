import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactNode } from 'react';
import { http, HttpResponse } from 'msw';
import { server } from '@/test/mocks/server';
import { mockServers, mockServerDetails } from '@/test/mocks/handlers';
import type { Server } from '@/api/types';
import {
  useServers,
  useServer,
  useRegisterServer,
  useUpdateServer,
  useDeleteServer,
  useToggleServer,
  useRefreshServer,
  filterServers,
  type ServerFilter,
} from '../hooks';

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

describe('Server Hooks', () => {
  beforeEach(() => {
    server.resetHandlers();
  });

  describe('useServers', () => {
    it('returns servers from the API', async () => {
      const { result } = renderHook(() => useServers(), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(true);

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.servers).toHaveLength(5);
      expect(result.current.servers[0].display_name).toBe('SQLite Server');
      expect(result.current.isError).toBe(false);
    });

    it('handles API errors', async () => {
      server.use(
        http.get('/api/servers', () => {
          return HttpResponse.json(
            { detail: 'Internal server error' },
            { status: 500 }
          );
        })
      );

      const { result } = renderHook(() => useServers(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.isError).toBe(true);
      expect(result.current.servers).toHaveLength(0);
    });
  });

  describe('useServer', () => {
    it('returns server details from the API', async () => {
      const { result } = renderHook(() => useServer('sqlite'), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.server).not.toBeNull();
      expect(result.current.server?.display_name).toBe('SQLite Server');
      expect(result.current.server?.tools).toHaveLength(3);
      expect(result.current.isError).toBe(false);
    });

    it('handles non-existent server', async () => {
      const { result } = renderHook(() => useServer('non-existent'), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.isError).toBe(true);
      expect(result.current.server).toBeNull();
    });
  });

  describe('useRegisterServer', () => {
    it('registers a new server', async () => {
      const { result } = renderHook(() => useRegisterServer(), {
        wrapper: createWrapper(),
      });

      expect(result.current.isRegistering).toBe(false);

      let newServer: Server | undefined;
      await act(async () => {
        newServer = await result.current.registerServer({
          name: 'New Server',
          path: 'new-server',
          proxy_pass_url: 'http://localhost:9000/mcp',
          description: 'A new test server',
          tags: ['test'],
        });
      });

      expect(newServer).toBeDefined();
      expect(newServer?.display_name).toBe('New Server');
      expect(newServer?.path).toBe('new-server');
    });
  });

  describe('useUpdateServer', () => {
    it('updates an existing server', async () => {
      const { result } = renderHook(() => useUpdateServer(), {
        wrapper: createWrapper(),
      });

      let updatedServer: Server | undefined;
      await act(async () => {
        updatedServer = await result.current.updateServer('sqlite', {
          name: 'Updated SQLite Server',
        });
      });

      expect(updatedServer).toBeDefined();
      expect(updatedServer?.display_name).toBe('Updated SQLite Server');
    });
  });

  describe('useDeleteServer', () => {
    it('deletes a server', async () => {
      const { result } = renderHook(() => useDeleteServer(), {
        wrapper: createWrapper(),
      });

      await act(async () => {
        await result.current.deleteServer('sqlite');
      });

      expect(result.current.isDeleting).toBe(false);
      expect(result.current.error).toBeNull();
    });
  });

  describe('useToggleServer', () => {
    it('toggles server enabled state', async () => {
      const { result } = renderHook(() => useToggleServer(), {
        wrapper: createWrapper(),
      });

      let toggled: Server | undefined;
      await act(async () => {
        toggled = await result.current.toggleServer('sqlite', false);
      });

      expect(toggled).toBeDefined();
      expect(toggled?.is_enabled).toBe(false);
    });
  });

  describe('useRefreshServer', () => {
    it('refreshes server health', async () => {
      const { result } = renderHook(() => useRefreshServer(), {
        wrapper: createWrapper(),
      });

      let refreshed: Server | undefined;
      await act(async () => {
        refreshed = await result.current.refreshServer('sqlite');
      });

      expect(refreshed).toBeDefined();
      expect(refreshed?.health_status).toBe('healthy');
    });
  });

  describe('filterServers', () => {
    const servers = mockServers;

    it('filters by search term (name)', () => {
      const filter: ServerFilter = { search: 'sqlite', status: 'all', healthStatus: 'all' };
      const result = filterServers(servers, filter);

      expect(result).toHaveLength(1);
      expect(result[0].display_name).toBe('SQLite Server');
    });

    it('filters by search term (tag)', () => {
      const filter: ServerFilter = { search: 'database', status: 'all', healthStatus: 'all' };
      const result = filterServers(servers, filter);

      expect(result).toHaveLength(1);
      expect(result[0].path).toBe('sqlite');
    });

    it('filters by enabled status', () => {
      const filter: ServerFilter = { search: '', status: 'enabled', healthStatus: 'all' };
      const result = filterServers(servers, filter);

      expect(result).toHaveLength(4);
      expect(result.every(s => s.is_enabled)).toBe(true);
    });

    it('filters by disabled status', () => {
      const filter: ServerFilter = { search: '', status: 'disabled', healthStatus: 'all' };
      const result = filterServers(servers, filter);

      expect(result).toHaveLength(1);
      expect(result[0].is_enabled).toBe(false);
    });

    it('filters by health status', () => {
      const filter: ServerFilter = { search: '', status: 'all', healthStatus: 'healthy' };
      const result = filterServers(servers, filter);

      expect(result).toHaveLength(4);
      expect(result.every(s => s.health_status === 'healthy')).toBe(true);
    });

    it('combines multiple filters', () => {
      const filter: ServerFilter = { search: 'server', status: 'enabled', healthStatus: 'healthy' };
      const result = filterServers(servers, filter);

      expect(result).toHaveLength(4);
    });

    it('returns all servers with no filters', () => {
      const filter: ServerFilter = { search: '', status: 'all', healthStatus: 'all' };
      const result = filterServers(servers, filter);

      expect(result).toHaveLength(5);
    });
  });
});
