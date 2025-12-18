import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactNode } from 'react';
import { http, HttpResponse } from 'msw';
import { server } from '@/test/mocks/server';
import {
  mockServers,
  mockA2AAgents,
  mockEnforceAIAgents,
} from '@/test/mocks/handlers';
import {
  useServers,
  useServerStats,
  useA2AAgents,
  useA2AAgentStats,
  useEnforceAIAgents,
  useEnforceAIAgentStats,
  useConnectionTest,
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

describe('Overview Hooks', () => {
  beforeEach(() => {
    server.resetHandlers();
  });

  describe('useServers', () => {
    it('returns servers from the API', async () => {
      const { result } = renderHook(() => useServers(), {
        wrapper: createWrapper(),
      });

      // Initially loading
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

  describe('useServerStats', () => {
    it('calculates correct server statistics', async () => {
      const { result } = renderHook(() => useServerStats(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Mock data has 5 servers: 4 enabled, 1 disabled
      expect(result.current.total).toBe(5);
      expect(result.current.enabled).toBe(4);
      expect(result.current.disabled).toBe(1);
    });
  });

  describe('useA2AAgents', () => {
    it('returns A2A agents from the API', async () => {
      const { result } = renderHook(() => useA2AAgents(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.agents).toHaveLength(2);
      expect(result.current.agents[0].name).toBe('Code Assistant');
      expect(result.current.isError).toBe(false);
    });
  });

  describe('useA2AAgentStats', () => {
    it('calculates correct A2A agent statistics', async () => {
      const { result } = renderHook(() => useA2AAgentStats(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Mock data has 2 agents: 1 enabled, 1 disabled
      expect(result.current.total).toBe(2);
      expect(result.current.enabled).toBe(1);
      expect(result.current.disabled).toBe(1);
    });
  });

  describe('useEnforceAIAgents', () => {
    it('returns EnforceAI agents from the API', async () => {
      const { result } = renderHook(() => useEnforceAIAgents(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.agents).toHaveLength(2);
      expect(result.current.agents[0].agent_id).toBe(
        '9d2724e9-1753-4493-8993-0d6986754414'
      );
      expect(result.current.isError).toBe(false);
    });

    it('handles 404 (EnforceAI not enabled) gracefully', async () => {
      server.use(
        http.get('/enforceai/agents', () => {
          return HttpResponse.json(
            { detail: 'EnforceAI not enabled' },
            { status: 404 }
          );
        })
      );

      const { result } = renderHook(() => useEnforceAIAgents(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.isError).toBe(true);
      expect(result.current.errorCode).toBe(404);
      expect(result.current.agents).toHaveLength(0);
    });
  });

  describe('useEnforceAIAgentStats', () => {
    it('calculates correct EnforceAI agent statistics', async () => {
      const { result } = renderHook(() => useEnforceAIAgentStats(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      // Mock data has 2 agents: 1 active, 1 revoked
      expect(result.current.total).toBe(2);
      expect(result.current.active).toBe(1);
      expect(result.current.revoked).toBe(1);
      expect(result.current.enforceAIEnabled).toBe(true);
    });

    it('indicates EnforceAI not enabled when 404', async () => {
      server.use(
        http.get('/enforceai/agents', () => {
          return HttpResponse.json(
            { detail: 'EnforceAI not enabled' },
            { status: 404 }
          );
        })
      );

      const { result } = renderHook(() => useEnforceAIAgentStats(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.enforceAIEnabled).toBe(false);
      expect(result.current.isError).toBe(false); // 404 is not treated as error
    });
  });

  describe('useConnectionTest', () => {
    it('initializes with not tested status', () => {
      const { result } = renderHook(() => useConnectionTest(), {
        wrapper: createWrapper(),
      });

      expect(result.current.status.registry.tested).toBe(false);
      expect(result.current.status.enforceai.tested).toBe(false);
      expect(result.current.isTesting).toBe(false);
    });

    it('tests connections successfully', async () => {
      const { result } = renderHook(() => useConnectionTest(), {
        wrapper: createWrapper(),
      });

      // Trigger connection test
      result.current.testConnections();

      // Wait for test to complete - the mutation may complete quickly
      await waitFor(
        () => {
          expect(result.current.status.registry.tested).toBe(true);
        },
        { timeout: 5000 }
      );

      // Check results
      expect(result.current.status.registry.reachable).toBe(true);
      expect(result.current.status.registry.elapsed_ms).toBeGreaterThanOrEqual(0);

      expect(result.current.status.enforceai.tested).toBe(true);
      expect(result.current.status.enforceai.reachable).toBe(true);
    });

    it('handles connection failures', async () => {
      server.use(
        http.get('/api/servers', () => {
          return HttpResponse.json(
            { detail: 'Connection failed' },
            { status: 503 }
          );
        }),
        http.get('/enforceai/agents', () => {
          return HttpResponse.json(
            { detail: 'Connection failed' },
            { status: 503 }
          );
        })
      );

      const { result } = renderHook(() => useConnectionTest(), {
        wrapper: createWrapper(),
      });

      // Trigger connection test
      result.current.testConnections();

      // Wait for test to complete
      await waitFor(() => {
        expect(result.current.status.registry.tested).toBe(true);
      });

      // Check results - should show as unreachable
      expect(result.current.status.registry.reachable).toBe(false);
      expect(result.current.status.enforceai.reachable).toBe(false);
    });
  });
});
