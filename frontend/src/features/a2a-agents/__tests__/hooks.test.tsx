import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactNode } from 'react';
import { http, HttpResponse } from 'msw';
import { server } from '@/test/mocks/server';
import { mockA2AAgents } from '@/test/mocks/handlers';
import type { A2AAgent } from '@/api/types';
import {
  useAgents,
  useAgent,
  useRegisterAgent,
  useUpdateAgent,
  useDeleteAgent,
  useToggleAgent,
  filterAgents,
  type AgentFilter,
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

describe('A2A Agent Hooks', () => {
  beforeEach(() => {
    server.resetHandlers();
  });

  describe('useAgents', () => {
    it('returns agents from the API', async () => {
      const { result } = renderHook(() => useAgents(), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(true);

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.agents).toHaveLength(2);
      expect(result.current.agents[0].name).toBe('Code Assistant');
      expect(result.current.isError).toBe(false);
    });

    it('handles API errors', async () => {
      server.use(
        http.get('/api/agents', () => {
          return HttpResponse.json(
            { detail: 'Internal server error' },
            { status: 500 }
          );
        })
      );

      const { result } = renderHook(() => useAgents(), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.isError).toBe(true);
      expect(result.current.agents).toHaveLength(0);
    });
  });

  describe('useAgent', () => {
    it('returns agent details from the API', async () => {
      const { result } = renderHook(() => useAgent('code-assistant'), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.agent).not.toBeNull();
      expect(result.current.agent?.name).toBe('Code Assistant');
      expect(result.current.agent?.skills).toHaveLength(2);
      expect(result.current.isError).toBe(false);
    });

    it('handles non-existent agent', async () => {
      const { result } = renderHook(() => useAgent('non-existent'), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.isError).toBe(true);
      expect(result.current.agent).toBeNull();
    });
  });

  describe('useRegisterAgent', () => {
    it('registers a new agent', async () => {
      const { result } = renderHook(() => useRegisterAgent(), {
        wrapper: createWrapper(),
      });

      expect(result.current.isRegistering).toBe(false);

      let newAgent: A2AAgent | undefined;
      await act(async () => {
        newAgent = await result.current.registerAgent({
          name: 'New Agent',
          path: 'new-agent',
          description: 'A new test agent',
          skills: ['testing'],
          tags: ['test'],
          visibility: 'public',
        });
      });

      expect(newAgent).toBeDefined();
      expect(newAgent?.name).toBe('New Agent');
      expect(newAgent?.path).toBe('new-agent');
    });
  });

  describe('useUpdateAgent', () => {
    it('updates an existing agent', async () => {
      const { result } = renderHook(() => useUpdateAgent(), {
        wrapper: createWrapper(),
      });

      let updatedAgent: A2AAgent | undefined;
      await act(async () => {
        updatedAgent = await result.current.updateAgent('code-assistant', {
          name: 'Updated Code Assistant',
        });
      });

      expect(updatedAgent).toBeDefined();
      expect(updatedAgent?.name).toBe('Updated Code Assistant');
    });
  });

  describe('useDeleteAgent', () => {
    it('deletes an agent', async () => {
      const { result } = renderHook(() => useDeleteAgent(), {
        wrapper: createWrapper(),
      });

      await act(async () => {
        await result.current.deleteAgent('code-assistant');
      });

      expect(result.current.isDeleting).toBe(false);
      expect(result.current.error).toBeNull();
    });
  });

  describe('useToggleAgent', () => {
    it('toggles agent enabled state', async () => {
      const { result } = renderHook(() => useToggleAgent(), {
        wrapper: createWrapper(),
      });

      let toggled: A2AAgent | undefined;
      await act(async () => {
        toggled = await result.current.toggleAgent('code-assistant', false);
      });

      expect(toggled).toBeDefined();
      expect(toggled?.is_enabled).toBe(false);
    });
  });

  describe('filterAgents', () => {
    const agents = mockA2AAgents;

    it('filters by search term (name)', () => {
      const filter: AgentFilter = { search: 'code', status: 'all', healthStatus: 'all', visibility: 'all' };
      const result = filterAgents(agents, filter);

      expect(result).toHaveLength(1);
      expect(result[0].name).toBe('Code Assistant');
    });

    it('filters by search term (skill)', () => {
      const filter: AgentFilter = { search: 'code-review', status: 'all', healthStatus: 'all', visibility: 'all' };
      const result = filterAgents(agents, filter);

      expect(result).toHaveLength(1);
      expect(result[0].path).toBe('code-assistant');
    });

    it('filters by search term (tag)', () => {
      const filter: AgentFilter = { search: 'data', status: 'all', healthStatus: 'all', visibility: 'all' };
      const result = filterAgents(agents, filter);

      expect(result).toHaveLength(1);
      expect(result[0].name).toBe('Data Analyst');
    });

    it('filters by enabled status', () => {
      const filter: AgentFilter = { search: '', status: 'enabled', healthStatus: 'all', visibility: 'all' };
      const result = filterAgents(agents, filter);

      expect(result).toHaveLength(1);
      expect(result.every(a => a.is_enabled)).toBe(true);
    });

    it('filters by disabled status', () => {
      const filter: AgentFilter = { search: '', status: 'disabled', healthStatus: 'all', visibility: 'all' };
      const result = filterAgents(agents, filter);

      expect(result).toHaveLength(1);
      expect(result[0].is_enabled).toBe(false);
    });

    it('filters by health status', () => {
      const filter: AgentFilter = { search: '', status: 'all', healthStatus: 'healthy', visibility: 'all' };
      const result = filterAgents(agents, filter);

      expect(result).toHaveLength(1);
      expect(result.every(a => a.health_status === 'healthy')).toBe(true);
    });

    it('filters by visibility', () => {
      const filter: AgentFilter = { search: '', status: 'all', healthStatus: 'all', visibility: 'public' };
      const result = filterAgents(agents, filter);

      expect(result).toHaveLength(1);
      expect(result[0].visibility).toBe('public');
    });

    it('combines multiple filters', () => {
      const filter: AgentFilter = { search: 'assistant', status: 'enabled', healthStatus: 'healthy', visibility: 'public' };
      const result = filterAgents(agents, filter);

      expect(result).toHaveLength(1);
      expect(result[0].name).toBe('Code Assistant');
    });

    it('returns all agents with no filters', () => {
      const filter: AgentFilter = { search: '', status: 'all', healthStatus: 'all', visibility: 'all' };
      const result = filterAgents(agents, filter);

      expect(result).toHaveLength(2);
    });
  });
});
