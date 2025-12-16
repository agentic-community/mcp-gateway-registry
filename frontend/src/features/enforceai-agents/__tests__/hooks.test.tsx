import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactNode } from 'react';
import { http, HttpResponse } from 'msw';
import { server } from '@/test/mocks/server';
import { mockEnforceAIAgents } from '@/test/mocks/handlers';
import type { EnforceAIAgent } from '@/api/types';
import {
  useEnforceAIAgents,
  useEnforceAIAgent,
  useCreateEnforceAIAgent,
  useUpdateEnforceAIAgent,
  useRevokeEnforceAIAgent,
  useRevokeAllTokens,
  filterEnforceAIAgents,
  type EnforceAIAgentFilter,
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

describe('EnforceAI Agent Hooks', () => {
  beforeEach(() => {
    server.resetHandlers();
  });

  describe('useEnforceAIAgents', () => {
    it('returns agents from the API', async () => {
      const { result } = renderHook(() => useEnforceAIAgents(), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(true);

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.agents).toHaveLength(2);
      expect(result.current.agents[0].alias).toBe('sqlite-agent');
      expect(result.current.isError).toBe(false);
    });

    it('handles API errors', async () => {
      server.use(
        http.get('/enforceai/agents', () => {
          return HttpResponse.json(
            { detail: 'Internal server error' },
            { status: 500 }
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
      expect(result.current.agents).toHaveLength(0);
    });
  });

  describe('useEnforceAIAgent', () => {
    it('returns agent details from the API', async () => {
      const { result } = renderHook(
        () => useEnforceAIAgent('9d2724e9-1753-4493-8993-0d6986754414'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.agent).not.toBeNull();
      expect(result.current.agent?.alias).toBe('sqlite-agent');
      expect(result.current.agent?.scopes).toContain('sqlite.manage');
      expect(result.current.isError).toBe(false);
    });

    it('handles non-existent agent', async () => {
      const { result } = renderHook(
        () => useEnforceAIAgent('non-existent'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.isError).toBe(true);
      expect(result.current.agent).toBeNull();
    });
  });

  describe('useCreateEnforceAIAgent', () => {
    it('creates a new agent', async () => {
      const { result } = renderHook(() => useCreateEnforceAIAgent(), {
        wrapper: createWrapper(),
      });

      expect(result.current.isCreating).toBe(false);

      let newAgent: EnforceAIAgent | undefined;
      await act(async () => {
        newAgent = await result.current.createAgent({
          scopes: ['test.scope'],
          alias: 'test-agent',
        });
      });

      expect(newAgent).toBeDefined();
      expect(newAgent?.alias).toBe('test-agent');
      expect(newAgent?.scopes).toContain('test.scope');
    });
  });

  describe('useUpdateEnforceAIAgent', () => {
    it('updates an existing agent', async () => {
      const { result } = renderHook(() => useUpdateEnforceAIAgent(), {
        wrapper: createWrapper(),
      });

      let updatedAgent: EnforceAIAgent | undefined;
      await act(async () => {
        updatedAgent = await result.current.updateAgent(
          '9d2724e9-1753-4493-8993-0d6986754414',
          { alias: 'updated-alias' }
        );
      });

      expect(updatedAgent).toBeDefined();
      expect(updatedAgent?.alias).toBe('updated-alias');
    });
  });

  describe('useRevokeEnforceAIAgent', () => {
    it('revokes an agent', async () => {
      const { result } = renderHook(() => useRevokeEnforceAIAgent(), {
        wrapper: createWrapper(),
      });

      let revokedAgent: EnforceAIAgent | undefined;
      await act(async () => {
        revokedAgent = await result.current.revokeAgent(
          '9d2724e9-1753-4493-8993-0d6986754414'
        );
      });

      expect(revokedAgent).toBeDefined();
      expect(revokedAgent?.revoked_at).not.toBeNull();
    });
  });

  describe('useRevokeAllTokens', () => {
    it('revokes all tokens for an agent', async () => {
      const { result } = renderHook(() => useRevokeAllTokens(), {
        wrapper: createWrapper(),
      });

      let updatedAgent: EnforceAIAgent | undefined;
      await act(async () => {
        updatedAgent = await result.current.revokeAllTokens(
          '9d2724e9-1753-4493-8993-0d6986754414'
        );
      });

      expect(updatedAgent).toBeDefined();
      expect(updatedAgent?.tokens_valid_after).not.toBeNull();
    });
  });

  describe('filterEnforceAIAgents', () => {
    const agents = mockEnforceAIAgents;

    it('filters by search term (alias)', () => {
      const filter: EnforceAIAgentFilter = { search: 'sqlite', status: 'all' };
      const result = filterEnforceAIAgents(agents, filter);

      expect(result).toHaveLength(1);
      expect(result[0].alias).toBe('sqlite-agent');
    });

    it('filters by search term (agent_id)', () => {
      const filter: EnforceAIAgentFilter = { search: '9d2724', status: 'all' };
      const result = filterEnforceAIAgents(agents, filter);

      expect(result).toHaveLength(1);
      expect(result[0].alias).toBe('sqlite-agent');
    });

    it('filters by search term (scope)', () => {
      const filter: EnforceAIAgentFilter = { search: 'filesystem', status: 'all' };
      const result = filterEnforceAIAgents(agents, filter);

      expect(result).toHaveLength(2); // Both agents have filesystem scope
    });

    it('filters by active status', () => {
      const filter: EnforceAIAgentFilter = { search: '', status: 'active' };
      const result = filterEnforceAIAgents(agents, filter);

      expect(result).toHaveLength(1);
      expect(result.every((a) => a.revoked_at === null)).toBe(true);
    });

    it('filters by revoked status', () => {
      const filter: EnforceAIAgentFilter = { search: '', status: 'revoked' };
      const result = filterEnforceAIAgents(agents, filter);

      expect(result).toHaveLength(1);
      expect(result[0].revoked_at).not.toBeNull();
    });

    it('combines multiple filters', () => {
      const filter: EnforceAIAgentFilter = { search: 'fs', status: 'revoked' };
      const result = filterEnforceAIAgents(agents, filter);

      expect(result).toHaveLength(1);
      expect(result[0].alias).toBe('fs-reader');
    });

    it('returns all agents with no filters', () => {
      const filter: EnforceAIAgentFilter = { search: '', status: 'all' };
      const result = filterEnforceAIAgents(agents, filter);

      expect(result).toHaveLength(2);
    });
  });
});
