import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook, waitFor } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactNode } from 'react';
import { server } from '@/test/mocks/server';
import type {
  EnforceAIAgent,
  ApiKeySummary,
  TokenRevocationRecord,
} from '@/api/types';
import {
  useAdminUsers,
  useAdminUser,
  useAdminUserAgents,
  useAdminUserAgentApiKeys,
  useAdminCreateAgent,
  useAdminRevokeAgent,
  useAdminRevokeAllTokens,
  useAdminRevokeApiKey,
  useAdminRevokeToken,
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
    return <QueryClientProvider client={queryClient}>{children}</QueryClientProvider>;
  };
}

describe('Admin Query Hooks', () => {
  beforeEach(() => {
    server.resetHandlers();
  });

  describe('useAdminUsers', () => {
    it('fetches users when query is provided', async () => {
      const { result } = renderHook(() => useAdminUsers('test'), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(true);

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.users.length).toBeGreaterThan(0);
      expect(result.current.users[0]).toHaveProperty('user_id');
      expect(result.current.users[0]).toHaveProperty('email');
    });

    it('does not fetch when query is empty', async () => {
      const { result } = renderHook(() => useAdminUsers(''), {
        wrapper: createWrapper(),
      });

      // Should not be loading when disabled
      expect(result.current.isLoading).toBe(false);
      expect(result.current.users).toEqual([]);
    });
  });

  describe('useAdminUser', () => {
    it('fetches a single user by ID', async () => {
      const { result } = renderHook(() => useAdminUser('test|user123'), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.user).toBeDefined();
      expect(result.current.user?.user_id).toBe('test|user123');
    });

    it('returns null for empty userId', async () => {
      const { result } = renderHook(() => useAdminUser(''), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(false);
      expect(result.current.user).toBeNull();
    });
  });

  describe('useAdminUserAgents', () => {
    it('fetches agents for a user', async () => {
      const { result } = renderHook(() => useAdminUserAgents('test|user123'), {
        wrapper: createWrapper(),
      });

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.agents.length).toBeGreaterThan(0);
      expect(result.current.agents[0]).toHaveProperty('agent_id');
      expect(result.current.agents[0]).toHaveProperty('scopes');
    });
  });

  describe('useAdminUserAgentApiKeys', () => {
    it('fetches API keys for a user agent', async () => {
      const { result } = renderHook(
        () => useAdminUserAgentApiKeys('test|user123', '9d2724e9-1753-4493-8993-0d6986754414'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.apiKeys).toBeDefined();
      expect(Array.isArray(result.current.apiKeys)).toBe(true);
    });

    it('does not fetch when agentId is empty', async () => {
      const { result } = renderHook(() => useAdminUserAgentApiKeys('test|user123', ''), {
        wrapper: createWrapper(),
      });

      expect(result.current.isLoading).toBe(false);
      expect(result.current.apiKeys).toEqual([]);
    });
  });
});

describe('Admin Mutation Hooks', () => {
  beforeEach(() => {
    server.resetHandlers();
  });

  describe('useAdminCreateAgent', () => {
    it('creates an agent for a user', async () => {
      const { result } = renderHook(() => useAdminCreateAgent('test|user123'), {
        wrapper: createWrapper(),
      });

      let createdAgent: EnforceAIAgent | undefined;
      await waitFor(async () => {
        createdAgent = await result.current.createAgent({
          scopes: ['sqlite.manage'],
          alias: 'test-agent',
        });
      });

      expect(createdAgent).toBeDefined();
      expect(createdAgent?.scopes).toContain('sqlite.manage');
    });
  });

  describe('useAdminRevokeAgent', () => {
    it('revokes an agent for a user', async () => {
      const { result } = renderHook(() => useAdminRevokeAgent('test|user123'), {
        wrapper: createWrapper(),
      });

      let revokedAgent: EnforceAIAgent | undefined;
      await waitFor(async () => {
        revokedAgent = await result.current.revokeAgent(
          '9d2724e9-1753-4493-8993-0d6986754414'
        );
      });

      expect(revokedAgent).toBeDefined();
      expect(revokedAgent?.revoked_at).toBeDefined();
    });
  });

  describe('useAdminRevokeAllTokens', () => {
    it('revokes all tokens for a user agent', async () => {
      const { result } = renderHook(() => useAdminRevokeAllTokens('test|user123'), {
        wrapper: createWrapper(),
      });

      let updatedAgent: EnforceAIAgent | undefined;
      await waitFor(async () => {
        updatedAgent = await result.current.revokeAllTokens(
          '9d2724e9-1753-4493-8993-0d6986754414'
        );
      });

      expect(updatedAgent).toBeDefined();
      expect(updatedAgent?.tokens_valid_after).toBeDefined();
    });
  });

  describe('useAdminRevokeApiKey', () => {
    it('revokes an API key for a user', async () => {
      const { result } = renderHook(() => useAdminRevokeApiKey('test|user123'), {
        wrapper: createWrapper(),
      });

      let revokedKey: ApiKeySummary | undefined;
      await waitFor(async () => {
        revokedKey = await result.current.revokeApiKey('eak_abc123');
      });

      expect(revokedKey).toBeDefined();
      expect(revokedKey?.revoked_at).toBeDefined();
    });
  });

  describe('useAdminRevokeToken', () => {
    it('revokes a token for a user', async () => {
      const { result } = renderHook(() => useAdminRevokeToken('test|user123'), {
        wrapper: createWrapper(),
      });

      let revocation: TokenRevocationRecord | undefined;
      await waitFor(async () => {
        revocation = await result.current.revokeToken({
          agent_id: '9d2724e9-1753-4493-8993-0d6986754414',
          jti: 'test-jti-123',
          reason: 'Security concern',
        });
      });

      expect(revocation).toBeDefined();
      expect(revocation?.jti).toBe('test-jti-123');
      expect(revocation?.revoked_at).toBeDefined();
    });
  });
});
