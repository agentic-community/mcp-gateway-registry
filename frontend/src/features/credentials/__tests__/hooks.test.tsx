import { describe, it, expect, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { ReactNode } from 'react';
import { http, HttpResponse } from 'msw';
import { server } from '@/test/mocks/server';
import { mockApiKeys } from '@/test/mocks/handlers';
import type { CreateApiKeyResponse, MintTokenResponse, TokenRevocationRecord } from '@/api/types';
import {
  useApiKeys,
  useCreateApiKey,
  useRevokeApiKey,
  useMintToken,
  useRevokeToken,
  decodeToken,
  isTokenExpired,
  getTokenExpiration,
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

describe('Credentials Hooks', () => {
  beforeEach(() => {
    server.resetHandlers();
  });

  describe('useApiKeys', () => {
    it('returns API keys for an agent', async () => {
      const { result } = renderHook(
        () => useApiKeys('9d2724e9-1753-4493-8993-0d6986754414'),
        { wrapper: createWrapper() }
      );

      expect(result.current.isLoading).toBe(true);

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.apiKeys).toHaveLength(2);
      expect(result.current.apiKeys[0].key_id).toBe('eak_abc123');
      expect(result.current.isError).toBe(false);
    });

    it('returns empty array for agent with no keys', async () => {
      const { result } = renderHook(
        () => useApiKeys('non-existent-agent'),
        { wrapper: createWrapper() }
      );

      await waitFor(() => {
        expect(result.current.isLoading).toBe(false);
      });

      expect(result.current.apiKeys).toHaveLength(0);
    });

    it('does not fetch when agentId is empty', async () => {
      const { result } = renderHook(
        () => useApiKeys(''),
        { wrapper: createWrapper() }
      );

      // Should not start loading when disabled
      expect(result.current.isLoading).toBe(false);
      expect(result.current.apiKeys).toHaveLength(0);
    });
  });

  describe('useCreateApiKey', () => {
    it('creates a new API key', async () => {
      const { result } = renderHook(() => useCreateApiKey(), {
        wrapper: createWrapper(),
      });

      expect(result.current.isCreating).toBe(false);

      let response: CreateApiKeyResponse | undefined;
      await act(async () => {
        response = await result.current.createApiKey(
          '9d2724e9-1753-4493-8993-0d6986754414',
          { scopes: ['sqlite.manage'] }
        );
      });

      expect(response).toBeDefined();
      expect(response?.key_id).toContain('eak_');
      expect(response?.api_key_value).toContain('.');
    });
  });

  describe('useRevokeApiKey', () => {
    it('revokes an API key', async () => {
      const { result } = renderHook(() => useRevokeApiKey(), {
        wrapper: createWrapper(),
      });

      await act(async () => {
        await result.current.revokeApiKey(
          'eak_abc123',
          '9d2724e9-1753-4493-8993-0d6986754414'
        );
      });

      // Should complete without error
      expect(result.current.isRevoking).toBe(false);
      expect(result.current.error).toBeNull();
    });
  });

  describe('useMintToken', () => {
    it('mints a new gateway token', async () => {
      const { result } = renderHook(() => useMintToken(), {
        wrapper: createWrapper(),
      });

      expect(result.current.isMinting).toBe(false);

      let response: MintTokenResponse | undefined;
      await act(async () => {
        response = await result.current.mintToken(
          '9d2724e9-1753-4493-8993-0d6986754414',
          { scopes: ['sqlite.manage'], ttl_seconds: 3600 }
        );
      });

      expect(response).toBeDefined();
      expect(response?.token).toContain('.');
      expect(response?.token.split('.').length).toBe(3);
    });
  });

  describe('useRevokeToken', () => {
    it('revokes a token by JTI', async () => {
      const { result } = renderHook(() => useRevokeToken(), {
        wrapper: createWrapper(),
      });

      let response: TokenRevocationRecord | undefined;
      await act(async () => {
        response = await result.current.revokeToken({
          agent_id: '9d2724e9-1753-4493-8993-0d6986754414',
          jti: 'test-jti',
          reason: 'Test revocation',
        });
      });

      expect(response).toBeDefined();
      expect(response?.jti).toBe('test-jti');
      expect(response?.reason).toBe('Test revocation');
      expect(response?.revoked_at).toBeDefined();
    });

    it('revokes a token by pasting the token', async () => {
      // First mint a token to revoke
      const mintHook = renderHook(() => useMintToken(), {
        wrapper: createWrapper(),
      });

      let mintedToken: MintTokenResponse | undefined;
      await act(async () => {
        mintedToken = await mintHook.result.current.mintToken(
          '9d2724e9-1753-4493-8993-0d6986754414',
          { scopes: ['sqlite.manage'], ttl_seconds: 3600 }
        );
      });

      // Now revoke it
      const { result } = renderHook(() => useRevokeToken(), {
        wrapper: createWrapper(),
      });

      let response: TokenRevocationRecord | undefined;
      await act(async () => {
        response = await result.current.revokeToken({
          gateway_token: mintedToken!.token,
        });
      });

      expect(response).toBeDefined();
      expect(response?.jti).toBeDefined();
    });
  });

  describe('decodeToken', () => {
    it('decodes a valid JWT token', () => {
      // Create a mock JWT
      const header = btoa(JSON.stringify({ alg: 'RS256', typ: 'JWT', kid: 'test-kid' }));
      const payload = btoa(JSON.stringify({
        iss: 'enforceai-gateway',
        sub: 'test-agent',
        aud: ['mcp-gateway'],
        exp: Math.floor(Date.now() / 1000) + 3600,
        iat: Math.floor(Date.now() / 1000),
        jti: 'test-jti',
        type: 'agent',
        scopes: ['test.scope'],
      }));
      const token = `${header}.${payload}.signature`;

      const decoded = decodeToken(token);

      expect(decoded).not.toBeNull();
      expect(decoded?.header.alg).toBe('RS256');
      expect(decoded?.header.kid).toBe('test-kid');
      expect(decoded?.payload.iss).toBe('enforceai-gateway');
      expect(decoded?.payload.jti).toBe('test-jti');
      expect(decoded?.payload.scopes).toContain('test.scope');
    });

    it('returns null for invalid token', () => {
      const decoded = decodeToken('not-a-valid-token');
      expect(decoded).toBeNull();
    });

    it('returns null for token with wrong number of parts', () => {
      const decoded = decodeToken('only.two');
      expect(decoded).toBeNull();
    });
  });

  describe('isTokenExpired', () => {
    it('returns true for expired token', () => {
      const header = btoa(JSON.stringify({ alg: 'RS256', typ: 'JWT' }));
      const payload = btoa(JSON.stringify({
        iss: 'test',
        sub: 'test',
        aud: ['test'],
        exp: Math.floor(Date.now() / 1000) - 3600, // 1 hour ago
        iat: Math.floor(Date.now() / 1000) - 7200,
        jti: 'test',
        type: 'agent',
        scopes: [],
      }));
      const token = `${header}.${payload}.signature`;

      expect(isTokenExpired(token)).toBe(true);
    });

    it('returns false for valid token', () => {
      const header = btoa(JSON.stringify({ alg: 'RS256', typ: 'JWT' }));
      const payload = btoa(JSON.stringify({
        iss: 'test',
        sub: 'test',
        aud: ['test'],
        exp: Math.floor(Date.now() / 1000) + 3600, // 1 hour from now
        iat: Math.floor(Date.now() / 1000),
        jti: 'test',
        type: 'agent',
        scopes: [],
      }));
      const token = `${header}.${payload}.signature`;

      expect(isTokenExpired(token)).toBe(false);
    });

    it('returns true for invalid token', () => {
      expect(isTokenExpired('invalid')).toBe(true);
    });
  });

  describe('getTokenExpiration', () => {
    it('returns expiration date for valid token', () => {
      const expTime = Math.floor(Date.now() / 1000) + 3600;
      const header = btoa(JSON.stringify({ alg: 'RS256', typ: 'JWT' }));
      const payload = btoa(JSON.stringify({
        iss: 'test',
        sub: 'test',
        aud: ['test'],
        exp: expTime,
        iat: Math.floor(Date.now() / 1000),
        jti: 'test',
        type: 'agent',
        scopes: [],
      }));
      const token = `${header}.${payload}.signature`;

      const expDate = getTokenExpiration(token);

      expect(expDate).not.toBeNull();
      expect(expDate?.getTime()).toBe(expTime * 1000);
    });

    it('returns null for invalid token', () => {
      expect(getTokenExpiration('invalid')).toBeNull();
    });
  });
});
