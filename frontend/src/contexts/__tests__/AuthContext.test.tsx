import { describe, it, expect, vi, beforeEach } from 'vitest';
import { renderHook, waitFor, act } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { server } from '../../test/mocks/server';
import { AuthProvider, useAuth } from '../AuthContext';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';

function createWrapper() {
  const queryClient = new QueryClient({
    defaultOptions: {
      queries: { retry: false },
      mutations: { retry: false },
    },
  });

  return function Wrapper({ children }: { children: React.ReactNode }) {
    return (
      <QueryClientProvider client={queryClient}>
        <AuthProvider>{children}</AuthProvider>
      </QueryClientProvider>
    );
  };
}

describe('AuthContext', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  it('provides user data when authenticated', async () => {
    const { result } = renderHook(() => useAuth(), {
      wrapper: createWrapper(),
    });

    // Initially loading
    expect(result.current.loading).toBe(true);

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.user).toMatchObject({
      user_id: 'test|user123',
      email: 'test@example.com',
      username: 'testuser',
      auth_method: 'password',
      is_admin: false,
    });
  });

  it('provides null user when not authenticated', async () => {
    server.use(
      http.get('/api/auth/me', () => {
        return HttpResponse.json(
          { detail: 'Not authenticated' },
          { status: 401 }
        );
      })
    );

    const { result } = renderHook(() => useAuth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    expect(result.current.user).toBeNull();
  });

  it('fetches OAuth providers', async () => {
    const { result } = renderHook(() => useAuth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.oauthProvidersLoading).toBe(false);
    });

    expect(result.current.oauthProviders).toEqual([
      { name: 'google', display_name: 'Google' },
      { name: 'github', display_name: 'GitHub' },
    ]);
  });

  it('handles OAuth login redirect', async () => {
    const originalLocation = window.location;
    const mockHref = vi.fn();

    // Mock window.location.href
    Object.defineProperty(window, 'location', {
      value: {
        ...originalLocation,
        href: '',
        origin: 'http://localhost:3000',
      },
      writable: true,
    });

    Object.defineProperty(window.location, 'href', {
      set: mockHref,
      get: () => '',
    });

    const { result } = renderHook(() => useAuth(), {
      wrapper: createWrapper(),
    });

    await waitFor(() => {
      expect(result.current.loading).toBe(false);
    });

    act(() => {
      result.current.loginWithOAuth('google');
    });

    expect(mockHref).toHaveBeenCalledWith(
      '/api/auth/auth/google?redirect_uri=' +
        encodeURIComponent('http://localhost:3000/auth/callback')
    );

    // Restore
    Object.defineProperty(window, 'location', {
      value: originalLocation,
      writable: true,
    });
  });

  it('throws error when useAuth is used outside AuthProvider', () => {
    // Suppress console.error for this test
    const consoleSpy = vi.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => {
      renderHook(() => useAuth());
    }).toThrow('useAuth must be used within an AuthProvider');

    consoleSpy.mockRestore();
  });
});
