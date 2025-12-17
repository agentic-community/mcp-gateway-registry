import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import * as ReactRouter from 'react-router-dom';
import AdminUserDetailsPage from '../AdminUserDetailsPage';

// Mock useParams
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useParams: vi.fn(),
  };
});

describe('AdminUserDetailsPage', () => {
  const mockUser = {
    user_id: 'local|testuser',
    email: 'testuser@example.com',
    username: 'testuser',
    auth_method: 'password',
    role: 'user',
    last_seen_at: '2024-01-15T10:00:00Z',
    created_at: '2024-01-01T00:00:00Z',
  };

  const mockAgents = [
    {
      user_id: 'local|testuser',
      agent_id: 'agent-1',
      alias: 'Test Agent 1',
      scopes: ['scope1', 'scope2'],
      allowed_tools: ['tool1', 'tool2'],
      revoked_at: null,
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-01T00:00:00Z',
      tokens_valid_after: null,
      metadata: null,
    },
    {
      user_id: 'local|testuser',
      agent_id: 'agent-2',
      alias: null,
      scopes: ['scope3'],
      allowed_tools: [],
      revoked_at: '2024-01-10T00:00:00Z',
      created_at: '2024-01-01T00:00:00Z',
      updated_at: '2024-01-10T00:00:00Z',
      tokens_valid_after: null,
      metadata: null,
    },
  ];

  beforeEach(() => {
    vi.mocked(ReactRouter.useParams).mockReturnValue({ userId: 'local|testuser' });
    server.resetHandlers();

    server.use(
      http.get('/enforceai/admin/users/:userId', ({ params }) => {
        if (params.userId === 'local%7Ctestuser' || params.userId === 'local|testuser') {
          return HttpResponse.json(mockUser);
        }
        return HttpResponse.json({ detail: 'User not found' }, { status: 404 });
      }),
      http.get('/enforceai/admin/users/:userId/agents', () => {
        return HttpResponse.json(mockAgents);
      })
    );
  });

  describe('Page Rendering', () => {
    it('renders page header with user info', async () => {
      render(<AdminUserDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('User Details')).toBeInTheDocument();
        expect(screen.getByText(/Managing user: testuser@example.com/)).toBeInTheDocument();
      });
    });

    it('displays admin warning banner', async () => {
      render(<AdminUserDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('Admin Mode')).toBeInTheDocument();
      });
    });

    it('displays target user banner', async () => {
      render(<AdminUserDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText(/Acting on user: testuser@example.com/)).toBeInTheDocument();
      });
    });

    it('shows loading state initially', () => {
      render(<AdminUserDetailsPage />);

      expect(screen.getByText('Loading user information...')).toBeInTheDocument();
    });

    it('shows back button', async () => {
      render(<AdminUserDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('Back to Directory')).toBeInTheDocument();
      });
    });
  });

  describe('User Information Display', () => {
    it('displays user information card', async () => {
      render(<AdminUserDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('User Information')).toBeInTheDocument();
      });
    });

    it('displays all user fields', async () => {
      render(<AdminUserDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('testuser@example.com')).toBeInTheDocument();
        expect(screen.getByText('testuser')).toBeInTheDocument();
        expect(screen.getByText('local|testuser')).toBeInTheDocument();
        expect(screen.getByText('password')).toBeInTheDocument();
        expect(screen.getByText('user')).toBeInTheDocument();
      });
    });

    it('displays canonical user ID with copy button', async () => {
      render(<AdminUserDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('local|testuser')).toBeInTheDocument();
        // Copy button should be present
        const buttons = screen.getAllByRole('button');
        expect(buttons.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Agent Management', () => {
    it('displays agent management section', async () => {
      render(<AdminUserDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('EnforceAI Agents')).toBeInTheDocument();
      });
    });

    it('displays total agent count', async () => {
      render(<AdminUserDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('Total Agents')).toBeInTheDocument();
        expect(screen.getByText('2')).toBeInTheDocument();
      });
    });

    it('displays active and revoked counts', async () => {
      render(<AdminUserDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('Active Agents')).toBeInTheDocument();
        expect(screen.getByText('Revoked Agents')).toBeInTheDocument();
      });
    });

    it('shows create agent button', async () => {
      render(<AdminUserDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /Create Agent/i })).toBeInTheDocument();
      });
    });

    it('shows agent alias when available', async () => {
      render(<AdminUserDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('Test Agent 1')).toBeInTheDocument();
      });
    });
  });

  describe('Cross-User Operations', () => {
    it('shows revoke token section', async () => {
      render(<AdminUserDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('Revoke Gateway Token')).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Revoke Token by JTI/i })).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    it('shows error state when user not found', async () => {
      vi.mocked(ReactRouter.useParams).mockReturnValue({ userId: 'nonexistent' });

      server.use(
        http.get('/enforceai/admin/users/:userId', () => {
          return HttpResponse.json({ detail: 'User not found' }, { status: 404 });
        })
      );

      render(<AdminUserDetailsPage />);

      await waitFor(() => {
        // Multiple elements with "User not found" (header description and EmptyState title)
        const elements = screen.getAllByText('User not found');
        expect(elements.length).toBeGreaterThan(0);
        expect(
          screen.getByText(/The requested user could not be found/)
        ).toBeInTheDocument();
      });
    });

    it('shows back button on error', async () => {
      vi.mocked(ReactRouter.useParams).mockReturnValue({ userId: 'nonexistent' });

      server.use(
        http.get('/enforceai/admin/users/:userId', () => {
          return HttpResponse.json({ detail: 'User not found' }, { status: 404 });
        })
      );

      render(<AdminUserDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('Back to User Directory')).toBeInTheDocument();
      });
    });
  });
});
