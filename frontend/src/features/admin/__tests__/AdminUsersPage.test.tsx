import { describe, it, expect, beforeEach, vi } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import AdminUsersPage from '../AdminUsersPage';

describe('AdminUsersPage', () => {
  const mockUsers = [
    {
      user_id: 'local|user1',
      email: 'user1@example.com',
      username: 'user1',
      auth_method: 'password',
      last_seen_at: '2024-01-15T10:00:00Z',
    },
    {
      user_id: 'oidc|user2',
      email: 'user2@example.com',
      username: 'user2',
      auth_method: 'oidc',
      last_seen_at: '2024-01-14T10:00:00Z',
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();
  });

  describe('Page Rendering', () => {
    it('renders page header', () => {
      render(<AdminUsersPage />);

      expect(screen.getByText('User Directory')).toBeInTheDocument();
      expect(screen.getByText('Search for users and manage their accounts')).toBeInTheDocument();
    });

    it('displays admin warning banner', () => {
      render(<AdminUsersPage />);

      expect(screen.getByText('Admin Mode')).toBeInTheDocument();
    });

    it('renders search input', () => {
      render(<AdminUsersPage />);

      expect(screen.getByLabelText('Search Users')).toBeInTheDocument();
      expect(
        screen.getByPlaceholderText('Search by email or username...')
      ).toBeInTheDocument();
    });

    it('shows initial instructions when no search', () => {
      render(<AdminUsersPage />);

      expect(screen.getByText('Search for Users')).toBeInTheDocument();
      expect(
        screen.getByText(/Use the search box above to find users by email or username/)
      ).toBeInTheDocument();
    });
  });

  describe('Search Functionality', () => {
    it('triggers search when user types', async () => {
      server.use(
        http.get('/enforceai/admin/users', ({ request }) => {
          const url = new URL(request.url);
          const query = url.searchParams.get('query');
          if (query === 'user1') {
            return HttpResponse.json([mockUsers[0]]);
          }
          return HttpResponse.json([]);
        })
      );

      const { user } = render(<AdminUsersPage />);

      const searchInput = screen.getByPlaceholderText('Search by email or username...');
      await user.type(searchInput, 'user1');

      await waitFor(() => {
        expect(screen.getByText('Search Results')).toBeInTheDocument();
      });
    });

    it('displays search results', async () => {
      server.use(
        http.get('/enforceai/admin/users', () => {
          return HttpResponse.json(mockUsers);
        })
      );

      const { user } = render(<AdminUsersPage />);

      const searchInput = screen.getByPlaceholderText('Search by email or username...');
      await user.type(searchInput, 'user');

      await waitFor(() => {
        expect(screen.getByText('user1@example.com')).toBeInTheDocument();
        expect(screen.getByText('user2@example.com')).toBeInTheDocument();
      });
    });

    it('displays canonical user_id with copy button', async () => {
      server.use(
        http.get('/enforceai/admin/users', () => {
          return HttpResponse.json([mockUsers[0]]);
        })
      );

      const { user } = render(<AdminUsersPage />);

      const searchInput = screen.getByPlaceholderText('Search by email or username...');
      await user.type(searchInput, 'user1');

      await waitFor(() => {
        expect(screen.getByText('local|user1')).toBeInTheDocument();
      });

      // Copy button should be present
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });

    it('shows no results message when search returns empty', async () => {
      server.use(
        http.get('/enforceai/admin/users', () => {
          return HttpResponse.json([]);
        })
      );

      const { user } = render(<AdminUsersPage />);

      const searchInput = screen.getByPlaceholderText('Search by email or username...');
      await user.type(searchInput, 'nonexistent');

      await waitFor(() => {
        expect(screen.getByText('No users found')).toBeInTheDocument();
      });
    });

    it('shows error state when search fails', async () => {
      server.use(
        http.get('/enforceai/admin/users', () => {
          return HttpResponse.json({ detail: 'Search failed' }, { status: 500 });
        })
      );

      const { user } = render(<AdminUsersPage />);

      const searchInput = screen.getByPlaceholderText('Search by email or username...');
      await user.type(searchInput, 'user');

      await waitFor(() => {
        // Multiple elements with "Search failed" (title and description in EmptyState)
        const elements = screen.getAllByText('Search failed');
        expect(elements.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Result Display', () => {
    it('displays all user fields', async () => {
      server.use(
        http.get('/enforceai/admin/users', () => {
          return HttpResponse.json([mockUsers[0]]);
        })
      );

      const { user } = render(<AdminUsersPage />);

      const searchInput = screen.getByPlaceholderText('Search by email or username...');
      await user.type(searchInput, 'user1');

      await waitFor(() => {
        expect(screen.getByText('user1@example.com')).toBeInTheDocument();
        expect(screen.getByText('user1')).toBeInTheDocument();
        expect(screen.getByText('local|user1')).toBeInTheDocument();
        expect(screen.getByText('password')).toBeInTheDocument();
      });
    });

    it('renders result count', async () => {
      server.use(
        http.get('/enforceai/admin/users', () => {
          return HttpResponse.json(mockUsers);
        })
      );

      const { user } = render(<AdminUsersPage />);

      const searchInput = screen.getByPlaceholderText('Search by email or username...');
      await user.type(searchInput, 'user');

      await waitFor(() => {
        expect(screen.getByText('Found 2 users')).toBeInTheDocument();
      });
    });

    it('makes rows clickable', async () => {
      server.use(
        http.get('/enforceai/admin/users', () => {
          return HttpResponse.json([mockUsers[0]]);
        })
      );

      const { user } = render(<AdminUsersPage />);

      const searchInput = screen.getByPlaceholderText('Search by email or username...');
      await user.type(searchInput, 'user1');

      await waitFor(() => {
        const row = screen.getByText('user1@example.com').closest('tr');
        expect(row).toHaveClass('cursor-pointer');
      });
    });
  });
});
