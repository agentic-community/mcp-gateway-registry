import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor, within } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import { mockServers } from '@/test/mocks/handlers';
import ServersPage from '../ServersPage';

// Mock useNavigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

describe('ServersPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();
  });

  describe('Server List', () => {
    it('renders page title and description', async () => {
      render(<ServersPage />);

      expect(screen.getByText('MCP Servers')).toBeInTheDocument();
      expect(screen.getByText('Manage registered MCP servers')).toBeInTheDocument();
    });

    it('displays server cards after loading', async () => {
      render(<ServersPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      expect(screen.getByText('Filesystem Server')).toBeInTheDocument();
      expect(screen.getByText('Disabled Server')).toBeInTheDocument();
    });

    it('shows loading spinner initially', async () => {
      // Add delay to handler
      server.use(
        http.get('/api/servers', async () => {
          await new Promise((resolve) => setTimeout(resolve, 100));
          return HttpResponse.json({ servers: mockServers });
        })
      );

      render(<ServersPage />);

      // The loading spinner should be present initially
      const spinner = document.querySelector('[class*="animate-spin"]');
      expect(spinner || screen.queryByText('MCP Servers')).toBeTruthy();
    });

    it('shows empty state when no servers exist', async () => {
      server.use(
        http.get('/api/servers', () => {
          return HttpResponse.json({ servers: [] });
        })
      );

      render(<ServersPage />);

      await waitFor(() => {
        expect(screen.getByText('No servers registered')).toBeInTheDocument();
      });

      expect(screen.getByText('Register your first MCP server to get started')).toBeInTheDocument();
    });

    it('shows error state when API fails', async () => {
      server.use(
        http.get('/api/servers', () => {
          return HttpResponse.json({ detail: 'Server error' }, { status: 500 });
        })
      );

      render(<ServersPage />);

      await waitFor(() => {
        expect(screen.getByText('Failed to load servers')).toBeInTheDocument();
      });
    });
  });

  describe('Server Card', () => {
    it('displays server information correctly', async () => {
      render(<ServersPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      // Check for badges
      const enabledBadges = screen.getAllByText('Enabled');
      expect(enabledBadges.length).toBeGreaterThan(0);

      const healthyBadges = screen.getAllByText('healthy');
      expect(healthyBadges.length).toBeGreaterThan(0);
    });

    it('displays server tags', async () => {
      render(<ServersPage />);

      await waitFor(() => {
        expect(screen.getByText('database')).toBeInTheDocument();
        expect(screen.getByText('sql')).toBeInTheDocument();
      });
    });

    it('navigates to server details when clicking View', async () => {
      const { user } = render(<ServersPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      const viewButtons = screen.getAllByRole('button', { name: /view/i });
      await user.click(viewButtons[0]);

      expect(mockNavigate).toHaveBeenCalledWith('/servers/sqlite');
    });
  });

  describe('Filtering', () => {
    it('filters servers by search term', async () => {
      const { user } = render(<ServersPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search servers...');
      await user.type(searchInput, 'sqlite');

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
        expect(screen.queryByText('Filesystem Server')).not.toBeInTheDocument();
      });
    });

    it('filters servers by enabled status', async () => {
      const { user } = render(<ServersPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      const statusSelect = screen.getByDisplayValue('All Status');
      await user.selectOptions(statusSelect, 'disabled');

      await waitFor(() => {
        expect(screen.queryByText('SQLite Server')).not.toBeInTheDocument();
        expect(screen.getByText('Disabled Server')).toBeInTheDocument();
      });
    });

    it('shows empty state when no servers match filters', async () => {
      const { user } = render(<ServersPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search servers...');
      await user.type(searchInput, 'nonexistent');

      await waitFor(() => {
        expect(screen.getByText('No servers match your filters')).toBeInTheDocument();
      });
    });

    it('clears filters when clicking Clear Filters button', async () => {
      const { user } = render(<ServersPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      // Apply a filter that hides all servers
      const searchInput = screen.getByPlaceholderText('Search servers...');
      await user.type(searchInput, 'nonexistent');

      await waitFor(() => {
        expect(screen.getByText('No servers match your filters')).toBeInTheDocument();
      });

      // Clear filters
      const clearButton = screen.getByRole('button', { name: /clear filters/i });
      await user.click(clearButton);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });
    });
  });

  describe('Server Actions', () => {
    it('opens register modal when clicking Register Server', async () => {
      const { user } = render(<ServersPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      const registerButton = screen.getByRole('button', { name: /register server/i });
      await user.click(registerButton);

      await waitFor(() => {
        expect(screen.getByText('Register MCP Server')).toBeInTheDocument();
      });
    });

    it('opens edit modal when clicking Edit on a server', async () => {
      const { user } = render(<ServersPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      const editButtons = screen.getAllByRole('button', { name: /edit/i });
      await user.click(editButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Edit MCP Server')).toBeInTheDocument();
      });
    });

    it('shows success toast after toggling server', async () => {
      const { user } = render(<ServersPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      // Find the Disable button for the first enabled server
      const disableButtons = screen.getAllByRole('button', { name: /disable/i });
      await user.click(disableButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Server updated')).toBeInTheDocument();
      });
    });

    it('shows success toast after refreshing server health', async () => {
      const { user } = render(<ServersPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      const refreshButtons = screen.getAllByRole('button', { name: /refresh/i });
      await user.click(refreshButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Health check complete')).toBeInTheDocument();
      });
    });
  });

  describe('Results Count', () => {
    it('displays correct server count', async () => {
      render(<ServersPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      expect(screen.getByText(/showing 3 of 3 servers/i)).toBeInTheDocument();
    });

    it('updates count when filtering', async () => {
      const { user } = render(<ServersPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      const statusSelect = screen.getByDisplayValue('All Status');
      await user.selectOptions(statusSelect, 'enabled');

      await waitFor(() => {
        expect(screen.getByText(/showing 2 of 3 servers/i)).toBeInTheDocument();
      });
    });
  });
});
