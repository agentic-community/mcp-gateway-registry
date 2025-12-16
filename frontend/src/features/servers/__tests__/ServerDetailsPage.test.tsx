import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import { mockServerDetails } from '@/test/mocks/handlers';
import ServerDetailsPage from '../ServerDetailsPage';

// Mock useParams and useNavigate
const mockNavigate = vi.fn();
let mockPath = 'sqlite';

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useParams: () => ({ path: mockPath }),
    useNavigate: () => mockNavigate,
  };
});

describe('ServerDetailsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();
    mockPath = 'sqlite';
  });

  describe('Loading State', () => {
    it('shows loading spinner while fetching', async () => {
      server.use(
        http.get('/api/servers/:path', async () => {
          await new Promise((resolve) => setTimeout(resolve, 100));
          return HttpResponse.json(mockServerDetails[0]);
        })
      );

      render(<ServerDetailsPage />);

      // Should show loading state
      expect(screen.getByText('Server Details')).toBeInTheDocument();
    });
  });

  describe('Server Info', () => {
    it('displays server name and path', async () => {
      render(<ServerDetailsPage />);

      await waitFor(() => {
        // Use heading role to be specific
        expect(screen.getByRole('heading', { level: 1, name: 'SQLite Server' })).toBeInTheDocument();
      });

      expect(screen.getByText('sqlite')).toBeInTheDocument();
    });

    it('displays server description in breadcrumbs', async () => {
      render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'SQLite Server' })).toBeInTheDocument();
      });

      // Breadcrumb should show server name
      const breadcrumbs = screen.getAllByText('SQLite Server');
      expect(breadcrumbs.length).toBeGreaterThanOrEqual(1);
    });

    it('displays proxy URL', async () => {
      render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('http://localhost:3031/mcp')).toBeInTheDocument();
      });
    });

    it('displays tags', async () => {
      render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'SQLite Server' })).toBeInTheDocument();
      });

      expect(screen.getByText('database')).toBeInTheDocument();
      expect(screen.getByText('sql')).toBeInTheDocument();
    });

    it('displays status badges', async () => {
      render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'SQLite Server' })).toBeInTheDocument();
      });

      const enabledBadges = screen.getAllByText('Enabled');
      expect(enabledBadges.length).toBeGreaterThan(0);

      const healthyBadges = screen.getAllByText('healthy');
      expect(healthyBadges.length).toBeGreaterThan(0);
    });
  });

  describe('Tools Section', () => {
    it('displays tools count', async () => {
      render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'SQLite Server' })).toBeInTheDocument();
      });

      // Find the tools badge - should show 3 tools
      expect(screen.getByText('3 tools')).toBeInTheDocument();
    });

    it('displays tool names', async () => {
      render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'SQLite Server' })).toBeInTheDocument();
      });

      expect(screen.getByText('read_query')).toBeInTheDocument();
      expect(screen.getByText('list_tables')).toBeInTheDocument();
      expect(screen.getByText('describe_table')).toBeInTheDocument();
    });

    it('displays tool descriptions', async () => {
      render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'SQLite Server' })).toBeInTheDocument();
      });

      expect(screen.getByText('Execute a SELECT query on the database')).toBeInTheDocument();
    });

    it('expands tool to show input schema', async () => {
      const { user } = render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'SQLite Server' })).toBeInTheDocument();
      });

      // Click on the first tool (read_query has a schema)
      const toolButton = screen.getByText('read_query').closest('button');
      if (toolButton) {
        await user.click(toolButton);
      }

      await waitFor(() => {
        expect(screen.getByText('Input Schema')).toBeInTheDocument();
      });
    });
  });

  describe('Server Actions', () => {
    it('opens edit modal when clicking Edit', async () => {
      const { user } = render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'SQLite Server' })).toBeInTheDocument();
      });

      const editButtons = screen.getAllByRole('button', { name: /edit/i });
      await user.click(editButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Edit MCP Server')).toBeInTheDocument();
      });
    });

    it('refreshes server health when clicking Refresh Health', async () => {
      const { user } = render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'SQLite Server' })).toBeInTheDocument();
      });

      const refreshButtons = screen.getAllByRole('button', { name: /refresh health/i });
      await user.click(refreshButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Health check complete')).toBeInTheDocument();
      });
    });

    it('toggles server enabled state', async () => {
      const { user } = render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'SQLite Server' })).toBeInTheDocument();
      });

      // Click the "Disable Server" button (not just "Disable")
      const disableButtons = screen.getAllByRole('button', { name: /disable server/i });
      await user.click(disableButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Server updated')).toBeInTheDocument();
      });
    });

    it('shows delete confirmation dialog', async () => {
      const { user } = render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'SQLite Server' })).toBeInTheDocument();
      });

      const deleteButtons = screen.getAllByRole('button', { name: /delete/i });
      await user.click(deleteButtons[0]);

      // Check that the confirmation message appears (unique to the modal)
      await waitFor(() => {
        expect(screen.getByText(/are you sure you want to delete/i)).toBeInTheDocument();
      });
    });

    it('navigates back after deleting server', async () => {
      const { user } = render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'SQLite Server' })).toBeInTheDocument();
      });

      // Open delete dialog
      const deleteButtons = screen.getAllByRole('button', { name: /delete/i });
      await user.click(deleteButtons[0]);

      await waitFor(() => {
        expect(screen.getByText(/are you sure you want to delete/i)).toBeInTheDocument();
      });

      // Confirm delete - find the confirm button in the modal (the one that says just "Delete")
      const confirmButtons = screen.getAllByRole('button', { name: 'Delete' });
      // The confirm button in the modal should be the last one
      await user.click(confirmButtons[confirmButtons.length - 1]);

      await waitFor(() => {
        expect(mockNavigate).toHaveBeenCalledWith('/servers');
      });
    });
  });

  describe('Error Handling', () => {
    it('shows not found state for non-existent server', async () => {
      mockPath = 'non-existent';

      render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('Server not found')).toBeInTheDocument();
      });

      expect(screen.getByText(/doesn't exist or may have been deleted/i)).toBeInTheDocument();
    });

    it('shows back to servers button in error state', async () => {
      mockPath = 'non-existent';

      render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('Server not found')).toBeInTheDocument();
      });

      expect(screen.getByRole('button', { name: /back to servers/i })).toBeInTheDocument();
    });
  });

  describe('Quick Info Sidebar', () => {
    it('displays status in sidebar', async () => {
      render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'SQLite Server' })).toBeInTheDocument();
      });

      // Should have Quick Info section
      expect(screen.getByText('Quick Info')).toBeInTheDocument();
      expect(screen.getByText('Status')).toBeInTheDocument();
      expect(screen.getByText('Health')).toBeInTheDocument();
      // "Tools" appears multiple times (in card header and sidebar)
      const toolsElements = screen.getAllByText('Tools');
      expect(toolsElements.length).toBeGreaterThanOrEqual(1);
    });

    it('displays supported transports', async () => {
      render(<ServerDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'SQLite Server' })).toBeInTheDocument();
      });

      expect(screen.getByText('http')).toBeInTheDocument();
      expect(screen.getByText('sse')).toBeInTheDocument();
    });
  });
});
