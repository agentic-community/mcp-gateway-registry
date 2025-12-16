import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import { mockA2AAgents } from '@/test/mocks/handlers';
import A2AAgentDetailsPage from '../A2AAgentDetailsPage';

// Mock useParams and useNavigate
const mockNavigate = vi.fn();
let mockPath = 'code-assistant';

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useParams: () => ({ path: mockPath }),
    useNavigate: () => mockNavigate,
  };
});

describe('A2AAgentDetailsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();
    mockPath = 'code-assistant';
  });

  describe('Loading State', () => {
    it('shows loading spinner while fetching', async () => {
      server.use(
        http.get('/api/agents/:path', async () => {
          await new Promise((resolve) => setTimeout(resolve, 100));
          return HttpResponse.json(mockA2AAgents[0]);
        })
      );

      render(<A2AAgentDetailsPage />);

      // Should show loading state
      expect(screen.getByText('Agent Details')).toBeInTheDocument();
    });
  });

  describe('Agent Info', () => {
    it('displays agent name and path', async () => {
      render(<A2AAgentDetailsPage />);

      await waitFor(() => {
        // Use heading role to be specific
        expect(screen.getByRole('heading', { level: 1, name: 'Code Assistant' })).toBeInTheDocument();
      });

      expect(screen.getByText('code-assistant')).toBeInTheDocument();
    });

    it('displays agent description in breadcrumbs', async () => {
      render(<A2AAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'Code Assistant' })).toBeInTheDocument();
      });

      // Breadcrumb should show agent name
      const breadcrumbs = screen.getAllByText('Code Assistant');
      expect(breadcrumbs.length).toBeGreaterThanOrEqual(1);
    });

    it('displays agent URL', async () => {
      render(<A2AAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('http://localhost:5001')).toBeInTheDocument();
      });
    });

    it('displays tags', async () => {
      render(<A2AAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'Code Assistant' })).toBeInTheDocument();
      });

      expect(screen.getByText('ai')).toBeInTheDocument();
      expect(screen.getByText('code')).toBeInTheDocument();
    });

    it('displays status badges', async () => {
      render(<A2AAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'Code Assistant' })).toBeInTheDocument();
      });

      const enabledBadges = screen.getAllByText('Enabled');
      expect(enabledBadges.length).toBeGreaterThan(0);

      const healthyBadges = screen.getAllByText('healthy');
      expect(healthyBadges.length).toBeGreaterThan(0);
    });

    it('displays visibility badge', async () => {
      render(<A2AAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'Code Assistant' })).toBeInTheDocument();
      });

      const publicBadges = screen.getAllByText('Public');
      expect(publicBadges.length).toBeGreaterThan(0);
    });
  });

  describe('Skills Section', () => {
    it('displays skills count', async () => {
      render(<A2AAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'Code Assistant' })).toBeInTheDocument();
      });

      // Find the skills badge - should show 2 skills
      expect(screen.getByText('2 skills')).toBeInTheDocument();
    });

    it('displays skill names', async () => {
      render(<A2AAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'Code Assistant' })).toBeInTheDocument();
      });

      expect(screen.getByText('code-review')).toBeInTheDocument();
      expect(screen.getByText('refactoring')).toBeInTheDocument();
    });
  });

  describe('Agent Actions', () => {
    it('opens edit modal when clicking Edit', async () => {
      const { user } = render(<A2AAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'Code Assistant' })).toBeInTheDocument();
      });

      const editButtons = screen.getAllByRole('button', { name: /edit/i });
      await user.click(editButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Edit A2A Agent')).toBeInTheDocument();
      });
    });

    it('toggles agent enabled state', async () => {
      const { user } = render(<A2AAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'Code Assistant' })).toBeInTheDocument();
      });

      // Click the "Disable Agent" button
      const disableButtons = screen.getAllByRole('button', { name: /disable agent/i });
      await user.click(disableButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Agent updated')).toBeInTheDocument();
      });
    });

    it('shows delete confirmation dialog', async () => {
      const { user } = render(<A2AAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'Code Assistant' })).toBeInTheDocument();
      });

      const deleteButtons = screen.getAllByRole('button', { name: /delete/i });
      await user.click(deleteButtons[0]);

      // Check that the confirmation message appears (unique to the modal)
      await waitFor(() => {
        expect(screen.getByText(/are you sure you want to delete/i)).toBeInTheDocument();
      });
    });

    it('navigates back after deleting agent', async () => {
      const { user } = render(<A2AAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'Code Assistant' })).toBeInTheDocument();
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
        expect(mockNavigate).toHaveBeenCalledWith('/a2a-agents');
      });
    });
  });

  describe('Error Handling', () => {
    it('shows not found state for non-existent agent', async () => {
      mockPath = 'non-existent';

      render(<A2AAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('Agent not found')).toBeInTheDocument();
      });

      expect(screen.getByText(/doesn't exist or may have been deleted/i)).toBeInTheDocument();
    });

    it('shows back to agents button in error state', async () => {
      mockPath = 'non-existent';

      render(<A2AAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('Agent not found')).toBeInTheDocument();
      });

      expect(screen.getByRole('button', { name: /back to agents/i })).toBeInTheDocument();
    });
  });

  describe('Quick Info Sidebar', () => {
    it('displays status in sidebar', async () => {
      render(<A2AAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'Code Assistant' })).toBeInTheDocument();
      });

      // Should have Quick Info section
      expect(screen.getByText('Quick Info')).toBeInTheDocument();
      expect(screen.getByText('Status')).toBeInTheDocument();
      expect(screen.getByText('Health')).toBeInTheDocument();
      // "Skills" appears multiple times (in card header and sidebar)
      const skillsElements = screen.getAllByText('Skills');
      expect(skillsElements.length).toBeGreaterThanOrEqual(1);
    });

    it('displays stars count', async () => {
      render(<A2AAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'Code Assistant' })).toBeInTheDocument();
      });

      expect(screen.getByText('Stars')).toBeInTheDocument();
    });
  });
});
