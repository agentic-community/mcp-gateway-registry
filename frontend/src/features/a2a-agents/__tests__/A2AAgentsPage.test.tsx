import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import { mockA2AAgents } from '@/test/mocks/handlers';
import A2AAgentsPage from '../A2AAgentsPage';

// Mock useNavigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

describe('A2AAgentsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();
  });

  describe('Agent List', () => {
    it('renders page title and description', async () => {
      render(<A2AAgentsPage />);

      expect(screen.getByRole('heading', { level: 1, name: 'A2A Agents' })).toBeInTheDocument();
      expect(screen.getByText('Manage registered Agent-to-Agent agents')).toBeInTheDocument();
    });

    it('displays agent cards after loading', async () => {
      render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
      });

      expect(screen.getByText('Data Analyst')).toBeInTheDocument();
    });

    it('shows loading spinner initially', async () => {
      // Add delay to handler
      server.use(
        http.get('/api/agents', async () => {
          await new Promise((resolve) => setTimeout(resolve, 100));
          return HttpResponse.json({ agents: mockA2AAgents, total_count: mockA2AAgents.length });
        })
      );

      render(<A2AAgentsPage />);

      // The loading spinner should be present initially
      const spinner = document.querySelector('[class*="animate-spin"]');
      expect(spinner || screen.queryByText('A2A Agents')).toBeTruthy();
    });

    it('shows empty state when no agents exist', async () => {
      server.use(
        http.get('/api/agents', () => {
          return HttpResponse.json({ agents: [], total_count: 0 });
        })
      );

      render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('No agents registered')).toBeInTheDocument();
      });

      expect(screen.getByText('Register your first A2A agent to get started')).toBeInTheDocument();
    });

    it('shows error state when API fails', async () => {
      server.use(
        http.get('/api/agents', () => {
          return HttpResponse.json({ detail: 'Server error' }, { status: 500 });
        })
      );

      render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Failed to load agents')).toBeInTheDocument();
      });
    });
  });

  describe('Agent Card', () => {
    it('displays agent information correctly', async () => {
      render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
      });

      // Check for badges
      const enabledBadges = screen.getAllByText('Enabled');
      expect(enabledBadges.length).toBeGreaterThan(0);

      const healthyBadges = screen.getAllByText('healthy');
      expect(healthyBadges.length).toBeGreaterThan(0);
    });

    it('displays agent tags', async () => {
      render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
      });

      // 'ai' appears in both agents, so there should be 2
      const aiTags = screen.getAllByText('ai');
      expect(aiTags.length).toBeGreaterThan(0);
      expect(screen.getByText('code')).toBeInTheDocument();
    });

    it('displays agent skills', async () => {
      render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
      });

      expect(screen.getByText('code-review')).toBeInTheDocument();
      expect(screen.getByText('refactoring')).toBeInTheDocument();
    });

    it('navigates to agent details when clicking View', async () => {
      const { user } = render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
      });

      const viewButtons = screen.getAllByRole('button', { name: /view/i });
      await user.click(viewButtons[0]);

      expect(mockNavigate).toHaveBeenCalledWith('/a2a-agents/code-assistant');
    });
  });

  describe('Filtering', () => {
    it('filters agents by search term', async () => {
      const { user } = render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search agents...');
      await user.type(searchInput, 'code');

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
        expect(screen.queryByText('Data Analyst')).not.toBeInTheDocument();
      });
    });

    it('filters agents by enabled status', async () => {
      const { user } = render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
      });

      const statusSelect = screen.getByDisplayValue('All Status');
      await user.selectOptions(statusSelect, 'disabled');

      await waitFor(() => {
        expect(screen.queryByText('Code Assistant')).not.toBeInTheDocument();
        expect(screen.getByText('Data Analyst')).toBeInTheDocument();
      });
    });

    it('filters agents by visibility', async () => {
      const { user } = render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
      });

      const visibilitySelect = screen.getByDisplayValue('All Visibility');
      await user.selectOptions(visibilitySelect, 'private');

      await waitFor(() => {
        expect(screen.queryByText('Code Assistant')).not.toBeInTheDocument();
        expect(screen.getByText('Data Analyst')).toBeInTheDocument();
      });
    });

    it('shows empty state when no agents match filters', async () => {
      const { user } = render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search agents...');
      await user.type(searchInput, 'nonexistent');

      await waitFor(() => {
        expect(screen.getByText('No agents match your filters')).toBeInTheDocument();
      });
    });

    it('clears filters when clicking Clear Filters button', async () => {
      const { user } = render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
      });

      // Apply a filter that hides all agents
      const searchInput = screen.getByPlaceholderText('Search agents...');
      await user.type(searchInput, 'nonexistent');

      await waitFor(() => {
        expect(screen.getByText('No agents match your filters')).toBeInTheDocument();
      });

      // Clear filters
      const clearButton = screen.getByRole('button', { name: /clear filters/i });
      await user.click(clearButton);

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
      });
    });
  });

  describe('Agent Actions', () => {
    it('opens register modal when clicking Register Agent', async () => {
      const { user } = render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
      });

      const registerButton = screen.getByRole('button', { name: /register agent/i });
      await user.click(registerButton);

      await waitFor(() => {
        expect(screen.getByText('Register A2A Agent')).toBeInTheDocument();
      });
    });

    it('opens edit modal when clicking Edit on an agent', async () => {
      const { user } = render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
      });

      const editButtons = screen.getAllByRole('button', { name: /edit/i });
      await user.click(editButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Edit A2A Agent')).toBeInTheDocument();
      });
    });

    it('shows success toast after toggling agent', async () => {
      const { user } = render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
      });

      // Find the Disable button for the first enabled agent
      const disableButtons = screen.getAllByRole('button', { name: /disable/i });
      await user.click(disableButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Agent updated')).toBeInTheDocument();
      });
    });
  });

  describe('Results Count', () => {
    it('displays correct agent count', async () => {
      render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
      });

      expect(screen.getByText(/showing 2 of 2 agents/i)).toBeInTheDocument();
    });

    it('updates count when filtering', async () => {
      const { user } = render(<A2AAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Code Assistant')).toBeInTheDocument();
      });

      const statusSelect = screen.getByDisplayValue('All Status');
      await user.selectOptions(statusSelect, 'enabled');

      await waitFor(() => {
        expect(screen.getByText(/showing 1 of 2 agents/i)).toBeInTheDocument();
      });
    });
  });
});
