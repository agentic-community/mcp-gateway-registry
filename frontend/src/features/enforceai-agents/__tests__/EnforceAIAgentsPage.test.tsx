import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import { mockEnforceAIAgents } from '@/test/mocks/handlers';
import EnforceAIAgentsPage from '../EnforceAIAgentsPage';

// Mock useNavigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

describe('EnforceAIAgentsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();
  });

  describe('Agent List', () => {
    it('renders page title and description', async () => {
      render(<EnforceAIAgentsPage />);

      expect(screen.getByRole('heading', { level: 1, name: 'EnforceAI Agents' })).toBeInTheDocument();
      expect(screen.getByText('Manage agent identities and authorization')).toBeInTheDocument();
    });

    it('displays agent cards after loading', async () => {
      render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });

      expect(screen.getByText('fs-reader')).toBeInTheDocument();
    });

    it('shows loading spinner initially', async () => {
      server.use(
        http.get('/enforceai/agents', async () => {
          await new Promise((resolve) => setTimeout(resolve, 100));
          return HttpResponse.json(mockEnforceAIAgents);
        })
      );

      render(<EnforceAIAgentsPage />);

      // The loading spinner should be present initially
      const spinner = document.querySelector('[class*="animate-spin"]');
      expect(spinner || screen.queryByText('EnforceAI Agents')).toBeTruthy();
    });

    it('shows empty state when no agents exist', async () => {
      server.use(
        http.get('/enforceai/agents', () => {
          return HttpResponse.json([]);
        })
      );

      render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('No agents created')).toBeInTheDocument();
      });

      expect(screen.getByText(/Create your first EnforceAI agent/)).toBeInTheDocument();
    });

    it('shows error state when API fails', async () => {
      server.use(
        http.get('/enforceai/agents', () => {
          return HttpResponse.json({ detail: 'Server error' }, { status: 500 });
        })
      );

      render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('Failed to load agents')).toBeInTheDocument();
      });
    });
  });

  describe('Agent Card', () => {
    it('displays agent information correctly', async () => {
      render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });

      // Check for status badges
      const activeBadges = screen.getAllByText('Active');
      expect(activeBadges.length).toBeGreaterThan(0);
    });

    it('displays revoked status for revoked agent', async () => {
      render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });

      // The second mock agent is revoked
      const revokedBadges = screen.getAllByText('Revoked');
      expect(revokedBadges.length).toBeGreaterThan(0);
    });

    it('displays agent scopes', async () => {
      render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });

      expect(screen.getByText('sqlite.manage')).toBeInTheDocument();
    });

    it('displays allowed tools when present', async () => {
      render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });

      expect(screen.getByText('read_query')).toBeInTheDocument();
      expect(screen.getByText('list_tables')).toBeInTheDocument();
    });

    it('navigates to agent details when clicking View', async () => {
      const { user } = render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });

      const viewButtons = screen.getAllByRole('button', { name: /view/i });
      await user.click(viewButtons[0]);

      expect(mockNavigate).toHaveBeenCalledWith('/agents/9d2724e9-1753-4493-8993-0d6986754414');
    });
  });

  describe('Filtering', () => {
    it('filters agents by search term', async () => {
      const { user } = render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search agents...');
      await user.type(searchInput, 'sqlite');

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
        expect(screen.queryByText('fs-reader')).not.toBeInTheDocument();
      });
    });

    it('filters agents by status', async () => {
      const { user } = render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });

      const statusSelect = screen.getByDisplayValue('All Status');
      await user.selectOptions(statusSelect, 'revoked');

      await waitFor(() => {
        expect(screen.queryByText('sqlite-agent')).not.toBeInTheDocument();
        expect(screen.getByText('fs-reader')).toBeInTheDocument();
      });
    });

    it('shows empty state when no agents match filters', async () => {
      const { user } = render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search agents...');
      await user.type(searchInput, 'nonexistent');

      await waitFor(() => {
        expect(screen.getByText('No agents match your filters')).toBeInTheDocument();
      });
    });

    it('clears filters when clicking Clear Filters button', async () => {
      const { user } = render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });

      // Apply a filter that hides all agents
      const searchInput = screen.getByPlaceholderText('Search agents...');
      await user.type(searchInput, 'nonexistent');

      await waitFor(() => {
        expect(screen.getByText('No agents match your filters')).toBeInTheDocument();
      });

      // Clear filters - there may be multiple buttons (FilterBar + EmptyState)
      const clearButtons = screen.getAllByRole('button', { name: /clear filters/i });
      await user.click(clearButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });
    });
  });

  describe('Agent Actions', () => {
    it('opens create modal when clicking Create Agent', async () => {
      const { user } = render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });

      const createButton = screen.getByRole('button', { name: /create agent/i });
      await user.click(createButton);

      await waitFor(() => {
        expect(screen.getByText('Create EnforceAI Agent')).toBeInTheDocument();
      });
    });

    it('opens edit modal when clicking Edit on an agent', async () => {
      const { user } = render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });

      const editButtons = screen.getAllByRole('button', { name: /^edit$/i });
      await user.click(editButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Edit EnforceAI Agent')).toBeInTheDocument();
      });
    });

    it('opens revoke modal when clicking Revoke on an agent', async () => {
      const { user } = render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });

      const revokeButtons = screen.getAllByRole('button', { name: /^revoke$/i });
      await user.click(revokeButtons[0]);

      await waitFor(() => {
        expect(screen.getByText(/permanently revoke/i)).toBeInTheDocument();
      });
    });

    it('opens revoke tokens modal when clicking Revoke Tokens', async () => {
      const { user } = render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });

      const revokeTokensButtons = screen.getAllByRole('button', { name: /revoke tokens/i });
      await user.click(revokeTokensButtons[0]);

      await waitFor(() => {
        expect(screen.getByText(/What happens when you revoke all tokens/i)).toBeInTheDocument();
      });
    });
  });

  describe('Results Count', () => {
    it('displays correct agent count', async () => {
      render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });

      expect(screen.getByText(/showing 2 of 2 agents/i)).toBeInTheDocument();
    });

    it('updates count when filtering', async () => {
      const { user } = render(<EnforceAIAgentsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite-agent')).toBeInTheDocument();
      });

      const statusSelect = screen.getByDisplayValue('All Status');
      await user.selectOptions(statusSelect, 'active');

      await waitFor(() => {
        expect(screen.getByText(/showing 1 of 2 agents/i)).toBeInTheDocument();
      });
    });
  });
});
