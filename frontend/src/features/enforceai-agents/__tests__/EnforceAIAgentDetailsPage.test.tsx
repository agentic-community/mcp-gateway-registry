import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import { mockEnforceAIAgents } from '@/test/mocks/handlers';
import EnforceAIAgentDetailsPage from '../EnforceAIAgentDetailsPage';

// Mock useParams and useNavigate
const mockNavigate = vi.fn();
let mockAgentId = '9d2724e9-1753-4493-8993-0d6986754414';

vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useParams: () => ({ agentId: mockAgentId }),
    useNavigate: () => mockNavigate,
  };
});

describe('EnforceAIAgentDetailsPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();
    mockAgentId = '9d2724e9-1753-4493-8993-0d6986754414';
  });

  describe('Loading State', () => {
    it('shows loading spinner while fetching', async () => {
      server.use(
        http.get('/enforceai/agents/:agentId', async () => {
          await new Promise((resolve) => setTimeout(resolve, 100));
          return HttpResponse.json(mockEnforceAIAgents[0]);
        })
      );

      render(<EnforceAIAgentDetailsPage />);

      // Should show loading state
      expect(screen.getByText('Agent Details')).toBeInTheDocument();
    });
  });

  describe('Agent Info', () => {
    it('displays agent alias and ID', async () => {
      render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'sqlite-agent' })).toBeInTheDocument();
      });

      expect(screen.getByText('9d2724e9-1753-4493-8993-0d6986754414')).toBeInTheDocument();
    });

    it('displays agent scopes', async () => {
      render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'sqlite-agent' })).toBeInTheDocument();
      });

      expect(screen.getByText('sqlite.manage')).toBeInTheDocument();
      expect(screen.getByText('filesystem.read')).toBeInTheDocument();
    });

    it('displays allowed tools', async () => {
      render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'sqlite-agent' })).toBeInTheDocument();
      });

      expect(screen.getByText('read_query')).toBeInTheDocument();
      expect(screen.getByText('list_tables')).toBeInTheDocument();
    });

    it('displays status badges', async () => {
      render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'sqlite-agent' })).toBeInTheDocument();
      });

      const activeBadges = screen.getAllByText('Active');
      expect(activeBadges.length).toBeGreaterThan(0);
    });

    it('displays user ID', async () => {
      render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'sqlite-agent' })).toBeInTheDocument();
      });

      expect(screen.getByText('test|user123')).toBeInTheDocument();
    });
  });

  describe('Scopes Section', () => {
    it('displays scopes count', async () => {
      render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'sqlite-agent' })).toBeInTheDocument();
      });

      expect(screen.getByText('2 scopes')).toBeInTheDocument();
    });

    it('displays scope names', async () => {
      render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'sqlite-agent' })).toBeInTheDocument();
      });

      expect(screen.getByText('sqlite.manage')).toBeInTheDocument();
      expect(screen.getByText('filesystem.read')).toBeInTheDocument();
    });
  });

  describe('Agent Actions', () => {
    it('opens edit modal when clicking Edit', async () => {
      const { user } = render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'sqlite-agent' })).toBeInTheDocument();
      });

      const editButtons = screen.getAllByRole('button', { name: /edit/i });
      await user.click(editButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Edit EnforceAI Agent')).toBeInTheDocument();
      });
    });

    it('opens revoke modal when clicking Revoke', async () => {
      const { user } = render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'sqlite-agent' })).toBeInTheDocument();
      });

      const revokeButtons = screen.getAllByRole('button', { name: /^revoke$/i });
      await user.click(revokeButtons[0]);

      await waitFor(() => {
        expect(screen.getByText(/permanently revoke/i)).toBeInTheDocument();
      });
    });

    it('opens revoke all tokens modal when clicking Revoke All Tokens', async () => {
      const { user } = render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'sqlite-agent' })).toBeInTheDocument();
      });

      const revokeTokensButtons = screen.getAllByRole('button', { name: /revoke all tokens/i });
      await user.click(revokeTokensButtons[0]);

      await waitFor(() => {
        expect(screen.getByText(/What happens when you revoke all tokens/i)).toBeInTheDocument();
      });
    });
  });

  describe('Error Handling', () => {
    it('shows not found state for non-existent agent', async () => {
      mockAgentId = 'non-existent';

      render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('Agent not found')).toBeInTheDocument();
      });

      expect(screen.getByText(/doesn't exist or may have been deleted/i)).toBeInTheDocument();
    });

    it('shows back to agents button in error state', async () => {
      mockAgentId = 'non-existent';

      render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByText('Agent not found')).toBeInTheDocument();
      });

      expect(screen.getByRole('button', { name: /back to agents/i })).toBeInTheDocument();
    });
  });

  describe('Quick Info Sidebar', () => {
    it('displays status in sidebar', async () => {
      render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'sqlite-agent' })).toBeInTheDocument();
      });

      // Should have Quick Info section
      expect(screen.getByText('Quick Info')).toBeInTheDocument();
      expect(screen.getByText('Status')).toBeInTheDocument();
      // "Scopes" appears multiple times (Quick Info sidebar + Scopes card header)
      const scopesElements = screen.getAllByText('Scopes');
      expect(scopesElements.length).toBeGreaterThanOrEqual(1);
      expect(screen.getByText('Tools')).toBeInTheDocument();
    });

    it('displays created date', async () => {
      render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'sqlite-agent' })).toBeInTheDocument();
      });

      // "Created" appears multiple times (Quick Info sidebar + Agent Info card)
      const createdElements = screen.getAllByText('Created');
      expect(createdElements.length).toBeGreaterThanOrEqual(1);
    });
  });

  describe('Revoked Agent', () => {
    it('shows revoked warning for revoked agent', async () => {
      mockAgentId = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890';

      render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'fs-reader' })).toBeInTheDocument();
      });

      expect(screen.getByText('Agent Revoked')).toBeInTheDocument();
    });

    it('hides action buttons for revoked agent', async () => {
      mockAgentId = 'a1b2c3d4-e5f6-7890-abcd-ef1234567890';

      render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'fs-reader' })).toBeInTheDocument();
      });

      // Edit Agent button should not be in the Actions card (sidebar)
      const actionsCard = screen.queryByText('Actions');
      expect(actionsCard).not.toBeInTheDocument();
    });
  });

  describe('Related Links', () => {
    it('displays link to view API keys', async () => {
      render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'sqlite-agent' })).toBeInTheDocument();
      });

      expect(screen.getByText('Related')).toBeInTheDocument();
      expect(screen.getByText(/View API Keys/i)).toBeInTheDocument();
    });

    it('displays link to scope catalog', async () => {
      render(<EnforceAIAgentDetailsPage />);

      await waitFor(() => {
        expect(screen.getByRole('heading', { level: 1, name: 'sqlite-agent' })).toBeInTheDocument();
      });

      expect(screen.getByText(/View Scope Catalog/i)).toBeInTheDocument();
    });
  });
});
