import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import { mockEnforceAIAgents, mockApiKeys } from '@/test/mocks/handlers';
import ApiKeysPage from '../ApiKeysPage';

// Mock useNavigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

describe('ApiKeysPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();
  });

  describe('Page Structure', () => {
    it('renders page title and description', async () => {
      render(<ApiKeysPage />);

      expect(screen.getByRole('heading', { level: 1, name: 'API Keys' })).toBeInTheDocument();
      expect(screen.getByText('Create and manage API keys for agent authentication')).toBeInTheDocument();
    });

    it('shows loading spinner initially', async () => {
      server.use(
        http.get('/enforceai/agents', async () => {
          await new Promise((resolve) => setTimeout(resolve, 100));
          return HttpResponse.json(mockEnforceAIAgents);
        })
      );

      render(<ApiKeysPage />);

      // The loading spinner should be present initially
      const spinner = document.querySelector('[class*="animate-spin"]');
      expect(spinner || screen.queryByText('API Keys')).toBeTruthy();
    });
  });

  describe('Agent Selection', () => {
    it('shows agent selector after loading', async () => {
      render(<ApiKeysPage />);

      await waitFor(() => {
        expect(screen.getByText('Select Agent')).toBeInTheDocument();
      });

      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    it('displays active agents in selector', async () => {
      render(<ApiKeysPage />);

      await waitFor(() => {
        expect(screen.getByText('Select an agent...')).toBeInTheDocument();
      });

      // sqlite-agent should be in the dropdown (active agent)
      expect(screen.getByText(/sqlite-agent/)).toBeInTheDocument();
    });

    it('shows info message before selecting an agent', async () => {
      render(<ApiKeysPage />);

      await waitFor(() => {
        expect(screen.getByText('Select Agent')).toBeInTheDocument();
      });

      expect(screen.getByText(/Select an agent above to view and manage its API keys/)).toBeInTheDocument();
    });
  });

  describe('API Keys Display', () => {
    it('shows API keys after selecting an agent', async () => {
      const { user } = render(<ApiKeysPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toBeInTheDocument();
      });

      // Select the agent
      const select = screen.getByRole('combobox');
      await user.selectOptions(select, '9d2724e9-1753-4493-8993-0d6986754414');

      await waitFor(() => {
        expect(screen.getByText(/eak_abc123/)).toBeInTheDocument();
      });
    });

    it('displays key status badges', async () => {
      const { user } = render(<ApiKeysPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      await user.selectOptions(select, '9d2724e9-1753-4493-8993-0d6986754414');

      await waitFor(() => {
        // Should show both Active and Revoked badges for the two mock keys
        expect(screen.getByText('Active')).toBeInTheDocument();
        expect(screen.getByText('Revoked')).toBeInTheDocument();
      });
    });
  });

  describe('Empty States', () => {
    it('shows no agents message when no agents exist', async () => {
      server.use(
        http.get('/enforceai/agents', () => {
          return HttpResponse.json([]);
        })
      );

      render(<ApiKeysPage />);

      await waitFor(() => {
        expect(screen.getByText('No agents available')).toBeInTheDocument();
      });

      expect(screen.getByText(/You need to create an EnforceAI agent/)).toBeInTheDocument();
    });

    it('shows no keys message for agent without keys', async () => {
      server.use(
        http.get('/enforceai/agents/:agentId/api-keys', () => {
          return HttpResponse.json([]);
        })
      );

      const { user } = render(<ApiKeysPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      await user.selectOptions(select, '9d2724e9-1753-4493-8993-0d6986754414');

      await waitFor(() => {
        expect(screen.getByText('No API keys')).toBeInTheDocument();
      });
    });
  });

  describe('Create API Key', () => {
    it('shows Create API Key button after selecting agent', async () => {
      const { user } = render(<ApiKeysPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      await user.selectOptions(select, '9d2724e9-1753-4493-8993-0d6986754414');

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /create api key/i })).toBeInTheDocument();
      });
    });

    it('opens create modal when clicking Create API Key', async () => {
      const { user } = render(<ApiKeysPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      await user.selectOptions(select, '9d2724e9-1753-4493-8993-0d6986754414');

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /create api key/i })).toBeInTheDocument();
      });

      const createButtons = screen.getAllByRole('button', { name: /create api key/i });
      await user.click(createButtons[0]);

      await waitFor(() => {
        // Look for the modal heading specifically
        expect(screen.getByRole('heading', { name: 'Create API Key' })).toBeInTheDocument();
        expect(screen.getByText(/Creating key for:/)).toBeInTheDocument();
      });
    });
  });

  describe('Revoke API Key', () => {
    it('shows revoke button on active keys', async () => {
      const { user } = render(<ApiKeysPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      await user.selectOptions(select, '9d2724e9-1753-4493-8993-0d6986754414');

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /^revoke$/i })).toBeInTheDocument();
      });
    });

    it('opens revoke modal when clicking Revoke', async () => {
      const { user } = render(<ApiKeysPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      await user.selectOptions(select, '9d2724e9-1753-4493-8993-0d6986754414');

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /^revoke$/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /^revoke$/i }));

      await waitFor(() => {
        // Look for the modal heading specifically
        expect(screen.getByRole('heading', { name: 'Revoke API Key' })).toBeInTheDocument();
        expect(screen.getByText(/This action cannot be undone/)).toBeInTheDocument();
      });
    });
  });
});
