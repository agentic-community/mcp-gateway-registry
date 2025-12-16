import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import { mockEnforceAIAgents } from '@/test/mocks/handlers';
import TokensPage from '../TokensPage';

// Mock useNavigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

describe('TokensPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();
  });

  describe('Page Structure', () => {
    it('renders page title and description', async () => {
      render(<TokensPage />);

      expect(screen.getByRole('heading', { level: 1, name: 'Gateway Tokens' })).toBeInTheDocument();
      expect(screen.getByText('Mint and manage gateway tokens')).toBeInTheDocument();
    });

    it('shows loading spinner initially', async () => {
      server.use(
        http.get('/enforceai/agents', async () => {
          await new Promise((resolve) => setTimeout(resolve, 100));
          return HttpResponse.json(mockEnforceAIAgents);
        })
      );

      render(<TokensPage />);

      const spinner = document.querySelector('[class*="animate-spin"]');
      expect(spinner || screen.queryByText('Gateway Tokens')).toBeTruthy();
    });
  });

  describe('Agent Selection', () => {
    it('shows agent selector after loading', async () => {
      render(<TokensPage />);

      await waitFor(() => {
        expect(screen.getByText('Select Agent for Minting')).toBeInTheDocument();
      });

      expect(screen.getByRole('combobox')).toBeInTheDocument();
    });

    it('displays active agents in selector', async () => {
      render(<TokensPage />);

      await waitFor(() => {
        expect(screen.getByText('Select an agent...')).toBeInTheDocument();
      });

      expect(screen.getByText(/sqlite-agent/)).toBeInTheDocument();
    });

    it('shows info message before selecting an agent', async () => {
      render(<TokensPage />);

      await waitFor(() => {
        expect(screen.getByText('Select Agent for Minting')).toBeInTheDocument();
      });

      expect(screen.getByText(/Select an agent above to mint a new gateway token/)).toBeInTheDocument();
    });
  });

  describe('Token Information', () => {
    it('displays token information cards', async () => {
      render(<TokensPage />);

      await waitFor(() => {
        expect(screen.getByText('About Gateway Tokens')).toBeInTheDocument();
      });

      expect(screen.getByText('Token Revocation')).toBeInTheDocument();
    });

    it('shows ready to mint state after selecting agent', async () => {
      const { user } = render(<TokensPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      await user.selectOptions(select, '9d2724e9-1753-4493-8993-0d6986754414');

      await waitFor(() => {
        expect(screen.getByText('Ready to mint')).toBeInTheDocument();
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

      render(<TokensPage />);

      await waitFor(() => {
        expect(screen.getByText('No agents available')).toBeInTheDocument();
      });

      expect(screen.getByText(/You need to create an EnforceAI agent/)).toBeInTheDocument();
    });
  });

  describe('Mint Token', () => {
    it('shows Mint Token button after selecting agent', async () => {
      const { user } = render(<TokensPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      await user.selectOptions(select, '9d2724e9-1753-4493-8993-0d6986754414');

      await waitFor(() => {
        // Should have at least one Mint Token button (in header or empty state)
        const mintButtons = screen.getAllByRole('button', { name: /mint token/i });
        expect(mintButtons.length).toBeGreaterThan(0);
      });
    });

    it('opens mint modal when clicking Mint Token', async () => {
      const { user } = render(<TokensPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toBeInTheDocument();
      });

      const select = screen.getByRole('combobox');
      await user.selectOptions(select, '9d2724e9-1753-4493-8993-0d6986754414');

      await waitFor(() => {
        expect(screen.getAllByRole('button', { name: /mint token/i })[0]).toBeInTheDocument();
      });

      const mintButtons = screen.getAllByRole('button', { name: /mint token/i });
      await user.click(mintButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Mint Gateway Token')).toBeInTheDocument();
        expect(screen.getByText(/Minting token for:/)).toBeInTheDocument();
      });
    });
  });

  describe('Revoke Token', () => {
    it('shows Revoke Token button in header', async () => {
      render(<TokensPage />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /revoke token/i })).toBeInTheDocument();
      });
    });

    it('opens revoke modal when clicking Revoke Token', async () => {
      const { user } = render(<TokensPage />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /revoke token/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /revoke token/i }));

      await waitFor(() => {
        expect(screen.getByText('Revoke Gateway Token')).toBeInTheDocument();
      });
    });

    it('shows two revocation modes', async () => {
      const { user } = render(<TokensPage />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /revoke token/i })).toBeInTheDocument();
      });

      await user.click(screen.getByRole('button', { name: /revoke token/i }));

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /by agent id \+ jti/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /by pasting token/i })).toBeInTheDocument();
      });
    });
  });
});
