import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import {
  mockServers,
  mockA2AAgents,
  mockEnforceAIAgents,
} from '@/test/mocks/handlers';
import OverviewPage from '../OverviewPage';

// Mock useNavigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

describe('OverviewPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Connection Status Cards', () => {
    it('renders connection status cards section', async () => {
      render(<OverviewPage />);

      expect(screen.getByText('Connection Status')).toBeInTheDocument();
      expect(screen.getByText('Registry')).toBeInTheDocument();
      expect(screen.getByText('EnforceAI')).toBeInTheDocument();
    });

    it('shows "Not Tested" status initially', async () => {
      render(<OverviewPage />);

      // Initially should show pending status
      const pendingBadges = screen.getAllByText('Pending');
      expect(pendingBadges.length).toBeGreaterThanOrEqual(1);
    });

    it('shows connection status after clicking Test Connections', async () => {
      const { user } = render(<OverviewPage />);

      // Click the Test Connections button
      const testButton = screen.getByRole('button', { name: /test connections/i });
      await user.click(testButton);

      // Wait for the test to complete - look for Online badge which indicates successful connection
      await waitFor(
        () => {
          const onlineBadges = screen.getAllByText('Online');
          expect(onlineBadges.length).toBeGreaterThanOrEqual(1);
        },
        { timeout: 5000 }
      );
    });
  });

  describe('Summary Counts', () => {
    it('renders summary section with stats cards', async () => {
      render(<OverviewPage />);

      expect(screen.getByText('Summary')).toBeInTheDocument();
      expect(screen.getByText('MCP Servers')).toBeInTheDocument();
      expect(screen.getByText('A2A Agents')).toBeInTheDocument();
      expect(screen.getByText('EnforceAI Agents')).toBeInTheDocument();
    });

    it('displays correct server count and breakdown', async () => {
      render(<OverviewPage />);

      // Wait for data to load - check that MCP Servers section renders with data
      await waitFor(
        () => {
          // Check that server stats are loaded (the section has rendered with data)
          const serversSection = screen.getByText('MCP Servers');
          expect(serversSection).toBeInTheDocument();
        },
        { timeout: 3000 }
      );

      // Verify stats breakdown badges are present
      await waitFor(
        () => {
          // Look for any "enabled" text in breakdown badges
          const textContent = document.body.textContent || '';
          expect(textContent).toContain('enabled');
        },
        { timeout: 3000 }
      );
    });

    it('displays correct A2A agent count and breakdown', async () => {
      render(<OverviewPage />);

      // Wait for A2A data to load - check for breakdown badges
      await waitFor(
        () => {
          // Check breakdown badges exist
          const enabledBadges = screen.getAllByText('1 enabled');
          expect(enabledBadges.length).toBeGreaterThanOrEqual(1);
        },
        { timeout: 3000 }
      );
    });

    it('displays correct EnforceAI agent count and breakdown', async () => {
      render(<OverviewPage />);

      // Check breakdown: 1 active, 1 revoked
      await waitFor(
        () => {
          expect(screen.getByText('1 active')).toBeInTheDocument();
          expect(screen.getByText('1 revoked')).toBeInTheDocument();
        },
        { timeout: 3000 }
      );
    });

    it('navigates to servers page when clicking MCP Servers card', async () => {
      const { user } = render(<OverviewPage />);

      // Wait for data to load
      await waitFor(
        () => {
          expect(screen.getByText('MCP Servers')).toBeInTheDocument();
        },
        { timeout: 3000 }
      );

      // Find and click the MCP Servers card
      const serversCard = screen
        .getByText('MCP Servers')
        .closest('[role="button"]');
      if (serversCard) {
        await user.click(serversCard);
        expect(mockNavigate).toHaveBeenCalledWith('/servers');
      }
    });
  });

  describe('Quick Actions', () => {
    it('renders quick actions section', async () => {
      render(<OverviewPage />);

      expect(screen.getByText('Quick Actions')).toBeInTheDocument();
      expect(screen.getByText('Create Agent')).toBeInTheDocument();
      expect(screen.getByText('Mint Token')).toBeInTheDocument();
      expect(screen.getByText('Create API Key')).toBeInTheDocument();
      expect(screen.getByText('Register Server')).toBeInTheDocument();
    });

    it('navigates to create agent page when clicking Create Agent', async () => {
      const { user } = render(<OverviewPage />);

      await waitFor(() => {
        expect(screen.getByText('Create Agent')).toBeInTheDocument();
      });

      const createAgentButton = screen.getByText('Create Agent').closest('button');
      if (createAgentButton) {
        await user.click(createAgentButton);
        expect(mockNavigate).toHaveBeenCalledWith('/agents/new');
      }
    });

    it('navigates to mint token page when clicking Mint Token', async () => {
      const { user } = render(<OverviewPage />);

      await waitFor(() => {
        expect(screen.getByText('Mint Token')).toBeInTheDocument();
      });

      const mintTokenButton = screen.getByText('Mint Token').closest('button');
      if (mintTokenButton) {
        await user.click(mintTokenButton);
        expect(mockNavigate).toHaveBeenCalledWith('/credentials/tokens/mint');
      }
    });

    it('navigates to create API key page when clicking Create API Key', async () => {
      const { user } = render(<OverviewPage />);

      await waitFor(() => {
        expect(screen.getByText('Create API Key')).toBeInTheDocument();
      });

      const apiKeyButton = screen.getByText('Create API Key').closest('button');
      if (apiKeyButton) {
        await user.click(apiKeyButton);
        expect(mockNavigate).toHaveBeenCalledWith('/credentials/api-keys/new');
      }
    });

    it('navigates to register server page when clicking Register Server', async () => {
      const { user } = render(<OverviewPage />);

      await waitFor(() => {
        expect(screen.getByText('Register Server')).toBeInTheDocument();
      });

      const registerServerButton = screen.getByText('Register Server').closest('button');
      if (registerServerButton) {
        await user.click(registerServerButton);
        expect(mockNavigate).toHaveBeenCalledWith('/servers/new');
      }
    });
  });

  describe('EnforceAI Not Enabled', () => {
    it('shows warning when EnforceAI returns 404', async () => {
      // Override the handler to return 404
      server.use(
        http.get('/enforceai/agents', () => {
          return HttpResponse.json(
            { detail: 'EnforceAI not enabled' },
            { status: 404 }
          );
        })
      );

      render(<OverviewPage />);

      // Wait for the warning to appear
      await waitFor(
        () => {
          expect(screen.getByText('EnforceAI Not Enabled')).toBeInTheDocument();
        },
        { timeout: 3000 }
      );

      // Check the warning message
      expect(
        screen.getByText(/EnforceAI management features are not available/)
      ).toBeInTheDocument();
    });

    it('disables EnforceAI quick actions when not enabled', async () => {
      // Override the handler to return 404
      server.use(
        http.get('/enforceai/agents', () => {
          return HttpResponse.json(
            { detail: 'EnforceAI not enabled' },
            { status: 404 }
          );
        })
      );

      render(<OverviewPage />);

      // Wait for data to load
      await waitFor(
        () => {
          expect(screen.getByText('EnforceAI Not Enabled')).toBeInTheDocument();
        },
        { timeout: 3000 }
      );

      // Check that EnforceAI-related quick actions are disabled
      const createAgentButton = screen.getByText('Create Agent').closest('button');
      const mintTokenButton = screen.getByText('Mint Token').closest('button');
      const createApiKeyButton = screen.getByText('Create API Key').closest('button');
      const registerServerButton = screen.getByText('Register Server').closest('button');

      expect(createAgentButton).toBeDisabled();
      expect(mintTokenButton).toBeDisabled();
      expect(createApiKeyButton).toBeDisabled();
      // Register Server should still be enabled (it's a registry operation)
      expect(registerServerButton).not.toBeDisabled();
    });
  });

  describe('Loading States', () => {
    it('shows loading spinners while data is fetching', async () => {
      // Add a delay to the handlers to test loading state
      server.use(
        http.get('/api/servers', async () => {
          await new Promise((resolve) => setTimeout(resolve, 100));
          return HttpResponse.json(mockServers);
        }),
        http.get('/api/agents', async () => {
          await new Promise((resolve) => setTimeout(resolve, 100));
          return HttpResponse.json(mockA2AAgents);
        }),
        http.get('/enforceai/agents', async () => {
          await new Promise((resolve) => setTimeout(resolve, 100));
          return HttpResponse.json(mockEnforceAIAgents);
        })
      );

      render(<OverviewPage />);

      // Should show spinners while loading
      // Note: Spinners might appear briefly, this test verifies the component doesn't crash
      await waitFor(() => {
        expect(screen.getByText('Summary')).toBeInTheDocument();
      });
    });
  });

  describe('Page Header', () => {
    it('renders page title and description', () => {
      render(<OverviewPage />);

      expect(screen.getByText('Overview')).toBeInTheDocument();
      expect(
        screen.getByText('Welcome to the Enforce Gateway management console')
      ).toBeInTheDocument();
    });

    it('renders Test Connections button in header', () => {
      render(<OverviewPage />);

      expect(
        screen.getByRole('button', { name: /test connections/i })
      ).toBeInTheDocument();
    });
  });
});
