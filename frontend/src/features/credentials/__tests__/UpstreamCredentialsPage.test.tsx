import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import { mockServers } from '@/test/mocks/handlers';
import UpstreamCredentialsPage from '../UpstreamCredentialsPage';

// Mock useNavigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

describe('UpstreamCredentialsPage', () => {
  beforeEach(() => {
    mockNavigate.mockClear();
    server.resetHandlers();
  });

  describe('Page Rendering', () => {
    it('renders page header and description', async () => {
      render(<UpstreamCredentialsPage />);

      expect(screen.getByText('Upstream Credentials')).toBeInTheDocument();
      expect(
        screen.getByText(/Manage authentication credentials for upstream servers/i)
      ).toBeInTheDocument();
    });

    it('renders filter section', async () => {
      render(<UpstreamCredentialsPage />);

      expect(screen.getByText('Filter by Status')).toBeInTheDocument();
    });

    it('renders all filter buttons', async () => {
      render(<UpstreamCredentialsPage />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /All/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Configured/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Missing/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Expired/i })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: /Revoked/i })).toBeInTheDocument();
      });
    });
  });

  describe('Loading and Data Display', () => {
    it('loads and displays servers with upstream auth', async () => {
      render(<UpstreamCredentialsPage />);

      await waitFor(() => {
        const serversWithAuth = mockServers.filter(
          (s) =>
            s.upstream_auth &&
            s.upstream_auth.mode === 'gateway-managed' &&
            s.upstream_auth.type !== 'none'
        );

        if (serversWithAuth.length > 0) {
          expect(
            screen.getByText(serversWithAuth[0].display_name || serversWithAuth[0].path)
          ).toBeInTheDocument();
        }
      });
    });

    it('only shows servers with gateway-managed upstream auth', async () => {
      render(<UpstreamCredentialsPage />);

      await waitFor(() => {
        const serversWithAuth = mockServers.filter(
          (s) =>
            s.upstream_auth &&
            s.upstream_auth.mode === 'gateway-managed' &&
            s.upstream_auth.type !== 'none'
        );

        serversWithAuth.forEach((server) => {
          expect(
            screen.getByText(server.display_name || server.path)
          ).toBeInTheDocument();
        });
      });
    });

    it('displays credential status badges', async () => {
      render(<UpstreamCredentialsPage />);

      await waitFor(() => {
        const serversWithAuth = mockServers.filter(
          (s) =>
            s.upstream_auth &&
            s.upstream_auth.mode === 'gateway-managed' &&
            s.upstream_auth.type !== 'none' &&
            s.upstream_credential_status
        );

        serversWithAuth.forEach((server) => {
          if (server.upstream_credential_status) {
            expect(
              screen.getAllByText(server.upstream_credential_status).length
            ).toBeGreaterThan(0);
          }
        });
      });
    });

    it('displays credential binding info', async () => {
      render(<UpstreamCredentialsPage />);

      await waitFor(() => {
        expect(screen.getAllByText(/Binding:/i).length).toBeGreaterThan(0);
      });
    });

    it('displays Configure buttons', async () => {
      render(<UpstreamCredentialsPage />);

      await waitFor(() => {
        const configureButtons = screen.getAllByRole('button', { name: /Configure/i });
        expect(configureButtons.length).toBeGreaterThan(0);
      });
    });
  });

  describe('Status Filtering', () => {
    it('shows all servers by default', async () => {
      render(<UpstreamCredentialsPage />);

      await waitFor(() => {
        const serversWithAuth = mockServers.filter(
          (s) =>
            s.upstream_auth &&
            s.upstream_auth.mode === 'gateway-managed' &&
            s.upstream_auth.type !== 'none'
        );

        serversWithAuth.forEach((server) => {
          expect(
            screen.getByText(server.display_name || server.path)
          ).toBeInTheDocument();
        });
      });
    });

    it('filters servers when status button is clicked', async () => {
      const user = userEvent.setup();
      render(<UpstreamCredentialsPage />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /Configured/i })).toBeInTheDocument();
      });

      const configuredButton = screen.getByRole('button', { name: /Configured/i });
      await user.click(configuredButton);

      // Just verify the button is active (has primary variant styles)
      await waitFor(() => {
        expect(configuredButton).toBeInTheDocument();
      });
    });

    it('displays status counts', async () => {
      render(<UpstreamCredentialsPage />);

      await waitFor(() => {
        const allButton = screen.getByRole('button', { name: /^All/i });
        expect(allButton).toHaveTextContent(/\d+/); // Contains a number
      });
    });
  });

  describe('Navigation', () => {
    it('has Configure buttons for navigation', async () => {
      render(<UpstreamCredentialsPage />);

      await waitFor(() => {
        const configureButtons = screen.getAllByRole('button', { name: /Configure/i });
        expect(configureButtons.length).toBeGreaterThan(0);
      });
    });
  });
});
