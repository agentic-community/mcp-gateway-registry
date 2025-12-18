import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import { UpstreamCredentialModal } from '../UpstreamCredentialModal';
import type { Server } from '@/api/types';

describe('UpstreamCredentialModal', () => {
  const mockOnClose = vi.fn();
  const mockOnSuccess = vi.fn();

  const mockServerApiKey: Server = {
    display_name: 'GitHub MCP Server',
    path: 'github-mcp-test',
    proxy_pass_url: 'http://localhost:3001',
    is_enabled: true,
    upstream_auth: {
      mode: 'gateway-managed',
      type: 'api-key',
      credential_binding: 'user',
    },
    upstream_credential_status: 'missing',
  };

  const mockServerJwt: Server = {
    display_name: 'JWT Auth Server',
    path: 'jwt-server-test',
    proxy_pass_url: 'http://localhost:3002',
    is_enabled: true,
    upstream_auth: {
      mode: 'gateway-managed',
      type: 'jwt',
      credential_binding: 'user',
    },
    upstream_credential_status: 'missing',
  };

  const mockServerOauth: Server = {
    display_name: 'OAuth Server',
    path: 'oauth-server',
    proxy_pass_url: 'http://localhost:3003',
    is_enabled: true,
    upstream_auth: {
      mode: 'gateway-managed',
      type: 'oauth2',
      provider: 'github',
      credential_binding: 'user',
    },
    upstream_credential_status: 'missing',
  };

  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();

    // Default handler returns 404 (no credential)
    server.use(
      http.get('/enforceai/upstream/servers/:serverPath/credentials', () => {
        return HttpResponse.json(
          { detail: 'No credential configured' },
          { status: 404 }
        );
      })
    );
  });

  describe('Form Display', () => {
    it('renders API key form for api-key auth type', async () => {
      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerApiKey}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Configure API Key')).toBeInTheDocument();
      });

      // Use id selector to avoid matching multiple elements with "API Key" text
      expect(document.getElementById('secret')).toBeInTheDocument();
    });

    it('renders JWT form for jwt auth type', async () => {
      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerJwt}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Configure JWT Token')).toBeInTheDocument();
      });

      // Use id selector
      expect(document.getElementById('secret')).toBeInTheDocument();
    });

    it('shows OAuth connect view for OAuth type', async () => {
      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerOauth}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /Connect Github/i })).toBeInTheDocument();
      });

      expect(screen.getByText('OAuth Connection Required')).toBeInTheDocument();
    });

    it('displays server info', async () => {
      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerApiKey}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('GitHub MCP Server')).toBeInTheDocument();
      });
    });
  });

  describe('Form Validation', () => {
    it('requires secret field', async () => {
      const user = userEvent.setup();

      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerApiKey}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Configure API Key')).toBeInTheDocument();
      });

      const submitButton = screen.getByRole('button', { name: /Save Credential/i });
      await user.click(submitButton);

      // Error may appear in multiple places (input error and validation message)
      await waitFor(() => {
        expect(screen.getAllByText(/API Key is required/i).length).toBeGreaterThan(0);
      });
    });

    it('validates expiration date is in the future', async () => {
      const user = userEvent.setup();

      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerApiKey}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Configure API Key')).toBeInTheDocument();
      });

      const secretInput = document.getElementById('secret') as HTMLInputElement;
      await user.type(secretInput, 'test-api-key');

      const expiresInput = document.getElementById('expiresAt') as HTMLInputElement;
      // Set to past date
      await user.type(expiresInput, '2020-01-01T00:00');

      const submitButton = screen.getByRole('button', { name: /Save Credential/i });
      await user.click(submitButton);

      // Error may appear in multiple places (input error and validation message)
      await waitFor(() => {
        expect(screen.getAllByText(/Expiration date must be in the future/i).length).toBeGreaterThan(0);
      });
    });
  });

  describe('Credential Creation', () => {
    it('creates credential and shows secret once', async () => {
      const user = userEvent.setup();

      server.use(
        http.post('/enforceai/upstream/servers/:serverPath/credentials', async ({ request }) => {
          const body = (await request.json()) as { secret: string };
          return HttpResponse.json({
            credential_id: 'new-cred-123',
            server_path: 'github-mcp-test',
            credential_type: 'api-key',
            secret: body.secret,
            created_at: new Date().toISOString(),
          });
        })
      );

      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerApiKey}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Configure API Key')).toBeInTheDocument();
      });

      const secretInput = document.getElementById('secret') as HTMLInputElement;
      await user.type(secretInput, 'sk-test-api-key-12345');

      const submitButton = screen.getByRole('button', { name: /Save Credential/i });
      await user.click(submitButton);

      // Should show the secret display view
      await waitFor(() => {
        expect(screen.getByText('Credential Configured')).toBeInTheDocument();
      });

      // Secret should be visible
      expect(screen.getByText('sk-test-api-key-12345')).toBeInTheDocument();
    });

    it('prevents closing without acknowledgment', async () => {
      const user = userEvent.setup();

      server.use(
        http.post('/enforceai/upstream/servers/:serverPath/credentials', async ({ request }) => {
          const body = (await request.json()) as { secret: string };
          return HttpResponse.json({
            credential_id: 'new-cred-123',
            server_path: 'github-mcp-test',
            credential_type: 'api-key',
            secret: body.secret,
            created_at: new Date().toISOString(),
          });
        })
      );

      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerApiKey}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Configure API Key')).toBeInTheDocument();
      });

      const secretInput = document.getElementById('secret') as HTMLInputElement;
      await user.type(secretInput, 'sk-test-key');

      const submitButton = screen.getByRole('button', { name: /Save Credential/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText('Credential Configured')).toBeInTheDocument();
      });

      // Done button should be disabled without acknowledgment
      const doneButton = screen.getByRole('button', { name: /Done/i });
      expect(doneButton).toBeDisabled();

      // Click acknowledgment checkbox
      const checkbox = screen.getByRole('checkbox');
      await user.click(checkbox);

      // Now Done button should be enabled
      expect(doneButton).not.toBeDisabled();
    });

    it('calls onSuccess after acknowledging and clicking Done', async () => {
      const user = userEvent.setup();

      server.use(
        http.post('/enforceai/upstream/servers/:serverPath/credentials', async ({ request }) => {
          const body = (await request.json()) as { secret: string };
          return HttpResponse.json({
            credential_id: 'new-cred-123',
            server_path: 'github-mcp-test',
            credential_type: 'api-key',
            secret: body.secret,
            created_at: new Date().toISOString(),
          });
        })
      );

      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerApiKey}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Configure API Key')).toBeInTheDocument();
      });

      const secretInput = document.getElementById('secret') as HTMLInputElement;
      await user.type(secretInput, 'sk-test-key');

      const submitButton = screen.getByRole('button', { name: /Save Credential/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText('Credential Configured')).toBeInTheDocument();
      });

      const checkbox = screen.getByRole('checkbox');
      await user.click(checkbox);

      const doneButton = screen.getByRole('button', { name: /Done/i });
      await user.click(doneButton);

      expect(mockOnSuccess).toHaveBeenCalled();
    });
  });

  describe('Existing Credential', () => {
    it('shows existing credential view when credential exists', async () => {
      server.use(
        http.get('/enforceai/upstream/servers/:serverPath/credentials', () => {
          return HttpResponse.json({
            credential_id: 'existing-cred-123',
            server_path: 'github-mcp-test',
            credential_type: 'api-key',
            binding: 'user',
            user_id: 'local|testuser',
            status: 'configured',
            expires_at: null,
            created_at: '2024-01-01T00:00:00Z',
            updated_at: '2024-01-01T00:00:00Z',
          });
        })
      );

      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerApiKey}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Upstream Credential')).toBeInTheDocument();
      });

      expect(screen.getByText('configured')).toBeInTheDocument();
      expect(screen.getByText(/Revoke/i)).toBeInTheDocument();
      expect(screen.getByText(/Configure New/i)).toBeInTheDocument();
    });

    it('shows revoke confirmation dialog', async () => {
      const user = userEvent.setup();

      server.use(
        http.get('/enforceai/upstream/servers/:serverPath/credentials', () => {
          return HttpResponse.json({
            credential_id: 'existing-cred-123',
            server_path: 'github-mcp-test',
            credential_type: 'api-key',
            binding: 'user',
            status: 'configured',
            created_at: '2024-01-01T00:00:00Z',
            updated_at: '2024-01-01T00:00:00Z',
          });
        })
      );

      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerApiKey}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Upstream Credential')).toBeInTheDocument();
      });

      const revokeButton = screen.getByRole('button', { name: /^Revoke$/i });
      await user.click(revokeButton);

      // Wait for confirmation dialog to appear
      await waitFor(() => {
        expect(screen.getByText(/Are you sure/i)).toBeInTheDocument();
      });

      // Verify the confirm button is present
      expect(screen.getByRole('button', { name: /Revoke Credential/i })).toBeInTheDocument();
    });

    it('revokes credential successfully', async () => {
      const user = userEvent.setup();

      server.use(
        http.get('/enforceai/upstream/servers/:serverPath/credentials', () => {
          return HttpResponse.json({
            credential_id: 'existing-cred-123',
            server_path: 'github-mcp-test',
            credential_type: 'api-key',
            binding: 'user',
            status: 'configured',
            created_at: '2024-01-01T00:00:00Z',
            updated_at: '2024-01-01T00:00:00Z',
          });
        }),
        http.post('/enforceai/upstream/credentials/:credentialId/revoke', () => {
          return new HttpResponse(null, { status: 204 });
        })
      );

      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerApiKey}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Upstream Credential')).toBeInTheDocument();
      });

      const revokeButton = screen.getByRole('button', { name: /^Revoke$/i });
      await user.click(revokeButton);

      // Wait for confirmation dialog
      await waitFor(() => {
        expect(screen.getByText(/Are you sure/i)).toBeInTheDocument();
      });

      const confirmRevokeButton = screen.getByRole('button', { name: /Revoke Credential/i });
      await user.click(confirmRevokeButton);

      await waitFor(() => {
        expect(mockOnSuccess).toHaveBeenCalled();
      });
    });

    it('allows configuring new credential from existing view', async () => {
      const user = userEvent.setup();

      server.use(
        http.get('/enforceai/upstream/servers/:serverPath/credentials', () => {
          return HttpResponse.json({
            credential_id: 'existing-cred-123',
            server_path: 'github-mcp-test',
            credential_type: 'api-key',
            binding: 'user',
            status: 'configured',
            created_at: '2024-01-01T00:00:00Z',
            updated_at: '2024-01-01T00:00:00Z',
          });
        })
      );

      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerApiKey}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Upstream Credential')).toBeInTheDocument();
      });

      const configureNewButton = screen.getByRole('button', { name: /Configure New/i });
      await user.click(configureNewButton);

      await waitFor(() => {
        expect(screen.getByText('Configure API Key')).toBeInTheDocument();
      });
    });
  });

  describe('Modal Controls', () => {
    it('calls onClose when Cancel is clicked in form view', async () => {
      const user = userEvent.setup();

      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerApiKey}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Configure API Key')).toBeInTheDocument();
      });

      const cancelButton = screen.getByRole('button', { name: /Cancel/i });
      await user.click(cancelButton);

      expect(mockOnClose).toHaveBeenCalled();
    });

    it('does not render when closed', () => {
      render(
        <UpstreamCredentialModal
          open={false}
          server={mockServerApiKey}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      expect(screen.queryByText('Configure API Key')).not.toBeInTheDocument();
    });
  });

  describe('Action Buttons', () => {
    it('provides Download as .txt button after creation', async () => {
      const user = userEvent.setup();

      server.use(
        http.post('/enforceai/upstream/servers/:serverPath/credentials', async ({ request }) => {
          const body = (await request.json()) as { secret: string };
          return HttpResponse.json({
            credential_id: 'new-cred-123',
            server_path: 'github-mcp-test',
            credential_type: 'api-key',
            secret: body.secret,
            created_at: new Date().toISOString(),
          });
        })
      );

      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerApiKey}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Configure API Key')).toBeInTheDocument();
      });

      const secretInput = document.getElementById('secret') as HTMLInputElement;
      await user.type(secretInput, 'sk-test-key');

      const submitButton = screen.getByRole('button', { name: /Save Credential/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText('Credential Configured')).toBeInTheDocument();
      });

      expect(screen.getByRole('button', { name: /Download as .txt/i })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: /Copy as Header/i })).toBeInTheDocument();
    });
  });

  describe('OAuth Connect Flow', () => {
    const mockServerOAuthWithProvider: Server = {
      display_name: 'GitHub OAuth Server',
      path: 'github-oauth-test',
      proxy_pass_url: 'http://localhost:3010',
      is_enabled: true,
      upstream_auth: {
        mode: 'gateway-managed',
        type: 'provider-oauth',
        provider: 'github',
        credential_binding: 'user',
      },
      upstream_credential_status: 'missing',
    };

    it('shows connect button for OAuth types', async () => {
      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerOAuthWithProvider}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /Connect Github/i })).toBeInTheDocument();
      });

      expect(screen.getByText('OAuth Connection Required')).toBeInTheDocument();
    });

    it('displays gateway-terminated explanation', async () => {
      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerOAuthWithProvider}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /Connect Github/i })).toBeInTheDocument();
      });

      expect(screen.getByText(/gateway will manage tokens automatically/i)).toBeInTheDocument();
    });

    it('calls start OAuth flow when Connect is clicked', async () => {
      const user = userEvent.setup();
      let startCalled = false;
      const originalLocation = window.location.href;

      server.use(
        http.post('/enforceai/upstream/servers/:serverPath/oauth/start', async () => {
          startCalled = true;
          return HttpResponse.json({
            authorization_url: 'https://github.com/login/oauth/authorize?state=test',
            state: 'test-state',
          });
        })
      );

      // Mock window.location.href assignment
      const hrefSetter = vi.fn();
      Object.defineProperty(window, 'location', {
        value: { ...window.location, href: originalLocation },
        writable: true,
      });
      Object.defineProperty(window.location, 'href', {
        set: hrefSetter,
        get: () => originalLocation,
      });

      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerOAuthWithProvider}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByRole('button', { name: /Connect Github/i })).toBeInTheDocument();
      });

      const connectButton = screen.getByRole('button', { name: /Connect Github/i });
      await user.click(connectButton);

      await waitFor(() => {
        expect(startCalled).toBe(true);
      });
    });
  });

  describe('OAuth View (Connected)', () => {
    const mockServerOAuthConnected: Server = {
      display_name: 'GitHub OAuth Server',
      path: 'github-oauth-connected',
      proxy_pass_url: 'http://localhost:3010',
      is_enabled: true,
      upstream_auth: {
        mode: 'gateway-managed',
        type: 'provider-oauth',
        provider: 'github',
        credential_binding: 'user',
      },
      upstream_credential_status: 'configured',
    };

    beforeEach(() => {
      server.use(
        http.get('/enforceai/upstream/servers/:serverPath/credentials', () => {
          return HttpResponse.json({
            credential_id: 'oauth-cred-123',
            server_path: 'github-oauth-connected',
            credential_type: 'provider-oauth',
            binding: 'user',
            user_id: 'local|testuser',
            status: 'configured',
            provider: 'github',
            oauth_scopes: ['repo', 'read:user'],
            token_expires_at: new Date(Date.now() + 3600000).toISOString(),
            has_refresh_token: true,
            expires_at: null,
            created_at: '2024-01-01T00:00:00Z',
            updated_at: '2024-01-01T00:00:00Z',
            last_used_at: '2024-01-15T10:30:00Z',
          });
        })
      );
    });

    it('shows connected OAuth credential view', async () => {
      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerOAuthConnected}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Github Connection')).toBeInTheDocument();
      });

      // There are multiple "Connected" texts - the status label and the field label
      const connectedElements = screen.getAllByText('Connected');
      expect(connectedElements.length).toBeGreaterThanOrEqual(1);
      expect(screen.getByRole('button', { name: /Disconnect/i })).toBeInTheDocument();
    });

    it('displays OAuth provider name', async () => {
      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerOAuthConnected}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Github Connection')).toBeInTheDocument();
      });

      expect(screen.getByText('github')).toBeInTheDocument();
    });

    it('displays OAuth scopes', async () => {
      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerOAuthConnected}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Github Connection')).toBeInTheDocument();
      });

      expect(screen.getByText('repo')).toBeInTheDocument();
      expect(screen.getByText('read:user')).toBeInTheDocument();
    });

    it('shows auto-refresh status', async () => {
      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerOAuthConnected}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Github Connection')).toBeInTheDocument();
      });

      expect(screen.getByText('Auto-Refresh')).toBeInTheDocument();
      expect(screen.getByText('Enabled')).toBeInTheDocument();
    });
  });

  describe('OAuth Disconnect Flow', () => {
    const mockServerOAuthConnected: Server = {
      display_name: 'GitHub OAuth Server',
      path: 'github-oauth-to-disconnect',
      proxy_pass_url: 'http://localhost:3010',
      is_enabled: true,
      upstream_auth: {
        mode: 'gateway-managed',
        type: 'provider-oauth',
        provider: 'github',
        credential_binding: 'user',
      },
      upstream_credential_status: 'configured',
    };

    beforeEach(() => {
      server.use(
        http.get('/enforceai/upstream/servers/:serverPath/credentials', () => {
          return HttpResponse.json({
            credential_id: 'oauth-cred-456',
            server_path: 'github-oauth-to-disconnect',
            credential_type: 'provider-oauth',
            binding: 'user',
            status: 'configured',
            provider: 'github',
            oauth_scopes: ['repo'],
            has_refresh_token: true,
            created_at: '2024-01-01T00:00:00Z',
            updated_at: '2024-01-01T00:00:00Z',
          });
        }),
        http.post('/enforceai/upstream/servers/:serverPath/oauth/disconnect', () => {
          return new HttpResponse(null, { status: 204 });
        })
      );
    });

    it('shows disconnect confirmation dialog', async () => {
      const user = userEvent.setup();

      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerOAuthConnected}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Github Connection')).toBeInTheDocument();
      });

      const disconnectButton = screen.getByRole('button', { name: /Disconnect/i });
      await user.click(disconnectButton);

      await waitFor(() => {
        expect(screen.getByText('Disconnect OAuth')).toBeInTheDocument();
      });

      expect(screen.getByText(/Are you sure/i)).toBeInTheDocument();
    });

    it('disconnects OAuth successfully', async () => {
      const user = userEvent.setup();

      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerOAuthConnected}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Github Connection')).toBeInTheDocument();
      });

      // Click Disconnect to show confirmation
      const disconnectButton = screen.getByRole('button', { name: /Disconnect/i });
      await user.click(disconnectButton);

      await waitFor(() => {
        expect(screen.getByText('Disconnect OAuth')).toBeInTheDocument();
      });

      // After clicking, the modal changes to confirmation dialog with only one Disconnect button
      // Find the Disconnect button in the confirmation dialog (now it's the only one)
      const confirmButton = screen.getByRole('button', { name: /^Disconnect$/i });
      await user.click(confirmButton);

      await waitFor(() => {
        expect(mockOnSuccess).toHaveBeenCalled();
      });
    });

    it('allows canceling disconnect', async () => {
      const user = userEvent.setup();

      render(
        <UpstreamCredentialModal
          open={true}
          server={mockServerOAuthConnected}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('Github Connection')).toBeInTheDocument();
      });

      const disconnectButton = screen.getByRole('button', { name: /Disconnect/i });
      await user.click(disconnectButton);

      await waitFor(() => {
        expect(screen.getByText('Disconnect OAuth')).toBeInTheDocument();
      });

      const cancelButton = screen.getByRole('button', { name: /Cancel/i });
      await user.click(cancelButton);

      // Should return to view
      await waitFor(() => {
        expect(screen.getByText('Github Connection')).toBeInTheDocument();
      });

      // onSuccess should not have been called
      expect(mockOnSuccess).not.toHaveBeenCalled();
    });
  });
});
