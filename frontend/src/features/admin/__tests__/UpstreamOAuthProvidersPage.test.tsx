import { describe, it, expect, beforeEach } from 'vitest';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '@/test/utils';
import { mockUpstreamOAuthProviders } from '@/test/mocks/handlers';
import type { UpstreamOAuthProviderPublic } from '@/api/types';
import UpstreamOAuthProvidersPage from '../UpstreamOAuthProvidersPage';

function cloneProviders(
  providers: UpstreamOAuthProviderPublic[]
): UpstreamOAuthProviderPublic[] {
  return JSON.parse(JSON.stringify(providers)) as UpstreamOAuthProviderPublic[];
}

const initialProviders = cloneProviders(mockUpstreamOAuthProviders);

describe('UpstreamOAuthProvidersPage', () => {
  beforeEach(() => {
    mockUpstreamOAuthProviders.splice(
      0,
      mockUpstreamOAuthProviders.length,
      ...cloneProviders(initialProviders)
    );
  });

  it('renders providers from the API', async () => {
    render(<UpstreamOAuthProvidersPage />);

    expect(screen.getByText('Upstream OAuth Providers')).toBeInTheDocument();

    await waitFor(() => {
      expect(screen.getByText('github')).toBeInTheDocument();
      expect(screen.getByText('slack')).toBeInTheDocument();
    });
  });

  it('creates a provider without showing the secret afterwards', async () => {
    const user = userEvent.setup();
    render(<UpstreamOAuthProvidersPage />);

    await waitFor(() => {
      expect(screen.getByText('github')).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: /Add Provider/i }));

    await waitFor(() => {
      expect(screen.getByText('Create Upstream OAuth Provider')).toBeInTheDocument();
    });

    await user.type(screen.getByLabelText('Provider ID'), 'test-provider');
    await user.type(screen.getByLabelText('Client ID'), 'test-client-id');
    await user.type(
      screen.getByLabelText('Authorization Endpoint'),
      'https://example.com/oauth/authorize'
    );
    await user.type(
      screen.getByLabelText('Token Endpoint'),
      'https://example.com/oauth/token'
    );
    await user.type(screen.getByLabelText('Client Secret'), 'super-secret');
    await user.type(screen.getByLabelText('Default Scopes'), 'repo user:email');

    await user.click(screen.getByRole('button', { name: /Create Provider/i }));

    await waitFor(() => {
      expect(screen.getByText('Provider created')).toBeInTheDocument();
      expect(screen.getByText('test-provider')).toBeInTheDocument();
    });

    expect(screen.queryByText('super-secret')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Edit test-provider' }));

    await waitFor(() => {
      expect(screen.getByText('Edit Upstream OAuth Provider')).toBeInTheDocument();
    });

    expect(screen.getByLabelText('Client Secret')).toHaveValue('');
  });

  it('updates a provider and rotates secret when provided', async () => {
    const user = userEvent.setup();
    render(<UpstreamOAuthProvidersPage />);

    await waitFor(() => {
      expect(screen.getByText('slack')).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: 'Edit slack' }));

    await waitFor(() => {
      expect(screen.getByText('Edit Upstream OAuth Provider')).toBeInTheDocument();
    });

    await user.type(screen.getByLabelText('Client Secret'), 'rotated-secret');
    await user.click(screen.getByRole('button', { name: /Save Changes/i }));

    await waitFor(() => {
      expect(screen.getByText('Provider updated')).toBeInTheDocument();
    });

    await waitFor(() => {
      const row = screen.getByText('slack').closest('tr');
      expect(row).not.toBeNull();
      expect(within(row as HTMLElement).getByText('Set')).toBeInTheDocument();
    });

    expect(screen.queryByText('rotated-secret')).not.toBeInTheDocument();

    await user.click(screen.getByRole('button', { name: 'Edit slack' }));
    await waitFor(() => {
      expect(screen.getByText('Edit Upstream OAuth Provider')).toBeInTheDocument();
    });
    expect(screen.getByLabelText('Client Secret')).toHaveValue('');
  });

  it('shows an error toast when deletion is blocked by references', async () => {
    const user = userEvent.setup();
    render(<UpstreamOAuthProvidersPage />);

    await waitFor(() => {
      expect(screen.getByText('github')).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: 'Delete github' }));

    await waitFor(() => {
      expect(screen.getByText('Delete Provider')).toBeInTheDocument();
    });

    const confirmButtons = screen.getAllByRole('button', { name: /^Delete$/i });
    await user.click(confirmButtons[confirmButtons.length - 1]);

    await waitFor(() => {
      expect(screen.getByText('Failed to delete provider')).toBeInTheDocument();
      expect(
        screen.getByText('Provider is referenced by one or more servers')
      ).toBeInTheDocument();
    });
  });
});
