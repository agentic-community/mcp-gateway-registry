import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';

import ConnectedAccountsPage from '../ConnectedAccountsPage';
import * as egressAuth from '../../utils/egressAuth';

jest.mock('../../utils/egressAuth');
const mocked = egressAuth as jest.Mocked<typeof egressAuth>;

// The page uses useNavigate() for the "Back to Dashboard" link, so it must be
// rendered inside a Router.
const renderPage = () =>
  render(
    <MemoryRouter>
      <ConnectedAccountsPage />
    </MemoryRouter>
  );

describe('ConnectedAccountsPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Default: no egress-enabled servers available (overridden per test).
    mocked.listAvailableServers.mockResolvedValue([]);
  });

  it('lists existing connections', async () => {
    mocked.listConnections.mockResolvedValue([
      {
        provider: 'github',
        server_path: '/github-mcp',
        scopes: ['repo'],
        expires_at: null,
        status: 'active',
        last_refreshed_at: null,
      },
    ]);

    renderPage();

    expect(await screen.findByText('github')).toBeInTheDocument();
    expect(screen.getByText('/github-mcp')).toBeInTheDocument();
    expect(screen.getByText('repo')).toBeInTheDocument();
  });

  it('shows empty state when no connections', async () => {
    mocked.listConnections.mockResolvedValue([]);
    renderPage();
    expect(await screen.findByText('No connected accounts yet.')).toBeInTheDocument();
  });

  it('opens the authorize URL on Connect', async () => {
    mocked.listConnections.mockResolvedValue([]);
    mocked.listAvailableServers.mockResolvedValue([
      { server_path: '/github-mcp', server_name: 'GitHub MCP', provider: 'github' },
    ]);
    mocked.initiateConsent.mockResolvedValue('https://github.com/login/oauth/authorize?x=1');
    const openSpy = jest.spyOn(window, 'open').mockImplementation(() => null);

    renderPage();
    await screen.findByText('No connected accounts yet.');

    fireEvent.change(screen.getByLabelText('Server requiring per-user authentication'), {
      target: { value: '/github-mcp' },
    });
    fireEvent.click(screen.getByRole('button', { name: /connect/i }));

    await waitFor(() => expect(mocked.initiateConsent).toHaveBeenCalledWith('/github-mcp'));
    expect(openSpy).toHaveBeenCalledWith(
      'https://github.com/login/oauth/authorize?x=1',
      '_blank',
      'noopener,noreferrer'
    );
    openSpy.mockRestore();
  });

  it('disconnects and refreshes', async () => {
    mocked.listConnections
      .mockResolvedValueOnce([
        {
          provider: 'github',
          server_path: '/github-mcp',
          scopes: [],
          expires_at: null,
          status: 'active',
          last_refreshed_at: null,
        },
      ])
      .mockResolvedValueOnce([]);
    mocked.disconnect.mockResolvedValue();

    renderPage();
    fireEvent.click(await screen.findByRole('button', { name: /disconnect github/i }));

    await waitFor(() =>
      expect(mocked.disconnect).toHaveBeenCalledWith('github', '/github-mcp')
    );
    expect(await screen.findByText('No connected accounts yet.')).toBeInTheDocument();
  });

  it('surfaces a load error', async () => {
    mocked.listConnections.mockRejectedValue(new Error('boom'));
    renderPage();
    expect(await screen.findByText('Could not load connections.')).toBeInTheDocument();
  });
});
