import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { MemoryRouter, Routes, Route } from 'react-router-dom';
import { render } from '@/test/utils';
import UpstreamOAuthCallbackPage from '../UpstreamOAuthCallbackPage';

const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

describe('UpstreamOAuthCallbackPage', () => {
  beforeEach(() => {
    mockNavigate.mockClear();
  });

  it('renders success and navigates back to upstream credentials with configure param', async () => {
    const user = userEvent.setup();

    render(
      <MemoryRouter
        initialEntries={[
          '/credentials/upstream/oauth/callback?upstream_oauth=success&server_path=/github-mcp&provider=github&credential_id=cred-1',
        ]}
      >
        <Routes>
          <Route
            path="/credentials/upstream/oauth/callback"
            element={<UpstreamOAuthCallbackPage />}
          />
        </Routes>
      </MemoryRouter>,
      { withRouter: false }
    );

    expect(screen.getByText('Successfully Connected')).toBeInTheDocument();
    expect(screen.getByText(/github account has been connected/i)).toBeInTheDocument();

    const returnButton = screen.getByRole('button', {
      name: /Return to Upstream Credentials/i,
    });
    await user.click(returnButton);

    expect(mockNavigate).toHaveBeenCalledWith(
      '/credentials/upstream?configure=github-mcp',
      { replace: true }
    );
  });

  it('renders error and allows retry navigation', async () => {
    const user = userEvent.setup();

    render(
      <MemoryRouter
        initialEntries={[
          '/credentials/upstream/oauth/callback?upstream_oauth=error&error_code=token_exchange_failed&server_path=/github-mcp&provider=github',
        ]}
      >
        <Routes>
          <Route
            path="/credentials/upstream/oauth/callback"
            element={<UpstreamOAuthCallbackPage />}
          />
        </Routes>
      </MemoryRouter>,
      { withRouter: false }
    );

    expect(screen.getByText('Connection Failed')).toBeInTheDocument();
    expect(screen.getByText(/token_exchange_failed/i)).toBeInTheDocument();

    const tryAgainButton = screen.getByRole('button', { name: /Try Again/i });
    await user.click(tryAgainButton);

    expect(mockNavigate).toHaveBeenCalledWith(
      '/credentials/upstream?configure=github-mcp',
      { replace: true }
    );
  });
});

