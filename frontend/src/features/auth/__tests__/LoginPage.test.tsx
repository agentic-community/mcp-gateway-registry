import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen, waitFor } from '../../../test/utils';
import { http, HttpResponse } from 'msw';
import { server } from '../../../test/mocks/server';
import LoginPage from '../LoginPage';

describe('LoginPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    // Start unauthenticated
    server.use(
      http.get('/api/auth/me', () => {
        return HttpResponse.json(
          { detail: 'Not authenticated' },
          { status: 401 }
        );
      })
    );
  });

  it('renders login form', async () => {
    render(<LoginPage />, { withAuth: true });

    await waitFor(() => {
      expect(
        screen.getByText('Sign in to Enforce Gateway')
      ).toBeInTheDocument();
    });

    expect(screen.getByLabelText(/username or email/i)).toBeInTheDocument();
    expect(screen.getByLabelText(/password/i)).toBeInTheDocument();
    expect(screen.getByRole('button', { name: /sign in/i })).toBeInTheDocument();
  });

  it('renders OAuth provider buttons when available', async () => {
    render(<LoginPage />, { withAuth: true });

    await waitFor(() => {
      expect(screen.getByText(/continue with google/i)).toBeInTheDocument();
    });

    expect(screen.getByText(/continue with github/i)).toBeInTheDocument();
    expect(
      screen.getByText(/or continue with password/i)
    ).toBeInTheDocument();
  });

  it('hides OAuth buttons when no providers available', async () => {
    server.use(
      http.get('/api/auth/providers', () => {
        return HttpResponse.json({ providers: [] });
      })
    );

    render(<LoginPage />, { withAuth: true });

    // Wait for providers to load
    await waitFor(() => {
      expect(
        screen.queryByRole('button', { name: /continue with/i })
      ).not.toBeInTheDocument();
    });

    expect(
      screen.queryByText(/or continue with password/i)
    ).not.toBeInTheDocument();
  });

  it('shows error message on login failure', async () => {
    server.use(
      http.post('/api/auth/login', () => {
        return HttpResponse.json(
          { detail: 'Invalid credentials' },
          { status: 401 }
        );
      })
    );

    const { user } = render(<LoginPage />, { withAuth: true });

    await user.type(screen.getByLabelText(/username or email/i), 'wronguser');
    await user.type(screen.getByLabelText(/password/i), 'wrongpass');
    await user.click(screen.getByRole('button', { name: /sign in/i }));

    await waitFor(() => {
      expect(
        screen.getByText(/login failed/i)
      ).toBeInTheDocument();
    });
  });

  it('renders help link', async () => {
    render(<LoginPage />, { withAuth: true });

    await waitFor(() => {
      expect(screen.getByText(/need help signing in/i)).toBeInTheDocument();
    });

    const helpLink = screen.getByText(/need help signing in/i);
    expect(helpLink).toHaveAttribute('href', '/help');
  });
});
