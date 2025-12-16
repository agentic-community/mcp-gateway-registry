import { describe, it, expect, vi, beforeEach } from 'vitest';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { SessionExpiryWarning } from '../SessionExpiryWarning';

// Mock useAuth hook
const mockRefreshSession = vi.fn().mockResolvedValue(undefined);
const mockLogout = vi.fn().mockResolvedValue(undefined);
let mockSessionExpiring = false;

vi.mock('../../../contexts/AuthContext', () => ({
  useAuth: () => ({
    sessionExpiring: mockSessionExpiring,
    refreshSession: mockRefreshSession,
    logout: mockLogout,
  }),
}));

describe('SessionExpiryWarning', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    mockSessionExpiring = false;
  });

  it('does not render when session is not expiring', () => {
    mockSessionExpiring = false;
    render(<SessionExpiryWarning />);

    expect(
      screen.queryByText(/session expiring soon/i)
    ).not.toBeInTheDocument();
  });

  it('renders warning when session is expiring', () => {
    mockSessionExpiring = true;
    render(<SessionExpiryWarning />);

    expect(screen.getByText(/session expiring soon/i)).toBeInTheDocument();
    expect(
      screen.getByText(/your session will expire in a few minutes/i)
    ).toBeInTheDocument();
  });

  it('renders action buttons when session is expiring', () => {
    mockSessionExpiring = true;
    render(<SessionExpiryWarning />);

    expect(
      screen.getByRole('button', { name: /stay signed in/i })
    ).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /sign out/i })
    ).toBeInTheDocument();
  });

  it('calls refreshSession when Stay Signed In is clicked', async () => {
    mockSessionExpiring = true;
    const user = userEvent.setup();
    render(<SessionExpiryWarning />);

    await user.click(screen.getByRole('button', { name: /stay signed in/i }));

    expect(mockRefreshSession).toHaveBeenCalled();
  });

  it('calls logout when Sign Out is clicked', async () => {
    mockSessionExpiring = true;
    const user = userEvent.setup();
    render(<SessionExpiryWarning />);

    await user.click(screen.getByRole('button', { name: /sign out/i }));

    expect(mockLogout).toHaveBeenCalled();
  });

  it('has alert role for accessibility', () => {
    mockSessionExpiring = true;
    render(<SessionExpiryWarning />);

    expect(screen.getByRole('alert')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    mockSessionExpiring = true;
    render(<SessionExpiryWarning className="custom-class" />);

    const alert = screen.getByRole('alert');
    expect(alert).toHaveClass('custom-class');
  });
});
