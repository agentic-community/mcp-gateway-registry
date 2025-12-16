import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '../../../test/utils';
import { Header } from '../Header';

describe('Header', () => {
  const defaultProps = {
    user: null,
    onMenuClick: vi.fn(),
    onLogout: vi.fn(),
  };

  it('renders logo and title', () => {
    render(<Header {...defaultProps} />);

    expect(screen.getByText('Enforce Gateway')).toBeInTheDocument();
    expect(screen.getByText('EG')).toBeInTheDocument();
  });

  it('renders menu toggle button', () => {
    render(<Header {...defaultProps} />);

    const menuButton = screen.getByRole('button', { name: /toggle sidebar/i });
    expect(menuButton).toBeInTheDocument();
  });

  it('calls onMenuClick when menu button is clicked', async () => {
    const onMenuClick = vi.fn();
    const result = render(<Header {...defaultProps} onMenuClick={onMenuClick} />);

    const menuButton = screen.getByRole('button', { name: /toggle sidebar/i });
    await result.user.click(menuButton);

    expect(onMenuClick).toHaveBeenCalled();
  });

  it('renders theme toggle button', () => {
    render(<Header {...defaultProps} />);

    const themeButton = screen.getByRole('button', { name: /switch to/i });
    expect(themeButton).toBeInTheDocument();
  });

  it('does not render user menu when user is null', () => {
    render(<Header {...defaultProps} user={null} />);

    expect(screen.queryByText('User')).not.toBeInTheDocument();
  });

  it('renders user menu when user is authenticated', () => {
    const user = { username: 'testuser', email: 'test@example.com' };
    render(<Header {...defaultProps} user={user} />);

    expect(screen.getByText('test@example.com')).toBeInTheDocument();
  });

  it('shows admin badge for admin users', async () => {
    const testUser = { username: 'admin', email: 'admin@example.com', is_admin: true };
    const result = render(<Header {...defaultProps} user={testUser} />);

    // Open menu
    const userButton = screen.getByText('admin@example.com').closest('button');
    await result.user.click(userButton!);

    expect(screen.getByText('Administrator')).toBeInTheDocument();
  });

  it('renders settings link in user dropdown', async () => {
    const testUser = { username: 'testuser', email: 'test@example.com' };
    const result = render(<Header {...defaultProps} user={testUser} />);

    // Open menu
    const userButton = screen.getByText('test@example.com').closest('button');
    await result.user.click(userButton!);

    expect(screen.getByText('Settings')).toBeInTheDocument();
  });

  it('calls onLogout when sign out is clicked', async () => {
    const onLogout = vi.fn();
    const testUser = { username: 'testuser', email: 'test@example.com' };
    const result = render(
      <Header {...defaultProps} user={testUser} onLogout={onLogout} />
    );

    // Open menu
    const userButton = screen.getByText('test@example.com').closest('button');
    await result.user.click(userButton!);

    // Click sign out
    const signOutButton = screen.getByText('Sign out');
    await result.user.click(signOutButton);

    expect(onLogout).toHaveBeenCalled();
  });
});
