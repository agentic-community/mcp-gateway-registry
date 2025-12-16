import { describe, it, expect, vi } from 'vitest';
import { render, screen } from '../../../test/utils';
import { Sidebar } from '../Sidebar';

describe('Sidebar', () => {
  it('renders all navigation sections when open', () => {
    render(<Sidebar isOpen={true} isAdmin={true} />);

    // Check main navigation items are rendered
    expect(screen.getByText('Overview')).toBeInTheDocument();
    expect(screen.getByText('Servers')).toBeInTheDocument();
    expect(screen.getByText('A2A Agents')).toBeInTheDocument();
    expect(screen.getByText('Agents')).toBeInTheDocument();
    expect(screen.getByText('Credentials')).toBeInTheDocument();
    expect(screen.getByText('Scopes')).toBeInTheDocument();
    expect(screen.getByText('Tools')).toBeInTheDocument();
    expect(screen.getByText('Audit')).toBeInTheDocument();
    expect(screen.getByText('Settings')).toBeInTheDocument();
    expect(screen.getByText('Help')).toBeInTheDocument();
  });

  it('renders Admin link for admin users', () => {
    render(<Sidebar isOpen={true} isAdmin={true} />);

    expect(screen.getByText('Admin')).toBeInTheDocument();
  });

  it('hides Admin link for non-admin users', () => {
    render(<Sidebar isOpen={true} isAdmin={false} />);

    expect(screen.queryByText('Admin')).not.toBeInTheDocument();
  });

  it('does not render sidebar content when closed', () => {
    render(<Sidebar isOpen={false} isAdmin={false} />);

    // Sidebar should not render any navigation items when closed
    expect(screen.queryByText('Overview')).not.toBeInTheDocument();
    expect(screen.queryByText('Servers')).not.toBeInTheDocument();
  });

  it('renders section titles correctly', () => {
    render(<Sidebar isOpen={true} isAdmin={true} />);

    expect(screen.getByText('Registry')).toBeInTheDocument();
    expect(screen.getByText('EnforceAI')).toBeInTheDocument();
    expect(screen.getByText('Policy')).toBeInTheDocument();
    expect(screen.getByText('Monitoring')).toBeInTheDocument();
    expect(screen.getByText('Administration')).toBeInTheDocument();
  });

  it('navigates to correct routes when links are clicked', () => {
    render(<Sidebar isOpen={true} isAdmin={true} />);

    const overviewLink = screen.getByText('Overview').closest('a');
    const serversLink = screen.getByText('Servers').closest('a');
    const settingsLink = screen.getByText('Settings').closest('a');

    expect(overviewLink).toHaveAttribute('href', '/');
    expect(serversLink).toHaveAttribute('href', '/servers');
    expect(settingsLink).toHaveAttribute('href', '/settings');
  });

  it('calls onClose when nav item is clicked on mobile', () => {
    // Mock window.innerWidth for mobile
    const originalInnerWidth = window.innerWidth;
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: 500,
    });

    const onClose = vi.fn();
    render(<Sidebar isOpen={true} isAdmin={false} onClose={onClose} />);

    const overviewLink = screen.getByText('Overview');
    overviewLink.click();

    expect(onClose).toHaveBeenCalled();

    // Restore
    Object.defineProperty(window, 'innerWidth', {
      writable: true,
      configurable: true,
      value: originalInnerWidth,
    });
  });

  it('expands credentials submenu when clicked', async () => {
    const result = render(<Sidebar isOpen={true} isAdmin={false} />);

    const credentialsButton = screen.getByText('Credentials');
    await result.user.click(credentialsButton);

    // Should show child links
    expect(screen.getByText('API Keys')).toBeInTheDocument();
    expect(screen.getByText('Gateway Tokens')).toBeInTheDocument();
  });

  it('renders version info at bottom', () => {
    render(<Sidebar isOpen={true} isAdmin={false} />);

    expect(screen.getByText('Enforce Gateway v1.0')).toBeInTheDocument();
  });
});
