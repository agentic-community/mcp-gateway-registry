import React from 'react';
import { render, screen } from '@testing-library/react';
import EmptyState from '../EmptyState';

describe('EmptyState', () => {
  it('renders the title and subtitle', () => {
    render(<EmptyState title="No servers found" subtitle="None registered yet" />);
    expect(screen.getByText('No servers found')).toBeInTheDocument();
    expect(screen.getByText('None registered yet')).toBeInTheDocument();
  });

  it('omits the subtitle when not provided', () => {
    const { container } = render(<EmptyState title="Empty" />);
    expect(container.querySelector('p')).toBeNull();
  });

  it('renders a CTA when provided', () => {
    render(<EmptyState title="Empty" cta={<button>Register</button>} />);
    expect(screen.getByRole('button', { name: 'Register' })).toBeInTheDocument();
  });

  it('uses the error tone styling for failures', () => {
    render(<EmptyState title="Failed to load" tone="error" />);
    expect(screen.getByText('Failed to load').className).toContain('text-danger-500');
  });

  it('applies a tinted background for accent tones', () => {
    const { container } = render(<EmptyState title="No agents found" tone="cyan" />);
    expect((container.firstChild as HTMLElement).className).toContain('bg-agent-50');
  });
});
