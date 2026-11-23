import React from 'react';
import { render, screen } from '@testing-library/react';
import ProviderBadge from '../ProviderBadge';

describe('ProviderBadge', () => {
  it('renders the provider label', () => {
    render(<ProviderBadge provider="manual" />);
    expect(screen.getByText('manual')).toBeInTheDocument();
  });

  it('highlights the manual provider distinctly from IdPs', () => {
    const { rerender } = render(<ProviderBadge provider="manual" />);
    expect(screen.getByText('manual').className).toContain('bg-info-100');
    rerender(<ProviderBadge provider="okta" />);
    expect(screen.getByText('okta').className).toContain('bg-gray-100');
  });

  it('falls back to the neutral style for unknown providers', () => {
    render(<ProviderBadge provider="mystery" />);
    expect(screen.getByText('mystery').className).toContain('bg-gray-100');
  });
});
