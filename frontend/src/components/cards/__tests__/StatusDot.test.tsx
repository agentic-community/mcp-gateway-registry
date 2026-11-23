import React from 'react';
import { render, screen } from '@testing-library/react';
import StatusDot, { StatusDivider } from '../StatusDot';

describe('StatusDot', () => {
  it('renders the label', () => {
    render(<StatusDot tone="green" label="Enabled" />);
    expect(screen.getByText('Enabled')).toBeInTheDocument();
  });

  it('applies the glow class for an active tone', () => {
    const { container } = render(<StatusDot tone="green" label="Healthy" />);
    const dot = container.querySelector('div.rounded-full');
    expect(dot?.className).toContain('bg-success-400');
    expect(dot?.className).toContain('shadow-success-400/30');
  });

  it('applies the muted class for the off tone', () => {
    const { container } = render(<StatusDot tone="off" label="Disabled" />);
    const dot = container.querySelector('div.rounded-full');
    expect(dot?.className).toContain('bg-gray-300');
  });

  it('exposes the title as a tooltip on the label', () => {
    render(<StatusDot tone="emerald" label="Local" title="runs via stdio" />);
    expect(screen.getByText('Local')).toHaveAttribute('title', 'runs via stdio');
  });
});

describe('StatusDivider', () => {
  it('renders a thin vertical rule', () => {
    const { container } = render(<StatusDivider />);
    const rule = container.firstChild as HTMLElement;
    expect(rule.className).toContain('w-px');
    expect(rule.className).toContain('h-4');
  });
});
