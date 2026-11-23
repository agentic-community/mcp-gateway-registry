import React from 'react';
import { render, screen } from '@testing-library/react';
import Badge from '../Badge';

describe('Badge', () => {
  it('renders its content', () => {
    render(<Badge tone="teal">VIRTUAL</Badge>);
    expect(screen.getByText('VIRTUAL')).toBeInTheDocument();
  });

  it('applies the tone classes', () => {
    render(<Badge tone="amber">SKILL</Badge>);
    expect(screen.getByText('SKILL').className).toContain('bg-warning-100');
    expect(screen.getByText('SKILL').className).toContain('dark:text-warning-300');
  });

  it('defaults to a pill shape and supports square', () => {
    const { rerender } = render(<Badge>A</Badge>);
    expect(screen.getByText('A').className).toContain('rounded-full');
    rerender(<Badge shape="square">B</Badge>);
    expect(screen.getByText('B').className).toContain('rounded');
    expect(screen.getByText('B').className).not.toContain('rounded-full');
  });

  it('adds a border when bordered', () => {
    render(<Badge tone="teal" bordered>VIRTUAL</Badge>);
    expect(screen.getByText('VIRTUAL').className).toContain('border-virtual-200');
  });

  it('passes through a title', () => {
    render(<Badge tone="red" title="No longer exists">ORPHANED</Badge>);
    expect(screen.getByText('ORPHANED')).toHaveAttribute('title', 'No longer exists');
  });
});
