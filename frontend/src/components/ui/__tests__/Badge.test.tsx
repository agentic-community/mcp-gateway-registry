import { describe, it, expect } from 'vitest';
import { render, screen } from '../../../test/utils';
import { Badge } from '../Badge';

describe('Badge', () => {
  it('renders with children', () => {
    render(<Badge>Test Badge</Badge>);
    expect(screen.getByText('Test Badge')).toBeInTheDocument();
  });

  it('renders neutral variant by default', () => {
    render(<Badge>Neutral</Badge>);
    const badge = screen.getByText('Neutral');
    expect(badge).toHaveClass('bg-gray-100');
  });

  it('renders success variant', () => {
    render(<Badge variant="success">Success</Badge>);
    const badge = screen.getByText('Success');
    expect(badge).toHaveClass('bg-green-100');
  });

  it('renders warning variant', () => {
    render(<Badge variant="warning">Warning</Badge>);
    const badge = screen.getByText('Warning');
    expect(badge).toHaveClass('bg-yellow-100');
  });

  it('renders error variant', () => {
    render(<Badge variant="error">Error</Badge>);
    const badge = screen.getByText('Error');
    expect(badge).toHaveClass('bg-red-100');
  });

  it('renders info variant', () => {
    render(<Badge variant="info">Info</Badge>);
    const badge = screen.getByText('Info');
    expect(badge).toHaveClass('bg-blue-100');
  });

  it('renders small size by default', () => {
    render(<Badge>Small</Badge>);
    const badge = screen.getByText('Small');
    expect(badge).toHaveClass('text-xs');
  });

  it('renders medium size', () => {
    render(<Badge size="md">Medium</Badge>);
    const badge = screen.getByText('Medium');
    expect(badge).toHaveClass('text-sm');
  });

  it('applies custom className', () => {
    render(<Badge className="custom-class">Custom</Badge>);
    const badge = screen.getByText('Custom');
    expect(badge).toHaveClass('custom-class');
  });
});
