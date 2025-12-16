import { describe, it, expect } from 'vitest';
import { render, screen } from '../../../test/utils';
import { Spinner } from '../Spinner';

describe('Spinner', () => {
  it('renders spinner element', () => {
    render(<Spinner />);
    expect(screen.getByTestId('spinner')).toBeInTheDocument();
  });

  it('renders medium size by default', () => {
    render(<Spinner />);
    const spinner = screen.getByTestId('spinner');
    expect(spinner).toHaveClass('h-6', 'w-6');
  });

  it('renders small size', () => {
    render(<Spinner size="sm" />);
    const spinner = screen.getByTestId('spinner');
    expect(spinner).toHaveClass('h-4', 'w-4');
  });

  it('renders large size', () => {
    render(<Spinner size="lg" />);
    const spinner = screen.getByTestId('spinner');
    expect(spinner).toHaveClass('h-8', 'w-8');
  });

  it('applies custom className', () => {
    render(<Spinner className="custom-class" />);
    const spinner = screen.getByTestId('spinner');
    expect(spinner).toHaveClass('custom-class');
  });

  it('has animate-spin class', () => {
    render(<Spinner />);
    const spinner = screen.getByTestId('spinner');
    expect(spinner).toHaveClass('animate-spin');
  });

  it('has aria-hidden attribute', () => {
    render(<Spinner />);
    const spinner = screen.getByTestId('spinner');
    expect(spinner).toHaveAttribute('aria-hidden', 'true');
  });
});
