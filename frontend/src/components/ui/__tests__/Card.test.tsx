import { describe, it, expect } from 'vitest';
import { render, screen } from '../../../test/utils';
import { Card, CardHeader, CardFooter } from '../Card';

describe('Card', () => {
  it('renders with children', () => {
    render(<Card>Card content</Card>);
    expect(screen.getByText('Card content')).toBeInTheDocument();
  });

  it('renders with medium padding by default', () => {
    render(<Card>Content</Card>);
    // The Card component wraps content directly, so get the container div
    const card = screen.getByText('Content').closest('div');
    expect(card).toHaveClass('p-6');
  });

  it('renders with no padding', () => {
    render(<Card padding="none">Content</Card>);
    const card = screen.getByText('Content').closest('div');
    expect(card).not.toHaveClass('p-4', 'p-6', 'p-8');
  });

  it('renders with small padding', () => {
    render(<Card padding="sm">Content</Card>);
    const card = screen.getByText('Content').closest('div');
    expect(card).toHaveClass('p-4');
  });

  it('renders with large padding', () => {
    render(<Card padding="lg">Content</Card>);
    const card = screen.getByText('Content').closest('div');
    expect(card).toHaveClass('p-8');
  });

  it('applies custom className', () => {
    render(<Card className="custom-class">Content</Card>);
    const card = screen.getByText('Content').closest('div');
    expect(card).toHaveClass('custom-class');
  });
});

describe('CardHeader', () => {
  it('renders with title', () => {
    render(<CardHeader title="Header Title" />);
    expect(screen.getByText('Header Title')).toBeInTheDocument();
  });

  it('renders with description', () => {
    render(<CardHeader title="Title" description="Header description" />);
    expect(screen.getByText('Header description')).toBeInTheDocument();
  });

  it('renders with action', () => {
    render(
      <CardHeader
        title="Title"
        action={<button>Action</button>}
      />
    );
    expect(screen.getByRole('button', { name: /action/i })).toBeInTheDocument();
  });

  it('applies custom className', () => {
    render(<CardHeader title="Title" className="custom-class" />);
    const header = screen.getByText('Title').parentElement?.parentElement;
    expect(header).toHaveClass('custom-class');
  });
});

describe('CardFooter', () => {
  it('renders with children', () => {
    render(<CardFooter>Footer content</CardFooter>);
    expect(screen.getByText('Footer content')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    render(<CardFooter className="custom-class">Footer</CardFooter>);
    const footer = screen.getByText('Footer').closest('div');
    expect(footer).toHaveClass('custom-class');
  });
});
