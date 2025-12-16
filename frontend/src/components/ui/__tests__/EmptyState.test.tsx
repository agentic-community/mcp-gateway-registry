import { describe, it, expect } from 'vitest';
import { render, screen } from '../../../test/utils';
import { EmptyState } from '../EmptyState';

describe('EmptyState', () => {
  it('renders with title', () => {
    render(<EmptyState title="No items found" />);
    expect(screen.getByText('No items found')).toBeInTheDocument();
  });

  it('renders with description', () => {
    render(
      <EmptyState
        title="No items"
        description="Try adding some items to get started"
      />
    );
    expect(
      screen.getByText('Try adding some items to get started')
    ).toBeInTheDocument();
  });

  it('renders with icon', () => {
    render(
      <EmptyState
        title="No items"
        icon={<span data-testid="empty-icon">Icon</span>}
      />
    );
    expect(screen.getByTestId('empty-icon')).toBeInTheDocument();
  });

  it('renders with action', () => {
    render(
      <EmptyState
        title="No items"
        action={<button>Add Item</button>}
      />
    );
    expect(screen.getByRole('button', { name: /add item/i })).toBeInTheDocument();
  });

  it('applies custom className', () => {
    render(<EmptyState title="No items" className="custom-class" />);
    const container = screen.getByText('No items').parentElement;
    expect(container).toHaveClass('custom-class');
  });
});
