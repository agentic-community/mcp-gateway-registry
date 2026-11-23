import React from 'react';
import { render, screen } from '@testing-library/react';
import TagList from '../TagList';

describe('TagList', () => {
  it('renders nothing when there are no tags', () => {
    const { container } = render(<TagList tags={[]} />);
    expect(container.firstChild).toBeNull();
  });

  it('shows up to max tags and collapses the rest into +N', () => {
    render(<TagList tags={['a', 'b', 'c', 'd', 'e']} max={3} />);
    expect(screen.getByText('a')).toBeInTheDocument();
    expect(screen.getByText('c')).toBeInTheDocument();
    expect(screen.queryByText('d')).not.toBeInTheDocument();
    expect(screen.getByText('+2')).toBeInTheDocument();
  });

  it('applies a prefix to each tag', () => {
    render(<TagList tags={['db']} prefix="#" />);
    expect(screen.getByText('#db')).toBeInTheDocument();
  });

  it('does not render an overflow pill when tags fit', () => {
    render(<TagList tags={['a', 'b']} max={3} />);
    expect(screen.queryByText(/^\+/)).not.toBeInTheDocument();
  });

  it('applies a per-tag class override', () => {
    render(
      <TagList
        tags={['security-pending']}
        tagClassName={(t) =>
          t === 'security-pending' ? 'bg-warning-100' : undefined
        }
      />,
    );
    expect(screen.getByText('security-pending').className).toContain(
      'bg-warning-100',
    );
  });
});
