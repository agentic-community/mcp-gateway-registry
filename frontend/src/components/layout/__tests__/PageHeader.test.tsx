import { describe, it, expect } from 'vitest';
import { render, screen } from '../../../test/utils';
import { PageHeader } from '../PageHeader';
import { Button } from '../../ui';

describe('PageHeader', () => {
  it('renders title', () => {
    render(<PageHeader title="Test Title" />);

    expect(screen.getByText('Test Title')).toBeInTheDocument();
  });

  it('renders description when provided', () => {
    render(<PageHeader title="Title" description="Test description" />);

    expect(screen.getByText('Test description')).toBeInTheDocument();
  });

  it('does not render description when not provided', () => {
    render(<PageHeader title="Title" />);

    // No description paragraph should exist
    expect(screen.queryByText('Test description')).not.toBeInTheDocument();
  });

  it('renders breadcrumbs when provided', () => {
    render(
      <PageHeader
        title="Details"
        breadcrumbs={[
          { name: 'Parent', href: '/parent' },
          { name: 'Child' },
        ]}
      />
    );

    expect(screen.getByText('Parent')).toBeInTheDocument();
    expect(screen.getByText('Child')).toBeInTheDocument();
  });

  it('renders breadcrumb links correctly', () => {
    render(
      <PageHeader
        title="Details"
        breadcrumbs={[
          { name: 'Parent', href: '/parent' },
          { name: 'Current' },
        ]}
      />
    );

    const parentLink = screen.getByText('Parent').closest('a');
    expect(parentLink).toHaveAttribute('href', '/parent');
  });

  it('renders home icon in breadcrumbs', () => {
    render(
      <PageHeader
        title="Page"
        breadcrumbs={[{ name: 'Section' }]}
      />
    );

    expect(screen.getByRole('link', { name: /home/i })).toBeInTheDocument();
  });

  it('renders actions when provided', () => {
    render(
      <PageHeader
        title="Title"
        actions={<Button>Action</Button>}
      />
    );

    expect(screen.getByRole('button', { name: 'Action' })).toBeInTheDocument();
  });

  it('applies custom className', () => {
    const { container } = render(
      <PageHeader title="Title" className="custom-class" />
    );

    expect(container.firstChild).toHaveClass('custom-class');
  });
});
