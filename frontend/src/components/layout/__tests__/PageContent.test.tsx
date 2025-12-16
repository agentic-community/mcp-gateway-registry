import { describe, it, expect } from 'vitest';
import { render, screen } from '../../../test/utils';
import { PageContent } from '../PageContent';

describe('PageContent', () => {
  it('renders children', () => {
    render(
      <PageContent>
        <div>Test content</div>
      </PageContent>
    );

    expect(screen.getByText('Test content')).toBeInTheDocument();
  });

  it('applies default padding classes', () => {
    const { container } = render(
      <PageContent>Content</PageContent>
    );

    expect(container.firstChild).toHaveClass('px-4');
    expect(container.firstChild).toHaveClass('py-4');
  });

  it('removes padding when noPadding is true', () => {
    const { container } = render(
      <PageContent noPadding>Content</PageContent>
    );

    expect(container.firstChild).not.toHaveClass('px-4');
    expect(container.firstChild).not.toHaveClass('py-4');
  });

  it('applies max-width by default', () => {
    const { container } = render(
      <PageContent>Content</PageContent>
    );

    expect(container.firstChild).toHaveClass('max-w-7xl');
  });

  it('removes max-width when fullWidth is true', () => {
    const { container } = render(
      <PageContent fullWidth>Content</PageContent>
    );

    expect(container.firstChild).not.toHaveClass('max-w-7xl');
  });

  it('applies custom className', () => {
    const { container } = render(
      <PageContent className="custom-class">Content</PageContent>
    );

    expect(container.firstChild).toHaveClass('custom-class');
  });

  it('has flex-1 class for layout', () => {
    const { container } = render(
      <PageContent>Content</PageContent>
    );

    expect(container.firstChild).toHaveClass('flex-1');
  });
});
