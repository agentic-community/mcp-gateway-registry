import React from 'react';
import { render, screen } from '@testing-library/react';
import { SafeLink, safeMarkdownAnchor } from '../SafeLink';

describe('SafeLink', () => {
  it('renders a live anchor for a safe https URL', () => {
    render(
      <SafeLink href="https://example.com/repo" target="_blank" rel="noopener noreferrer">
        View Repo
      </SafeLink>,
    );
    const link = screen.getByText('View Repo').closest('a');
    expect(link).not.toBeNull();
    expect(link).toHaveAttribute('href', 'https://example.com/repo');
  });

  it('does NOT render a navigable anchor for a javascript: URL', () => {
    // eslint-disable-next-line react/jsx-no-script-url -- intentional negative-test payload
    render(<SafeLink href="javascript:alert(document.cookie)">View Repo</SafeLink>);
    // The label is still shown, but never as a clickable <a href=...>.
    const text = screen.getByText('View Repo');
    expect(text.closest('a')).toBeNull();
    expect(document.querySelector('a[href]')).toBeNull();
  });

  it('does NOT render a navigable anchor for a data: URL', () => {
    render(<SafeLink href="data:text/html,<script>alert(1)</script>">Open</SafeLink>);
    expect(screen.getByText('Open').closest('a')).toBeNull();
    expect(document.querySelector('a[href]')).toBeNull();
  });

  it('does NOT render a navigable anchor for an obfuscated javascript: URL', () => {
    render(<SafeLink href={'java\tscript:alert(1)'}>Open</SafeLink>);
    expect(screen.getByText('Open').closest('a')).toBeNull();
    expect(document.querySelector('a[href]')).toBeNull();
  });

  it('renders the fallback text (not a link) for a null href', () => {
    render(<SafeLink href={null}>No URL</SafeLink>);
    expect(screen.getByText('No URL').closest('a')).toBeNull();
  });

  it('does not reflect the unsafe URL into a title tooltip', () => {
    // eslint-disable-next-line react/jsx-no-script-url -- intentional negative-test payload
    render(<SafeLink href="javascript:alert(1)">View</SafeLink>);
    const span = screen.getByText('View');
    expect(span.getAttribute('title')).toBeNull();
  });
});

describe('safeMarkdownAnchor', () => {
  it('renders a live anchor for a safe URL from markdown', () => {
    render(safeMarkdownAnchor({ href: 'https://example.com', children: 'link' }));
    const link = screen.getByText('link').closest('a');
    expect(link).toHaveAttribute('href', 'https://example.com');
    expect(link).toHaveAttribute('rel', 'noopener noreferrer nofollow');
  });

  it('renders inert text for a javascript: URL from markdown', () => {
    render(safeMarkdownAnchor({ href: 'javascript:alert(1)', children: 'link' }));
    expect(screen.getByText('link').closest('a')).toBeNull();
    expect(document.querySelector('a[href]')).toBeNull();
  });
});
