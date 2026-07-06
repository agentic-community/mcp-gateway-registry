import React from 'react';
import { isSafeUrl } from '../utils/safeUrl';

interface SafeLinkProps extends Omit<React.AnchorHTMLAttributes<HTMLAnchorElement>, 'href'> {
  /** The candidate URL. Server-registration or federation-supplied values are untrusted. */
  href: string | null | undefined;
  /** What to show when the URL is unsafe. Defaults to the anchor's children. */
  unsafeFallback?: React.ReactNode;
}

/**
 * Anchor that only renders a live link when `href` uses a script-safe scheme
 * (http/https/mailto). For any other scheme (javascript:, data:, vbscript:,
 * relative, ...) it renders the content as plain text so a malicious
 * registration/federation URL cannot execute script in the viewer's session.
 *
 * Use this in place of a raw `<a href={dynamicUrl}>` for any URL that originates
 * from server payloads or ARD federation discovery.
 */
export const SafeLink: React.FC<SafeLinkProps> = ({
  href,
  children,
  unsafeFallback,
  className,
  // target/rel are anchor-only; pull them out so they aren't spread onto a span.
  target,
  rel,
  ...rest
}) => {
  if (isSafeUrl(href)) {
    return (
      <a href={href as string} target={target} rel={rel} className={className} {...rest}>
        {children}
      </a>
    );
  }

  // Unsafe scheme: render as inert text, never a navigable link. We do NOT
  // reflect the raw href into a title tooltip — an unsafe value (e.g.
  // `data:text/html,...`) should not be surfaced verbatim to the user.
  return <span className={className}>{unsafeFallback ?? children}</span>;
};

/**
 * `components={{ a: safeMarkdownAnchor }}` override for react-markdown.
 *
 * react-markdown emits raw `<a href>` at runtime from markdown link syntax in
 * server-/federation-supplied content (e.g. a registered SKILL.md). That path
 * is not JSX, so the `react/jsx-no-script-url` lint rule cannot see it and the
 * default sanitization depends on library internals. Routing every rendered
 * link through this override applies the same scheme guard explicitly, so a
 * `[x](javascript:...)` link in remote content renders as inert text.
 */
export const safeMarkdownAnchor = ({
  href,
  children,
  ...rest
}: React.AnchorHTMLAttributes<HTMLAnchorElement>): React.ReactElement => {
  if (isSafeUrl(href)) {
    return (
      <a href={href} target="_blank" rel="noopener noreferrer nofollow" {...rest}>
        {children}
      </a>
    );
  }
  return <span>{children}</span>;
};

export default SafeLink;
