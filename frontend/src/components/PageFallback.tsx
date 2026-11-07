import React from 'react';
import clsx from 'clsx';

interface PageFallbackProps {
  /** Fill the viewport (routes outside the app shell) instead of the content area. */
  fullScreen?: boolean;
}

/** Suspense fallback shown while a lazily loaded page chunk downloads. */
const PageFallback: React.FC<PageFallbackProps> = ({ fullScreen = false }) => (
  <div
    role="status"
    aria-live="polite"
    className={clsx(
      'flex items-center justify-center',
      fullScreen ? 'min-h-screen bg-canvas' : 'flex-1 py-24',
    )}
  >
    <div className="h-8 w-8 animate-spin rounded-full border-2 border-line border-t-accent" />
    <span className="sr-only">Loading</span>
  </div>
);

export default PageFallback;
