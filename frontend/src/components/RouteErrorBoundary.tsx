import React from 'react';
import { ArrowPathIcon, ExclamationTriangleIcon } from '@heroicons/react/24/outline';

/**
 * Messages browsers use when a dynamic import or Vite's CSS preload fails.
 * Chrome/Edge, Firefox, Safari, and Vite's own preload error respectively.
 */
const CHUNK_ERROR_PATTERNS: RegExp[] = [
  /Failed to fetch dynamically imported module/i,
  /error loading dynamically imported module/i,
  /Importing a module script failed/i,
  /Unable to preload CSS/i,
];

/** True when the error means a page's code could not be downloaded. */
export function isChunkLoadError(error: unknown): boolean {
  if (!(error instanceof Error)) return false;
  return CHUNK_ERROR_PATTERNS.some((pattern) => pattern.test(error.message));
}

interface RouteErrorBoundaryProps {
  children: React.ReactNode;
}

interface RouteErrorBoundaryState {
  error: Error | null;
}

/**
 * Catches errors from the lazily loaded routes so a failure shows a reload
 * prompt instead of a blank page. The common case is a tab left open across a
 * registry upgrade: the old entry script asks for page chunks whose hashed
 * names no longer exist, and a reload fetches the new build.
 */
class RouteErrorBoundary extends React.Component<RouteErrorBoundaryProps, RouteErrorBoundaryState> {
  state: RouteErrorBoundaryState = { error: null };

  static getDerivedStateFromError(error: Error): RouteErrorBoundaryState {
    return { error };
  }

  componentDidCatch(error: Error, info: React.ErrorInfo): void {
    console.error('Route failed to render', error, info.componentStack);
  }

  render(): React.ReactNode {
    const { error } = this.state;
    if (!error) return this.props.children;

    const chunkError = isChunkLoadError(error);
    return (
      <div role="alert" className="flex min-h-screen items-center justify-center bg-canvas p-6">
        <div className="card max-w-md p-6 text-center">
          <ExclamationTriangleIcon className="mx-auto h-10 w-10 text-warning-500" aria-hidden="true" />
          <h1 className="mt-3 text-lg font-semibold text-fg">
            {chunkError ? 'This page could not be loaded' : 'Something went wrong'}
          </h1>
          <p className="mt-2 text-sm text-fg-muted">
            {chunkError
              ? 'The registry may have been updated since this tab was opened. Reload to get the latest version.'
              : 'An unexpected error stopped this page from showing. Reloading usually fixes it.'}
          </p>
          <button type="button" className="btn-primary mt-5" onClick={() => window.location.reload()}>
            <ArrowPathIcon className="h-4 w-4" aria-hidden="true" />
            Reload
          </button>
        </div>
      </div>
    );
  }
}

export default RouteErrorBoundary;
