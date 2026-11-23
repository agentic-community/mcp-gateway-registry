import React, { Suspense, lazy } from 'react';
import { render, screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import RouteErrorBoundary, { isChunkLoadError } from '../RouteErrorBoundary';

/** A lazy route whose chunk download fails, as after an upgrade removed it. */
const _failingRoute = (message: string) =>
  lazy(() => Promise.reject(new TypeError(message)));

const _renderRoute = (Route: React.ComponentType) =>
  render(
    <RouteErrorBoundary>
      <Suspense fallback={<span>Loading</span>}>
        <Route />
      </Suspense>
    </RouteErrorBoundary>,
  );

describe('RouteErrorBoundary', () => {
  const realLocation = window.location;
  let reload: jest.Mock;

  beforeEach(() => {
    jest.spyOn(console, 'error').mockImplementation(() => undefined);
    reload = jest.fn();
    Object.defineProperty(window, 'location', { configurable: true, value: { ...realLocation, reload } });
  });

  afterEach(() => {
    Object.defineProperty(window, 'location', { configurable: true, value: realLocation });
    jest.restoreAllMocks();
  });

  it('shows a reload prompt when a lazy page chunk fails to load', async () => {
    _renderRoute(_failingRoute('Failed to fetch dynamically imported module: /static/js/Settings.abc.js'));

    expect(await screen.findByRole('heading', { name: 'This page could not be loaded' })).toBeInTheDocument();
    await userEvent.click(screen.getByRole('button', { name: 'Reload' }));
    expect(reload).toHaveBeenCalledTimes(1);
  });

  it('shows a generic reload prompt for other render errors', async () => {
    const Broken = () => {
      throw new Error('boom');
    };
    _renderRoute(Broken);

    expect(await screen.findByRole('heading', { name: 'Something went wrong' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Reload' })).toBeInTheDocument();
  });

  it('renders the route when nothing fails', async () => {
    const Ok = lazy(() => Promise.resolve({ default: () => <p>Audit Logs</p> }));
    _renderRoute(Ok);

    expect(await screen.findByText('Audit Logs')).toBeInTheDocument();
    expect(screen.queryByRole('alert')).not.toBeInTheDocument();
  });
});

describe('isChunkLoadError', () => {
  it.each([
    'Failed to fetch dynamically imported module: https://x/static/js/a.js',
    'error loading dynamically imported module: https://x/static/js/a.js',
    'Importing a module script failed.',
    'Unable to preload CSS for /static/css/a.css',
  ])('recognizes %j', (message) => {
    expect(isChunkLoadError(new TypeError(message))).toBe(true);
  });

  it('does not treat other errors as chunk failures', () => {
    expect(isChunkLoadError(new Error('Cannot read properties of undefined'))).toBe(false);
    expect(isChunkLoadError('Failed to fetch dynamically imported module')).toBe(false);
  });
});
