import { useCallback, useSyncExternalStore } from 'react';

/** Tailwind's `md` breakpoint: the app shell switches to a docked sidebar here. */
export const DESKTOP_QUERY = '(min-width: 768px)';

/**
 * Subscribe to a CSS media query. Re-renders when the match changes (window
 * resize, device rotation), unlike a one-off `window.innerWidth` read.
 * Returns `false` where `matchMedia` is unavailable.
 */
export function useMediaQuery(query: string): boolean {
  const subscribe = useCallback(
    (onChange: () => void) => {
      if (typeof window === 'undefined' || !window.matchMedia) {
        return () => undefined;
      }
      const media = window.matchMedia(query);
      media.addEventListener('change', onChange);
      return () => media.removeEventListener('change', onChange);
    },
    [query],
  );

  const getSnapshot = () =>
    typeof window !== 'undefined' && !!window.matchMedia && window.matchMedia(query).matches;

  return useSyncExternalStore(subscribe, getSnapshot, () => false);
}
