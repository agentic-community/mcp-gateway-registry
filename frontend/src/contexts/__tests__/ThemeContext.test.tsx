import React from 'react';
import { renderHook, act } from '@testing-library/react';
import { ThemeProvider, useTheme } from '../ThemeContext';
import { BrandingManifest, EMPTY_MANIFEST } from '../../branding/manifest';

let mockManifest: BrandingManifest = EMPTY_MANIFEST;
jest.mock('../BrandingContext', () => ({
  useBranding: () => ({ manifest: mockManifest, loaded: true }),
}));

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <ThemeProvider>{children}</ThemeProvider>
);

const root = () => document.documentElement;

/** Install a matchMedia stub reporting the given system dark-mode preference. */
function _mockMatchMedia(prefersDark: boolean) {
  const listeners: Array<(e: MediaQueryListEvent) => void> = [];
  const mql = {
    matches: prefersDark,
    media: '(prefers-color-scheme: dark)',
    addEventListener: (_: string, cb: (e: MediaQueryListEvent) => void) =>
      listeners.push(cb),
    removeEventListener: (_: string, cb: (e: MediaQueryListEvent) => void) => {
      const i = listeners.indexOf(cb);
      if (i >= 0) listeners.splice(i, 1);
    },
  };
  Object.defineProperty(window, 'matchMedia', {
    configurable: true,
    writable: true,
    value: jest.fn().mockReturnValue(mql),
  });
  // Allow tests to drive a live OS change.
  return (nextPrefersDark: boolean) => {
    mql.matches = nextPrefersDark;
    listeners.forEach((cb) => cb({ matches: nextPrefersDark } as MediaQueryListEvent));
  };
}

const ACME: BrandingManifest = {
  ...EMPTY_MANIFEST,
  themes: [
    { id: 'acme', label: 'Acme' },
    { id: 'acme-night', label: 'Acme Night' },
  ],
};

describe('ThemeContext', () => {
  beforeEach(() => {
    localStorage.clear();
    root().className = '';
    delete root().dataset.brandTheme;
    Reflect.deleteProperty(window, 'matchMedia');
    mockManifest = EMPTY_MANIFEST;
  });

  it('defaults to dark mode and the default palette', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });
    expect(result.current.theme).toBe('dark');
    expect(result.current.themeName).toBe('default');
    expect(root().classList.contains('dark')).toBe(true);
  });

  it('follows the system preference when no choice is saved', () => {
    _mockMatchMedia(false); // OS prefers light
    const { result } = renderHook(() => useTheme(), { wrapper });
    expect(result.current.theme).toBe('light');
    expect(root().classList.contains('dark')).toBe(false);
    // Following the OS is not a saved choice.
    expect(localStorage.getItem('theme')).toBeNull();
  });

  it('mirrors live OS changes until the user picks a mode', () => {
    const emit = _mockMatchMedia(false); // start light
    const { result } = renderHook(() => useTheme(), { wrapper });
    expect(result.current.theme).toBe('light');

    act(() => emit(true)); // OS switches to dark
    expect(result.current.theme).toBe('dark');

    // Once chosen, OS changes no longer override.
    act(() => result.current.toggleTheme()); // -> light, explicit
    expect(result.current.theme).toBe('light');
    act(() => emit(true));
    expect(result.current.theme).toBe('light');
  });

  it('prefers a saved choice over the system preference', () => {
    _mockMatchMedia(false); // OS prefers light
    localStorage.setItem('theme', 'dark');
    const { result } = renderHook(() => useTheme(), { wrapper });
    expect(result.current.theme).toBe('dark');
  });

  it('resumes following the OS after choosing system', () => {
    const emit = _mockMatchMedia(false); // OS prefers light
    localStorage.setItem('theme', 'dark');
    const { result } = renderHook(() => useTheme(), { wrapper });

    act(() => result.current.setPreference('system'));
    expect(result.current.theme).toBe('light');
    expect(localStorage.getItem('theme')).toBeNull();

    act(() => emit(true));
    expect(result.current.theme).toBe('dark');
  });

  it('uses the branding default mode only until the user chooses', () => {
    _mockMatchMedia(true); // OS prefers dark
    mockManifest = { ...EMPTY_MANIFEST, defaultMode: 'light' };
    const { result } = renderHook(() => useTheme(), { wrapper });
    expect(result.current.theme).toBe('light');

    act(() => result.current.setPreference('dark'));
    expect(result.current.theme).toBe('dark');
  });

  it('lets a branding lock override a saved mode', () => {
    localStorage.setItem('theme', 'dark');
    mockManifest = { ...EMPTY_MANIFEST, lockMode: 'light' };
    const { result } = renderHook(() => useTheme(), { wrapper });
    expect(result.current.modeLocked).toBe(true);
    expect(result.current.theme).toBe('light');
    expect(root().classList.contains('dark')).toBe(false);
  });

  it('applies a branding theme, persists it, and swaps the class on change', () => {
    mockManifest = ACME;
    const { result } = renderHook(() => useTheme(), { wrapper });

    act(() => result.current.setThemeName('acme'));
    expect(root().classList.contains('acme')).toBe(true);
    expect(localStorage.getItem('themeName')).toBe('acme');

    act(() => result.current.setThemeName('acme-night'));
    expect(root().classList.contains('acme')).toBe(false);
    expect(root().classList.contains('acme-night')).toBe(true);

    act(() => result.current.setThemeName('default'));
    expect(root().classList.contains('acme-night')).toBe(false);
  });

  it('ignores theme names the deployment does not offer', () => {
    mockManifest = ACME;
    const { result } = renderHook(() => useTheme(), { wrapper });
    act(() => result.current.setThemeName('bogus'));
    expect(result.current.themeName).toBe('default');
    expect(localStorage.getItem('themeName')).toBeNull();
  });

  it('starts on the branding default theme, and a saved default wins over it', () => {
    mockManifest = { ...ACME, defaultTheme: 'acme' };
    const first = renderHook(() => useTheme(), { wrapper });
    expect(first.result.current.themeName).toBe('acme');
    act(() => first.result.current.setThemeName('default'));
    first.unmount();

    const second = renderHook(() => useTheme(), { wrapper });
    expect(second.result.current.themeName).toBe('default');
  });

  it('falls back to default when a saved theme was removed from the manifest', () => {
    localStorage.setItem('themeName', 'retired');
    mockManifest = ACME;
    const { result } = renderHook(() => useTheme(), { wrapper });
    expect(result.current.themeName).toBe('default');
  });

  it('removes only the theme class it applied, never other classes on <html>', () => {
    mockManifest = ACME;
    root().classList.add('vendor-widget');
    const { result } = renderHook(() => useTheme(), { wrapper });
    act(() => result.current.setThemeName('acme'));
    act(() => result.current.setThemeName('acme-night'));
    expect(root().classList.contains('vendor-widget')).toBe(true);
  });

  it('combines high contrast with a branding theme and dark mode', () => {
    mockManifest = ACME;
    const { result } = renderHook(() => useTheme(), { wrapper });
    act(() => result.current.setThemeName('acme'));
    act(() => result.current.setHighContrast(true));
    expect(['dark', 'acme', 'high-contrast'].every((c) => root().classList.contains(c))).toBe(true);
    expect(localStorage.getItem('contrast')).toBe('high');
  });

  it('migrates the legacy high-contrast theme name to the contrast setting', () => {
    localStorage.setItem('themeName', 'high-contrast');
    const { result } = renderHook(() => useTheme(), { wrapper });
    expect(result.current.highContrast).toBe(true);
    expect(result.current.themeName).toBe('default');
    expect(localStorage.getItem('contrast')).toBe('high');
    expect(localStorage.getItem('themeName')).toBeNull();
  });

  it('uses the branding theme-color for the browser chrome', () => {
    document.head.innerHTML = '<meta name="theme-color" content="#000000">';
    mockManifest = { ...EMPTY_MANIFEST, themeColor: { dark: '#112233' } };
    renderHook(() => useTheme(), { wrapper });
    expect(document.querySelector('meta[name="theme-color"]')?.getAttribute('content')).toBe('#112233');
  });
});
