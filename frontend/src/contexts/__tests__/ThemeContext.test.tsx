import React from 'react';
import { renderHook, act } from '@testing-library/react';
import { ThemeProvider, useTheme, AVAILABLE_THEMES } from '../ThemeContext';

const wrapper = ({ children }: { children: React.ReactNode }) => (
  <ThemeProvider>{children}</ThemeProvider>
);

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
  (window as any).matchMedia = jest.fn().mockReturnValue(mql);
  // Allow tests to drive a live OS change.
  return (nextPrefersDark: boolean) => {
    mql.matches = nextPrefersDark;
    listeners.forEach((cb) =>
      cb({ matches: nextPrefersDark } as MediaQueryListEvent),
    );
  };
}

describe('ThemeContext', () => {
  beforeEach(() => {
    localStorage.clear();
    document.documentElement.className = '';
    delete (window as any).matchMedia;
  });

  it('defaults to dark mode and the default palette', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });
    expect(result.current.theme).toBe('dark');
    expect(result.current.themeName).toBe('default');
    expect(document.documentElement.classList.contains('dark')).toBe(true);
  });

  it('follows the system preference when no choice is saved', () => {
    _mockMatchMedia(false); // OS prefers light
    const { result } = renderHook(() => useTheme(), { wrapper });
    expect(result.current.theme).toBe('light');
    expect(document.documentElement.classList.contains('dark')).toBe(false);
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

  it('toggles light/dark mode and persists it', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });
    act(() => result.current.toggleTheme());
    expect(result.current.theme).toBe('light');
    expect(document.documentElement.classList.contains('dark')).toBe(false);
    expect(localStorage.getItem('theme')).toBe('light');
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

  it('restores a saved palette theme on mount', () => {
    localStorage.setItem('themeName', 'high-contrast');
    renderHook(() => useTheme(), { wrapper });
    expect(document.documentElement.classList.contains('high-contrast')).toBe(true);
  });

  it('applies a named palette theme class and persists it', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });
    act(() => result.current.setThemeName('high-contrast'));
    expect(result.current.themeName).toBe('high-contrast');
    expect(document.documentElement.classList.contains('high-contrast')).toBe(true);
    expect(localStorage.getItem('themeName')).toBe('high-contrast');
  });

  it('clears the previous theme class when switching back to default', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });
    act(() => result.current.setThemeName('high-contrast'));
    expect(document.documentElement.classList.contains('high-contrast')).toBe(true);
    act(() => result.current.setThemeName('default'));
    expect(document.documentElement.classList.contains('high-contrast')).toBe(false);
  });

  it('keeps dark mode independent of the palette theme', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });
    act(() => result.current.setThemeName('high-contrast'));
    // dark class is unaffected by the palette theme.
    expect(document.documentElement.classList.contains('dark')).toBe(true);
    expect(document.documentElement.classList.contains('high-contrast')).toBe(true);
  });

  it('ignores unknown theme names', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });
    act(() => result.current.setThemeName('bogus' as any));
    expect(result.current.themeName).toBe('default');
  });

  it('exposes the available theme list', () => {
    const { result } = renderHook(() => useTheme(), { wrapper });
    expect(result.current.availableThemes).toEqual(AVAILABLE_THEMES);
  });
});
