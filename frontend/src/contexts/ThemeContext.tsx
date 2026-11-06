import React, { createContext, useCallback, useContext, useEffect, useMemo, useState, ReactNode } from 'react';

/**
 * Light/dark mode. Drives Tailwind's `dark:` variants and the semantic color
 * tokens in index.css via the `dark` class on <html>. Kept as a distinct axis
 * from the named palette theme.
 */
type Mode = 'light' | 'dark';

/**
 * What the user asked for: an explicit mode, or `system` to follow the OS
 * preference (including live changes to it).
 */
export type ModePreference = Mode | 'system';

/**
 * Named palette theme. `default` uses the base token values; other names map to
 * an `html.<name>` block in index.css that remaps the CSS color variables at
 * runtime. Add a theme by defining that block and listing the name here.
 */
export type ThemeName = 'default' | 'high-contrast';

export const AVAILABLE_THEMES: ThemeName[] = ['default', 'high-contrast'];

interface ThemeContextType {
  /** Resolved light/dark mode currently applied. */
  theme: Mode;
  /** Flip the resolved mode and persist it as an explicit choice. */
  toggleTheme: () => void;
  /** The user's stored choice; `system` when following the OS. */
  preference: ModePreference;
  setPreference: (preference: ModePreference) => void;
  /** Active palette theme name. */
  themeName: ThemeName;
  setThemeName: (name: ThemeName) => void;
  availableThemes: ThemeName[];
}

const ThemeContext = createContext<ThemeContextType | undefined>(undefined);

export const useTheme = () => {
  const context = useContext(ThemeContext);
  if (context === undefined) {
    throw new Error('useTheme must be used within a ThemeProvider');
  }
  return context;
};

interface ThemeProviderProps {
  children: ReactNode;
}

// Storage keys are shared with the pre-paint bootstrap script in index.html;
// change both together.
const MODE_STORAGE_KEY = 'theme';
const THEME_NAME_STORAGE_KEY = 'themeName';

const THEME_CLASSES: ThemeName[] = AVAILABLE_THEMES.filter((t) => t !== 'default');

const SYSTEM_DARK_QUERY = '(prefers-color-scheme: dark)';

// Browser chrome color per mode; matches --color-surface in index.css.
const THEME_COLOR: Record<Mode, string> = {
  light: '#ffffff',
  dark: '#1d2939',
};


/** Read the OS-level light/dark preference, defaulting to dark when unknown. */
function _systemMode(): Mode {
  if (typeof window === 'undefined' || !window.matchMedia) {
    return 'dark';
  }
  return window.matchMedia(SYSTEM_DARK_QUERY).matches ? 'dark' : 'light';
}


function _readStorage(key: string): string | null {
  try {
    return window.localStorage.getItem(key);
  } catch {
    // Storage can throw in privacy modes; treat as "no saved choice".
    return null;
  }
}


function _writeStorage(key: string, value: string | null): void {
  try {
    if (value === null) {
      window.localStorage.removeItem(key);
    } else {
      window.localStorage.setItem(key, value);
    }
  } catch {
    // Non-persistent session; the in-memory choice still applies.
  }
}


function _initialPreference(): ModePreference {
  const saved = _readStorage(MODE_STORAGE_KEY);
  return saved === 'light' || saved === 'dark' ? saved : 'system';
}


function _initialThemeName(): ThemeName {
  const saved = _readStorage(THEME_NAME_STORAGE_KEY) as ThemeName | null;
  return saved && AVAILABLE_THEMES.includes(saved) ? saved : 'default';
}


function _applyMode(mode: Mode): void {
  const root = window.document.documentElement;
  root.classList.toggle('dark', mode === 'dark');
  document
    .querySelectorAll('meta[name="theme-color"]')
    .forEach((meta) => meta.setAttribute('content', THEME_COLOR[mode]));
}


export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  // Initial values are read synchronously so the first render already matches
  // what the index.html bootstrap script painted; there is no flash.
  const [preference, setPreferenceState] = useState<ModePreference>(_initialPreference);
  const [systemMode, setSystemMode] = useState<Mode>(_systemMode);
  const [themeName, setThemeNameState] = useState<ThemeName>(_initialThemeName);

  const theme: Mode = preference === 'system' ? systemMode : preference;

  useEffect(() => {
    // Track the OS preference so `system` follows live changes.
    if (typeof window === 'undefined' || !window.matchMedia) {
      return;
    }
    const media = window.matchMedia(SYSTEM_DARK_QUERY);
    const handleChange = (event: MediaQueryListEvent) => {
      setSystemMode(event.matches ? 'dark' : 'light');
    };
    media.addEventListener('change', handleChange);
    return () => media.removeEventListener('change', handleChange);
  }, []);

  useEffect(() => {
    _applyMode(theme);
  }, [theme]);

  useEffect(() => {
    // Apply the active palette theme class, clearing any other theme class so
    // switching themes never leaves a stale class behind.
    const root = window.document.documentElement;
    THEME_CLASSES.forEach((name) => root.classList.remove(name));
    if (themeName !== 'default') {
      root.classList.add(themeName);
    }
    _writeStorage(THEME_NAME_STORAGE_KEY, themeName);
  }, [themeName]);

  const setPreference = useCallback((next: ModePreference) => {
    // Only an explicit mode is persisted, so a system-following session never
    // masquerades as a saved preference.
    _writeStorage(MODE_STORAGE_KEY, next === 'system' ? null : next);
    setPreferenceState(next);
  }, []);

  const toggleTheme = useCallback(() => {
    setPreference(theme === 'light' ? 'dark' : 'light');
  }, [theme, setPreference]);

  const setThemeName = useCallback((name: ThemeName) => {
    if (AVAILABLE_THEMES.includes(name)) {
      setThemeNameState(name);
    }
  }, []);

  const value = useMemo<ThemeContextType>(
    () => ({
      theme,
      toggleTheme,
      preference,
      setPreference,
      themeName,
      setThemeName,
      availableThemes: AVAILABLE_THEMES,
    }),
    [theme, toggleTheme, preference, setPreference, themeName, setThemeName],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
};
