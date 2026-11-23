import React, { createContext, useCallback, useContext, useEffect, useMemo, useState, ReactNode } from 'react';
import { useBranding } from './BrandingContext';
import { BrandTheme, ModeChoice, themeClass } from '../branding/manifest';

/**
 * Light/dark mode. Drives Tailwind's `dark:` variants and the semantic color
 * tokens in index.css via the `dark` class on <html>.
 */
type Mode = 'light' | 'dark';

/**
 * What the user asked for: an explicit mode, or `system` to follow the OS
 * preference (including live changes to it).
 */
export type ModePreference = ModeChoice;

/** `default` is the built-in look; any other value is a branding theme id. */
export type ThemeName = string;

export const DEFAULT_THEME: ThemeName = 'default';

interface ThemeContextType {
  /** Resolved light/dark mode currently applied. */
  theme: Mode;
  /** Flip the resolved mode and persist it as an explicit choice. */
  toggleTheme: () => void;
  /** The user's stored choice; `system` when following the OS. */
  preference: ModePreference;
  setPreference: (preference: ModePreference) => void;
  /** True when the deployment's branding forces one mode. */
  modeLocked: boolean;
  /** Active palette theme: `default` or a branding theme id. */
  themeName: ThemeName;
  setThemeName: (name: ThemeName) => void;
  /** Branding themes the deployment offers, in addition to `default`. */
  availableThemes: BrandTheme[];
  /** High-contrast palette, independent of mode and theme. */
  highContrast: boolean;
  setHighContrast: (enabled: boolean) => void;
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
const CONTRAST_STORAGE_KEY = 'contrast';
const HIGH_CONTRAST_CLASS = 'high-contrast';

const SYSTEM_DARK_QUERY = '(prefers-color-scheme: dark)';

// Browser chrome color per mode when branding sets none; matches --color-surface.
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


function _savedPreference(): ModePreference | null {
  const saved = _readStorage(MODE_STORAGE_KEY);
  return saved === 'light' || saved === 'dark' ? saved : null;
}


/**
 * High contrast used to be stored as themeName 'high-contrast'. Move that to
 * its own key so it combines with branding themes.
 */
function _migrateLegacyContrast(): void {
  if (_readStorage(THEME_NAME_STORAGE_KEY) === HIGH_CONTRAST_CLASS) {
    _writeStorage(CONTRAST_STORAGE_KEY, 'high');
    _writeStorage(THEME_NAME_STORAGE_KEY, null);
  }
}


export const ThemeProvider: React.FC<ThemeProviderProps> = ({ children }) => {
  const { manifest } = useBranding();
  // Initial values are read synchronously so the first render already matches
  // what the index.html bootstrap script painted; there is no flash.
  const [savedPreference, setSavedPreference] = useState<ModePreference | null>(_savedPreference);
  const [systemMode, setSystemMode] = useState<Mode>(_systemMode);
  const [savedThemeName, setSavedThemeName] = useState<string | null>(() => {
    _migrateLegacyContrast();
    return _readStorage(THEME_NAME_STORAGE_KEY);
  });
  const [highContrast, setHighContrastState] = useState<boolean>(
    () => _readStorage(CONTRAST_STORAGE_KEY) === 'high',
  );

  const modeLocked = manifest.lockMode !== undefined;
  const preference: ModePreference =
    manifest.lockMode ?? savedPreference ?? manifest.defaultMode ?? 'system';
  const theme: Mode = preference === 'system' ? systemMode : preference;

  const availableThemes = manifest.themes;
  const themeName: ThemeName =
    savedThemeName === DEFAULT_THEME
      ? DEFAULT_THEME
      : availableThemes.find((t) => t.id === savedThemeName)?.id ?? manifest.defaultTheme ?? DEFAULT_THEME;

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

  const themeColor = manifest.themeColor[theme] ?? THEME_COLOR[theme];

  useEffect(() => {
    document.documentElement.classList.toggle('dark', theme === 'dark');
    document
      .querySelectorAll('meta[name="theme-color"]')
      .forEach((meta) => meta.setAttribute('content', themeColor));
  }, [theme, themeColor]);

  useEffect(() => {
    // `data-brand-theme` records which theme id is applied as a `theme-<id>`
    // class on <html> (the pre-paint script sets it too), so switching removes
    // exactly that class and never touches classes other scripts may have added.
    const root = document.documentElement;
    const applied = root.dataset.brandTheme;
    if (applied && applied !== themeName) {
      root.classList.remove(themeClass(applied));
    }
    if (themeName === DEFAULT_THEME) {
      delete root.dataset.brandTheme;
    } else {
      root.classList.add(themeClass(themeName));
      root.dataset.brandTheme = themeName;
    }
  }, [themeName]);

  useEffect(() => {
    document.documentElement.classList.toggle(HIGH_CONTRAST_CLASS, highContrast);
  }, [highContrast]);

  const setPreference = useCallback((next: ModePreference) => {
    // Only an explicit mode is persisted, so a system-following session never
    // masquerades as a saved preference.
    const explicit = next === 'system' ? null : next;
    _writeStorage(MODE_STORAGE_KEY, explicit);
    setSavedPreference(explicit);
  }, []);

  const toggleTheme = useCallback(() => {
    setPreference(theme === 'light' ? 'dark' : 'light');
  }, [theme, setPreference]);

  const setThemeName = useCallback(
    (name: ThemeName) => {
      if (name === DEFAULT_THEME || availableThemes.some((t) => t.id === name)) {
        _writeStorage(THEME_NAME_STORAGE_KEY, name);
        setSavedThemeName(name);
      }
    },
    [availableThemes],
  );

  const setHighContrast = useCallback((enabled: boolean) => {
    _writeStorage(CONTRAST_STORAGE_KEY, enabled ? 'high' : null);
    setHighContrastState(enabled);
  }, []);

  const value = useMemo<ThemeContextType>(
    () => ({
      theme,
      toggleTheme,
      preference,
      setPreference,
      modeLocked,
      themeName,
      setThemeName,
      availableThemes,
      highContrast,
      setHighContrast,
    }),
    [theme, toggleTheme, preference, setPreference, modeLocked, themeName, setThemeName, availableThemes, highContrast, setHighContrast],
  );

  return <ThemeContext.Provider value={value}>{children}</ThemeContext.Provider>;
};
