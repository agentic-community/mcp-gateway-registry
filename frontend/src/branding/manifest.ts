/**
 * Deployment branding manifest (`{ROOT_PATH}/branding/manifest.json`).
 *
 * An operator supplies the branding folder at deploy time (docs/ui-branding.md),
 * so everything read here is untrusted input: unknown keys are ignored, and any
 * value that fails validation falls back to the built-in default rather than
 * reaching the DOM.
 */
import { getBaseURL } from '../utils/basePath';

export type ModeChoice = 'light' | 'dark' | 'system';
export type LockedMode = 'light' | 'dark';

export interface BrandTheme {
  id: string;
  label: string;
}

export interface BrandingManifest {
  logo: { light?: string; dark?: string; alt?: string };
  favicon?: string;
  themeColor: { light?: string; dark?: string };
  defaultMode?: ModeChoice;
  lockMode?: LockedMode;
  themes: BrandTheme[];
  defaultTheme?: string;
}

export const EMPTY_MANIFEST: BrandingManifest = { logo: {}, themeColor: {}, themes: [] };

/** Same rule the registry applies before serving a file from the branding folder. */
const FILE_NAME = /^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$/;
const IMAGE_EXTENSION = /\.(svg|png|webp|jpe?g|ico)$/i;
const THEME_ID = /^[a-z][a-z0-9-]{0,31}$/;
const RESERVED_THEME_IDS = ['default', 'light', 'dark', 'high-contrast'];
const HEX_COLOR = /^#[0-9a-fA-F]{6}$/;
const MAX_THEMES = 8;
const MAX_LABEL = 40;
const MAX_ALT = 80;


/** Whether `id` can name a branding theme (and so become a class on <html>). */
export function isValidThemeId(id: unknown): id is string {
  return typeof id === 'string' && THEME_ID.test(id) && !RESERVED_THEME_IDS.includes(id);
}


function _imageFile(value: unknown): string | undefined {
  return typeof value === 'string' && FILE_NAME.test(value) && IMAGE_EXTENSION.test(value)
    ? value
    : undefined;
}


function _boundedText(value: unknown, max: number): string | undefined {
  if (typeof value !== 'string') return undefined;
  const text = value.trim();
  return text && text.length <= max ? text : undefined;
}


function _hexColor(value: unknown): string | undefined {
  return typeof value === 'string' && HEX_COLOR.test(value) ? value : undefined;
}


function _record(value: unknown): Record<string, unknown> {
  return value !== null && typeof value === 'object' && !Array.isArray(value)
    ? (value as Record<string, unknown>)
    : {};
}


function _themes(value: unknown): BrandTheme[] {
  if (!Array.isArray(value)) return [];
  const themes: BrandTheme[] = [];
  for (const entry of value) {
    const theme = _record(entry);
    const label = _boundedText(theme.label, MAX_LABEL);
    if (isValidThemeId(theme.id) && label && !themes.some((t) => t.id === theme.id)) {
      themes.push({ id: theme.id, label });
    }
    if (themes.length === MAX_THEMES) break;
  }
  return themes;
}


/**
 * Validate a parsed manifest. Never throws: invalid keys are dropped and a
 * warning names them so an operator can see why a setting had no effect.
 */
export function parseBrandingManifest(raw: unknown): BrandingManifest {
  const source = _record(raw);
  const ignored: string[] = [];
  // Run a validator; when a key was present but did not validate, record it.
  const take = <T>(key: string, present: unknown, value: T | undefined): T | undefined => {
    if (present !== undefined && present !== null && value === undefined) ignored.push(key);
    return value;
  };

  const logo = _record(source.logo);
  const themeColor = _record(source.themeColor);
  const themes = _themes(source.themes);
  if (source.themes !== undefined && (!Array.isArray(source.themes) || source.themes.length !== themes.length)) {
    ignored.push('themes');
  }

  const manifest: BrandingManifest = {
    logo: {
      light: take('logo.light', logo.light, _imageFile(logo.light)),
      dark: take('logo.dark', logo.dark, _imageFile(logo.dark)),
      alt: take('logo.alt', logo.alt, _boundedText(logo.alt, MAX_ALT)),
    },
    favicon: take('favicon', source.favicon, _imageFile(source.favicon)),
    themeColor: {
      light: take('themeColor.light', themeColor.light, _hexColor(themeColor.light)),
      dark: take('themeColor.dark', themeColor.dark, _hexColor(themeColor.dark)),
    },
    defaultMode: take(
      'defaultMode',
      source.defaultMode,
      ['light', 'dark', 'system'].includes(source.defaultMode as string)
        ? (source.defaultMode as ModeChoice)
        : undefined,
    ),
    lockMode: take(
      'lockMode',
      source.lockMode,
      ['light', 'dark'].includes(source.lockMode as string) ? (source.lockMode as LockedMode) : undefined,
    ),
    themes,
    defaultTheme:
      source.defaultTheme === 'default'
        ? undefined
        : take(
            'defaultTheme',
            source.defaultTheme,
            themes.find((t) => t.id === source.defaultTheme)?.id,
          ),
  };

  const known = ['version', 'logo', 'favicon', 'themeColor', 'defaultMode', 'lockMode', 'themes', 'defaultTheme'];
  ignored.push(...Object.keys(source).filter((key) => !known.includes(key)));
  if (ignored.length > 0) {
    console.warn(`Branding manifest: ignored invalid or unknown keys: ${ignored.join(', ')}`);
  }
  return manifest;
}


/** URL of a file in the deployment's branding folder, honoring ROOT_PATH. */
export function brandingAssetUrl(name: string): string {
  return `${getBaseURL()}/branding/${encodeURIComponent(name)}`;
}
