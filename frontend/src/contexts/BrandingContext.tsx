import React, { createContext, useContext, useEffect, useState, ReactNode } from 'react';
import axios from 'axios';
import {
  BrandingManifest,
  EMPTY_MANIFEST,
  brandingAssetUrl,
  parseBrandingManifest,
} from '../branding/manifest';

/**
 * localStorage key holding the last validated manifest. The pre-paint script in
 * index.html reads `defaultMode`, `lockMode` and `defaultTheme` from it so a
 * returning visitor gets the branded mode and theme before first paint; change
 * both together.
 */
export const BRANDING_CACHE_KEY = 'branding';

interface BrandingContextType {
  manifest: BrandingManifest;
  /** True once the deployment's manifest has been fetched (or failed to). */
  loaded: boolean;
}

const BrandingContext = createContext<BrandingContextType>({ manifest: EMPTY_MANIFEST, loaded: false });

export const useBranding = () => useContext(BrandingContext);


function _cachedManifest(): BrandingManifest {
  try {
    const raw = window.localStorage.getItem(BRANDING_CACHE_KEY);
    // Re-validate: localStorage is writable by any script on the origin.
    return raw ? parseBrandingManifest(JSON.parse(raw)) : EMPTY_MANIFEST;
  } catch {
    return EMPTY_MANIFEST;
  }
}


function _storeManifest(manifest: BrandingManifest): void {
  try {
    window.localStorage.setItem(BRANDING_CACHE_KEY, JSON.stringify(manifest));
  } catch {
    // Non-persistent session; the in-memory manifest still applies.
  }
}


/** Point every icon link at the branded favicon (index.html ships the default). */
function _applyFavicon(name: string): void {
  const href = brandingAssetUrl(name);
  document.querySelectorAll<HTMLLinkElement>('link[rel="icon"]').forEach((link) => {
    link.href = href;
    link.removeAttribute('sizes');
    link.removeAttribute('type');
  });
}


/**
 * Loads `{ROOT_PATH}/branding/manifest.json` once. The registry answers `{}`
 * when no branding is configured, so an unbranded deployment renders the
 * defaults after one small request.
 */
export const BrandingProvider: React.FC<{ children: ReactNode }> = ({ children }) => {
  const [state, setState] = useState<BrandingContextType>(() => ({
    manifest: _cachedManifest(),
    loaded: false,
  }));

  useEffect(() => {
    let active = true;
    axios
      .get<unknown>('/branding/manifest.json', { withCredentials: false })
      .then((response) => {
        if (!active) return;
        const manifest = parseBrandingManifest(response.data);
        _storeManifest(manifest);
        setState({ manifest, loaded: true });
      })
      .catch((error: unknown) => {
        console.warn('Branding manifest unavailable; using the default look.', error);
        if (active) setState((prev) => ({ ...prev, loaded: true }));
      });
    return () => {
      active = false;
    };
  }, []);

  useEffect(() => {
    if (state.manifest.favicon) {
      _applyFavicon(state.manifest.favicon);
    }
  }, [state.manifest.favicon]);

  return <BrandingContext.Provider value={state}>{children}</BrandingContext.Provider>;
};
