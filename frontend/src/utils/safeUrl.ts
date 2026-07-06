// Shared URL scheme guard for dynamic href / window.open targets.
//
// React does NOT block `javascript:` (or `data:` / `vbscript:`) in an href, so
// any anchor whose href is built from server-registration payloads or from ARD
// federation discovery (untrusted remote peers) can execute script in the
// viewer's authenticated session when clicked. Every dynamic link/window.open
// that renders such a URL MUST route through this guard. When the scheme is not
// allowlisted, callers render the value as plain text (or drop the link)
// instead of a live link.
//
// This is the single hardened URL guard for the frontend — do not reintroduce
// per-component variants; the drift between copies is where the gaps live.

// Only these schemes may become a live link. Everything else (javascript:,
// data:, vbscript:, blob:, file:, unknown, or scheme-relative) is rejected.
const ALLOWED_SCHEMES: ReadonlySet<string> = new Set(['http:', 'https:', 'mailto:']);

// Control characters (including TAB, LF, CR, NUL) can be interleaved inside a
// scheme to evade a naive prefix check — browsers strip many of them before
// dispatching the URL, so `java<TAB>script:alert(1)` still runs. We strip the
// same class of characters before parsing so the guard sees what the browser
// sees: all C0 controls + DEL via \p{Cc}, plus every whitespace char via \s.
const STRIPPED_CHARS = /[\p{Cc}\s]/gu;

/**
 * Returns true only when `url` uses an allowlisted, script-safe scheme.
 *
 * Fails closed: null/undefined/empty, relative or scheme-relative URLs, and any
 * value that cannot be parsed to an explicit allowlisted scheme return false.
 * Whitespace, control characters, and mixed case in the scheme are normalized
 * before the check so obfuscated payloads (e.g. an embedded TAB in `javascript`
 * or a leading-space `  javascript:`) cannot bypass it.
 */
export const isSafeUrl = (url: string | null | undefined): boolean => {
  if (!url) {
    return false;
  }

  // Strip control chars / whitespace the browser would ignore, then match the
  // leading scheme. Relative URLs (no scheme) and scheme-relative URLs (`//x`)
  // deliberately fail — dynamic hrefs here are absolute external targets, and
  // treating a schemeless value as "safe" would let `//evil.example` through.
  const cleaned = url.replace(STRIPPED_CHARS, '');
  const schemeMatch = cleaned.match(/^([a-zA-Z][a-zA-Z0-9+.-]*:)/);
  if (!schemeMatch) {
    return false;
  }

  return ALLOWED_SCHEMES.has(schemeMatch[1].toLowerCase());
};

/**
 * Returns `url` when it is safe to use as a live href/window.open target,
 * otherwise `undefined`. Use in JSX (`href={safeHref(url)}`) so an unsafe value
 * yields no navigable href, or branch on the return to render plain text.
 */
export const safeHref = (url: string | null | undefined): string | undefined =>
  isSafeUrl(url) ? (url as string) : undefined;
