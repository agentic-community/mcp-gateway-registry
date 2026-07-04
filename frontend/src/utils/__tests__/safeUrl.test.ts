import { isSafeUrl, safeHref } from '../safeUrl';

describe('isSafeUrl', () => {
  it('allows http, https, and mailto', () => {
    expect(isSafeUrl('http://example.com')).toBe(true);
    expect(isSafeUrl('https://example.com/path?q=1')).toBe(true);
    expect(isSafeUrl('HTTPS://EXAMPLE.COM')).toBe(true);
    expect(isSafeUrl('mailto:user@example.com')).toBe(true);
  });

  it('rejects empty / nullish values (fail closed)', () => {
    expect(isSafeUrl('')).toBe(false);
    expect(isSafeUrl(null)).toBe(false);
    expect(isSafeUrl(undefined)).toBe(false);
  });

  it('rejects javascript: URIs', () => {
    expect(isSafeUrl('javascript:alert(1)')).toBe(false);
    expect(isSafeUrl('JavaScript:alert(document.cookie)')).toBe(false);
    expect(isSafeUrl("javascript:fetch('/api/x',{method:'DELETE'})")).toBe(false);
  });

  it('rejects data:, vbscript:, blob:, and file: URIs', () => {
    expect(isSafeUrl('data:text/html,<script>alert(1)</script>')).toBe(false);
    expect(isSafeUrl('vbscript:msgbox(1)')).toBe(false);
    expect(isSafeUrl('blob:https://example.com/uuid')).toBe(false);
    expect(isSafeUrl('file:///etc/passwd')).toBe(false);
  });

  it('rejects obfuscation via leading whitespace and control chars', () => {
    expect(isSafeUrl('  javascript:alert(1)')).toBe(false);
    expect(isSafeUrl('\t javascript:alert(1)')).toBe(false);
    // Embedded TAB / newline / carriage-return inside the scheme keyword.
    expect(isSafeUrl('java\tscript:alert(1)')).toBe(false);
    expect(isSafeUrl('java\nscript:alert(1)')).toBe(false);
    expect(isSafeUrl('java\rscript:alert(1)')).toBe(false);
    // Embedded NUL byte.
    expect(isSafeUrl('java\x00script:alert(1)')).toBe(false);
    // Leading control character before the scheme.
    expect(isSafeUrl('\x01javascript:alert(1)')).toBe(false);
  });

  it('rejects relative and scheme-relative URLs (no explicit safe scheme)', () => {
    expect(isSafeUrl('/relative/path')).toBe(false);
    expect(isSafeUrl('//evil.example/path')).toBe(false);
    expect(isSafeUrl('example.com/no-scheme')).toBe(false);
  });
});

describe('safeHref', () => {
  it('returns the URL for safe schemes', () => {
    expect(safeHref('https://example.com')).toBe('https://example.com');
  });

  it('returns undefined for unsafe schemes so no navigable href is set', () => {
    expect(safeHref('javascript:alert(1)')).toBeUndefined();
    expect(safeHref('java\tscript:alert(1)')).toBeUndefined();
    expect(safeHref(null)).toBeUndefined();
    expect(safeHref('')).toBeUndefined();
  });
});
