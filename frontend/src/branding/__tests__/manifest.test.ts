import { parseBrandingManifest, isValidThemeId } from '../manifest';

describe('parseBrandingManifest', () => {
  let warn: jest.SpyInstance;
  beforeEach(() => {
    warn = jest.spyOn(console, 'warn').mockImplementation(() => undefined);
  });
  afterEach(() => warn.mockRestore());

  it('keeps a fully valid manifest', () => {
    const manifest = parseBrandingManifest({
      version: 1,
      logo: { light: 'logo-light.svg', dark: 'logo-dark.png', alt: 'Acme' },
      favicon: 'favicon.ico',
      themeColor: { light: '#ffffff', dark: '#101828' },
      defaultMode: 'light',
      lockMode: 'dark',
      themes: [{ id: 'acme', label: 'Acme' }],
      defaultTheme: 'acme',
    });
    expect(manifest).toEqual({
      logo: { light: 'logo-light.svg', dark: 'logo-dark.png', alt: 'Acme' },
      favicon: 'favicon.ico',
      themeColor: { light: '#ffffff', dark: '#101828' },
      defaultMode: 'light',
      lockMode: 'dark',
      themes: [{ id: 'acme', label: 'Acme' }],
      defaultTheme: 'acme',
    });
    expect(warn).not.toHaveBeenCalled();
  });

  it.each([
    ['a URL', 'https://evil.example/logo.svg'],
    ['a path', 'img/logo.svg'],
    ['a traversal', '../logo.svg'],
    ['a dotfile', '.logo.svg'],
    ['a non-image', 'logo.js'],
    ['a javascript: URL', 'javascript:alert(1)'],
  ])('drops a logo given as %s', (_label, value) => {
    const manifest = parseBrandingManifest({ logo: { light: value } });
    expect(manifest.logo.light).toBeUndefined();
    expect(warn).toHaveBeenCalledWith(expect.stringContaining('logo.light'));
  });

  it('rejects reserved and malformed theme ids and caps the list', () => {
    const themes = [
      { id: 'dark', label: 'Dark' },
      { id: 'high-contrast', label: 'HC' },
      { id: 'Acme', label: 'Upper' },
      { id: 'acme', label: 'Acme' },
      { id: 'acme', label: 'Duplicate' },
      ...Array.from({ length: 10 }, (_, i) => ({ id: `t${i}`, label: `T${i}` })),
    ];
    const manifest = parseBrandingManifest({ themes });
    expect(manifest.themes[0]).toEqual({ id: 'acme', label: 'Acme' });
    expect(manifest.themes).toHaveLength(8);
    expect(manifest.themes.map((t) => t.id)).not.toContain('dark');
  });

  it('ignores a defaultTheme that is not in the theme list', () => {
    const manifest = parseBrandingManifest({ themes: [{ id: 'acme', label: 'Acme' }], defaultTheme: 'other' });
    expect(manifest.defaultTheme).toBeUndefined();
  });

  it('treats a non-object body (for example an HTML error page) as no branding', () => {
    expect(parseBrandingManifest('<html>oops</html>')).toEqual({ logo: {}, themeColor: {}, themes: [] });
  });

  it('accepts lockMode null and defaultTheme default without warning', () => {
    parseBrandingManifest({ lockMode: null, defaultTheme: 'default' });
    expect(warn).not.toHaveBeenCalled();
  });

  it('drops invalid colors and modes and names them', () => {
    const manifest = parseBrandingManifest({ themeColor: { light: 'red' }, defaultMode: 'auto', extra: 1 });
    expect(manifest.themeColor.light).toBeUndefined();
    expect(manifest.defaultMode).toBeUndefined();
    expect(warn).toHaveBeenCalledWith(expect.stringMatching(/themeColor\.light.*defaultMode.*extra/));
  });
});

describe('isValidThemeId', () => {
  it.each(['acme', 'acme-2', 'a'])('accepts %s', (id) => expect(isValidThemeId(id)).toBe(true));
  it.each(['default', 'light', 'dark', 'high-contrast', '1acme', 'Acme', 'a b', '', 'a'.repeat(33)])(
    'rejects %s',
    (id) => expect(isValidThemeId(id)).toBe(false),
  );
});
