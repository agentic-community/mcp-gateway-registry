import { slugify, pathFromName, pathForRenamed } from '../slug';

describe('slugify', () => {
  it('lowercases and hyphenates', () => {
    expect(slugify('My Cool Server')).toBe('my-cool-server');
  });
  it('collapses non-alphanumeric runs and trims hyphens', () => {
    expect(slugify('  Foo / Bar!! ')).toBe('foo-bar');
  });
});

describe('pathFromName', () => {
  it('returns empty for an empty name', () => {
    expect(pathFromName('')).toBe('');
  });
  it('builds a root path without a prefix', () => {
    expect(pathFromName('My Server')).toBe('/my-server');
  });
  it('builds a prefixed path', () => {
    expect(pathFromName('My Server', 'virtual')).toBe('/virtual/my-server');
  });
});

describe('pathForRenamed', () => {
  it('fills an empty path from the new name', () => {
    expect(pathForRenamed('', '', 'W')).toBe('/w');
  });
  it('keeps following while the path still matches the previous name', () => {
    expect(pathForRenamed('W', '/w', 'We')).toBe('/we');
  });
  it('keeps a path the user changed', () => {
    expect(pathForRenamed('Weather', '/weather-2', 'Weather API')).toBe('/weather-2');
  });
  it('empties the path when a following name is cleared', () => {
    expect(pathForRenamed('W', '/w', '')).toBe('');
  });
  it('follows within a prefix', () => {
    expect(pathForRenamed('Tools', '/virtual/tools', 'Tools Plus', 'virtual')).toBe('/virtual/tools-plus');
  });
});
