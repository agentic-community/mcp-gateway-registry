import { gatewayClientUrl } from '../gatewayClientUrl';

describe('gatewayClientUrl', () => {
  it('returns an absolute same-origin URL with a trailing slash', () => {
    // The nginx location for a proxied entity ends in "/", so the bare path 301s.
    // A URL we hand a user to copy must already carry the slash.
    expect(gatewayClientUrl('/gateway/skill/pdf')).toBe(
      `${window.location.origin}/gateway/skill/pdf/`,
    );
  });

  it('never doubles an existing trailing slash', () => {
    expect(gatewayClientUrl('/gateway/skill/pdf/')).toBe(
      `${window.location.origin}/gateway/skill/pdf/`,
    );
    expect(gatewayClientUrl('/gateway/skill/pdf///')).toBe(
      `${window.location.origin}/gateway/skill/pdf/`,
    );
  });

  it('returns an empty string when the entity is not proxied', () => {
    // No client path exists unless the server derived one, so callers can treat
    // "" as "nothing to show" rather than rendering a bare origin.
    expect(gatewayClientUrl(null)).toBe('');
    expect(gatewayClientUrl(undefined)).toBe('');
    expect(gatewayClientUrl('')).toBe('');
  });
});
