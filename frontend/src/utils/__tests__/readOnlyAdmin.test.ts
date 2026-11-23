import { readOnlyAdminDenial, VIEW_ALL_DENIED_PREFIX } from '../readOnlyAdmin';

/** Read-only admin refusals (issue #2099) are recognised only by status and prefix. */
describe('readOnlyAdminDenial', () => {
  const detail = `${VIEW_ALL_DENIED_PREFIX}: connection settings can include stored credentials.`;

  it('returns the detail for a 403 that starts with the prefix', () => {
    expect(readOnlyAdminDenial({ response: { status: 403, data: { detail } } })).toBe(detail);
  });

  it('ignores a 403 with another detail', () => {
    const error = { response: { status: 403, data: { detail: 'CSRF validation failed' } } };
    expect(readOnlyAdminDenial(error)).toBeNull();
  });

  it('ignores the prefix on a status other than 403', () => {
    expect(readOnlyAdminDenial({ response: { status: 404, data: { detail } } })).toBeNull();
  });

  it('handles errors without a response', () => {
    expect(readOnlyAdminDenial(new Error('network'))).toBeNull();
    expect(readOnlyAdminDenial(undefined)).toBeNull();
  });
});
