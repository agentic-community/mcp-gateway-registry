import {
  buildEgressAuthConfigPayload,
  isGatewayManagedEgress,
  normalizeEgressAuthMode,
  type EgressAuthConfigForm,
} from '../egressAuth';

const baseForm: EgressAuthConfigForm = {
  egress_auth_mode: 'none',
  egress_provider: ' github ',
  egress_client_id: ' client-id ',
  egress_client_secret: '',
  egress_scopes: 'read, write',
  egress_custom_authorize_url: '',
  egress_custom_token_url: '',
  egress_custom_token_auth_style: '',
  egress_custom_resource: '',
  egress_target_audience: ' api://backend ',
};

describe('egress auth config helpers', () => {
  test('normalizes unknown modes to none', () => {
    expect(normalizeEgressAuthMode('operator_credential')).toBe('operator_credential');
    expect(normalizeEgressAuthMode('unknown')).toBe('none');
    expect(normalizeEgressAuthMode(null)).toBe('none');
  });

  test('treats only explicit none as unmanaged egress', () => {
    expect(isGatewayManagedEgress('none')).toBe(false);
    expect(isGatewayManagedEgress('')).toBe(false);
    expect(isGatewayManagedEgress(null)).toBe(false);
    expect(isGatewayManagedEgress(undefined)).toBe(false);
    expect(isGatewayManagedEgress('operator_credential')).toBe(true);
    expect(isGatewayManagedEgress('oauth_user')).toBe(true);
    expect(isGatewayManagedEgress('future_mode')).toBe(true);
  });

  test('builds the minimal operator credential payload', () => {
    expect(
      buildEgressAuthConfigPayload({
        ...baseForm,
        egress_auth_mode: 'operator_credential',
      })
    ).toEqual({ egress_auth_mode: 'operator_credential' });
  });

  test('builds the OBO payload and normalizes scopes', () => {
    expect(
      buildEgressAuthConfigPayload({
        ...baseForm,
        egress_auth_mode: 'obo_exchange',
      })
    ).toEqual({
      egress_auth_mode: 'obo_exchange',
      target_audience: 'api://backend',
      scopes: ['read', 'write'],
    });
  });

  test('builds the OAuth payload without an empty secret', () => {
    expect(
      buildEgressAuthConfigPayload({
        ...baseForm,
        egress_auth_mode: 'oauth_user',
      })
    ).toEqual({
      egress_auth_mode: 'oauth_user',
      egress_provider: 'github',
      client_id: 'client-id',
      client_secret: undefined,
      scopes: ['read', 'write'],
      custom_authorize_url: undefined,
      custom_token_url: undefined,
      custom_token_auth_style: undefined,
      custom_resource: undefined,
    });
  });
});
