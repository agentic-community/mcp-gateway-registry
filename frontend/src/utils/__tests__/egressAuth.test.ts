import { buildEgressConfigBody, type EgressConfigFormFields } from '../egressAuth';

const form: EgressConfigFormFields = {
  egress_auth_mode: 'none',
  egress_provider: ' github ',
  egress_client_id: ' cid ',
  egress_client_secret: '',
  egress_scopes: 'repo, read:org',
  egress_custom_authorize_url: '',
  egress_custom_token_url: '',
  egress_custom_token_auth_style: '',
  egress_custom_resource: '',
  egress_expected_issuer: '',
  egress_target_audience: ' api://internal ',
  egress_list_tools_before_consent: true,
};

describe('buildEgressConfigBody', () => {
  it.each(['oauth_user', 'pat'] as const)(
    'sends list_tools_before_consent in %s mode',
    (mode) => {
      const body = buildEgressConfigBody({ ...form, egress_auth_mode: mode });
      expect(body.list_tools_before_consent).toBe(true);
    }
  );

  it('sends false explicitly, because the request replaces the whole config', () => {
    const body = buildEgressConfigBody({
      ...form,
      egress_auth_mode: 'pat',
      egress_list_tools_before_consent: false,
    });
    expect(body.list_tools_before_consent).toBe(false);
  });

  it.each(['none', 'obo_exchange'] as const)(
    'omits list_tools_before_consent in %s mode, so the server stores false',
    (mode) => {
      const body = buildEgressConfigBody({ ...form, egress_auth_mode: mode });
      expect(body).not.toHaveProperty('list_tools_before_consent');
    }
  );

  it('builds the oauth_user body with trimmed values and split scopes', () => {
    const body = buildEgressConfigBody({ ...form, egress_auth_mode: 'oauth_user' });
    expect(body).toMatchObject({
      egress_auth_mode: 'oauth_user',
      egress_provider: 'github',
      client_id: 'cid',
      client_secret: undefined,
      scopes: ['repo', 'read:org'],
    });
  });

  it('builds the pat body with the provider only', () => {
    const body = buildEgressConfigBody({ ...form, egress_auth_mode: 'pat' });
    expect(body).toEqual({
      egress_auth_mode: 'pat',
      egress_provider: 'github',
      list_tools_before_consent: true,
    });
  });

  it('builds the obo_exchange body', () => {
    const body = buildEgressConfigBody({ ...form, egress_auth_mode: 'obo_exchange' });
    expect(body).toEqual({
      egress_auth_mode: 'obo_exchange',
      target_audience: 'api://internal',
      scopes: ['repo', 'read:org'],
    });
  });

  it('builds the disabled body', () => {
    expect(buildEgressConfigBody(form)).toEqual({ egress_auth_mode: 'none' });
  });
});
