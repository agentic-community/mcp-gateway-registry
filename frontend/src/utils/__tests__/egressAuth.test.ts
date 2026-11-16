import {
  buildEgressConfigBody,
  isDcrEgressProvider,
  type EgressConfigFormFields,
} from '../egressAuth';

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
  egress_re_register: false,
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

  describe('dynamic client registration provider (atlassian-rovo)', () => {
    const rovo: EgressConfigFormFields = {
      ...form,
      egress_auth_mode: 'oauth_user',
      egress_provider: 'atlassian-rovo',
      egress_client_id: '',
      egress_client_secret: '',
      egress_scopes: '',
    };

    it('recognizes the provider the gateway registers a client for', () => {
      expect(isDcrEgressProvider('atlassian-rovo')).toBe(true);
      expect(isDcrEgressProvider(' atlassian-rovo ')).toBe(true);
      expect(isDcrEgressProvider('atlassian')).toBe(false);
      expect(isDcrEgressProvider('custom')).toBe(false);
    });

    it('omits client_id and client_secret when blank, so the gateway registers a client', () => {
      const body = buildEgressConfigBody(rovo);
      expect(body.egress_provider).toBe('atlassian-rovo');
      expect(body.client_id).toBeUndefined();
      expect(body.client_secret).toBeUndefined();
      expect(body.scopes).toEqual([]);
    });

    it('sends the stored client_id back, which the backend treats as no client_id', () => {
      const body = buildEgressConfigBody({ ...rovo, egress_client_id: ' stored-id ' });
      expect(body.client_id).toBe('stored-id');
    });

    it('never sends a client secret, even one left in the form state', () => {
      const body = buildEgressConfigBody({ ...rovo, egress_client_secret: 'stale-secret' });
      expect(body.client_secret).toBeUndefined();
    });

    it('sends re_register only when asked', () => {
      expect(buildEgressConfigBody(rovo).re_register).toBeUndefined();
      expect(buildEgressConfigBody({ ...rovo, egress_re_register: true }).re_register).toBe(true);
    });

    it('drops a typed or stored client_id when re_register is set', () => {
      const body = buildEgressConfigBody({
        ...rovo,
        egress_client_id: ' typed-id ',
        egress_re_register: true,
      });
      expect(body.client_id).toBeUndefined();
      expect(body.re_register).toBe(true);
    });

    it('does not send re_register for a provider that does not register a client', () => {
      const body = buildEgressConfigBody({
        ...form,
        egress_auth_mode: 'oauth_user',
        egress_re_register: true,
      });
      expect(body).not.toHaveProperty('re_register');
    });

    it('still sends a blank client_id as an empty string for the classic atlassian provider', () => {
      const body = buildEgressConfigBody({
        ...rovo,
        egress_provider: 'atlassian',
        egress_client_id: ' cid ',
        egress_client_secret: 's3cret',
      });
      expect(body).toMatchObject({ client_id: 'cid', client_secret: 's3cret' });
    });
  });
});
