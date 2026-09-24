import React, { useState } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import ServerEditModal, { ServerEditForm } from '../ServerEditModal';

const baseForm: ServerEditForm = {
  name: 'My Server',
  path: '/my-server',
  proxyPass: 'http://localhost:8080',
  description: 'desc',
  tags: ['a', 'b'],
  license: 'MIT',
  num_tools: 3,
  mcp_endpoint: '',
  metadata: '',
  auth_scheme: 'none',
  auth_credential: '',
  auth_header_name: 'X-API-Key',
  oauth_token_url: '',
  oauth_client_id: '',
  oauth_client_secret: '',
  oauth_scopes: '',
  oauth_token_auth_style: 'post_body',
  oauth_resource: '',
  status: 'active',
  deployment: 'remote',
  local_runtime: {
    type: 'npx',
    package: '',
    version: '',
    image_digest: '',
    argList: [],
    envRows: [],
  },
  custom_headers: [],
  egress_auth_mode: 'none',
  egress_provider: '',
  egress_client_id: '',
  egress_client_secret: '',
  egress_scopes: '',
  egress_custom_authorize_url: '',
  egress_custom_token_url: '',
  egress_custom_token_auth_style: '',
  egress_custom_resource: '',
  egress_target_audience: '',
  oauth_discovery_enabled: false,
  oauth_discovery_provider: '',
  oauth_discovery_client_id: '',
  oauth_discovery_client_secret: '',
  oauth_discovery_scopes: '',
  oauth_discovery_custom_authorize_url: '',
  oauth_discovery_custom_token_url: '',
  oauth_discovery_custom_scope_separator: '',
  oauth_discovery_custom_token_auth_style: '',
  oauth_discovery_custom_resource: '',
};

// Harness that owns the form state so controlled-input edits are observable.
function Harness({
  initial = baseForm,
  onSave = jest.fn(),
  onClose = jest.fn(),
  loading = false,
  egressEnabled = false,
  withGateway = true,
}: {
  initial?: ServerEditForm;
  onSave?: () => void;
  onClose?: () => void;
  loading?: boolean;
  egressEnabled?: boolean;
  withGateway?: boolean;
}) {
  const [form, setForm] = useState<ServerEditForm>(initial);
  return (
    <ServerEditModal
      serverName={form.name}
      form={form}
      setForm={setForm}
      loading={loading}
      egressEnabled={egressEnabled}
      withGateway={withGateway}
      onSave={onSave}
      onClose={onClose}
    />
  );
}

describe('ServerEditModal', () => {
  it('renders the header and pre-fills fields from the form', () => {
    render(<Harness />);
    expect(screen.getByText('Edit Server: My Server')).toBeInTheDocument();
    expect(screen.getByDisplayValue('My Server')).toBeInTheDocument();
    expect(screen.getByDisplayValue('http://localhost:8080')).toBeInTheDocument();
    expect(screen.getByDisplayValue('a,b')).toBeInTheDocument();
  });

  it('edits a controlled field', () => {
    render(<Harness />);
    const nameInput = screen.getByDisplayValue('My Server');
    fireEvent.change(nameInput, { target: { value: 'Renamed' } });
    expect(screen.getByText('Edit Server: Renamed')).toBeInTheDocument();
  });

  it('shows the proxy pass field for remote', () => {
    render(<Harness />);
    expect(screen.getByText('Proxy Pass URL *')).toBeInTheDocument();
  });

  it('hides the proxy pass field for local deployments', () => {
    render(<Harness initial={{ ...baseForm, deployment: 'local' }} />);
    expect(screen.queryByText('Proxy Pass URL *')).not.toBeInTheDocument();
  });

  it('reveals the credential input when an auth scheme is chosen', () => {
    render(<Harness />);
    // No password (credential) input while the scheme is "none".
    expect(
      document.querySelector('input[type="password"]'),
    ).not.toBeInTheDocument();
    fireEvent.change(screen.getByDisplayValue('None'), { target: { value: 'bearer' } });
    expect(document.querySelector('input[type="password"]')).toBeInTheDocument();
  });

  it('calls onSave when the form is submitted', () => {
    const onSave = jest.fn();
    render(<Harness onSave={onSave} />);
    fireEvent.click(screen.getByRole('button', { name: 'Save Changes' }));
    expect(onSave).toHaveBeenCalled();
  });

  it('calls onClose when cancel is clicked', () => {
    const onClose = jest.fn();
    render(<Harness onClose={onClose} />);
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onClose).toHaveBeenCalled();
  });

  it('disables the save button and shows Saving while loading', () => {
    render(<Harness loading />);
    const save = screen.getByRole('button', { name: 'Saving...' });
    expect(save).toBeDisabled();
  });

  it('hides the egress section when the feature is disabled', () => {
    render(<Harness egressEnabled={false} />);
    expect(screen.queryByText('Egress Auth')).not.toBeInTheDocument();
  });

  it('shows the egress section for remote servers when the feature is enabled', () => {
    render(<Harness egressEnabled />);
    expect(screen.getByText('Egress Auth')).toBeInTheDocument();
  });

  it('hides the egress section for local deployments even when enabled', () => {
    render(<Harness initial={{ ...baseForm, deployment: 'local' }} egressEnabled />);
    expect(screen.queryByText('Egress Auth')).not.toBeInTheDocument();
  });

  it('hides the egress section in registry-only mode even when the feature is enabled', () => {
    render(<Harness egressEnabled withGateway={false} />);
    expect(screen.queryByText('Egress Auth')).not.toBeInTheDocument();
  });

  it('hides the discovery-identity block while the flag is off', () => {
    render(<Harness egressEnabled />);
    // The toggle is offered, but the config block stays collapsed.
    expect(
      screen.getByRole('checkbox', { name: /Discovery Identity \(OAuth 2\.1\)/ }),
    ).not.toBeChecked();
    expect(
      screen.queryByRole('heading', { name: 'Discovery Identity (OAuth 2.1)' }),
    ).not.toBeInTheDocument();
  });

  it('reveals the discovery-identity block when the flag is on', () => {
    render(<Harness initial={{ ...baseForm, oauth_discovery_enabled: true }} egressEnabled />);
    expect(
      screen.getByRole('heading', { name: 'Discovery Identity (OAuth 2.1)' }),
    ).toBeInTheDocument();
  });

  it('reveals the discovery-identity block when the toggle is checked', () => {
    render(<Harness egressEnabled />);
    fireEvent.click(
      screen.getByRole('checkbox', { name: /Discovery Identity \(OAuth 2\.1\)/ }),
    );
    expect(
      screen.getByRole('heading', { name: 'Discovery Identity (OAuth 2.1)' }),
    ).toBeInTheDocument();
  });

  it('offers no discovery identity when the egress feature is disabled', () => {
    // Discovery borrows a vaulted per-user token, which the backend only vends
    // when EGRESS_AUTH_ENABLED — so neither the toggle nor the block may show,
    // even for a server that already has discovery enabled.
    render(
      <Harness
        initial={{ ...baseForm, oauth_discovery_enabled: true }}
        egressEnabled={false}
      />,
    );
    expect(
      screen.queryByRole('checkbox', { name: /Discovery Identity \(OAuth 2\.1\)/ }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole('heading', { name: 'Discovery Identity (OAuth 2.1)' }),
    ).not.toBeInTheDocument();
  });

  it('offers no discovery identity in registry-only mode', () => {
    render(
      <Harness
        initial={{ ...baseForm, oauth_discovery_enabled: true }}
        egressEnabled
        withGateway={false}
      />,
    );
    expect(
      screen.queryByRole('checkbox', { name: /Discovery Identity \(OAuth 2\.1\)/ }),
    ).not.toBeInTheDocument();
    expect(
      screen.queryByRole('heading', { name: 'Discovery Identity (OAuth 2.1)' }),
    ).not.toBeInTheDocument();
  });

  it('warns that a discovery identity is inert alongside an explicit scheme', () => {
    // The two are orthogonal in the DATA MODEL -- a server may carry both, and the
    // backend stores both -- but they are NOT both effective. resolve_discovery_bearer
    // bows out for any explicit auth_scheme, because a resolved OAuth bearer
    // short-circuits the header builders and would silently drop the static
    // credential. An earlier revision of this test asserted both simply rendering,
    // which let an operator configure a discovery identity, complete an interactive
    // OAuth consent, and vault a token nothing ever reads. The UI must say so.
    render(
      <Harness
        initial={{ ...baseForm, auth_scheme: 'bearer', oauth_discovery_enabled: true }}
        egressEnabled
      />,
    );
    // Both are still editable -- the config is storable, just not in effect.
    expect(
      screen.getByPlaceholderText('Leave blank to keep current credential'),
    ).toHaveAttribute('type', 'password');
    expect(
      screen.getByRole('heading', { name: 'Discovery Identity (OAuth 2.1)' }),
    ).toBeInTheDocument();
    // ...and the operator is told which one actually wins.
    expect(screen.getByText(/Not in effect/)).toBeInTheDocument();
  });

  it('does not warn when the scheme is none, where discovery does apply', () => {
    render(
      <Harness
        initial={{ ...baseForm, auth_scheme: 'none', oauth_discovery_enabled: true }}
        egressEnabled
      />,
    );
    expect(screen.queryByText(/Not in effect/)).not.toBeInTheDocument();
  });

  it('disables discovery Connect when no provider is selected', () => {
    render(<Harness initial={{ ...baseForm, oauth_discovery_enabled: true }} egressEnabled />);
    expect(
      screen.getByRole('button', { name: 'Connect account for discovery' }),
    ).toBeDisabled();
  });

  it('enables discovery Connect for a saved config but disables it on unsaved edits', () => {
    render(
      <Harness
        initial={{
          ...baseForm,
          oauth_discovery_enabled: true,
          oauth_discovery_provider: 'github',
          oauth_discovery_client_id: 'abc',
        }}
        egressEnabled
      />,
    );
    expect(
      screen.getByRole('button', { name: 'Connect account for discovery' }),
    ).toBeEnabled();
    // Editing a discovery field diverges from the saved snapshot -> Connect
    // disabled until the change is saved (it would otherwise consent against
    // the stale saved client).
    fireEvent.change(screen.getByDisplayValue('abc'), { target: { value: 'xyz' } });
    expect(
      screen.getByRole('button', { name: 'Connect account for discovery' }),
    ).toBeDisabled();
  });

  it('shows the target audience field only in obo_exchange mode', () => {
    render(<Harness initial={{ ...baseForm, egress_auth_mode: 'obo_exchange' }} egressEnabled />);
    expect(screen.getByText('Target Audience')).toBeInTheDocument();
    // 3LO provider picker is hidden in obo_exchange mode.
    expect(screen.queryByText('Provider')).not.toBeInTheDocument();
  });

  it('shows the provider picker in oauth_user mode, not the target audience', () => {
    render(<Harness initial={{ ...baseForm, egress_auth_mode: 'oauth_user' }} egressEnabled />);
    expect(screen.getByText('Provider')).toBeInTheDocument();
    expect(screen.queryByText('Target Audience')).not.toBeInTheDocument();
  });

  it('shows the provider field in pat mode, not the target audience or a header input', () => {
    render(<Harness initial={{ ...baseForm, egress_auth_mode: 'pat' }} egressEnabled />);
    expect(screen.getByText('Provider (vault key)')).toBeInTheDocument();
    // pat inherits the inject header from Backend Authentication: no header inputs here.
    expect(screen.queryByText('Auth header name')).not.toBeInTheDocument();
    expect(screen.queryByText('Value prefix')).not.toBeInTheDocument();
    expect(screen.queryByText('Target Audience')).not.toBeInTheDocument();
    // pat needs only a provider (no client id/secret fields).
    expect(screen.queryByText('Client ID')).not.toBeInTheDocument();
  });

  it('shows token auth style and resource fields for a custom provider', () => {
    render(
      <Harness
        initial={{ ...baseForm, egress_auth_mode: 'oauth_user', egress_provider: 'custom' }}
        egressEnabled
      />,
    );
    expect(screen.getByText('Token Endpoint Authentication')).toBeInTheDocument();
    expect(screen.getByText('Resource (RFC 8707, optional)')).toBeInTheDocument();
    // Confidential styles (default post_body) still show the secret field.
    expect(screen.getByText('Client Secret')).toBeInTheDocument();
  });

  it('hides the client secret field for a custom public client (token auth none)', () => {
    render(
      <Harness
        initial={{
          ...baseForm,
          egress_auth_mode: 'oauth_user',
          egress_provider: 'custom',
          egress_custom_token_auth_style: 'none',
        }}
        egressEnabled
      />,
    );
    expect(screen.queryByText('Client Secret')).not.toBeInTheDocument();
  });

  it('shows neither provider nor target audience when egress mode is none', () => {
    render(<Harness initial={{ ...baseForm, egress_auth_mode: 'none' }} egressEnabled />);
    expect(screen.getByText('Egress Auth')).toBeInTheDocument();
    expect(screen.queryByText('Provider')).not.toBeInTheDocument();
    expect(screen.queryByText('Target Audience')).not.toBeInTheDocument();
  });
});
