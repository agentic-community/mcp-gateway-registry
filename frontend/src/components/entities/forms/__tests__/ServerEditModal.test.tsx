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
  egress_target_audience: '',
};

// Harness that owns the form state so controlled-input edits are observable.
function Harness({
  initial = baseForm,
  onSave = jest.fn(),
  onClose = jest.fn(),
  loading = false,
  egressEnabled = false,
}: {
  initial?: ServerEditForm;
  onSave?: () => void;
  onClose?: () => void;
  loading?: boolean;
  egressEnabled?: boolean;
}) {
  const [form, setForm] = useState<ServerEditForm>(initial);
  return (
    <ServerEditModal
      serverName={form.name}
      form={form}
      setForm={setForm}
      loading={loading}
      egressEnabled={egressEnabled}
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

  it('shows neither provider nor target audience when egress mode is none', () => {
    render(<Harness initial={{ ...baseForm, egress_auth_mode: 'none' }} egressEnabled />);
    expect(screen.getByText('Egress Auth')).toBeInTheDocument();
    expect(screen.queryByText('Provider')).not.toBeInTheDocument();
    expect(screen.queryByText('Target Audience')).not.toBeInTheDocument();
  });
});
