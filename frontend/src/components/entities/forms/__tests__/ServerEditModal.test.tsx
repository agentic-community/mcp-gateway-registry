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
  egress_provider: '',
  egress_client_id: '',
  egress_client_secret: '',
  egress_scopes: '',
  egress_custom_authorize_url: '',
  egress_custom_token_url: '',
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
    expect(screen.queryByText('Per-User Egress Auth (OAuth)')).not.toBeInTheDocument();
  });

  it('shows the egress section for remote servers when the feature is enabled', () => {
    render(<Harness egressEnabled />);
    expect(screen.getByText('Per-User Egress Auth (OAuth)')).toBeInTheDocument();
  });

  it('hides the egress section for local deployments even when enabled', () => {
    render(<Harness initial={{ ...baseForm, deployment: 'local' }} egressEnabled />);
    expect(screen.queryByText('Per-User Egress Auth (OAuth)')).not.toBeInTheDocument();
  });
});
