import React, { useState } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import SkillFormModal, { SkillForm } from '../SkillFormModal';

const baseForm: SkillForm = {
  name: '',
  description: '',
  skill_md_url: '',
  repository_url: '',
  version: '',
  visibility: 'public',
  tags: '',
  target_agents: '',
  metadata: '',
  status: 'draft',
  auth_scheme: 'none',
  auth_credential: '',
  auth_header_name: '',
  is_proxied: false,
  proxy_target_url: '',
  proxy_client_url: '',
  custom_headers: [],
};

function Harness({
  editing = null,
  initial = baseForm,
  loading = false,
  onSubmit = jest.fn((e: React.FormEvent) => e.preventDefault()),
  onParse = jest.fn(),
  onClose = jest.fn(),
}: {
  editing?: { name: string; path: string } | null;
  initial?: SkillForm;
  loading?: boolean;
  onSubmit?: (e: React.FormEvent) => void;
  onParse?: () => void;
  onClose?: () => void;
}) {
  const [form, setForm] = useState<SkillForm>(initial);
  const [autoFill, setAutoFill] = useState(true);
  return (
    <SkillFormModal
      editing={editing}
      form={form}
      setForm={setForm}
      loading={loading}
      autoFill={autoFill}
      setAutoFill={setAutoFill}
      parseLoading={false}
      onParse={onParse}
      onSubmit={onSubmit}
      onClose={onClose}
    />
  );
}

describe('SkillFormModal', () => {
  it('shows create title, the auto-fill toggle, and the Register button in create mode', () => {
    render(<Harness />);
    expect(screen.getByText('Register New Skill')).toBeInTheDocument();
    expect(screen.getByText('Auto-fill from SKILL.md')).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Register Skill' })).toBeInTheDocument();
  });

  it('shows edit title, locks the name, shows path, and the Save button in edit mode', () => {
    render(
      <Harness
        editing={{ name: 'doc-writer', path: '/skills/doc-writer' }}
        initial={{ ...baseForm, name: 'doc-writer' }}
      />,
    );
    expect(screen.getByText('Edit Skill: doc-writer')).toBeInTheDocument();
    // Name input is disabled in edit mode.
    expect(screen.getByDisplayValue('doc-writer')).toBeDisabled();
    // Path (read-only) is shown.
    expect(screen.getByDisplayValue('/skills/doc-writer')).toBeInTheDocument();
    // Auto-fill toggle is hidden in edit mode.
    expect(screen.queryByText('Auto-fill from SKILL.md')).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Save Changes' })).toBeInTheDocument();
  });

  it('reveals the credential field for bearer/api_key schemes', () => {
    render(<Harness initial={{ ...baseForm, auth_scheme: 'bearer' }} />);
    expect(document.querySelector('input[type="password"]')).toBeInTheDocument();
  });

  it('hides the header-name field for bearer', () => {
    render(<Harness initial={{ ...baseForm, auth_scheme: 'bearer' }} />);
    expect(screen.queryByText('Header Name')).not.toBeInTheDocument();
  });

  it('shows the header-name field for api_key', () => {
    render(<Harness initial={{ ...baseForm, auth_scheme: 'api_key' }} />);
    expect(screen.getByText('Header Name')).toBeInTheDocument();
  });

  it('calls onSubmit when the form is submitted', () => {
    const onSubmit = jest.fn((e: React.FormEvent) => e.preventDefault());
    const { container } = render(<Harness onSubmit={onSubmit} />);
    fireEvent.submit(container.querySelector('form')!);
    expect(onSubmit).toHaveBeenCalled();
  });

  it('calls onClose when cancel is clicked', () => {
    const onClose = jest.fn();
    render(<Harness onClose={onClose} />);
    fireEvent.click(screen.getByRole('button', { name: 'Cancel' }));
    expect(onClose).toHaveBeenCalled();
  });

  it('shows scanning label while loading in create mode', () => {
    render(<Harness loading />);
    expect(
      screen.getByRole('button', { name: 'Registering & Scanning...' }),
    ).toBeDisabled();
  });

  it('renders the upstream-headers editor with per-row labels when proxied', () => {
    render(
      <Harness
        initial={{
          ...baseForm,
          is_proxied: true,
          proxy_target_url: 'https://backend.example.com/',
          custom_headers: [
            { name: 'X-Api-Key', value: 'sk', overridable: false },
            // Fixed Authorization is invalid — the field flags it inline so the
            // operator sees why the Dashboard save will refuse it.
            { name: 'Authorization', value: 'Bearer x', overridable: false },
          ],
        }}
      />,
    );
    // Per-row accessible names are unique.
    expect(screen.getByLabelText('Header name (row 1)')).toHaveValue('X-Api-Key');
    expect(screen.getByLabelText('Header name (row 2)')).toHaveValue('Authorization');
    // The invalid fixed-Authorization row surfaces its inline error.
    expect(screen.getByText(/caller-overridable/)).toBeInTheDocument();
  });
});
