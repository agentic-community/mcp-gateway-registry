import React, { useState } from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import UpstreamHeadersField, {
  upstreamHeaderRowError,
  type UpstreamHeader,
} from '../UpstreamHeadersField';

describe('upstreamHeaderRowError', () => {
  it('accepts a fixed header with a value', () => {
    expect(
      upstreamHeaderRowError({ name: 'X-Api-Key', value: 'sk', overridable: false }),
    ).toBeNull();
  });

  it('accepts an overridable default (value + overridable)', () => {
    expect(
      upstreamHeaderRowError({ name: 'X-Tenant', value: 'acme', overridable: true }),
    ).toBeNull();
  });

  it('accepts a caller-only slot (no value + overridable)', () => {
    expect(
      upstreamHeaderRowError({ name: 'X-Tenant', value: '', overridable: true }),
    ).toBeNull();
  });

  it('rejects a fixed header with no value', () => {
    expect(
      upstreamHeaderRowError({ name: 'X-Tenant', value: '', overridable: false }),
    ).toMatch(/fixed header needs a value/);
  });

  it('treats a fully-blank row as ignorable (no error)', () => {
    expect(upstreamHeaderRowError({ name: '', value: '', overridable: false })).toBeNull();
  });

  it('flags a value-only row with no name', () => {
    expect(
      upstreamHeaderRowError({ name: '', value: 'sk', overridable: false }),
    ).toMatch(/name is required/);
  });

  it('rejects a reserved gateway/internal header name in any form', () => {
    // Covers ingress creds, identity headers, all internal-token variants, and
    // the generic-proxy markers / routing headers (parity with the backend
    // RESERVED_CUSTOM_HEADER_NAMES so the client hint matches the 400).
    for (const name of [
      'Cookie',
      'X-Authorization',
      'X-Internal-Token',
      'X-Internal-Token-Generic',
      'Host',
      'X-User',
      'X-Scopes',
      'X-Generic-Proxy-Kind',
      'X-Generic-Has-Upstream-Auth',
      'X-Resolved-Generic-Upstream',
      'X-Upstream-Url',
      'X-Body',
    ]) {
      expect(
        upstreamHeaderRowError({ name, value: 'x', overridable: false }),
      ).toMatch(/managed by the gateway/);
      expect(
        upstreamHeaderRowError({ name, value: '', overridable: true }),
      ).toMatch(/managed by the gateway/);
    }
  });

  it('rejects a fixed Authorization but allows an overridable one', () => {
    expect(
      upstreamHeaderRowError({ name: 'Authorization', value: 'Bearer x', overridable: false }),
    ).toMatch(/caller-overridable/);
    expect(
      upstreamHeaderRowError({ name: 'authorization', value: 'Bearer x', overridable: true }),
    ).toBeNull();
    // Caller-only Authorization slot (no default) is fine.
    expect(
      upstreamHeaderRowError({ name: 'Authorization', value: '', overridable: true }),
    ).toBeNull();
  });
});

function Harness({
  initial = [] as UpstreamHeader[],
  editMode = false,
}: {
  initial?: UpstreamHeader[];
  editMode?: boolean;
}) {
  const [headers, setHeaders] = useState<UpstreamHeader[]>(initial);
  return (
    <UpstreamHeadersField headers={headers} onChange={setHeaders} editMode={editMode} />
  );
}

describe('UpstreamHeadersField', () => {
  it('adds a header row on "Add header"', () => {
    render(<Harness />);
    fireEvent.click(screen.getByText('+ Add header'));
    expect(screen.getByLabelText('Header name (row 1)')).toBeInTheDocument();
  });

  it('shows the inline error for a fixed header with no value', () => {
    render(
      <Harness initial={[{ name: 'X-Tenant', value: '', overridable: false }]} />,
    );
    expect(screen.getByText(/fixed header needs a value/)).toBeInTheDocument();
  });

  it('renders the Authorization warning when it is overridable', () => {
    render(
      <Harness initial={[{ name: 'Authorization', value: '', overridable: true }]} />,
    );
    expect(screen.getByText(/equal-token guard/)).toBeInTheDocument();
  });

  it('removes a row on "Remove"', () => {
    render(<Harness initial={[{ name: 'X-A', value: 'v', overridable: false }]} />);
    expect(screen.getByLabelText('Header name (row 1)')).toBeInTheDocument();
    fireEvent.click(screen.getByLabelText('Remove header (row 1)'));
    expect(screen.queryByLabelText('Header name (row 1)')).not.toBeInTheDocument();
  });

  it('gives each row a unique accessible name', () => {
    render(
      <Harness
        initial={[
          { name: 'X-A', value: 'a', overridable: false },
          { name: 'X-B', value: 'b', overridable: false },
        ]}
      />,
    );
    expect(screen.getByLabelText('Header name (row 1)')).toBeInTheDocument();
    expect(screen.getByLabelText('Header name (row 2)')).toBeInTheDocument();
    expect(screen.getByLabelText('Header value (row 1)')).toBeInTheDocument();
    expect(screen.getByLabelText('Header value (row 2)')).toBeInTheDocument();
    expect(screen.getByLabelText('Remove header (row 1)')).toBeInTheDocument();
    expect(screen.getByLabelText('Remove header (row 2)')).toBeInTheDocument();
  });

  it('toggles the overridable flag', () => {
    render(<Harness initial={[{ name: 'X-A', value: '', overridable: false }]} />);
    // Fixed + no value = error initially.
    expect(screen.getByText(/fixed header needs a value/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('checkbox'));
    // Now a caller-only slot = valid, error gone.
    expect(screen.queryByText(/fixed header needs a value/)).not.toBeInTheDocument();
  });

  it('renders edit-mode rows rebuilt from names with blank (write-only) values', () => {
    // On edit the parent maps custom_header_names -> rows with value:'' and the
    // overridable flag from custom_header_overridable_names. The field must show
    // the name, keep the value blank, and hint that blank keeps the stored value.
    render(
      <Harness
        editMode
        initial={[
          { name: 'X-Api-Key', value: '', overridable: false },
          { name: 'X-Tenant', value: '', overridable: true },
        ]}
      />,
    );
    // Names are populated from the mapping...
    expect(screen.getByLabelText('Header name (row 1)')).toHaveValue('X-Api-Key');
    expect(screen.getByLabelText('Header name (row 2)')).toHaveValue('X-Tenant');
    // ...values stay blank (never echoed back), with the keep-stored placeholder.
    const value1 = screen.getByLabelText('Header value (row 1)');
    expect(value1).toHaveValue('');
    expect(value1).toHaveAttribute('placeholder', 'Blank = keep stored value');
    // The overridable flag maps through per row.
    const [override1, override2] = screen.getAllByRole('checkbox');
    expect(override1).not.toBeChecked();
    expect(override2).toBeChecked();
    // A blank write-only value on edit is NOT an error (keep-stored, no gating).
    expect(screen.queryByText(/fixed header needs a value/)).not.toBeInTheDocument();
  });
});
