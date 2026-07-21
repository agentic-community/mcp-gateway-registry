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

function Harness({ initial = [] as UpstreamHeader[] }) {
  const [headers, setHeaders] = useState<UpstreamHeader[]>(initial);
  return <UpstreamHeadersField headers={headers} onChange={setHeaders} />;
}

describe('UpstreamHeadersField', () => {
  it('adds a header row on "Add header"', () => {
    render(<Harness />);
    fireEvent.click(screen.getByText('+ Add header'));
    expect(screen.getByLabelText('Header name')).toBeInTheDocument();
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
    expect(screen.getByLabelText('Header name')).toBeInTheDocument();
    fireEvent.click(screen.getByText('Remove'));
    expect(screen.queryByLabelText('Header name')).not.toBeInTheDocument();
  });

  it('toggles the overridable flag', () => {
    render(<Harness initial={[{ name: 'X-A', value: '', overridable: false }]} />);
    // Fixed + no value = error initially.
    expect(screen.getByText(/fixed header needs a value/)).toBeInTheDocument();
    fireEvent.click(screen.getByRole('checkbox'));
    // Now a caller-only slot = valid, error gone.
    expect(screen.queryByText(/fixed header needs a value/)).not.toBeInTheDocument();
  });
});
