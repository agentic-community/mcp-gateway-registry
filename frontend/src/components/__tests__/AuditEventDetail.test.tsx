import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import AuditEventDetail from '../AuditEventDetail';
import { AuditEvent } from '../AuditLogTable';

// The durable identity claims reach the browser on two record
// shapes: nested under `identity` for the registry_api / mcp_access streams, and
// flat at top level for token_mint (which has no `identity` block). The panel
// must present them in both cases, and must stay visually unchanged for the
// majority of records that carry none.
const baseIdentity: NonNullable<AuditEvent['identity']> = {
  username: 'alice@example.com',
  auth_method: 'entra',
  credential_type: 'bearer_token',
  is_admin: false,
};

const baseEvent: AuditEvent = {
  timestamp: '2026-09-14T12:00:00Z',
  request_id: 'req-1',
  log_type: 'mcp_server_access',
  identity: baseIdentity,
};

// Synthetic values that keep the SHAPE of real Entra claims: `oid`/`tid`/`appid`
// are GUIDs (36 chars) and `canonical_id` is `oid@tid` (73 chars). The shared
// prefix matters: truncated to a detail-panel column these two collapse to the
// same string, which is the defect this panel has to avoid.
const OBJECT_ID = '00000000-0000-4000-8000-0000000000ab';
const TENANT_ID = '11111111-1111-4111-8111-111111111111';
const APP_ID = '22222222-2222-4222-8222-222222222222';

const CLAIMS = {
  subject: 'kAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA',
  canonical_id: `${OBJECT_ID}@${TENANT_ID}`,
  principal_name: 'alice@example.com',
  object_id: OBJECT_ID,
  tenant_id: TENANT_ID,
  app_id: APP_ID,
};

const CLAIM_LABELS = ['Principal', 'Canonical ID', 'Subject', 'Object ID', 'Tenant ID', 'App ID'];

/** The <dd> value element for a claim row, located from its <dt> label. */
const claimValue = (label: string): HTMLElement => {
  const cell = screen.getByText(label).closest('div') as HTMLElement;
  return cell.querySelector('dd') as HTMLElement;
};

describe('AuditEventDetail identity claims', () => {
  it('renders the claims as labeled rows for the nested identity shape', () => {
    render(
      <AuditEventDetail
        event={{ ...baseEvent, identity: { ...baseIdentity, ...CLAIMS } }}
        onClose={() => {}}
      />,
    );

    expect(screen.getByText('Identity Claims')).toBeInTheDocument();
    expect(screen.getByText('Canonical ID')).toBeInTheDocument();
    expect(screen.getByText(CLAIMS.canonical_id)).toBeInTheDocument();
    expect(screen.getByText('Object ID')).toBeInTheDocument();
    expect(screen.getByText('Tenant ID')).toBeInTheDocument();
    expect(screen.getByText('App ID')).toBeInTheDocument();
    // The opaque `sub` is still shown -- it is what an operator correlates on --
    // it is just no longer the only identity in the panel.
    expect(screen.getByText(CLAIMS.subject)).toBeInTheDocument();
  });

  it('renders the claims for the flat token_mint shape', () => {
    // token_mint records carry no `identity` block at all.
    const tokenMintEvent: AuditEvent = {
      timestamp: '2026-09-14T12:00:00Z',
      request_id: 'req-2',
      log_type: 'token_mint',
      username: 'alice@example.com',
      ...CLAIMS,
    };

    render(<AuditEventDetail event={tokenMintEvent} onClose={() => {}} />);

    expect(screen.getByText('Identity Claims')).toBeInTheDocument();
    expect(screen.getByText(CLAIMS.canonical_id)).toBeInTheDocument();
    expect(screen.getByTitle('Copy Canonical ID')).toBeInTheDocument();
  });

  it('omits the section entirely when the token carried no claims', () => {
    render(<AuditEventDetail event={baseEvent} onClose={() => {}} />);

    expect(screen.queryByText('Identity Claims')).not.toBeInTheDocument();
    // No dangling placeholder rows for the absent fields.
    expect(screen.queryByText('Canonical ID')).not.toBeInTheDocument();
    expect(screen.queryByText('Tenant ID')).not.toBeInTheDocument();
  });

  it('omits claims the API returned as explicit null', () => {
    // The API serializes a claim the token did not carry as `null`, not as an
    // absent key: a null must be dropped, not rendered as an empty row.
    render(
      <AuditEventDetail
        event={{
          ...baseEvent,
          identity: {
            ...baseIdentity,
            subject: 'keycloak-sub-1',
            canonical_id: 'keycloak-sub-1',
            object_id: null,
            tenant_id: null,
            app_id: null,
          },
        }}
        onClose={() => {}}
      />
    );

    expect(screen.getByText('Subject')).toBeInTheDocument();
    // A non-Entra token has no oid/tid: those labels must not appear at all.
    expect(screen.queryByText('Object ID')).not.toBeInTheDocument();
    expect(screen.queryByText('Tenant ID')).not.toBeInTheDocument();
    expect(screen.queryByTitle('Copy Tenant ID')).not.toBeInTheDocument();
  });

  it('uses monospace for opaque ids and plain text for the readable principal', () => {
    render(
      <AuditEventDetail
        event={{ ...baseEvent, identity: { ...baseIdentity, ...CLAIMS } }}
        onClose={() => {}}
      />,
    );

    expect(claimValue('Object ID').className).toContain('font-mono');
    // `principal_name` is a human-readable handle, so it is not monospaced.
    expect(claimValue('Principal').className).not.toContain('font-mono');
  });

  it('shows every claim value in full instead of truncating it', () => {
    render(
      <AuditEventDetail
        event={{ ...baseEvent, identity: { ...baseIdentity, ...CLAIMS } }}
        onClose={() => {}}
      />,
    );

    // `canonical_id` (73 chars) starts with the whole of `object_id` (36), so a
    // truncating value cell renders the two identically. Both must carry their
    // complete text and be allowed to wrap rather than clip.
    const canonical = claimValue('Canonical ID');
    const objectId = claimValue('Object ID');

    expect(canonical.textContent).toBe(CLAIMS.canonical_id);
    expect(objectId.textContent).toBe(CLAIMS.object_id);
    expect(canonical.textContent).not.toBe(objectId.textContent);
    for (const el of [canonical, objectId]) {
      expect(el.className).toContain('break-all');
      expect(el.className).not.toContain('truncate');
    }

    // The claims grid tracks the page grid: two columns while the detail panel
    // is full width, one column from `lg`, where the page drops the panel to a
    // one-third sidebar (measured: a 270px value column there, vs 127px if the
    // block stayed two-up and 85px in the original three-up layout).
    const grid = canonical.closest('.grid') as HTMLElement;
    expect(grid.className).toContain('grid-cols-1');
    expect(grid.className).toContain('md:grid-cols-2');
    expect(grid.className).toContain('lg:grid-cols-1');
    expect(grid.className).not.toContain('grid-cols-3');
  });

  it('copies a single claim value without the surrounding record', async () => {
    const writeText = jest.fn().mockResolvedValue(undefined);
    Object.assign(navigator, { clipboard: { writeText } });

    render(
      <AuditEventDetail
        event={{ ...baseEvent, identity: { ...baseIdentity, ...CLAIMS } }}
        onClose={() => {}}
      />,
    );

    // Every claim is individually copyable: hover-only `title` text is dead on
    // touch devices and "Copy JSON" only yields the whole record.
    for (const label of CLAIM_LABELS) {
      expect(screen.getByTitle(`Copy ${label}`)).toBeInTheDocument();
    }

    fireEvent.click(screen.getByTitle('Copy Object ID'));
    await waitFor(() => expect(writeText).toHaveBeenCalledWith(CLAIMS.object_id));

    fireEvent.click(screen.getByTitle('Copy Canonical ID'));
    await waitFor(() => expect(writeText).toHaveBeenLastCalledWith(CLAIMS.canonical_id));
  });
});
