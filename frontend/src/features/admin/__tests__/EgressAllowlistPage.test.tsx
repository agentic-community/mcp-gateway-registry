import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import { mockEgressAllowlistEntries } from '@/test/mocks/handlers';
import EgressAllowlistPage from '../EgressAllowlistPage';

describe('EgressAllowlistPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();
  });

  it('renders page header and SSRF banner', async () => {
    render(<EgressAllowlistPage />);

    expect(screen.getByText('Egress Allowlist')).toBeInTheDocument();
    expect(
      screen.getByText(/Control which upstream servers can be proxied to prevent SSRF attacks/i)
    ).toBeInTheDocument();
    expect(screen.getByText('SSRF Protection')).toBeInTheDocument();
  });

  it('loads and displays entries', async () => {
    render(<EgressAllowlistPage />);

    await waitFor(() => {
      expect(screen.getByText(mockEgressAllowlistEntries[0].value)).toBeInTheDocument();
    });

    for (const entry of mockEgressAllowlistEntries) {
      expect(screen.getByText(entry.value)).toBeInTheDocument();
    }

    // Kind badges may repeat across entries; check presence by unique kinds.
    const uniqueKinds = Array.from(new Set(mockEgressAllowlistEntries.map((e) => e.kind)));
    for (const kind of uniqueKinds) {
      expect(screen.getAllByText(kind).length).toBeGreaterThan(0);
    }
  });

  it('opens create modal when Add Entry is clicked', async () => {
    const user = userEvent.setup();
    render(<EgressAllowlistPage />);

    await waitFor(() => {
      expect(screen.getByRole('button', { name: /Add Entry/i })).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: /Add Entry/i }));

    await waitFor(() => {
      expect(screen.getByText('Add Allowlist Entry')).toBeInTheDocument();
    });
  });

  it('opens edit modal when Edit is clicked', async () => {
    const user = userEvent.setup();
    render(<EgressAllowlistPage />);

    await waitFor(() => {
      expect(screen.getByText(mockEgressAllowlistEntries[0].value)).toBeInTheDocument();
    });

    await user.click(screen.getAllByRole('button', { name: /Edit/i })[0]);

    await waitFor(() => {
      expect(screen.getByText('Edit Allowlist Entry')).toBeInTheDocument();
    });
  });

  it('opens delete confirmation dialog when Delete is clicked', async () => {
    const user = userEvent.setup();
    render(<EgressAllowlistPage />);

    await waitFor(() => {
      expect(screen.getByText(mockEgressAllowlistEntries[0].value)).toBeInTheDocument();
    });

    await user.click(screen.getAllByRole('button', { name: /Delete/i })[0]);

    await waitFor(() => {
      expect(screen.getByText('Delete Allowlist Entry')).toBeInTheDocument();
      expect(screen.getByText(/Are you sure you want to delete/i)).toBeInTheDocument();
    });
  });

  it('shows empty state when no entries exist', async () => {
    server.use(
      http.get('/enforceai/admin/egress-allowlist', () => {
        return HttpResponse.json([]);
      })
    );

    render(<EgressAllowlistPage />);

    await waitFor(() => {
      expect(screen.getByText('No allowlist entries')).toBeInTheDocument();
      expect(
        screen.getByText(/Add entries to control which upstream servers can be proxied/i)
      ).toBeInTheDocument();
    });
  });
});
