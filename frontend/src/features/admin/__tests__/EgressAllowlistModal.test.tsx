import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import { EgressAllowlistModal } from '../EgressAllowlistModal';
import type { EgressAllowlistEntry } from '@/api/types';

describe('EgressAllowlistModal', () => {
  const mockOnClose = vi.fn();
  const mockOnSuccess = vi.fn();

  const mockEntry: EgressAllowlistEntry = {
    entry_id: 1,
    kind: 'hostname',
    value: 'localhost',
    comment: 'Allow localhost connections',
    expires_at: null,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  };

  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();
  });

  it('renders create mode', () => {
    render(
      <EgressAllowlistModal open={true} onClose={mockOnClose} onSuccess={mockOnSuccess} entry={null} />
    );

    expect(screen.getByText('Add Allowlist Entry')).toBeInTheDocument();
    expect(
      screen.getByText(/Allow a destination host\/network for upstream proxying/i)
    ).toBeInTheDocument();
    expect(screen.getByLabelText(/Kind/i)).toHaveValue('hostname');
    expect(screen.getByLabelText(/Value/i)).toHaveValue('');
    expect(screen.getByLabelText(/Comment/i)).toHaveValue('');
  });

  it('creates an entry successfully', async () => {
    const user = userEvent.setup();

    server.use(
      http.post('/enforceai/admin/egress-allowlist', async ({ request }) => {
        const body = (await request.json()) as { kind: string; value: string };
        return HttpResponse.json({
          entry_id: 999,
          kind: body.kind,
          value: body.value,
          comment: null,
          expires_at: null,
          created_at: new Date().toISOString(),
          updated_at: new Date().toISOString(),
        });
      })
    );

    render(
      <EgressAllowlistModal open={true} onClose={mockOnClose} onSuccess={mockOnSuccess} entry={null} />
    );

    const valueInput = screen.getByLabelText(/Value/i);
    await user.type(valueInput, 'host.docker.internal');

    const submitButton = screen.getByRole('button', { name: /Add Entry/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(mockOnSuccess).toHaveBeenCalled();
      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  it('populates edit mode with existing entry', () => {
    render(
      <EgressAllowlistModal open={true} onClose={mockOnClose} onSuccess={mockOnSuccess} entry={mockEntry} />
    );

    expect(screen.getByText('Edit Allowlist Entry')).toBeInTheDocument();
    expect(screen.getByLabelText(/Kind/i)).toHaveValue('hostname');
    expect(screen.getByLabelText(/Value/i)).toHaveValue('localhost');
    expect(screen.getByLabelText(/Comment/i)).toHaveValue('Allow localhost connections');
  });

  it('updates an entry successfully', async () => {
    const user = userEvent.setup();

    server.use(
      http.put('/enforceai/admin/egress-allowlist/:entryId', async ({ request }) => {
        const body = (await request.json()) as { value?: string };
        return HttpResponse.json({
          ...mockEntry,
          value: body.value ?? mockEntry.value,
          updated_at: new Date().toISOString(),
        });
      })
    );

    render(
      <EgressAllowlistModal open={true} onClose={mockOnClose} onSuccess={mockOnSuccess} entry={mockEntry} />
    );

    const valueInput = screen.getByLabelText(/Value/i);
    await user.clear(valueInput);
    await user.type(valueInput, 'updated-host');

    const submitButton = screen.getByRole('button', { name: /Update Entry/i });
    await user.click(submitButton);

    await waitFor(() => {
      expect(mockOnSuccess).toHaveBeenCalled();
      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  it('shows a warning for localhost hostname', async () => {
    const user = userEvent.setup();
    render(
      <EgressAllowlistModal open={true} onClose={mockOnClose} onSuccess={mockOnSuccess} entry={null} />
    );

    const valueInput = screen.getByLabelText(/Value/i);
    await user.type(valueInput, 'localhost');

    await waitFor(() => {
      expect(screen.getByText(/This entry includes localhost/i)).toBeInTheDocument();
    });
  });
});

