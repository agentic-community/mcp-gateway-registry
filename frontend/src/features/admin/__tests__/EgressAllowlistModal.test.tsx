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
    entry_id: 'entry-1',
    pattern: 'localhost:*',
    description: 'Allow localhost connections',
    expires_at: null,
    created_at: '2024-01-01T00:00:00Z',
    updated_at: '2024-01-01T00:00:00Z',
  };

  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();
  });

  describe('Create Mode', () => {
    it('renders modal with create title', () => {
      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      expect(screen.getByText('Add Allowlist Entry')).toBeInTheDocument();
      expect(
        screen.getByText(/Add a new pattern to control which upstream servers/i)
      ).toBeInTheDocument();
    });

    it('renders empty form fields', () => {
      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      const patternInput = screen.getByLabelText(/Pattern/i);
      const descriptionInput = screen.getByLabelText(/Description/i);

      expect(patternInput).toHaveValue('');
      expect(descriptionInput).toHaveValue('');
    });

    it('requires pattern to submit', async () => {
      const user = userEvent.setup();
      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      // Wait for modal to open
      await waitFor(() => {
        expect(screen.getByRole('button', { name: /Add Entry/i })).toBeInTheDocument();
      });

      // Verify pattern input has required attribute
      const patternInput = screen.getByLabelText(/Pattern/i);
      expect(patternInput).toHaveAttribute('required');
    });

    it('creates entry successfully', async () => {
      const user = userEvent.setup();

      server.use(
        http.post('/enforceai/admin/egress-allowlist', async ({ request }) => {
          const body = await request.json() as { pattern: string };
          return HttpResponse.json({
            entry_id: 'new-entry',
            pattern: body.pattern,
            description: null,
            expires_at: null,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          });
        })
      );

      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      const patternInput = screen.getByLabelText(/Pattern/i);
      await user.type(patternInput, 'example.com:*');

      const submitButton = screen.getByRole('button', { name: /Add Entry/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockOnSuccess).toHaveBeenCalled();
        expect(mockOnClose).toHaveBeenCalled();
      });
    });

    it('creates entry with description and expiration', async () => {
      const user = userEvent.setup();

      server.use(
        http.post('/enforceai/admin/egress-allowlist', async ({ request }) => {
          const body = await request.json() as { pattern: string; description: string };
          return HttpResponse.json({
            entry_id: 'new-entry',
            pattern: body.pattern,
            description: body.description,
            expires_at: null,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          });
        })
      );

      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      const patternInput = screen.getByLabelText(/Pattern/i);
      const descriptionInput = screen.getByLabelText(/Description/i);

      await user.type(patternInput, 'api.example.com');
      await user.type(descriptionInput, 'Allow API access');

      const submitButton = screen.getByRole('button', { name: /Add Entry/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockOnSuccess).toHaveBeenCalled();
      });
    });

    it('shows error toast on creation failure', async () => {
      const user = userEvent.setup();

      server.use(
        http.post('/enforceai/admin/egress-allowlist', () => {
          return new HttpResponse(
            JSON.stringify({ message: 'Pattern already exists' }),
            { status: 400 }
          );
        })
      );

      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      const patternInput = screen.getByLabelText(/Pattern/i);
      await user.type(patternInput, 'existing-pattern');

      const submitButton = screen.getByRole('button', { name: /Add Entry/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText('Failed to create entry')).toBeInTheDocument();
      });

      expect(mockOnSuccess).not.toHaveBeenCalled();
      expect(mockOnClose).not.toHaveBeenCalled();
    });
  });

  describe('Edit Mode', () => {
    it('renders modal with edit title', () => {
      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={mockEntry}
        />
      );

      expect(screen.getByText('Edit Allowlist Entry')).toBeInTheDocument();
      expect(
        screen.getByText(/Update the pattern, description, or expiration date/i)
      ).toBeInTheDocument();
    });

    it('populates form with existing entry data', () => {
      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={mockEntry}
        />
      );

      const patternInput = screen.getByLabelText(/Pattern/i);
      const descriptionInput = screen.getByLabelText(/Description/i);

      expect(patternInput).toHaveValue(mockEntry.pattern);
      expect(descriptionInput).toHaveValue(mockEntry.description);
    });

    it('updates entry successfully', async () => {
      const user = userEvent.setup();

      server.use(
        http.put('/enforceai/admin/egress-allowlist/:entryId', async ({ request }) => {
          const body = await request.json() as { pattern: string };
          return HttpResponse.json({
            ...mockEntry,
            pattern: body.pattern,
            updated_at: new Date().toISOString(),
          });
        })
      );

      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={mockEntry}
        />
      );

      const patternInput = screen.getByLabelText(/Pattern/i);
      await user.clear(patternInput);
      await user.type(patternInput, 'updated-pattern:*');

      const submitButton = screen.getByRole('button', { name: /Update Entry/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockOnSuccess).toHaveBeenCalled();
        expect(mockOnClose).toHaveBeenCalled();
      });
    });

    it('shows error toast on update failure', async () => {
      const user = userEvent.setup();

      server.use(
        http.put('/enforceai/admin/egress-allowlist/:entryId', () => {
          return new HttpResponse(
            JSON.stringify({ message: 'Update failed' }),
            { status: 500 }
          );
        })
      );

      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={mockEntry}
        />
      );

      const submitButton = screen.getByRole('button', { name: /Update Entry/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(screen.getByText('Failed to update entry')).toBeInTheDocument();
      });

      expect(mockOnSuccess).not.toHaveBeenCalled();
      expect(mockOnClose).not.toHaveBeenCalled();
    });
  });

  describe('Form Fields', () => {
    it('allows entering pattern text', async () => {
      const user = userEvent.setup();
      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      // Wait for modal to open
      await waitFor(() => {
        expect(screen.getByLabelText(/Pattern/i)).toBeInTheDocument();
      });

      const patternInput = screen.getByLabelText(/Pattern/i);
      await user.type(patternInput, 'test-pattern:*');

      expect(patternInput).toHaveValue('test-pattern:*');
    });

    it('allows entering description text', async () => {
      const user = userEvent.setup();
      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      // Wait for modal to open
      await waitFor(() => {
        expect(screen.getByLabelText(/Description/i)).toBeInTheDocument();
      });

      const descriptionInput = screen.getByLabelText(/Description/i);
      await user.type(descriptionInput, 'Test description');

      expect(descriptionInput).toHaveValue('Test description');
    });
  });

  describe('Modal Controls', () => {
    it('calls onClose when Cancel is clicked', async () => {
      const user = userEvent.setup();
      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      const cancelButton = screen.getByRole('button', { name: /Cancel/i });
      await user.click(cancelButton);

      expect(mockOnClose).toHaveBeenCalled();
    });

    it('does not render when closed', () => {
      render(
        <EgressAllowlistModal
          open={false}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      expect(screen.queryByText('Add Allowlist Entry')).not.toBeInTheDocument();
    });

    it('resets form when modal opens with different entry', async () => {
      const { rerender } = render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={mockEntry}
        />
      );

      expect(screen.getByLabelText(/Pattern/i)).toHaveValue(mockEntry.pattern);

      // Close and reopen with null entry
      rerender(
        <EgressAllowlistModal
          open={false}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      rerender(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      expect(screen.getByLabelText(/Pattern/i)).toHaveValue('');
    });
  });

  describe('Helper Text', () => {
    it('displays pattern help text', () => {
      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      expect(
        screen.getByText(/Use wildcards \(\*\) for flexible matching/i)
      ).toBeInTheDocument();
    });

    it('displays expiration help text', () => {
      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      expect(
        screen.getByText(/Set an expiration date for temporary access/i)
      ).toBeInTheDocument();
    });
  });

  describe('Security Warnings', () => {
    it('shows warning for localhost patterns', async () => {
      const user = userEvent.setup();
      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      const patternInput = screen.getByLabelText(/Pattern/i);
      await user.type(patternInput, 'localhost:3000');

      await waitFor(() => {
        expect(
          screen.getByText(/This pattern matches localhost/i)
        ).toBeInTheDocument();
      });
    });

    it('shows warning for loopback address patterns', async () => {
      const user = userEvent.setup();
      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      const patternInput = screen.getByLabelText(/Pattern/i);
      await user.type(patternInput, '127.0.0.1:8080');

      await waitFor(() => {
        expect(
          screen.getByText(/This pattern matches loopback addresses/i)
        ).toBeInTheDocument();
      });
    });

    it('shows warning for private network patterns (10.x)', async () => {
      const user = userEvent.setup();
      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      const patternInput = screen.getByLabelText(/Pattern/i);
      await user.type(patternInput, '10.0.0.1');

      await waitFor(() => {
        expect(
          screen.getByText(/This pattern matches private network \(10\.x\.x\.x\)/i)
        ).toBeInTheDocument();
      });
    });

    it('shows warning for private network patterns (192.168.x)', async () => {
      const user = userEvent.setup();
      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      const patternInput = screen.getByLabelText(/Pattern/i);
      await user.type(patternInput, '192.168.1.1');

      await waitFor(() => {
        expect(
          screen.getByText(/This pattern matches private network \(192\.168\.x\.x\)/i)
        ).toBeInTheDocument();
      });
    });

    it('shows warning for wildcard-only pattern', async () => {
      const user = userEvent.setup();
      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      const patternInput = screen.getByLabelText(/Pattern/i);
      await user.type(patternInput, '*');

      await waitFor(() => {
        expect(
          screen.getByText(/This pattern matches wildcard matching all hosts/i)
        ).toBeInTheDocument();
      });
    });

    it('does not show warning for public domain patterns', async () => {
      const user = userEvent.setup();
      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      const patternInput = screen.getByLabelText(/Pattern/i);
      await user.type(patternInput, 'api.example.com');

      // Wait for potential warning
      await waitFor(() => {
        expect(patternInput).toHaveValue('api.example.com');
      });

      // Verify no warning is shown
      expect(
        screen.queryByText(/This pattern matches/i)
      ).not.toBeInTheDocument();
    });

    it('allows submission with warning patterns', async () => {
      const user = userEvent.setup();

      server.use(
        http.post('/enforceai/admin/egress-allowlist', async ({ request }) => {
          const body = (await request.json()) as { pattern: string };
          return HttpResponse.json({
            entry_id: 'new-entry',
            pattern: body.pattern,
            description: null,
            expires_at: null,
            created_at: new Date().toISOString(),
            updated_at: new Date().toISOString(),
          });
        })
      );

      render(
        <EgressAllowlistModal
          open={true}
          onClose={mockOnClose}
          onSuccess={mockOnSuccess}
          entry={null}
        />
      );

      const patternInput = screen.getByLabelText(/Pattern/i);
      await user.type(patternInput, 'localhost:*');

      // Verify warning is shown
      await waitFor(() => {
        expect(
          screen.getByText(/This pattern matches localhost/i)
        ).toBeInTheDocument();
      });

      // Submit should still work
      const submitButton = screen.getByRole('button', { name: /Add Entry/i });
      await user.click(submitButton);

      await waitFor(() => {
        expect(mockOnSuccess).toHaveBeenCalled();
      });
    });
  });
});
