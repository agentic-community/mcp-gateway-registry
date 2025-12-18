import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor, within } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import { mockEgressAllowlistEntries } from '@/test/mocks/handlers';
import EgressAllowlistPage from '../EgressAllowlistPage';

describe('EgressAllowlistPage', () => {
  describe('Page Rendering', () => {
    it('renders page header and description', async () => {
      render(<EgressAllowlistPage />);

      expect(screen.getByText('Egress Allowlist')).toBeInTheDocument();
      expect(
        screen.getByText(/Control which upstream servers can be proxied to prevent SSRF attacks/i)
      ).toBeInTheDocument();
    });

    it('renders SSRF warning banner', async () => {
      render(<EgressAllowlistPage />);

      expect(screen.getByText('SSRF Protection')).toBeInTheDocument();
      expect(
        screen.getByText(/This allowlist prevents Server-Side Request Forgery/i)
      ).toBeInTheDocument();
      expect(
        screen.getByText(/Exercise caution when adding patterns/i)
      ).toBeInTheDocument();
    });

    it('renders section title and add button', async () => {
      render(<EgressAllowlistPage />);

      await waitFor(() => {
        expect(screen.getByText('Allowed Patterns')).toBeInTheDocument();
      });

      const addButton = screen.getByRole('button', { name: /Add Pattern/i });
      expect(addButton).toBeInTheDocument();
    });
  });

  describe('Loading State', () => {
    it('loads data successfully', async () => {
      render(<EgressAllowlistPage />);

      // Wait for loading to complete
      await waitFor(() => {
        if (mockEgressAllowlistEntries.length > 0) {
          expect(screen.getByText(mockEgressAllowlistEntries[0].pattern)).toBeInTheDocument();
        }
      });
    });
  });

  describe('Entries List', () => {
    it('displays all entries', async () => {
      render(<EgressAllowlistPage />);

      await waitFor(() => {
        mockEgressAllowlistEntries.forEach((entry) => {
          expect(screen.getByText(entry.pattern)).toBeInTheDocument();
        });
      });
    });

    it('displays entry count badge', async () => {
      render(<EgressAllowlistPage />);

      await waitFor(() => {
        const expectedCount = mockEgressAllowlistEntries.length;
        const badge = screen.getByText(new RegExp(`${expectedCount} entr`));
        expect(badge).toBeInTheDocument();
      });
    });

    it('displays entry descriptions', async () => {
      render(<EgressAllowlistPage />);

      await waitFor(() => {
        const entryWithDescription = mockEgressAllowlistEntries.find(
          (e) => e.description
        );
        if (entryWithDescription && entryWithDescription.description) {
          expect(
            screen.getByText(entryWithDescription.description)
          ).toBeInTheDocument();
        }
      });
    });

    it('displays created timestamp', async () => {
      render(<EgressAllowlistPage />);

      await waitFor(() => {
        expect(screen.getAllByText(/Added/i).length).toBeGreaterThan(0);
      });
    });

    it('displays updated timestamp when different from created', async () => {
      render(<EgressAllowlistPage />);

      await waitFor(() => {
        const entries = mockEgressAllowlistEntries.filter(
          (e) => e.updated_at !== e.created_at
        );
        if (entries.length > 0) {
          expect(screen.getAllByText(/Updated/i).length).toBeGreaterThan(0);
        }
      });
    });
  });

  describe('Expiration Status', () => {
    it('displays expired badge for expired entries', async () => {
      const expiredEntry = {
        entry_id: 'expired-1',
        pattern: 'expired.example.com',
        description: 'Expired entry',
        expires_at: '2020-01-01T00:00:00Z', // Past date
        created_at: '2019-01-01T00:00:00Z',
        updated_at: '2019-01-01T00:00:00Z',
      };

      server.use(
        http.get('/enforceai/admin/egress-allowlist', () => {
          return HttpResponse.json([expiredEntry]);
        })
      );

      render(<EgressAllowlistPage />);

      await waitFor(() => {
        expect(screen.getByText('Expired')).toBeInTheDocument();
      });
    });

    it('displays expiration time for future expiration', async () => {
      const futureDate = new Date();
      futureDate.setDate(futureDate.getDate() + 7);

      const expiringEntry = {
        entry_id: 'expiring-1',
        pattern: 'expiring.example.com',
        description: 'Expiring entry',
        expires_at: futureDate.toISOString(),
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
      };

      server.use(
        http.get('/enforceai/admin/egress-allowlist', () => {
          return HttpResponse.json([expiringEntry]);
        })
      );

      render(<EgressAllowlistPage />);

      await waitFor(() => {
        expect(screen.getByText(/Expires in/i)).toBeInTheDocument();
      });
    });

    it('does not display expiration badge for entries without expiration', async () => {
      const noExpiryEntry = {
        entry_id: 'no-expiry-1',
        pattern: 'noexpiry.example.com',
        description: 'No expiry',
        expires_at: null,
        created_at: '2024-01-01T00:00:00Z',
        updated_at: '2024-01-01T00:00:00Z',
      };

      server.use(
        http.get('/enforceai/admin/egress-allowlist', () => {
          return HttpResponse.json([noExpiryEntry]);
        })
      );

      render(<EgressAllowlistPage />);

      await waitFor(() => {
        expect(screen.getByText('noexpiry.example.com')).toBeInTheDocument();
      });

      expect(screen.queryByText(/Expires/i)).not.toBeInTheDocument();
      expect(screen.queryByText('Expired')).not.toBeInTheDocument();
    });
  });

  describe('Entry Actions', () => {
    it('displays Edit and Delete buttons for each entry', async () => {
      render(<EgressAllowlistPage />);

      await waitFor(() => {
        const editButtons = screen.getAllByRole('button', { name: /Edit/i });
        const deleteButtons = screen.getAllByRole('button', { name: /Delete/i });

        expect(editButtons.length).toBe(mockEgressAllowlistEntries.length);
        expect(deleteButtons.length).toBe(mockEgressAllowlistEntries.length);
      });
    });

    it('opens edit modal when Edit is clicked', async () => {
      const user = userEvent.setup();
      render(<EgressAllowlistPage />);

      await waitFor(() => {
        expect(screen.getByText(mockEgressAllowlistEntries[0].pattern)).toBeInTheDocument();
      });

      const editButtons = screen.getAllByRole('button', { name: /Edit/i });
      await user.click(editButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Edit Allowlist Entry')).toBeInTheDocument();
      });
    });

    it('opens create modal when Add Pattern is clicked', async () => {
      const user = userEvent.setup();
      render(<EgressAllowlistPage />);

      await waitFor(() => {
        const addButton = screen.getByRole('button', { name: /Add Pattern/i });
        expect(addButton).toBeInTheDocument();
      });

      const addButton = screen.getByRole('button', { name: /Add Pattern/i });
      await user.click(addButton);

      await waitFor(() => {
        expect(screen.getByText('Add Allowlist Entry')).toBeInTheDocument();
      });
    });
  });

  describe('Delete Functionality', () => {
    it('opens confirmation dialog when Delete is clicked', async () => {
      const user = userEvent.setup();
      render(<EgressAllowlistPage />);

      await waitFor(() => {
        expect(screen.getByText(mockEgressAllowlistEntries[0].pattern)).toBeInTheDocument();
      });

      const deleteButtons = screen.getAllByRole('button', { name: /Delete/i });
      await user.click(deleteButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Delete Allowlist Entry')).toBeInTheDocument();
        expect(
          screen.getByText(/Are you sure you want to delete the pattern/i)
        ).toBeInTheDocument();
      });
    });

    it('closes dialog when Cancel is clicked', async () => {
      const user = userEvent.setup();
      render(<EgressAllowlistPage />);

      await waitFor(() => {
        expect(screen.getByText(mockEgressAllowlistEntries[0].pattern)).toBeInTheDocument();
      });

      const deleteButtons = screen.getAllByRole('button', { name: /Delete/i });
      await user.click(deleteButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Delete Allowlist Entry')).toBeInTheDocument();
      });

      const cancelButton = screen.getByRole('button', { name: /Cancel/i });
      await user.click(cancelButton);

      await waitFor(() => {
        expect(screen.queryByText('Delete Allowlist Entry')).not.toBeInTheDocument();
      });
    });

    it('deletes entry successfully', async () => {
      const user = userEvent.setup();
      const entryToDelete = mockEgressAllowlistEntries[0];

      server.use(
        http.delete('/enforceai/admin/egress-allowlist/:entryId', ({ params }) => {
          if (params.entryId === entryToDelete.entry_id) {
            return new HttpResponse(null, { status: 204 });
          }
          return new HttpResponse(null, { status: 404 });
        })
      );

      render(<EgressAllowlistPage />);

      await waitFor(() => {
        expect(screen.getByText(entryToDelete.pattern)).toBeInTheDocument();
      });

      const deleteButtons = screen.getAllByRole('button', { name: /Delete/i });
      await user.click(deleteButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Delete Allowlist Entry')).toBeInTheDocument();
      });

      // Find the Delete button in the dialog (not in the list)
      const confirmButtons = screen.getAllByRole('button', {
        name: /^Delete$/i,
      });
      // The last one should be in the confirm dialog
      await user.click(confirmButtons[confirmButtons.length - 1]);

      // Just verify the success toast appeared
      await waitFor(() => {
        expect(screen.getByText('Entry deleted')).toBeInTheDocument();
      });
    });

    it('shows error toast when delete fails', async () => {
      const user = userEvent.setup();

      server.use(
        http.delete('/enforceai/admin/egress-allowlist/:entryId', () => {
          return new HttpResponse(
            JSON.stringify({ message: 'Delete failed' }),
            { status: 500 }
          );
        })
      );

      render(<EgressAllowlistPage />);

      await waitFor(() => {
        expect(screen.getByText(mockEgressAllowlistEntries[0].pattern)).toBeInTheDocument();
      });

      const deleteButtons = screen.getAllByRole('button', { name: /Delete/i });
      await user.click(deleteButtons[0]);

      await waitFor(() => {
        expect(screen.getByText('Delete Allowlist Entry')).toBeInTheDocument();
      });

      // Find the Delete button in the dialog (not in the list)
      const confirmButtons = screen.getAllByRole('button', {
        name: /^Delete$/i,
      });
      // The last one should be in the confirm dialog
      await user.click(confirmButtons[confirmButtons.length - 1]);

      await waitFor(() => {
        expect(screen.getByText('Failed to delete entry')).toBeInTheDocument();
      });
    });
  });

  describe('Error State', () => {
    it('displays error message when fetch fails', async () => {
      server.use(
        http.get('/enforceai/admin/egress-allowlist', () => {
          return new HttpResponse(
            JSON.stringify({ message: 'Server error' }),
            { status: 500 }
          );
        })
      );

      render(<EgressAllowlistPage />);

      await waitFor(() => {
        expect(screen.getAllByText('Failed to load allowlist')[0]).toBeInTheDocument();
      });
    });

    it('displays retry button on error', async () => {
      server.use(
        http.get('/enforceai/admin/egress-allowlist', () => {
          return new HttpResponse(
            JSON.stringify({ message: 'Server error' }),
            { status: 500 }
          );
        })
      );

      render(<EgressAllowlistPage />);

      await waitFor(() => {
        const retryButton = screen.getByRole('button', { name: /Retry/i });
        expect(retryButton).toBeInTheDocument();
      });
    });

    it('retries fetch when retry button is clicked', async () => {
      const user = userEvent.setup();
      let callCount = 0;

      server.use(
        http.get('/enforceai/admin/egress-allowlist', () => {
          callCount++;
          if (callCount === 1) {
            return new HttpResponse(
              JSON.stringify({ message: 'Server error' }),
              { status: 500 }
            );
          }
          return HttpResponse.json(mockEgressAllowlistEntries);
        })
      );

      render(<EgressAllowlistPage />);

      await waitFor(() => {
        expect(screen.getAllByText('Failed to load allowlist')[0]).toBeInTheDocument();
      });

      const retryButton = screen.getByRole('button', { name: /Retry/i });
      await user.click(retryButton);

      await waitFor(() => {
        expect(screen.getByText(mockEgressAllowlistEntries[0].pattern)).toBeInTheDocument();
      });
    });
  });

  describe('Empty State', () => {
    it('displays empty state when no entries exist', async () => {
      server.use(
        http.get('/enforceai/admin/egress-allowlist', () => {
          return HttpResponse.json([]);
        })
      );

      render(<EgressAllowlistPage />);

      await waitFor(() => {
        expect(screen.getByText('No allowlist entries')).toBeInTheDocument();
        expect(
          screen.getByText('Add patterns to control which upstream servers can be proxied')
        ).toBeInTheDocument();
      });
    });

    it('displays add button in empty state', async () => {
      server.use(
        http.get('/enforceai/admin/egress-allowlist', () => {
          return HttpResponse.json([]);
        })
      );

      render(<EgressAllowlistPage />);

      await waitFor(() => {
        const addButton = screen.getByRole('button', { name: /Add First Pattern/i });
        expect(addButton).toBeInTheDocument();
      });
    });
  });
});
