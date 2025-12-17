import { describe, it, expect, beforeEach } from 'vitest';
import { screen, waitFor, within } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import ScopesPage from '../ScopesPage';

describe('Delete Scope Flow', () => {
  beforeEach(() => {
    server.resetHandlers();
  });

  it('allows admin to delete a scope with typed confirmation', async () => {
    let deleted = false;

    server.use(
      http.get('/api/auth/me', () => {
        return HttpResponse.json({
          user_id: 'test|admin123',
          email: 'admin@example.com',
          username: 'admin',
          auth_method: 'password',
          is_admin: true,
          roles: [],
          groups: ['enforceai-admin'],
        });
      }),
      http.get('/enforceai/scopes/catalog', () => {
        if (deleted) {
          return HttpResponse.json({
            version: '1.0',
            generated_at: new Date().toISOString(),
            etag: 'etag-2',
            group_mappings: {},
            scopes: {},
          });
        }

        return HttpResponse.json({
          version: '1.0',
          generated_at: new Date().toISOString(),
          etag: 'etag-1',
          group_mappings: {},
          scopes: {
            'scope-a': {
              name: 'scope-a',
              server_permissions: [],
              agent_permissions: [],
            },
          },
        });
      }),
      http.delete('/enforceai/admin/scopes/scope-a', ({ request }) => {
        const ifMatch = request.headers.get('If-Match');
        expect(ifMatch).toBe('etag-1');
        deleted = true;
        return HttpResponse.json({
          ok: true,
          scope_name: 'scope-a',
          etag: 'etag-2',
          last_modified: new Date().toISOString(),
        });
      })
    );

    const { user } = render(<ScopesPage />, { withAuth: true });

    await waitFor(() => {
      expect(screen.getByText('Total Scopes')).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: 'Delete' }));

    const dialog = screen.getByRole('dialog');
    await user.type(within(dialog).getByPlaceholderText('scope-a'), 'scope-a');
    await user.click(within(dialog).getByRole('button', { name: 'Delete Scope' }));

    await waitFor(() => {
      expect(screen.queryByText('scope-a')).not.toBeInTheDocument();
    });

    await waitFor(() => {
      expect(screen.getByText('Scope deleted')).toBeInTheDocument();
    });
  });

  it('shows conflict toast when scope is referenced', async () => {
    server.use(
      http.get('/api/auth/me', () => {
        return HttpResponse.json({
          user_id: 'test|admin123',
          email: 'admin@example.com',
          username: 'admin',
          auth_method: 'password',
          is_admin: true,
          roles: [],
          groups: ['enforceai-admin'],
        });
      }),
      http.get('/enforceai/scopes/catalog', () => {
        return HttpResponse.json({
          version: '1.0',
          generated_at: new Date().toISOString(),
          etag: 'etag-1',
          group_mappings: { 'some-group': ['scope-a'] },
          scopes: {
            'scope-a': {
              name: 'scope-a',
              server_permissions: [],
              agent_permissions: [],
            },
          },
        });
      }),
      http.delete('/enforceai/admin/scopes/scope-a', () => {
        return HttpResponse.json(
          { detail: 'Scope is referenced by group_mappings: some-group' },
          { status: 409 }
        );
      })
    );

    const { user } = render(<ScopesPage />, { withAuth: true });

    await waitFor(() => {
      expect(screen.getByText('Total Scopes')).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: 'Delete' }));

    const dialog = screen.getByRole('dialog');
    await user.type(within(dialog).getByPlaceholderText('scope-a'), 'scope-a');
    await user.click(within(dialog).getByRole('button', { name: 'Delete Scope' }));

    await waitFor(() => {
      expect(screen.getByText('Delete failed')).toBeInTheDocument();
      expect(screen.getByText(/referenced by group_mappings/i)).toBeInTheDocument();
    });
  });
});

