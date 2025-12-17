import { describe, it, expect, beforeEach } from 'vitest';
import { screen, waitFor, within } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import ScopesPage from '../ScopesPage';

describe('Create Scope Flow', () => {
  beforeEach(() => {
    server.resetHandlers();
  });

  it('allows admin to create a blank scope', async () => {
    let created = false;

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
        if (created) {
          return HttpResponse.json({
            version: '1.0',
            generated_at: new Date().toISOString(),
            etag: 'etag-2',
            group_mappings: {},
            scopes: {
              'scope-created': {
                name: 'scope-created',
                server_permissions: [],
                agent_permissions: [],
              },
            },
          });
        }

        return HttpResponse.json({
          version: '1.0',
          generated_at: new Date().toISOString(),
          etag: 'etag-1',
          group_mappings: {},
          scopes: {},
        });
      }),
      http.post('/enforceai/admin/scopes', async ({ request }) => {
        const ifMatch = request.headers.get('If-Match');
        expect(ifMatch).toBe('etag-1');

        const body = (await request.json()) as { name?: string };
        created = true;
        return HttpResponse.json({
          ok: true,
          scope_name: body.name,
          etag: 'etag-2',
          last_modified: new Date().toISOString(),
        });
      })
    );

    const { user } = render(<ScopesPage />, { withAuth: true });

    await waitFor(() => {
      expect(screen.getByText('Total Scopes')).toBeInTheDocument();
    });

    await user.click(screen.getByRole('button', { name: 'Create Scope' }));

    const dialog = screen.getByRole('dialog');
    await user.type(within(dialog).getByLabelText('Scope name'), 'scope-created');
    await user.click(within(dialog).getByRole('button', { name: 'Create Scope' }));

    await waitFor(() => {
      expect(screen.getAllByText('scope-created').length).toBeGreaterThan(0);
    });
  });
});
