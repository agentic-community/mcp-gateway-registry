import { describe, it, expect, beforeEach } from 'vitest';
import { screen, waitFor, within } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import ScopesPage from '../ScopesPage';

describe('Edit Scope Flow', () => {
  beforeEach(() => {
    server.resetHandlers();
  });

  it('allows admin to edit a scope with If-Match etag', async () => {
    let updated = false;

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
        if (updated) {
          return HttpResponse.json({
            version: '1.0',
            generated_at: new Date().toISOString(),
            etag: 'etag-2',
            group_mappings: {},
            scopes: {
              'scope-a': {
                name: 'scope-a',
                server_permissions: [
                  {
                    server: 'mcpgw',
                    methods: { all_methods: true, methods: [] },
                    tools: { all_tools: true, tools: [] },
                  },
                ],
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
          scopes: {
            'scope-a': {
              name: 'scope-a',
              server_permissions: [
                {
                  server: 'mcpgw',
                  methods: { all_methods: false, methods: ['tools/list', 'tools/call'] },
                  tools: { all_tools: false, tools: ['tool-a'] },
                },
              ],
              agent_permissions: [],
            },
          },
        });
      }),
      http.put('/enforceai/admin/scopes/scope-a', async ({ request }) => {
        const ifMatch = request.headers.get('If-Match');
        expect(ifMatch).toBe('etag-1');

        const body = (await request.json()) as {
          server_permissions?: unknown[];
        };
        expect(Array.isArray(body.server_permissions)).toBe(true);
        updated = true;

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

    const scopeCard = screen.getByText('scope-a').closest('button');
    expect(scopeCard).toBeTruthy();
    if (scopeCard) {
      await user.click(scopeCard);
    }

    await user.click(screen.getByRole('button', { name: 'Edit' }));

    const dialog = screen.getByRole('dialog');
    await user.click(within(dialog).getByLabelText('Allow all methods'));
    await user.click(within(dialog).getByLabelText('Allow all tools (*)'));

    await user.click(within(dialog).getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => {
      expect(screen.getAllByText('scope-a').length).toBeGreaterThan(0);
    });
  });

  it('shows guidance when etag precondition fails', async () => {
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
      http.put('/enforceai/admin/scopes/scope-a', () => {
        return HttpResponse.json({ detail: 'ETag mismatch' }, { status: 412 });
      })
    );

    const { user } = render(<ScopesPage />, { withAuth: true });

    await waitFor(() => {
      expect(screen.getByText('Total Scopes')).toBeInTheDocument();
    });

    const scopeCard = screen.getByText('scope-a').closest('button');
    if (scopeCard) {
      await user.click(scopeCard);
    }

    await user.click(screen.getByRole('button', { name: 'Edit' }));

    const dialog = screen.getByRole('dialog');
    await user.click(within(dialog).getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => {
      expect(
        within(dialog).getByText(/catalog changed since you loaded it/i)
      ).toBeInTheDocument();
    });
  });

  it('disables removing tools policy when tools/call is allowed', async () => {
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
          group_mappings: {},
          scopes: {
            'scope-a': {
              name: 'scope-a',
              server_permissions: [
                {
                  server: 'mcpgw',
                  methods: { all_methods: false, methods: ['tools/call'] },
                  tools: { all_tools: true, tools: [] },
                },
              ],
              agent_permissions: [],
            },
          },
        });
      })
    );

    const { user } = render(<ScopesPage />, { withAuth: true });

    await waitFor(() => {
      expect(screen.getByText('Total Scopes')).toBeInTheDocument();
    });

    const scopeCard = screen.getByText('scope-a').closest('button');
    if (scopeCard) {
      await user.click(scopeCard);
    }

    await user.click(screen.getByRole('button', { name: 'Edit' }));

    const dialog = screen.getByRole('dialog');
    const noToolsPolicy = within(dialog).getByLabelText('No tools policy');
    expect(noToolsPolicy).toBeDisabled();
  });

  it('requires an action when resources are specified', async () => {
    let putCalled = false;

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
      http.put('/enforceai/admin/scopes/scope-a', () => {
        putCalled = true;
        return HttpResponse.json({ ok: true, scope_name: 'scope-a', etag: 'etag-2' });
      })
    );

    const { user } = render(<ScopesPage />, { withAuth: true });

    await waitFor(() => {
      expect(screen.getByText('Total Scopes')).toBeInTheDocument();
    });

    const scopeCard = screen.getByText('scope-a').closest('button');
    if (scopeCard) {
      await user.click(scopeCard);
    }

    await user.click(screen.getByRole('button', { name: 'Edit' }));

    const dialog = screen.getByRole('dialog');
    await user.click(within(dialog).getByRole('button', { name: 'Add agent permission' }));
    await user.click(within(dialog).getByRole('button', { name: 'Add resource' }));
    await user.type(within(dialog).getByLabelText('Resource 1.1'), '*');

    await user.click(within(dialog).getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => {
      expect(
        within(dialog).getByText(/requires an action name/i)
      ).toBeInTheDocument();
    });
    expect(putCalled).toBe(false);
  });
});
