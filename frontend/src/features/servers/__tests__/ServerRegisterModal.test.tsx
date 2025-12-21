import { describe, it, expect, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import { ServerRegisterModal } from '../ServerRegisterModal';

describe('ServerRegisterModal', () => {
  beforeEach(() => {
    server.resetHandlers();
  });

  it('submits upstream_auth when registering a server', async () => {
    let capturedBody: unknown = null;

    server.use(
      http.post('/api/servers', async ({ request }) => {
        capturedBody = await request.json();
        return HttpResponse.json(
          {
            display_name: 'My MCP Server',
            path: 'my-server',
            proxy_pass_url: 'http://localhost:3000/mcp',
            is_enabled: true,
            health_status: 'unknown',
          },
          { status: 201 }
        );
      })
    );

    const { user } = render(
      <ServerRegisterModal open={true} onClose={() => {}} onSuccess={() => {}} />
    );

    await user.type(screen.getByLabelText('Display Name'), 'My MCP Server');
    await user.type(screen.getByLabelText('Path'), 'my-server');
    await user.type(screen.getByLabelText('Proxy URL'), 'http://localhost:3000/mcp');

    await user.selectOptions(screen.getByLabelText('Auth Type'), 'api-key');
    await user.selectOptions(screen.getByLabelText('Credential Binding'), 'user');
    await user.type(
      screen.getByLabelText('Injection Header Override (optional)'),
      'X-API-Key'
    );

    await user.click(screen.getByRole('button', { name: 'Register Server' }));

    await waitFor(() => {
      expect(capturedBody).not.toBeNull();
    });

    const body = capturedBody as {
      upstream_auth?: {
        mode?: string;
        type?: string;
        credential_binding?: string;
        injection?: { header_name?: string; scheme?: string | null } | null;
      };
    };

    expect(body.upstream_auth?.mode).toBe('gateway-managed');
    expect(body.upstream_auth?.type).toBe('api-key');
    expect(body.upstream_auth?.credential_binding).toBe('user');
    expect(body.upstream_auth?.injection?.header_name).toBe('X-API-Key');
  });

  it('requires provider for OAuth upstream auth types', async () => {
    for (const authType of ['oauth2', 'oidc', 'provider-oauth']) {
      server.resetHandlers();
      let called = false;

      server.use(
        http.post('/api/servers', () => {
          called = true;
          return HttpResponse.json({}, { status: 201 });
        })
      );

      const { user, unmount } = render(
        <ServerRegisterModal open={true} onClose={() => {}} onSuccess={() => {}} />
      );

      await user.type(screen.getByLabelText('Display Name'), 'My MCP Server');
      await user.type(screen.getByLabelText('Path'), 'my-server');
      await user.type(screen.getByLabelText('Proxy URL'), 'http://localhost:3000/mcp');

      await user.selectOptions(screen.getByLabelText('Auth Type'), authType);

      await user.click(screen.getByRole('button', { name: 'Register Server' }));

      expect(
        await screen.findByText('Provider is required for OAuth upstream auth')
      ).toBeInTheDocument();
      expect(called).toBe(false);

      unmount();
    }
  });
});
