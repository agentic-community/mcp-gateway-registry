import { describe, it, expect, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import { mockServers } from '@/test/mocks/handlers';
import { ServerEditModal } from '../ServerEditModal';

describe('ServerEditModal', () => {
  beforeEach(() => {
    server.resetHandlers();
  });

  it('submits upstream_auth when updating a server', async () => {
    let capturedBody: unknown = null;

    server.use(
      http.put('/api/servers/:path', async ({ request }) => {
        capturedBody = await request.json();
        return HttpResponse.json(mockServers[0]);
      })
    );

    const { user } = render(
      <ServerEditModal
        open={true}
        server={mockServers[0]}
        onClose={() => {}}
        onSuccess={() => {}}
      />
    );

    await user.selectOptions(screen.getByLabelText('Auth Type'), 'jwt');

    await user.click(screen.getByRole('button', { name: 'Save Changes' }));

    await waitFor(() => {
      expect(capturedBody).not.toBeNull();
    });

    const body = capturedBody as {
      upstream_auth?: { mode?: string; type?: string };
    };

    expect(body.upstream_auth?.mode).toBe('gateway-managed');
    expect(body.upstream_auth?.type).toBe('jwt');
  });
});

