import { http, HttpResponse } from 'msw';

// Default mock handlers for API endpoints
export const handlers = [
  // Auth endpoints
  http.get('/api/auth/me', () => {
    return HttpResponse.json({
      user_id: 'test|user123',
      email: 'test@example.com',
      username: 'testuser',
      auth_method: 'password',
      is_admin: false,
      roles: [],
      groups: [],
    });
  }),

  http.get('/api/auth/csrf', () => {
    return HttpResponse.json({
      csrf_token: 'test-csrf-token-12345',
    });
  }),

  http.post('/api/auth/login', async ({ request }) => {
    const body = await request.json() as { username?: string; password?: string };
    if (body.username === 'testuser' && body.password === 'testpass') {
      return HttpResponse.json({ success: true });
    }
    return HttpResponse.json(
      { detail: 'Invalid credentials' },
      { status: 401 }
    );
  }),

  http.post('/api/auth/logout', () => {
    return HttpResponse.json({ success: true });
  }),

  http.get('/api/auth/providers', () => {
    return HttpResponse.json({
      providers: [
        { name: 'google', display_name: 'Google' },
        { name: 'github', display_name: 'GitHub' },
      ],
    });
  }),

  http.post('/api/auth/refresh', () => {
    return HttpResponse.json({ success: true });
  }),

  // Registry endpoints
  http.get('/api/servers', () => {
    return HttpResponse.json({
      servers: [],
    });
  }),

  http.get('/api/agents', () => {
    return HttpResponse.json({
      agents: [],
    });
  }),

  // EnforceAI endpoints
  http.get('/enforceai/agents', () => {
    return HttpResponse.json([]);
  }),
];
