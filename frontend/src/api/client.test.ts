import { describe, it, expect, beforeEach } from 'vitest';
import { http, HttpResponse } from 'msw';
import { server } from '@/test/mocks/server';
import {
  apiClient,
  apiGet,
  apiPost,
  clearCsrfToken,
  getCsrfToken,
} from './client';

describe('API Client', () => {
  beforeEach(() => {
    clearCsrfToken();
  });

  describe('CSRF token handling', () => {
    it('fetches CSRF token and attaches to POST requests', async () => {
      let receivedCsrfToken: string | null = null;

      server.use(
        http.post('/api/test', async ({ request }) => {
          receivedCsrfToken = request.headers.get('X-CSRF-Token');
          return HttpResponse.json({ success: true });
        })
      );

      await apiPost('/api/test', { data: 'test' });

      expect(receivedCsrfToken).toBe('test-csrf-token-12345');
      expect(getCsrfToken()).toBe('test-csrf-token-12345');
    });

    it('does not attach CSRF token to GET requests', async () => {
      let receivedCsrfToken: string | null = null;

      server.use(
        http.get('/api/test', async ({ request }) => {
          receivedCsrfToken = request.headers.get('X-CSRF-Token');
          return HttpResponse.json({ data: 'test' });
        })
      );

      await apiGet('/api/test');

      expect(receivedCsrfToken).toBeNull();
    });

    it('caches CSRF token for subsequent requests', async () => {
      let csrfFetchCount = 0;

      server.use(
        http.get('/api/auth/csrf', () => {
          csrfFetchCount++;
          return HttpResponse.json({ csrf_token: 'cached-token' });
        }),
        http.post('/api/test', () => {
          return HttpResponse.json({ success: true });
        })
      );

      await apiPost('/api/test', { data: 'first' });
      await apiPost('/api/test', { data: 'second' });
      await apiPost('/api/test', { data: 'third' });

      expect(csrfFetchCount).toBe(1);
    });

    it('refreshes CSRF token on 403 CSRF error', async () => {
      let requestCount = 0;
      let csrfFetchCount = 0;

      server.use(
        http.get('/api/auth/csrf', () => {
          csrfFetchCount++;
          const token = csrfFetchCount === 1 ? 'old-token' : 'new-token';
          return HttpResponse.json({ csrf_token: token });
        }),
        http.post('/api/test', async ({ request }) => {
          requestCount++;
          const csrfToken = request.headers.get('X-CSRF-Token');
          if (csrfToken === 'old-token') {
            return HttpResponse.json(
              { detail: 'CSRF token invalid' },
              { status: 403 }
            );
          }
          return HttpResponse.json({ success: true });
        })
      );

      await apiPost('/api/test', { data: 'test' });

      expect(csrfFetchCount).toBe(2);
      expect(requestCount).toBe(2);
      expect(getCsrfToken()).toBe('new-token');
    });
  });

  describe('Request ID generation', () => {
    it('attaches X-Request-Id header to all requests', async () => {
      let receivedRequestId: string | null = null;

      server.use(
        http.get('/api/test', async ({ request }) => {
          receivedRequestId = request.headers.get('X-Request-Id');
          return HttpResponse.json({ data: 'test' });
        })
      );

      await apiGet('/api/test');

      expect(receivedRequestId).toBeTruthy();
      // UUID v4 format check
      expect(receivedRequestId).toMatch(
        /^[0-9a-f]{8}-[0-9a-f]{4}-4[0-9a-f]{3}-[89ab][0-9a-f]{3}-[0-9a-f]{12}$/i
      );
    });

    it('generates unique request IDs for each request', async () => {
      const requestIds: string[] = [];

      server.use(
        http.get('/api/test', async ({ request }) => {
          requestIds.push(request.headers.get('X-Request-Id') || '');
          return HttpResponse.json({ data: 'test' });
        })
      );

      await apiGet('/api/test');
      await apiGet('/api/test');
      await apiGet('/api/test');

      expect(new Set(requestIds).size).toBe(3);
    });
  });

  describe('Error normalization', () => {
    it('normalizes 401 errors', async () => {
      server.use(
        http.get('/api/test', () => {
          return HttpResponse.json(
            { detail: 'Invalid credentials' },
            { status: 401 }
          );
        })
      );

      try {
        await apiGet('/api/test');
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toHaveProperty('status', 401);
        expect(error).toHaveProperty('message', 'Invalid credentials');
      }
    });

    it('normalizes 403 errors (non-CSRF)', async () => {
      server.use(
        http.get('/api/test', () => {
          return HttpResponse.json(
            { detail: 'Access denied' },
            { status: 403 }
          );
        })
      );

      try {
        await apiGet('/api/test');
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toHaveProperty('status', 403);
        expect(error).toHaveProperty('message', 'Access denied');
      }
    });

    it('normalizes 503 errors', async () => {
      server.use(
        http.get('/api/test', () => {
          return HttpResponse.json(
            { detail: 'Service temporarily unavailable' },
            { status: 503 }
          );
        })
      );

      try {
        await apiGet('/api/test');
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toHaveProperty('status', 503);
        expect(error).toHaveProperty('message', 'Service temporarily unavailable');
      }
    });

    it('handles errors without detail field', async () => {
      server.use(
        http.get('/api/test', () => {
          return HttpResponse.json({}, { status: 500 });
        })
      );

      try {
        await apiGet('/api/test');
        expect.fail('Should have thrown');
      } catch (error) {
        expect(error).toHaveProperty('status', 500);
        expect(error).toHaveProperty('message', 'Server error');
      }
    });
  });

  describe('Credentials handling', () => {
    it('includes credentials in requests', async () => {
      // Check that withCredentials is set
      expect(apiClient.defaults.withCredentials).toBe(true);
    });
  });
});
