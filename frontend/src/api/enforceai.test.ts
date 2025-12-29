import { describe, it, expect } from 'vitest';
import { vi } from 'vitest';
import { http, HttpResponse } from 'msw';
import { server } from '@/test/mocks/server';
import { getAuditEvents, exportAdminAuditEvents } from './enforceai';
import type { AuditEventsListResponse } from './types';

describe('EnforceAI API - Audit Events', () => {
  describe('getAuditEvents', () => {
    it('fetches audit events without query parameters', async () => {
      const mockResponse: AuditEventsListResponse = {
        items: [
          {
            event_id: 1,
            occurred_at: '2024-01-15T10:00:00Z',
            user_id: 'test|user123',
            agent_id: '9d2724e9-1753-4493-8993-0d6986754414',
            action: 'tools/call',
            outcome: 'allow',
            request_id: 'req-001',
            details: { server: 'sqlite', tool: 'read_query' },
          },
        ],
        next_cursor: null,
        server_time: '2024-01-15T10:30:00Z',
      };

      server.use(
        http.get('/enforceai/audit/events', () => {
          return HttpResponse.json(mockResponse);
        })
      );

      const result = await getAuditEvents();

      expect(result.items).toHaveLength(1);
      expect(result.items[0].event_id).toBe(1);
      expect(result.items[0].action).toBe('tools/call');
      expect(result.next_cursor).toBeNull();
    });

    it('includes time range parameters in query string', async () => {
      let receivedUrl: URL | null = null;

      server.use(
        http.get('/enforceai/audit/events', ({ request }) => {
          receivedUrl = new URL(request.url);
          return HttpResponse.json({
            items: [],
            next_cursor: null,
            server_time: new Date().toISOString(),
          });
        })
      );

      await getAuditEvents({
        since: '2024-01-15T00:00:00Z',
        until: '2024-01-15T23:59:59Z',
      });

      expect(receivedUrl).not.toBeNull();
      expect(receivedUrl!.searchParams.get('since')).toBe('2024-01-15T00:00:00Z');
      expect(receivedUrl!.searchParams.get('until')).toBe('2024-01-15T23:59:59Z');
    });

    it('includes limit and cursor parameters', async () => {
      let receivedUrl: URL | null = null;

      server.use(
        http.get('/enforceai/audit/events', ({ request }) => {
          receivedUrl = new URL(request.url);
          return HttpResponse.json({
            items: [],
            next_cursor: null,
            server_time: new Date().toISOString(),
          });
        })
      );

      await getAuditEvents({
        limit: 50,
        cursor: '12345',
      });

      expect(receivedUrl).not.toBeNull();
      expect(receivedUrl!.searchParams.get('limit')).toBe('50');
      expect(receivedUrl!.searchParams.get('cursor')).toBe('12345');
    });

    it('includes agent_id parameter', async () => {
      let receivedUrl: URL | null = null;

      server.use(
        http.get('/enforceai/audit/events', ({ request }) => {
          receivedUrl = new URL(request.url);
          return HttpResponse.json({
            items: [],
            next_cursor: null,
            server_time: new Date().toISOString(),
          });
        })
      );

      await getAuditEvents({
        agent_id: '9d2724e9-1753-4493-8993-0d6986754414',
      });

      expect(receivedUrl).not.toBeNull();
      expect(receivedUrl!.searchParams.get('agent_id')).toBe(
        '9d2724e9-1753-4493-8993-0d6986754414'
      );
    });

    it('includes multiple action parameters', async () => {
      let receivedUrl: URL | null = null;

      server.use(
        http.get('/enforceai/audit/events', ({ request }) => {
          receivedUrl = new URL(request.url);
          return HttpResponse.json({
            items: [],
            next_cursor: null,
            server_time: new Date().toISOString(),
          });
        })
      );

      await getAuditEvents({
        action: ['tools/list', 'tools/call'],
      });

      expect(receivedUrl).not.toBeNull();
      const actions = receivedUrl!.searchParams.getAll('action');
      expect(actions).toEqual(['tools/list', 'tools/call']);
    });

    it('includes multiple outcome parameters', async () => {
      let receivedUrl: URL | null = null;

      server.use(
        http.get('/enforceai/audit/events', ({ request }) => {
          receivedUrl = new URL(request.url);
          return HttpResponse.json({
            items: [],
            next_cursor: null,
            server_time: new Date().toISOString(),
          });
        })
      );

      await getAuditEvents({
        outcome: ['allow', 'deny'],
      });

      expect(receivedUrl).not.toBeNull();
      const outcomes = receivedUrl!.searchParams.getAll('outcome');
      expect(outcomes).toEqual(['allow', 'deny']);
    });

    it('includes request_id, server, and tool parameters', async () => {
      let receivedUrl: URL | null = null;

      server.use(
        http.get('/enforceai/audit/events', ({ request }) => {
          receivedUrl = new URL(request.url);
          return HttpResponse.json({
            items: [],
            next_cursor: null,
            server_time: new Date().toISOString(),
          });
        })
      );

      await getAuditEvents({
        request_id: 'req-001',
        server: 'sqlite',
        tool: 'read_query',
      });

      expect(receivedUrl).not.toBeNull();
      expect(receivedUrl!.searchParams.get('request_id')).toBe('req-001');
      expect(receivedUrl!.searchParams.get('server')).toBe('sqlite');
      expect(receivedUrl!.searchParams.get('tool')).toBe('read_query');
    });

    it('handles paginated responses with next_cursor', async () => {
      const mockResponse: AuditEventsListResponse = {
        items: [
          {
            event_id: 10,
            occurred_at: '2024-01-15T10:00:00Z',
            user_id: 'test|user123',
            agent_id: '9d2724e9-1753-4493-8993-0d6986754414',
            action: 'tools/call',
            outcome: 'allow',
            request_id: null,
            details: null,
          },
        ],
        next_cursor: '10',
        server_time: '2024-01-15T10:30:00Z',
      };

      server.use(
        http.get('/enforceai/audit/events', () => {
          return HttpResponse.json(mockResponse);
        })
      );

      const result = await getAuditEvents({ limit: 1 });

      expect(result.items).toHaveLength(1);
      expect(result.next_cursor).toBe('10');
    });

    it('omits empty/undefined parameters from query string', async () => {
      let receivedUrl: URL | null = null;

      server.use(
        http.get('/enforceai/audit/events', ({ request }) => {
          receivedUrl = new URL(request.url);
          return HttpResponse.json({
            items: [],
            next_cursor: null,
            server_time: new Date().toISOString(),
          });
        })
      );

      await getAuditEvents({
        action: [], // Empty array
        outcome: undefined,
      });

      expect(receivedUrl).not.toBeNull();
      // Empty arrays should not generate params
      expect(receivedUrl!.searchParams.getAll('action')).toEqual([]);
      expect(receivedUrl!.searchParams.has('outcome')).toBe(false);
    });
  });

  describe('exportAdminAuditEvents', () => {
    it('downloads a CSV file using the export endpoint', async () => {
      const createObjectUrl = vi.fn(() => 'blob:mock');
      const revokeObjectUrl = vi.fn();
      const originalCreateObjectUrl = window.URL.createObjectURL;
      const originalRevokeObjectUrl = window.URL.revokeObjectURL;

      window.URL.createObjectURL = createObjectUrl;
      window.URL.revokeObjectURL = revokeObjectUrl;

      const clickSpy = vi.spyOn(HTMLAnchorElement.prototype, 'click').mockImplementation(() => {});

      const originalCreateElement = document.createElement.bind(document);
      let createdAnchor: HTMLAnchorElement | null = null;
      const createElementSpy = vi.spyOn(document, 'createElement').mockImplementation((tagName: string) => {
        const el = originalCreateElement(tagName);
        if (tagName.toLowerCase() === 'a') {
          createdAnchor = el as HTMLAnchorElement;
        }
        return el;
      });

      try {
        await exportAdminAuditEvents({ user_id: 'test|user123' });

        expect(createObjectUrl).toHaveBeenCalledTimes(1);
        expect(clickSpy).toHaveBeenCalledTimes(1);
        expect(createdAnchor?.download).toBe('audit_events_mock.csv');
      } finally {
        clickSpy.mockRestore();
        createElementSpy.mockRestore();
        window.URL.createObjectURL = originalCreateObjectUrl;
        window.URL.revokeObjectURL = originalRevokeObjectUrl;
      }
    });
  });
});
