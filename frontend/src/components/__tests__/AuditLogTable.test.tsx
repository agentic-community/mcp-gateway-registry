import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import axios from 'axios';
import AuditLogTable from '../AuditLogTable';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

const TOTAL = 1500;

/** Answer /api/audit/events with `limit` distinct events starting at `offset`. */
function _serveEvents(url: string) {
  const params = new URL(url, 'http://registry.test').searchParams;
  const offset = Number(params.get('offset'));
  const limit = Number(params.get('limit'));
  const count = Math.max(0, Math.min(limit, TOTAL - offset));
  const events = Array.from({ length: count }, (_, i) => ({
    _id: `${params.get('sort_order')}-${offset + i}`,
    log_type: 'registry_api_access',
    request_id: `r${offset + i}`,
    timestamp: '2026-11-07T10:30:12+00:00',
  }));
  return Promise.resolve({ data: { events, total: TOTAL } });
}

/** The (sort_order, offset) pairs the grid requested, in order. */
const _requests = () =>
  mockedAxios.get.mock.calls.map(([url]) => {
    const params = new URL(String(url), 'http://registry.test').searchParams;
    return `${params.get('sort_order')}@${params.get('offset')}`;
  });

describe('AuditLogTable load order', () => {
  beforeEach(() => {
    mockedAxios.get.mockReset();
    mockedAxios.get.mockImplementation((url: string) => _serveEvents(url));
  });

  it('loads newest first by default', async () => {
    render(<AuditLogTable filters={{ stream: 'registry_api' }} />);

    await waitFor(() => expect(_requests()).toEqual(['-1@0', '-1@500']));
  });

  it('reloads from the oldest event and keeps that order for "load more"', async () => {
    render(<AuditLogTable filters={{ stream: 'registry_api' }} />);
    await waitFor(() => expect(_requests()).toHaveLength(2));
    mockedAxios.get.mockClear();

    await userEvent.selectOptions(screen.getByLabelText('Load'), 'oldest');
    await waitFor(() => expect(_requests()).toEqual(['1@0', '1@500']));

    mockedAxios.get.mockClear();
    await userEvent.click(await screen.findByRole('button', { name: /Load 500 more/ }));
    await waitFor(() => expect(_requests()).toEqual(['1@1000']));
  });
});
