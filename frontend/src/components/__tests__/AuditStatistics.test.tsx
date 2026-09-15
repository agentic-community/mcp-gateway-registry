import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import axios from 'axios';
import AuditStatistics from '../AuditStatistics';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

// The statistics endpoint buckets `$response.status_code`. token_mint records
// carry no `response` block, so that bucket is structurally always empty for
// the token_mint stream and the card must not be rendered at all.
const statistics = {
  total_events: 12,
  top_users: [{ name: 'alice@example.com', count: 12 }],
  top_servers: [],
  top_operations: [{ name: 'resource', count: 12 }],
  activity_timeline: [{ period: '2026-09-14', count: 12 }],
  activity_timeline_prior: [],
  status_distribution: { status_2xx: 0, status_4xx: 0, status_5xx: 0 },
  user_activity: [
    { username: 'alice@example.com', total: 12, operations: [{ name: 'resource', count: 12 }] },
  ],
};

describe('AuditStatistics status distribution card', () => {
  beforeEach(() => {
    mockedAxios.get.mockReset();
    mockedAxios.get.mockResolvedValue({ data: statistics });
    localStorage.clear();
  });

  it('omits the card for the token_mint stream, which has no status codes', async () => {
    render(<AuditStatistics stream="token_mint" />);

    await waitFor(() =>
      expect(screen.getByRole('heading', { name: 'Top Operations' })).toBeInTheDocument(),
    );
    expect(screen.queryByText('Status Distribution')).not.toBeInTheDocument();
    // ...and therefore no titled panel whose only content is an empty state.
    expect(screen.queryByText('No data available')).not.toBeInTheDocument();
  });

  it('keeps the card for streams that do carry status codes', async () => {
    render(<AuditStatistics stream="registry_api" />);

    await waitFor(() => expect(screen.getByText('Status Distribution')).toBeInTheDocument());
  });
});
