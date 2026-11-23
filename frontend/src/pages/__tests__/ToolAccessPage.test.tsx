import React from 'react';
import { render, screen, fireEvent, waitFor } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import axios from 'axios';

import ToolAccessPage from '../ToolAccessPage';
import type { ToolAccessGroup, ToolAccessResponse, ToolAccessRow } from '../../hooks/useToolAccess';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

const GROUPS: Record<string, ToolAccessGroup> = {
  'mcp-registry-admin': { group: 'mcp-registry-admin', idp_groups: ['mcp-registry-admin'], is_admin: true },
  'docs-readers': { group: 'docs-readers', idp_groups: ['eng-docs'], is_admin: false },
  'docs-callers': { group: 'docs-callers', idp_groups: [], is_admin: false },
};

const row = (overrides: Partial<ToolAccessRow>): ToolAccessRow => ({
  tool_name: 'search',
  server_name: 'Docs',
  server_path: '/docs/',
  server_type: 'mcp',
  enabled: true,
  blocked: false,
  blocked_source: null,
  backend_server_path: null,
  backend_tool_name: null,
  can_see: { alone: ['mcp-registry-admin', 'docs-readers'], together: [] },
  can_call: { alone: ['mcp-registry-admin', 'docs-readers'], together: [] },
  ...overrides,
});

const response = (rows: ToolAccessRow[], total = rows.length): { data: ToolAccessResponse } => ({
  data: {
    rows,
    groups: GROUPS,
    total_count: total,
    limit: 20,
    offset: 0,
    has_next: total > rows.length,
    generated_at: '2026-11-22T18:00:00Z',
  },
});

const renderPage = (url = '/tool-access') =>
  render(
    <MemoryRouter initialEntries={[url]}>
      <ToolAccessPage />
    </MemoryRouter>,
  );

describe('ToolAccessPage', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockedAxios.isAxiosError.mockReturnValue(false);
  });

  it('shows group names with IdP groups in the tooltip and one Admins chip', async () => {
    mockedAxios.get.mockResolvedValue(response([row({})]));
    renderPage();
    expect(await screen.findByText('search')).toBeInTheDocument();
    const chip = screen.getAllByText('docs-readers')[0];
    expect(chip).toHaveAttribute('title', 'docs-readers\nIdP groups: eng-docs');
    expect(screen.getAllByText('Admins')).toHaveLength(2);
    expect(screen.queryByText('mcp-registry-admin')).not.toBeInTheDocument();
    expect(screen.queryByText('eng-docs')).not.toBeInTheDocument();
  });

  it('asks for 20 rows per page', async () => {
    mockedAxios.get.mockResolvedValue(response([row({})]));
    renderPage();
    await screen.findByText('search');
    expect(mockedAxios.get.mock.calls[0][1]?.params).toMatchObject({ limit: 20, offset: 0 });
  });

  it('hides combined groups until the toggle is opened', async () => {
    const together = [
      { label: 'Open a session on /docs/ (initialize)', groups: ['docs-readers'] },
      { label: 'Call install on /docs/', groups: ['docs-callers'] },
    ];
    mockedAxios.get.mockResolvedValue(
      response([row({ tool_name: 'install', blocked: true, can_call: { alone: ['mcp-registry-admin'], together } })]),
    );
    renderPage();
    expect(await screen.findByText('Blocked')).toBeInTheDocument();
    expect(screen.queryByText('Call install on /docs/')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('Combined groups'));
    expect(screen.getByText('Call install on /docs/')).toBeInTheDocument();
    expect(screen.getByText('docs-callers')).toBeInTheDocument();
  });

  it('collapses long group lists behind +N more', async () => {
    const many = ['g1', 'g2', 'g3', 'g4', 'g5'];
    mockedAxios.get.mockResolvedValue(response([row({ can_see: { alone: many, together: [] } })]));
    renderPage();
    expect(await screen.findByText('+2 more')).toBeInTheDocument();
    expect(screen.queryByText('g5')).not.toBeInTheDocument();
    fireEvent.click(screen.getByText('+2 more'));
    expect(screen.getByText('g5')).toBeInTheDocument();
  });

  it('shows a virtual row with its backend tool name', async () => {
    mockedAxios.get.mockResolvedValue(
      response([
        row({
          tool_name: 'docs_search',
          server_type: 'virtual',
          server_path: '/virtual/eng',
          backend_tool_name: 'search',
          can_call: { alone: [], together: [{ label: 'Member of eng-users', groups: ['eng-users'] }] },
        }),
      ]),
    );
    renderPage();
    expect(await screen.findByText('docs_search')).toBeInTheDocument();
    expect(screen.getByText('backend: search')).toBeInTheDocument();
    expect(screen.getByText('No single group')).toBeInTheDocument();
  });

  it('filters as the user types', async () => {
    mockedAxios.get.mockResolvedValue(response([row({})]));
    renderPage();
    await screen.findByText('search');
    fireEvent.change(screen.getByLabelText('Filter by tool, server or group'), { target: { value: 'sea' } });
    await waitFor(() => {
      expect(mockedAxios.get.mock.calls.at(-1)?.[1]?.params).toMatchObject({ q: 'sea' });
    });
  });

  it('filters by a group when its chip is clicked', async () => {
    mockedAxios.get.mockResolvedValue(response([row({})]));
    renderPage();
    fireEvent.click((await screen.findAllByText('docs-readers'))[0]);
    await waitFor(() => {
      expect(mockedAxios.get.mock.calls.at(-1)?.[1]?.params).toMatchObject({ group: 'docs-readers' });
    });
  });

  it('sends the server filter from the URL', async () => {
    mockedAxios.get.mockResolvedValue(response([row({})]));
    renderPage('/tool-access?server_path=%2Fdocs%2F');
    await screen.findByText('search');
    expect(mockedAxios.get.mock.calls[0][1]?.params).toMatchObject({ server_path: '/docs/' });
    expect(screen.getByText('Server /docs/')).toBeInTheDocument();
  });

  it('shows the empty state when the caller has no rows', async () => {
    mockedAxios.get.mockResolvedValue(response([]));
    renderPage();
    expect(await screen.findByText(/No tools to show/)).toBeInTheDocument();
  });

  it('shows the error detail from the API', async () => {
    mockedAxios.get.mockRejectedValue({ response: { data: { detail: 'Could not load tool access.' } } });
    mockedAxios.isAxiosError.mockReturnValue(true);
    renderPage();
    expect(await screen.findByText('Could not load tool access.')).toBeInTheDocument();
  });
});
