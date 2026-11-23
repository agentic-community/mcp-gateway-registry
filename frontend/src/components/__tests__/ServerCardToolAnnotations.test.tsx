/**
 * MCP tool annotations in the server card's tools modal: READ ONLY and
 * DESTRUCTIVE badges appear only for hints the server declared.
 */
import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import axios from 'axios';

import ServerCard, { Server } from '../ServerCard';

jest.mock('react-markdown', () => {
  return ({ children }: { children: React.ReactNode }) => <div>{children}</div>;
});
jest.mock('remark-gfm', () => ({ __esModule: true, default: () => {} }));
jest.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({
    user: {
      username: 'admin',
      is_admin: true,
      can_modify_servers: true,
      ui_permissions: {},
    },
  }),
}));
jest.mock('axios');
jest.mock('../ServerConfigModal', () => () => null);
jest.mock('../SecurityScanModal', () => () => null);
jest.mock('../VersionSelectorModal', () => () => null);
jest.mock('../ServerDetailsModal', () => () => null);
jest.mock('../PullCardPreviewModal', () => () => null);
jest.mock('../ResourceBoundTokenButton', () => () => null);
jest.mock('../StarRatingWidget', () => () => <div data-testid="stars" />);

const mockedAxios = axios as jest.Mocked<typeof axios>;
const noop = () => {};


function _server(): Server {
  return {
    name: 'Notes',
    path: '/notes',
    enabled: true,
    num_tools: 5,
  };
}


describe('ServerCard tools modal annotation badges', () => {
  beforeEach(() => {
    mockedAxios.get.mockResolvedValue({
      data: {
        tools: [
          { name: 'read_notes', annotations: { readOnlyHint: true } },
          { name: 'delete_note', annotations: { readOnlyHint: false, destructiveHint: true } },
          { name: 'update_note', annotations: { readOnlyHint: false } },
          { name: 'odd_note', annotations: { readOnlyHint: true, destructiveHint: true } },
          { name: 'ping' },
        ],
      },
    });
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  it('shows only the declared hints, and never DESTRUCTIVE on a read-only tool', async () => {
    render(<ServerCard server={_server()} onToggle={noop} />);

    fireEvent.click(screen.getByTitle('View tools'));

    await screen.findByText('delete_note');
    // read_notes and odd_note are read-only; only delete_note is destructive.
    // update_note declares readOnlyHint false but no destructiveHint, so it gets
    // no badge: the UI does not apply the spec's destructive default.
    expect(screen.getAllByText('READ ONLY')).toHaveLength(2);
    expect(screen.getAllByText('DESTRUCTIVE')).toHaveLength(1);
  });

  it('says in the tooltip that the registry does not verify the hint', async () => {
    render(<ServerCard server={_server()} onToggle={noop} />);

    fireEvent.click(screen.getByTitle('View tools'));

    const badge = await screen.findByTitle(/declares this tool can delete or overwrite data/);
    expect(badge.getAttribute('title')).toMatch(/does not verify/);
  });
});
