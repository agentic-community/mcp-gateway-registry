/**
 * MCP Apps markers on the server card: the header badge and the per-tool
 * badges in the tools modal.
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


function _server(
  overrides: Partial<Server> = {},
): Server {
  return {
    name: 'Budget Allocator',
    path: '/mcp-apps-budget',
    enabled: true,
    num_tools: 2,
    ...overrides,
  };
}


describe('ServerCard MCP Apps badge', () => {
  it('shows MCP APP when the server has tools with a page', () => {
    render(<ServerCard server={_server({ num_mcp_app_tools: 1 })} onToggle={noop} />);

    expect(screen.getByText('MCP APP')).toBeInTheDocument();
  });

  it('hides the badge when no tool has a page', () => {
    render(<ServerCard server={_server({ num_mcp_app_tools: 0 })} onToggle={noop} />);

    expect(screen.queryByText('MCP APP')).not.toBeInTheDocument();
  });

  it('hides the badge when the field is missing', () => {
    render(<ServerCard server={_server()} onToggle={noop} />);

    expect(screen.queryByText('MCP APP')).not.toBeInTheDocument();
  });
});


describe('ServerCard tools modal MCP Apps markers', () => {
  beforeEach(() => {
    mockedAxios.get.mockResolvedValue({
      data: {
        tools: [
          {
            name: 'get-budget-data',
            description: 'Open the budget',
            ui_resource_uri: 'ui://budget/view.html',
            ui_visibility: ['model', 'app'],
          },
          {
            name: 'save-budget',
            description: 'Called by the page',
            ui_resource_uri: 'ui://budget/view.html',
            ui_visibility: ['app'],
          },
          { name: 'plain-tool', description: 'No page' },
          { name: 'forged-tool', description: 'Bad address', ui_resource_uri: 'javascript:alert(1)' },
        ],
      },
    });
  });

  afterEach(() => {
    jest.resetAllMocks();
  });

  it('marks tools with a ui:// page and the app-only tool, not a forged address', async () => {
    render(<ServerCard server={_server({ num_tools: 4 })} onToggle={noop} />);

    fireEvent.click(screen.getByTitle('View tools'));

    await screen.findAllByTitle(/^Opens the MCP App page ui:\/\/budget/);
    // No header badge here (num_mcp_app_tools is unset), so every MCP APP label
    // is a per-tool marker: two real pages, none for the forged address.
    expect(screen.getAllByText('MCP APP')).toHaveLength(2);
    expect(screen.getAllByText('APP ONLY')).toHaveLength(1);
  });
});
