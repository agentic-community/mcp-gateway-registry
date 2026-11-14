/**
 * "Manage APIs" on the ordinary server card (issue #2022): shown only to an
 * admin, only with the feature on, and only on an OpenAPI MCP server.
 * "Your API keys" shows for any user, only on an OpenAPI MCP server that has
 * per-user targets.
 */
import React from 'react';
import { render, screen, waitFor } from '@testing-library/react';
import axios from 'axios';

import ServerCard, { Server } from '../ServerCard';

let mockIsAdmin = true;

jest.mock('react-markdown', () => {
  return ({ children }: { children: React.ReactNode }) => <div>{children}</div>;
});
jest.mock('remark-gfm', () => ({ __esModule: true, default: () => {} }));
jest.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({
    user: { username: 'u', is_admin: mockIsAdmin, can_modify_servers: true, ui_permissions: {} },
  }),
}));
jest.mock('axios');
jest.mock('../ServerConfigModal', () => () => null);
jest.mock('../SecurityScanModal', () => () => null);
jest.mock('../VersionSelectorModal', () => () => null);
jest.mock('../ServerDetailsModal', () => () => null);
jest.mock('../PullCardPreviewModal', () => () => null);
jest.mock('../ResourceBoundTokenButton', () => () => null);
jest.mock('../OpenApiServerDetailsModal', () => () => null);
jest.mock('../OpenApiMyKeysModal', () => () => null);
jest.mock('../StarRatingWidget', () => () => <div data-testid="stars" />);

const noop = () => {};
const mockedAxios = axios as jest.Mocked<typeof axios>;


function _server(
  overrides: Partial<Server> = {},
): Server {
  return {
    name: 'Weather APIs',
    path: '/weather-apis',
    enabled: true,
    openapi_server_slug: 'weather-apis',
    ...overrides,
  };
}


beforeEach(() => {
  jest.clearAllMocks();
  mockIsAdmin = true;
});


describe('ServerCard Manage APIs link', () => {
  it('shows Manage APIs to an admin when the feature is on', () => {
    render(<ServerCard server={_server()} onToggle={noop} openApiEnabled />);

    expect(screen.getByRole('button', { name: /Manage APIs/ })).toBeInTheDocument();
  });

  it('hides Manage APIs when the feature is off', () => {
    render(<ServerCard server={_server()} onToggle={noop} openApiEnabled={false} />);

    expect(screen.queryByRole('button', { name: /Manage APIs/ })).not.toBeInTheDocument();
  });

  it('hides Manage APIs from a non-admin', () => {
    mockIsAdmin = false;
    render(<ServerCard server={_server()} onToggle={noop} openApiEnabled />);

    expect(screen.queryByRole('button', { name: /Manage APIs/ })).not.toBeInTheDocument();
  });

  it('hides Manage APIs on an ordinary server', () => {
    render(
      <ServerCard server={_server({ openapi_server_slug: undefined })} onToggle={noop} openApiEnabled />,
    );

    expect(screen.queryByRole('button', { name: /Manage APIs/ })).not.toBeInTheDocument();
  });
});


describe('ServerCard Your API keys button', () => {
  const MY_CREDS_URL = '/api/openapi-servers/weather-apis/my-credentials';

  function _mockMyCredentials(targets: Array<Record<string, unknown>>) {
    mockedAxios.get.mockImplementation((url: string) => {
      if (url === MY_CREDS_URL) {
        return Promise.resolve({ data: { slug: 'weather-apis', targets } });
      }
      return Promise.resolve({ data: {} });
    });
  }

  it('shows Your API keys to a non-admin on an OpenAPI server with per-user targets', async () => {
    mockIsAdmin = false;
    _mockMyCredentials([{ target: 'petstore', configured: false, expires_at: null }]);
    render(<ServerCard server={_server()} onToggle={noop} />);

    expect(await screen.findByRole('button', { name: /Your API keys/ })).toBeInTheDocument();
  });

  it('shows nothing extra when the OpenAPI server has no per-user targets', async () => {
    _mockMyCredentials([]);
    render(<ServerCard server={_server()} onToggle={noop} />);

    await waitFor(() => expect(mockedAxios.get).toHaveBeenCalledWith(MY_CREDS_URL));
    expect(screen.queryByRole('button', { name: /Your API keys/ })).not.toBeInTheDocument();
  });

  it('does not show or fetch Your API keys on an ordinary server', async () => {
    _mockMyCredentials([{ target: 'petstore', configured: false, expires_at: null }]);
    render(<ServerCard server={_server({ openapi_server_slug: undefined })} onToggle={noop} />);

    await waitFor(() => expect(screen.getByText('Weather APIs')).toBeInTheDocument());
    expect(screen.queryByRole('button', { name: /Your API keys/ })).not.toBeInTheDocument();
    const urls = mockedAxios.get.mock.calls.map((c) => c[0]);
    expect(urls).not.toContain(MY_CREDS_URL);
  });
});
