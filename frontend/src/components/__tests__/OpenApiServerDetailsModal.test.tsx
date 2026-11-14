/**
 * OpenApiServerDetailsModal (issue #2022): polling while a sync is pending,
 * the credential shown only as yes or no, and third-party text and URLs
 * rendered inertly.
 */
import React from 'react';
import { render, screen, act } from '@testing-library/react';
import axios from 'axios';

import OpenApiServerDetailsModal, {
  OPENAPI_POLL_INTERVAL_MS,
  OPENAPI_POLL_MAX_MS,
} from '../OpenApiServerDetailsModal';
import { OpenApiServer } from '../../types/openApiServer';

jest.mock('axios');

const mockedAxios = axios as jest.Mocked<typeof axios>;


function _server(
  overrides: Partial<OpenApiServer> = {},
): OpenApiServer {
  return {
    slug: 'weather-apis',
    name: 'Weather APIs',
    description: 'Forecasts',
    path: '/weather-apis',
    mcp_url: 'https://registry.example.com/weather-apis/mcp',
    sync_status: 'applied',
    targets: [
      {
        name: 'petstore',
        spec_title: 'Petstore',
        spec_version: '1.0.0',
        base_url: 'https://petstore.example.com/api',
        auth: { scheme: 'bearer', credential_set: true },
        tool_names: ['petstore_getPetById'],
        status: 'ok',
      },
    ],
    ...overrides,
  };
}


function _serverGets(): number {
  return mockedAxios.get.mock.calls.filter((c) => c[0] === '/api/openapi-servers/weather-apis').length;
}


async function _advance(ms: number) {
  await act(async () => {
    jest.advanceTimersByTime(ms);
  });
}


async function _renderModal() {
  await act(async () => {
    render(<OpenApiServerDetailsModal slug="weather-apis" isOpen onClose={jest.fn()} />);
  });
}


beforeEach(() => {
  jest.resetAllMocks();
});


describe('OpenApiServerDetailsModal polling', () => {
  beforeEach(() => {
    jest.useFakeTimers();
  });

  afterEach(() => {
    jest.useRealTimers();
  });

  it('polls every 5 seconds while pending and stops after 2 minutes', async () => {
    mockedAxios.get.mockResolvedValue({ data: _server({ sync_status: 'pending' }) });
    await _renderModal();
    expect(_serverGets()).toBe(1);

    await _advance(OPENAPI_POLL_INTERVAL_MS);
    expect(_serverGets()).toBe(2);

    const maxPolls = OPENAPI_POLL_MAX_MS / OPENAPI_POLL_INTERVAL_MS;
    for (let i = 1; i < maxPolls; i++) {
      await _advance(OPENAPI_POLL_INTERVAL_MS);
    }
    expect(_serverGets()).toBe(1 + maxPolls);
    expect(screen.getByRole('status')).toHaveTextContent('Still syncing. Check the openapi-mcp container.');

    await _advance(OPENAPI_POLL_INTERVAL_MS * 4);
    expect(_serverGets()).toBe(1 + maxPolls);
  });

  it('stops polling once the server is applied', async () => {
    mockedAxios.get
      .mockResolvedValueOnce({ data: _server({ sync_status: 'pending' }) })
      .mockResolvedValue({ data: _server({ sync_status: 'applied' }) });
    await _renderModal();

    await _advance(OPENAPI_POLL_INTERVAL_MS);
    expect(_serverGets()).toBe(2);

    await _advance(OPENAPI_POLL_INTERVAL_MS * 3);
    expect(_serverGets()).toBe(2);
    expect(screen.queryByRole('status')).not.toBeInTheDocument();
  });

  it('does not poll when nothing is pending', async () => {
    mockedAxios.get.mockResolvedValue({ data: _server() });
    await _renderModal();

    await _advance(OPENAPI_POLL_INTERVAL_MS * 3);
    expect(_serverGets()).toBe(1);
  });
});


describe('OpenApiServerDetailsModal rendering', () => {
  it('shows only whether a credential is set', async () => {
    mockedAxios.get.mockResolvedValue({ data: _server() });
    await _renderModal();

    expect(screen.getByTestId('openapi-credential-set-petstore')).toHaveTextContent('yes');
  });

  it('offers Refresh spec only for a target fetched from a URL', async () => {
    const server = _server();
    server.targets.push({ ...server.targets[0], name: 'uploaded', spec_source: 'upload' });
    mockedAxios.get.mockResolvedValue({ data: server });
    await _renderModal();

    expect(screen.getAllByRole('button', { name: 'Refresh spec' })).toHaveLength(1);
  });

  it('renders spec titles as plain text, not HTML', async () => {
    const server = _server();
    server.targets[0].spec_title = '<img src=x onerror="alert(1)">Evil';
    mockedAxios.get.mockResolvedValue({ data: server });
    await _renderModal();

    expect(screen.getByText(/<img src=x onerror="alert\(1\)">Evil/)).toBeInTheDocument();
    expect(document.querySelector('img[src="x"]')).toBeNull();
  });

  it('renders an unsafe base URL as text, not a link', async () => {
    const server = _server();
    server.targets[0].base_url = 'javascript:alert(1)';
    mockedAxios.get.mockResolvedValue({ data: server });
    await _renderModal();

    const node = screen.getByText('javascript:alert(1)');
    expect(node.tagName).not.toBe('A');
    expect(node.closest('a')).toBeNull();
  });
});


describe('OpenApiServerDetailsModal per-user targets', () => {
  function _perUserServer(): OpenApiServer {
    const server = _server();
    server.targets[0].auth = { scheme: 'bearer', credential_set: false, credential_mode: 'per_user' };
    return server;
  }

  function _mockGets(server: OpenApiServer) {
    mockedAxios.get.mockImplementation((url: string) => {
      if (url === '/api/openapi-servers/weather-apis/my-credentials') {
        return Promise.resolve({
          data: {
            slug: 'weather-apis',
            targets: [{ target: 'petstore', configured: false, expires_at: null }],
          },
        });
      }
      return Promise.resolve({ data: server });
    });
  }

  it('shows a Per user badge and no Rotate credential', async () => {
    _mockGets(_perUserServer());
    await _renderModal();

    expect(screen.getByTestId('openapi-credential-set-petstore')).toHaveTextContent('Per user');
    expect(screen.queryByRole('button', { name: 'Rotate credential' })).not.toBeInTheDocument();
  });

  it("shows the current user's key status from my-credentials", async () => {
    _mockGets(_perUserServer());
    await _renderModal();

    expect(await screen.findByTestId('openapi-your-keys')).toBeInTheDocument();
    expect(screen.getByTestId('openapi-my-key-status-petstore')).toHaveTextContent('Not set');
    expect(screen.getByRole('button', { name: 'Add key' })).toBeInTheDocument();
  });

  it('does not load or show Your key for shared targets', async () => {
    mockedAxios.get.mockResolvedValue({ data: _server() });
    await _renderModal();

    expect(screen.queryByTestId('openapi-your-keys')).not.toBeInTheDocument();
    const urls = mockedAxios.get.mock.calls.map((c) => c[0]);
    expect(urls).not.toContain('/api/openapi-servers/weather-apis/my-credentials');
  });
});
