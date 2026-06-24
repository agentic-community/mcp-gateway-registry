import React from 'react';
import { render, screen, waitFor, fireEvent } from '@testing-library/react';
import axios from 'axios';
import ServerConfigModal from '../ServerConfigModal';
import type { Server } from '../ServerCard';

// Mock axios so the connect-config / csrf / token fetches resolve deterministically.
jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Mock the useRegistryConfig hook
const mockUseRegistryConfig = jest.fn();
jest.mock('../../hooks/useRegistryConfig', () => ({
  useRegistryConfig: () => mockUseRegistryConfig(),
}));

// Mock clipboard API
Object.assign(navigator, {
  clipboard: { writeText: jest.fn().mockResolvedValue(undefined) },
});

const baseServer: Server = {
  name: 'Test Server',
  path: '/test-server',
  enabled: true,
  proxy_pass_url: 'http://internal-host:8080/mcp',
};

// Mutable connect-config payload for the current test.
let connectConfig: Record<string, unknown> = { custom_headers: [] };

function withGatewayConfig() {
  return {
    config: {
      deployment_mode: 'with-gateway',
      registry_mode: 'full',
      nginx_updates_enabled: true,
      features: { mcp_servers: true, agents: true, skills: true, federation: true, gateway_proxy: true },
    },
    loading: false,
    error: null,
  };
}

function registryOnlyConfig() {
  return {
    config: {
      deployment_mode: 'registry-only',
      registry_mode: 'full',
      nginx_updates_enabled: false,
      features: { mcp_servers: true, agents: true, skills: true, federation: true, gateway_proxy: false },
    },
    loading: false,
    error: null,
  };
}

function renderModal(serverOverrides: Partial<Server> = {}) {
  const server = { ...baseServer, ...serverOverrides };
  return render(
    <ServerConfigModal
      server={server}
      isOpen={true}
      onClose={jest.fn()}
      onShowToast={jest.fn()}
    />
  );
}

function getDisplayedConfig(): any {
  // The config JSON is rendered inside a <pre> tag
  const preElement = screen.getByText(/{/, { selector: 'pre' });
  return JSON.parse(preElement.textContent || '');
}

beforeEach(() => {
  jest.clearAllMocks();
  connectConfig = { custom_headers: [] };
  mockedAxios.get.mockImplementation((url: string) => {
    if (url.includes('csrf-token')) {
      return Promise.resolve({ data: { csrf_token: 'test-csrf' } });
    }
    if (url.includes('connect-config')) {
      return Promise.resolve({ data: connectConfig });
    }
    return Promise.resolve({ data: {} });
  });
  mockedAxios.post.mockResolvedValue({ data: { token: 'test-jwt' } });
  mockUseRegistryConfig.mockReturnValue(withGatewayConfig());
});

describe('ServerConfigModal URL generation', () => {
  test('should use gateway URL in with-gateway mode', async () => {
    // proxy_pass_url without an MCP transport suffix, so the gateway appends /mcp
    renderModal({ proxy_pass_url: 'http://internal-host:8080' });

    await waitFor(() => {
      // Cursor is the default IDE — config uses the "mcpServers" key
      const serverConfig = getDisplayedConfig().mcpServers['test-server'];
      expect(serverConfig.url).toBe('http://localhost/test-server/mcp');
      // Gateway mode embeds the gateway token via the X-Authorization header
      expect(serverConfig.headers['X-Authorization']).toContain('Bearer');
    });
  });

  test('should use proxy_pass_url in registry-only mode', async () => {
    mockUseRegistryConfig.mockReturnValue(registryOnlyConfig());

    renderModal({ proxy_pass_url: 'http://internal-host:8080/mcp' });

    await waitFor(() => {
      const serverConfig = getDisplayedConfig().mcpServers['test-server'];
      expect(serverConfig.url).toBe('http://internal-host:8080/mcp');
      // Registry-only mode should NOT include auth headers
      expect(serverConfig.headers).toBeUndefined();
    });
  });

  test('should always use mcp_endpoint when provided', async () => {
    renderModal({
      mcp_endpoint: 'https://custom-endpoint.example.com/mcp',
      proxy_pass_url: 'http://internal-host:8080/mcp',
    });

    await waitFor(() => {
      const serverConfig = getDisplayedConfig().mcpServers['test-server'];
      expect(serverConfig.url).toBe('https://custom-endpoint.example.com/mcp');
    });
  });
});

describe('ServerConfigModal per-server append_mcp_path override', () => {
  test('append_mcp_path:false strips the auto /mcp suffix', async () => {
    // proxy_pass_url has no transport suffix → would normally get /mcp appended
    connectConfig = { custom_headers: [], append_mcp_path: false };

    renderModal({ proxy_pass_url: 'http://internal-host:8080' });

    await waitFor(() => {
      const serverConfig = getDisplayedConfig().mcpServers['test-server'];
      expect(serverConfig.url).toBe('http://localhost/test-server');
    });
  });

  test('append_mcp_path:true forces the /mcp suffix even when proxy already ends in /mcp', async () => {
    connectConfig = { custom_headers: [], append_mcp_path: true };

    renderModal({ proxy_pass_url: 'http://internal-host:8080/mcp' });

    await waitFor(() => {
      const serverConfig = getDisplayedConfig().mcpServers['test-server'];
      expect(serverConfig.url).toBe('http://localhost/test-server/mcp');
    });
  });
});

describe('ServerConfigModal IDE OAuth login (oauth_client_id)', () => {
  test('Cursor: emits auth.CLIENT_ID and omits the gateway token when oauth_client_id is set', async () => {
    connectConfig = { custom_headers: [], oauth_client_id: 'mcp-gateway' };

    renderModal();

    await waitFor(() => {
      const serverConfig = getDisplayedConfig().mcpServers['test-server'];
      expect(serverConfig.auth).toEqual({ CLIENT_ID: 'mcp-gateway' });
    });
    const serverConfig = getDisplayedConfig().mcpServers['test-server'];
    expect(serverConfig.headers).toBeUndefined();
  });

  test('Cursor: omits the server Authorization header for a bearer server under OAuth login', async () => {
    // The gateway injects the stored egress credential upstream, so the client
    // config must not carry the [YOUR_SERVER_AUTH_TOKEN] placeholder.
    connectConfig = { custom_headers: [], oauth_client_id: 'mcp-gateway' };

    renderModal({ auth_scheme: 'bearer' } as Partial<Server>);

    await waitFor(() => {
      const serverConfig = getDisplayedConfig().mcpServers['test-server'];
      expect(serverConfig.auth).toEqual({ CLIENT_ID: 'mcp-gateway' });
    });
    const serverConfig = getDisplayedConfig().mcpServers['test-server'];
    // No headers block at all: gateway token omitted (OAuth) and server auth
    // header suppressed.
    expect(serverConfig.headers).toBeUndefined();
  });

  test('Roo Code: drops server Authorization but keeps the static gateway token under OAuth login', async () => {
    // Roo Code can't run the IDE OAuth-login config, so it keeps the static
    // X-Authorization gateway token — but the gateway still injects the egress
    // credential, so the server Authorization header must be omitted.
    connectConfig = { custom_headers: [], oauth_client_id: 'mcp-gateway' };

    renderModal({ auth_scheme: 'bearer' } as Partial<Server>);

    await waitFor(() => screen.getByRole('button', { name: 'Roo Code' }));
    fireEvent.click(screen.getByRole('button', { name: 'Roo Code' }));

    await waitFor(() => {
      const serverConfig = getDisplayedConfig().mcpServers['test-server'];
      expect(serverConfig.type).toBe('streamable-http');
    });
    const serverConfig = getDisplayedConfig().mcpServers['test-server'];
    expect(serverConfig.headers['X-Authorization']).toContain('Bearer');
    expect(serverConfig.headers.Authorization).toBeUndefined();
  });

  test('Kiro: drops server Authorization but keeps the static gateway token under OAuth login', async () => {
    connectConfig = { custom_headers: [], oauth_client_id: 'mcp-gateway' };

    renderModal({ auth_scheme: 'bearer' } as Partial<Server>);

    await waitFor(() => screen.getByRole('button', { name: 'Kiro' }));
    fireEvent.click(screen.getByRole('button', { name: 'Kiro' }));

    await waitFor(() => {
      const serverConfig = getDisplayedConfig().mcpServers['test-server'];
      expect(serverConfig.autoApprove).toBeDefined();
    });
    const serverConfig = getDisplayedConfig().mcpServers['test-server'];
    expect(serverConfig.headers['X-Authorization']).toContain('Bearer');
    expect(serverConfig.headers.Authorization).toBeUndefined();
  });

  test('Cursor: keeps the static gateway token when oauth_client_id is absent', async () => {
    connectConfig = { custom_headers: [] };

    renderModal();

    await waitFor(() => {
      const serverConfig = getDisplayedConfig().mcpServers['test-server'];
      expect(serverConfig.headers['X-Authorization']).toContain('Bearer');
    });
    const serverConfig = getDisplayedConfig().mcpServers['test-server'];
    expect(serverConfig.auth).toBeUndefined();
  });

  test('registry-only deployment never enables OAuth login', async () => {
    mockUseRegistryConfig.mockReturnValue(registryOnlyConfig());
    connectConfig = { custom_headers: [], oauth_client_id: 'mcp-gateway' };

    renderModal({ proxy_pass_url: 'http://internal-host:8080/mcp' });

    await waitFor(() => {
      expect(mockedAxios.get).toHaveBeenCalledWith(
        expect.stringContaining('connect-config'),
        expect.anything()
      );
    });
    const serverConfig = getDisplayedConfig().mcpServers['test-server'];
    expect(serverConfig.auth).toBeUndefined();
    expect(serverConfig.headers).toBeUndefined();
  });
});
