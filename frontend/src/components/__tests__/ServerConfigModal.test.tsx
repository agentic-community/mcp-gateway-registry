import React from 'react';
import { render, screen } from '@testing-library/react';
import ServerConfigModal from '../ServerConfigModal';
import type { Server } from '../ServerCard';

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

function renderModal(serverOverrides: Partial<Server> = {}, configOverride?: ReturnType<typeof mockUseRegistryConfig>) {
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

describe('ServerConfigModal URL generation', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    // Default: jsdom sets window.location.origin to http://localhost
  });

  test('should use gateway URL in with-gateway mode', () => {
    mockUseRegistryConfig.mockReturnValue({
      config: {
        deployment_mode: 'with-gateway',
        registry_mode: 'full',
        nginx_updates_enabled: true,
        features: { mcp_servers: true, agents: true, skills: true, federation: true, gateway_proxy: true },
      },
      loading: false,
      error: null,
    });

    // proxy_pass_url without an MCP transport suffix, so the gateway appends /mcp
    renderModal({ proxy_pass_url: 'http://internal-host:8080' });
    const config = getDisplayedConfig();

    // Cursor is the default IDE — config uses the "mcpServers" key
    const serverConfig = config.mcpServers['test-server'];
    expect(serverConfig.url).toBe('http://localhost/test-server/mcp');
    // Gateway mode embeds the gateway token via the X-Authorization header
    expect(serverConfig.headers).toBeDefined();
    expect(serverConfig.headers['X-Authorization']).toContain('Bearer');
  });

  test('should use proxy_pass_url in registry-only mode', () => {
    mockUseRegistryConfig.mockReturnValue({
      config: {
        deployment_mode: 'registry-only',
        registry_mode: 'full',
        nginx_updates_enabled: false,
        features: { mcp_servers: true, agents: true, skills: true, federation: true, gateway_proxy: false },
      },
      loading: false,
      error: null,
    });

    renderModal({ proxy_pass_url: 'http://internal-host:8080/mcp' });
    const config = getDisplayedConfig();

    const serverConfig = config.mcpServers['test-server'];
    expect(serverConfig.url).toBe('http://internal-host:8080/mcp');
    // Registry-only mode should NOT include auth headers
    expect(serverConfig.headers).toBeUndefined();
  });

  test('should always use mcp_endpoint when provided', () => {
    // Test with with-gateway mode
    mockUseRegistryConfig.mockReturnValue({
      config: {
        deployment_mode: 'with-gateway',
        registry_mode: 'full',
        nginx_updates_enabled: true,
        features: { mcp_servers: true, agents: true, skills: true, federation: true, gateway_proxy: true },
      },
      loading: false,
      error: null,
    });

    const { unmount } = renderModal({
      mcp_endpoint: 'https://custom-endpoint.example.com/mcp',
      proxy_pass_url: 'http://internal-host:8080/mcp',
    });
    let config = getDisplayedConfig();
    let serverConfig = config.mcpServers['test-server'];
    expect(serverConfig.url).toBe('https://custom-endpoint.example.com/mcp');

    unmount();

    // Test with registry-only mode — mcp_endpoint still takes precedence
    mockUseRegistryConfig.mockReturnValue({
      config: {
        deployment_mode: 'registry-only',
        registry_mode: 'full',
        nginx_updates_enabled: false,
        features: { mcp_servers: true, agents: true, skills: true, federation: true, gateway_proxy: false },
      },
      loading: false,
      error: null,
    });

    renderModal({
      mcp_endpoint: 'https://custom-endpoint.example.com/mcp',
      proxy_pass_url: 'http://internal-host:8080/mcp',
    });
    config = getDisplayedConfig();
    serverConfig = config.mcpServers['test-server'];
    expect(serverConfig.url).toBe('https://custom-endpoint.example.com/mcp');
  });
});

describe('ServerConfigModal IDE OAuth login (ide_oauth_client_id)', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  function withGatewayConfig(extra: Record<string, unknown> = {}) {
    return {
      config: {
        deployment_mode: 'with-gateway',
        registry_mode: 'full',
        auth_provider: 'keycloak',
        nginx_updates_enabled: true,
        features: { mcp_servers: true, agents: true, skills: true, federation: true, gateway_proxy: true },
        ...extra,
      },
      loading: false,
      error: null,
    };
  }

  test('emits auth.CLIENT_ID and omits the gateway token when ide_oauth_client_id is set', () => {
    mockUseRegistryConfig.mockReturnValue(
      withGatewayConfig({ ide_oauth_client_id: 'mcp-gateway' })
    );

    renderModal();
    const serverConfig = getDisplayedConfig().mcpServers['test-server'];

    // Cursor (default IDE) advertises the pre-registered client for the login flow
    expect(serverConfig.auth).toEqual({ CLIENT_ID: 'mcp-gateway' });
    // No static gateway token is embedded
    expect(serverConfig.headers).toBeUndefined();
  });

  test('keeps the static gateway token when ide_oauth_client_id is not set', () => {
    mockUseRegistryConfig.mockReturnValue(withGatewayConfig());

    renderModal();
    const serverConfig = getDisplayedConfig().mcpServers['test-server'];

    expect(serverConfig.auth).toBeUndefined();
    expect(serverConfig.headers['X-Authorization']).toContain('Bearer');
  });

  test('does not enable OAuth login in registry-only mode', () => {
    mockUseRegistryConfig.mockReturnValue({
      config: {
        deployment_mode: 'registry-only',
        registry_mode: 'full',
        auth_provider: 'keycloak',
        ide_oauth_client_id: 'mcp-gateway',
        nginx_updates_enabled: false,
        features: { mcp_servers: true, agents: true, skills: true, federation: true, gateway_proxy: false },
      },
      loading: false,
      error: null,
    });

    renderModal({ proxy_pass_url: 'http://internal-host:8080/mcp' });
    const serverConfig = getDisplayedConfig().mcpServers['test-server'];

    // Registry-only servers are reached directly, so no gateway OAuth login
    expect(serverConfig.auth).toBeUndefined();
    expect(serverConfig.headers).toBeUndefined();
  });
});
