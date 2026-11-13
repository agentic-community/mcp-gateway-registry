import React from 'react';
import axios from 'axios';
import { render, screen, fireEvent, within } from '@testing-library/react';
import { MemoryRouter } from 'react-router-dom';
import Dashboard from '../Dashboard';

/**
 * Render-level safety net for the Dashboard. Pins down the per-entity collection
 * behavior (cards render per tab, empty states show, tab switching works) BEFORE
 * the Section-component decomposition, so structural changes that alter what the
 * user sees fail loudly.
 *
 * Every data hook and the heavy card components are mocked to lightweight stubs;
 * the test asserts on the Dashboard's own composition, not card internals.
 */

// --- Mutable hook state the mocks read from (reset per test) ---
const stats = {
  servers: [] as any[],
  agents: [] as any[],
  customRecordsByType: [] as any[],
  loading: false,
  error: null as string | null,
  refreshData: jest.fn(),
  setServers: jest.fn(),
  setAgents: jest.fn(),
};
const skillsState = {
  skills: [] as any[],
  setSkills: jest.fn(),
  loading: false,
  error: null as string | null,
  refreshData: jest.fn(),
};
const virtualState = {
  virtualServers: [] as any[],
  loading: false,
  error: null as string | null,
  toggleVirtualServer: jest.fn(),
  deleteVirtualServer: jest.fn(),
  updateVirtualServer: jest.fn(),
  refreshData: jest.fn(),
};
let authUser: any = { username: 'admin', is_admin: true, can_modify_servers: true };
let registryFeatures: Record<string, boolean> = {};

jest.mock('../../hooks/useServerStats', () => ({
  useServerStats: () => stats,
}));
jest.mock('../../hooks/useSkills', () => ({
  useSkills: () => skillsState,
}));
jest.mock('../../hooks/useVirtualServers', () => ({
  useVirtualServers: () => virtualState,
  useVirtualServer: () => ({ virtualServer: null, loading: false }),
}));
jest.mock('../../contexts/AuthContext', () => ({
  useAuth: () => ({ user: authUser }),
}));
jest.mock('../../hooks/useRegistryConfig', () => ({
  useRegistryConfig: () => ({ config: { features: registryFeatures } }),
}));
jest.mock('../../hooks/useSemanticSearch', () => ({
  useSemanticSearch: () => ({ results: null, loading: false, error: null, debouncedQuery: '' }),
}));
jest.mock('../../hooks/useDuplicateCheck', () => ({
  useDuplicateCheck: () => ({
    runCheck: jest.fn(),
    duplicates: [],
    loading: false,
    reset: jest.fn(),
  }),
}));
jest.mock('../../hooks/useCustomEntities', () => ({
  uuidFromPath: (p: string) => p,
}));

// Card stubs surface the entity name + a testid so we can assert per-tab rendering.
jest.mock('../../components/ServerCard', () => {
  const M = (props: any) => <div data-testid="server-card">{props.server.name}</div>;
  M.displayName = 'ServerCard';
  return M;
});
jest.mock('../../components/AgentCard', () => {
  const M = (props: any) => <div data-testid="agent-card">{props.agent.name}</div>;
  M.displayName = 'AgentCard';
  return M;
});
jest.mock('../../components/SkillCard', () => {
  const M = (props: any) => <div data-testid="skill-card">{props.skill.name}</div>;
  M.displayName = 'SkillCard';
  return M;
});
jest.mock('../../components/VirtualServerCard', () => {
  const M = (props: any) => (
    <div data-testid="virtual-card">{props.virtualServer.server_name}</div>
  );
  M.displayName = 'VirtualServerCard';
  return M;
});
jest.mock('../../components/OpenApiServerList', () => {
  const M = () => <div data-testid="openapi-server-list" />;
  M.displayName = 'OpenApiServerList';
  return M;
});
// Other heavy children that aren't under test.
jest.mock('../../components/DiscoverTab', () => {
  const M = () => <div data-testid="discover-tab" />;
  M.displayName = 'DiscoverTab';
  return M;
});
jest.mock('../../components/SemanticSearchResults', () => {
  const M = () => <div data-testid="semantic-results" />;
  M.displayName = 'SemanticSearchResults';
  return M;
});
jest.mock('../../components/DuplicateCheckModal', () => {
  const M = () => null;
  M.displayName = 'DuplicateCheckModal';
  return M;
});

const makeServer = (name: string, overrides = {}) => ({
  name,
  path: `/${name}/`,
  enabled: true,
  status: 'healthy',
  tags: [],
  rating_details: [],
  ...overrides,
});
const makeAgent = (name: string, overrides = {}) => ({
  name,
  path: `/${name}/`,
  enabled: true,
  status: 'healthy',
  tags: [],
  rating_details: [],
  ...overrides,
});
const makeSkill = (name: string, overrides = {}) => ({
  name,
  path: `/skills/${name}`,
  visibility: 'public',
  is_enabled: true,
  tags: [],
  ...overrides,
});
const makeVirtual = (name: string, overrides = {}) => ({
  server_name: name,
  path: `/${name}`,
  is_enabled: true,
  tags: [],
  tool_count: 0,
  rating_details: [],
  ...overrides,
});

function renderDashboard(filter = 'all') {
  return render(
    <MemoryRouter>
      <Dashboard activeFilter={filter} setActiveFilter={jest.fn()} selectedTags={[]} />
    </MemoryRouter>,
  );
}

beforeEach(() => {
  stats.servers = [];
  stats.agents = [];
  stats.customRecordsByType = [];
  stats.loading = false;
  stats.error = null;
  skillsState.skills = [];
  skillsState.loading = false;
  skillsState.error = null;
  virtualState.virtualServers = [];
  virtualState.loading = false;
  authUser = { username: 'admin', is_admin: true, can_modify_servers: true };
  registryFeatures = {};
  // No built-in federation source is configured unless a test says so. Also
  // keeps the Dashboard's background GETs from reaching the network and
  // logging after a test has finished.
  mockFederationConfig({});
});

afterEach(() => {
  jest.restoreAllMocks();
});

/**
 * Answer GET /api/federation/config with `config`. The Dashboard shows a
 * built-in source tab (Anthropic, ASOR, AWS) only when that source is
 * configured there. Every other GET keeps failing, as it does unmocked.
 */
function mockFederationConfig(config: Record<string, unknown>) {
  jest.spyOn(axios, 'get').mockImplementation((url: string) =>
    url === '/api/federation/config'
      ? Promise.resolve({ data: config })
      : Promise.reject(new Error(`unmocked GET ${url}`)),
  );
}

describe('Dashboard entity collections', () => {
  it('renders server cards on the MCP Servers tab', () => {
    stats.servers = [makeServer('Alpha'), makeServer('Beta')];
    renderDashboard();
    fireEvent.click(screen.getByRole('button', { name: 'MCP Servers' }));
    const cards = screen.getAllByTestId('server-card');
    expect(cards).toHaveLength(2);
    expect(screen.getByText('Alpha')).toBeInTheDocument();
  });

  it('shows the server empty state with a register CTA when there are none', () => {
    renderDashboard();
    fireEvent.click(screen.getByRole('button', { name: 'MCP Servers' }));
    expect(screen.getByText('No servers found')).toBeInTheDocument();
    expect(
      screen.getByRole('button', { name: /Register Server/i }),
    ).toBeInTheDocument();
  });

  it('renders agent cards on the Agents tab', () => {
    stats.agents = [makeAgent('AgentOne')];
    renderDashboard();
    fireEvent.click(screen.getByRole('button', { name: 'Agents' }));
    expect(screen.getByTestId('agent-card')).toHaveTextContent('AgentOne');
  });

  it('renders skill cards on the Skills tab', () => {
    skillsState.skills = [makeSkill('doc-writer')];
    renderDashboard();
    fireEvent.click(screen.getByRole('button', { name: 'Agent Skills' }));
    expect(screen.getByTestId('skill-card')).toHaveTextContent('doc-writer');
  });

  it('renders virtual server cards on the Virtual MCP tab', () => {
    virtualState.virtualServers = [makeVirtual('VS One')];
    renderDashboard();
    fireEvent.click(screen.getByRole('button', { name: 'Virtual MCP Servers' }));
    expect(screen.getByTestId('virtual-card')).toHaveTextContent('VS One');
  });

  it('groups servers into collapsible registries when a federated server exists', () => {
    stats.servers = [
      makeServer('LocalOne'),
      makeServer('PeerOne', {
        sync_metadata: { is_federated: true, source_peer_id: 'peer-registry-lob1' },
      }),
    ];
    renderDashboard();
    fireEvent.click(screen.getByRole('button', { name: 'MCP Servers' }));

    // Both registry group headers render (local + the federated peer).
    expect(screen.getByText('Local Registry')).toBeInTheDocument();
    expect(screen.getByText(/LOB1 \(Federated\)/)).toBeInTheDocument();
    // Both cards are visible while the groups are expanded by default.
    expect(screen.getByText('LocalOne')).toBeInTheDocument();
    expect(screen.getByText('PeerOne')).toBeInTheDocument();
  });

  it('collapses a server registry group when its header is clicked', () => {
    stats.servers = [
      makeServer('LocalOne'),
      makeServer('PeerOne', {
        sync_metadata: { is_federated: true, source_peer_id: 'peer-registry-lob1' },
      }),
    ];
    renderDashboard();
    fireEvent.click(screen.getByRole('button', { name: 'MCP Servers' }));
    expect(screen.getByText('LocalOne')).toBeInTheDocument();

    // Collapsing the local group hides its card but keeps the header.
    fireEvent.click(screen.getByText('Local Registry'));
    expect(screen.queryByText('LocalOne')).not.toBeInTheDocument();
    expect(screen.getByText('Local Registry')).toBeInTheDocument();
    // The other group stays expanded.
    expect(screen.getByText('PeerOne')).toBeInTheDocument();
  });

  it('groups agents into collapsible registries when a federated agent exists', () => {
    stats.agents = [
      makeAgent('LocalAgent'),
      makeAgent('PeerAgent', {
        sync_metadata: { is_federated: true, source_peer_id: 'peer-registry-lob2' },
      }),
    ];
    renderDashboard();
    fireEvent.click(screen.getByRole('button', { name: 'Agents' }));
    expect(screen.getByText('Local Registry')).toBeInTheDocument();
    expect(screen.getByText(/LOB2 \(Federated\)/)).toBeInTheDocument();
    expect(screen.getByText('LocalAgent')).toBeInTheDocument();
    expect(screen.getByText('PeerAgent')).toBeInTheDocument();
  });

  it('renders the External Registries tab with federated cards grouped by type', async () => {
    mockFederationConfig({ anthropic: { enabled: true } });
    // Anthropic federation marks imported servers with this tag (no peer sync_metadata).
    stats.servers = [makeServer('ExtServer', { tags: ['anthropic-registry'] })];
    renderDashboard();
    fireEvent.click(screen.getByRole('button', { name: 'External Registries' }));
    // The federated server shows under the Servers subsection of the Anthropic tab.
    expect(await screen.findByRole('heading', { name: 'Servers' })).toBeInTheDocument();
    expect(screen.getByRole('button', { name: 'Anthropic' })).toBeInTheDocument();
    expect(screen.getByText('ExtServer')).toBeInTheDocument();
  });

  it('shows the external empty state when no external registries are configured', () => {
    renderDashboard();
    fireEvent.click(screen.getByRole('button', { name: 'External Registries' }));
    expect(
      screen.getByText('No External Registries Available'),
    ).toBeInTheDocument();
  });

  it('switches collections when tabs change', () => {
    stats.servers = [makeServer('Alpha')];
    skillsState.skills = [makeSkill('doc-writer')];
    renderDashboard();

    fireEvent.click(screen.getByRole('button', { name: 'MCP Servers' }));
    expect(screen.getByTestId('server-card')).toBeInTheDocument();
    expect(screen.queryByTestId('skill-card')).not.toBeInTheDocument();

    fireEvent.click(screen.getByRole('button', { name: 'Agent Skills' }));
    expect(screen.getByTestId('skill-card')).toBeInTheDocument();
    expect(screen.queryByTestId('server-card')).not.toBeInTheDocument();
  });
});


describe('Dashboard OpenAPI MCP servers view (issue #2022)', () => {
  it('hides the OpenAPI tab when the feature flag is off', () => {
    registryFeatures = { openapi_servers: false };
    renderDashboard();
    expect(screen.queryByRole('button', { name: 'OpenAPI MCP Servers' })).not.toBeInTheDocument();
    expect(screen.queryByTestId('openapi-server-list')).not.toBeInTheDocument();
  });

  it('hides the OpenAPI tab when the backend omits the flag', () => {
    renderDashboard();
    expect(screen.queryByRole('button', { name: 'OpenAPI MCP Servers' })).not.toBeInTheDocument();
  });

  it('hides the OpenAPI tab from a non-admin even when the flag is on', () => {
    registryFeatures = { openapi_servers: true };
    authUser = { username: 'dev', is_admin: false, can_modify_servers: true };
    renderDashboard();
    expect(screen.queryByRole('button', { name: 'OpenAPI MCP Servers' })).not.toBeInTheDocument();
  });

  it('renders the OpenAPI list for an admin when the flag is on', () => {
    registryFeatures = { openapi_servers: true };
    renderDashboard();
    fireEvent.click(screen.getByRole('button', { name: 'OpenAPI MCP Servers' }));
    expect(screen.getByTestId('openapi-server-list')).toBeInTheDocument();
  });
});
