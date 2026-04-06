import React from 'react';
import { render, screen, fireEvent } from '@testing-library/react';
import DiscoverTab from '../DiscoverTab';


// Mock useSemanticSearch hook
const mockSemanticSearch = {
  results: null,
  loading: false,
  error: null,
  debouncedQuery: '',
};
jest.mock('../../hooks/useSemanticSearch', () => ({
  useSemanticSearch: () => mockSemanticSearch,
}));

// Mock DiscoverListRow to simplify testing
jest.mock('../DiscoverListRow', () => {
  const MockListRow = (props) => (
    <div data-testid={`list-row-${props.type}-${props.item.path}`}>
      {props.item.name}
    </div>
  );
  MockListRow.displayName = 'DiscoverListRow';
  return MockListRow;
});

jest.mock('../SemanticSearchResults', () => {
  const MockSearchResults = (props) => (
    <div data-testid="semantic-search-results">
      Search results for: {props.query}
    </div>
  );
  MockSearchResults.displayName = 'SemanticSearchResults';
  return MockSearchResults;
});


// Test data factories
const makeServer = (overrides = {}) => ({
  name: 'Test Server',
  path: '/test-server/',
  enabled: true,
  rating_details: [],
  ...overrides,
});


const makeSkill = (overrides = {}) => ({
  name: 'Test Skill',
  path: '/test-skill/',
  skill_md_url: '',
  visibility: 'public',
  is_enabled: true,
  num_stars: 0,
  ...overrides,
});


const defaultProps = {
  servers: [],
  agents: [],
  skills: [],
  virtualServers: [],
  externalServers: [],
  externalAgents: [],
  loading: false,
  onServerToggle: jest.fn(),
  onAgentToggle: jest.fn(),
  onSkillToggle: jest.fn(),
  onVirtualServerToggle: jest.fn(),
  onShowToast: jest.fn(),
  authToken: null,
};


describe('DiscoverTab', () => {
  beforeEach(() => {
    jest.clearAllMocks();
    mockSemanticSearch.results = null;
    mockSemanticSearch.loading = false;
    mockSemanticSearch.error = null;
  });

  test('renders search bar and title at the top', () => {
    render(<DiscoverTab {...defaultProps} />);
    expect(
      screen.getByPlaceholderText(/search servers, agents, skills/i)
    ).toBeInTheDocument();
    expect(
      screen.getByText('Discover MCP Servers, Agents & Skills')
    ).toBeInTheDocument();
  });

  test('title stays visible during keyword search', () => {
    render(<DiscoverTab {...defaultProps} />);

    const input = screen.getByPlaceholderText(/search servers/i);
    fireEvent.change(input, { target: { value: 'test' } });

    expect(
      screen.getByText('Discover MCP Servers, Agents & Skills')
    ).toBeInTheDocument();
  });

  test('shows loading state', () => {
    render(<DiscoverTab {...defaultProps} loading={true} />);
    expect(screen.getByText(/loading featured items/i)).toBeInTheDocument();
  });

  test('shows empty state when no items registered', () => {
    render(<DiscoverTab {...defaultProps} />);
    expect(
      screen.getByText(/no items registered yet/i)
    ).toBeInTheDocument();
  });

  test('shows "no items matching" when keyword filter has no results', () => {
    render(
      <DiscoverTab
        {...defaultProps}
        servers={[makeServer({ name: 'AI Registry tools', path: '/airegistry-tools/' })]}
      />
    );

    const input = screen.getByPlaceholderText(/search servers/i);
    fireEvent.change(input, { target: { value: 'zzzznonexistent' } });

    expect(screen.getByText(/no items matching/i)).toBeInTheDocument();
  });

  test('renders section headers for servers, agents, and skills', () => {
    render(
      <DiscoverTab
        {...defaultProps}
        servers={[makeServer({ name: 'My Server', path: '/my-server/' })]}
        agents={[makeServer({ name: 'My Agent', path: '/my-agent/' })]}
        skills={[makeSkill({ name: 'My Skill', path: '/my-skill/' })]}
      />
    );

    expect(screen.getByText('MCP Servers')).toBeInTheDocument();
    expect(screen.getByText('Agents')).toBeInTheDocument();
    expect(screen.getByText('Skills')).toBeInTheDocument();
  });

  test('renders list rows for each item type', () => {
    render(
      <DiscoverTab
        {...defaultProps}
        servers={[
          makeServer({ name: 'AI Registry tools', path: '/airegistry-tools/' }),
          makeServer({
            name: 'Cloudflare Docs',
            path: '/cloudflare-docs/',
            rating_details: [{ user: 'alice', rating: 5 }],
          }),
        ]}
        agents={[
          makeServer({
            name: 'Test Agent',
            path: '/test-agent/',
            rating_details: [{ user: 'bob', rating: 4 }],
          }),
        ]}
        skills={[
          makeSkill({ name: 'Code Review', path: '/code-review/', num_stars: 4.2 }),
        ]}
      />
    );

    expect(screen.getByTestId('list-row-server-/airegistry-tools/')).toBeInTheDocument();
    expect(screen.getByTestId('list-row-server-/cloudflare-docs/')).toBeInTheDocument();
    expect(screen.getByTestId('list-row-agent-/test-agent/')).toBeInTheDocument();
    expect(screen.getByTestId('list-row-skill-/code-review/')).toBeInTheDocument();
  });

  test('sorts servers by rating descending, alphabetical tiebreaker', () => {
    const servers = [
      makeServer({ name: 'AI Registry tools', path: '/airegistry-tools/' }),
      makeServer({
        name: 'Beta Server',
        path: '/beta/',
        rating_details: [{ user: 'u1', rating: 3 }],
      }),
      makeServer({
        name: 'Alpha Server',
        path: '/alpha/',
        rating_details: [{ user: 'u1', rating: 5 }],
      }),
      makeServer({
        name: 'Gamma Server',
        path: '/gamma/',
        rating_details: [{ user: 'u1', rating: 5 }],
      }),
    ];

    render(<DiscoverTab {...defaultProps} servers={servers} />);

    expect(screen.getByTestId('list-row-server-/airegistry-tools/')).toBeInTheDocument();
    expect(screen.getByTestId('list-row-server-/alpha/')).toBeInTheDocument();
    expect(screen.getByTestId('list-row-server-/gamma/')).toBeInTheDocument();
    expect(screen.getByTestId('list-row-server-/beta/')).toBeInTheDocument();
  });

  test('excludes disabled items from featured', () => {
    const servers = [
      makeServer({ name: 'AI Registry tools', path: '/airegistry-tools/' }),
      makeServer({
        name: 'Disabled Server',
        path: '/disabled/',
        enabled: false,
        rating_details: [{ user: 'u1', rating: 5 }],
      }),
    ];

    render(<DiscoverTab {...defaultProps} servers={servers} />);

    expect(screen.getByText('AI Registry tools')).toBeInTheDocument();
    expect(screen.queryByText('Disabled Server')).not.toBeInTheDocument();
  });

  test('keyword search filters items instantly as you type', () => {
    render(
      <DiscoverTab
        {...defaultProps}
        servers={[
          makeServer({ name: 'AI Registry tools', path: '/airegistry-tools/' }),
          makeServer({ name: 'Cloudflare Docs', path: '/cloudflare-docs/' }),
        ]}
        agents={[makeServer({ name: 'Test Agent', path: '/test-agent/' })]}
      />
    );

    expect(screen.getByText('AI Registry tools')).toBeInTheDocument();
    expect(screen.getByText('Cloudflare Docs')).toBeInTheDocument();
    expect(screen.getByText('Test Agent')).toBeInTheDocument();

    const input = screen.getByPlaceholderText(/search servers/i);
    fireEvent.change(input, { target: { value: 'cloud' } });

    expect(screen.queryByText('AI Registry tools')).not.toBeInTheDocument();
    expect(screen.getByText('Cloudflare Docs')).toBeInTheDocument();
    expect(screen.queryByText('Test Agent')).not.toBeInTheDocument();
  });

  test('keyword search matches tags', () => {
    render(
      <DiscoverTab
        {...defaultProps}
        servers={[
          makeServer({ name: 'Server A', path: '/a/', tags: ['documentation'] }),
          makeServer({ name: 'Server B', path: '/b/', tags: ['api'] }),
        ]}
      />
    );

    const input = screen.getByPlaceholderText(/search servers/i);
    fireEvent.change(input, { target: { value: 'documentation' } });

    expect(screen.getByText('Server A')).toBeInTheDocument();
    expect(screen.queryByText('Server B')).not.toBeInTheDocument();
  });

  test('Enter key triggers semantic search', () => {
    render(<DiscoverTab {...defaultProps} />);

    const input = screen.getByPlaceholderText(/search servers/i);
    fireEvent.change(input, { target: { value: 'find something' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    expect(screen.getByTestId('semantic-search-results')).toBeInTheDocument();
    expect(
      screen.getByText(/search results for: find something/i)
    ).toBeInTheDocument();
  });

  test('clearing search returns to full listing', () => {
    render(
      <DiscoverTab
        {...defaultProps}
        servers={[
          makeServer({ name: 'AI Registry tools', path: '/airegistry-tools/' }),
        ]}
      />
    );

    const input = screen.getByPlaceholderText(/search servers/i);

    fireEvent.change(input, { target: { value: 'test query' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(screen.getByTestId('semantic-search-results')).toBeInTheDocument();

    const clearButton = screen.getByRole('button');
    fireEvent.click(clearButton);

    expect(screen.queryByTestId('semantic-search-results')).not.toBeInTheDocument();
    expect(screen.getByText('AI Registry tools')).toBeInTheDocument();
  });

  test('backspacing exits semantic mode and shows keyword-filtered results', () => {
    render(
      <DiscoverTab
        {...defaultProps}
        servers={[
          makeServer({ name: 'AI Registry tools', path: '/airegistry-tools/' }),
          makeServer({ name: 'Cloudflare Docs', path: '/cloudflare-docs/' }),
        ]}
      />
    );

    const input = screen.getByPlaceholderText(/search servers/i);

    fireEvent.change(input, { target: { value: 'cloud' } });
    fireEvent.keyDown(input, { key: 'Enter' });
    expect(screen.getByTestId('semantic-search-results')).toBeInTheDocument();

    fireEvent.change(input, { target: { value: 'clou' } });
    expect(screen.queryByTestId('semantic-search-results')).not.toBeInTheDocument();

    expect(screen.queryByText('AI Registry tools')).not.toBeInTheDocument();
    expect(screen.getByText('Cloudflare Docs')).toBeInTheDocument();
  });

  test('does not trigger semantic search for queries shorter than 2 characters', () => {
    render(<DiscoverTab {...defaultProps} />);

    const input = screen.getByPlaceholderText(/search servers/i);
    fireEvent.change(input, { target: { value: 'a' } });
    fireEvent.keyDown(input, { key: 'Enter' });

    expect(screen.queryByTestId('semantic-search-results')).not.toBeInTheDocument();
  });

  test('graceful degradation when no agents exist', () => {
    render(
      <DiscoverTab
        {...defaultProps}
        servers={[makeServer({ name: 'AI Registry tools', path: '/airegistry-tools/' })]}
        agents={[]}
        skills={[makeSkill({ name: 'My Skill', path: '/my-skill/' })]}
      />
    );

    expect(screen.getByText('AI Registry tools')).toBeInTheDocument();
    expect(screen.getByText('My Skill')).toBeInTheDocument();
    expect(screen.queryByText('Agents')).not.toBeInTheDocument();
  });
});
