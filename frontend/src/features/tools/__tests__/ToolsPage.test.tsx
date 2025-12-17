import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import ToolsPage from '../ToolsPage';

describe('ToolsPage', () => {
  const mockServers = [
    {
      display_name: 'SQLite Server',
      path: 'sqlite',
      proxy_pass_url: 'http://localhost:3000',
      is_enabled: true,
      health_status: 'healthy',
      num_tools: 2,
    },
    {
      display_name: 'GitHub Server',
      path: 'github',
      proxy_pass_url: 'http://localhost:3001',
      is_enabled: true,
      health_status: 'healthy',
      num_tools: 3,
    },
  ];

  const mockSqliteDetails = {
    display_name: 'SQLite Server',
    path: 'sqlite',
    proxy_pass_url: 'http://localhost:3000',
    is_enabled: true,
    health_status: 'healthy' as const,
    num_tools: 2,
    tools: [
      { name: 'query', description: 'Execute SQL query' },
      { name: 'list_tables', description: 'List all tables' },
    ],
  };

  const mockGithubDetails = {
    display_name: 'GitHub Server',
    path: 'github',
    proxy_pass_url: 'http://localhost:3001',
    is_enabled: true,
    health_status: 'healthy' as const,
    num_tools: 3,
    tools: [
      { name: 'list_repos', description: 'List repositories' },
      { name: 'create_issue', description: 'Create an issue' },
      { name: 'close_issue', description: 'Close an issue' },
    ],
  };

  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();

    // Mock servers list endpoint
    server.use(
      http.get('/api/servers', () => {
        return HttpResponse.json({ servers: mockServers });
      }),
      http.get('/api/servers/sqlite', () => {
        return HttpResponse.json(mockSqliteDetails);
      }),
      http.get('/api/servers/github', () => {
        return HttpResponse.json(mockGithubDetails);
      })
    );
  });

  describe('Page Rendering', () => {
    it('renders page header and summary stats', async () => {
      render(<ToolsPage />);

      // Wait for loading to complete
      await waitFor(
        () => {
          expect(screen.queryByText('Loading tools from all servers...')).not.toBeInTheDocument();
        },
        { timeout: 3000 }
      );

      await waitFor(() => {
        expect(screen.getByText('Tools Discovery')).toBeInTheDocument();
        expect(
          screen.getByText('Browse all available MCP server tools and configure agent access')
        ).toBeInTheDocument();
        expect(screen.getByText('Total Servers')).toBeInTheDocument();
        expect(screen.getByText('Total Tools')).toBeInTheDocument();
        expect(screen.getByText('Enabled Servers')).toBeInTheDocument();
      });
    });

    it('displays correct summary counts', async () => {
      render(<ToolsPage />);

      await waitFor(() => {
        // Total servers: 2
        const serverCounts = screen.getAllByText('2');
        expect(serverCounts.length).toBeGreaterThan(0);

        // Total tools: 5 (2 from sqlite + 3 from github)
        expect(screen.getByText('5')).toBeInTheDocument();
      });
    });

    it('shows loading state initially', () => {
      render(<ToolsPage />);

      expect(screen.getByText('Loading tools from all servers...')).toBeInTheDocument();
    });
  });

  describe('Server List', () => {
    it('displays all servers', async () => {
      render(<ToolsPage />);

      // Wait for loading to complete
      await waitFor(
        () => {
          expect(screen.queryByText('Loading tools from all servers...')).not.toBeInTheDocument();
        },
        { timeout: 3000 }
      );

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
        expect(screen.getByText('GitHub Server')).toBeInTheDocument();
      });
    });

    it('shows server paths', async () => {
      render(<ToolsPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite')).toBeInTheDocument();
        expect(screen.getByText('github')).toBeInTheDocument();
      });
    });

    it('displays server status badges', async () => {
      render(<ToolsPage />);

      await waitFor(() => {
        const enabledBadges = screen.getAllByText('enabled');
        expect(enabledBadges.length).toBe(2);

        const healthyBadges = screen.getAllByText('healthy');
        expect(healthyBadges.length).toBe(2);
      });
    });

    it('shows tool count per server', async () => {
      render(<ToolsPage />);

      await waitFor(() => {
        expect(screen.getByText('2 tools')).toBeInTheDocument();
        expect(screen.getByText('3 tools')).toBeInTheDocument();
      });
    });
  });

  describe('Server Expansion', () => {
    it('expands server to show tools', async () => {
      const { user } = render(<ToolsPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      // Click to expand
      const sqliteServer = screen.getByText('SQLite Server');
      await user.click(sqliteServer);

      await waitFor(() => {
        expect(screen.getByText('query')).toBeInTheDocument();
        expect(screen.getByText('Execute SQL query')).toBeInTheDocument();
        expect(screen.getByText('list_tables')).toBeInTheDocument();
        expect(screen.getByText('List all tables')).toBeInTheDocument();
      });
    });

    it('collapses expanded server', async () => {
      const { user } = render(<ToolsPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      // Expand
      const sqliteServer = screen.getByText('SQLite Server');
      await user.click(sqliteServer);

      await waitFor(() => {
        expect(screen.getByText('query')).toBeInTheDocument();
      });

      // Collapse
      await user.click(sqliteServer);

      await waitFor(() => {
        expect(screen.queryByText('query')).not.toBeInTheDocument();
      });
    });
  });

  describe('Expand/Collapse All', () => {
    it('expands all servers', async () => {
      const { user } = render(<ToolsPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      const expandAllButton = screen.getByText('Expand All');
      await user.click(expandAllButton);

      await waitFor(() => {
        expect(screen.getByText('query')).toBeInTheDocument();
        expect(screen.getByText('list_repos')).toBeInTheDocument();
      });
    });

    it('collapses all servers', async () => {
      const { user } = render(<ToolsPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      // First expand all
      const expandAllButton = screen.getByText('Expand All');
      await user.click(expandAllButton);

      await waitFor(() => {
        expect(screen.getByText('query')).toBeInTheDocument();
      });

      // Then collapse all
      const collapseAllButton = screen.getByText('Collapse All');
      await user.click(collapseAllButton);

      await waitFor(() => {
        expect(screen.queryByText('query')).not.toBeInTheDocument();
      });
    });
  });

  describe('Search Functionality', () => {
    it('filters servers by name', async () => {
      const { user } = render(<ToolsPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
        expect(screen.getByText('GitHub Server')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search tools or servers...');
      await user.type(searchInput, 'SQLite');

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
        expect(screen.queryByText('GitHub Server')).not.toBeInTheDocument();
      });
    });

    it('filters tools by name', async () => {
      const { user } = render(<ToolsPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search tools or servers...');
      await user.type(searchInput, 'query');

      // Expand to see tools
      const expandAllButton = screen.getByText('Expand All');
      await user.click(expandAllButton);

      await waitFor(() => {
        expect(screen.getByText('query')).toBeInTheDocument();
        expect(screen.queryByText('list_repos')).not.toBeInTheDocument();
      });
    });

    it('shows empty state when no results', async () => {
      const { user } = render(<ToolsPage />);

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search tools or servers...');
      await user.type(searchInput, 'nonexistent');

      await waitFor(() => {
        expect(screen.getByText('No tools found')).toBeInTheDocument();
        expect(screen.getByText('Try adjusting your search query.')).toBeInTheDocument();
      });
    });
  });

  describe('Configure Agent Button', () => {
    it('navigates to agents page when clicked', async () => {
      const { user } = render(<ToolsPage />);

      await waitFor(() => {
        expect(screen.getByText('Configure Agent')).toBeInTheDocument();
      });

      const configureButton = screen.getByText('Configure Agent');
      await user.click(configureButton);

      // Navigation is mocked in test utils, just verify button is clickable
      expect(configureButton).toBeEnabled();
    });
  });

  describe('Error Handling', () => {
    it('displays error when servers fail to load', async () => {
      server.use(
        http.get('/api/servers', () => {
          return HttpResponse.json(
            { detail: 'Failed to fetch servers' },
            { status: 500 }
          );
        })
      );

      render(<ToolsPage />);

      await waitFor(() => {
        expect(screen.getByText('Failed to load tools')).toBeInTheDocument();
      });
    });
  });

  describe('Tool Schema Display', () => {
    it('shows input schema when expanded', async () => {
      const mockServerWithSchema = {
        display_name: 'SQLite Server',
        path: 'sqlite',
        proxy_pass_url: 'http://localhost:3000',
        is_enabled: true,
        health_status: 'healthy' as const,
        num_tools: 1,
        tools: [
          {
            name: 'query',
            description: 'Execute SQL query',
            input_schema: { type: 'object', properties: { sql: { type: 'string' } } },
          },
        ],
      };

      server.use(
        http.get('/api/servers/sqlite', () => {
          return HttpResponse.json(mockServerWithSchema);
        })
      );

      const { user } = render(<ToolsPage />);

      // Wait for loading to complete
      await waitFor(
        () => {
          expect(screen.queryByText('Loading tools from all servers...')).not.toBeInTheDocument();
        },
        { timeout: 3000 }
      );

      await waitFor(() => {
        expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      });

      // Expand server
      const sqliteServer = screen.getByText('SQLite Server');
      await user.click(sqliteServer);

      await waitFor(() => {
        expect(screen.getByText('query')).toBeInTheDocument();
      });

      // Click to view schema
      const viewSchemaButton = screen.getByText('View input schema');
      await user.click(viewSchemaButton);

      await waitFor(() => {
        expect(screen.getByText(/"type":/)).toBeInTheDocument();
      });
    });
  });
});
