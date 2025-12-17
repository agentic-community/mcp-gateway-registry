import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { render } from '@/test/utils';
import { ToolPicker } from '../ToolPicker';
import { ServerWithTools } from '@/features/servers/hooks';

describe('ToolPicker', () => {
  const mockOnChange = vi.fn();

  const mockServersWithTools: ServerWithTools[] = [
    {
      server: {
        display_name: 'SQLite Server',
        path: 'sqlite',
        proxy_pass_url: 'http://localhost:3000',
        is_enabled: true,
      },
      tools: [
        { name: 'query', description: 'Execute SQL query' },
        { name: 'list_tables', description: 'List all tables' },
      ],
      isLoading: false,
      error: null,
    },
    {
      server: {
        display_name: 'GitHub Server',
        path: 'github',
        proxy_pass_url: 'http://localhost:3001',
        is_enabled: true,
      },
      tools: [
        { name: 'list_repos', description: 'List repositories' },
        { name: 'create_issue', description: 'Create an issue' },
      ],
      isLoading: false,
      error: null,
    },
  ];

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Basic Rendering', () => {
    it('renders with no tools selected', () => {
      render(
        <ToolPicker
          serversWithTools={mockServersWithTools}
          selectedTools={[]}
          onChange={mockOnChange}
        />
      );

      expect(screen.getByPlaceholderText('Search tools...')).toBeInTheDocument();
      expect(screen.getByText('SQLite Server')).toBeInTheDocument();
      expect(screen.getByText('GitHub Server')).toBeInTheDocument();
    });

    it('renders loading state', () => {
      render(
        <ToolPicker
          serversWithTools={[]}
          selectedTools={[]}
          onChange={mockOnChange}
          isLoading={true}
        />
      );

      expect(screen.getByText('Loading tools...')).toBeInTheDocument();
    });

    it('renders error state', () => {
      const error = new Error('Failed to load tools');
      render(
        <ToolPicker
          serversWithTools={[]}
          selectedTools={[]}
          onChange={mockOnChange}
          error={error}
        />
      );

      const errorElements = screen.getAllByText('Failed to load tools');
      expect(errorElements.length).toBeGreaterThan(0);
    });

    it('renders empty state when no servers', () => {
      render(
        <ToolPicker
          serversWithTools={[]}
          selectedTools={[]}
          onChange={mockOnChange}
        />
      );

      expect(screen.getByText('No servers available')).toBeInTheDocument();
    });
  });

  describe('Tool Selection', () => {
    it('displays selected tools count', () => {
      render(
        <ToolPicker
          serversWithTools={mockServersWithTools}
          selectedTools={['query', 'list_repos']}
          onChange={mockOnChange}
        />
      );

      expect(screen.getByText('2 tools selected')).toBeInTheDocument();
    });

    it('calls onChange when tool is selected', async () => {
      const { user } = render(
        <ToolPicker
          serversWithTools={mockServersWithTools}
          selectedTools={[]}
          onChange={mockOnChange}
        />
      );

      // Find all checkboxes
      const checkboxes = screen.getAllByRole('checkbox');
      // The first checkbox should be for 'query' tool (from SQLite Server)
      await user.click(checkboxes[0]);

      expect(mockOnChange).toHaveBeenCalledWith(['query']);
    });

    it('calls onChange when tool is deselected', async () => {
      const { user } = render(
        <ToolPicker
          serversWithTools={mockServersWithTools}
          selectedTools={['query', 'list_repos']}
          onChange={mockOnChange}
        />
      );

      // Find all checkboxes
      const checkboxes = screen.getAllByRole('checkbox');
      // Click the first checkbox (query) to deselect it
      await user.click(checkboxes[0]);

      expect(mockOnChange).toHaveBeenCalledWith(['list_repos']);
    });
  });

  describe('Search Functionality', () => {
    it('filters tools by name', async () => {
      const { user } = render(
        <ToolPicker
          serversWithTools={mockServersWithTools}
          selectedTools={[]}
          onChange={mockOnChange}
        />
      );

      const searchInput = screen.getByPlaceholderText('Search tools...');
      await user.type(searchInput, 'query');

      expect(screen.getByText('query')).toBeInTheDocument();
      expect(screen.queryByText('list_repos')).not.toBeInTheDocument();
    });

    it('shows no results when search matches nothing', async () => {
      const { user } = render(
        <ToolPicker
          serversWithTools={mockServersWithTools}
          selectedTools={[]}
          onChange={mockOnChange}
        />
      );

      const searchInput = screen.getByPlaceholderText('Search tools...');
      await user.type(searchInput, 'nonexistent');

      expect(screen.getByText('No tools found')).toBeInTheDocument();
    });
  });

  describe('Duplicate Tool Warning', () => {
    it('shows warning when duplicate tool names exist', () => {
      const serversWithDuplicates: ServerWithTools[] = [
        {
          server: {
            display_name: 'Server A',
            path: 'server-a',
            proxy_pass_url: 'http://localhost:3000',
            is_enabled: true,
          },
          tools: [{ name: 'duplicate_tool', description: 'Tool from server A' }],
          isLoading: false,
          error: null,
        },
        {
          server: {
            display_name: 'Server B',
            path: 'server-b',
            proxy_pass_url: 'http://localhost:3001',
            is_enabled: true,
          },
          tools: [{ name: 'duplicate_tool', description: 'Tool from server B' }],
          isLoading: false,
          error: null,
        },
      ];

      render(
        <ToolPicker
          serversWithTools={serversWithDuplicates}
          selectedTools={[]}
          onChange={mockOnChange}
          showDuplicateWarning={true}
        />
      );

      expect(screen.getByText('Duplicate tool names detected')).toBeInTheDocument();
      expect(screen.getAllByText('duplicate_tool').length).toBeGreaterThan(1);
    });

    it('can hide duplicate warning', () => {
      const serversWithDuplicates: ServerWithTools[] = [
        {
          server: {
            display_name: 'Server A',
            path: 'server-a',
            proxy_pass_url: 'http://localhost:3000',
            is_enabled: true,
          },
          tools: [{ name: 'duplicate_tool', description: 'Tool from server A' }],
          isLoading: false,
          error: null,
        },
        {
          server: {
            display_name: 'Server B',
            path: 'server-b',
            proxy_pass_url: 'http://localhost:3001',
            is_enabled: true,
          },
          tools: [{ name: 'duplicate_tool', description: 'Tool from server B' }],
          isLoading: false,
          error: null,
        },
      ];

      render(
        <ToolPicker
          serversWithTools={serversWithDuplicates}
          selectedTools={[]}
          onChange={mockOnChange}
          showDuplicateWarning={false}
        />
      );

      expect(screen.queryByText('Duplicate tool names detected')).not.toBeInTheDocument();
    });
  });

  describe('Server Select All', () => {
    it('selects all tools for a server', async () => {
      const { user } = render(
        <ToolPicker
          serversWithTools={mockServersWithTools}
          selectedTools={[]}
          onChange={mockOnChange}
        />
      );

      const selectAllButton = screen.getAllByText('Select all')[0];
      await user.click(selectAllButton);

      expect(mockOnChange).toHaveBeenCalledWith(['query', 'list_tables']);
    });

    it('deselects all tools for a server', async () => {
      const { user } = render(
        <ToolPicker
          serversWithTools={mockServersWithTools}
          selectedTools={['query', 'list_tables']}
          onChange={mockOnChange}
        />
      );

      const deselectAllButton = screen.getAllByText('Deselect all')[0];
      await user.click(deselectAllButton);

      expect(mockOnChange).toHaveBeenCalledWith([]);
    });
  });
});
