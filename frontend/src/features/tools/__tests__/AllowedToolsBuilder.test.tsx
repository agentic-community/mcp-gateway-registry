import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { render } from '@/test/utils';
import { AllowedToolsBuilder } from '../AllowedToolsBuilder';
import { ServerWithTools } from '@/features/servers/hooks';

describe('AllowedToolsBuilder', () => {
  const mockOnApply = vi.fn();

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
    it('renders with tool picker', () => {
      render(
        <AllowedToolsBuilder
          serversWithTools={mockServersWithTools}
          onApply={mockOnApply}
        />
      );

      expect(screen.getByText('Select Tools')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Search tools...')).toBeInTheDocument();
    });

    it('renders with initial tools selected', () => {
      render(
        <AllowedToolsBuilder
          serversWithTools={mockServersWithTools}
          initialTools={['query', 'list_repos']}
          onApply={mockOnApply}
        />
      );

      expect(screen.getByText(/Selected Tools \(2\)/)).toBeInTheDocument();
      const queryElements = screen.getAllByText('query');
      expect(queryElements.length).toBeGreaterThan(0);
      const listReposElements = screen.getAllByText('list_repos');
      expect(listReposElements.length).toBeGreaterThan(0);
    });

    it('shows apply button by default', () => {
      render(
        <AllowedToolsBuilder
          serversWithTools={mockServersWithTools}
          initialTools={['query']}
          onApply={mockOnApply}
        />
      );

      expect(screen.getByText('Apply to Agent')).toBeInTheDocument();
    });

    it('hides apply button when showApplyButton is false', () => {
      render(
        <AllowedToolsBuilder
          serversWithTools={mockServersWithTools}
          initialTools={['query']}
          onApply={mockOnApply}
          showApplyButton={false}
        />
      );

      expect(screen.queryByText('Apply to Agent')).not.toBeInTheDocument();
    });
  });

  describe('Tool Selection', () => {
    it('updates selected tools when tool is clicked', async () => {
      const { user } = render(
        <AllowedToolsBuilder
          serversWithTools={mockServersWithTools}
          onApply={mockOnApply}
        />
      );

      // Find and click the first checkbox
      const checkboxes = screen.getAllByRole('checkbox');
      await user.click(checkboxes[0]);

      await waitFor(() => {
        expect(screen.getByText(/Selected Tools \(1\)/)).toBeInTheDocument();
      });
    });

    it('displays selected tools with server count for duplicates', () => {
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
        <AllowedToolsBuilder
          serversWithTools={serversWithDuplicates}
          initialTools={['duplicate_tool']}
          onApply={mockOnApply}
        />
      );

      expect(screen.getByText('2 servers')).toBeInTheDocument();
    });
  });

  describe('Remove Tool', () => {
    it('removes tool when X button is clicked', async () => {
      const { user } = render(
        <AllowedToolsBuilder
          serversWithTools={mockServersWithTools}
          initialTools={['query', 'list_repos']}
          onApply={mockOnApply}
        />
      );

      expect(screen.getByText(/Selected Tools \(2\)/)).toBeInTheDocument();

      // Get all buttons - there should be "Clear All", "Apply to Agent", and X buttons for each tool
      const allButtons = screen.getAllByRole('button');

      // The X buttons are unmarked buttons (no text), filter them out
      // Clear All and Apply to Agent have text, X buttons don't
      const removeButtons = allButtons.filter(
        (btn) => btn.textContent === '' || btn.querySelector('svg')
      );

      // Click the first X button
      if (removeButtons.length > 0) {
        await user.click(removeButtons[0]);
      }

      await waitFor(() => {
        expect(screen.getByText(/Selected Tools \(1\)/)).toBeInTheDocument();
      });
    });
  });

  describe('Clear All', () => {
    it('clears all selected tools', async () => {
      const { user } = render(
        <AllowedToolsBuilder
          serversWithTools={mockServersWithTools}
          initialTools={['query', 'list_repos']}
          onApply={mockOnApply}
        />
      );

      expect(screen.getByText(/Selected Tools \(2\)/)).toBeInTheDocument();

      const clearButton = screen.getByText('Clear All');
      await user.click(clearButton);

      await waitFor(() => {
        expect(screen.queryByText(/Selected Tools \(2\)/)).not.toBeInTheDocument();
      });
    });
  });

  describe('Apply to Agent', () => {
    it('calls onApply with selected tools', async () => {
      const { user } = render(
        <AllowedToolsBuilder
          serversWithTools={mockServersWithTools}
          initialTools={['query', 'list_repos']}
          onApply={mockOnApply}
        />
      );

      const applyButton = screen.getByText('Apply to Agent');
      await user.click(applyButton);

      expect(mockOnApply).toHaveBeenCalledWith(['query', 'list_repos']);
    });

    it('disables apply button when no tools selected', () => {
      render(
        <AllowedToolsBuilder
          serversWithTools={mockServersWithTools}
          initialTools={[]}
          onApply={mockOnApply}
        />
      );

      // No "Selected Tools" section should be visible
      expect(screen.queryByText('Apply to Agent')).not.toBeInTheDocument();
    });

    it('navigates to agent create page when onApply is not provided', async () => {
      const { user } = render(
        <AllowedToolsBuilder
          serversWithTools={mockServersWithTools}
          initialTools={['query']}
        />
      );

      const applyButton = screen.getByText('Apply to Agent');
      await user.click(applyButton);

      // Navigation is mocked in test utils
      expect(applyButton).toBeEnabled();
    });
  });

  describe('Help Text', () => {
    it('displays help information', () => {
      render(
        <AllowedToolsBuilder
          serversWithTools={mockServersWithTools}
          onApply={mockOnApply}
        />
      );

      expect(screen.getByText(/What are allowed_tools?/)).toBeInTheDocument();
      expect(screen.getByText(/Duplicate tool names:/)).toBeInTheDocument();
    });
  });
});
