import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '@/test/utils';
import { AuditAdvancedFilters } from '../AuditAdvancedFilters';
import { INITIAL_ADVANCED_FILTERS, type AdvancedFilters } from '../hooks';

describe('AuditAdvancedFilters', () => {
  const defaultProps = {
    value: INITIAL_ADVANCED_FILTERS,
    onChange: vi.fn(),
  };

  describe('Disclosure Toggle', () => {
    it('renders the Filters button', () => {
      render(<AuditAdvancedFilters {...defaultProps} />);
      expect(screen.getByText('Filters')).toBeInTheDocument();
    });

    it('shows filter count badge when filters are active', () => {
      const activeFilters: AdvancedFilters = {
        ...INITIAL_ADVANCED_FILTERS,
        agentId: 'test-agent',
        server: 'test-server',
      };
      render(<AuditAdvancedFilters {...defaultProps} value={activeFilters} />);
      expect(screen.getByText('2')).toBeInTheDocument();
    });

    it('does not show badge when no filters are active', () => {
      render(<AuditAdvancedFilters {...defaultProps} />);
      expect(screen.queryByText('0')).not.toBeInTheDocument();
    });

    it('expands panel when clicked', async () => {
      const user = userEvent.setup();
      render(<AuditAdvancedFilters {...defaultProps} />);

      await user.click(screen.getByText('Filters'));

      expect(screen.getByText('Advanced Filters')).toBeInTheDocument();
    });
  });

  describe('Text Input Fields', () => {
    async function openFilters() {
      const user = userEvent.setup();
      await user.click(screen.getByText('Filters'));
      return user;
    }

    it('renders Agent ID input', async () => {
      render(<AuditAdvancedFilters {...defaultProps} />);
      const user = await openFilters();

      const input = screen.getByLabelText('Agent ID');
      expect(input).toBeInTheDocument();
      expect(input).toHaveAttribute('placeholder', 'e.g., 9d2724e9-...');
    });

    it('calls onChange when Agent ID is changed', async () => {
      const onChange = vi.fn();
      render(<AuditAdvancedFilters {...defaultProps} onChange={onChange} />);
      const user = await openFilters();

      const input = screen.getByLabelText('Agent ID');
      await user.type(input, 'x');

      // onChange is called for keystrokes
      expect(onChange).toHaveBeenCalled();
      // First call should have the character
      const firstCall = onChange.mock.calls[0][0];
      expect(firstCall.agentId).toBe('x');
    });

    it('renders Request ID input', async () => {
      render(<AuditAdvancedFilters {...defaultProps} />);
      await openFilters();

      const input = screen.getByLabelText('Request ID');
      expect(input).toBeInTheDocument();
      expect(input).toHaveAttribute('placeholder', 'e.g., req-abc123...');
    });

    it('calls onChange when Request ID is changed', async () => {
      const onChange = vi.fn();
      render(<AuditAdvancedFilters {...defaultProps} onChange={onChange} />);
      const user = await openFilters();

      const input = screen.getByLabelText('Request ID');
      await user.type(input, 'r');

      // onChange is called for keystrokes
      expect(onChange).toHaveBeenCalled();
      // First call should have the character
      const firstCall = onChange.mock.calls[0][0];
      expect(firstCall.requestId).toBe('r');
    });

    it('renders Server input', async () => {
      render(<AuditAdvancedFilters {...defaultProps} />);
      await openFilters();

      const input = screen.getByLabelText('Server');
      expect(input).toBeInTheDocument();
      expect(input).toHaveAttribute('placeholder', 'e.g., sqlite');
    });

    it('renders Tool input', async () => {
      render(<AuditAdvancedFilters {...defaultProps} />);
      await openFilters();

      const input = screen.getByLabelText('Tool');
      expect(input).toBeInTheDocument();
      expect(input).toHaveAttribute('placeholder', 'e.g., read_query');
    });
  });

  describe('Actions Multi-Select', () => {
    async function openFilters() {
      const user = userEvent.setup();
      await user.click(screen.getByText('Filters'));
      return user;
    }

    it('renders Actions label', async () => {
      render(<AuditAdvancedFilters {...defaultProps} />);
      await openFilters();

      expect(screen.getByText('Actions')).toBeInTheDocument();
    });

    it('shows placeholder when no actions selected', async () => {
      render(<AuditAdvancedFilters {...defaultProps} />);
      await openFilters();

      expect(screen.getByText('Select actions...')).toBeInTheDocument();
    });

    it('shows action count when actions are selected', async () => {
      const filtersWithActions: AdvancedFilters = {
        ...INITIAL_ADVANCED_FILTERS,
        actions: ['tools/list', 'tools/call'],
      };
      render(<AuditAdvancedFilters {...defaultProps} value={filtersWithActions} />);
      await openFilters();

      expect(screen.getByText('2 action(s) selected')).toBeInTheDocument();
    });

    it('opens dropdown when clicked', async () => {
      render(<AuditAdvancedFilters {...defaultProps} />);
      const user = await openFilters();

      await user.click(screen.getByText('Select actions...'));

      expect(screen.getByText('MCP Operations')).toBeInTheDocument();
      expect(screen.getByText('Agent Management')).toBeInTheDocument();
    });

    it('shows action categories in dropdown', async () => {
      render(<AuditAdvancedFilters {...defaultProps} />);
      const user = await openFilters();

      await user.click(screen.getByText('Select actions...'));

      expect(screen.getByText('MCP Operations')).toBeInTheDocument();
      expect(screen.getByText('Agent Management')).toBeInTheDocument();
      expect(screen.getByText('API Key Management')).toBeInTheDocument();
      expect(screen.getByText('Token Management')).toBeInTheDocument();
    });

    it('calls onChange when action is toggled', async () => {
      const onChange = vi.fn();
      render(<AuditAdvancedFilters {...defaultProps} onChange={onChange} />);
      const user = await openFilters();

      await user.click(screen.getByText('Select actions...'));
      await user.click(screen.getByText('tools/list'));

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          actions: ['tools/list'],
        })
      );
    });

    it('renders selected action tags', async () => {
      const filtersWithActions: AdvancedFilters = {
        ...INITIAL_ADVANCED_FILTERS,
        actions: ['tools/list'],
      };
      render(<AuditAdvancedFilters {...defaultProps} value={filtersWithActions} />);
      await openFilters();

      // Find the tag for tools/list (not inside the dropdown)
      const tags = screen.getAllByText('tools/list');
      expect(tags.length).toBeGreaterThan(0);
    });

    it('removes action when tag X is clicked', async () => {
      const onChange = vi.fn();
      const filtersWithActions: AdvancedFilters = {
        ...INITIAL_ADVANCED_FILTERS,
        actions: ['tools/list', 'tools/call'],
      };
      render(<AuditAdvancedFilters {...defaultProps} value={filtersWithActions} onChange={onChange} />);
      const user = await openFilters();

      await user.click(screen.getByRole('button', { name: 'Remove action tools/list' }));

      expect(onChange).toHaveBeenCalledWith(
        expect.objectContaining({
          actions: ['tools/call'],
        })
      );
    });
  });

  describe('Clear All', () => {
    async function openFilters() {
      const user = userEvent.setup();
      await user.click(screen.getByText('Filters'));
      return user;
    }

    it('shows Clear all button when filters are active', async () => {
      const activeFilters: AdvancedFilters = {
        ...INITIAL_ADVANCED_FILTERS,
        agentId: 'test-agent',
      };
      render(<AuditAdvancedFilters {...defaultProps} value={activeFilters} />);
      await openFilters();

      expect(screen.getByText('Clear all')).toBeInTheDocument();
    });

    it('does not show Clear all when no filters are active', async () => {
      render(<AuditAdvancedFilters {...defaultProps} />);
      await openFilters();

      expect(screen.queryByText('Clear all')).not.toBeInTheDocument();
    });

    it('calls onChange with initial filters when Clear all is clicked', async () => {
      const onChange = vi.fn();
      const activeFilters: AdvancedFilters = {
        agentId: 'test-agent',
        actions: ['tools/list'],
        requestId: 'req-123',
        server: 'sqlite',
        tool: 'read_query',
        userId: '',
      };
      render(<AuditAdvancedFilters {...defaultProps} value={activeFilters} onChange={onChange} />);
      const user = await openFilters();

      await user.click(screen.getByText('Clear all'));

      expect(onChange).toHaveBeenCalledWith(INITIAL_ADVANCED_FILTERS);
    });
  });

  describe('Filter Badge Count', () => {
    it('counts each active filter', () => {
      const testCases = [
        { filters: { ...INITIAL_ADVANCED_FILTERS, agentId: 'test' }, expected: 1 },
        { filters: { ...INITIAL_ADVANCED_FILTERS, actions: ['tools/list'] }, expected: 1 },
        { filters: { ...INITIAL_ADVANCED_FILTERS, requestId: 'req-123' }, expected: 1 },
        { filters: { ...INITIAL_ADVANCED_FILTERS, server: 'sqlite' }, expected: 1 },
        { filters: { ...INITIAL_ADVANCED_FILTERS, tool: 'read_query' }, expected: 1 },
        {
          filters: {
            agentId: 'test',
            actions: ['tools/list'],
            requestId: 'req-123',
            server: 'sqlite',
            tool: 'read_query',
            userId: '',
          },
          expected: 5,
        },
      ];

      testCases.forEach(({ filters, expected }) => {
        const { container } = render(<AuditAdvancedFilters {...defaultProps} value={filters} />);
        const badge = container.querySelector('.rounded-full');
        if (expected > 0) {
          expect(badge).toHaveTextContent(expected.toString());
        }
      });
    });
  });

  describe('Admin Mode', () => {
    async function openFilters() {
      const user = userEvent.setup();
      await user.click(screen.getByText('Filters'));
      return user;
    }

    it('shows User ID field when isAdmin is true', async () => {
      render(<AuditAdvancedFilters {...defaultProps} isAdmin={true} />);
      await openFilters();

      expect(screen.getByLabelText('User ID')).toBeInTheDocument();
    });

    it('hides User ID field when isAdmin is false', async () => {
      render(<AuditAdvancedFilters {...defaultProps} isAdmin={false} />);
      await openFilters();

      expect(screen.queryByLabelText('User ID')).not.toBeInTheDocument();
    });

    it('calls onChange when User ID is typed in admin mode', async () => {
      const onChange = vi.fn();
      render(<AuditAdvancedFilters {...defaultProps} onChange={onChange} isAdmin={true} />);
      const user = await openFilters();

      const userIdInput = screen.getByLabelText('User ID');
      await user.type(userIdInput, 'l');

      expect(onChange).toHaveBeenCalled();
      const lastCall = onChange.mock.calls[onChange.mock.calls.length - 1][0];
      expect(lastCall.userId).toBe('l');
    });
  });
});
