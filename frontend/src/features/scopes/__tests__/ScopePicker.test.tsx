import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import { ScopePicker } from '../ScopePicker';

describe('ScopePicker', () => {
  const mockOnChange = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();
  });

  describe('Basic Rendering', () => {
    it('renders with placeholder', async () => {
      render(
        <ScopePicker
          value={[]}
          onChange={mockOnChange}
          placeholder="Select scopes..."
        />
      );

      expect(screen.getByText('Select scopes...')).toBeInTheDocument();
    });

    it('renders with label', async () => {
      render(
        <ScopePicker
          value={[]}
          onChange={mockOnChange}
          label="Scopes"
        />
      );

      expect(screen.getByText('Scopes')).toBeInTheDocument();
    });

    it('renders with helper text', async () => {
      render(
        <ScopePicker
          value={[]}
          onChange={mockOnChange}
          helperText="Select one or more scopes"
        />
      );

      expect(screen.getByText('Select one or more scopes')).toBeInTheDocument();
    });

    it('renders with error message', async () => {
      render(
        <ScopePicker
          value={[]}
          onChange={mockOnChange}
          error="At least one scope is required"
        />
      );

      expect(screen.getByText('At least one scope is required')).toBeInTheDocument();
    });
  });

  describe('Selected Values', () => {
    it('displays selected scopes as badges', async () => {
      render(
        <ScopePicker
          value={['sqlite.manage', 'registry-admins']}
          onChange={mockOnChange}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('sqlite.manage')).toBeInTheDocument();
        expect(screen.getByText('registry-admins')).toBeInTheDocument();
      });
    });

    it('calls onChange when removing a scope', async () => {
      const { user } = render(
        <ScopePicker
          value={['sqlite.manage', 'registry-admins']}
          onChange={mockOnChange}
        />
      );

      await waitFor(() => {
        expect(screen.getByText('sqlite.manage')).toBeInTheDocument();
      });

      // Find the X button for sqlite.manage and click it
      const sqliteBadge = screen.getByText('sqlite.manage').parentElement;
      const removeButton = sqliteBadge?.querySelector('[role="button"]');
      if (removeButton) {
        await user.click(removeButton);
      }

      expect(mockOnChange).toHaveBeenCalledWith(['registry-admins']);
    });
  });

  describe('Dropdown Behavior', () => {
    it('opens dropdown on click', async () => {
      const { user } = render(
        <ScopePicker
          value={[]}
          onChange={mockOnChange}
        />
      );

      const trigger = screen.getByText('Select scopes...').closest('button');
      if (trigger) {
        await user.click(trigger);
      }

      await waitFor(() => {
        expect(screen.getByPlaceholderText('Search scopes...')).toBeInTheDocument();
      });
    });

    it('shows scope options when dropdown is open', async () => {
      const { user } = render(
        <ScopePicker
          value={[]}
          onChange={mockOnChange}
        />
      );

      const trigger = screen.getByText('Select scopes...').closest('button');
      if (trigger) {
        await user.click(trigger);
      }

      await waitFor(() => {
        expect(screen.getByText('sqlite.manage')).toBeInTheDocument();
        expect(screen.getByText('registry-admins')).toBeInTheDocument();
      });
    });

    it('filters scopes by search query', async () => {
      const { user } = render(
        <ScopePicker
          value={[]}
          onChange={mockOnChange}
        />
      );

      const trigger = screen.getByText('Select scopes...').closest('button');
      if (trigger) {
        await user.click(trigger);
      }

      await waitFor(() => {
        expect(screen.getByPlaceholderText('Search scopes...')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search scopes...');
      await user.type(searchInput, 'sqlite');

      await waitFor(() => {
        expect(screen.getByText('sqlite.manage')).toBeInTheDocument();
      });
    });
  });

  describe('Selection', () => {
    it('selects a scope when clicked', async () => {
      const { user } = render(
        <ScopePicker
          value={[]}
          onChange={mockOnChange}
        />
      );

      const trigger = screen.getByText('Select scopes...').closest('button');
      if (trigger) {
        await user.click(trigger);
      }

      await waitFor(() => {
        expect(screen.getByText('sqlite.manage')).toBeInTheDocument();
      });

      // Click on the scope option
      const scopeOption = screen.getByText('sqlite.manage').closest('button');
      if (scopeOption) {
        await user.click(scopeOption);
      }

      expect(mockOnChange).toHaveBeenCalledWith(['sqlite.manage']);
    });

    it('deselects a scope when clicked again', async () => {
      const { user } = render(
        <ScopePicker
          value={['sqlite.manage']}
          onChange={mockOnChange}
        />
      );

      // The selected scope should be visible as a badge
      expect(screen.getByText('sqlite.manage')).toBeInTheDocument();

      // Click to open dropdown
      const trigger = screen.getByText('sqlite.manage').closest('button');
      if (trigger) {
        await user.click(trigger);
      }

      await waitFor(() => {
        // Wait for dropdown to open and show scope options
        expect(screen.getByPlaceholderText('Search scopes...')).toBeInTheDocument();
      });

      // Find the sqlite.manage option in the dropdown list (not the badge)
      // The dropdown list has the scrollable container with max-h-48
      const dropdownList = document.querySelector('.max-h-48');
      const scopeOption = dropdownList?.querySelector('button');

      // Find and click the sqlite.manage option to deselect
      const allButtons = dropdownList?.querySelectorAll('button');
      const sqliteOption = Array.from(allButtons || []).find(
        btn => btn.textContent?.includes('sqlite.manage')
      );

      if (sqliteOption) {
        await user.click(sqliteOption);
      }

      expect(mockOnChange).toHaveBeenCalledWith([]);
    });
  });

  describe('Max Selection', () => {
    it('shows selection count when maxSelect is set', async () => {
      const { user } = render(
        <ScopePicker
          value={['sqlite.manage']}
          onChange={mockOnChange}
          maxSelect={3}
        />
      );

      const trigger = screen.getByText('sqlite.manage').closest('button');
      if (trigger) {
        await user.click(trigger);
      }

      await waitFor(() => {
        expect(screen.getByText('1 / 3 selected')).toBeInTheDocument();
      });
    });
  });

  describe('Disabled State', () => {
    it('does not open dropdown when disabled', async () => {
      const { user } = render(
        <ScopePicker
          value={[]}
          onChange={mockOnChange}
          disabled
        />
      );

      const trigger = screen.getByText('Select scopes...').closest('button');
      if (trigger) {
        await user.click(trigger);
      }

      // Dropdown should not open
      expect(screen.queryByPlaceholderText('Search scopes...')).not.toBeInTheDocument();
    });
  });

  describe('Catalog Unavailable', () => {
    it('shows unavailable message when API fails', async () => {
      server.use(
        http.get('/enforceai/scopes/catalog', () => {
          return HttpResponse.json({ detail: 'Not found' }, { status: 404 });
        })
      );

      const { user } = render(
        <ScopePicker
          value={[]}
          onChange={mockOnChange}
        />
      );

      const trigger = screen.getByText('Select scopes...').closest('button');
      if (trigger) {
        await user.click(trigger);
      }

      // Allow extra time for retry to complete
      await waitFor(
        () => {
          expect(screen.getByText('Scope catalog unavailable')).toBeInTheDocument();
        },
        { timeout: 3000 }
      );
    });
  });
});
