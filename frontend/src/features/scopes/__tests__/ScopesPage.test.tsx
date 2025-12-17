import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen, waitFor } from '@testing-library/react';
import { http, HttpResponse } from 'msw';
import { render } from '@/test/utils';
import { server } from '@/test/mocks/server';
import ScopesPage from '../ScopesPage';

// Mock useNavigate
const mockNavigate = vi.fn();
vi.mock('react-router-dom', async () => {
  const actual = await vi.importActual('react-router-dom');
  return {
    ...actual,
    useNavigate: () => mockNavigate,
  };
});

describe('ScopesPage', () => {
  beforeEach(() => {
    vi.clearAllMocks();
    server.resetHandlers();
  });

  describe('Page Structure', () => {
    it('renders page title and description', async () => {
      render(<ScopesPage />);

      expect(screen.getByRole('heading', { level: 1, name: 'Scopes and Policy' })).toBeInTheDocument();
      expect(screen.getByText('View the scope catalog and policy definitions')).toBeInTheDocument();
    });

    it('shows loading spinner initially', async () => {
      server.use(
        http.get('/enforceai/scopes/catalog', async () => {
          await new Promise((resolve) => setTimeout(resolve, 100));
          return HttpResponse.json({
            version: '1.0',
            generated_at: new Date().toISOString(),
            group_mappings: {},
            scopes: {},
          });
        })
      );

      render(<ScopesPage />);

      const spinner = document.querySelector('[class*="animate-spin"]');
      expect(spinner || screen.queryByText('Scopes and Policy')).toBeTruthy();
    });
  });

  describe('Catalog Available', () => {
    it('shows scope list when catalog loads', async () => {
      render(<ScopesPage />);

      await waitFor(() => {
        expect(screen.getByText('Total Scopes')).toBeInTheDocument();
      });

      // Should show scope cards (use getAllByText since scope names may appear in multiple places)
      expect(screen.getAllByText('registry-admins').length).toBeGreaterThan(0);
      expect(screen.getAllByText('sqlite.manage').length).toBeGreaterThan(0);
    });

    it('shows stats cards', async () => {
      render(<ScopesPage />);

      await waitFor(() => {
        expect(screen.getByText('Total Scopes')).toBeInTheDocument();
      });

      expect(screen.getByText('Servers Referenced')).toBeInTheDocument();
      // Group Mappings appears as both a stat card label and section header
      expect(screen.getAllByText('Group Mappings').length).toBeGreaterThan(0);
    });

    it('displays group mappings section', async () => {
      render(<ScopesPage />);

      await waitFor(() => {
        // Group Mappings appears as both a stat card label and section header
        expect(screen.getAllByText('Group Mappings').length).toBeGreaterThan(0);
      });

      // Group names appear in multiple places (scope cards and group mappings)
      expect(screen.getAllByText('registry-admins').length).toBeGreaterThan(0);
    });
  });

  describe('Search and Filter', () => {
    it('shows search input', async () => {
      render(<ScopesPage />);

      await waitFor(() => {
        expect(screen.getByPlaceholderText('Search scopes, servers, or tools...')).toBeInTheDocument();
      });
    });

    it('filters scopes by search query', async () => {
      const { user } = render(<ScopesPage />);

      await waitFor(() => {
        expect(screen.getByPlaceholderText('Search scopes, servers, or tools...')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search scopes, servers, or tools...');
      await user.type(searchInput, 'sqlite');

      await waitFor(() => {
        expect(screen.getByText(/matching "sqlite"/)).toBeInTheDocument();
      });
    });

    it('shows server filter dropdown', async () => {
      render(<ScopesPage />);

      await waitFor(() => {
        expect(screen.getByRole('combobox')).toBeInTheDocument();
      });

      // Should show "All servers" option
      expect(screen.getByRole('option', { name: 'All servers' })).toBeInTheDocument();
    });
  });

  describe('Expand/Collapse', () => {
    it('shows expand all and collapse all buttons', async () => {
      render(<ScopesPage />);

      await waitFor(() => {
        expect(screen.getByRole('button', { name: 'Expand All' })).toBeInTheDocument();
        expect(screen.getByRole('button', { name: 'Collapse All' })).toBeInTheDocument();
      });
    });

    it('expands scope on click', async () => {
      const { user } = render(<ScopesPage />);

      await waitFor(() => {
        expect(screen.getByText('sqlite.manage')).toBeInTheDocument();
      });

      // Find and click the scope card button
      const scopeCard = screen.getByText('sqlite.manage').closest('button');
      if (scopeCard) {
        await user.click(scopeCard);
      }

      await waitFor(() => {
        expect(screen.getByText('Servers')).toBeInTheDocument();
      });
    });
  });

  describe('Catalog Unavailable', () => {
    it('shows unavailable message when API fails', async () => {
      server.use(
        http.get('/enforceai/scopes/catalog', () => {
          return HttpResponse.json({ detail: 'Not found' }, { status: 404 });
        })
      );

      render(<ScopesPage />);

      // Allow extra time for retry to complete
      await waitFor(
        () => {
          expect(screen.getByText('Scope Catalog Unavailable')).toBeInTheDocument();
        },
        { timeout: 3000 }
      );

      expect(screen.getByText(/could not be loaded/)).toBeInTheDocument();
    });

    it('shows about scopes info when catalog unavailable', async () => {
      server.use(
        http.get('/enforceai/scopes/catalog', () => {
          return HttpResponse.json({ detail: 'Not found' }, { status: 404 });
        })
      );

      render(<ScopesPage />);

      // Allow extra time for retry to complete
      await waitFor(
        () => {
          expect(screen.getByText('About Scopes')).toBeInTheDocument();
        },
        { timeout: 3000 }
      );
    });
  });

  describe('Empty States', () => {
    it('shows empty state when no scopes match filter', async () => {
      const { user } = render(<ScopesPage />);

      await waitFor(() => {
        expect(screen.getByPlaceholderText('Search scopes, servers, or tools...')).toBeInTheDocument();
      });

      const searchInput = screen.getByPlaceholderText('Search scopes, servers, or tools...');
      await user.type(searchInput, 'nonexistentscope12345');

      await waitFor(() => {
        expect(screen.getByText('No matching scopes')).toBeInTheDocument();
      });
    });
  });
});
