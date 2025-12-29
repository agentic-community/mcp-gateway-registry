import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '@/test/utils';
import AuditPage from '../AuditPage';

describe('AuditPage', () => {
  // All tests need withAuth because AuditExplorer uses useAuth
  const renderOptions = { withAuth: true };

  describe('Page Rendering', () => {
    it('renders page header', () => {
      render(<AuditPage />, renderOptions);

      expect(screen.getByText('Audit')).toBeInTheDocument();
      expect(
        screen.getByText('Explore audit events and access enforcement logs')
      ).toBeInTheDocument();
    });

    it('renders tab navigation', () => {
      render(<AuditPage />, renderOptions);

      expect(screen.getByRole('tab', { name: /explorer/i })).toBeInTheDocument();
      expect(screen.getByRole('tab', { name: /guidance/i })).toBeInTheDocument();
    });

    it('shows Explorer tab by default', () => {
      render(<AuditPage />, renderOptions);

      const explorerTab = screen.getByRole('tab', { name: /explorer/i });
      expect(explorerTab).toHaveAttribute('aria-selected', 'true');
    });
  });

  describe('Explorer Tab', () => {
    it('renders the audit explorer component', async () => {
      render(<AuditPage />, renderOptions);

      // Explorer shows time filter presets
      expect(screen.getByText('Time:')).toBeInTheDocument();
      expect(screen.getByText('15 minutes')).toBeInTheDocument();
      expect(screen.getByText('1 hour')).toBeInTheDocument();
      expect(screen.getByText('24 hours')).toBeInTheDocument();
      expect(screen.getByText('7 days')).toBeInTheDocument();
    });

    it('renders outcome filter', () => {
      render(<AuditPage />, renderOptions);

      expect(screen.getByText('Outcome:')).toBeInTheDocument();
      expect(screen.getByText('All')).toBeInTheDocument();
      expect(screen.getByText('Allowed')).toBeInTheDocument();
      expect(screen.getByText('Denied')).toBeInTheDocument();
    });

    it('renders refresh button', () => {
      render(<AuditPage />, renderOptions);

      expect(screen.getByRole('button', { name: /refresh/i })).toBeInTheDocument();
    });
  });

  describe('Guidance Tab', () => {
    async function switchToGuidanceTab() {
      const user = userEvent.setup();
      const guidanceTab = screen.getByRole('tab', { name: /guidance/i });
      await user.click(guidanceTab);
    }

    it('shows guidance content when clicked', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(screen.getByText('How to Access Audit Events')).toBeInTheDocument();
    });

    it('renders how to access audit events section', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(screen.getByText('How to Access Audit Events')).toBeInTheDocument();
      expect(screen.getByText('Docker Compose Logs')).toBeInTheDocument();
      expect(screen.getByText('SQLite Database')).toBeInTheDocument();
      expect(screen.getByText('Filtering by Request ID')).toBeInTheDocument();
    });

    it('renders request ID display section', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(screen.getByText('Current Session Request ID')).toBeInTheDocument();
      expect(
        screen.getByText('Use this request ID to correlate audit events from your current session:')
      ).toBeInTheDocument();
    });

    it('renders audit actions glossary section', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(screen.getByText('Common Audit Actions')).toBeInTheDocument();
      expect(
        screen.getByText(/Audit events are categorized by action type/)
      ).toBeInTheDocument();
    });

    it('renders additional resources section', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(screen.getByText('Additional Resources')).toBeInTheDocument();
    });
  });

  describe('Docker Compose Guidance', () => {
    async function switchToGuidanceTab() {
      const user = userEvent.setup();
      const guidanceTab = screen.getByRole('tab', { name: /guidance/i });
      await user.click(guidanceTab);
    }

    it('shows docker compose logs command', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(screen.getByText('docker-compose logs -f auth_server')).toBeInTheDocument();
    });

    it('describes audit events in stdout', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(
        screen.getByText(/Audit events are written to stdout and can be viewed using Docker Compose/)
      ).toBeInTheDocument();
    });
  });

  describe('SQLite Database Guidance', () => {
    async function switchToGuidanceTab() {
      const user = userEvent.setup();
      const guidanceTab = screen.getByRole('tab', { name: /guidance/i });
      await user.click(guidanceTab);
    }

    it('shows SQLite database path', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(screen.getByText('$ENFORCEAI_STATE_DIR/enforceai_audit.db')).toBeInTheDocument();
    });

    it('describes SQLite audit storage', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(
        screen.getByText(/If SQLite audit storage is configured, audit events are persisted to/)
      ).toBeInTheDocument();
    });
  });

  describe('Request ID Display', () => {
    async function switchToGuidanceTab() {
      const user = userEvent.setup();
      const guidanceTab = screen.getByRole('tab', { name: /guidance/i });
      await user.click(guidanceTab);
    }

    it('displays a sample request ID', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      // Request ID should match the pattern req-{timestamp}-{random}
      const codeElements = screen.getAllByRole('code');
      const requestIdElement = codeElements.find((el) => el.textContent?.startsWith('req-'));
      expect(requestIdElement).toBeDefined();
    });

    it('shows note about real implementation', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(
        screen.getByText(/Note: In a real implementation, this would show the actual X-Request-Id/)
      ).toBeInTheDocument();
    });
  });

  describe('Audit Actions Glossary', () => {
    async function switchToGuidanceTab() {
      const user = userEvent.setup();
      const guidanceTab = screen.getByRole('tab', { name: /guidance/i });
      await user.click(guidanceTab);
    }

    it('displays MCP Operations category', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(screen.getByText('MCP Operations')).toBeInTheDocument();
      expect(screen.getByText('tools/list')).toBeInTheDocument();
      expect(screen.getByText('tools/call')).toBeInTheDocument();
    });

    it('displays Agent Management category', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(screen.getByText('Agent Management')).toBeInTheDocument();
      expect(screen.getByText('management/agents/create')).toBeInTheDocument();
      expect(screen.getByText('management/agents/update')).toBeInTheDocument();
      expect(screen.getByText('management/agents/revoke')).toBeInTheDocument();
      expect(screen.getByText('management/agents/list')).toBeInTheDocument();
      expect(screen.getByText('management/agents/get')).toBeInTheDocument();
    });

    it('displays API Key Management category', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(screen.getByText('API Key Management')).toBeInTheDocument();
      expect(screen.getByText('management/api-keys/create')).toBeInTheDocument();
      expect(screen.getByText('management/api-keys/revoke')).toBeInTheDocument();
      expect(screen.getByText('management/api-keys/list')).toBeInTheDocument();
    });

    it('displays Token Management category', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(screen.getByText('Token Management')).toBeInTheDocument();
      expect(screen.getByText('management/tokens/mint')).toBeInTheDocument();
      expect(screen.getByText('management/tokens/revoke')).toBeInTheDocument();
      expect(screen.getByText('management/tokens/revoke-all')).toBeInTheDocument();
    });

    it('displays action descriptions', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(screen.getByText('List available tools from a server')).toBeInTheDocument();
      expect(screen.getByText('Execute a tool')).toBeInTheDocument();
      expect(screen.getByText('Create a new EnforceAI agent')).toBeInTheDocument();
      expect(screen.getByText('Update agent configuration')).toBeInTheDocument();
    });

    it('displays severity badges', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      // Check for all severity levels
      const infoBadges = screen.getAllByText('info');
      expect(infoBadges.length).toBeGreaterThan(0);

      const warningBadges = screen.getAllByText('warning');
      expect(warningBadges.length).toBeGreaterThan(0);

      const errorBadges = screen.getAllByText('error');
      expect(errorBadges.length).toBeGreaterThan(0);
    });
  });

  describe('Additional Resources', () => {
    async function switchToGuidanceTab() {
      const user = userEvent.setup();
      const guidanceTab = screen.getByRole('tab', { name: /guidance/i });
      await user.click(guidanceTab);
    }

    it('links to audit retention documentation', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(
        screen.getByText('enforceai/instructions/ENFORCEAI_AUDIT_RETENTION.md')
      ).toBeInTheDocument();
      expect(screen.getByText(/for retention policy configuration/)).toBeInTheDocument();
    });

    it('links to management guide', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(
        screen.getByText('enforceai/instructions/ENFORCEAI_MANAGEMENT.md')
      ).toBeInTheDocument();
      expect(screen.getByText(/for operational procedures/)).toBeInTheDocument();
    });

    it('links to setup guide', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      expect(screen.getByText('docs/enforceai-setup-guide.md')).toBeInTheDocument();
      expect(screen.getByText(/for initial configuration/)).toBeInTheDocument();
    });
  });

  describe('Code Examples', () => {
    async function switchToGuidanceTab() {
      const user = userEvent.setup();
      const guidanceTab = screen.getByRole('tab', { name: /guidance/i });
      await user.click(guidanceTab);
    }

    it('shows grep command example with request ID', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      // The grep command should contain "docker-compose logs" and "grep"
      const text = screen.getByText(/docker-compose logs auth_server \| grep/);
      expect(text).toBeInTheDocument();
    });

    it('renders X-Request-Id header mention', async () => {
      render(<AuditPage />, renderOptions);
      await switchToGuidanceTab();

      // X-Request-Id appears in multiple places (code element and note)
      const elements = screen.getAllByText(/X-Request-Id/);
      expect(elements.length).toBeGreaterThan(0);
    });
  });
});
