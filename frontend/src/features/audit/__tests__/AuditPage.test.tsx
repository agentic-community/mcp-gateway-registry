import { describe, it, expect } from 'vitest';
import { screen } from '@testing-library/react';
import { render } from '@/test/utils';
import AuditPage from '../AuditPage';

describe('AuditPage', () => {
  describe('Page Rendering', () => {
    it('renders page header', () => {
      render(<AuditPage />);

      expect(screen.getByText('Audit')).toBeInTheDocument();
      expect(
        screen.getByText('Guidance on accessing audit events and understanding enforcement logs')
      ).toBeInTheDocument();
    });

    it('renders future viewer placeholder', () => {
      render(<AuditPage />);

      expect(screen.getByText('Audit Event Viewer Coming Soon')).toBeInTheDocument();
      expect(
        screen.getByText(
          /A dedicated audit event viewer with filtering capabilities is planned for a future release/
        )
      ).toBeInTheDocument();
    });

    it('renders how to access audit events section', () => {
      render(<AuditPage />);

      expect(screen.getByText('How to Access Audit Events')).toBeInTheDocument();
      expect(screen.getByText('Docker Compose Logs')).toBeInTheDocument();
      expect(screen.getByText('SQLite Database')).toBeInTheDocument();
      expect(screen.getByText('Filtering by Request ID')).toBeInTheDocument();
    });

    it('renders request ID display section', () => {
      render(<AuditPage />);

      expect(screen.getByText('Current Session Request ID')).toBeInTheDocument();
      expect(
        screen.getByText('Use this request ID to correlate audit events from your current session:')
      ).toBeInTheDocument();
    });

    it('renders audit actions glossary section', () => {
      render(<AuditPage />);

      expect(screen.getByText('Common Audit Actions')).toBeInTheDocument();
      expect(
        screen.getByText(/Audit events are categorized by action type/)
      ).toBeInTheDocument();
    });

    it('renders additional resources section', () => {
      render(<AuditPage />);

      expect(screen.getByText('Additional Resources')).toBeInTheDocument();
    });
  });

  describe('Docker Compose Guidance', () => {
    it('shows docker compose logs command', () => {
      render(<AuditPage />);

      expect(screen.getByText('docker-compose logs -f auth_server')).toBeInTheDocument();
    });

    it('describes audit events in stdout', () => {
      render(<AuditPage />);

      expect(
        screen.getByText(/Audit events are written to stdout and can be viewed using Docker Compose/)
      ).toBeInTheDocument();
    });
  });

  describe('SQLite Database Guidance', () => {
    it('shows SQLite database path', () => {
      render(<AuditPage />);

      expect(screen.getByText('$ENFORCEAI_STATE_DIR/enforceai_audit.db')).toBeInTheDocument();
    });

    it('describes SQLite audit storage', () => {
      render(<AuditPage />);

      expect(
        screen.getByText(/If SQLite audit storage is configured, audit events are persisted to/)
      ).toBeInTheDocument();
    });
  });

  describe('Request ID Display', () => {
    it('displays a sample request ID', () => {
      render(<AuditPage />);

      // Request ID should match the pattern req-{timestamp}-{random}
      const codeElements = screen.getAllByRole('code');
      const requestIdElement = codeElements.find((el) => el.textContent?.startsWith('req-'));
      expect(requestIdElement).toBeDefined();
    });

    it('shows copy button for request ID', () => {
      render(<AuditPage />);

      // CopyButton component should be present
      // It's implemented as a button, so we can check for buttons
      const buttons = screen.getAllByRole('button');
      expect(buttons.length).toBeGreaterThan(0);
    });

    it('shows note about real implementation', () => {
      render(<AuditPage />);

      expect(
        screen.getByText(/Note: In a real implementation, this would show the actual X-Request-Id/)
      ).toBeInTheDocument();
    });
  });

  describe('Audit Actions Glossary', () => {
    it('displays MCP Operations category', () => {
      render(<AuditPage />);

      expect(screen.getByText('MCP Operations')).toBeInTheDocument();
      expect(screen.getByText('tools/list')).toBeInTheDocument();
      expect(screen.getByText('tools/call')).toBeInTheDocument();
    });

    it('displays Agent Management category', () => {
      render(<AuditPage />);

      expect(screen.getByText('Agent Management')).toBeInTheDocument();
      expect(screen.getByText('management/agents/create')).toBeInTheDocument();
      expect(screen.getByText('management/agents/update')).toBeInTheDocument();
      expect(screen.getByText('management/agents/revoke')).toBeInTheDocument();
      expect(screen.getByText('management/agents/list')).toBeInTheDocument();
      expect(screen.getByText('management/agents/get')).toBeInTheDocument();
    });

    it('displays API Key Management category', () => {
      render(<AuditPage />);

      expect(screen.getByText('API Key Management')).toBeInTheDocument();
      expect(screen.getByText('management/api-keys/create')).toBeInTheDocument();
      expect(screen.getByText('management/api-keys/revoke')).toBeInTheDocument();
      expect(screen.getByText('management/api-keys/list')).toBeInTheDocument();
    });

    it('displays Token Management category', () => {
      render(<AuditPage />);

      expect(screen.getByText('Token Management')).toBeInTheDocument();
      expect(screen.getByText('management/tokens/mint')).toBeInTheDocument();
      expect(screen.getByText('management/tokens/revoke')).toBeInTheDocument();
      expect(screen.getByText('management/tokens/revoke-all')).toBeInTheDocument();
    });

    it('displays action descriptions', () => {
      render(<AuditPage />);

      expect(screen.getByText('List available tools from a server')).toBeInTheDocument();
      expect(screen.getByText('Execute a tool')).toBeInTheDocument();
      expect(screen.getByText('Create a new EnforceAI agent')).toBeInTheDocument();
      expect(screen.getByText('Update agent configuration')).toBeInTheDocument();
    });

    it('displays severity badges', () => {
      render(<AuditPage />);

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
    it('links to audit retention documentation', () => {
      render(<AuditPage />);

      expect(
        screen.getByText('enforceai/instructions/ENFORCEAI_AUDIT_RETENTION.md')
      ).toBeInTheDocument();
      expect(screen.getByText(/for retention policy configuration/)).toBeInTheDocument();
    });

    it('links to management guide', () => {
      render(<AuditPage />);

      expect(
        screen.getByText('enforceai/instructions/ENFORCEAI_MANAGEMENT.md')
      ).toBeInTheDocument();
      expect(screen.getByText(/for operational procedures/)).toBeInTheDocument();
    });

    it('links to setup guide', () => {
      render(<AuditPage />);

      expect(screen.getByText('docs/enforceai-setup-guide.md')).toBeInTheDocument();
      expect(screen.getByText(/for initial configuration/)).toBeInTheDocument();
    });
  });

  describe('Code Examples', () => {
    it('shows grep command example with request ID', () => {
      render(<AuditPage />);

      // The grep command should contain "docker-compose logs" and "grep"
      const text = screen.getByText(/docker-compose logs auth_server \| grep/);
      expect(text).toBeInTheDocument();
    });

    it('renders X-Request-Id header mention', () => {
      render(<AuditPage />);

      // X-Request-Id appears in multiple places (code element and note)
      const elements = screen.getAllByText(/X-Request-Id/);
      expect(elements.length).toBeGreaterThan(0);
    });
  });
});
