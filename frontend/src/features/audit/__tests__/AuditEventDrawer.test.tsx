import { describe, it, expect, vi } from 'vitest';
import { screen } from '@testing-library/react';
import userEvent from '@testing-library/user-event';
import { render } from '@/test/utils';
import { AuditEventDrawer } from '../AuditEventDrawer';
import type { AuditEvent } from '@/api/types';

describe('AuditEventDrawer', () => {
  const mockEvent: AuditEvent = {
    event_id: 123,
    occurred_at: '2024-01-15T10:30:00.000Z',
    user_id: 'user-123',
    agent_id: '9d2724e9-1753-4493-8993-0d69867544',
    action: 'tools/call',
    outcome: 'allow',
    request_id: 'req-abc123',
    details: {
      server: 'sqlite',
      tool: 'read_query',
      args: { query: 'SELECT * FROM users' },
    },
  };

  const defaultProps = {
    event: mockEvent,
    isOpen: true,
    onClose: vi.fn(),
  };

  describe('Rendering', () => {
    it('renders nothing when event is null', () => {
      render(<AuditEventDrawer {...defaultProps} event={null} />);
      // Drawer content should not be rendered
      expect(screen.queryByText('Audit Event Details')).not.toBeInTheDocument();
    });

    it('renders drawer title when open', () => {
      render(<AuditEventDrawer {...defaultProps} />);
      expect(screen.getByText('Audit Event Details')).toBeInTheDocument();
    });

    it('renders outcome badge for allowed event', () => {
      render(<AuditEventDrawer {...defaultProps} />);
      expect(screen.getByText('Allowed')).toBeInTheDocument();
    });

    it('renders outcome badge for denied event', () => {
      const deniedEvent = { ...mockEvent, outcome: 'deny' };
      render(<AuditEventDrawer {...defaultProps} event={deniedEvent} />);
      expect(screen.getByText('Denied')).toBeInTheDocument();
    });
  });

  describe('Event Details', () => {
    it('displays event ID', () => {
      render(<AuditEventDrawer {...defaultProps} />);
      expect(screen.getByText('Event ID')).toBeInTheDocument();
      expect(screen.getByText('123')).toBeInTheDocument();
    });

    it('displays action', () => {
      render(<AuditEventDrawer {...defaultProps} />);
      expect(screen.getByText('Action')).toBeInTheDocument();
      expect(screen.getByText('tools/call')).toBeInTheDocument();
    });

    it('displays user ID', () => {
      render(<AuditEventDrawer {...defaultProps} />);
      expect(screen.getByText('User ID')).toBeInTheDocument();
      expect(screen.getByText('user-123')).toBeInTheDocument();
    });

    it('displays agent ID', () => {
      render(<AuditEventDrawer {...defaultProps} />);
      expect(screen.getByText('Agent ID')).toBeInTheDocument();
      expect(screen.getByText('9d2724e9-1753-4493-8993-0d69867544')).toBeInTheDocument();
    });

    it('displays request ID when present', () => {
      render(<AuditEventDrawer {...defaultProps} />);
      expect(screen.getByText('Request ID')).toBeInTheDocument();
      expect(screen.getByText('req-abc123')).toBeInTheDocument();
    });

    it('does not display request ID row when null', () => {
      const eventWithoutRequestId = { ...mockEvent, request_id: null };
      render(<AuditEventDrawer {...defaultProps} event={eventWithoutRequestId} />);
      expect(screen.queryByText('Request ID')).not.toBeInTheDocument();
    });
  });

  describe('Details JSON', () => {
    it('renders details section when details exist', () => {
      render(<AuditEventDrawer {...defaultProps} />);
      expect(screen.getByText('Details')).toBeInTheDocument();
    });

    it('renders details as JSON', () => {
      render(<AuditEventDrawer {...defaultProps} />);
      // Should contain parts of the JSON
      expect(screen.getByText(/"server": "sqlite"/)).toBeInTheDocument();
    });

    it('shows empty state when no details', () => {
      const eventWithoutDetails = { ...mockEvent, details: null };
      render(<AuditEventDrawer {...defaultProps} event={eventWithoutDetails} />);
      expect(screen.getByText('No additional details available')).toBeInTheDocument();
    });
  });

  describe('Close Behavior', () => {
    it('calls onClose when X button is clicked', async () => {
      const onClose = vi.fn();
      const user = userEvent.setup();
      render(<AuditEventDrawer {...defaultProps} onClose={onClose} />);

      // Find the X button (it's the first button in the header)
      const closeButton = screen.getAllByRole('button')[0];
      await user.click(closeButton);

      expect(onClose).toHaveBeenCalled();
    });

    it('calls onClose when Close button is clicked', async () => {
      const onClose = vi.fn();
      const user = userEvent.setup();
      render(<AuditEventDrawer {...defaultProps} onClose={onClose} />);

      await user.click(screen.getByRole('button', { name: 'Close' }));

      expect(onClose).toHaveBeenCalled();
    });
  });

  describe('Copy Buttons', () => {
    it('has copy button for event ID', () => {
      render(<AuditEventDrawer {...defaultProps} />);
      // Event ID row has a copy button
      const eventIdLabel = screen.getByText('Event ID');
      const eventIdRow = eventIdLabel.closest('div.py-3');
      expect(eventIdRow?.querySelector('button')).toBeInTheDocument();
    });

    it('has copy button for agent ID', () => {
      render(<AuditEventDrawer {...defaultProps} />);
      const agentIdLabel = screen.getByText('Agent ID');
      const agentIdRow = agentIdLabel.closest('div.py-3');
      expect(agentIdRow?.querySelector('button')).toBeInTheDocument();
    });

    it('has copy button for details JSON', () => {
      render(<AuditEventDrawer {...defaultProps} />);
      const detailsLabel = screen.getByText('Details');
      const detailsSection = detailsLabel.closest('div.mt-6');
      expect(detailsSection?.querySelector('button')).toBeInTheDocument();
    });
  });

  describe('Formatted Date', () => {
    it('displays formatted occurred at date', () => {
      render(<AuditEventDrawer {...defaultProps} />);
      expect(screen.getByText('Occurred At')).toBeInTheDocument();
      // The formatted date should be visible (format depends on locale)
      expect(screen.getAllByText(/2024/).length).toBeGreaterThan(0);
    });
  });

  describe('Correlation View', () => {
    it('shows View All Events button when request ID exists and callback provided', () => {
      const onFilterByRequestId = vi.fn();
      render(
        <AuditEventDrawer {...defaultProps} onFilterByRequestId={onFilterByRequestId} />
      );
      expect(screen.getByText('View All Events for This Request')).toBeInTheDocument();
    });

    it('does not show View All Events button when request ID is null', () => {
      const onFilterByRequestId = vi.fn();
      const eventWithoutRequestId = { ...mockEvent, request_id: null };
      render(
        <AuditEventDrawer
          {...defaultProps}
          event={eventWithoutRequestId}
          onFilterByRequestId={onFilterByRequestId}
        />
      );
      expect(screen.queryByText('View All Events for This Request')).not.toBeInTheDocument();
    });

    it('does not show View All Events button when callback not provided', () => {
      render(<AuditEventDrawer {...defaultProps} />);
      expect(screen.queryByText('View All Events for This Request')).not.toBeInTheDocument();
    });

    it('calls onFilterByRequestId and onClose when View All Events is clicked', async () => {
      const onFilterByRequestId = vi.fn();
      const onClose = vi.fn();
      const user = userEvent.setup();
      render(
        <AuditEventDrawer
          {...defaultProps}
          onFilterByRequestId={onFilterByRequestId}
          onClose={onClose}
        />
      );

      await user.click(screen.getByText('View All Events for This Request'));

      expect(onFilterByRequestId).toHaveBeenCalledWith('req-abc123');
      expect(onClose).toHaveBeenCalled();
    });
  });
});
