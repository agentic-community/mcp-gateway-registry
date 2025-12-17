import { describe, it, expect, vi, beforeEach } from 'vitest';
import { screen } from '@testing-library/react';
import { render } from '@/test/utils';
import { TypeToConfirmDialog } from '../TypeToConfirmDialog';

describe('TypeToConfirmDialog', () => {
  const mockOnClose = vi.fn();
  const mockOnConfirm = vi.fn();

  beforeEach(() => {
    vi.clearAllMocks();
  });

  describe('Basic Rendering', () => {
    it('renders title and message', () => {
      render(
        <TypeToConfirmDialog
          open={true}
          onClose={mockOnClose}
          onConfirm={mockOnConfirm}
          title="Confirm Action"
          message="Are you sure you want to proceed?"
          confirmValue="confirm"
        />
      );

      expect(screen.getByText('Confirm Action')).toBeInTheDocument();
      expect(screen.getByText('Are you sure you want to proceed?')).toBeInTheDocument();
    });

    it('shows confirm value instruction', () => {
      render(
        <TypeToConfirmDialog
          open={true}
          onClose={mockOnClose}
          onConfirm={mockOnConfirm}
          title="Delete Item"
          message="This action cannot be undone."
          confirmValue="delete-me"
        />
      );

      expect(screen.getByText('delete-me')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('delete-me')).toBeInTheDocument();
    });

    it('renders confirm and cancel buttons', () => {
      render(
        <TypeToConfirmDialog
          open={true}
          onClose={mockOnClose}
          onConfirm={mockOnConfirm}
          title="Test"
          message="Test message"
          confirmValue="test"
        />
      );

      expect(screen.getByRole('button', { name: 'Confirm' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Cancel' })).toBeInTheDocument();
    });

    it('uses custom button labels', () => {
      render(
        <TypeToConfirmDialog
          open={true}
          onClose={mockOnClose}
          onConfirm={mockOnConfirm}
          title="Test"
          message="Test message"
          confirmValue="test"
          confirmLabel="Delete"
          cancelLabel="Go Back"
        />
      );

      expect(screen.getByRole('button', { name: 'Delete' })).toBeInTheDocument();
      expect(screen.getByRole('button', { name: 'Go Back' })).toBeInTheDocument();
    });
  });

  describe('Confirmation Flow', () => {
    it('confirm button is disabled initially', () => {
      render(
        <TypeToConfirmDialog
          open={true}
          onClose={mockOnClose}
          onConfirm={mockOnConfirm}
          title="Test"
          message="Test message"
          confirmValue="confirm"
        />
      );

      expect(screen.getByRole('button', { name: 'Confirm' })).toBeDisabled();
    });

    it('enables confirm button when correct value is typed', async () => {
      const { user } = render(
        <TypeToConfirmDialog
          open={true}
          onClose={mockOnClose}
          onConfirm={mockOnConfirm}
          title="Test"
          message="Test message"
          confirmValue="confirm"
        />
      );

      const input = screen.getByPlaceholderText('confirm');
      await user.type(input, 'confirm');

      expect(screen.getByRole('button', { name: 'Confirm' })).not.toBeDisabled();
    });

    it('keeps confirm button disabled for incorrect value', async () => {
      const { user } = render(
        <TypeToConfirmDialog
          open={true}
          onClose={mockOnClose}
          onConfirm={mockOnConfirm}
          title="Test"
          message="Test message"
          confirmValue="confirm"
        />
      );

      const input = screen.getByPlaceholderText('confirm');
      await user.type(input, 'wrong');

      expect(screen.getByRole('button', { name: 'Confirm' })).toBeDisabled();
    });

    it('calls onConfirm when confirm button is clicked', async () => {
      const { user } = render(
        <TypeToConfirmDialog
          open={true}
          onClose={mockOnClose}
          onConfirm={mockOnConfirm}
          title="Test"
          message="Test message"
          confirmValue="confirm"
        />
      );

      const input = screen.getByPlaceholderText('confirm');
      await user.type(input, 'confirm');
      await user.click(screen.getByRole('button', { name: 'Confirm' }));

      expect(mockOnConfirm).toHaveBeenCalled();
    });

    it('calls onClose when cancel button is clicked', async () => {
      const { user } = render(
        <TypeToConfirmDialog
          open={true}
          onClose={mockOnClose}
          onConfirm={mockOnConfirm}
          title="Test"
          message="Test message"
          confirmValue="confirm"
        />
      );

      await user.click(screen.getByRole('button', { name: 'Cancel' }));

      expect(mockOnClose).toHaveBeenCalled();
    });
  });

  describe('Reason Field', () => {
    it('shows reason field when requireReason is true', () => {
      render(
        <TypeToConfirmDialog
          open={true}
          onClose={mockOnClose}
          onConfirm={mockOnConfirm}
          title="Test"
          message="Test message"
          confirmValue="confirm"
          requireReason
        />
      );

      expect(screen.getByText('Reason (required)')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Enter reason for this action...')).toBeInTheDocument();
    });

    it('confirm button disabled without reason when required', async () => {
      const { user } = render(
        <TypeToConfirmDialog
          open={true}
          onClose={mockOnClose}
          onConfirm={mockOnConfirm}
          title="Test"
          message="Test message"
          confirmValue="confirm"
          requireReason
        />
      );

      // Type correct confirm value
      const input = screen.getByPlaceholderText('confirm');
      await user.type(input, 'confirm');

      // Button should still be disabled without reason
      expect(screen.getByRole('button', { name: 'Confirm' })).toBeDisabled();
    });

    it('enables confirm when both value and reason are provided', async () => {
      const { user } = render(
        <TypeToConfirmDialog
          open={true}
          onClose={mockOnClose}
          onConfirm={mockOnConfirm}
          title="Test"
          message="Test message"
          confirmValue="confirm"
          requireReason
        />
      );

      // Type correct confirm value
      const confirmInput = screen.getByPlaceholderText('confirm');
      await user.type(confirmInput, 'confirm');

      // Type reason
      const reasonInput = screen.getByPlaceholderText('Enter reason for this action...');
      await user.type(reasonInput, 'Security concern');

      expect(screen.getByRole('button', { name: 'Confirm' })).not.toBeDisabled();
    });

    it('passes reason to onConfirm when required', async () => {
      const { user } = render(
        <TypeToConfirmDialog
          open={true}
          onClose={mockOnClose}
          onConfirm={mockOnConfirm}
          title="Test"
          message="Test message"
          confirmValue="confirm"
          requireReason
        />
      );

      const confirmInput = screen.getByPlaceholderText('confirm');
      await user.type(confirmInput, 'confirm');

      const reasonInput = screen.getByPlaceholderText('Enter reason for this action...');
      await user.type(reasonInput, 'Security concern');

      await user.click(screen.getByRole('button', { name: 'Confirm' }));

      expect(mockOnConfirm).toHaveBeenCalledWith('Security concern');
    });

    it('uses custom reason label and placeholder', () => {
      render(
        <TypeToConfirmDialog
          open={true}
          onClose={mockOnClose}
          onConfirm={mockOnConfirm}
          title="Test"
          message="Test message"
          confirmValue="confirm"
          requireReason
          reasonLabel="Justification"
          reasonPlaceholder="Explain why..."
        />
      );

      expect(screen.getByText('Justification')).toBeInTheDocument();
      expect(screen.getByPlaceholderText('Explain why...')).toBeInTheDocument();
    });
  });

  describe('Loading State', () => {
    it('shows loading state on confirm button', () => {
      render(
        <TypeToConfirmDialog
          open={true}
          onClose={mockOnClose}
          onConfirm={mockOnConfirm}
          title="Test"
          message="Test message"
          confirmValue="confirm"
          loading
        />
      );

      // Cancel should be disabled during loading
      expect(screen.getByRole('button', { name: 'Cancel' })).toBeDisabled();
    });
  });

  describe('Dialog State', () => {
    it('does not render when closed', () => {
      render(
        <TypeToConfirmDialog
          open={false}
          onClose={mockOnClose}
          onConfirm={mockOnConfirm}
          title="Test"
          message="Test message"
          confirmValue="confirm"
        />
      );

      expect(screen.queryByText('Test')).not.toBeInTheDocument();
    });
  });
});
