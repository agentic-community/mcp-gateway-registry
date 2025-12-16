import { describe, it, expect, vi } from 'vitest';
import { render, screen, userEvent } from '../../../test/utils';
import { ConfirmDialog } from '../ConfirmDialog';

describe('ConfirmDialog', () => {
  const defaultProps = {
    open: true,
    onClose: vi.fn(),
    onConfirm: vi.fn(),
    title: 'Confirm Action',
    message: 'Are you sure you want to proceed?',
  };

  it('renders when open', () => {
    render(<ConfirmDialog {...defaultProps} />);
    expect(screen.getByText('Confirm Action')).toBeInTheDocument();
    expect(
      screen.getByText('Are you sure you want to proceed?')
    ).toBeInTheDocument();
  });

  it('does not render when closed', () => {
    render(<ConfirmDialog {...defaultProps} open={false} />);
    expect(screen.queryByText('Confirm Action')).not.toBeInTheDocument();
  });

  it('calls onConfirm when confirm button is clicked', async () => {
    const handleConfirm = vi.fn();
    const user = userEvent.setup();

    render(<ConfirmDialog {...defaultProps} onConfirm={handleConfirm} />);
    await user.click(screen.getByRole('button', { name: /confirm/i }));

    expect(handleConfirm).toHaveBeenCalledTimes(1);
  });

  it('calls onClose when cancel button is clicked', async () => {
    const handleClose = vi.fn();
    const user = userEvent.setup();

    render(<ConfirmDialog {...defaultProps} onClose={handleClose} />);
    await user.click(screen.getByRole('button', { name: /cancel/i }));

    expect(handleClose).toHaveBeenCalledTimes(1);
  });

  it('renders custom confirm label', () => {
    render(<ConfirmDialog {...defaultProps} confirmLabel="Delete" />);
    expect(screen.getByRole('button', { name: /delete/i })).toBeInTheDocument();
  });

  it('renders custom cancel label', () => {
    render(<ConfirmDialog {...defaultProps} cancelLabel="Go Back" />);
    expect(screen.getByRole('button', { name: /go back/i })).toBeInTheDocument();
  });

  it('shows loading state on confirm button', () => {
    render(<ConfirmDialog {...defaultProps} loading />);
    const confirmButton = screen.getByRole('button', { name: /confirm/i });
    expect(confirmButton).toBeDisabled();
  });

  it('disables cancel button when loading', () => {
    render(<ConfirmDialog {...defaultProps} loading />);
    const cancelButton = screen.getByRole('button', { name: /cancel/i });
    expect(cancelButton).toBeDisabled();
  });

  it('renders danger variant with red confirm button', () => {
    render(<ConfirmDialog {...defaultProps} variant="danger" />);
    const confirmButton = screen.getByRole('button', { name: /confirm/i });
    expect(confirmButton).toHaveClass('bg-red-600');
  });

  it('renders warning variant', () => {
    render(<ConfirmDialog {...defaultProps} variant="warning" />);
    expect(screen.getByText('Confirm Action')).toBeInTheDocument();
  });

  it('renders info variant with primary confirm button', () => {
    render(<ConfirmDialog {...defaultProps} variant="info" />);
    const confirmButton = screen.getByRole('button', { name: /confirm/i });
    expect(confirmButton).toHaveClass('bg-primary-600');
  });
});
