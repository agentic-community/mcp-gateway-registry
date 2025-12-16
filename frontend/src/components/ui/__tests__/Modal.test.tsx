import { describe, it, expect, vi } from 'vitest';
import { render, screen, userEvent, waitFor } from '../../../test/utils';
import { Modal } from '../Modal';

describe('Modal', () => {
  it('renders when open', () => {
    render(
      <Modal open={true} onClose={() => {}}>
        Modal content
      </Modal>
    );
    expect(screen.getByText('Modal content')).toBeInTheDocument();
  });

  it('does not render when closed', () => {
    render(
      <Modal open={false} onClose={() => {}}>
        Modal content
      </Modal>
    );
    expect(screen.queryByText('Modal content')).not.toBeInTheDocument();
  });

  it('renders with title', () => {
    render(
      <Modal open={true} onClose={() => {}} title="Test Title">
        Content
      </Modal>
    );
    expect(screen.getByText('Test Title')).toBeInTheDocument();
  });

  it('renders with description', () => {
    render(
      <Modal
        open={true}
        onClose={() => {}}
        title="Title"
        description="Test description"
      >
        Content
      </Modal>
    );
    expect(screen.getByText('Test description')).toBeInTheDocument();
  });

  it('calls onClose when close button is clicked', async () => {
    const handleClose = vi.fn();
    const user = userEvent.setup();

    render(
      <Modal open={true} onClose={handleClose} title="Title">
        Content
      </Modal>
    );

    const closeButton = screen.getByRole('button', { name: /close/i });
    await user.click(closeButton);

    expect(handleClose).toHaveBeenCalled();
  });

  it('calls onClose when clicking outside modal', async () => {
    const handleClose = vi.fn();
    const user = userEvent.setup();

    render(
      <Modal open={true} onClose={handleClose}>
        Content
      </Modal>
    );

    // Click on the backdrop
    const backdrop = document.querySelector('[aria-hidden="true"]');
    if (backdrop) {
      await user.click(backdrop);
    }

    await waitFor(() => {
      expect(handleClose).toHaveBeenCalled();
    });
  });

  it('hides close button when showCloseButton is false', () => {
    render(
      <Modal open={true} onClose={() => {}} showCloseButton={false}>
        Content
      </Modal>
    );
    expect(screen.queryByRole('button', { name: /close/i })).not.toBeInTheDocument();
  });

  it('renders with different sizes', () => {
    const { rerender } = render(
      <Modal open={true} onClose={() => {}} size="sm">
        Content
      </Modal>
    );
    expect(screen.getByText('Content').closest('[class*="max-w-"]')).toHaveClass('max-w-md');

    rerender(
      <Modal open={true} onClose={() => {}} size="lg">
        Content
      </Modal>
    );
    expect(screen.getByText('Content').closest('[class*="max-w-"]')).toHaveClass('max-w-2xl');
  });

  it('applies custom className', () => {
    render(
      <Modal open={true} onClose={() => {}} className="custom-class">
        Content
      </Modal>
    );
    // Find the dialog panel which contains the custom class
    const panel = screen.getByText('Content').closest('[class*="rounded-lg"]');
    expect(panel).toHaveClass('custom-class');
  });
});
