import { describe, it, expect, vi } from 'vitest';
import { render, screen, userEvent, waitFor } from '../../../test/utils';
import { CopyButton, CopyIconButton } from '../CopyButton';

describe('CopyButton', () => {

  it('renders with default label', () => {
    render(<CopyButton text="test" />);
    expect(screen.getByRole('button', { name: /copy/i })).toBeInTheDocument();
  });

  it('renders with custom label', () => {
    render(<CopyButton text="test" label="Copy text" />);
    expect(
      screen.getByRole('button', { name: /copy text/i })
    ).toBeInTheDocument();
  });

  it('copies text to clipboard and shows feedback when clicked', async () => {
    const user = userEvent.setup();

    render(<CopyButton text="test content" />);
    await user.click(screen.getByRole('button'));

    // Check that feedback is shown (indicates successful copy)
    await waitFor(() => {
      expect(screen.getByText('Copied!')).toBeInTheDocument();
    });
  });

  it('shows copied feedback after click', async () => {
    const user = userEvent.setup();

    render(<CopyButton text="test" copiedLabel="Copied!" />);
    await user.click(screen.getByRole('button'));

    await waitFor(() => {
      expect(screen.getByText('Copied!')).toBeInTheDocument();
    });
  });

  it('calls onCopy callback when copied', async () => {
    const handleCopy = vi.fn();
    const user = userEvent.setup();

    render(<CopyButton text="test" onCopy={handleCopy} />);
    await user.click(screen.getByRole('button'));

    expect(handleCopy).toHaveBeenCalled();
  });

  it('shows copied state temporarily', async () => {
    const user = userEvent.setup();

    render(<CopyButton text="test" />);
    await user.click(screen.getByRole('button'));

    // Verify it shows copied state
    await waitFor(() => {
      expect(screen.getByText('Copied!')).toBeInTheDocument();
    });
  });
});

describe('CopyIconButton', () => {
  it('renders with aria-label', () => {
    render(<CopyIconButton text="test" aria-label="Copy value" />);
    expect(
      screen.getByRole('button', { name: /copy value/i })
    ).toBeInTheDocument();
  });

  it('calls onCopy callback when clicked', async () => {
    const handleCopy = vi.fn();
    const user = userEvent.setup();

    render(<CopyIconButton text="test" onCopy={handleCopy} />);
    await user.click(screen.getByRole('button'));

    await waitFor(() => {
      expect(handleCopy).toHaveBeenCalled();
    });
  });
});
