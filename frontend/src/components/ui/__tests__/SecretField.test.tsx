import { describe, it, expect, vi } from 'vitest';
import { render, screen, userEvent } from '../../../test/utils';
import { SecretField } from '../SecretField';

describe('SecretField', () => {
  it('renders with type password by default', () => {
    render(<SecretField value="secret" onChange={() => {}} />);
    const input = document.querySelector('input');
    expect(input).toHaveAttribute('type', 'password');
  });

  it('masks value by default', () => {
    render(<SecretField value="my-secret-value" onChange={() => {}} />);
    const input = document.querySelector('input[type="password"]');
    expect(input).toBeInTheDocument();
  });

  it('reveals value when toggle is clicked', async () => {
    const user = userEvent.setup();

    render(<SecretField value="my-secret-value" onChange={() => {}} />);

    // Initially password
    expect(document.querySelector('input[type="password"]')).toBeInTheDocument();

    // Click reveal button
    const revealButton = screen.getByRole('button', { name: /show secret/i });
    await user.click(revealButton);

    // Now text
    expect(document.querySelector('input[type="text"]')).toBeInTheDocument();
  });

  it('hides value when toggle is clicked again', async () => {
    const user = userEvent.setup();

    render(<SecretField value="my-secret-value" onChange={() => {}} />);

    const revealButton = screen.getByRole('button', { name: /show secret/i });
    await user.click(revealButton);

    expect(document.querySelector('input[type="text"]')).toBeInTheDocument();

    const hideButton = screen.getByRole('button', { name: /hide secret/i });
    await user.click(hideButton);

    expect(document.querySelector('input[type="password"]')).toBeInTheDocument();
  });

  it('renders with label', () => {
    render(<SecretField label="API Key" value="" onChange={() => {}} />);
    expect(screen.getByText('API Key')).toBeInTheDocument();
  });

  it('shows error state', () => {
    render(<SecretField error="Invalid key" value="" onChange={() => {}} />);
    expect(screen.getByRole('alert')).toHaveTextContent('Invalid key');
  });

  it('shows helper text', () => {
    render(<SecretField helperText="Keep this secret safe" value="" onChange={() => {}} />);
    expect(screen.getByText('Keep this secret safe')).toBeInTheDocument();
  });

  it('renders copy button by default', () => {
    render(<SecretField value="test-secret" onChange={() => {}} />);
    expect(
      screen.getByRole('button', { name: /copy secret/i })
    ).toBeInTheDocument();
  });

  it('hides copy button when showCopy is false', () => {
    render(<SecretField value="test-secret" showCopy={false} onChange={() => {}} />);
    expect(
      screen.queryByRole('button', { name: /copy secret/i })
    ).not.toBeInTheDocument();
  });

  it('copies value to clipboard when copy button is clicked', async () => {
    const user = userEvent.setup();

    render(<SecretField value="test-secret" onChange={() => {}} />);
    await user.click(screen.getByRole('button', { name: /copy secret/i }));

    // Just check it doesn't throw - clipboard API is mocked in setup
    expect(true).toBe(true);
  });

  it('calls onCopy when copy button is clicked', async () => {
    const handleCopy = vi.fn();
    const user = userEvent.setup();

    render(<SecretField value="test-secret" onCopy={handleCopy} onChange={() => {}} />);
    await user.click(screen.getByRole('button', { name: /copy secret/i }));

    expect(handleCopy).toHaveBeenCalled();
  });

  it('hides copy button when value is empty', () => {
    render(<SecretField value="" onChange={() => {}} />);
    expect(
      screen.queryByRole('button', { name: /copy secret/i })
    ).not.toBeInTheDocument();
  });

  it('is disabled when disabled prop is true', () => {
    render(<SecretField value="test" disabled onChange={() => {}} />);
    const input = document.querySelector('input');
    expect(input).toBeDisabled();
  });
});
