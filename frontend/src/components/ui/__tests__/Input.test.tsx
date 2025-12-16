import { describe, it, expect, vi } from 'vitest';
import { render, screen, userEvent } from '../../../test/utils';
import { Input } from '../Input';

describe('Input', () => {
  it('renders with default props', () => {
    render(<Input />);
    expect(screen.getByRole('textbox')).toBeInTheDocument();
  });

  it('renders with label', () => {
    render(<Input label="Email" />);
    expect(screen.getByLabelText('Email')).toBeInTheDocument();
  });

  it('shows error state', () => {
    render(<Input error="This field is required" />);
    const input = screen.getByRole('textbox');
    const error = screen.getByRole('alert');

    expect(input).toHaveAttribute('aria-invalid', 'true');
    expect(error).toHaveTextContent('This field is required');
  });

  it('shows helper text', () => {
    render(<Input helperText="Enter your email address" />);
    expect(screen.getByText('Enter your email address')).toBeInTheDocument();
  });

  it('hides helper text when error is present', () => {
    render(
      <Input helperText="Enter your email address" error="Invalid email" />
    );
    expect(
      screen.queryByText('Enter your email address')
    ).not.toBeInTheDocument();
    expect(screen.getByText('Invalid email')).toBeInTheDocument();
  });

  it('calls onChange when value changes', async () => {
    const handleChange = vi.fn();
    const user = userEvent.setup();

    render(<Input onChange={handleChange} />);
    await user.type(screen.getByRole('textbox'), 'test');

    expect(handleChange).toHaveBeenCalledTimes(4);
  });

  it('renders with left addon', () => {
    render(<Input leftAddon="$" />);
    expect(screen.getByText('$')).toBeInTheDocument();
  });

  it('renders with right addon', () => {
    render(<Input rightAddon=".00" />);
    expect(screen.getByText('.00')).toBeInTheDocument();
  });

  it('is disabled when disabled prop is true', () => {
    render(<Input disabled />);
    expect(screen.getByRole('textbox')).toBeDisabled();
  });

  it('renders with placeholder', () => {
    render(<Input placeholder="Enter text" />);
    expect(screen.getByPlaceholderText('Enter text')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    render(<Input className="custom-class" />);
    expect(screen.getByRole('textbox')).toHaveClass('custom-class');
  });

  it('has proper aria-describedby for error', () => {
    render(<Input id="test-input" error="Error message" />);
    const input = screen.getByRole('textbox');
    expect(input).toHaveAttribute('aria-describedby', 'test-input-error');
  });

  it('has proper aria-describedby for helper text', () => {
    render(<Input id="test-input" helperText="Helper text" />);
    const input = screen.getByRole('textbox');
    expect(input).toHaveAttribute('aria-describedby', 'test-input-helper');
  });
});
