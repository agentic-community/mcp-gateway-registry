import { describe, it, expect, vi } from 'vitest';
import { render, screen, userEvent } from '../../../test/utils';
import { Textarea } from '../Textarea';

describe('Textarea', () => {
  it('renders with default props', () => {
    render(<Textarea />);
    expect(screen.getByRole('textbox')).toBeInTheDocument();
  });

  it('renders with label', () => {
    render(<Textarea label="Description" />);
    expect(screen.getByLabelText('Description')).toBeInTheDocument();
  });

  it('shows error state', () => {
    render(<Textarea error="This field is required" />);
    const textarea = screen.getByRole('textbox');
    const error = screen.getByRole('alert');

    expect(textarea).toHaveAttribute('aria-invalid', 'true');
    expect(error).toHaveTextContent('This field is required');
  });

  it('shows helper text', () => {
    render(<Textarea helperText="Enter a detailed description" />);
    expect(screen.getByText('Enter a detailed description')).toBeInTheDocument();
  });

  it('hides helper text when error is present', () => {
    render(
      <Textarea helperText="Helper" error="Error" />
    );
    expect(screen.queryByText('Helper')).not.toBeInTheDocument();
    expect(screen.getByText('Error')).toBeInTheDocument();
  });

  it('calls onChange when value changes', async () => {
    const handleChange = vi.fn();
    const user = userEvent.setup();

    render(<Textarea onChange={handleChange} />);
    await user.type(screen.getByRole('textbox'), 'test');

    expect(handleChange).toHaveBeenCalledTimes(4);
  });

  it('renders with default rows', () => {
    render(<Textarea />);
    const textarea = screen.getByRole('textbox');
    expect(textarea).toHaveAttribute('rows', '4');
  });

  it('renders with custom rows', () => {
    render(<Textarea rows={10} />);
    const textarea = screen.getByRole('textbox');
    expect(textarea).toHaveAttribute('rows', '10');
  });

  it('is disabled when disabled prop is true', () => {
    render(<Textarea disabled />);
    expect(screen.getByRole('textbox')).toBeDisabled();
  });

  it('renders with placeholder', () => {
    render(<Textarea placeholder="Enter text here" />);
    expect(screen.getByPlaceholderText('Enter text here')).toBeInTheDocument();
  });

  it('applies custom className', () => {
    render(<Textarea className="custom-class" />);
    expect(screen.getByRole('textbox')).toHaveClass('custom-class');
  });
});
