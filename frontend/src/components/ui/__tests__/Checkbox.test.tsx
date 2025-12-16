import { describe, it, expect, vi } from 'vitest';
import { render, screen, userEvent } from '../../../test/utils';
import { Checkbox } from '../Checkbox';

describe('Checkbox', () => {
  it('renders checkbox input', () => {
    render(<Checkbox />);
    expect(screen.getByRole('checkbox')).toBeInTheDocument();
  });

  it('renders with label', () => {
    render(<Checkbox label="Accept terms" />);
    expect(screen.getByLabelText('Accept terms')).toBeInTheDocument();
  });

  it('renders with description', () => {
    render(
      <Checkbox
        label="Newsletter"
        description="Receive weekly updates"
      />
    );
    expect(screen.getByText('Receive weekly updates')).toBeInTheDocument();
  });

  it('toggles checked state when clicked', async () => {
    const user = userEvent.setup();

    render(<Checkbox label="Test" />);
    const checkbox = screen.getByRole('checkbox');

    expect(checkbox).not.toBeChecked();
    await user.click(checkbox);
    expect(checkbox).toBeChecked();
    await user.click(checkbox);
    expect(checkbox).not.toBeChecked();
  });

  it('calls onChange when checked state changes', async () => {
    const handleChange = vi.fn();
    const user = userEvent.setup();

    render(<Checkbox onChange={handleChange} />);
    await user.click(screen.getByRole('checkbox'));

    expect(handleChange).toHaveBeenCalledTimes(1);
  });

  it('is disabled when disabled prop is true', () => {
    render(<Checkbox disabled />);
    expect(screen.getByRole('checkbox')).toBeDisabled();
  });

  it('can be controlled with checked prop', () => {
    const { rerender } = render(<Checkbox checked={false} onChange={() => {}} />);
    expect(screen.getByRole('checkbox')).not.toBeChecked();

    rerender(<Checkbox checked={true} onChange={() => {}} />);
    expect(screen.getByRole('checkbox')).toBeChecked();
  });

  it('applies custom className', () => {
    render(<Checkbox className="custom-class" />);
    expect(screen.getByRole('checkbox')).toHaveClass('custom-class');
  });
});
