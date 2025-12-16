import { describe, it, expect, vi } from 'vitest';
import { render, screen, userEvent, waitFor } from '../../../test/utils';
import { ToastProvider, useToast } from '../Toast';

function TestComponent() {
  const { success, error, warning, info, toasts, removeToast } = useToast();

  return (
    <div>
      <button onClick={() => success('Success message')}>Show Success</button>
      <button onClick={() => error('Error message')}>Show Error</button>
      <button onClick={() => warning('Warning message')}>Show Warning</button>
      <button onClick={() => info('Info message')}>Show Info</button>
      <button onClick={() => success('With description', 'This is a description')}>
        With Description
      </button>
      <div data-testid="toast-count">{toasts.length}</div>
      {toasts.map((toast) => (
        <button
          key={toast.id}
          onClick={() => removeToast(toast.id)}
          data-testid={`dismiss-${toast.id}`}
        >
          Dismiss {toast.id}
        </button>
      ))}
    </div>
  );
}

describe('Toast', () => {
  it('shows success toast when success is called', async () => {
    const user = userEvent.setup();

    render(
      <ToastProvider>
        <TestComponent />
      </ToastProvider>,
      { withToast: false }
    );

    await user.click(screen.getByText('Show Success'));

    await waitFor(() => {
      expect(screen.getByText('Success message')).toBeInTheDocument();
    });
  });

  it('shows error toast when error is called', async () => {
    const user = userEvent.setup();

    render(
      <ToastProvider>
        <TestComponent />
      </ToastProvider>,
      { withToast: false }
    );

    await user.click(screen.getByText('Show Error'));

    await waitFor(() => {
      expect(screen.getByText('Error message')).toBeInTheDocument();
    });
  });

  it('shows warning toast when warning is called', async () => {
    const user = userEvent.setup();

    render(
      <ToastProvider>
        <TestComponent />
      </ToastProvider>,
      { withToast: false }
    );

    await user.click(screen.getByText('Show Warning'));

    await waitFor(() => {
      expect(screen.getByText('Warning message')).toBeInTheDocument();
    });
  });

  it('shows info toast when info is called', async () => {
    const user = userEvent.setup();

    render(
      <ToastProvider>
        <TestComponent />
      </ToastProvider>,
      { withToast: false }
    );

    await user.click(screen.getByText('Show Info'));

    await waitFor(() => {
      expect(screen.getByText('Info message')).toBeInTheDocument();
    });
  });

  it('shows toast with description', async () => {
    const user = userEvent.setup();

    render(
      <ToastProvider>
        <TestComponent />
      </ToastProvider>,
      { withToast: false }
    );

    await user.click(screen.getByText('With Description'));

    await waitFor(() => {
      expect(screen.getByText('With description')).toBeInTheDocument();
      expect(screen.getByText('This is a description')).toBeInTheDocument();
    });
  });

  it('dismisses toast when dismiss button is clicked', async () => {
    const user = userEvent.setup();

    render(
      <ToastProvider>
        <TestComponent />
      </ToastProvider>,
      { withToast: false }
    );

    await user.click(screen.getByText('Show Success'));

    await waitFor(() => {
      expect(screen.getByText('Success message')).toBeInTheDocument();
    });

    // Click the close button on the toast
    const closeButtons = screen.getAllByRole('button', { name: /close/i });
    await user.click(closeButtons[0]);

    await waitFor(() => {
      expect(screen.queryByText('Success message')).not.toBeInTheDocument();
    });
  });

  it('can show and track multiple toasts', async () => {
    const user = userEvent.setup();

    render(
      <ToastProvider>
        <TestComponent />
      </ToastProvider>,
      { withToast: false }
    );

    await user.click(screen.getByText('Show Success'));

    await waitFor(() => {
      expect(screen.getByTestId('toast-count')).toHaveTextContent('1');
    });

    await user.click(screen.getByText('Show Error'));

    await waitFor(() => {
      expect(screen.getByTestId('toast-count')).toHaveTextContent('2');
    });
  });

  it('throws error when useToast is used outside ToastProvider', () => {
    const consoleError = vi.spyOn(console, 'error').mockImplementation(() => {});

    expect(() => {
      render(<TestComponent />, { withToast: false, withTheme: false });
    }).toThrow('useToast must be used within a ToastProvider');

    consoleError.mockRestore();
  });
});
