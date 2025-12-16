import { ReactElement, ReactNode } from 'react';
import { render, RenderOptions } from '@testing-library/react';
import { QueryClient, QueryClientProvider } from '@tanstack/react-query';
import { BrowserRouter } from 'react-router-dom';
import userEvent from '@testing-library/user-event';
import { ThemeProvider } from '../contexts/ThemeContext';
import { AuthProvider } from '../contexts/AuthContext';
import { ToastProvider } from '../components/ui/Toast';

// Create a new query client for each test
function createTestQueryClient() {
  return new QueryClient({
    defaultOptions: {
      queries: {
        retry: false,
        gcTime: 0,
        staleTime: 0,
      },
      mutations: {
        retry: false,
      },
    },
  });
}

interface WrapperProps {
  children: ReactNode;
}

interface CustomRenderOptions extends Omit<RenderOptions, 'wrapper'> {
  queryClient?: QueryClient;
  withRouter?: boolean;
  withTheme?: boolean;
  withToast?: boolean;
  withAuth?: boolean;
}

/**
 * Custom render function that wraps components with necessary providers
 */
function customRender(
  ui: ReactElement,
  {
    queryClient = createTestQueryClient(),
    withRouter = true,
    withTheme = true,
    withToast = true,
    withAuth = false,
    ...options
  }: CustomRenderOptions = {}
) {
  function Wrapper({ children }: WrapperProps) {
    let content = (
      <QueryClientProvider client={queryClient}>
        {children}
      </QueryClientProvider>
    );

    if (withAuth) {
      content = <AuthProvider>{content}</AuthProvider>;
    }

    if (withToast) {
      content = <ToastProvider>{content}</ToastProvider>;
    }

    if (withTheme) {
      content = <ThemeProvider defaultTheme="light">{content}</ThemeProvider>;
    }

    if (withRouter) {
      return <BrowserRouter>{content}</BrowserRouter>;
    }

    return content;
  }

  const user = userEvent.setup();

  return {
    ...render(ui, { wrapper: Wrapper, ...options }),
    queryClient,
    user,
  };
}

// Re-export everything from testing-library
export * from '@testing-library/react';
export { userEvent } from '@testing-library/user-event';

// Override render method
export { customRender as render };
