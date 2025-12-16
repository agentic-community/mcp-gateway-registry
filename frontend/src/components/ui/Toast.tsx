import {
  createContext,
  useContext,
  useState,
  useCallback,
  type ReactNode,
} from 'react';
import { Transition } from '@headlessui/react';
import {
  CheckCircleIcon,
  ExclamationCircleIcon,
  ExclamationTriangleIcon,
  InformationCircleIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { cn } from '../../lib/cn';

export type ToastType = 'success' | 'error' | 'warning' | 'info';

export interface Toast {
  id: string;
  type: ToastType;
  title: string;
  message?: string;
  duration?: number;
}

interface ToastContextValue {
  toasts: Toast[];
  addToast: (toast: Omit<Toast, 'id'>) => string;
  removeToast: (id: string) => void;
  success: (title: string, message?: string) => string;
  error: (title: string, message?: string) => string;
  warning: (title: string, message?: string) => string;
  info: (title: string, message?: string) => string;
}

const ToastContext = createContext<ToastContextValue | undefined>(undefined);

const DEFAULT_DURATION = 5000;

let toastIdCounter = 0;

function generateToastId(): string {
  return `toast-${++toastIdCounter}-${Date.now()}`;
}

interface ToastProviderProps {
  children: ReactNode;
}

export function ToastProvider({ children }: ToastProviderProps) {
  const [toasts, setToasts] = useState<Toast[]>([]);

  const removeToast = useCallback((id: string) => {
    setToasts((prev) => prev.filter((toast) => toast.id !== id));
  }, []);

  const addToast = useCallback(
    (toast: Omit<Toast, 'id'>) => {
      const id = generateToastId();
      const duration = toast.duration ?? DEFAULT_DURATION;

      setToasts((prev) => [...prev, { ...toast, id }]);

      if (duration > 0) {
        setTimeout(() => {
          removeToast(id);
        }, duration);
      }

      return id;
    },
    [removeToast]
  );

  const success = useCallback(
    (title: string, message?: string) =>
      addToast({ type: 'success', title, message }),
    [addToast]
  );

  const error = useCallback(
    (title: string, message?: string) =>
      addToast({ type: 'error', title, message }),
    [addToast]
  );

  const warning = useCallback(
    (title: string, message?: string) =>
      addToast({ type: 'warning', title, message }),
    [addToast]
  );

  const info = useCallback(
    (title: string, message?: string) =>
      addToast({ type: 'info', title, message }),
    [addToast]
  );

  return (
    <ToastContext.Provider
      value={{
        toasts,
        addToast,
        removeToast,
        success,
        error,
        warning,
        info,
      }}
    >
      {children}
      <ToastContainer toasts={toasts} onDismiss={removeToast} />
    </ToastContext.Provider>
  );
}

export function useToast(): ToastContextValue {
  const context = useContext(ToastContext);
  if (context === undefined) {
    throw new Error('useToast must be used within a ToastProvider');
  }
  return context;
}

const iconMap: Record<ToastType, typeof CheckCircleIcon> = {
  success: CheckCircleIcon,
  error: ExclamationCircleIcon,
  warning: ExclamationTriangleIcon,
  info: InformationCircleIcon,
};

const iconColorMap: Record<ToastType, string> = {
  success: 'text-green-400',
  error: 'text-red-400',
  warning: 'text-yellow-400',
  info: 'text-blue-400',
};

interface ToastContainerProps {
  toasts: Toast[];
  onDismiss: (id: string) => void;
}

function ToastContainer({ toasts, onDismiss }: ToastContainerProps) {
  return (
    <div
      aria-live="assertive"
      className="pointer-events-none fixed inset-0 z-50 flex items-end px-4 py-6 sm:items-start sm:p-6"
    >
      <div className="flex w-full flex-col items-center space-y-4 sm:items-end">
        {toasts.map((toast) => (
          <ToastItem key={toast.id} toast={toast} onDismiss={onDismiss} />
        ))}
      </div>
    </div>
  );
}

interface ToastItemProps {
  toast: Toast;
  onDismiss: (id: string) => void;
}

function ToastItem({ toast, onDismiss }: ToastItemProps) {
  const Icon = iconMap[toast.type];

  return (
    <Transition
      show={true}
      enter="transform ease-out duration-300 transition"
      enterFrom="translate-y-2 opacity-0 sm:translate-y-0 sm:translate-x-2"
      enterTo="translate-y-0 opacity-100 sm:translate-x-0"
      leave="transition ease-in duration-100"
      leaveFrom="opacity-100"
      leaveTo="opacity-0"
    >
      <div
        className={cn(
          'pointer-events-auto w-full max-w-sm overflow-hidden rounded-lg shadow-lg ring-1 ring-black ring-opacity-5',
          'bg-white dark:bg-gray-800'
        )}
        role="alert"
      >
        <div className="p-4">
          <div className="flex items-start">
            <div className="flex-shrink-0">
              <Icon
                className={cn('h-6 w-6', iconColorMap[toast.type])}
                aria-hidden="true"
              />
            </div>
            <div className="ml-3 w-0 flex-1 pt-0.5">
              <p className="text-sm font-medium text-gray-900 dark:text-gray-100">
                {toast.title}
              </p>
              {toast.message && (
                <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
                  {toast.message}
                </p>
              )}
            </div>
            <div className="ml-4 flex flex-shrink-0">
              <button
                type="button"
                className={cn(
                  'inline-flex rounded-md',
                  'text-gray-400 hover:text-gray-500',
                  'dark:text-gray-500 dark:hover:text-gray-400',
                  'focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2'
                )}
                onClick={() => onDismiss(toast.id)}
              >
                <span className="sr-only">Close</span>
                <XMarkIcon className="h-5 w-5" aria-hidden="true" />
              </button>
            </div>
          </div>
        </div>
      </div>
    </Transition>
  );
}
