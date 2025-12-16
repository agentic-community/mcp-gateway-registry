import { forwardRef, type InputHTMLAttributes, type ReactNode, useId } from 'react';
import { cn } from '../../lib/cn';

export interface InputProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'size'> {
  label?: string;
  error?: string;
  helperText?: string;
  leftAddon?: ReactNode;
  rightAddon?: ReactNode;
}

export const Input = forwardRef<HTMLInputElement, InputProps>(
  (
    {
      className,
      label,
      error,
      helperText,
      leftAddon,
      rightAddon,
      id: providedId,
      ...props
    },
    ref
  ) => {
    const generatedId = useId();
    const id = providedId || generatedId;
    const hasError = Boolean(error);
    const describedBy = error ? `${id}-error` : helperText ? `${id}-helper` : undefined;

    return (
      <div className="w-full">
        {label && (
          <label
            htmlFor={id}
            className={cn(
              'block text-sm font-medium mb-1',
              'text-gray-700 dark:text-gray-300'
            )}
          >
            {label}
          </label>
        )}
        <div className="relative">
          {leftAddon && (
            <div className="absolute inset-y-0 left-0 pl-3 flex items-center pointer-events-none">
              <span className="text-gray-500 dark:text-gray-400 sm:text-sm">
                {leftAddon}
              </span>
            </div>
          )}
          <input
            ref={ref}
            id={id}
            className={cn(
              'block w-full rounded-md shadow-sm',
              'text-sm',
              'transition-colors duration-150',
              'focus:outline-none focus:ring-2 focus:ring-offset-0',
              'disabled:bg-gray-100 disabled:cursor-not-allowed',
              'dark:disabled:bg-gray-800',
              leftAddon && 'pl-10',
              rightAddon && 'pr-10',
              hasError
                ? cn(
                    'border-red-300 text-red-900 placeholder-red-300',
                    'focus:border-red-500 focus:ring-red-500',
                    'dark:border-red-600 dark:text-red-400'
                  )
                : cn(
                    'border-gray-300 text-gray-900 placeholder-gray-400',
                    'focus:border-primary-500 focus:ring-primary-500',
                    'dark:border-gray-600 dark:bg-gray-800 dark:text-gray-100',
                    'dark:placeholder-gray-500'
                  ),
              className
            )}
            aria-invalid={hasError}
            aria-describedby={describedBy}
            {...props}
          />
          {rightAddon && (
            <div className="absolute inset-y-0 right-0 pr-3 flex items-center pointer-events-none">
              <span className="text-gray-500 dark:text-gray-400 sm:text-sm">
                {rightAddon}
              </span>
            </div>
          )}
        </div>
        {error && (
          <p
            id={`${id}-error`}
            className="mt-1 text-sm text-red-600 dark:text-red-400"
            role="alert"
          >
            {error}
          </p>
        )}
        {helperText && !error && (
          <p
            id={`${id}-helper`}
            className="mt-1 text-sm text-gray-500 dark:text-gray-400"
          >
            {helperText}
          </p>
        )}
      </div>
    );
  }
);

Input.displayName = 'Input';
