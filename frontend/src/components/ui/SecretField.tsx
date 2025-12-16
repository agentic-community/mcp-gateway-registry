import { useState, forwardRef, type InputHTMLAttributes, useId } from 'react';
import { EyeIcon, EyeSlashIcon } from '@heroicons/react/24/outline';
import { cn } from '../../lib/cn';
import { CopyIconButton } from './CopyButton';

export interface SecretFieldProps
  extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string;
  error?: string;
  helperText?: string;
  showCopy?: boolean;
  onCopy?: () => void;
}

export const SecretField = forwardRef<HTMLInputElement, SecretFieldProps>(
  (
    {
      className,
      label,
      error,
      helperText,
      showCopy = true,
      onCopy,
      id: providedId,
      value,
      ...props
    },
    ref
  ) => {
    const generatedId = useId();
    const id = providedId || generatedId;
    const [revealed, setRevealed] = useState(false);
    const hasError = Boolean(error);
    const describedBy = error
      ? `${id}-error`
      : helperText
        ? `${id}-helper`
        : undefined;

    const stringValue = typeof value === 'string' ? value : '';

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
        <div className="relative flex items-center">
          <input
            ref={ref}
            id={id}
            type={revealed ? 'text' : 'password'}
            value={value}
            className={cn(
              'block w-full rounded-md shadow-sm',
              'text-sm font-mono',
              'transition-colors duration-150',
              'focus:outline-none focus:ring-2 focus:ring-offset-0',
              'disabled:bg-gray-100 disabled:cursor-not-allowed',
              'dark:disabled:bg-gray-800',
              'pr-20',
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
          <div className="absolute inset-y-0 right-0 flex items-center pr-2 gap-1">
            <button
              type="button"
              onClick={() => setRevealed(!revealed)}
              className={cn(
                'p-1 rounded',
                'text-gray-500 hover:text-gray-700',
                'dark:text-gray-400 dark:hover:text-gray-200',
                'focus:outline-none focus:ring-2 focus:ring-primary-500'
              )}
              aria-label={revealed ? 'Hide secret' : 'Show secret'}
            >
              {revealed ? (
                <EyeSlashIcon className="h-5 w-5" />
              ) : (
                <EyeIcon className="h-5 w-5" />
              )}
            </button>
            {showCopy && stringValue && (
              <CopyIconButton
                text={stringValue}
                onCopy={onCopy}
                aria-label="Copy secret"
              />
            )}
          </div>
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

SecretField.displayName = 'SecretField';
