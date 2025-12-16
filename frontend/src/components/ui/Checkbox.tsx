import { forwardRef, type InputHTMLAttributes, useId } from 'react';
import { cn } from '../../lib/cn';

export interface CheckboxProps extends Omit<InputHTMLAttributes<HTMLInputElement>, 'type'> {
  label?: string;
  description?: string;
}

export const Checkbox = forwardRef<HTMLInputElement, CheckboxProps>(
  ({ className, label, description, id: providedId, ...props }, ref) => {
    const generatedId = useId();
    const id = providedId || generatedId;

    return (
      <div className="relative flex items-start">
        <div className="flex h-6 items-center">
          <input
            ref={ref}
            id={id}
            type="checkbox"
            className={cn(
              'h-4 w-4 rounded',
              'border-gray-300 dark:border-gray-600',
              'text-primary-600 dark:text-primary-500',
              'focus:ring-primary-500 dark:focus:ring-primary-400',
              'focus:ring-2 focus:ring-offset-0',
              'bg-white dark:bg-gray-800',
              'disabled:opacity-50 disabled:cursor-not-allowed',
              className
            )}
            {...props}
          />
        </div>
        {(label || description) && (
          <div className="ml-3 text-sm leading-6">
            {label && (
              <label
                htmlFor={id}
                className={cn(
                  'font-medium',
                  'text-gray-900 dark:text-gray-100',
                  props.disabled && 'opacity-50 cursor-not-allowed'
                )}
              >
                {label}
              </label>
            )}
            {description && (
              <p className="text-gray-500 dark:text-gray-400">{description}</p>
            )}
          </div>
        )}
      </div>
    );
  }
);

Checkbox.displayName = 'Checkbox';
