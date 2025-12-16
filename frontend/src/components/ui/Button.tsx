import { forwardRef, type ButtonHTMLAttributes, type ReactNode } from 'react';
import { cn } from '../../lib/cn';
import { Spinner } from './Spinner';

export type ButtonVariant = 'primary' | 'secondary' | 'danger' | 'ghost' | 'outline';
export type ButtonSize = 'sm' | 'md' | 'lg';

export interface ButtonProps extends ButtonHTMLAttributes<HTMLButtonElement> {
  variant?: ButtonVariant;
  size?: ButtonSize;
  loading?: boolean;
  leftIcon?: ReactNode;
  rightIcon?: ReactNode;
}

const variantStyles: Record<ButtonVariant, string> = {
  primary: cn(
    'bg-primary-600 text-white',
    'hover:bg-primary-700',
    'focus:ring-primary-500',
    'disabled:bg-primary-300 dark:disabled:bg-primary-800'
  ),
  secondary: cn(
    'bg-white text-gray-700 border border-gray-300',
    'hover:bg-gray-50',
    'focus:ring-primary-500',
    'dark:bg-gray-800 dark:text-gray-200 dark:border-gray-600',
    'dark:hover:bg-gray-700',
    'disabled:bg-gray-100 dark:disabled:bg-gray-900'
  ),
  danger: cn(
    'bg-red-600 text-white',
    'hover:bg-red-700',
    'focus:ring-red-500',
    'disabled:bg-red-300 dark:disabled:bg-red-800'
  ),
  ghost: cn(
    'bg-transparent text-gray-700',
    'hover:bg-gray-100',
    'focus:ring-gray-500',
    'dark:text-gray-300 dark:hover:bg-gray-800',
    'disabled:text-gray-400 dark:disabled:text-gray-600'
  ),
  outline: cn(
    'bg-transparent text-gray-700 border border-gray-300',
    'hover:bg-gray-50',
    'focus:ring-primary-500',
    'dark:text-gray-200 dark:border-gray-600',
    'dark:hover:bg-gray-800',
    'disabled:text-gray-400 dark:disabled:text-gray-600'
  ),
};

const sizeStyles: Record<ButtonSize, string> = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-4 py-2 text-sm',
  lg: 'px-6 py-3 text-base',
};

export const Button = forwardRef<HTMLButtonElement, ButtonProps>(
  (
    {
      className,
      variant = 'primary',
      size = 'md',
      loading = false,
      disabled,
      leftIcon,
      rightIcon,
      children,
      ...props
    },
    ref
  ) => {
    const isDisabled = disabled || loading;

    return (
      <button
        ref={ref}
        className={cn(
          'inline-flex items-center justify-center font-medium rounded-md',
          'transition-colors duration-150',
          'focus:outline-none focus:ring-2 focus:ring-offset-2',
          'dark:focus:ring-offset-gray-900',
          'disabled:cursor-not-allowed',
          variantStyles[variant],
          sizeStyles[size],
          className
        )}
        disabled={isDisabled}
        {...props}
      >
        {loading ? (
          <Spinner
            size={size === 'lg' ? 'md' : 'sm'}
            className={cn(
              children ? 'mr-2' : '',
              variant === 'primary' || variant === 'danger'
                ? 'text-white'
                : 'text-gray-500'
            )}
          />
        ) : leftIcon ? (
          <span className="mr-2">{leftIcon}</span>
        ) : null}
        {children}
        {rightIcon && !loading && <span className="ml-2">{rightIcon}</span>}
      </button>
    );
  }
);

Button.displayName = 'Button';
