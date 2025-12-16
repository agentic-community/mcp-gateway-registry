import { type HTMLAttributes, type ReactNode } from 'react';
import { cn } from '../../lib/cn';

export type BadgeVariant = 'success' | 'warning' | 'error' | 'neutral' | 'info';
export type BadgeSize = 'sm' | 'md';

export interface BadgeProps extends HTMLAttributes<HTMLSpanElement> {
  variant?: BadgeVariant;
  size?: BadgeSize;
  children: ReactNode;
}

const variantStyles: Record<BadgeVariant, string> = {
  success: cn(
    'bg-green-100 text-green-800',
    'dark:bg-green-900/30 dark:text-green-400'
  ),
  warning: cn(
    'bg-yellow-100 text-yellow-800',
    'dark:bg-yellow-900/30 dark:text-yellow-400'
  ),
  error: cn(
    'bg-red-100 text-red-800',
    'dark:bg-red-900/30 dark:text-red-400'
  ),
  neutral: cn(
    'bg-gray-100 text-gray-800',
    'dark:bg-gray-700 dark:text-gray-300'
  ),
  info: cn(
    'bg-blue-100 text-blue-800',
    'dark:bg-blue-900/30 dark:text-blue-400'
  ),
};

const sizeStyles: Record<BadgeSize, string> = {
  sm: 'px-2 py-0.5 text-xs',
  md: 'px-2.5 py-0.5 text-sm',
};

export function Badge({
  variant = 'neutral',
  size = 'sm',
  children,
  className,
  ...props
}: BadgeProps) {
  return (
    <span
      {...props}
      className={cn(
        'inline-flex items-center font-medium rounded-full',
        variantStyles[variant],
        sizeStyles[size],
        className
      )}
    >
      {children}
    </span>
  );
}
