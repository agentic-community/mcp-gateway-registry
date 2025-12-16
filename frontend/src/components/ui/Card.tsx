import { type ReactNode } from 'react';
import { cn } from '../../lib/cn';

export interface CardProps {
  children: ReactNode;
  className?: string;
  padding?: 'none' | 'sm' | 'md' | 'lg';
}

export interface CardHeaderProps {
  title: string;
  description?: string;
  action?: ReactNode;
  className?: string;
}

export interface CardFooterProps {
  children: ReactNode;
  className?: string;
}

const paddingStyles = {
  none: '',
  sm: 'p-4',
  md: 'p-6',
  lg: 'p-8',
};

export function Card({ children, className, padding = 'md' }: CardProps) {
  return (
    <div
      className={cn(
        'bg-white dark:bg-gray-800',
        'border border-gray-200 dark:border-gray-700',
        'rounded-lg shadow-sm',
        paddingStyles[padding],
        className
      )}
    >
      {children}
    </div>
  );
}

export function CardHeader({
  title,
  description,
  action,
  className,
}: CardHeaderProps) {
  return (
    <div
      className={cn(
        'flex items-center justify-between',
        'pb-4 mb-4',
        'border-b border-gray-200 dark:border-gray-700',
        className
      )}
    >
      <div>
        <h3 className="text-lg font-medium text-gray-900 dark:text-gray-100">
          {title}
        </h3>
        {description && (
          <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
            {description}
          </p>
        )}
      </div>
      {action && <div className="ml-4">{action}</div>}
    </div>
  );
}

export function CardFooter({ children, className }: CardFooterProps) {
  return (
    <div
      className={cn(
        'pt-4 mt-4',
        'border-t border-gray-200 dark:border-gray-700',
        className
      )}
    >
      {children}
    </div>
  );
}
