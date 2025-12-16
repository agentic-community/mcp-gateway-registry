import { type ReactNode } from 'react';
import { cn } from '../../lib/cn';

interface PageContentProps {
  children: ReactNode;
  className?: string;
  fullWidth?: boolean;
  noPadding?: boolean;
}

export function PageContent({
  children,
  className,
  fullWidth = false,
  noPadding = false,
}: PageContentProps) {
  return (
    <div
      className={cn(
        'flex-1',
        !noPadding && 'px-4 sm:px-6 lg:px-8 py-4 md:py-6',
        !fullWidth && 'max-w-7xl mx-auto w-full',
        className
      )}
    >
      {children}
    </div>
  );
}
