import { Fragment, type ReactNode } from 'react';
import { Link } from 'react-router-dom';
import { ChevronRightIcon, HomeIcon } from '@heroicons/react/24/outline';
import { cn } from '../../lib/cn';

export interface Breadcrumb {
  name: string;
  href?: string;
}

interface PageHeaderProps {
  title: string;
  description?: string;
  breadcrumbs?: Breadcrumb[];
  actions?: ReactNode;
  className?: string;
}

export function PageHeader({
  title,
  description,
  breadcrumbs,
  actions,
  className,
}: PageHeaderProps) {
  return (
    <div className={cn('mb-6', className)}>
      {/* Breadcrumbs */}
      {breadcrumbs && breadcrumbs.length > 0 && (
        <nav className="flex mb-4" aria-label="Breadcrumb">
          <ol className="flex items-center space-x-2">
            <li>
              <Link
                to="/"
                className="text-gray-400 hover:text-gray-500 dark:text-gray-500 dark:hover:text-gray-400"
              >
                <HomeIcon className="h-4 w-4 flex-shrink-0" aria-hidden="true" />
                <span className="sr-only">Home</span>
              </Link>
            </li>
            {breadcrumbs.map((crumb, index) => (
              <Fragment key={crumb.name}>
                <li className="flex items-center">
                  <ChevronRightIcon
                    className="h-4 w-4 flex-shrink-0 text-gray-400 dark:text-gray-500"
                    aria-hidden="true"
                  />
                </li>
                <li>
                  {crumb.href && index < breadcrumbs.length - 1 ? (
                    <Link
                      to={crumb.href}
                      className="ml-2 text-sm font-medium text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-300"
                    >
                      {crumb.name}
                    </Link>
                  ) : (
                    <span
                      className="ml-2 text-sm font-medium text-gray-700 dark:text-gray-300"
                      aria-current={index === breadcrumbs.length - 1 ? 'page' : undefined}
                    >
                      {crumb.name}
                    </span>
                  )}
                </li>
              </Fragment>
            ))}
          </ol>
        </nav>
      )}

      {/* Title and actions */}
      <div className="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold text-gray-900 dark:text-white">
            {title}
          </h1>
          {description && (
            <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
              {description}
            </p>
          )}
        </div>
        {actions && (
          <div className="flex items-center gap-3 flex-shrink-0">
            {actions}
          </div>
        )}
      </div>
    </div>
  );
}
