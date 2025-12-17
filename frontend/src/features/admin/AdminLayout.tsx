/**
 * AdminLayout - Layout wrapper for admin pages with warning banner
 */

import { type ReactNode } from 'react';
import { ExclamationTriangleIcon } from '@heroicons/react/24/outline';

interface AdminLayoutProps {
  children: ReactNode;
}

/**
 * Layout wrapper for admin pages that displays a warning banner
 * to indicate elevated permissions context
 */
export function AdminLayout({ children }: AdminLayoutProps) {
  return (
    <div className="space-y-4">
      {/* Admin Warning Banner */}
      <div className="bg-amber-50 dark:bg-amber-900/20 border-l-4 border-amber-400 dark:border-amber-600 p-4 rounded">
        <div className="flex items-start gap-3">
          <ExclamationTriangleIcon className="h-6 w-6 text-amber-600 dark:text-amber-400 flex-shrink-0 mt-0.5" />
          <div className="flex-1">
            <h3 className="text-sm font-semibold text-amber-800 dark:text-amber-200 mb-1">
              Admin Mode
            </h3>
            <p className="text-sm text-amber-700 dark:text-amber-300">
              You are viewing the admin interface with elevated permissions. Actions taken here
              can affect other users' data and system configuration. Proceed with caution.
            </p>
          </div>
        </div>
      </div>

      {/* Admin Page Content */}
      {children}
    </div>
  );
}
