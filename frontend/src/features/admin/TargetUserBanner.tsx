/**
 * TargetUserBanner - Shows admin context when acting on another user
 */

import { ExclamationTriangleIcon } from '@heroicons/react/24/outline';

export interface TargetUserBannerProps {
  targetUserEmail: string;
  targetUserId: string;
}

export function TargetUserBanner({ targetUserEmail, targetUserId }: TargetUserBannerProps) {
  return (
    <div className="bg-yellow-50 dark:bg-yellow-900/30 border-l-4 border-yellow-400 dark:border-yellow-600 p-4 mb-6">
      <div className="flex items-start">
        <div className="flex-shrink-0">
          <ExclamationTriangleIcon
            className="h-5 w-5 text-yellow-400 dark:text-yellow-500"
            aria-hidden="true"
          />
        </div>
        <div className="ml-3">
          <h3 className="text-sm font-medium text-yellow-800 dark:text-yellow-200">
            Acting on user: {targetUserEmail}
          </h3>
          <div className="mt-1 text-sm text-yellow-700 dark:text-yellow-300">
            <p>
              You are performing administrative actions on behalf of another user. All actions
              will be logged with your admin identity.
            </p>
            <p className="mt-1 font-mono text-xs bg-yellow-100 dark:bg-yellow-900/50 px-2 py-1 rounded inline-block">
              user_id: {targetUserId}
            </p>
          </div>
        </div>
      </div>
    </div>
  );
}
