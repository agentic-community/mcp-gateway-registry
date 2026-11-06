import React from 'react';

type LifecycleStatus = 'active' | 'deprecated' | 'draft' | 'beta';

interface StatusBadgeProps {
  status: LifecycleStatus;
  className?: string;
}

const STATUS_CONFIG: Record<
  LifecycleStatus,
  {
    label: string;
    tooltip: string;
    colorClasses: string;
  }
> = {
  active: {
    label: 'Active',
    tooltip: 'This item is active and ready for use',
    colorClasses:
      'bg-green-50 text-green-700 dark:bg-green-900/30 dark:text-green-300',
  },
  deprecated: {
    label: 'Deprecated',
    tooltip: 'This item is deprecated and may be removed in the future',
    colorClasses:
      'bg-orange-50 text-orange-700 dark:bg-orange-900/30 dark:text-orange-300',
  },
  draft: {
    label: 'Draft',
    tooltip: 'This item is in draft mode and not yet ready for production',
    colorClasses:
      'bg-gray-500/15 text-gray-700 dark:text-gray-300',
  },
  beta: {
    label: 'Beta',
    tooltip: 'This item is in beta testing phase',
    colorClasses:
      'bg-blue-50 text-blue-700 dark:bg-blue-900/30 dark:text-blue-300',
  },
};

/**
 * StatusBadge component displays the lifecycle status of a server or agent.
 *
 * Features:
 * - Color-coded badges for different statuses
 * - Tooltip with status description
 * - Dark mode support
 */
const StatusBadge: React.FC<StatusBadgeProps> = ({ status, className = '' }) => {
  const config = STATUS_CONFIG[status] || STATUS_CONFIG.active;

  return (
    <span
      className={`
        inline-flex items-center px-2 py-0.5 text-xs font-medium rounded
        ${config.colorClasses}
        transition-colors duration-200
        ${className}
      `}
      title={config.tooltip}
    >
      {config.label}
    </span>
  );
};

export default StatusBadge;
