import React from 'react';
import { ChevronDownIcon } from '@heroicons/react/24/outline';


interface ServerVersion {
  version: string;
  proxy_pass_url: string;
  status: string;
  is_default: boolean;
  released?: string;
  sunset_date?: string;
  description?: string;
}


interface VersionBadgeProps {
  versions?: ServerVersion[] | null;
  defaultVersion?: string | null;
  onClick?: () => void;
}


/**
 * VersionBadge component displays the current version of a server.
 *
 * - Shows the default version as a clickable badge
 * - Displays dropdown arrow when multiple versions exist
 * - Hidden when server has no versions (single-version backward compatibility)
 */
const VersionBadge: React.FC<VersionBadgeProps> = ({
  versions,
  defaultVersion,
  onClick
}) => {
  // Don't render badge if no versions configured (backward compatibility)
  if (!versions || versions.length === 0) {
    return null;
  }

  // Find the current default version
  const currentVersion = defaultVersion ||
    versions.find(v => v.is_default)?.version ||
    versions[0]?.version ||
    'v1.0.0';

  const hasMultipleVersions = versions.length > 1;

  return (
    <button
      onClick={onClick}
      disabled={!onClick || !hasMultipleVersions}
      className={`
        inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium rounded
        ${hasMultipleVersions
          ? 'bg-indigo-50 text-indigo-700 hover:bg-indigo-100 dark:bg-indigo-900/30 dark:text-indigo-300 dark:hover:bg-indigo-900/50 cursor-pointer'
          : 'bg-gray-500/10 text-fg-muted cursor-default'
        }
        transition-colors duration-200
      `}
      title={hasMultipleVersions ? 'Click to manage versions' : `Version: ${currentVersion}`}
    >
      {currentVersion}
      {hasMultipleVersions && (
        <ChevronDownIcon className="h-3 w-3" />
      )}
    </button>
  );
};


export default VersionBadge;
