import React, { useState } from 'react';
import axios from 'axios';
import {
  XMarkIcon,
  CheckCircleIcon,
  ExclamationTriangleIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';
import useEscapeKey from '../hooks/useEscapeKey';


interface ServerVersion {
  version: string;
  proxy_pass_url: string;
  status: string;
  is_default: boolean;
  released?: string;
  sunset_date?: string;
  description?: string;
}


interface VersionSelectorModalProps {
  isOpen: boolean;
  onClose: () => void;
  serverName: string;
  serverPath: string;
  versions: ServerVersion[];
  defaultVersion: string | null;
  onVersionChange?: (newDefaultVersion: string) => void;
  onRefreshServer?: () => void;
  onShowToast?: (message: string, type: 'success' | 'error') => void;
  authToken?: string | null;
  canModify?: boolean;
}


/**
 * VersionSelectorModal displays all versions of a server and allows
 * administrators to switch the default version.
 */
const VersionSelectorModal: React.FC<VersionSelectorModalProps> = ({
  isOpen,
  onClose,
  serverName,
  serverPath,
  versions,
  defaultVersion,
  onVersionChange,
  onRefreshServer,
  onShowToast,
  authToken,
  canModify = false,
}) => {
  const [loading, setLoading] = useState<string | null>(null);

  useEscapeKey(onClose, isOpen);

  if (!isOpen) {
    return null;
  }

  const handleSetDefault = async (version: string) => {
    if (loading || version === defaultVersion) {
      return;
    }

    setLoading(version);
    try {
      const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
      await axios.put(
        `/api/servers${serverPath}/versions/default`,
        { version },
        headers ? { headers } : undefined
      );

      if (onVersionChange) {
        onVersionChange(version);
      }

      if (onShowToast) {
        onShowToast(`Switched to ${version}`, 'success');
      }

      // Trigger a server refresh to get updated data
      if (onRefreshServer) {
        onRefreshServer();
      }

      onClose();
    } catch (error: any) {
      console.error('Failed to set default version:', error);
      if (onShowToast) {
        onShowToast(
          error.response?.data?.detail || 'Failed to switch version',
          'error'
        );
      }
    } finally {
      setLoading(null);
    }
  };

  const getStatusBadge = (status: string, isDefault: boolean) => {
    if (isDefault) {
      return (
        <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-semibold bg-green-100 text-green-700 dark:bg-green-900/30 dark:text-green-400 rounded-full">
          <CheckCircleIcon className="h-3 w-3" />
          ACTIVE
        </span>
      );
    }

    switch (status) {
      case 'deprecated':
        return (
          <span className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-medium bg-amber-100 text-amber-700 dark:bg-amber-900/30 dark:text-amber-400 rounded-full">
            <ExclamationTriangleIcon className="h-3 w-3" />
            deprecated
          </span>
        );
      case 'beta':
        return (
          <span className="px-2 py-0.5 text-xs font-medium bg-blue-100 text-blue-700 dark:bg-blue-900/30 dark:text-blue-400 rounded-full">
            beta
          </span>
        );
      default:
        return (
          <span className="px-2 py-0.5 text-xs font-medium bg-surface-muted text-fg-muted rounded-full">
            stable
          </span>
        );
    }
  };

  return (
    <div className="fixed inset-0 bg-black/50 backdrop-blur-sm flex items-center justify-center z-50">
      <div className="bg-surface-overlay border border-line rounded-xl p-6 max-w-lg w-full mx-4 max-h-[80vh] overflow-auto shadow-2xl">
        {/* Header */}
        <div className="flex items-center justify-between mb-6">
          <div>
            <h3 className="text-lg font-semibold text-fg">
              Select Version
            </h3>
            <p className="text-sm text-fg-muted mt-1">
              {serverName}
            </p>
          </div>
          <button
            onClick={onClose}
            className="btn-icon"
          >
            <XMarkIcon className="h-5 w-5" />
          </button>
        </div>

        {/* Version Cards */}
        <div className="space-y-3">
          {versions.map((version) => {
            const isCurrentDefault = version.version === defaultVersion || version.is_default;
            const isLoading = loading === version.version;

            return (
              <div
                key={version.version}
                className={`
                  border rounded-lg p-4 transition-all
                  ${isCurrentDefault
                    ? 'border-green-300 bg-green-50/50 dark:border-green-700 dark:bg-green-900/20'
                    : 'border-line hover:border-line-strong'
                  }
                `}
              >
                {/* Version Header */}
                <div className="flex items-center justify-between mb-2">
                  <div className="flex items-center gap-2">
                    <span className="font-semibold text-fg">
                      {version.version}
                    </span>
                    {getStatusBadge(version.status, isCurrentDefault)}
                  </div>

                  {canModify && !isCurrentDefault && (
                    <button
                      onClick={() => handleSetDefault(version.version)}
                      disabled={isLoading}
                      className="btn-ghost px-3 py-1.5 text-accent-text hover:bg-accent-soft hover:text-accent-soft-fg"
                    >
                      {isLoading ? (
                        <ArrowPathIcon className="h-4 w-4 animate-spin" />
                      ) : (
                        'Set Active'
                      )}
                    </button>
                  )}
                </div>

                {/* Version Details */}
                <div className="space-y-1 text-sm">
                  <div className="text-fg-muted">
                    <span className="font-medium">Backend:</span>{' '}
                    <code className="text-xs bg-surface-muted text-fg px-1 py-0.5 rounded">
                      {version.proxy_pass_url}
                    </code>
                  </div>

                  {version.released && (
                    <div className="text-fg-muted">
                      <span className="font-medium">Released:</span> {version.released}
                    </div>
                  )}

                  {version.sunset_date && (
                    <div className="text-amber-600 dark:text-amber-400">
                      <span className="font-medium">Sunset:</span> {version.sunset_date}
                    </div>
                  )}

                  {version.description && (
                    <div className="text-fg-muted mt-2">
                      {version.description}
                    </div>
                  )}
                </div>
              </div>
            );
          })}
        </div>

        {/* Info Footer */}
        <div className="mt-6 pt-4 border-t border-line">
          <p className="text-xs text-fg-muted">
            Clients can request specific versions using the{' '}
            <code className="bg-surface-muted text-fg px-1 py-0.5 rounded">
              X-MCP-Server-Version
            </code>{' '}
            header.
          </p>
        </div>
      </div>
    </div>
  );
};


export default VersionSelectorModal;
