import React, { useEffect, useRef } from 'react';

interface FieldChange {
  field: string;
  current_value: any;
  remote_value: any;
}

interface PullCardPreviewModalProps {
  isOpen: boolean;
  onClose: () => void;
  onApply: () => void;
  loading: boolean;
  result: {
    agent_path: string;
    remote_card_url: string;
    changes: FieldChange[];
    has_changes: boolean;
    remote_card: any;
  } | null;
  agentName: string;
}

const PullCardPreviewModal: React.FC<PullCardPreviewModalProps> = ({
  isOpen,
  onClose,
  onApply,
  loading,
  result,
  agentName,
}) => {
  // U4: close on Escape and move focus into the dialog when it opens.
  const closeButtonRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!isOpen) return;
    const onKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') onClose();
    };
    document.addEventListener('keydown', onKeyDown);
    closeButtonRef.current?.focus();
    return () => document.removeEventListener('keydown', onKeyDown);
  }, [isOpen, onClose]);

  if (!isOpen) return null;

  const formatValue = (value: any): string => {
    if (value === null || value === undefined) return 'null';
    if (typeof value === 'object') return JSON.stringify(value, null, 2);
    return String(value);
  };

  // U3: the remote card version, shown in the no-changes state for confirmation.
  const remoteVersion =
    result?.remote_card?.version ?? result?.remote_card?.protocolVersion ?? null;

  return (
    <div className="fixed inset-0 z-50 overflow-y-auto">
      <div className="flex items-center justify-center min-h-screen p-4">
        {/* Backdrop */}
        <div className="fixed inset-0 bg-black/50" onClick={onClose} />

        {/* Modal */}
        <div
          role="dialog"
          aria-modal="true"
          aria-label={`Pull agent card for ${agentName}`}
          className="relative bg-white dark:bg-gray-800 rounded-xl shadow-2xl max-w-2xl w-full max-h-[80vh] flex flex-col"
        >
          {/* Header */}
          <div className="px-6 py-4 border-b border-gray-200 dark:border-gray-700">
            <h3 className="text-lg font-bold text-gray-900 dark:text-white">
              Pull Agent Card: {agentName}
            </h3>
            {result && (
              <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
                Source: {result.remote_card_url}
              </p>
            )}
          </div>

          {/* Body */}
          <div className="px-6 py-4 overflow-y-auto flex-1">
            {loading && (
              <div className="flex items-center justify-center py-8">
                <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-cyan-600" />
                <span className="ml-3 text-gray-500">Fetching remote card...</span>
              </div>
            )}

            {result && !result.has_changes && (
              <div className="text-center py-8">
                <p className="text-gray-500 dark:text-gray-400">
                  No changes detected. The local card matches the remote card.
                </p>
                {remoteVersion && (
                  <p className="text-xs text-gray-400 dark:text-gray-500 mt-2">
                    Remote card version: {String(remoteVersion)}
                  </p>
                )}
              </div>
            )}

            {result && result.has_changes && (
              <div className="space-y-4">
                <p className="text-sm text-gray-600 dark:text-gray-300">
                  {result.changes.length} field(s) will be updated:
                </p>
                {result.changes.map((change, idx) => (
                  <div
                    key={idx}
                    className="border border-gray-200 dark:border-gray-700 rounded-lg p-3"
                  >
                    <div className="font-mono text-sm font-bold text-gray-800 dark:text-gray-200 mb-2">
                      {change.field}
                    </div>
                    <div className="grid grid-cols-2 gap-3">
                      <div>
                        <span className="text-xs font-medium text-red-600 dark:text-red-400">
                          Current
                        </span>
                        <pre className="mt-1 text-xs bg-red-50 dark:bg-red-900/20 text-red-800 dark:text-red-200 p-2 rounded overflow-x-auto max-h-32">
                          {formatValue(change.current_value)}
                        </pre>
                      </div>
                      <div>
                        <span className="text-xs font-medium text-green-600 dark:text-green-400">
                          Remote
                        </span>
                        <pre className="mt-1 text-xs bg-green-50 dark:bg-green-900/20 text-green-800 dark:text-green-200 p-2 rounded overflow-x-auto max-h-32">
                          {formatValue(change.remote_value)}
                        </pre>
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="px-6 py-4 border-t border-gray-200 dark:border-gray-700 flex justify-end gap-3">
            <button
              ref={closeButtonRef}
              onClick={onClose}
              className="px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700 hover:bg-gray-200 dark:hover:bg-gray-600 rounded-lg transition-colors"
            >
              Cancel
            </button>
            {result && result.has_changes && (
              <button
                onClick={onApply}
                className="px-4 py-2 text-sm font-medium text-white bg-cyan-600 hover:bg-cyan-700 rounded-lg transition-colors"
              >
                Apply Changes
              </button>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PullCardPreviewModal;
