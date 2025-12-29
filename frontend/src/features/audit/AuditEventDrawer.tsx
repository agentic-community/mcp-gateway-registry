/**
 * AuditEventDrawer - Slide-out drawer showing full event details
 */

import { Dialog, DialogPanel, DialogTitle } from '@headlessui/react';
import { XMarkIcon, ClipboardDocumentIcon, MagnifyingGlassIcon } from '@heroicons/react/24/outline';
import { Badge } from '@/components/ui/Badge';
import { CopyButton } from '@/components/ui/CopyButton';
import type { AuditEvent } from '@/api/types';

interface AuditEventDrawerProps {
  event: AuditEvent | null;
  isOpen: boolean;
  onClose: () => void;
  onFilterByRequestId?: (requestId: string) => void;
}

/**
 * Format a date string for display
 */
function formatDateTime(isoString: string): string {
  const date = new Date(isoString);
  return date.toLocaleString(undefined, {
    year: 'numeric',
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
    timeZoneName: 'short',
  });
}

/**
 * Detail row component
 */
function DetailRow({
  label,
  value,
  copyable = false,
  mono = false,
}: {
  label: string;
  value: string | null | undefined;
  copyable?: boolean;
  mono?: boolean;
}) {
  if (!value) return null;

  return (
    <div className="py-3 border-b border-gray-100 dark:border-gray-800">
      <dt className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider mb-1">
        {label}
      </dt>
      <dd className="flex items-center gap-2">
        <span
          className={`text-sm text-gray-900 dark:text-gray-100 ${mono ? 'font-mono' : ''} break-all`}
        >
          {value}
        </span>
        {copyable && <CopyButton text={value} size="sm" />}
      </dd>
    </div>
  );
}

export function AuditEventDrawer({ event, isOpen, onClose, onFilterByRequestId }: AuditEventDrawerProps) {
  if (!event) return null;

  const handleViewRelatedEvents = () => {
    if (event.request_id && onFilterByRequestId) {
      onFilterByRequestId(event.request_id);
      onClose();
    }
  };

  const detailsJson = event.details ? JSON.stringify(event.details, null, 2) : null;

  return (
    <Dialog open={isOpen} onClose={onClose} className="relative z-50">
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/30" aria-hidden="true" />

      {/* Drawer container */}
      <div className="fixed inset-0 flex justify-end">
        <DialogPanel className="w-full max-w-lg bg-white dark:bg-gray-900 shadow-xl overflow-y-auto">
          {/* Header */}
          <div className="sticky top-0 z-10 bg-white dark:bg-gray-900 border-b border-gray-200 dark:border-gray-700 px-6 py-4">
            <div className="flex items-center justify-between">
              <DialogTitle className="text-lg font-semibold text-gray-900 dark:text-gray-100">
                Audit Event Details
              </DialogTitle>
              <button
                onClick={onClose}
                className="p-2 text-gray-400 hover:text-gray-600 dark:hover:text-gray-300 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-800"
              >
                <XMarkIcon className="h-5 w-5" />
              </button>
            </div>
            <div className="flex items-center gap-2 mt-2">
              <Badge variant={event.outcome === 'allow' ? 'success' : 'error'}>
                {event.outcome === 'allow' ? 'Allowed' : 'Denied'}
              </Badge>
              <span className="text-sm text-gray-500 dark:text-gray-400">
                {formatDateTime(event.occurred_at)}
              </span>
            </div>
          </div>

          {/* Content */}
          <div className="px-6 py-4">
            <dl>
              <DetailRow label="Event ID" value={event.event_id.toString()} mono copyable />
              <DetailRow label="Action" value={event.action} mono />
              <DetailRow label="Occurred At" value={formatDateTime(event.occurred_at)} />
              <DetailRow label="User ID" value={event.user_id} mono copyable />
              <DetailRow label="Agent ID" value={event.agent_id} mono copyable />
              <DetailRow label="Request ID" value={event.request_id} mono copyable />
            </dl>

            {/* View Related Events Button */}
            {event.request_id && onFilterByRequestId && (
              <div className="mt-4">
                <button
                  onClick={handleViewRelatedEvents}
                  className="w-full flex items-center justify-center gap-2 px-4 py-2 text-sm font-medium text-primary-700 dark:text-primary-300 bg-primary-50 dark:bg-primary-900/30 hover:bg-primary-100 dark:hover:bg-primary-900/50 rounded-lg transition-colors"
                >
                  <MagnifyingGlassIcon className="h-4 w-4" />
                  View All Events for This Request
                </button>
              </div>
            )}

            {/* Details JSON */}
            {detailsJson && (
              <div className="mt-6">
                <div className="flex items-center justify-between mb-2">
                  <h4 className="text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider">
                    Details
                  </h4>
                  <CopyButton text={detailsJson} size="sm" />
                </div>
                <pre className="p-4 bg-gray-50 dark:bg-gray-800 rounded-lg text-xs font-mono text-gray-800 dark:text-gray-200 overflow-x-auto whitespace-pre-wrap break-all">
                  {detailsJson}
                </pre>
              </div>
            )}

            {/* Empty details state */}
            {!event.details && (
              <div className="mt-6 text-center py-8 bg-gray-50 dark:bg-gray-800 rounded-lg">
                <ClipboardDocumentIcon className="h-8 w-8 text-gray-400 mx-auto mb-2" />
                <p className="text-sm text-gray-500 dark:text-gray-400">
                  No additional details available
                </p>
              </div>
            )}
          </div>

          {/* Footer */}
          <div className="sticky bottom-0 bg-white dark:bg-gray-900 border-t border-gray-200 dark:border-gray-700 px-6 py-4">
            <button
              onClick={onClose}
              className="w-full px-4 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-800 hover:bg-gray-200 dark:hover:bg-gray-700 rounded-lg"
            >
              Close
            </button>
          </div>
        </DialogPanel>
      </div>
    </Dialog>
  );
}
