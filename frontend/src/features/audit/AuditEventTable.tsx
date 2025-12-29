/**
 * AuditEventTable - Displays audit events in a table format
 */

import { Badge } from '@/components/ui/Badge';
import { CopyButton } from '@/components/ui/CopyButton';
import type { AuditEvent } from '@/api/types';

interface AuditEventTableProps {
  events: AuditEvent[];
  onEventClick?: (event: AuditEvent) => void;
}

/**
 * Format a UTC ISO date to local time
 */
function formatTime(isoDate: string): { local: string; utc: string } {
  const date = new Date(isoDate);
  return {
    local: date.toLocaleString(),
    utc: date.toISOString().replace('T', ' ').substring(0, 19) + ' UTC',
  };
}

/**
 * Shorten a UUID for display
 */
function shortenUuid(uuid: string): string {
  if (uuid.length < 8) return uuid;
  return uuid.substring(0, 8) + '...';
}

/**
 * Get a summary from event details
 */
function getEventSummary(event: AuditEvent): string {
  if (!event.details) return '-';

  const parts: string[] = [];
  const details = event.details as Record<string, unknown>;

  if (details.server) {
    parts.push(`server: ${details.server}`);
  }
  if (details.tool) {
    parts.push(`tool: ${details.tool}`);
  }
  if (details.reason) {
    parts.push(`reason: ${details.reason}`);
  }

  return parts.length > 0 ? parts.join(', ') : '-';
}

export function AuditEventTable({ events, onEventClick }: AuditEventTableProps) {
  if (events.length === 0) {
    return (
      <div className="text-center py-12">
        <p className="text-gray-500 dark:text-gray-400">No audit events found</p>
        <p className="text-sm text-gray-400 dark:text-gray-500 mt-1">
          Try adjusting your filters or time range
        </p>
      </div>
    );
  }

  return (
    <div className="overflow-x-auto">
      <table className="min-w-full divide-y divide-gray-200 dark:divide-gray-700">
        <thead className="bg-gray-50 dark:bg-gray-800">
          <tr>
            <th
              scope="col"
              className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
            >
              Time
            </th>
            <th
              scope="col"
              className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
            >
              Outcome
            </th>
            <th
              scope="col"
              className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
            >
              Action
            </th>
            <th
              scope="col"
              className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
            >
              Agent
            </th>
            <th
              scope="col"
              className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
            >
              Summary
            </th>
            <th
              scope="col"
              className="px-4 py-3 text-left text-xs font-medium text-gray-500 dark:text-gray-400 uppercase tracking-wider"
            >
              Request ID
            </th>
          </tr>
        </thead>
        <tbody className="bg-white dark:bg-gray-900 divide-y divide-gray-200 dark:divide-gray-700">
          {events.map((event) => {
            const time = formatTime(event.occurred_at);
            return (
              <tr
                key={event.event_id}
                onClick={() => onEventClick?.(event)}
                className={`hover:bg-gray-50 dark:hover:bg-gray-800/50 transition-colors ${onEventClick ? 'cursor-pointer' : ''}`}
              >
                <td className="px-4 py-3 whitespace-nowrap">
                  <span
                    className="text-sm text-gray-900 dark:text-gray-100"
                    title={time.utc}
                  >
                    {time.local}
                  </span>
                </td>
                <td className="px-4 py-3 whitespace-nowrap">
                  <Badge
                    variant={event.outcome === 'allow' ? 'success' : 'error'}
                    size="sm"
                  >
                    {event.outcome}
                  </Badge>
                </td>
                <td className="px-4 py-3 whitespace-nowrap">
                  <code className="text-sm font-mono text-gray-900 dark:text-gray-100">
                    {event.action}
                  </code>
                </td>
                <td className="px-4 py-3 whitespace-nowrap">
                  <div className="flex items-center gap-1">
                    <span
                      className="text-sm font-mono text-gray-600 dark:text-gray-400"
                      title={event.agent_id}
                    >
                      {shortenUuid(event.agent_id)}
                    </span>
                    <span onClick={(e) => e.stopPropagation()}>
                      <CopyButton text={event.agent_id} />
                    </span>
                  </div>
                </td>
                <td className="px-4 py-3">
                  <span className="text-sm text-gray-600 dark:text-gray-400 max-w-xs truncate block">
                    {getEventSummary(event)}
                  </span>
                </td>
                <td className="px-4 py-3 whitespace-nowrap">
                  {event.request_id ? (
                    <div className="flex items-center gap-1">
                      <span
                        className="text-sm font-mono text-gray-600 dark:text-gray-400"
                        title={event.request_id}
                      >
                        {shortenUuid(event.request_id)}
                      </span>
                      <span onClick={(e) => e.stopPropagation()}>
                        <CopyButton text={event.request_id} />
                      </span>
                    </div>
                  ) : (
                    <span className="text-sm text-gray-400 dark:text-gray-500">-</span>
                  )}
                </td>
              </tr>
            );
          })}
        </tbody>
      </table>
    </div>
  );
}
