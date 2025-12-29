/**
 * AuditExplorer - Main container for audit event exploration
 */

import { useState, useMemo } from 'react';
import { Card } from '@/components/ui/Card';
import { Spinner } from '@/components/ui/Spinner';
import { AuditEventTable } from './AuditEventTable';
import { AuditTimeFilter } from './AuditTimeFilter';
import { AuditOutcomeFilter } from './AuditOutcomeFilter';
import { AuditAdvancedFilters } from './AuditAdvancedFilters';
import { AuditEventDrawer } from './AuditEventDrawer';
import {
  useAuditEvents,
  useAdminAuditEvents,
  getTimePresetSince,
  type TimePresetValue,
  type AdvancedFilters,
  INITIAL_ADVANCED_FILTERS,
} from './hooks';
import { useAuth } from '@/contexts/AuthContext';
import { exportAdminAuditEvents } from '@/api/enforceai';
import { ArrowPathIcon, ShieldCheckIcon, ArrowDownTrayIcon } from '@heroicons/react/24/outline';
import type { AuditEvent } from '@/api/types';
import { TypeToConfirmDialog } from '@/components/ui/TypeToConfirmDialog';

export function AuditExplorer() {
  const { user } = useAuth();
  const isAdmin = user?.is_admin ?? false;
  const [adminMode, setAdminMode] = useState(false);
  const [timePreset, setTimePreset] = useState<TimePresetValue>('1h');
  const [outcomes, setOutcomes] = useState<string[]>([]);
  const [advancedFilters, setAdvancedFilters] = useState<AdvancedFilters>(INITIAL_ADVANCED_FILTERS);
  const [selectedEvent, setSelectedEvent] = useState<AuditEvent | null>(null);
  const [isDrawerOpen, setIsDrawerOpen] = useState(false);
  const [isExporting, setIsExporting] = useState(false);
  const [isExportConfirmOpen, setIsExportConfirmOpen] = useState(false);

  // Determine if we're actually in admin mode (must be admin AND have toggled it on)
  const effectiveAdminMode = isAdmin && adminMode;
  const isExportAllUsers = effectiveAdminMode && !advancedFilters.userId.trim();

  const handleEventClick = (event: AuditEvent) => {
    setSelectedEvent(event);
    setIsDrawerOpen(true);
  };

  const handleDrawerClose = () => {
    setIsDrawerOpen(false);
  };

  const handleFilterByRequestId = (requestId: string) => {
    setAdvancedFilters((prev) => ({
      ...prev,
      requestId,
    }));
  };

  const handleAdminModeToggle = () => {
    setAdminMode(!adminMode);
    // Clear user ID filter when toggling off admin mode
    if (adminMode) {
      setAdvancedFilters((prev) => ({ ...prev, userId: '' }));
    }
  };

  const handleExport = async () => {
    if (!effectiveAdminMode) return;
    setIsExportConfirmOpen(true);
  };

  const handleConfirmExport = async () => {
    if (!effectiveAdminMode) return;

    setIsExportConfirmOpen(false);
    setIsExporting(true);
    try {
      await exportAdminAuditEvents(queryParams);
    } catch (error) {
      console.error('Export failed:', error);
    } finally {
      setIsExporting(false);
    }
  };

  // Build query parameters
  const queryParams = useMemo(() => {
    const params: Record<string, unknown> = {
      since: getTimePresetSince(timePreset),
    };

    if (outcomes.length > 0) {
      params.outcome = outcomes;
    }

    // Add advanced filters
    if (advancedFilters.agentId.trim()) {
      params.agent_id = advancedFilters.agentId.trim();
    }
    if (advancedFilters.actions.length > 0) {
      params.action = advancedFilters.actions;
    }
    if (advancedFilters.requestId.trim()) {
      params.request_id = advancedFilters.requestId.trim();
    }
    if (advancedFilters.server.trim()) {
      params.server = advancedFilters.server.trim();
    }
    if (advancedFilters.tool.trim()) {
      params.tool = advancedFilters.tool.trim();
    }

    // Add user_id filter for admin mode
    if (effectiveAdminMode && advancedFilters.userId.trim()) {
      params.user_id = advancedFilters.userId.trim();
    }

    return params;
  }, [timePreset, outcomes, advancedFilters, effectiveAdminMode]);

  // Use the appropriate hook based on mode
  const selfServiceQuery = useAuditEvents(queryParams, { enabled: !effectiveAdminMode });
  const adminQuery = useAdminAuditEvents(queryParams, { enabled: effectiveAdminMode });

  const { data, isLoading, isFetching, refetch, isError, error } = effectiveAdminMode
    ? adminQuery
    : selfServiceQuery;

  return (
    <Card>
      <TypeToConfirmDialog
        open={isExportConfirmOpen}
        onClose={() => setIsExportConfirmOpen(false)}
        onConfirm={handleConfirmExport}
        title={isExportAllUsers ? 'Export audit events (all users)' : 'Export audit events'}
        message={
          isExportAllUsers
            ? 'This will download audit events for all users as a CSV file. Audit exports can contain sensitive information. Narrow filters if possible.'
            : 'This will download audit events as a CSV file. Audit exports can contain sensitive information.'
        }
        confirmValue={isExportAllUsers ? 'EXPORT ALL' : 'EXPORT'}
        confirmLabel="Export CSV"
        loading={isExporting}
      />

      {/* Admin Mode Banner */}
      {effectiveAdminMode && (
        <div className="px-4 py-2 bg-amber-50 dark:bg-amber-900/20 border-b border-amber-200 dark:border-amber-800">
          <div className="flex items-center gap-2 text-sm text-amber-700 dark:text-amber-300">
            <ShieldCheckIcon className="h-4 w-4" />
            <span className="font-medium">Admin Mode</span>
            <span className="text-amber-600 dark:text-amber-400">
              - Viewing audit events for all users
            </span>
          </div>
        </div>
      )}

      <div className="p-4 border-b border-gray-200 dark:border-gray-700">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <div className="flex flex-wrap items-center gap-4">
            <AuditTimeFilter value={timePreset} onChange={setTimePreset} />
            <AuditOutcomeFilter value={outcomes} onChange={setOutcomes} />
          </div>

          <div className="flex items-center gap-2">
            {/* Admin Mode Toggle (only visible to admins) */}
            {isAdmin && (
              <button
                onClick={handleAdminModeToggle}
                className={`flex items-center gap-2 px-3 py-1.5 text-sm font-medium rounded-md transition-colors ${
                  adminMode
                    ? 'bg-amber-100 dark:bg-amber-900/30 text-amber-700 dark:text-amber-300 hover:bg-amber-200 dark:hover:bg-amber-900/50'
                    : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 hover:bg-gray-100 dark:hover:bg-gray-800'
                }`}
              >
                <ShieldCheckIcon className="h-4 w-4" />
                Admin
              </button>
            )}
            <AuditAdvancedFilters
              value={advancedFilters}
              onChange={setAdvancedFilters}
              isAdmin={effectiveAdminMode}
            />
            {effectiveAdminMode && (
              <button
                onClick={handleExport}
                disabled={isExporting}
                className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 disabled:opacity-50"
              >
                <ArrowDownTrayIcon className={`h-4 w-4 ${isExporting ? 'animate-pulse' : ''}`} />
                Export CSV
              </button>
            )}
            <button
              onClick={() => refetch()}
              disabled={isFetching}
              className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100 disabled:opacity-50"
            >
              <ArrowPathIcon className={`h-4 w-4 ${isFetching ? 'animate-spin' : ''}`} />
              Refresh
            </button>
          </div>
        </div>
      </div>

      <div className="min-h-[400px]">
        {isLoading ? (
          <div className="flex items-center justify-center py-12">
            <Spinner size="lg" />
          </div>
        ) : isError ? (
          <div className="text-center py-12">
            <p className="text-red-600 dark:text-red-400">
              Failed to load audit events
            </p>
            <p className="text-sm text-gray-500 dark:text-gray-400 mt-1">
              {error?.message || 'An unexpected error occurred'}
            </p>
            <button
              onClick={() => refetch()}
              className="mt-4 px-4 py-2 text-sm font-medium text-white bg-primary-600 hover:bg-primary-700 rounded-md"
            >
              Try Again
            </button>
          </div>
        ) : (
          <AuditEventTable events={data?.items || []} onEventClick={handleEventClick} />
        )}
      </div>

      {data?.next_cursor && !isLoading && (
        <div className="p-4 border-t border-gray-200 dark:border-gray-700 text-center">
          <p className="text-sm text-gray-500 dark:text-gray-400">
            Showing {data.items.length} events. More events available.
          </p>
        </div>
      )}

      <AuditEventDrawer
        event={selectedEvent}
        isOpen={isDrawerOpen}
        onClose={handleDrawerClose}
        onFilterByRequestId={handleFilterByRequestId}
      />
    </Card>
  );
}
