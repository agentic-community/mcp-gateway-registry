import React, { useState, useCallback } from 'react';
import { Transition } from '@headlessui/react';
import { ShieldExclamationIcon, ArrowDownTrayIcon } from '@heroicons/react/24/outline';
import { useAuth } from '../contexts/AuthContext';
import AuditFilterBar from '../components/AuditFilterBar';
import type { AuditFilters } from '../components/AuditFilterBar';
import AuditLogTable from '../components/AuditLogTable';
import AuditEventDetail from '../components/AuditEventDetail';
import AuditStatistics from '../components/AuditStatistics';
import AuditExecutiveSummary from '../components/AuditExecutiveSummary';
import type { AuditEvent } from '../types/audit';
import { buildAuditQueryParams } from '../utils/auditQuery';
import useEscapeKey from '../hooks/useEscapeKey';

/** Audit log viewer, rendered inside Settings > Audit. Admin only. */
const AuditLogsPage: React.FC = () => {
  const { user } = useAuth();
  const [filters, setFilters] = useState<AuditFilters>({
    stream: 'registry_api',
  });
  const [selectedEvent, setSelectedEvent] = useState<AuditEvent | null>(null);
  const [refreshKey, setRefreshKey] = useState(0);

  const handleFilterChange = useCallback((newFilters: AuditFilters) => {
    setFilters(newFilters);
    setSelectedEvent(null);
  }, []);

  const handleRefresh = useCallback(() => {
    setRefreshKey((prev) => prev + 1);
  }, []);

  const handleCloseDetail = useCallback(() => {
    setSelectedEvent(null);
  }, []);
  useEscapeKey(handleCloseDetail, selectedEvent !== null);

  const handleExport = useCallback((format: 'jsonl' | 'csv') => {
    const params = buildAuditQueryParams(filters);
    params.set('format', format);
    // Trigger download by opening the export URL
    window.open(`/api/audit/export?${params.toString()}`, '_blank');
  }, [filters]);

  if (!user?.is_admin) {
    return (
      <div className="flex items-center justify-center p-4">
        <div className="card max-w-md p-8 text-center">
          <ShieldExclamationIcon className="mx-auto mb-4 h-14 w-14 text-danger-500" />
          <h1 className="mb-2 text-xl font-semibold text-fg">Access Denied</h1>
          <p className="text-fg-muted">You need administrator privileges to view audit logs.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="space-y-6">
      <div className="flex flex-wrap items-start justify-between gap-4">
        <div>
          <h2 className="text-xl font-semibold tracking-tight text-fg">Audit Logs</h2>
          <p className="mt-1 text-sm text-fg-muted">
            Search system audit events for compliance and security monitoring.
          </p>
        </div>
        <div className="flex items-center gap-2">
          <span className="hidden text-xs text-fg-subtle md:inline">Export all matching:</span>
          <button
            type="button"
            onClick={() => handleExport('jsonl')}
            className="btn-secondary px-3"
            title="Export every event matching the filters as JSONL"
          >
            <ArrowDownTrayIcon className="h-4 w-4" />
            JSONL
          </button>
          <button
            type="button"
            onClick={() => handleExport('csv')}
            className="btn-secondary px-3"
            title="Export every event matching the filters as CSV"
          >
            <ArrowDownTrayIcon className="h-4 w-4" />
            CSV
          </button>
        </div>
      </div>

      <AuditFilterBar
        filters={filters}
        onFilterChange={handleFilterChange}
        onRefresh={handleRefresh}
      />

      {/* Executive Summary - hero tiles for platform leads */}
      <AuditExecutiveSummary />

      {/* Statistics Dashboard */}
      <AuditStatistics stream={filters.stream} username={filters.username} />

      <AuditLogTable
        key={refreshKey}
        filters={filters}
        onEventSelect={setSelectedEvent}
        selectedEventId={selectedEvent?.request_id}
      />

      {/* Inspector drawer: non-modal so another row can be picked while it is
          open; Escape or the close button dismisses it. */}
      <Transition
        show={selectedEvent !== null}
        enter="transition ease-out duration-200"
        enterFrom="translate-x-full opacity-0"
        enterTo="translate-x-0 opacity-100"
        leave="transition ease-in duration-150"
        leaveFrom="translate-x-0 opacity-100"
        leaveTo="translate-x-full opacity-0"
      >
        <aside
          aria-label="Audit event details"
          className="fixed bottom-0 right-0 top-16 z-40 w-full max-w-xl overflow-y-auto border-l border-line bg-surface-overlay p-4 shadow-xl"
        >
          {selectedEvent && <AuditEventDetail event={selectedEvent} onClose={handleCloseDetail} />}
        </aside>
      </Transition>
    </div>
  );
};

export default AuditLogsPage;
