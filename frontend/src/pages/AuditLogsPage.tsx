import React, { useState, useCallback } from 'react';
import { useAuth } from '../contexts/AuthContext';
import AuditFilterBar, { AuditFilters } from '../components/AuditFilterBar';
import AuditLogTable, { AuditEvent } from '../components/AuditLogTable';
import AuditEventDetail from '../components/AuditEventDetail';
import AuditStatistics from '../components/AuditStatistics';
import { ShieldExclamationIcon, ArrowDownTrayIcon } from '@heroicons/react/24/outline';

interface AuditLogsPageProps {
  embedded?: boolean;
}

const AuditLogsPage: React.FC<AuditLogsPageProps> = ({ embedded = false }) => {
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

  const handleEventSelect = useCallback((event: AuditEvent) => {
    setSelectedEvent(event);
  }, []);

  const handleCloseDetail = useCallback(() => {
    setSelectedEvent(null);
  }, []);

  const handleExport = useCallback((format: 'jsonl' | 'csv') => {
    const params = new URLSearchParams();
    params.set('stream', filters.stream);
    params.set('format', format);
    
    if (filters.from) {
      params.set('from', new Date(filters.from).toISOString());
    }
    if (filters.to) {
      params.set('to', new Date(filters.to).toISOString());
    }
    if (filters.username) {
      params.set('username', filters.username);
    }
    if (filters.operation) {
      params.set('operation', filters.operation);
    }
    if (filters.resourceType) {
      params.set('resource_type', filters.resourceType);
    }
    if (filters.statusMin !== undefined) {
      params.set('status_min', filters.statusMin.toString());
    }
    if (filters.statusMax !== undefined) {
      params.set('status_max', filters.statusMax.toString());
    }
    
    // Trigger download by opening the export URL
    window.open(`/api/audit/export?${params.toString()}`, '_blank');
  }, [filters]);

  // Check if user is admin
  if (!user?.is_admin) {
    return (
      <div className={embedded ? "flex items-center justify-center p-4" : "min-h-screen bg-gray-50 dark:bg-gray-900 flex items-center justify-center p-4"}>
        <div className="bg-white dark:bg-gray-800 rounded-lg shadow-lg p-8 max-w-md text-center">
          <ShieldExclamationIcon className="h-16 w-16 text-red-500 mx-auto mb-4" />
          <h1 className="text-xl font-semibold text-gray-900 dark:text-gray-100 mb-2">
            Access Denied
          </h1>
          <p className="text-gray-600 dark:text-gray-400">
            You need administrator privileges to view audit logs.
          </p>
        </div>
      </div>
    );
  }

  // Embedded mode - no outer container
  if (embedded) {
    return (
      <div>
        {/* Page Header */}
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h2 className="text-xl font-bold text-gray-900 dark:text-gray-100">
              Audit Logs
            </h2>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              View and search system audit events for compliance and security monitoring.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleExport('jsonl')}
              className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors"
              title="Export as JSONL"
            >
              <ArrowDownTrayIcon className="h-4 w-4" />
              <span>JSONL</span>
            </button>
            <button
              onClick={() => handleExport('csv')}
              className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors"
              title="Export as CSV"
            >
              <ArrowDownTrayIcon className="h-4 w-4" />
              <span>CSV</span>
            </button>
          </div>
        </div>

        {/* Filter Bar */}
        <div className="mb-6">
          <AuditFilterBar
            filters={filters}
            onFilterChange={handleFilterChange}
            onRefresh={handleRefresh}
          />
        </div>

        {/* Statistics Dashboard */}
        <AuditStatistics stream={filters.stream} username={filters.username} />

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Table - takes 2 columns when detail is shown, full width otherwise */}
          <div className={selectedEvent ? 'lg:col-span-2' : 'lg:col-span-3'}>
            <AuditLogTable
              key={refreshKey}
              filters={filters}
              onEventSelect={handleEventSelect}
              selectedEventId={selectedEvent?.request_id}
            />
          </div>

          {/* Event Detail Panel */}
          {selectedEvent && (
            <div className="lg:col-span-1">
              <div className="sticky top-8">
                <AuditEventDetail
                  event={selectedEvent}
                  onClose={handleCloseDetail}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen bg-gray-50 dark:bg-gray-900">
      <div className="max-w-7xl mx-auto px-4 sm:px-6 lg:px-8 py-8">
        {/* Page Header */}
        <div className="mb-6 flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-gray-900 dark:text-gray-100">
              Audit Logs
            </h1>
            <p className="mt-1 text-sm text-gray-600 dark:text-gray-400">
              View and search system audit events for compliance and security monitoring.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <button
              onClick={() => handleExport('jsonl')}
              className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors"
              title="Export as JSONL"
            >
              <ArrowDownTrayIcon className="h-4 w-4" />
              <span>JSONL</span>
            </button>
            <button
              onClick={() => handleExport('csv')}
              className="flex items-center gap-2 px-3 py-2 text-sm font-medium text-gray-700 dark:text-gray-300 bg-white dark:bg-gray-700 border border-gray-300 dark:border-gray-600 rounded-lg hover:bg-gray-50 dark:hover:bg-gray-600 transition-colors"
              title="Export as CSV"
            >
              <ArrowDownTrayIcon className="h-4 w-4" />
              <span>CSV</span>
            </button>
          </div>
        </div>

        {/* Filter Bar */}
        <div className="mb-6">
          <AuditFilterBar
            filters={filters}
            onFilterChange={handleFilterChange}
            onRefresh={handleRefresh}
          />
        </div>

        {/* Statistics Dashboard */}
        <AuditStatistics stream={filters.stream} username={filters.username} />

        {/* Main Content */}
        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
          {/* Table - takes 2 columns when detail is shown, full width otherwise */}
          <div className={selectedEvent ? 'lg:col-span-2' : 'lg:col-span-3'}>
            <AuditLogTable
              key={refreshKey}
              filters={filters}
              onEventSelect={handleEventSelect}
              selectedEventId={selectedEvent?.request_id}
            />
          </div>

          {/* Event Detail Panel */}
          {selectedEvent && (
            <div className="lg:col-span-1">
              <div className="sticky top-8">
                <AuditEventDetail
                  event={selectedEvent}
                  onClose={handleCloseDetail}
                />
              </div>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

export default AuditLogsPage;
