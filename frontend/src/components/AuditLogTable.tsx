import React, { useCallback, useDeferredValue, useEffect, useMemo, useRef, useState } from 'react';
import axios from 'axios';
import clsx from 'clsx';
import { Popover, Transition } from '@headlessui/react';
import { AgGridReact } from 'ag-grid-react';
import {
  CellStyleModule,
  ClientSideRowModelModule,
  ColumnApiModule,
  CsvExportModule,
  DateFilterModule,
  ModuleRegistry,
  NumberFilterModule,
  QuickFilterModule,
  RowApiModule,
  RowSelectionModule,
  TextFilterModule,
  ValidationModule,
} from 'ag-grid-community';
import type {
  CellKeyDownEvent,
  GridApi,
  GridReadyEvent,
  RowClickedEvent,
  RowSelectionOptions,
} from 'ag-grid-community';
import {
  ArrowDownTrayIcon,
  ArrowPathIcon,
  ExclamationTriangleIcon,
  MagnifyingGlassIcon,
  ViewColumnsIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import type { AuditFilters } from './AuditFilterBar';
import type { AuditEvent } from '../types/audit';
import { buildAuditQueryParams } from '../utils/auditQuery';
import { processCsvCell } from '../utils/csvSafe';
import { AUDIT_COLUMNS, DEFAULT_COL_DEF } from './audit/auditColumns';
import type { AuditRow } from './audit/auditColumns';
import 'ag-grid-community/styles/ag-grid.css';
import 'ag-grid-community/styles/ag-theme-quartz.css';
import './audit/auditGrid.css';

ModuleRegistry.registerModules([
  ClientSideRowModelModule,
  QuickFilterModule,
  TextFilterModule,
  NumberFilterModule,
  DateFilterModule,
  CsvExportModule,
  RowSelectionModule,
  CellStyleModule,
  ColumnApiModule,
  RowApiModule,
  // Readable warnings for misconfigured grid options, development builds only.
  ...(import.meta.env.DEV ? [ValidationModule] : []),
]);

/** Server-side maximum for `limit` on /api/audit/events. */
const PAGE_LIMIT = 500;

/** How many matching events to pull into the grid for instant client-side search. */
const WINDOW_OPTIONS = [500, 1000, 2500, 5000];
const DEFAULT_WINDOW = 1000;

const ROW_SELECTION: RowSelectionOptions = {
  mode: 'singleRow',
  checkboxes: false,
  enableClickSelection: true,
};

const NO_ROWS_TEMPLATE =
  '<span class="text-sm text-fg-subtle">No audit events match these filters.</span>';

interface AuditLogTableProps {
  filters: AuditFilters;
  onEventSelect?: (event: AuditEvent) => void;
  selectedEventId?: string;
}

interface ColumnToggle {
  id: string;
  label: string;
  visible: boolean;
}

interface FetchResult {
  events: AuditEvent[];
  total: number;
}


/**
 * Fetch `count` matching events starting at `offset`, newest first. The first
 * page reports the total, which bounds how many further pages (fetched in
 * parallel) are worth requesting.
 */
async function _fetchEvents(
  filters: AuditFilters,
  offset: number,
  count: number,
  signal: AbortSignal,
): Promise<FetchResult> {
  const base = buildAuditQueryParams(filters);
  const fetchPage = async (pageOffset: number, limit: number) => {
    const params = new URLSearchParams(base);
    params.set('limit', String(limit));
    params.set('offset', String(pageOffset));
    params.set('sort_order', '-1');
    const response = await axios.get<{ events?: AuditEvent[]; total?: number }>(
      `/api/audit/events?${params.toString()}`,
      { signal },
    );
    return response.data;
  };

  const first = await fetchPage(offset, Math.min(PAGE_LIMIT, count));
  const total = first.total ?? 0;
  const end = Math.min(offset + count, total);
  const rest: ReturnType<typeof fetchPage>[] = [];
  for (let pageOffset = offset + PAGE_LIMIT; pageOffset < end; pageOffset += PAGE_LIMIT) {
    rest.push(fetchPage(pageOffset, Math.min(PAGE_LIMIT, end - pageOffset)));
  }
  const pages = await Promise.all(rest);
  return { events: [first, ...pages].flatMap((page) => page.events ?? []), total };
}


/**
 * Append events as grid rows with a stable id, dropping any already loaded.
 * New events written between two "load more" requests shift server offsets,
 * so the same record can come back twice.
 */
function _appendRows(existing: AuditRow[], events: AuditEvent[]): AuditRow[] {
  const seen = new Set(existing.map((row) => row._rowId));
  const added: AuditRow[] = [];
  for (const event of events) {
    const rowId = event._id ?? `${event.log_type}|${event.request_id}|${event.timestamp}`;
    if (!seen.has(rowId)) {
      seen.add(rowId);
      added.push({ ...event, _rowId: rowId });
    }
  }
  return existing.concat(added);
}


function _errorMessage(err: unknown): string {
  if (axios.isAxiosError(err)) {
    if (err.response?.status === 403) {
      return 'Access denied. Admin permissions required.';
    }
    const detail = (err.response?.data as { detail?: unknown } | undefined)?.detail;
    if (typeof detail === 'string') {
      return detail;
    }
  }
  return 'Failed to load audit events';
}


function _readColumns(api: GridApi<AuditRow>): ColumnToggle[] {
  return (api.getColumns() ?? []).map((column) => ({
    id: column.getColId(),
    label: column.getColDef().headerName ?? column.getColId(),
    visible: column.isVisible(),
  }));
}


const AuditLogTable: React.FC<AuditLogTableProps> = ({
  filters,
  onEventSelect,
  selectedEventId,
}) => {
  const gridApiRef = useRef<GridApi<AuditRow> | null>(null);
  const searchRef = useRef<HTMLInputElement>(null);
  const abortRef = useRef<AbortController | null>(null);

  const [rows, setRows] = useState<AuditRow[]>([]);
  const [total, setTotal] = useState(0);
  const [loading, setLoading] = useState(false);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [windowSize, setWindowSize] = useState(DEFAULT_WINDOW);
  const [search, setSearch] = useState('');
  const deferredSearch = useDeferredValue(search);
  const [displayedCount, setDisplayedCount] = useState(0);
  const [columns, setColumns] = useState<ColumnToggle[]>([]);
  const [hasColumnFilters, setHasColumnFilters] = useState(false);

  const columnDefs = useMemo(() => AUDIT_COLUMNS[filters.stream], [filters.stream]);

  const load = useCallback(
    async (append: boolean) => {
      abortRef.current?.abort();
      const controller = new AbortController();
      abortRef.current = controller;
      const offset = append ? rows.length : 0;

      if (append) setLoadingMore(true);
      else setLoading(true);
      setError(null);

      try {
        const result = await _fetchEvents(filters, offset, windowSize, controller.signal);
        setRows((prev) => _appendRows(append ? prev : [], result.events));
        setTotal(result.total);
      } catch (err) {
        if (axios.isCancel(err)) return;
        console.error('Failed to fetch audit events:', err);
        setError(_errorMessage(err));
        if (!append) setRows([]);
      } finally {
        if (abortRef.current === controller) {
          setLoading(false);
          setLoadingMore(false);
        }
      }
    },
    [filters, windowSize, rows.length],
  );

  // Reload from the top whenever the server-side filters or the window change.
  useEffect(() => {
    void load(false);
    // `load` changes with rows.length; only filter/window changes should refetch.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [filters, windowSize]);

  useEffect(() => () => abortRef.current?.abort(), []);

  // "/" focuses the search box from anywhere on the page.
  useEffect(() => {
    const handleKeyDown = (event: KeyboardEvent) => {
      const target = event.target as HTMLElement | null;
      const typing = target && (target.isContentEditable || /^(INPUT|TEXTAREA|SELECT)$/.test(target.tagName));
      if (event.key === '/' && !typing) {
        event.preventDefault();
        searchRef.current?.focus();
      }
    };
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, []);

  // Closing the detail panel clears the grid's selection highlight.
  useEffect(() => {
    if (!selectedEventId) {
      gridApiRef.current?.deselectAll();
    }
  }, [selectedEventId]);

  const syncCounts = useCallback(() => {
    const api = gridApiRef.current;
    if (!api) return;
    setDisplayedCount(api.getDisplayedRowCount());
    setHasColumnFilters(api.isColumnFilterPresent());
  }, []);

  const handleGridReady = useCallback(
    (event: GridReadyEvent<AuditRow>) => {
      gridApiRef.current = event.api;
      setColumns(_readColumns(event.api));
      syncCounts();
    },
    [syncCounts],
  );

  const handleRowClicked = useCallback(
    (event: RowClickedEvent<AuditRow>) => {
      if (event.data) onEventSelect?.(event.data);
    },
    [onEventSelect],
  );

  const handleCellKeyDown = useCallback(
    (event: CellKeyDownEvent<AuditRow>) => {
      const key = (event.event as KeyboardEvent | null | undefined)?.key;
      if (key === 'Enter' && event.data) {
        event.node.setSelected(true);
        onEventSelect?.(event.data);
      }
    },
    [onEventSelect],
  );

  const toggleColumn = (id: string, visible: boolean) => {
    gridApiRef.current?.setColumnsVisible([id], visible);
  };

  const resetView = () => {
    const api = gridApiRef.current;
    setSearch('');
    if (!api) return;
    api.setFilterModel(null);
    api.resetColumnState();
  };

  const exportView = () => {
    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, '-');
    gridApiRef.current?.exportDataAsCsv({
      fileName: `audit-${filters.stream}-${stamp}.csv`,
      processCellCallback: processCsvCell,
    });
  };

  const loaded = rows.length;
  const canLoadMore = loaded < total;
  const viewIsFiltered = deferredSearch.trim() !== '' || hasColumnFilters;

  return (
    <div className="card overflow-hidden">
      {/* Toolbar */}
      <div className="flex flex-wrap items-center gap-2 border-b border-line p-3">
        <div className="relative min-w-[16rem] flex-1">
          <MagnifyingGlassIcon className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-fg-subtle" />
          <input
            ref={searchRef}
            type="search"
            value={search}
            onChange={(event) => setSearch(event.target.value)}
            onKeyDown={(event) => {
              if (event.key === 'Escape') setSearch('');
            }}
            placeholder="Search loaded events: user, path, tool, IP, request ID..."
            aria-label="Search loaded audit events"
            className="input pl-9 pr-16"
          />
          <kbd className={clsx('pointer-events-none absolute right-3 top-1/2 -translate-y-1/2 rounded border border-line bg-surface-muted px-1.5 font-mono text-[10px] text-fg-subtle', search ? 'hidden' : 'hidden sm:block')}>
            /
          </kbd>
        </div>

        <Popover className="relative">
          <Popover.Button className="btn-secondary px-3" title="Show or hide columns">
            <ViewColumnsIcon className="h-4 w-4" />
            <span className="hidden sm:inline">Columns</span>
          </Popover.Button>
          <Transition
            enter="transition ease-out duration-100"
            enterFrom="opacity-0 translate-y-1"
            enterTo="opacity-100 translate-y-0"
            leave="transition ease-in duration-75"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <Popover.Panel className="absolute right-0 z-30 mt-2 max-h-80 w-56 overflow-y-auto rounded-xl border border-line bg-surface-overlay p-1.5 shadow-lg">
              {columns.map((column) => (
                <label
                  key={column.id}
                  className="flex cursor-pointer items-center gap-2.5 rounded-lg px-2.5 py-1.5 text-sm text-fg hover:bg-surface-hover"
                >
                  <input
                    type="checkbox"
                    checked={column.visible}
                    onChange={(event) => toggleColumn(column.id, event.target.checked)}
                    className="h-4 w-4 rounded border-line-strong bg-surface text-accent focus:ring-focus"
                  />
                  {column.label}
                </label>
              ))}
            </Popover.Panel>
          </Transition>
        </Popover>

        <button
          type="button"
          onClick={resetView}
          disabled={!viewIsFiltered}
          className="btn-ghost px-3"
          title="Clear search and column filters, restore default columns and sort"
        >
          <XMarkIcon className="h-4 w-4" />
          <span className="hidden sm:inline">Reset view</span>
        </button>

        <button
          type="button"
          onClick={exportView}
          disabled={displayedCount === 0}
          className="btn-ghost px-3"
          title="Download the rows and columns currently shown as CSV"
        >
          <ArrowDownTrayIcon className="h-4 w-4" />
          <span className="hidden sm:inline">Export view</span>
        </button>
      </div>

      {error ? (
        <div className="flex items-center gap-2 bg-danger-soft px-4 py-3 text-sm text-danger-soft-fg" role="alert">
          <ExclamationTriangleIcon className="h-5 w-5 shrink-0" />
          <span className="flex-1">{error}</span>
          <button type="button" onClick={() => void load(false)} className="btn-ghost px-2 py-1 text-xs">
            Retry
          </button>
        </div>
      ) : null}

      {/* Grid */}
      <div className="ag-theme-quartz audit-grid h-[min(70vh,44rem)] min-h-[26rem]">
        <AgGridReact<AuditRow>
          key={filters.stream}
          theme="legacy"
          rowData={rows}
          columnDefs={columnDefs}
          defaultColDef={DEFAULT_COL_DEF}
          getRowId={(params) => params.data._rowId}
          quickFilterText={deferredSearch}
          cacheQuickFilter
          includeHiddenColumnsInQuickFilter
          rowSelection={ROW_SELECTION}
          loading={loading}
          overlayNoRowsTemplate={NO_ROWS_TEMPLATE}
          animateRows={false}
          enableCellTextSelection
          ensureDomOrder
          onGridReady={handleGridReady}
          onRowClicked={handleRowClicked}
          onCellKeyDown={handleCellKeyDown}
          onModelUpdated={syncCounts}
          onFilterChanged={syncCounts}
          onDisplayedColumnsChanged={(event) => setColumns(_readColumns(event.api))}
        />
      </div>

      {/* Footer: what is loaded vs. what matches on the server */}
      <div className="flex flex-wrap items-center justify-between gap-3 border-t border-line bg-surface-muted px-4 py-2.5 text-sm text-fg-muted">
        <div className="tabular-nums" aria-live="polite">
          {viewIsFiltered ? (
            <>
              <span className="font-semibold text-fg">{displayedCount.toLocaleString()}</span> of{' '}
              {loaded.toLocaleString()} loaded events match
            </>
          ) : (
            <>
              <span className="font-semibold text-fg">{loaded.toLocaleString()}</span> loaded
            </>
          )}
          <span className="text-fg-subtle">
            {' '}
            &middot; {total.toLocaleString()} match the server filters
          </span>
        </div>

        <div className="flex items-center gap-2">
          <label htmlFor="audit-window" className="text-xs text-fg-subtle">
            Load per batch
          </label>
          <select
            id="audit-window"
            value={windowSize}
            onChange={(event) => setWindowSize(Number(event.target.value))}
            className="input w-auto py-1 pr-8 text-xs"
          >
            {WINDOW_OPTIONS.map((size) => (
              <option key={size} value={size}>
                {size.toLocaleString()}
              </option>
            ))}
          </select>
          <button
            type="button"
            onClick={() => void load(true)}
            disabled={!canLoadMore || loading || loadingMore}
            className={clsx('btn-secondary px-3 py-1 text-xs', !canLoadMore && 'hidden')}
          >
            <ArrowPathIcon className={clsx('h-3.5 w-3.5', loadingMore && 'animate-spin')} />
            Load {Math.min(windowSize, total - loaded).toLocaleString()} more
          </button>
        </div>
      </div>
    </div>
  );
};

export default AuditLogTable;
