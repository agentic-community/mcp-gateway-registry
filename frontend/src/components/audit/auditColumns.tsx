import React from 'react';
import clsx from 'clsx';
import type { ColDef, ICellRendererParams, ValueFormatterParams } from 'ag-grid-community';
import type { AuditEvent, AuditStream } from '../../types/audit';

/** A loaded audit event plus the stable grid row id assigned at fetch time. */
export interface AuditRow extends AuditEvent {
  _rowId: string;
}

type Renderer = ICellRendererParams<AuditRow>;

const PILL = 'inline-flex items-center rounded-md px-1.5 py-0.5 text-xs font-medium tabular-nums';

const TONE = {
  success: 'bg-success-50 text-success-700 ring-1 ring-inset ring-success-600/20 dark:bg-success-500/10 dark:text-success-300 dark:ring-success-400/20',
  info: 'bg-info-50 text-info-700 ring-1 ring-inset ring-info-600/20 dark:bg-info-500/10 dark:text-info-300 dark:ring-info-400/20',
  warning: 'bg-warning-50 text-warning-800 ring-1 ring-inset ring-warning-600/20 dark:bg-warning-500/10 dark:text-warning-300 dark:ring-warning-400/20',
  danger: 'bg-danger-50 text-danger-700 ring-1 ring-inset ring-danger-600/20 dark:bg-danger-500/10 dark:text-danger-300 dark:ring-danger-400/20',
  neutral: 'bg-gray-100 text-gray-700 ring-1 ring-inset ring-gray-500/20 dark:bg-gray-500/10 dark:text-gray-300 dark:ring-gray-400/20',
  accent: 'bg-accent-soft text-accent-soft-fg ring-1 ring-inset ring-primary-500/20',
} as const;

type Tone = keyof typeof TONE;

const METHOD_TONE: Record<string, string> = {
  GET: 'text-info-600 dark:text-info-400',
  POST: 'text-success-600 dark:text-success-400',
  PUT: 'text-warning-600 dark:text-warning-400',
  PATCH: 'text-warning-600 dark:text-warning-400',
  DELETE: 'text-danger-600 dark:text-danger-400',
};

const MCP_STATUS_TONE: Record<string, Tone> = {
  success: 'success',
  error: 'danger',
  timeout: 'warning',
};

const AUTH_DECISION_TONE: Record<string, Tone> = {
  ALLOW: 'success',
  DENY: 'danger',
  NOT_REQUIRED: 'neutral',
};


function _httpStatusTone(code: number): Tone {
  if (code >= 500) return 'danger';
  if (code >= 400) return 'warning';
  if (code >= 300) return 'info';
  if (code >= 200) return 'success';
  return 'neutral';
}


function _pill(value: React.ReactNode, tone: Tone): React.ReactElement {
  return <span className={clsx(PILL, TONE[tone])}>{value}</span>;
}


function _formatTimestamp(params: ValueFormatterParams<AuditRow, string>): string {
  if (!params.value) return '';
  const date = new Date(params.value);
  if (Number.isNaN(date.getTime())) return params.value;
  return date.toLocaleString(undefined, {
    month: 'short',
    day: '2-digit',
    hour: '2-digit',
    minute: '2-digit',
    second: '2-digit',
  });
}


function _formatDuration(params: ValueFormatterParams<AuditRow, number>): string {
  return params.value === null || params.value === undefined ? '' : `${params.value.toFixed(1)} ms`;
}


/** Date filter compares the calendar day of an ISO timestamp. */
function _compareDay(filterDate: Date, cellValue: string | null): number {
  if (!cellValue) return -1;
  const cell = new Date(cellValue);
  const day = new Date(cell.getFullYear(), cell.getMonth(), cell.getDate()).getTime();
  return day === filterDate.getTime() ? 0 : day < filterDate.getTime() ? -1 : 1;
}


function _resourceLabel(type?: string, id?: string): string {
  if (!type) return '';
  return id ? `${type}/${id}` : type;
}


const UserCell: React.FC<Renderer> = ({ value, data }) => (
  <span className="flex min-w-0 items-center gap-1.5">
    <span className="truncate">{value || '-'}</span>
    {data?.identity?.is_admin && _pill('Admin', 'accent')}
  </span>
);

const MethodCell: React.FC<Renderer> = ({ value }) =>
  value ? (
    <span className={clsx('font-mono text-xs font-semibold', METHOD_TONE[String(value).toUpperCase()] ?? 'text-fg-muted')}>
      {value}
    </span>
  ) : null;

const MonoCell: React.FC<Renderer> = ({ value }) =>
  value ? <span className="font-mono text-xs">{value}</span> : null;

const HttpStatusCell: React.FC<Renderer> = ({ value }) =>
  typeof value === 'number' ? _pill(value, _httpStatusTone(value)) : null;

const McpStatusCell: React.FC<Renderer> = ({ value }) =>
  value ? _pill(value, MCP_STATUS_TONE[String(value).toLowerCase()] ?? 'neutral') : null;

const AuthDecisionCell: React.FC<Renderer> = ({ value }) =>
  value ? _pill(value, AUTH_DECISION_TONE[String(value)] ?? 'neutral') : null;

const OutcomeCell: React.FC<Renderer> = ({ value }) =>
  value ? _pill(value, value === 'success' ? 'success' : 'danger') : null;

const TokenKindCell: React.FC<Renderer> = ({ value }) =>
  value ? _pill(value, value === 'resource' ? 'info' : 'neutral') : null;


/** Defaults applied to every audit column. */
export const DEFAULT_COL_DEF: ColDef<AuditRow> = {
  sortable: true,
  resizable: true,
  filter: 'agTextColumnFilter',
  floatingFilter: true,
  suppressHeaderMenuButton: true,
  minWidth: 90,
  flex: 1,
};

const TIMESTAMP_COLUMN: ColDef<AuditRow> = {
  colId: 'timestamp',
  field: 'timestamp',
  headerName: 'Time',
  sort: 'desc',
  pinned: 'left',
  flex: 0,
  width: 180,
  filter: 'agDateColumnFilter',
  // The server filter bar owns date ranges; the header menu still offers a day filter.
  floatingFilter: false,
  suppressHeaderMenuButton: false,
  filterParams: { comparator: _compareDay },
  valueFormatter: _formatTimestamp,
  // Search matches the on-screen format; CSV export gets the raw ISO 8601 value
  // (year and UTC offset included), see processCsvCell.
  useValueFormatterForExport: false,
  getQuickFilterText: (params) => _formatTimestamp(params as ValueFormatterParams<AuditRow, string>),
  cellClass: 'tabular-nums',
};

const DURATION_PROPS: Partial<ColDef<AuditRow>> = {
  headerName: 'Duration',
  flex: 0,
  width: 120,
  filter: 'agNumberColumnFilter',
  valueFormatter: _formatDuration,
  type: 'rightAligned',
  cellClass: 'tabular-nums',
};

const REQUEST_ID_COLUMN: ColDef<AuditRow> = {
  colId: 'request_id',
  field: 'request_id',
  headerName: 'Request ID',
  hide: true,
  cellRenderer: MonoCell,
};

const REGISTRY_COLUMNS: ColDef<AuditRow>[] = [
  TIMESTAMP_COLUMN,
  { colId: 'user', headerName: 'User', valueGetter: (p) => p.data?.identity?.username, cellRenderer: UserCell, minWidth: 160 },
  { colId: 'method', headerName: 'Method', valueGetter: (p) => p.data?.request?.method, cellRenderer: MethodCell, flex: 0, width: 110 },
  { colId: 'path', headerName: 'Path', valueGetter: (p) => p.data?.request?.path, cellRenderer: MonoCell, minWidth: 200, flex: 2 },
  { colId: 'operation', headerName: 'Operation', valueGetter: (p) => p.data?.action?.operation },
  {
    colId: 'resource',
    headerName: 'Resource',
    valueGetter: (p) => _resourceLabel(p.data?.action?.resource_type, p.data?.action?.resource_id),
    minWidth: 160,
  },
  {
    colId: 'status',
    headerName: 'Status',
    valueGetter: (p) => p.data?.response?.status_code,
    cellRenderer: HttpStatusCell,
    filter: 'agNumberColumnFilter',
    flex: 0,
    width: 110,
  },
  { colId: 'duration', valueGetter: (p) => p.data?.response?.duration_ms, ...DURATION_PROPS },
  { colId: 'auth_decision', headerName: 'Auth', valueGetter: (p) => p.data?.authorization?.decision, cellRenderer: AuthDecisionCell, flex: 0, width: 130 },
  { colId: 'client_ip', headerName: 'Client IP', valueGetter: (p) => p.data?.request?.client_ip, cellRenderer: MonoCell, hide: true },
  { colId: 'auth_method', headerName: 'Auth Method', valueGetter: (p) => p.data?.identity?.auth_method, hide: true },
  { colId: 'user_agent', headerName: 'User Agent', valueGetter: (p) => p.data?.request?.user_agent, hide: true, minWidth: 200 },
  REQUEST_ID_COLUMN,
];

const MCP_COLUMNS: ColDef<AuditRow>[] = [
  TIMESTAMP_COLUMN,
  { colId: 'user', headerName: 'User', valueGetter: (p) => p.data?.identity?.username, cellRenderer: UserCell, minWidth: 160 },
  { colId: 'server', headerName: 'MCP Server', valueGetter: (p) => p.data?.mcp_server?.name, minWidth: 150 },
  { colId: 'mcp_method', headerName: 'MCP Method', valueGetter: (p) => p.data?.mcp_request?.method, cellRenderer: MonoCell, minWidth: 140 },
  {
    colId: 'target',
    headerName: 'Tool / Resource',
    valueGetter: (p) => p.data?.mcp_request?.tool_name || p.data?.mcp_request?.resource_uri,
    cellRenderer: MonoCell,
    minWidth: 180,
    flex: 2,
  },
  { colId: 'mcp_status', headerName: 'Status', valueGetter: (p) => p.data?.mcp_response?.status, cellRenderer: McpStatusCell, flex: 0, width: 120 },
  { colId: 'duration', valueGetter: (p) => p.data?.mcp_response?.duration_ms, ...DURATION_PROPS },
  { colId: 'error', headerName: 'Error', valueGetter: (p) => p.data?.mcp_response?.error_message, minWidth: 180 },
  { colId: 'transport', headerName: 'Transport', valueGetter: (p) => p.data?.mcp_request?.transport, hide: true },
  { colId: 'session', headerName: 'Session ID', valueGetter: (p) => p.data?.mcp_request?.mcp_session_id, cellRenderer: MonoCell, hide: true },
  { colId: 'client_ip', headerName: 'Client IP', valueGetter: (p) => p.data?.request?.client_ip, cellRenderer: MonoCell, hide: true },
  REQUEST_ID_COLUMN,
];

const TOKEN_MINT_COLUMNS: ColDef<AuditRow>[] = [
  TIMESTAMP_COLUMN,
  { colId: 'user', headerName: 'User', valueGetter: (p) => p.data?.username || p.data?.username_hash, cellRenderer: UserCell, minWidth: 180 },
  { colId: 'token_kind', field: 'token_kind', headerName: 'Token Kind', cellRenderer: TokenKindCell, flex: 0, width: 130 },
  { colId: 'token_path', field: 'token_path', headerName: 'Path', cellRenderer: MonoCell, minWidth: 180, flex: 2 },
  { colId: 'resource', headerName: 'Resource', valueGetter: (p) => _resourceLabel(p.data?.resource_type, p.data?.resource_id), minWidth: 160 },
  { colId: 'outcome', field: 'outcome', headerName: 'Outcome', cellRenderer: OutcomeCell, flex: 0, width: 120 },
  {
    colId: 'expires',
    headerName: 'Expires',
    valueGetter: (p) => (p.data?.expires_in_seconds ? p.data.expires_in_seconds / 3600 : undefined),
    valueFormatter: (p) => (p.value ? `${Math.round(p.value)}h` : ''),
    filter: 'agNumberColumnFilter',
    type: 'rightAligned',
    flex: 0,
    width: 110,
  },
  { colId: 'failure_reason', field: 'failure_reason', headerName: 'Failure Reason', minWidth: 180 },
  { colId: 'principal', field: 'principal_name', headerName: 'Principal', hide: true },
  { colId: 'object_id', field: 'object_id', headerName: 'Object ID', cellRenderer: MonoCell, hide: true },
  REQUEST_ID_COLUMN,
];

export const AUDIT_COLUMNS: Record<AuditStream, ColDef<AuditRow>[]> = {
  registry_api: REGISTRY_COLUMNS,
  mcp_access: MCP_COLUMNS,
  token_mint: TOKEN_MINT_COLUMNS,
};
