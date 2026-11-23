import {
  ClientSideRowModelModule,
  CsvExportModule,
  DateFilterModule,
  ModuleRegistry,
  createGrid,
} from 'ag-grid-community';
import type { ColDef } from 'ag-grid-community';
import { AUDIT_COLUMNS } from '../../components/audit/auditColumns';
import type { AuditRow } from '../../components/audit/auditColumns';
import { csvSafe, processCsvCell } from '../csvSafe';

ModuleRegistry.registerModules([ClientSideRowModelModule, CsvExportModule, DateFilterModule]);

describe('csvSafe', () => {
  it.each([
    ['=HYPERLINK("https://x.example","y")'],
    ['+1+1'],
    ['-2+3'],
    ['@SUM(A1)'],
    ['\t=cmd'],
    ['\r=cmd'],
    [' =cmd'],
    ['\n=cmd'],
    ['  @x'],
  ])('neutralizes %j', (value) => {
    expect(csvSafe(value)).toBe(`'${value}`);
  });

  it.each([['/api/servers'], ['alice'], ['a=b'], [''], ['   '], ['200 OK']])(
    'leaves %j unchanged',
    (value) => {
      expect(csvSafe(value)).toBe(value);
    },
  );
});

interface Row {
  user: string;
  path: string;
  status: number;
  duration: number;
}

/** Export rows through a real grid, as the audit page's Export view does. */
function _exportCsv(rows: Row[]): string[] {
  const columnDefs: ColDef<Row>[] = [
    { colId: 'user', valueGetter: (p) => p.data?.user },
    { field: 'path' },
    { field: 'status' },
    { field: 'duration', valueFormatter: (p) => `${p.value} ms` },
  ];
  const host = document.createElement('div');
  document.body.appendChild(host);
  const api = createGrid(host, { columnDefs, rowData: rows });
  const csv = api.getDataAsCsv({ processCellCallback: processCsvCell }) ?? '';
  api.destroy();
  host.remove();
  return csv.split(/\r?\n/).slice(1);
}

describe('processCsvCell', () => {
  it('neutralizes formula text from row data in the exported CSV', () => {
    const [line] = _exportCsv([
      { user: '=HYPERLINK("https://x.example","y")', path: ' @cmd', status: 200, duration: 5 },
    ]);

    expect(line).toBe(`"'=HYPERLINK(""https://x.example"",""y"")","' @cmd","200","5 ms"`);
  });

  it('keeps formatted values and leaves numbers alone', () => {
    const [line] = _exportCsv([{ user: 'alice', path: '/api/x', status: -1, duration: 12 }]);

    expect(line).toBe('"alice","/api/x","-1","12 ms"');
  });

  it("exports the audit Time column's raw ISO value, not the on-screen format", () => {
    const [timestampColumn] = AUDIT_COLUMNS.registry_api;
    const host = document.createElement('div');
    document.body.appendChild(host);
    const api = createGrid(host, {
      columnDefs: [timestampColumn],
      rowData: [{ _rowId: '1', timestamp: '2026-11-07T10:30:12.123000+00:00' } as AuditRow],
    });
    const csv = api.getDataAsCsv({ processCellCallback: processCsvCell }) ?? '';
    api.destroy();
    host.remove();

    expect(csv.split(/\r?\n/)[1]).toBe('"2026-11-07T10:30:12.123000+00:00"');
  });
});
