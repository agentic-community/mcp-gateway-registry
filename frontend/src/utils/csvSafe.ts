import type { ProcessCellForExportParams } from 'ag-grid-community';

/**
 * Spreadsheet formula neutralization for CSV the browser generates.
 *
 * Mirrors `_csv_safe` in registry/audit/routes.py, which guards the server's
 * `/api/audit/export?format=csv`. Keep the two in step: audit records hold
 * caller-authored strings (request paths, usernames, upstream error text), and
 * a cell starting with one of these characters runs as a formula when an admin
 * opens the export in Excel, Calc or Sheets (=HYPERLINK exfiltration, DDE).
 */
const CSV_FORMULA_PREFIXES = ['=', '+', '-', '@', '\t', '\r'];

const _startsWithFormulaPrefix = (text: string): boolean =>
  CSV_FORMULA_PREFIXES.some((prefix) => text.startsWith(prefix));

/**
 * Prefix a single quote when the text would be read as a formula.
 *
 * Checks position 0 and, because spreadsheets trim a cell before deciding,
 * the first character after leading whitespace (" =cmd", "\n=cmd").
 */
export function csvSafe(text: string): string {
  return _startsWithFormulaPrefix(text) || _startsWithFormulaPrefix(text.trimStart())
    ? `'${text}`
    : text;
}

/**
 * AG Grid `processCellCallback` for CSV export: the column's formatted text,
 * neutralized. Numbers are left alone, as on the server, so a negative number
 * stays a number.
 */
export function processCsvCell(params: ProcessCellForExportParams): string {
  const { value } = params;
  if (value == null) return '';
  const text = String(params.formatValue(value) ?? '');
  return typeof value === 'number' ? text : csvSafe(text);
}
