import { themeQuartz } from 'ag-grid-community';

/** Reference an RGB-channel design token from index.css as a CSS color. */
const token = (name: string, alpha?: number) =>
  alpha === undefined ? `rgb(var(--${name}))` : `rgb(var(--${name}) / ${alpha})`;

/**
 * AG Grid theme bound to the app's design tokens. Every color is a CSS
 * variable reference, so the grid follows light, dark and high-contrast mode
 * with no theme switching in JavaScript.
 */
export const auditGridTheme = themeQuartz.withParams({
  browserColorScheme: 'inherit',
  fontFamily: 'inherit',
  fontSize: 13,
  headerFontSize: 12,
  headerFontWeight: 600,
  spacing: 7,
  rowHeight: 40,
  headerHeight: 40,
  wrapperBorderRadius: 0,
  borderRadius: 6,
  backgroundColor: token('color-surface'),
  foregroundColor: token('color-fg'),
  textColor: token('color-fg'),
  subtleTextColor: token('color-fg-subtle'),
  chromeBackgroundColor: token('color-surface-muted'),
  headerBackgroundColor: token('color-surface-muted'),
  headerTextColor: token('color-fg-muted'),
  borderColor: token('color-line'),
  wrapperBorder: false,
  rowBorder: { color: token('color-line') },
  columnBorder: false,
  headerColumnBorder: { color: token('color-line') },
  headerColumnResizeHandleColor: token('color-line-strong'),
  accentColor: token('color-accent'),
  rowHoverColor: token('color-surface-hover', 0.6),
  selectedRowBackgroundColor: token('color-accent-soft'),
  oddRowBackgroundColor: token('color-surface'),
  inputBorder: { color: token('color-line-strong') },
  inputBackgroundColor: token('color-surface'),
  inputFocusBorder: { color: token('color-accent') },
  menuBackgroundColor: token('color-surface-overlay'),
  menuBorder: { color: token('color-line') },
  panelBackgroundColor: token('color-surface-overlay'),
  cellHorizontalPadding: 14,
});
