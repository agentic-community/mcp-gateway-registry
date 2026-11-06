# MCP Gateway Registry Frontend

The registry web UI: React 18, TypeScript, Vite and Tailwind CSS 3.

## Development

You need Node.js 20 or newer and a registry backend on `http://localhost:7860`. The Vite dev server proxies `/api`, `/auth`, `/oauth2` and `/.well-known` to that address (see `vite.config.ts`).

```bash
npm ci
npm start          # dev server on http://localhost:3000
```

| Script | What it does |
|---|---|
| `npm start` | Vite dev server with hot reload |
| `npm run build` | Production build into `build/` (the registry serves `build/static/`) |
| `npm run preview` | Serve the production build locally |
| `npm run lint` | ESLint over `src/` |
| `npm test` | Jest unit tests (`src/**/__tests__/`) |

Keep `assetsDir: 'static'` and the output names in `vite.config.ts`: `registry/main.py` rewrites `/static/` paths in `index.html` when the registry runs under a `ROOT_PATH`.

## Theming

Every color comes from a CSS custom property declared in `src/index.css`, and `tailwind.config.cjs` maps Tailwind color names onto those properties. Light mode, dark mode and the high-contrast palette swap variable values at runtime, so no component needs to know which mode is active.

There are two layers of tokens:

- The palette ramps `--gray-*` and `--primary-*` back Tailwind's `gray`, `primary` and `purple` classes (`purple` is an alias of the brand ramp). Existing components style each mode with `dark:` variants on these, as before.
- The semantic tokens change value per mode, so a class needs no `dark:` twin. Use them in new code:

| Tailwind class | Use it for |
|---|---|
| `bg-canvas` | Page background |
| `bg-surface`, `bg-surface-muted` | Cards; inset areas and table headers |
| `bg-surface-hover`, `bg-surface-overlay` | Hovered rows and buttons; menus, modals and popovers |
| `border-line`, `border-line-strong` | Dividers and card borders; input borders |
| `text-fg`, `text-fg-muted`, `text-fg-subtle` | Body text; secondary text; placeholders and captions |
| `bg-accent`, `text-accent-text`, `bg-accent-soft` + `text-accent-soft-fg` | Solid brand buttons; links; selected items |
| `bg-danger`, `bg-danger-soft` + `text-danger-soft-fg` | Destructive buttons; error banners |

`src/index.css` also defines the shared component classes `.btn-primary`, `.btn-secondary`, `.btn-danger`, `.btn-ghost`, `.btn-icon`, `.card`, `.input` and `.label`. Reach for these before writing a new button or input by hand.

The header's appearance menu offers Light, Dark and System, plus a High contrast toggle. `src/contexts/ThemeContext.tsx` stores an explicit choice in `localStorage` (`theme`, `themeName`). System follows the OS setting and stores nothing. A small inline script in `index.html` applies the saved mode before first paint, so a light-mode user never sees a dark flash. It shares the storage keys with `ThemeContext.tsx`, so change both together. The registry ships no Content-Security-Policy; if you add one, allow that script by hash.

To add a palette theme, add an `html.<name>` block to `src/index.css` that overrides the variables, then list the name in `AVAILABLE_THEMES`.

## Audit log grid

Settings > Audit > Audit Logs shows events in an [AG Grid Community](https://www.ag-grid.com/react-data-grid/) table (`src/components/AuditLogTable.tsx`).

- The filter bar above the grid sends stream, date range, user, operation, resource and status filters to `/api/audit/events`.
- The grid then loads a batch of matching events (1,000 by default; 500 to 5,000 via "Load per batch") in parallel 500-event pages, newest first. "Load more" appends the next batch.
- The search box filters the loaded events as you type, across every column including hidden ones (request ID, client IP, user agent). Press `/` to focus it.
- Each column has a filter row and sorts on click. Columns resize, and you can show or hide them from the Columns menu.
- "Export view" downloads the rows and columns on screen as CSV. The JSONL and CSV buttons at the top of the page export every event that matches the server filters.
- Clicking a row opens the event in a side panel; `Escape` closes it.

The grid registers only the AG Grid modules it uses (`ModuleRegistry.registerModules` at the top of `AuditLogTable.tsx`). If you use a new grid feature, register its module there; development builds log a warning that names the missing module. Grid colors come from `src/components/audit/auditGridTheme.ts`, which points every AG Grid theme parameter at the CSS tokens above. Per-stream column definitions live in `src/components/audit/auditColumns.tsx`.

## Code splitting

`src/App.tsx` loads each page with `React.lazy`, and `SettingsPage.tsx` lazy-loads the audit page, so a visitor downloads AG Grid only when they open Audit Logs.

## Project structure

```
frontend/
├── index.html         # App shell and pre-paint theme script
├── src/
│   ├── components/    # Shared and feature components (audit/ holds the grid config)
│   ├── contexts/      # Auth, theme and server stats providers
│   ├── hooks/         # Data and UI hooks (useMediaQuery, useEscapeKey, ...)
│   ├── pages/         # Route-level pages, lazy-loaded
│   ├── theme/         # Entity accent classes for cards
│   ├── types/         # Shared TypeScript types
│   ├── utils/         # Helpers (safeUrl, auditQuery, ...)
│   └── index.css      # Design tokens and component classes
├── e2e/               # Playwright tests (run against http://localhost)
└── tailwind.config.cjs
```
