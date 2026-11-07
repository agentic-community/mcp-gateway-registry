# MCP Gateway Registry Frontend

The registry web UI: React 18, TypeScript, Vite and Tailwind CSS 3.

## Development

You need Node.js 20 or newer and a registry backend on `http://localhost:7860`. The Vite dev server proxies `/api`, `/auth`, `/oauth2`, `/.well-known` and `/branding` to that address (see `vite.config.ts`).

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

- The palette ramps back Tailwind's color names: `gray` for neutrals, `primary` for the brand (`purple` is an alias), `success`, `warning`, `danger` and `info` for status, and `server`, `agent`, `skill`, `virtual` and `custom` for the registry entity types. Components style each mode with `dark:` variants on these. Raw Tailwind hues such as `text-red-600` or `bg-cyan-50` cannot be rebranded, so an ESLint rule (`no-restricted-syntax` in `package.json`) rejects them: write `text-danger-600` or `bg-agent-50` instead.
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

The header's appearance menu offers Light, Dark and System, any themes the deployment's branding defines, and a High contrast toggle that combines with all of them. `src/contexts/ThemeContext.tsx` stores explicit choices in `localStorage` (`theme`, `themeName`, `contrast`). System follows the OS setting and stores nothing. A small inline script in `index.html` applies the saved mode, theme and contrast before first paint, so a light-mode or branded user never sees the default look flash. It shares the storage keys with `ThemeContext.tsx` and `BrandingContext.tsx`, so change them together. The registry ships no Content-Security-Policy; if you add one, allow that script by hash.

## Branding

Deployments reskin the UI at runtime from a branding folder that the registry serves at `{ROOT_PATH}/branding/`. [docs/ui-branding.md](../docs/ui-branding.md) is the operator guide. The frontend side:

- `src/contexts/BrandingContext.tsx` fetches `branding/manifest.json` once, validates it with `src/branding/manifest.ts`, applies the favicon and caches the result in `localStorage` (`branding`) for the pre-paint script.
- `index.html` links `branding/theme.css` at the start of `<body>`, after the app's stylesheet, so its variable overrides win.
- `Layout.tsx` shows the branded logos, and `ThemeContext.tsx` applies the manifest's default mode, mode lock, theme list and theme color.

The CSS variables in `src/index.css` are the public contract `theme.css` overrides. Adding one is fine. Renaming or removing one breaks deployments that set it, so record it in the release notes and in docs/ui-branding.md. Some changes alter what an override does without renaming anything, and those need the docs updated too:

- Giving a role variable a fixed value instead of a `var(--scale-step)` reference, or the reverse, changes the "Values that do not follow the color scales" table.
- Moving the high-contrast rules, or changing which steps they replace, changes the "High contrast" section.
- Switching a component to `rounded-control`/`rounded-card` (or the `.btn-*`, `.input` and `.card` classes), or away from them, changes which corners `--radius-*` reaches, listed under "Corners and fonts".
- Changing a soft-tint value in `html.dark` means updating the blend constants in `scripts/generate-ui-theme.py` (`DARK_SURFACE`, `DARK_SOFT_BLEND_ALPHA`) to match.

## Audit log grid

Settings > Audit > Audit Logs shows events in an [AG Grid Community](https://www.ag-grid.com/react-data-grid/) table (`src/components/AuditLogTable.tsx`).

- The filter bar above the grid sends stream, date range, user, operation, resource and status filters to `/api/audit/events`.
- The grid then loads a batch of matching events (1,000 by default; 500 to 5,000 via "Per batch") in parallel 500-event pages. "Load" picks which end the server starts from, newest first (the default) or oldest first, and "Load more" appends the next batch in that order. Sorting a column only reorders the loaded events, so reaching the oldest events of a large range takes "Oldest first" rather than a column sort.
- The search box filters the loaded events as you type, across every column including hidden ones (request ID, client IP, user agent). Press `/` to focus it.
- Each column has a filter row and sorts on click. Columns resize, and you can show or hide them from the Columns menu.
- "Export view" downloads the rows and columns on screen as CSV. Time cells export as the raw ISO 8601 value, with year and UTC offset, rather than the on-screen format. The JSONL and CSV buttons at the top of the page export every event that matches the server filters.
- Clicking a row opens the event in a side panel; `Escape` closes it.

The grid registers only the AG Grid modules it uses (`ModuleRegistry.registerModules` at the top of `AuditLogTable.tsx`). If you use a new grid feature, register its module there; development builds log a warning that names the missing module. Per-stream column definitions live in `src/components/audit/auditColumns.tsx`.

The grid is styled with AG Grid's static stylesheets (`ag-grid.css` and the Quartz theme), imported by `AuditLogTable.tsx` with `theme="legacy"`, rather than AG Grid's JavaScript theming API. The JavaScript API injects `<style>` elements at runtime, which a Content-Security-Policy without `'unsafe-inline'` blocks; static files need only `style-src 'self'`. `src/components/audit/auditGrid.css` maps AG Grid's `--ag-*` variables onto the design tokens, so the grid follows the mode, high contrast and branding themes. Keep it that way: do not pass a `Theme` object to the grid.

## Code splitting

`src/App.tsx` loads each page with `React.lazy`, and `SettingsPage.tsx` lazy-loads the audit page, so a visitor downloads AG Grid only when they open Audit Logs.

The lazy pages work under a `ROOT_PATH` because `vite.config.ts` sets `experimental.renderBuiltUrl` to emit relative URLs in JS and CSS: a page chunk, its CSS and the fonts load from wherever the entry script came from. `index.html` keeps absolute `/static/` URLs, which `registry/main.py` rewrites under the prefix. Do not change `base` to `'./'`; that rewrite and deep links rely on absolute URLs in `index.html`.

`src/components/RouteErrorBoundary.tsx` wraps the routes. If a page chunk fails to download, typically in a tab left open across an upgrade whose old chunk names no longer exist, it shows a reload prompt instead of a blank page.

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
