# Enforce Gateway UI — Frontend Implementation Plan (Phased)
*Created: 2025-12-15*

This plan implements a complete rewrite of the Enforce Gateway UI to meet the requirements in `docs/enforceai-ui-requirements.md`. The backend contract (`docs/enforceai-ui-backend-changes.md`) and backend implementation (`enforceai/plans/plan-enforce-gw-ui-backend-phased.md`) are assumed complete.

## Primary Constraints
- Each phase is scoped to be finishable in a single agent run.
- Each phase adds tests for the phase's scope.
- The full test suite must pass at the end of every phase (`npm run test`).
- Build must succeed after each phase (`npm run build`).

## Current State Analysis
The existing `frontend/` is a React 18.2 + TypeScript + Tailwind CSS app using Create React App with:
- 4 pages: Dashboard, Login, TokenGeneration, OAuthCallback
- No tests
- Basic auth context with OAuth support
- Server and A2A agent cards (registry only)
- No EnforceAI management surfaces

## Target Architecture

### Tech Stack (retained/upgraded)
- **Framework**: React 18.2 + TypeScript 5.x
- **Build**: Vite 5.x (migrate from CRA for faster DX and better testing)
- **Styling**: Tailwind CSS 3.4 + @tailwindcss/forms
- **UI Components**: Headless UI + Heroicons (retained)
- **Routing**: React Router 6.x (retained)
- **Server State**: TanStack Query (React Query) 5.x (new)
- **Testing**: Vitest + React Testing Library + MSW (Mock Service Worker)
- **Form Handling**: React Hook Form + Zod validation (new)
- **Utilities**: clsx, date-fns (new)

### Navigation Structure (11 sections)
```
1. Overview
2. Registry: Servers
3. Registry: A2A Agents
4. EnforceAI: Agents
5. EnforceAI: Credentials
6. Scopes and Policy
7. Tools (Discovery)
8. Audit
9. Admin
10. Settings
11. Help
```

### Folder Structure (target)
```
frontend/
├── src/
│   ├── main.tsx                    # Entry point
│   ├── App.tsx                     # Root with providers
│   ├── router.tsx                  # Route definitions
│   ├── api/                        # API client layer
│   │   ├── client.ts               # Axios instance + interceptors
│   │   ├── registry.ts             # Registry API functions
│   │   ├── enforceai.ts            # EnforceAI API functions
│   │   └── types.ts                # API response types
│   ├── components/                 # Shared UI components
│   │   ├── ui/                     # Primitives (Button, Input, Modal, etc.)
│   │   ├── layout/                 # Layout components (Shell, Nav, Header)
│   │   └── common/                 # Shared domain components
│   ├── features/                   # Feature modules (page + components)
│   │   ├── overview/
│   │   ├── servers/
│   │   ├── agents/
│   │   ├── enforceai-agents/
│   │   ├── credentials/
│   │   ├── scopes/
│   │   ├── tools/
│   │   ├── audit/
│   │   ├── admin/
│   │   ├── settings/
│   │   └── help/
│   ├── hooks/                      # Shared custom hooks
│   ├── contexts/                   # React contexts
│   ├── lib/                        # Utilities
│   └── test/                       # Test utilities and mocks
├── tests/                          # Test files (colocated or here)
├── index.html
├── vite.config.ts
├── vitest.config.ts
├── tailwind.config.js
├── tsconfig.json
└── package.json
```

---

# Phase 1 — Project Foundation + Build System Migration
**Goal**: Migrate from CRA to Vite, establish testing infrastructure, and create foundational utilities.

## Deliverables
1. **Vite migration**:
   - Create new `vite.config.ts` with proxy to backend (`http://localhost:7860`).
   - Update `package.json` scripts: `dev`, `build`, `preview`, `test`, `test:ui`, `lint`, `typecheck`.
   - Create `index.html` (Vite entry point).
   - Update `tsconfig.json` for Vite compatibility.
   - Remove CRA dependencies (`react-scripts`).

2. **Testing infrastructure**:
   - Add Vitest + React Testing Library + jsdom.
   - Add MSW for API mocking.
   - Create `src/test/setup.ts` (test environment setup).
   - Create `src/test/mocks/handlers.ts` (MSW request handlers skeleton).
   - Create `src/test/utils.tsx` (custom render with providers).

3. **API client foundation**:
   - Create `src/api/client.ts` with:
     - Axios instance with `withCredentials: true`.
     - Request interceptor for `X-Request-Id` generation.
     - Response interceptor for error normalization.
     - CSRF token handling (fetch and attach `X-CSRF-Token`).
   - Create `src/api/types.ts` with shared API types.

4. **Core utilities**:
   - Create `src/lib/cn.ts` (classname utility using clsx + tailwind-merge).
   - Create `src/lib/format.ts` (date formatting helpers).
   - Create `src/lib/errors.ts` (error type guards and message extraction).

5. **Dependencies**:
   - Add: `vite`, `@vitejs/plugin-react`, `vitest`, `@testing-library/react`, `@testing-library/jest-dom`, `@testing-library/user-event`, `msw`, `@tanstack/react-query`, `react-hook-form`, `@hookform/resolvers`, `zod`, `date-fns`, `tailwind-merge`, `uuid`.
   - Retain: `react`, `react-dom`, `react-router-dom`, `axios`, `@headlessui/react`, `@heroicons/react`, `clsx`, `tailwindcss`, `@tailwindcss/forms`, `autoprefixer`, `postcss`, `typescript`.
   - Remove: `react-scripts`, `ajv`, `@types/node` (Vite has built-in types).

## Tests to Add
- Unit test: `src/api/client.test.ts` — CSRF token fetch, request ID generation, error normalization.
- Unit test: `src/lib/errors.test.ts` — error extraction from various response shapes.
- Smoke test: Vitest can run and `npm run test` passes.

## Required Test Gate
- `npm run typecheck` (tsc --noEmit) passes.
- `npm run build` succeeds.
- `npm run test` passes (new tests).

---

# Phase 2 — UI Primitives + Theme System
**Goal**: Build reusable UI primitive components and establish the theme/dark mode system.

## Deliverables
1. **UI primitive components** (`src/components/ui/`):
   - `Button.tsx` — primary/secondary/danger/ghost variants, sizes, loading state.
   - `Input.tsx` — text input with label, error state, helper text.
   - `Select.tsx` — dropdown using Headless UI Listbox.
   - `Textarea.tsx` — multiline input.
   - `Checkbox.tsx` — with label.
   - `Badge.tsx` — status badges (success/warning/error/neutral).
   - `Card.tsx` — container with optional header/footer.
   - `Modal.tsx` — dialog wrapper using Headless UI Dialog.
   - `ConfirmDialog.tsx` — destructive action confirmation modal.
   - `Toast.tsx` — notification toast system (context + hook + component).
   - `Spinner.tsx` — loading indicator.
   - `EmptyState.tsx` — empty list/error placeholder.
   - `CopyButton.tsx` — copy-to-clipboard with feedback.
   - `SecretField.tsx` — masked input with reveal toggle.
   - `Tooltip.tsx` — hover tooltip.

2. **Theme context** (`src/contexts/ThemeContext.tsx`):
   - Dark/light mode toggle persisted to localStorage.
   - System preference detection.
   - `useTheme()` hook.

3. **Tailwind config updates**:
   - Ensure dark mode is `class` strategy.
   - Add custom colors for brand consistency.
   - Add animation utilities for transitions.

## Tests to Add
- Unit tests for each UI component:
  - Button: renders variants, handles click, shows loading state.
  - Input: shows error state, calls onChange.
  - Modal: opens/closes, handles escape key.
  - ConfirmDialog: requires confirmation before action.
  - CopyButton: copies to clipboard, shows feedback.
  - SecretField: masks by default, reveals on toggle.
  - Toast: displays and auto-dismisses.

## Required Test Gate
- `npm run typecheck` passes.
- `npm run build` succeeds.
- `npm run test` passes.

---

# Phase 3 — Layout Shell + Navigation + Routing
**Goal**: Build the main application shell, navigation sidebar, and route structure.

## Deliverables
1. **Layout components** (`src/components/layout/`):
   - `AppShell.tsx` — main layout with sidebar + header + content area.
   - `Sidebar.tsx` — navigation with 11 sections, active state, collapse on mobile.
   - `Header.tsx` — user menu, theme toggle, logout.
   - `MobileNav.tsx` — hamburger menu for mobile.
   - `PageHeader.tsx` — page title + breadcrumbs + actions slot.
   - `PageContent.tsx` — content wrapper with consistent padding.

2. **Router setup** (`src/router.tsx`):
   - Define all routes with lazy loading:
     ```
     /                       → Overview
     /servers                → Registry: Servers
     /servers/:path          → Server Details
     /a2a-agents             → Registry: A2A Agents
     /a2a-agents/:path       → A2A Agent Details
     /agents                 → EnforceAI: Agents
     /agents/:agentId        → Agent Details
     /credentials            → EnforceAI: Credentials
     /credentials/api-keys   → API Keys
     /credentials/tokens     → Gateway Tokens
     /scopes                 → Scopes and Policy
     /tools                  → Tools Discovery
     /audit                  → Audit
     /admin                  → Admin
     /admin/users            → Admin: Users
     /admin/users/:userId    → Admin: User Details
     /settings               → Settings
     /help                   → Help
     /login                  → Login (public)
     /auth/callback          → OAuth Callback (public)
     ```
   - `ProtectedRoute` wrapper for authenticated routes.
   - `AdminRoute` wrapper for admin-only routes.

3. **Placeholder pages** (one per feature):
   - Create minimal placeholder components for each route that render page title + "Coming soon".
   - These will be replaced in subsequent phases.

4. **App root** (`src/App.tsx`):
   - Wrap with providers: QueryClientProvider, ThemeProvider, AuthProvider (skeleton), ToastProvider.
   - Render RouterProvider.

## Tests to Add
- Integration test: navigation renders all 11 sections.
- Integration test: clicking nav item navigates to correct route.
- Integration test: protected routes redirect to login when unauthenticated.
- Unit test: Sidebar collapses on mobile.
- Unit test: Header shows user menu when authenticated.

## Required Test Gate
- `npm run typecheck` passes.
- `npm run build` succeeds.
- `npm run test` passes.

---

# Phase 4 — Authentication + Session Management
**Goal**: Implement complete auth flows (OIDC, password), CSRF, session handling.

## Deliverables
1. **Auth context** (`src/contexts/AuthContext.tsx`):
   - State: `user`, `isLoading`, `isAuthenticated`, `isAdmin`.
   - User shape: `{ user_id, email, username, auth_method, is_admin, roles, groups }`.
   - Methods: `login()`, `logout()`, `refreshUser()`.
   - Auto-fetch user on mount via `/api/auth/me`.
   - Handle session expiry (401 response → clear state → redirect to login).

2. **CSRF handling** (in `src/api/client.ts`):
   - Fetch CSRF token from `GET /api/auth/csrf` on app init.
   - Attach `X-CSRF-Token` header to all non-GET requests.
   - Refresh CSRF token on 403 CSRF errors and retry.

3. **Login page** (`src/features/auth/LoginPage.tsx`):
   - OAuth provider selection (fetch from `/api/auth/providers`).
   - Username/password form with validation.
   - Remember me checkbox.
   - Error display for 401/403.
   - Redirect to intended destination after login.

4. **OAuth callback** (`src/features/auth/OAuthCallback.tsx`):
   - Handle OAuth redirect, extract any error params.
   - Fetch user data and redirect to home.

5. **Logout flow**:
   - Call `POST /api/auth/logout`.
   - Clear local state.
   - Redirect to login.

6. **Session display**:
   - Show current user in Header.
   - Show session info (auth method, email) in Settings page.

## Tests to Add
- Unit test: AuthContext initializes, fetches user, handles 401.
- Unit test: CSRF token is attached to POST/PUT/PATCH/DELETE.
- Integration test: Login form submits credentials, handles success/error.
- Integration test: OAuth flow redirects correctly.
- Integration test: Logout clears session and redirects.
- Integration test: Protected route redirects unauthenticated user.

## Required Test Gate
- `npm run typecheck` passes.
- `npm run build` succeeds.
- `npm run test` passes.

---

# Phase 5 — Overview Page
**Goal**: Implement the Overview dashboard with connection status and quick actions.

## Deliverables
1. **Overview page** (`src/features/overview/OverviewPage.tsx`):
   - Connection status cards:
     - Registry reachable (GET `/api/servers`).
     - EnforceAI reachable (GET `/enforceai/agents`).
     - Show success/failure, elapsed time.
   - Summary counts:
     - MCP servers (total, enabled, disabled).
     - A2A agents (total, enabled, disabled).
     - EnforceAI agents (total, active, revoked).
   - Quick action buttons:
     - Create EnforceAI agent → `/agents/new`.
     - Mint token → `/credentials/tokens/mint`.
     - Create API key → `/credentials/api-keys/new`.
     - Revoke all tokens → modal.
   - "Test connection" button with visual feedback.

2. **API hooks**:
   - `useServers()` — fetch registry servers.
   - `useA2AAgents()` — fetch A2A agents.
   - `useEnforceAIAgents()` — fetch EnforceAI agents.
   - `useConnectionTest()` — test connectivity mutation.

3. **Error handling**:
   - Show explicit message if EnforceAI management is not enabled (404 on `/enforceai/*`).

## Tests to Add
- Unit test: Overview renders connection status cards.
- Unit test: Summary counts display correctly.
- Unit test: Quick actions navigate to correct routes.
- Integration test: Connection test shows success/failure states.
- Integration test: Handles EnforceAI not enabled gracefully.

## Required Test Gate
- `npm run typecheck` passes.
- `npm run build` succeeds.
- `npm run test` passes.

---

# Phase 6 — Registry: Servers
**Goal**: Complete MCP server management (list, register, edit, toggle, refresh, details).

## Deliverables
1. **Servers list page** (`src/features/servers/ServersPage.tsx`):
   - Table/grid of servers with columns: name, path, status, tags, last health check, actions.
   - Search/filter by name, path, tags.
   - Filter by enabled/disabled.
   - Actions per row: View details, Edit, Toggle, Refresh.

2. **Server details page** (`src/features/servers/ServerDetailsPage.tsx`):
   - Server info card (name, path, URL, description, tags).
   - Health status with refresh button.
   - Tools list (expandable).
   - Edit and Toggle buttons.

3. **Server forms**:
   - `ServerRegisterModal.tsx` — register new server form.
   - `ServerEditModal.tsx` — edit existing server form.
   - Validation: required fields, URL format.

4. **API hooks**:
   - `useServers()` — list servers with React Query.
   - `useServer(path)` — single server details.
   - `useServerTools(path)` — server tools.
   - `useRegisterServer()` — mutation.
   - `useEditServer()` — mutation.
   - `useToggleServer()` — mutation.
   - `useRefreshServer()` — mutation.

5. **Bulk operations** (if backend supports):
   - Select multiple servers.
   - Bulk toggle, bulk refresh.

## Tests to Add
- Unit test: Server list renders, filters work.
- Unit test: Server card displays all fields.
- Unit test: Register form validates inputs.
- Integration test: Register server flow (mock API).
- Integration test: Toggle server updates UI optimistically.
- Integration test: Refresh server shows loading state.

## Required Test Gate
- `npm run typecheck` passes.
- `npm run build` succeeds.
- `npm run test` passes.

---

# Phase 7 — Registry: A2A Agents
**Goal**: Complete A2A agent management (list, register, edit, delete, toggle, discovery).

## Deliverables
1. **A2A Agents list page** (`src/features/agents/A2AAgentsPage.tsx`):
   - Table/grid with columns: name, path, skills, visibility, enabled, actions.
   - Search/filter by query, visibility, enabled.
   - Actions: View, Edit, Toggle, Delete.

2. **A2A Agent details page** (`src/features/agents/A2AAgentDetailsPage.tsx`):
   - Agent card info (name, description, skills, tags, visibility).
   - Health check button.
   - Edit, Toggle, Delete buttons.

3. **Agent forms**:
   - `AgentRegisterModal.tsx` — register new agent with all card fields.
   - `AgentEditModal.tsx` — edit existing agent.
   - Validation for required fields.

4. **Discovery features**:
   - `DiscoverAgentsModal.tsx` — POST `/api/agents/discover`.
   - `SemanticSearchModal.tsx` — POST `/api/agents/discover/semantic`.
   - Display results with "Add to registry" action.

5. **API hooks**:
   - `useA2AAgents()` — list with filtering.
   - `useA2AAgent(path)` — single agent.
   - `useRegisterA2AAgent()` — mutation.
   - `useUpdateA2AAgent()` — mutation.
   - `useDeleteA2AAgent()` — mutation with confirmation.
   - `useToggleA2AAgent()` — mutation.
   - `useDiscoverAgents()` — mutation.
   - `useSemanticSearch()` — mutation.

## Tests to Add
- Unit test: Agent list renders, filters work.
- Unit test: Agent form validates skill/tag arrays.
- Integration test: Register agent flow.
- Integration test: Delete agent shows confirmation, completes.
- Integration test: Discovery returns and displays results.
- Integration test: Semantic search with debouncing.

## Required Test Gate
- `npm run typecheck` passes.
- `npm run build` succeeds.
- `npm run test` passes.

---

# Phase 8 — EnforceAI: Agents (Identity/Enforcement)
**Goal**: Complete EnforceAI agent lifecycle management.

## Deliverables
1. **EnforceAI Agents list page** (`src/features/enforceai-agents/EnforceAIAgentsPage.tsx`):
   - Table with columns: alias, agent_id, scopes (count + expand), allowed_tools, revoked_at, tokens_valid_after, updated_at, actions.
   - Filter: active vs revoked, search by alias/agent_id.
   - Actions: View, Edit, Revoke, Revoke All Tokens, Copy agent_id.
   - Warnings for: empty scopes, empty allowed_tools.

2. **Agent details page** (`src/features/enforceai-agents/EnforceAIAgentDetailsPage.tsx`):
   - Full agent info display.
   - Scopes list (expandable with descriptions from catalog if available).
   - Allowed tools list.
   - Metadata JSON viewer.
   - Actions: Edit, Revoke, Revoke All Tokens.
   - Associated credentials summary (link to credentials page filtered by agent).

3. **Agent forms**:
   - `CreateAgentModal.tsx`:
     - Alias (optional).
     - Scopes (required, multi-select or tag input).
     - Allowed tools (optional, multi-select).
     - Metadata (JSON editor).
     - Scope validation against catalog.
   - `EditAgentModal.tsx`:
     - Edit scopes, allowed_tools, alias, metadata.
     - Diff-like confirmation before save (especially scope removal).

4. **Revocation flows**:
   - `RevokeAgentModal.tsx` — confirm destructive action, call revoke endpoint.
   - `RevokeAllTokensModal.tsx` — explain effect, call revoke-all, show updated tokens_valid_after.

5. **API hooks**:
   - `useEnforceAIAgents()` — list agents.
   - `useEnforceAIAgent(agentId)` — single agent.
   - `useCreateEnforceAIAgent()` — mutation.
   - `useUpdateEnforceAIAgent()` — mutation.
   - `useRevokeEnforceAIAgent()` — mutation.
   - `useRevokeAllTokens()` — mutation.

## Tests to Add
- Unit test: Agent list displays warnings for empty scopes.
- Unit test: Create form validates non-empty scopes.
- Unit test: Edit form shows diff confirmation.
- Integration test: Create agent flow end-to-end.
- Integration test: Revoke agent disables further actions.
- Integration test: Revoke all tokens updates tokens_valid_after.

## Required Test Gate
- `npm run typecheck` passes.
- `npm run build` succeeds.
- `npm run test` passes.

---

# Phase 9 — EnforceAI: Credentials (API Keys + Gateway Tokens)
**Goal**: Complete credential management (create, list, revoke API keys; mint, revoke tokens).

## Deliverables
1. **Credentials overview page** (`src/features/credentials/CredentialsPage.tsx`):
   - Tabs or sections: API Keys, Gateway Tokens.
   - Agent selector to filter credentials by agent.

2. **API Keys section** (`src/features/credentials/ApiKeysSection.tsx`):
   - List per agent: key_id, scopes, expires_at, revoked_at, created_at, last_used_at.
   - Create API key button → modal.
   - Revoke action per key.

3. **Create API Key flow** (`src/features/credentials/CreateApiKeyModal.tsx`):
   - Select agent.
   - Optional scopes (subset of agent scopes).
   - Optional expires_at.
   - On success: show secret ONCE with:
     - `api_key_value` (full `eak_<key_id>.<secret>`).
     - Copy button.
     - Download as .txt button.
     - "I have copied this secret" acknowledgment checkbox.
     - Cannot navigate away until acknowledged.
   - "How to use" snippet with header name.

4. **Gateway Tokens section** (`src/features/credentials/TokensSection.tsx`):
   - Mint token button → modal.
   - Revoke token by JTI form.
   - Revoke by pasting token form.

5. **Mint Token flow** (`src/features/credentials/MintTokenModal.tsx`):
   - Select agent.
   - Scopes (required, subset of agent scopes).
   - Expiry: TTL presets (1h, 1d, 7d, 30d, custom) OR explicit expires_at.
   - Mutual exclusivity validation.
   - On success: show token ONCE with:
     - Full token value.
     - Copy button.
     - Copy as Authorization header button.
     - "Decode (local only)" expander showing claims.
     - Strong warning to store securely.

6. **Revoke Token flow** (`src/features/credentials/RevokeTokenModal.tsx`):
   - Two modes: by agent_id + jti, or by pasting token.
   - Optional reason.
   - Show returned revocation record.
   - Never persist pasted token.

7. **API hooks**:
   - `useApiKeys(agentId)` — list API keys.
   - `useCreateApiKey()` — mutation.
   - `useRevokeApiKey()` — mutation.
   - `useMintToken()` — mutation.
   - `useRevokeToken()` — mutation.

## Tests to Add
- Unit test: API key list displays all fields.
- Unit test: Create API key form validates scope subset.
- Unit test: Secret display requires acknowledgment.
- Unit test: Mint token validates TTL/expires_at mutual exclusivity.
- Unit test: Token decode shows claims locally.
- Integration test: Create API key flow shows secret once.
- Integration test: Revoke token by JTI.
- Integration test: Revoke token by pasting.

## Required Test Gate
- `npm run typecheck` passes.
- `npm run build` succeeds.
- `npm run test` passes.

---

# Phase 10 — Scopes and Policy
**Goal**: Display scope catalog content for policy understanding.

## Deliverables
1. **Scopes page** (`src/features/scopes/ScopesPage.tsx`):
   - Fetch scope catalog (if API exists; otherwise show "not available" message).
   - List all scopes with:
     - Scope name.
     - Allowed servers.
     - Allowed methods.
     - Tool restrictions (`all`/`*` or specific tools).
   - Search/filter by scope name, server name.
   - Expandable details per scope.

2. **Scope explainer** (`src/features/scopes/ScopeExplainerCard.tsx`):
   - "What does this scope allow?" section.
   - Maps to: `tools/list` visibility, `tools/call` execution.

3. **Scope picker component** (`src/components/common/ScopePicker.tsx`):
   - Reusable multi-select for scopes.
   - Shows scope names with hover description.
   - Used in agent create/edit forms.

4. **API hooks**:
   - `useScopesCatalog()` — fetch catalog (or handle 404 gracefully).

## Tests to Add
- Unit test: Scopes list renders catalog entries.
- Unit test: Search filters scopes correctly.
- Unit test: Scope picker allows multi-selection.
- Integration test: Handles missing catalog gracefully.

## Required Test Gate
- `npm run typecheck` passes.
- `npm run build` succeeds.
- `npm run test` passes.

---

# Phase 11 — Tools Discovery
**Goal**: Help users configure `allowed_tools` and understand tool access.

## Deliverables
1. **Tools page** (`src/features/tools/ToolsPage.tsx`):
   - List all MCP servers from registry.
   - Per-server: expandable tools list.
   - Search across all tools.

2. **Allowed tools builder** (`src/features/tools/AllowedToolsBuilder.tsx`):
   - Multi-server tool selection.
   - Warn when same tool name appears in multiple servers.
   - "Apply to agent" action → navigate to agent edit with pre-filled tools.

3. **Effective access preview** (`src/features/tools/EffectiveAccessPreview.tsx`):
   - Select an agent.
   - Show visible tools per server based on:
     - Agent scopes vs scope catalog.
     - Agent allowed_tools.
   - Disclaimer: final enforcement is server-side.

4. **Tool picker component** (`src/components/common/ToolPicker.tsx`):
   - Reusable tool selector.
   - Groups by server.
   - Used in agent forms.

5. **API hooks**:
   - `useAllServerTools()` — aggregate tools from all servers.

## Tests to Add
- Unit test: Tools list renders per-server.
- Unit test: Search filters tools correctly.
- Unit test: Allowed tools builder warns on duplicates.
- Unit test: Effective access preview calculates correctly.
- Integration test: Tool picker integrates with agent form.

## Required Test Gate
- `npm run typecheck` passes.
- `npm run build` succeeds.
- `npm run test` passes.

---

# Phase 12 — Audit
**Goal**: Provide audit guidance and future viewer placeholder.

## Deliverables
1. **Audit page** (`src/features/audit/AuditPage.tsx`):
   - Guidance section:
     - How to find audit events (Docker Compose logs, stdout, SQLite).
     - Filtering by `X-Request-Id`.
   - Common audit actions glossary:
     - `tools/list`, `tools/call`.
     - `management/agents/*`, `management/api-keys/*`, `management/tokens/*`.
   - Request ID display (show current session's request IDs for correlation).

2. **Future audit viewer placeholder**:
   - "Audit event viewer coming in future release" message.
   - If API exists later, integrate filtering by time, agent_id, outcome, action.

## Tests to Add
- Unit test: Audit page renders guidance sections.
- Unit test: Glossary displays all action types.

## Required Test Gate
- `npm run typecheck` passes.
- `npm run build` succeeds.
- `npm run test` passes.

---

# Phase 13 — Admin: Users Directory
**Goal**: Admin user search and directory view.

## Deliverables
1. **Admin layout** (`src/features/admin/AdminLayout.tsx`):
   - Distinct admin mode UX (visual differentiation).
   - Admin-only route protection.
   - Warning banner for admin context.

2. **Users directory page** (`src/features/admin/UsersPage.tsx`):
   - Search by email/username (partial match).
   - Results table: email (primary), username, user_id (copyable), auth_method, last_seen_at.
   - Click row → user details.

3. **User details page** (`src/features/admin/UserDetailsPage.tsx`):
   - User info card.
   - Agent count summary.
   - Link to cross-user operations (Phase 14).

4. **API hooks**:
   - `useAdminUsers(query)` — search users.
   - `useAdminUser(userId)` — single user.

## Tests to Add
- Unit test: Users search triggers API call.
- Unit test: Results display canonical user_id.
- Unit test: Admin route rejects non-admin users.
- Integration test: Search returns and displays results.

## Required Test Gate
- `npm run typecheck` passes.
- `npm run build` succeeds.
- `npm run test` passes.

---

# Phase 14 — Admin: Cross-User Operations
**Goal**: Admin can manage agents/credentials for other users.

## Deliverables
1. **Target user context**:
   - Admin must explicitly select target user before cross-user operations.
   - Show "Acting on user: [email]" banner.

2. **Cross-user agent operations** (within UserDetailsPage):
   - View user's EnforceAI agents.
   - Create agent for user.
   - Revoke agent for user.
   - Revoke all tokens for user's agent.

3. **Cross-user credential operations**:
   - View user's API keys.
   - Revoke API key for user.
   - Revoke token by JTI for user.

4. **Confirmation dialogs**:
   - "Type agent_id to confirm" for revocations.
   - "Reason required" field for admin actions.

5. **A2A agent deletion** (Registry):
   - Admin can delete A2A agent records.
   - Confirmation required.

6. **API hooks** (admin variants):
   - `useAdminAgents(userId)` — list agents for target user.
   - `useAdminCreateAgent(userId)` — create agent for user.
   - `useAdminRevokeAgent(userId, agentId)` — revoke.
   - `useAdminRevokeAllTokens(userId, agentId)` — revoke all.
   - `useAdminApiKeys(userId, agentId)` — list keys.
   - `useAdminRevokeApiKey(keyId)` — revoke key.
   - `useAdminRevokeToken(userId, agentId, jti)` — revoke token.
   - `useAdminDeleteA2AAgent(path)` — delete A2A agent.

## Tests to Add
- Unit test: Target user context is required.
- Unit test: Confirmation requires typing agent_id.
- Unit test: Reason field is captured.
- Integration test: Admin create agent for another user.
- Integration test: Admin revoke agent shows in audit.
- Integration test: Non-admin cannot access admin routes.

## Required Test Gate
- `npm run typecheck` passes.
- `npm run build` succeeds.
- `npm run test` passes.

---

# Phase 15 — Settings + Help + Final Polish
**Goal**: Complete Settings and Help pages, final UX polish.

## Deliverables
1. **Settings page** (`src/features/settings/SettingsPage.tsx`):
   - API base URL configuration (view/edit).
   - Current session info display (auth_method, email, user_id).
   - Agent context selector (for OIDC X-Agent-Id).
   - Theme toggle.
   - "Clear session" action.
   - Operator config checklist (read-only):
     - ENFORCEAI_DB_PATH, ENFORCEAI_AUTH_PROVIDER, etc.
     - Token keyring paths.
     - Pepper path.

2. **Help page** (`src/features/help/HelpPage.tsx`):
   - Links to documentation:
     - `docs/enforceai-setup-guide.md`
     - `enforceai/instructions/ENFORCEAI_MANAGEMENT.md`
     - `enforceai/instructions/ENFORCEAI_AUDIT_RETENTION.md`
   - Troubleshooting decision tree for 401/403/503:
     - 401: Authentication failed — check credentials, token expiry.
     - 403: Forbidden — check ownership, revocation, agent binding.
     - 503: Enforcement unavailable — check DB, keyring, contact operator.

3. **Error pages**:
   - 404 page.
   - Error boundary with retry.

4. **Final polish**:
   - Consistent loading states across all pages.
   - Consistent empty states.
   - Keyboard navigation for all forms/modals.
   - Screen reader labels.
   - Responsive layout verification.

5. **Security verification**:
   - Audit all forms: no secrets in URLs.
   - Verify secrets masked by default.
   - Verify copy-to-clipboard is deliberate.
   - Verify "show once" warnings on secrets.

## Tests to Add
- Unit test: Settings page displays session info.
- Unit test: Help page renders all links.
- Unit test: Error boundary catches and displays errors.
- Accessibility test: All forms keyboard navigable.
- Accessibility test: Screen reader labels present.
- Integration test: Theme toggle persists across reload.
- Integration test: Clear session redirects to login.

## Required Test Gate
- `npm run typecheck` passes.
- `npm run build` succeeds.
- `npm run test` passes.

---

# Phase 16 — End-to-End Testing + Documentation
**Goal**: Add E2E tests, finalize documentation, prepare for release.

## Deliverables
1. **E2E test setup**:
   - Add Playwright for E2E tests.
   - Configure against running backend (docker-compose).
   - Test accounts for different roles (admin, non-admin).

2. **E2E test scenarios**:
   - Login flow (password, OAuth mock).
   - Server management (register, toggle, refresh).
   - A2A agent management (register, delete).
   - EnforceAI agent lifecycle (create, edit, revoke).
   - API key flow (create, copy secret, revoke).
   - Token mint flow (mint, copy, revoke).
   - Admin user search and cross-user revocation.

3. **Documentation**:
   - Update `frontend/README.md` with:
     - Development setup.
     - Build instructions.
     - Test commands.
     - Environment configuration.
   - Add inline JSDoc comments for complex components.

4. **Build verification**:
   - Production build succeeds.
   - Bundle size check (warn if >500KB gzipped).
   - No console errors in production build.

## Tests to Add
- E2E: Full user journey from login to agent creation.
- E2E: Admin journey from login to cross-user revocation.
- Build: Bundle size assertion.

## Required Test Gate
- `npm run typecheck` passes.
- `npm run build` succeeds.
- `npm run test` passes.
- `npm run test:e2e` passes (against running backend).

---

## Post-Plan Validation Checklist
After Phase 16 completion:
- [ ] All 11 navigation sections functional.
- [ ] CSRF protection working for all state-changing operations.
- [ ] No long-lived secrets in browser storage.
- [ ] Admin mode visually distinct.
- [ ] All forms keyboard accessible.
- [ ] Dark/light theme working.
- [ ] Mobile responsive.
- [ ] Production build <500KB gzipped.
- [ ] Docker Compose stack integration verified.
- [ ] Session survives page refresh.
- [ ] Logout invalidates session.

---

## Summary

| Phase | Focus | Key Deliverables |
|-------|-------|------------------|
| 1 | Foundation | Vite migration, testing infra, API client |
| 2 | UI Primitives | Button, Input, Modal, Toast, etc. |
| 3 | Layout + Routes | Shell, Nav, all routes defined |
| 4 | Auth | Login, CSRF, session handling |
| 5 | Overview | Dashboard, connection status, quick actions |
| 6 | Servers | Registry server management |
| 7 | A2A Agents | Registry agent management |
| 8 | EnforceAI Agents | Identity agent lifecycle |
| 9 | Credentials | API keys + gateway tokens |
| 10 | Scopes | Policy catalog view |
| 11 | Tools | Tool discovery + builder |
| 12 | Audit | Guidance + placeholder viewer |
| 13 | Admin Users | User directory |
| 14 | Admin Ops | Cross-user operations |
| 15 | Settings + Help | Configuration, docs, polish |
| 16 | E2E + Docs | Playwright tests, final docs |
