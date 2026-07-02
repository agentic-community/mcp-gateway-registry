/**
 * Pure scope-JSON build logic for the IAM scope editor, extracted from
 * IAMGroups.tsx so the create-preview and edit-save paths share ONE
 * implementation (they previously diverged) and so the authz-sensitive rules can
 * be unit-tested in isolation.
 *
 * Security-relevant invariants encoded here:
 * - A PROXIED non-MCP entity's `server` value is the canonical authz key
 *   `entity_type/registered_path` (supplied by the picker). It must be written
 *   VERBATIM — not slash-stripped to a bare name — or the auth-server's
 *   generic-hop authz key won't match and every request 403s.
 * - Proxied entities must NOT be synced into the MCP list_service/call_tool UI
 *   permissions (that would grant unintended MCP verbs). Only real MCP servers
 *   sync into MCP perms; virtual servers into list_virtual_server.
 */

export interface ServerAccessEntry {
  server: string;
  methods: string[];
  tools: string[];
}

// Membership in the proxied-authz-key set is the authoritative classifier
// (custom type names are open-set, so a prefix heuristic alone is unsafe).
export function isProxiedServerKey(server: string, proxiedKeys: Set<string>): boolean {
  return proxiedKeys.has(server.trim());
}

// Normalize a server_access `server` value for persistence:
// - proxied entity -> keep the canonical authz key verbatim (interior slashes
//   are meaningful: skill/skills/foo);
// - MCP / virtual / "*" -> strip leading & trailing slashes (legacy behavior).
export function normalizeServerKey(server: string, proxiedKeys: Set<string>): string {
  const s = server.trim();
  if (isProxiedServerKey(s, proxiedKeys)) return s;
  return s.replace(/^\/+|\/+$/g, '');
}

// Shared UI-permission auto-sync. MCP servers sync into the MCP service
// permissions; virtual servers into list_virtual_server; PROXIED non-MCP
// entities sync into NEITHER (their access is governed entirely by their
// server_access methods).
export function applyUiPermSync(
  perms: Record<string, string[]>,
  serverAccess: ServerAccessEntry[],
  selectedAgents: string[],
  proxiedKeys: Set<string>,
): void {
  const serverVals = serverAccess.filter((e) => e.server.trim()).map((e) => e.server.trim());

  const virtualServerPaths = serverVals.filter((p) => p.startsWith('/virtual/'));
  const mcpServerPaths = serverVals
    .filter((p) => !p.startsWith('/virtual/') && !isProxiedServerKey(p, proxiedKeys))
    .map((p) => p.replace(/^\/+|\/+$/g, ''));

  const MCP_PERM_KEYS = [
    'list_service',
    'health_check_service',
    'get_service',
    'list_tools',
    'call_tool',
  ];
  if (mcpServerPaths.length > 0) {
    for (const k of MCP_PERM_KEYS) perms[k] = mcpServerPaths;
  } else {
    for (const k of MCP_PERM_KEYS) delete perms[k];
  }

  if (virtualServerPaths.length > 0) {
    perms['list_virtual_server'] = virtualServerPaths;
  } else {
    delete perms['list_virtual_server'];
  }

  if (selectedAgents.length > 0) {
    perms['list_agents'] = selectedAgents;
    perms['get_agent'] = selectedAgents;
  } else {
    delete perms['list_agents'];
    delete perms['get_agent'];
  }
}

/**
 * Build the full scope JSON from form state for preview and API payload.
 */
export function buildScopeJson(
  name: string,
  description: string,
  serverAccess: ServerAccessEntry[],
  groupMappings: string,
  selectedAgents: string[],
  uiPermissions: Record<string, string>,
  createInIdp: boolean,
  proxiedKeys: Set<string>,
): Record<string, unknown> {
  const result: Record<string, unknown> = { scope_name: name };
  if (description) result.description = description;

  // Convert server access entries. Proxied non-MCP entities keep their canonical
  // authz key (entity_type/registered_path); MCP/virtual are slash-normalized.
  const access = serverAccess
    .filter((e) => e.server.trim())
    .map((e) => {
      const entry: Record<string, unknown> = {
        server: normalizeServerKey(e.server, proxiedKeys),
        methods: e.methods.length > 0 ? e.methods : ['all'],
      };
      if (e.tools.includes('*')) {
        entry.tools = '*';
      } else if (e.tools.length > 0) {
        entry.tools = e.tools;
      }
      return entry;
    });
  if (access.length > 0) result.server_access = access;

  const mappings = groupMappings
    .split(',')
    .map((m) => m.trim())
    .filter(Boolean);
  if (mappings.length > 0) result.group_mappings = mappings;

  if (selectedAgents.length > 0) result.agent_access = selectedAgents;

  const perms: Record<string, string[]> = {};
  for (const [key, val] of Object.entries(uiPermissions)) {
    const items = val.split(',').map((v) => v.trim()).filter(Boolean);
    if (items.length > 0) perms[key] = items;
  }

  applyUiPermSync(perms, serverAccess, selectedAgents, proxiedKeys);

  if (Object.keys(perms).length > 0) result.ui_permissions = perms;

  result.create_in_idp = createInIdp;
  return result;
}
