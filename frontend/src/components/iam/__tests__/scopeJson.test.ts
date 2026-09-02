/**
 * Tests for the IAM scope-JSON build logic (Task #11 / F1).
 *
 * The authz-critical properties: a proxied non-MCP entity's canonical authz key
 * (entity_type/registered_path) is written VERBATIM (not slash-stripped), and
 * proxied entities are NOT synced into MCP UI permissions. Getting either wrong
 * silently 403s the proxied route or over-grants MCP verbs.
 */

import {
  buildScopeJson,
  normalizeServerKey,
  applyUiPermSync,
  type ServerAccessEntry,
} from '../scopeJson';

const PROXIED = new Set<string>(['skill/skills/proxy-demo', 'a2a_agent/agents/code-reviewer']);

describe('normalizeServerKey', () => {
  it('keeps a proxied canonical key verbatim (interior slashes preserved)', () => {
    expect(normalizeServerKey('skill/skills/proxy-demo', PROXIED)).toBe('skill/skills/proxy-demo');
  });

  it('slash-strips a plain MCP server value', () => {
    expect(normalizeServerKey('/currenttime/', PROXIED)).toBe('currenttime');
  });

  it('slash-strips a virtual server value', () => {
    expect(normalizeServerKey('/virtual/dev/', PROXIED)).toBe('virtual/dev');
  });

  it('does not treat an unknown key as proxied', () => {
    expect(normalizeServerKey('/skill/skills/not-in-set/', PROXIED)).toBe(
      'skill/skills/not-in-set',
    );
  });
});

describe('applyUiPermSync', () => {
  const entry = (server: string): ServerAccessEntry => ({ server, methods: [], tools: [] });

  it('syncs MCP servers into MCP permissions', () => {
    const perms: Record<string, string[]> = {};
    applyUiPermSync(perms, [entry('/currenttime')], [], PROXIED);
    expect(perms['list_service']).toEqual(['currenttime']);
    expect(perms['call_tool']).toEqual(['currenttime']);
  });

  it('promotes the "*" (All servers) selection to the "all" wildcard token', () => {
    const perms: Record<string, string[]> = {};
    applyUiPermSync(perms, [entry('*')], [], PROXIED);
    // The auth-server treats 'all' (not '*') as the wildcard for MCP ui_permissions.
    expect(perms['list_service']).toEqual(['all']);
    expect(perms['get_service']).toEqual(['all']);
    expect(perms['call_tool']).toEqual(['all']);
  });

  it('does NOT sync a proxied entity into MCP permissions', () => {
    const perms: Record<string, string[]> = {};
    applyUiPermSync(perms, [entry('skill/skills/proxy-demo')], [], PROXIED);
    expect(perms['list_service']).toBeUndefined();
    expect(perms['call_tool']).toBeUndefined();
    expect(perms['list_virtual_server']).toBeUndefined();
  });

  it('syncs virtual servers into list_virtual_server only', () => {
    const perms: Record<string, string[]> = {};
    applyUiPermSync(perms, [entry('/virtual/dev')], [], PROXIED);
    expect(perms['list_virtual_server']).toEqual(['/virtual/dev']);
    expect(perms['list_service']).toBeUndefined();
  });

  it('mixed: MCP + proxied -> only the MCP server lands in MCP perms', () => {
    const perms: Record<string, string[]> = {};
    applyUiPermSync(
      perms,
      [entry('/currenttime'), entry('skill/skills/proxy-demo')],
      [],
      PROXIED,
    );
    expect(perms['list_service']).toEqual(['currenttime']);
  });

  it('clears agent perms when no agents selected', () => {
    const perms: Record<string, string[]> = { list_agents: ['x'], get_agent: ['x'] };
    applyUiPermSync(perms, [], [], PROXIED);
    expect(perms['list_agents']).toBeUndefined();
    expect(perms['get_agent']).toBeUndefined();
  });
});

describe('buildScopeJson', () => {
  it('emits the canonical authz key for a proxied entity with HTTP verbs', () => {
    const json = buildScopeJson(
      'proxy-scope',
      '',
      [{ server: 'skill/skills/proxy-demo', methods: ['GET', 'POST'], tools: [] }],
      '',
      [],
      {},
      false,
      PROXIED,
    );
    const access = json.server_access as Array<Record<string, unknown>>;
    expect(access[0].server).toBe('skill/skills/proxy-demo'); // NOT slash-stripped
    expect(access[0].methods).toEqual(['GET', 'POST']);
    // proxied entity is not wedged into MCP perms
    const perms = (json.ui_permissions || {}) as Record<string, unknown>;
    expect(perms['list_service']).toBeUndefined();
  });

  it('keeps MCP servers on the legacy bare-name key + MCP perms', () => {
    const json = buildScopeJson(
      'mcp-scope',
      '',
      [{ server: '/currenttime', methods: ['tools/call'], tools: ['*'] }],
      '',
      [],
      {},
      false,
      PROXIED,
    );
    const access = json.server_access as Array<Record<string, unknown>>;
    expect(access[0].server).toBe('currenttime');
    expect(access[0].tools).toBe('*');
    const perms = json.ui_permissions as Record<string, string[]>;
    expect(perms['list_service']).toEqual(['currenttime']);
  });

  it('defaults empty methods to ["all"]', () => {
    const json = buildScopeJson(
      's',
      '',
      [{ server: '/x', methods: [], tools: [] }],
      '',
      [],
      {},
      false,
      PROXIED,
    );
    const access = json.server_access as Array<Record<string, unknown>>;
    expect(access[0].methods).toEqual(['all']);
  });
});
