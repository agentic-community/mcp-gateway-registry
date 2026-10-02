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
  buildUpdateScopeConfig,
  parseGroupMappings,
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

describe('parseGroupMappings', () => {
  it('splits, trims and drops empty entries', () => {
    expect(parseGroupMappings(' a , b ,, c ')).toEqual(['a', 'b', 'c']);
  });

  it('returns an empty list for a blank input', () => {
    expect(parseGroupMappings('   ')).toEqual([]);
  });
});

describe('buildUpdateScopeConfig', () => {
  /**
   * Regression coverage: the group edit form rendered a Group Mappings input
   * but handleUpdate built a payload without group_mappings. The management API
   * preserves existing values for any omitted field, so the edit was silently
   * discarded while the UI still reported "updated successfully".
   */
  it('includes group_mappings from the form input', () => {
    const config = buildUpdateScopeConfig(
      [],
      'mcp-servers-unrestricted, other-group',
      [],
      {},
      PROXIED,
    );
    expect(config.group_mappings).toEqual(['mcp-servers-unrestricted', 'other-group']);
  });

  it('always sends group_mappings so the field can be cleared', () => {
    const config = buildUpdateScopeConfig([], '', [], {}, PROXIED);
    expect(config).toHaveProperty('group_mappings');
    expect(config.group_mappings).toEqual([]);
  });

  it('sends every scope_config field, since an omitted field means preserve', () => {
    const config = buildUpdateScopeConfig([], '', [], {}, PROXIED);
    expect(Object.keys(config).sort()).toEqual([
      'agent_access',
      'group_mappings',
      'server_access',
      'ui_permissions',
    ]);
  });

  it('normalizes server keys and defaults empty methods to all', () => {
    const entries: ServerAccessEntry[] = [
      { server: '/currenttime/', methods: [], tools: [] },
    ];
    const config = buildUpdateScopeConfig(entries, '', [], {}, PROXIED);
    expect(config.server_access).toEqual([{ server: 'currenttime', methods: ['all'] }]);
  });

  it('keeps a proxied canonical key verbatim', () => {
    const entries: ServerAccessEntry[] = [
      { server: 'skill/skills/proxy-demo', methods: ['all'], tools: [] },
    ];
    const config = buildUpdateScopeConfig(entries, '', [], {}, PROXIED);
    expect(config.server_access[0].server).toBe('skill/skills/proxy-demo');
  });
});
