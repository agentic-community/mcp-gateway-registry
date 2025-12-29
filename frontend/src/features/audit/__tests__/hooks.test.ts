import { describe, it, expect } from 'vitest';
import {
  hasActiveAdvancedFilters,
  countActiveAdvancedFilters,
  INITIAL_ADVANCED_FILTERS,
  ACTION_CATEGORIES,
  ALL_ACTIONS,
  TIME_PRESETS,
  OUTCOME_OPTIONS,
  getTimePresetSince,
  type AdvancedFilters,
} from '../hooks';

describe('Audit Hooks', () => {
  describe('hasActiveAdvancedFilters', () => {
    it('returns false for initial filters', () => {
      expect(hasActiveAdvancedFilters(INITIAL_ADVANCED_FILTERS)).toBe(false);
    });

    it('returns true when agentId is set', () => {
      const filters: AdvancedFilters = { ...INITIAL_ADVANCED_FILTERS, agentId: 'test-agent' };
      expect(hasActiveAdvancedFilters(filters)).toBe(true);
    });

    it('returns true when actions is not empty', () => {
      const filters: AdvancedFilters = { ...INITIAL_ADVANCED_FILTERS, actions: ['tools/list'] };
      expect(hasActiveAdvancedFilters(filters)).toBe(true);
    });

    it('returns true when requestId is set', () => {
      const filters: AdvancedFilters = { ...INITIAL_ADVANCED_FILTERS, requestId: 'req-123' };
      expect(hasActiveAdvancedFilters(filters)).toBe(true);
    });

    it('returns true when server is set', () => {
      const filters: AdvancedFilters = { ...INITIAL_ADVANCED_FILTERS, server: 'sqlite' };
      expect(hasActiveAdvancedFilters(filters)).toBe(true);
    });

    it('returns true when tool is set', () => {
      const filters: AdvancedFilters = { ...INITIAL_ADVANCED_FILTERS, tool: 'read_query' };
      expect(hasActiveAdvancedFilters(filters)).toBe(true);
    });

    it('returns false for whitespace-only values', () => {
      const filters: AdvancedFilters = {
        agentId: '   ',
        actions: [],
        requestId: '   ',
        server: '   ',
        tool: '   ',
        userId: '   ',
      };
      expect(hasActiveAdvancedFilters(filters)).toBe(false);
    });

    it('returns true when userId is set in admin mode', () => {
      const filters: AdvancedFilters = { ...INITIAL_ADVANCED_FILTERS, userId: 'local|admin' };
      expect(hasActiveAdvancedFilters(filters, true)).toBe(true);
    });

    it('returns false for userId when not in admin mode', () => {
      const filters: AdvancedFilters = { ...INITIAL_ADVANCED_FILTERS, userId: 'local|admin' };
      expect(hasActiveAdvancedFilters(filters, false)).toBe(false);
    });
  });

  describe('countActiveAdvancedFilters', () => {
    it('returns 0 for initial filters', () => {
      expect(countActiveAdvancedFilters(INITIAL_ADVANCED_FILTERS)).toBe(0);
    });

    it('counts each active filter', () => {
      const filters: AdvancedFilters = {
        agentId: 'test',
        actions: ['tools/list'],
        requestId: 'req-123',
        server: 'sqlite',
        tool: 'read_query',
        userId: '',
      };
      expect(countActiveAdvancedFilters(filters)).toBe(5);
    });

    it('counts partial filters correctly', () => {
      const filters: AdvancedFilters = {
        agentId: 'test',
        actions: [],
        requestId: '',
        server: 'sqlite',
        tool: '',
        userId: '',
      };
      expect(countActiveAdvancedFilters(filters)).toBe(2);
    });

    it('ignores whitespace-only values', () => {
      const filters: AdvancedFilters = {
        agentId: '   ',
        actions: ['tools/list'],
        requestId: '   ',
        server: '',
        tool: 'read_query',
        userId: '   ',
      };
      expect(countActiveAdvancedFilters(filters)).toBe(2);
    });

    it('counts userId in admin mode', () => {
      const filters: AdvancedFilters = {
        agentId: 'test',
        actions: [],
        requestId: '',
        server: '',
        tool: '',
        userId: 'local|admin',
      };
      expect(countActiveAdvancedFilters(filters, true)).toBe(2);
    });

    it('does not count userId when not in admin mode', () => {
      const filters: AdvancedFilters = {
        agentId: 'test',
        actions: [],
        requestId: '',
        server: '',
        tool: '',
        userId: 'local|admin',
      };
      expect(countActiveAdvancedFilters(filters, false)).toBe(1);
    });
  });

  describe('ACTION_CATEGORIES', () => {
    it('has expected categories', () => {
      const categories = ACTION_CATEGORIES.map((c) => c.category);
      expect(categories).toContain('MCP Operations');
      expect(categories).toContain('Agent Management');
      expect(categories).toContain('API Key Management');
      expect(categories).toContain('Token Management');
    });

    it('each category has actions', () => {
      ACTION_CATEGORIES.forEach((category) => {
        expect(category.actions.length).toBeGreaterThan(0);
      });
    });

    it('MCP Operations contains expected actions', () => {
      const mcpOps = ACTION_CATEGORIES.find((c) => c.category === 'MCP Operations');
      expect(mcpOps?.actions).toContain('tools/list');
      expect(mcpOps?.actions).toContain('tools/call');
    });
  });

  describe('ALL_ACTIONS', () => {
    it('contains all actions from categories', () => {
      const totalFromCategories = ACTION_CATEGORIES.reduce((sum, cat) => sum + cat.actions.length, 0);
      expect(ALL_ACTIONS.length).toBe(totalFromCategories);
    });

    it('contains expected actions', () => {
      expect(ALL_ACTIONS).toContain('tools/list');
      expect(ALL_ACTIONS).toContain('management/agents/create');
      expect(ALL_ACTIONS).toContain('management/api-keys/create');
      expect(ALL_ACTIONS).toContain('management/tokens/mint');
    });
  });

  describe('TIME_PRESETS', () => {
    it('has expected presets', () => {
      const values = TIME_PRESETS.map((p) => p.value);
      expect(values).toContain('15m');
      expect(values).toContain('1h');
      expect(values).toContain('24h');
      expect(values).toContain('7d');
    });

    it('each preset has label, value, and minutes', () => {
      TIME_PRESETS.forEach((preset) => {
        expect(preset.label).toBeTruthy();
        expect(preset.value).toBeTruthy();
        expect(preset.minutes).toBeGreaterThan(0);
      });
    });
  });

  describe('OUTCOME_OPTIONS', () => {
    it('has allow and deny options', () => {
      const values = OUTCOME_OPTIONS.map((o) => o.value);
      expect(values).toContain('allow');
      expect(values).toContain('deny');
    });

    it('each option has label and variant', () => {
      OUTCOME_OPTIONS.forEach((option) => {
        expect(option.label).toBeTruthy();
        expect(option.variant).toBeTruthy();
      });
    });
  });

  describe('getTimePresetSince', () => {
    it('returns valid ISO string for 15m preset', () => {
      const since = getTimePresetSince('15m');
      expect(since).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z$/);
    });

    it('returns valid ISO string for 1h preset', () => {
      const since = getTimePresetSince('1h');
      expect(since).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z$/);
    });

    it('returns valid ISO string for 24h preset', () => {
      const since = getTimePresetSince('24h');
      expect(since).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z$/);
    });

    it('returns valid ISO string for 7d preset', () => {
      const since = getTimePresetSince('7d');
      expect(since).toMatch(/^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}.\d{3}Z$/);
    });

    it('calculates correct time for 1h preset', () => {
      const now = Date.now();
      const since = getTimePresetSince('1h');
      const sinceTime = new Date(since).getTime();

      // Should be approximately 1 hour ago (allow 1 second tolerance)
      const expectedTime = now - 60 * 60 * 1000;
      expect(Math.abs(sinceTime - expectedTime)).toBeLessThan(1000);
    });

    it('throws error for unknown preset', () => {
      expect(() => getTimePresetSince('invalid' as any)).toThrow('Unknown time preset: invalid');
    });
  });

  describe('INITIAL_ADVANCED_FILTERS', () => {
    it('has empty string values', () => {
      expect(INITIAL_ADVANCED_FILTERS.agentId).toBe('');
      expect(INITIAL_ADVANCED_FILTERS.requestId).toBe('');
      expect(INITIAL_ADVANCED_FILTERS.server).toBe('');
      expect(INITIAL_ADVANCED_FILTERS.tool).toBe('');
      expect(INITIAL_ADVANCED_FILTERS.userId).toBe('');
    });

    it('has empty actions array', () => {
      expect(INITIAL_ADVANCED_FILTERS.actions).toEqual([]);
    });
  });
});
