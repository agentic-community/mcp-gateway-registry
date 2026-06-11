import { humanize, labelFor } from '../humanize';

describe('humanize', () => {
  it('splits on underscores and title-cases', () => {
    expect(humanize('trigger_type')).toBe('Trigger Type');
  });

  it('splits on dashes', () => {
    expect(humanize('owner-team')).toBe('Owner Team');
  });

  it('splits camelCase boundaries', () => {
    expect(humanize('workflowBody')).toBe('Workflow Body');
  });

  it('title-cases a single word (no acronym logic in v1)', () => {
    expect(humanize('url')).toBe('Url');
  });

  it('collapses repeated separators', () => {
    expect(humanize('a__b--c')).toBe('A B C');
  });
});

describe('labelFor', () => {
  it('prefers an explicit label', () => {
    expect(labelFor({ name: 'trigger_type', label: 'Trigger' })).toBe('Trigger');
  });

  it('falls back to humanized name when label is absent', () => {
    expect(labelFor({ name: 'trigger_type' })).toBe('Trigger Type');
  });

  it('falls back to humanized name when label is null', () => {
    expect(labelFor({ name: 'owner-team', label: null })).toBe('Owner Team');
  });
});
