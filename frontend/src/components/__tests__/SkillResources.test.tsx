/**
 * Tests for SkillResources component (issue #1111).
 *
 * Coverage: empty/full manifest, default-collapse, View / Download per file,
 * cap enforcement (file count + bytes), concurrency cap, partial / all-fail
 * paths, federation guard, and pure-helper sanity tests.
 */
import React from 'react';
import { render, screen, fireEvent, waitFor, act } from '@testing-library/react';
import axios from 'axios';

// react-markdown 10 is ESM-only and CRA's default jest config can't resolve
// its transitive deps (unist-util-visit-parents). For these tests we don't
// need real markdown rendering -- a minimal passthrough proves the preview
// content reaches the DOM, which is what the assertions check.
jest.mock('react-markdown', () => {
  const MockMarkdown = (props: { children: string }) => <div>{props.children}</div>;
  MockMarkdown.displayName = 'ReactMarkdown';
  return { __esModule: true, default: MockMarkdown };
});
jest.mock('remark-gfm', () => ({ __esModule: true, default: () => {} }));

import SkillResources, { __test__ } from '../SkillResources';
import type { Skill, SkillResourceManifest } from '../../types/skill';
import * as blobDownload from '../../utils/blobDownload';

jest.mock('axios');
const mockedAxios = axios as jest.Mocked<typeof axios>;

// Spy on triggerBlobDownload so we can assert without actually saving.
const triggerSpy = jest.spyOn(blobDownload, 'triggerBlobDownload');


// =============================================================================
// Fixtures
// =============================================================================


function makeSkill(overrides: Partial<Skill> = {}): Skill {
  return {
    name: 'Doc Coauthor',
    path: '/skills/doc-coauthor',
    skill_md_url: 'https://github.com/example/skills/blob/main/doc-coauthor/SKILL.md',
    visibility: 'public',
    is_enabled: true,
    num_stars: 0,
    registry_name: 'local',
    ...overrides,
  };
}


function makeManifest(overrides: Partial<SkillResourceManifest> = {}): SkillResourceManifest {
  return {
    references: [],
    scripts: [],
    agents: [],
    assets: [],
    ...overrides,
  };
}


function defaultProps(
  skillOverrides: Partial<Skill> & { resource_manifest?: SkillResourceManifest | null } = {},
  skillMdContent = '# SKILL\n',
) {
  // The `resource_manifest` field used to live on the Skill prop; it's now
  // a separate prop sourced from the /content fetch (the listing schema
  // doesn't include it). Tests still pass it via the skillOverrides bag for
  // brevity; we lift it out here so the helper still works without
  // updating every call site.
  const { resource_manifest, ...rest } = skillOverrides;
  return {
    skill: makeSkill(rest),
    skillApiPath: '/doc-coauthor',
    authToken: 'test-token',
    skillMdContent,
    resourceManifest: resource_manifest ?? null,
  };
}


// =============================================================================
// Pure helper tests (no React)
// =============================================================================


describe('SkillResources helpers', () => {
  test('_slugify produces filesystem-safe filenames', () => {
    const { _slugify } = __test__;
    expect(_slugify('My Skill')).toBe('my-skill');
    expect(_slugify('My Skill!')).toBe('my-skill');
    expect(_slugify('   ')).toBe('');
    expect(_slugify('../../etc/passwd')).toBe('etc-passwd');
    expect(_slugify('A/B\\C')).toBe('a-b-c');
    expect(_slugify('a__b__c')).toBe('a-b-c');
  });

  test('_formatSize formats bytes / KB / MB', () => {
    const { _formatSize } = __test__;
    expect(_formatSize(0)).toBe('0 B');
    expect(_formatSize(512)).toBe('512 B');
    expect(_formatSize(1024)).toBe('1.0 KB');
    expect(_formatSize(2048)).toBe('2.0 KB');
    expect(_formatSize(10 * 1024 * 1024)).toBe('10.0 MB');
  });

  test('_basename extracts the trailing path segment', () => {
    const { _basename } = __test__;
    expect(__test__._basename('scripts/run.sh')).toBe('run.sh');
    expect(__test__._basename('SKILL.md')).toBe('SKILL.md');
    expect(__test__._basename('a/b/c/d.txt')).toBe('d.txt');
  });

  test('_isOverCap rejects > MAX_BUNDLE_FILES', () => {
    const { _isOverCap, MAX_BUNDLE_FILES } = __test__;
    const many = Array.from({ length: MAX_BUNDLE_FILES }, (_, i) => ({
      path: `f${i}.txt`,
      type: 'reference' as const,
      size_bytes: 10,
    }));
    const result = _isOverCap(many);
    expect(result.over).toBe(true);
    if (result.over) expect(result.reason).toMatch(/cap is 50/);
  });

  test('_isOverCap rejects > MAX_BUNDLE_BYTES', () => {
    const { _isOverCap, MAX_BUNDLE_BYTES } = __test__;
    const big = [{ path: 'big.bin', type: 'asset' as const, size_bytes: MAX_BUNDLE_BYTES + 1 }];
    const result = _isOverCap(big);
    expect(result.over).toBe(true);
    if (result.over) expect(result.reason).toMatch(/cap is/);
  });

  test('_isOverCap accepts under-cap bundles', () => {
    const { _isOverCap } = __test__;
    expect(_isOverCap([{ path: 'a.txt', type: 'reference', size_bytes: 100 }]).over).toBe(false);
  });

  test('_sortByPath sorts ascending without mutating input', () => {
    const { _sortByPath } = __test__;
    const input = [
      { path: 'b.txt', type: 'reference' as const, size_bytes: 1 },
      { path: 'a.txt', type: 'reference' as const, size_bytes: 1 },
    ];
    const out = _sortByPath(input);
    expect(out.map((r) => r.path)).toEqual(['a.txt', 'b.txt']);
    expect(input.map((r) => r.path)).toEqual(['b.txt', 'a.txt']); // not mutated
  });
});


// =============================================================================
// Component: empty / federated / no-manifest cases
// =============================================================================


describe('SkillResources gating', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders nothing when manifest absent', () => {
    const { container } = render(
      <SkillResources {...defaultProps({ resource_manifest: null })} />,
    );
    expect(container.firstChild).toBeNull();
  });

  test('renders nothing when manifest groups are all empty', () => {
    const { container } = render(
      <SkillResources {...defaultProps({ resource_manifest: makeManifest() })} />,
    );
    expect(container.firstChild).toBeNull();
  });

  test('renders nothing for federated skills (registry_name !== local)', () => {
    const manifest = makeManifest({
      references: [{ path: 'r.md', type: 'reference', size_bytes: 100 }],
    });
    const { container } = render(
      <SkillResources {...defaultProps({
        registry_name: 'peer-registry',
        resource_manifest: manifest,
      })} />,
    );
    expect(container.firstChild).toBeNull();
  });
});


// =============================================================================
// Component: rendering populated manifest
// =============================================================================


describe('SkillResources rendering', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('renders header with totals and all four groups when present', () => {
    const manifest = makeManifest({
      references: [{ path: 'references/a.md', type: 'reference', size_bytes: 100 }],
      scripts: [{ path: 'scripts/run.sh', type: 'script', size_bytes: 200, language: 'shell' }],
      agents: [{ path: 'agents/c.md', type: 'agent', size_bytes: 300 }],
      assets: [{ path: 'assets/img.png', type: 'asset', size_bytes: 400 }],
    });

    render(<SkillResources {...defaultProps({ resource_manifest: manifest })} />);

    expect(screen.getByText(/Resources/)).toBeInTheDocument();
    expect(screen.getByText(/4 files/)).toBeInTheDocument();
    expect(screen.getByText(/References/)).toBeInTheDocument();
    expect(screen.getByText(/Scripts/)).toBeInTheDocument();
    expect(screen.getByText(/Agents/)).toBeInTheDocument();
    expect(screen.getByText(/Assets/)).toBeInTheDocument();
  });

  test('omits empty groups', () => {
    const manifest = makeManifest({
      references: [{ path: 'r.md', type: 'reference', size_bytes: 100 }],
    });
    render(<SkillResources {...defaultProps({ resource_manifest: manifest })} />);

    expect(screen.getByText(/References/)).toBeInTheDocument();
    expect(screen.queryByText(/Scripts/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Agents/)).not.toBeInTheDocument();
    expect(screen.queryByText(/Assets/)).not.toBeInTheDocument();
  });

  test('groups default-collapsed: rows hidden until clicked', () => {
    const manifest = makeManifest({
      references: [{ path: 'references/a.md', type: 'reference', size_bytes: 100 }],
    });
    render(<SkillResources {...defaultProps({ resource_manifest: manifest })} />);

    // Path text should not be in the DOM while collapsed.
    expect(screen.queryByText('references/a.md')).not.toBeInTheDocument();

    // Click the header to expand.
    const headerBtn = screen.getByRole('button', { name: /References/ });
    fireEvent.click(headerBtn);

    expect(screen.getByText('references/a.md')).toBeInTheDocument();
  });

  test('View hidden for assets (download-only)', () => {
    const manifest = makeManifest({
      assets: [{ path: 'assets/img.png', type: 'asset', size_bytes: 400 }],
    });
    render(<SkillResources {...defaultProps({ resource_manifest: manifest })} />);

    fireEvent.click(screen.getByRole('button', { name: /Assets/ }));

    expect(screen.queryByRole('button', { name: /^View assets/ })).not.toBeInTheDocument();
    expect(screen.getByRole('button', { name: /^Download assets\/img\.png/ })).toBeInTheDocument();
  });

  test('View visible for scripts/references/agents', () => {
    const manifest = makeManifest({
      scripts: [{ path: 'scripts/run.sh', type: 'script', size_bytes: 200 }],
    });
    render(<SkillResources {...defaultProps({ resource_manifest: manifest })} />);

    fireEvent.click(screen.getByRole('button', { name: /Scripts/ }));

    expect(screen.getByRole('button', { name: /^View scripts/ })).toBeInTheDocument();
  });
});


// =============================================================================
// Component: per-file Download
// =============================================================================


describe('Per-file Download', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('triggers blob download with basename filename', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: { content: '#!/bin/bash\necho hi' } });

    const manifest = makeManifest({
      scripts: [{ path: 'scripts/run.sh', type: 'script', size_bytes: 200 }],
    });
    render(<SkillResources {...defaultProps({ resource_manifest: manifest })} />);

    fireEvent.click(screen.getByRole('button', { name: /Scripts/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Download scripts\/run\.sh/ }));

    await waitFor(() => expect(triggerSpy).toHaveBeenCalled());
    const [, filename] = triggerSpy.mock.calls[0];
    expect(filename).toBe('run.sh');
    expect(mockedAxios.get).toHaveBeenCalledWith(
      '/api/skills/doc-coauthor/content',
      expect.objectContaining({
        params: { resource: 'scripts/run.sh' },
        headers: { Authorization: 'Bearer test-token' },
      }),
    );
  });

  test('shows error when per-file fetch fails', async () => {
    mockedAxios.get.mockRejectedValueOnce({ response: { status: 502 } });

    const manifest = makeManifest({
      scripts: [{ path: 'scripts/run.sh', type: 'script', size_bytes: 200 }],
    });
    render(<SkillResources {...defaultProps({ resource_manifest: manifest })} />);

    fireEvent.click(screen.getByRole('button', { name: /Scripts/ }));
    fireEvent.click(screen.getByRole('button', { name: /^Download scripts\/run\.sh/ }));

    await waitFor(() => expect(screen.getByText('HTTP 502')).toBeInTheDocument());
    expect(triggerSpy).not.toHaveBeenCalled();
  });
});


// =============================================================================
// Component: View / Preview
// =============================================================================


describe('Preview pane', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('opens preview, fetches resource, Back returns to listing', async () => {
    mockedAxios.get.mockResolvedValueOnce({ data: { content: '# Hello world\n' } });

    const manifest = makeManifest({
      references: [{ path: 'references/arch.md', type: 'reference', size_bytes: 100 }],
    });
    render(<SkillResources {...defaultProps({ resource_manifest: manifest })} />);

    fireEvent.click(screen.getByRole('button', { name: /References/ }));
    fireEvent.click(screen.getByRole('button', { name: /^View references\/arch\.md/ }));

    await waitFor(() => expect(screen.getByText(/Hello world/)).toBeInTheDocument());

    fireEvent.click(screen.getByRole('button', { name: /Back to SKILL\.md/ }));

    expect(screen.queryByText(/Hello world/)).not.toBeInTheDocument();
    expect(screen.getByText(/References/)).toBeInTheDocument();
  });
});


// =============================================================================
// Component: Download all (cap, concurrency, all-fail, partial)
// =============================================================================


function _makeResources(n: number, sizeBytes = 100): SkillResourceManifest {
  return {
    scripts: [],
    agents: [],
    assets: [],
    references: Array.from({ length: n }, (_, i) => ({
      path: `references/r${i}.md`,
      type: 'reference' as const,
      size_bytes: sizeBytes,
    })),
  };
}


describe('Download all', () => {
  beforeEach(() => {
    jest.clearAllMocks();
  });

  test('disables Download all when over file cap and surfaces reason via tooltip', () => {
    const manifest = _makeResources(__test__.MAX_BUNDLE_FILES + 1);
    render(<SkillResources {...defaultProps({ resource_manifest: manifest })} />);

    const btn = screen.getByRole('button', { name: /Download all classified resources/ });
    expect(btn).toBeDisabled();
    expect(btn).toHaveAttribute('title', expect.stringMatching(/cap is 50/));

    // Disabled buttons swallow clicks: no fetch, no zip.
    fireEvent.click(btn);
    expect(mockedAxios.get).not.toHaveBeenCalled();
    expect(triggerSpy).not.toHaveBeenCalled();
  });

  test('disables Download all when over byte cap', () => {
    // 5 files of 3 MB each = 15 MB > 10 MB cap.
    const manifest = makeManifest({
      references: Array.from({ length: 5 }, (_, i) => ({
        path: `r${i}.md`,
        type: 'reference' as const,
        size_bytes: 3 * 1024 * 1024,
      })),
    });
    render(<SkillResources {...defaultProps({ resource_manifest: manifest })} />);

    const btn = screen.getByRole('button', { name: /Download all classified resources/ });
    expect(btn).toBeDisabled();
    expect(btn).toHaveAttribute('title', expect.stringMatching(/cap is/));

    fireEvent.click(btn);
    expect(mockedAxios.get).not.toHaveBeenCalled();
  });

  test('happy path bundles SKILL.md + all successes', async () => {
    const manifest = makeManifest({
      references: [{ path: 'references/a.md', type: 'reference', size_bytes: 10 }],
      scripts: [{ path: 'scripts/run.sh', type: 'script', size_bytes: 20 }],
    });
    mockedAxios.get
      .mockResolvedValueOnce({ data: { content: 'AAA' } })
      .mockResolvedValueOnce({ data: { content: 'BBB' } });

    render(<SkillResources {...defaultProps({ resource_manifest: manifest }, '# Doc Coauthor\n')} />);

    fireEvent.click(screen.getByRole('button', { name: /Download all/ }));

    await waitFor(() => expect(triggerSpy).toHaveBeenCalled());
    const [, filename] = triggerSpy.mock.calls[0];
    expect(filename).toBe('doc-coauthor.zip');
  });

  test('all-fail guard: no zip when every fetch errors', async () => {
    const manifest = makeManifest({
      references: [
        { path: 'r1.md', type: 'reference', size_bytes: 10 },
        { path: 'r2.md', type: 'reference', size_bytes: 10 },
      ],
    });
    mockedAxios.get
      .mockRejectedValueOnce({ response: { status: 502 } })
      .mockRejectedValueOnce({ response: { status: 502 } });

    render(<SkillResources {...defaultProps({ resource_manifest: manifest })} />);

    fireEvent.click(screen.getByRole('button', { name: /Download all/ }));

    await waitFor(() =>
      expect(screen.getByText(/Could not fetch any resources/)).toBeInTheDocument(),
    );
    expect(triggerSpy).not.toHaveBeenCalled();
  });

  test('partial failure: zip still downloads, error banner shown', async () => {
    const manifest = makeManifest({
      references: [
        { path: 'r1.md', type: 'reference', size_bytes: 10 },
        { path: 'r2.md', type: 'reference', size_bytes: 10 },
      ],
    });
    mockedAxios.get
      .mockResolvedValueOnce({ data: { content: 'OK' } })
      .mockRejectedValueOnce({ response: { status: 502 } });

    render(<SkillResources {...defaultProps({ resource_manifest: manifest })} />);

    fireEvent.click(screen.getByRole('button', { name: /Download all/ }));

    await waitFor(() => expect(triggerSpy).toHaveBeenCalled());
    expect(screen.getByText(/1 file\(s\) failed/)).toBeInTheDocument();
  });

  test('respects FETCH_CONCURRENCY = 4', async () => {
    const total = 12;
    const manifest = _makeResources(total);

    let inFlight = 0;
    let maxInFlight = 0;
    mockedAxios.get.mockImplementation(() => {
      inFlight += 1;
      maxInFlight = Math.max(maxInFlight, inFlight);
      return new Promise((resolve) => {
        setTimeout(() => {
          inFlight -= 1;
          resolve({ data: { content: 'x' } } as any);
        }, 30);
      });
    });

    render(<SkillResources {...defaultProps({ resource_manifest: manifest })} />);

    await act(async () => {
      fireEvent.click(screen.getByRole('button', { name: /Download all/ }));
      await waitFor(() => expect(triggerSpy).toHaveBeenCalled(), { timeout: 5000 });
    });

    expect(maxInFlight).toBeLessThanOrEqual(__test__.FETCH_CONCURRENCY);
    expect(mockedAxios.get).toHaveBeenCalledTimes(total);
  });
});
