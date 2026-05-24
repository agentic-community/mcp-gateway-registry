import React, { useCallback, useState } from 'react';
import axios from 'axios';
import JSZip from 'jszip';
import ReactMarkdown from 'react-markdown';
import remarkGfm from 'remark-gfm';
import {
  ArrowDownTrayIcon,
  ArrowLeftIcon,
  EyeIcon,
  ChevronDownIcon,
  ChevronRightIcon,
} from '@heroicons/react/24/outline';
import { Skill, SkillResource, SkillResourceManifest } from '../types/skill';
import { triggerBlobDownload } from '../utils/blobDownload';

// Cap constants for the one-click "Download all" zip path. Per-file sizes
// are independently capped server-side at 512 KB (skill_routes.py:453).
const MAX_BUNDLE_FILES = 50;
const MAX_BUNDLE_BYTES = 10 * 1024 * 1024; // 10 MB
const FETCH_CONCURRENCY = 4;

// Resources whose contents render usefully inline (markdown / code).
// Assets are treated as opaque in v1 (download-only) since the server
// returns content as UTF-8 text and binary assets would be corrupted.
const TEXT_PREVIEWABLE_TYPES: ReadonlySet<SkillResource['type']> = new Set<SkillResource['type']>(
  ['script', 'reference', 'agent'],
);


// =============================================================================
// Private helpers
// =============================================================================


function _slugify(name: string): string {
  return name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-+|-+$/g, '');
}


function _formatSize(bytes: number): string {
  if (bytes < 1024) return `${bytes} B`;
  if (bytes < 1024 * 1024) return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}


function _basename(path: string): string {
  const idx = path.lastIndexOf('/');
  return idx >= 0 ? path.slice(idx + 1) : path;
}


function _flattenManifest(m: SkillResourceManifest): SkillResource[] {
  return [
    ...m.references,
    ...m.scripts,
    ...m.agents,
    ...m.assets,
  ];
}


function _hasAnyResource(m: SkillResourceManifest): boolean {
  return (
    m.scripts.length + m.references.length + m.agents.length + m.assets.length
  ) > 0;
}


function _sortByPath(resources: SkillResource[]): SkillResource[] {
  return [...resources].sort((a, b) => a.path.localeCompare(b.path));
}


type CapResult = { over: true; reason: string } | { over: false };


function _isOverCap(resources: SkillResource[]): CapResult {
  // SKILL.md adds 1 file to the bundle; its size is unknown until fetched
  // but is small in practice and not counted toward the byte cap here.
  if (resources.length + 1 > MAX_BUNDLE_FILES) {
    return {
      over: true,
      reason: `Bundle has ${resources.length + 1} files; cap is ${MAX_BUNDLE_FILES}.`,
    };
  }
  const total = resources.reduce((sum, r) => sum + r.size_bytes, 0);
  if (total > MAX_BUNDLE_BYTES) {
    return {
      over: true,
      reason: `Bundle is ${_formatSize(total)}; cap is ${_formatSize(MAX_BUNDLE_BYTES)}.`,
    };
  }
  return { over: false };
}


async function _fetchResource(
  skillApiPath: string,
  resourcePath: string,
  authToken: string | null,
): Promise<{ path: string; content: string }> {
  const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
  const resp = await axios.get(
    `/api/skills${skillApiPath}/content`,
    {
      params: { resource: resourcePath },
      ...(headers ? { headers } : {}),
    },
  );
  return { path: resourcePath, content: resp.data.content as string };
}


/**
 * Fetch every resource in `resources` with at most FETCH_CONCURRENCY
 * requests in flight at any time. Errors are captured per-file rather
 * than aborting the whole batch.
 */
async function _fetchAllWithConcurrency(
  skillApiPath: string,
  resources: SkillResource[],
  authToken: string | null,
  onProgress: (done: number, total: number) => void,
): Promise<{
  successes: { path: string; content: string }[];
  failures: { path: string; error: string }[];
}> {
  const successes: { path: string; content: string }[] = [];
  const failures: { path: string; error: string }[] = [];
  const queue = [...resources];
  let done = 0;

  async function _worker(): Promise<void> {
    while (queue.length > 0) {
      const next = queue.shift();
      if (!next) return;
      try {
        const result = await _fetchResource(skillApiPath, next.path, authToken);
        successes.push(result);
      } catch (e: any) {
        const status = e?.response?.status;
        failures.push({
          path: next.path,
          error: status ? `HTTP ${status}` : 'fetch failed',
        });
      } finally {
        done += 1;
        onProgress(done, resources.length);
      }
    }
  }

  const workerCount = Math.min(FETCH_CONCURRENCY, resources.length);
  await Promise.all(
    Array.from({ length: workerCount }, () => _worker()),
  );

  return { successes, failures };
}


async function _buildAndDownloadZip(
  skill: Skill,
  skillMdContent: string,
  fetched: { path: string; content: string }[],
): Promise<void> {
  const zip = new JSZip();
  zip.file('SKILL.md', skillMdContent);
  for (const r of fetched) {
    zip.file(r.path, r.content);
  }
  const blob = await zip.generateAsync({ type: 'blob' });
  triggerBlobDownload(blob, `${_slugify(skill.name)}.zip`);
}


// =============================================================================
// Subcomponent: ResourceItem (one row inside a group)
// =============================================================================


interface ResourceItemProps {
  resource: SkillResource;
  skillApiPath: string;
  authToken: string | null;
  onPreview: (resource: SkillResource) => void;
}


function ResourceItem({
  resource,
  skillApiPath,
  authToken,
  onPreview,
}: ResourceItemProps): React.ReactElement {
  const [downloading, setDownloading] = useState(false);
  const [downloadError, setDownloadError] = useState<string | null>(null);

  const _onDownload = useCallback(async () => {
    setDownloading(true);
    setDownloadError(null);
    try {
      const result = await _fetchResource(skillApiPath, resource.path, authToken);
      const blob = new Blob([result.content], { type: 'text/plain' });
      triggerBlobDownload(blob, _basename(resource.path));
    } catch (e: any) {
      const status = e?.response?.status;
      setDownloadError(status ? `HTTP ${status}` : 'fetch failed');
    } finally {
      setDownloading(false);
    }
  }, [skillApiPath, resource.path, authToken]);

  const isPreviewable = TEXT_PREVIEWABLE_TYPES.has(resource.type);

  return (
    <div className="flex items-center justify-between py-2 px-3 border-b border-gray-200 dark:border-gray-700 last:border-b-0 hover:bg-gray-50 dark:hover:bg-gray-800/50">
      <div className="flex-1 min-w-0">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="font-mono text-sm text-gray-900 dark:text-gray-100 truncate">
            {resource.path}
          </span>
          {resource.language && (
            <span className="px-2 py-0.5 text-xs font-semibold rounded bg-gray-100 dark:bg-gray-800 text-gray-700 dark:text-gray-300">
              {resource.language}
            </span>
          )}
          <span className="text-xs text-gray-500 dark:text-gray-400">
            {_formatSize(resource.size_bytes)}
          </span>
        </div>
        {downloadError && (
          <div className="text-xs text-red-600 dark:text-red-400 mt-1" role="alert">
            {downloadError}
          </div>
        )}
      </div>
      <div className="flex items-center gap-2 ml-3">
        {isPreviewable && (
          <button
            type="button"
            onClick={() => onPreview(resource)}
            aria-label={`View ${resource.path}`}
            className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded text-amber-700 dark:text-amber-300 hover:bg-amber-50 dark:hover:bg-amber-900/30"
          >
            <EyeIcon className="h-4 w-4" />
            View
          </button>
        )}
        <button
          type="button"
          onClick={_onDownload}
          disabled={downloading}
          aria-label={`Download ${resource.path}`}
          className="inline-flex items-center gap-1 px-2 py-1 text-xs font-medium rounded text-gray-700 dark:text-gray-300 hover:bg-gray-100 dark:hover:bg-gray-800 disabled:opacity-50 disabled:cursor-not-allowed"
        >
          <ArrowDownTrayIcon className="h-4 w-4" />
          {downloading ? 'Downloading…' : 'Download'}
        </button>
      </div>
    </div>
  );
}


// =============================================================================
// Subcomponent: ResourceGroup (collapsible section, one per type)
// =============================================================================


interface ResourceGroupProps {
  label: string;
  resources: SkillResource[];
  skillApiPath: string;
  authToken: string | null;
  onPreview: (resource: SkillResource) => void;
}


function ResourceGroup({
  label,
  resources,
  skillApiPath,
  authToken,
  onPreview,
}: ResourceGroupProps): React.ReactElement | null {
  // Default-collapsed (Resolution #2 in lld.md).
  const [expanded, setExpanded] = useState(false);

  if (resources.length === 0) return null;

  const sorted = _sortByPath(resources);

  return (
    <div className="rounded-lg border border-gray-200 dark:border-gray-700 overflow-hidden">
      <button
        type="button"
        onClick={() => setExpanded((e) => !e)}
        aria-expanded={expanded}
        className="w-full flex items-center justify-between px-3 py-2 bg-gray-50 dark:bg-gray-900/50 hover:bg-gray-100 dark:hover:bg-gray-800 text-left"
      >
        <span className="font-medium text-gray-800 dark:text-gray-200">
          {label} <span className="text-gray-500 dark:text-gray-400">({resources.length})</span>
        </span>
        {expanded ? (
          <ChevronDownIcon className="h-4 w-4 text-gray-500 dark:text-gray-400" />
        ) : (
          <ChevronRightIcon className="h-4 w-4 text-gray-500 dark:text-gray-400" />
        )}
      </button>
      {expanded && (
        <div className="bg-white dark:bg-gray-900">
          {sorted.map((r) => (
            <ResourceItem
              key={r.path}
              resource={r}
              skillApiPath={skillApiPath}
              authToken={authToken}
              onPreview={onPreview}
            />
          ))}
        </div>
      )}
    </div>
  );
}


// =============================================================================
// Subcomponent: ResourcePreview (in-modal text/markdown viewer)
// =============================================================================


interface ResourcePreviewProps {
  resource: SkillResource;
  skillApiPath: string;
  authToken: string | null;
  onBack: () => void;
}


function ResourcePreview({
  resource,
  skillApiPath,
  authToken,
  onBack,
}: ResourcePreviewProps): React.ReactElement {
  const [content, setContent] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  React.useEffect(() => {
    let cancelled = false;
    setContent(null);
    setError(null);
    _fetchResource(skillApiPath, resource.path, authToken)
      .then((result) => {
        if (!cancelled) setContent(result.content);
      })
      .catch((e: any) => {
        if (cancelled) return;
        const status = e?.response?.status;
        setError(status ? `HTTP ${status}` : 'fetch failed');
      });
    return () => {
      cancelled = true;
    };
  }, [skillApiPath, resource.path, authToken]);

  // NOTE: react-markdown disables raw HTML by default; do NOT add rehype-raw
  // without a security review (see review.md, Cipher recommendation #1).
  const isMarkdown = resource.path.toLowerCase().endsWith('.md');

  return (
    <div>
      <div className="flex items-center gap-2 mb-3 text-sm">
        <button
          type="button"
          onClick={onBack}
          className="inline-flex items-center gap-1 px-2 py-1 rounded text-amber-700 dark:text-amber-300 hover:bg-amber-50 dark:hover:bg-amber-900/30"
        >
          <ArrowLeftIcon className="h-4 w-4" />
          Back to SKILL.md
        </button>
        <span className="text-gray-400 dark:text-gray-500">›</span>
        <span className="font-mono text-gray-700 dark:text-gray-300 truncate">
          {resource.path}
        </span>
      </div>
      {error ? (
        <div className="text-center py-8 text-red-600 dark:text-red-400">
          Could not load resource: {error}
        </div>
      ) : content === null ? (
        <div className="flex items-center justify-center py-8">
          <div className="animate-spin rounded-full h-6 w-6 border-b-2 border-amber-600"></div>
        </div>
      ) : isMarkdown ? (
        <div className="prose prose-sm dark:prose-invert max-w-none prose-headings:text-amber-800 dark:prose-headings:text-amber-200 prose-a:text-amber-600 dark:prose-a:text-amber-400">
          <ReactMarkdown remarkPlugins={[remarkGfm]}>{content}</ReactMarkdown>
        </div>
      ) : (
        <pre className="bg-gray-100 dark:bg-gray-900 rounded p-4 overflow-x-auto text-xs">
          <code>{content}</code>
        </pre>
      )}
    </div>
  );
}


// =============================================================================
// Public component: SkillResources
// =============================================================================


type BundleStatus = 'idle' | 'preparing' | 'partial' | 'done' | 'error' | 'over-cap';


export interface SkillResourcesProps {
  skill: Skill;
  skillApiPath: string;
  // Accept undefined too -- SkillCardProps.authToken is `string | null | undefined`,
  // and we'd rather widen here than force callers to coalesce.
  authToken: string | null | undefined;
  skillMdContent: string | null;
  // Manifest captured from the /content API response. Passed in explicitly
  // because the skills LISTING schema (SkillInfo) intentionally omits
  // resource_manifest -- so skill.resource_manifest is undefined in the
  // Discover view and we can't read it off the skill prop.
  resourceManifest: SkillResourceManifest | null;
}


/**
 * Renders the "Resources" section inside a SkillCard modal.
 *
 * Self-gated:
 *   - returns null for federated skills (Resolution #1: registry_name !== 'local').
 *   - returns null when the manifest is absent or empty.
 */
export default function SkillResources({
  skill,
  skillApiPath,
  authToken,
  skillMdContent,
  resourceManifest,
}: SkillResourcesProps): React.ReactElement | null {
  // Federated skills are excluded for v1 (LLD Resolution #1).
  if (skill.registry_name && skill.registry_name !== 'local') return null;
  if (!resourceManifest) return null;
  if (!_hasAnyResource(resourceManifest)) return null;

  return (
    <SkillResourcesInner
      skill={skill}
      skillApiPath={skillApiPath}
      authToken={authToken ?? null}
      skillMdContent={skillMdContent}
      manifest={resourceManifest}
    />
  );
}


interface SkillResourcesInnerProps {
  skill: Skill;
  skillApiPath: string;
  authToken: string | null;
  skillMdContent: string | null;
  manifest: SkillResourceManifest;
}


// Inner component holds hook state, kept separate so the outer component
// can short-circuit before any hooks run.
function SkillResourcesInner({
  skill,
  skillApiPath,
  authToken,
  skillMdContent,
  manifest,
}: SkillResourcesInnerProps): React.ReactElement {
  const [bundleStatus, setBundleStatus] = useState<BundleStatus>('idle');
  const [bundleError, setBundleError] = useState<string | null>(null);
  const [progress, setProgress] = useState<{ done: number; total: number }>({ done: 0, total: 0 });
  const [selectedResource, setSelectedResource] = useState<SkillResource | null>(null);

  const all = _flattenManifest(manifest);
  const totalSize = all.reduce((sum, r) => sum + r.size_bytes, 0);
  const cap = _isOverCap(all);

  const _onDownloadAll = useCallback(async () => {
    setBundleStatus('preparing');
    setBundleError(null);
    setProgress({ done: 0, total: all.length });

    const capCheck = _isOverCap(all);
    if (capCheck.over) {
      setBundleStatus('over-cap');
      setBundleError(capCheck.reason);
      return;
    }

    // SKILL.md: prefer already-loaded content; fall back to fetch.
    let skillMd = skillMdContent;
    if (!skillMd) {
      try {
        const headers = authToken ? { Authorization: `Bearer ${authToken}` } : undefined;
        const resp = await axios.get(
          `/api/skills${skillApiPath}/content`,
          headers ? { headers } : undefined,
        );
        skillMd = resp.data.content as string;
      } catch {
        setBundleStatus('error');
        setBundleError('Could not fetch SKILL.md');
        return;
      }
    }

    const { successes, failures } = await _fetchAllWithConcurrency(
      skillApiPath,
      all,
      authToken,
      (done, total) => setProgress({ done, total }),
    );

    // All-fail guard (Resolution #3): do not deliver a SKILL.md-only zip
    // when every per-resource fetch failed.
    if (successes.length === 0 && all.length > 0) {
      setBundleStatus('error');
      setBundleError('Could not fetch any resources. Try again.');
      return;
    }

    await _buildAndDownloadZip(skill, skillMd!, successes);

    if (failures.length > 0) {
      setBundleStatus('partial');
      setBundleError(
        `${failures.length} file(s) failed: ${failures.map((f) => f.path).join(', ')}`,
      );
    } else {
      setBundleStatus('done');
      setBundleError(null);
    }
  }, [skill, skillApiPath, authToken, skillMdContent, all]);

  // Anti-double-click guard (Resolution #5): button is no-op while preparing.
  const _onDownloadAllClick = () => {
    if (bundleStatus === 'preparing') return;
    void _onDownloadAll();
  };

  if (selectedResource) {
    return (
      <ResourcePreview
        resource={selectedResource}
        skillApiPath={skillApiPath}
        authToken={authToken}
        onBack={() => setSelectedResource(null)}
      />
    );
  }

  return (
    <div className="mb-6">
      <div className="flex items-center justify-between mb-2">
        <h3 className="text-sm font-semibold text-gray-800 dark:text-gray-200">
          Resources{' '}
          <span className="font-normal text-gray-500 dark:text-gray-400">
            ({all.length} {all.length === 1 ? 'file' : 'files'}, {_formatSize(totalSize)})
          </span>
        </h3>
        <button
          type="button"
          onClick={_onDownloadAllClick}
          disabled={cap.over || bundleStatus === 'preparing'}
          aria-label="Download all classified resources as a zip"
          className="inline-flex items-center gap-1 px-3 py-1.5 text-xs font-medium rounded bg-amber-600 hover:bg-amber-700 text-white disabled:opacity-50 disabled:cursor-not-allowed"
          title={cap.over ? cap.reason : undefined}
        >
          <ArrowDownTrayIcon className="h-4 w-4" />
          {bundleStatus === 'preparing' ? 'Preparing…' : 'Download all'}
        </button>
      </div>

      <p className="text-xs text-gray-500 dark:text-gray-400 mb-3">
        Download all bundles classified resources only — files outside{' '}
        <code className="px-1 py-0.5 rounded bg-gray-100 dark:bg-gray-900">references/</code>,{' '}
        <code className="px-1 py-0.5 rounded bg-gray-100 dark:bg-gray-900">scripts/</code>,{' '}
        <code className="px-1 py-0.5 rounded bg-gray-100 dark:bg-gray-900">agents/</code>, and{' '}
        <code className="px-1 py-0.5 rounded bg-gray-100 dark:bg-gray-900">assets/</code> are not
        included.
      </p>

      {/* aria-live region for screen-reader announcements (Resolution #4). */}
      <div aria-live="polite" className="sr-only">
        {bundleStatus === 'preparing' && progress.total > 0 &&
          `${progress.done} of ${progress.total} files downloaded`}
        {bundleStatus === 'done' && 'Bundle ready'}
        {bundleStatus === 'partial' && 'Bundle ready with some failures'}
      </div>

      {bundleError && bundleStatus !== 'idle' && (
        <div
          role="alert"
          className={`mb-3 px-3 py-2 rounded text-sm ${
            bundleStatus === 'error' || bundleStatus === 'over-cap'
              ? 'bg-red-50 dark:bg-red-900/30 text-red-800 dark:text-red-200'
              : 'bg-amber-50 dark:bg-amber-900/30 text-amber-800 dark:text-amber-200'
          }`}
        >
          {bundleError}
        </div>
      )}

      <div className="space-y-2">
        <ResourceGroup
          label="References"
          resources={manifest.references}
          skillApiPath={skillApiPath}
          authToken={authToken}
          onPreview={setSelectedResource}
        />
        <ResourceGroup
          label="Scripts"
          resources={manifest.scripts}
          skillApiPath={skillApiPath}
          authToken={authToken}
          onPreview={setSelectedResource}
        />
        <ResourceGroup
          label="Agents"
          resources={manifest.agents}
          skillApiPath={skillApiPath}
          authToken={authToken}
          onPreview={setSelectedResource}
        />
        <ResourceGroup
          label="Assets"
          resources={manifest.assets}
          skillApiPath={skillApiPath}
          authToken={authToken}
          onPreview={setSelectedResource}
        />
      </div>
    </div>
  );
}


// Exported helpers for tests (pure functions; not part of the public API).
export const __test__ = {
  _slugify,
  _formatSize,
  _basename,
  _flattenManifest,
  _hasAnyResource,
  _isOverCap,
  _sortByPath,
  MAX_BUNDLE_FILES,
  MAX_BUNDLE_BYTES,
  FETCH_CONCURRENCY,
};
