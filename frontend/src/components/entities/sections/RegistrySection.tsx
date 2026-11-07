import React from 'react';
import {
  ChevronDownIcon,
  ChevronRightIcon,
  ArrowPathIcon,
} from '@heroicons/react/24/outline';
import EntityGrid from '../EntityGrid';

/**
 * Per-section color treatment for the federated (non-local) registry group.
 * Local groups always use the green/emerald treatment; this is the accent for
 * peer registries (cyan/blue for servers, violet/purple for agents).
 */
export interface RegistryAccent {
  headerBg: string;
  title: string;
  resyncButton: string;
  border: string;
}

interface RegistrySectionProps<T extends { path: string }> {
  /** Registry id (e.g. 'local' or 'peer-registry-lob1'). */
  registryId: string;
  /** DOM id for quick-nav scroll targeting (e.g. `server-registry-local`). */
  domId: string;
  /** Items in this registry, already filtered + paginated by the caller. */
  items: T[];
  /** Whether the group is expanded. */
  expanded: boolean;
  /** Toggle handler for the group header. */
  onToggle: () => void;
  /** Render one card for an item. */
  renderCard: (item: T) => React.ReactNode;
  /** Whether to render the collapsible header (false collapses to a bare grid). */
  showHeader: boolean;
  /** Registry endpoint URL shown in the header. */
  endpointUrl?: string;
  /** Count label, e.g. "3 servers". */
  countLabel: string;
  /** Display name, e.g. "Local Registry" or "LOB1 (Federated)". */
  displayName: string;
  /** Accent classes for the non-local (federated) header treatment. */
  accent: RegistryAccent;
  /** Resync handler for federated registries (omit for local). */
  onResync?: (e: React.MouseEvent) => void;
  /** Whether a resync is in flight. */
  syncing?: boolean;
  /**
   * Extra cards appended inside this group's grid (used to interleave virtual
   * servers into the local server registry). Rendered after the item cards.
   */
  extraCards?: React.ReactNode;
}

const LOCAL_HEADER_BG =
  'bg-gradient-to-r from-success-50 to-success-50 dark:from-success-900/20 dark:to-success-900/20 hover:from-success-100 hover:to-success-100 dark:hover:from-success-900/30 dark:hover:to-success-900/30';
const LOCAL_TITLE = 'text-success-700 dark:text-success-300';

/**
 * One registry group in the server/agent collections: an optional collapsible
 * header (registry name, endpoint, count, resync) wrapping a card grid. Shared
 * by the servers and agents sections, which differ only in accent color, the
 * cards they render, and the extra (virtual-server) cards interleaved into the
 * local group.
 *
 * When showHeader is false (single local registry), it renders just the grid,
 * matching the previous inline behavior.
 */
function RegistrySection<T extends { path: string }>({
  registryId,
  domId,
  items,
  expanded,
  onToggle,
  renderCard,
  showHeader,
  endpointUrl,
  countLabel,
  displayName,
  accent,
  onResync,
  syncing = false,
  extraCards,
}: RegistrySectionProps<T>): React.ReactElement {
  const isLocal = registryId === 'local';

  const grid = (
    <EntityGrid className="overflow-visible">
      {items.map((item) => renderCard(item))}
      {extraCards}
    </EntityGrid>
  );

  if (!showHeader) {
    return (
      <div className="overflow-visible">{grid}</div>
    );
  }

  return (
    <div
      id={domId}
      className={`border rounded-xl overflow-hidden scroll-mt-4 ${accent.border}`}
    >
      <button
        onClick={onToggle}
        className={`w-full flex items-center justify-between px-4 py-3 text-left transition-colors ${
          isLocal ? LOCAL_HEADER_BG : accent.headerBg
        }`}
      >
        <div className="flex items-center gap-3">
          {expanded ? (
            <ChevronDownIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          ) : (
            <ChevronRightIcon className="h-5 w-5 text-gray-500 dark:text-gray-400" />
          )}
          <span className={`font-semibold ${isLocal ? LOCAL_TITLE : accent.title}`}>
            {displayName}
          </span>
          <span
            className="text-xs text-gray-400 dark:text-gray-500 font-mono truncate max-w-[200px] lg:max-w-[300px]"
            title={endpointUrl}
          >
            | {endpointUrl || 'Loading...'}
          </span>
          <span className="px-2 py-0.5 text-xs font-medium bg-gray-100 dark:bg-gray-700 text-gray-600 dark:text-gray-300 rounded-full">
            {countLabel}
          </span>
          {!isLocal && onResync && (
            <button
              onClick={onResync}
              disabled={syncing}
              className={`ml-2 p-1 rounded-lg transition-colors disabled:opacity-50 ${accent.resyncButton}`}
              title={`Resync from ${endpointUrl || registryId}`}
            >
              <ArrowPathIcon className={`h-4 w-4 ${syncing ? 'animate-spin' : ''}`} />
            </button>
          )}
        </div>
      </button>

      {expanded && (
        <div className="p-4 bg-canvas overflow-visible">{grid}</div>
      )}
    </div>
  );
}

export default RegistrySection;
