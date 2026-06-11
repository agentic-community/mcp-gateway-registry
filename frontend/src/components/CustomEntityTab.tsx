import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import {
  PlusIcon,
  ArrowPathIcon,
  ChevronLeftIcon,
  ChevronRightIcon,
  MagnifyingGlassIcon,
  XMarkIcon,
} from '@heroicons/react/24/outline';
import { CustomEntityRecord } from '../types/customEntity';
import { useCustomEntities, uuidFromPath } from '../hooks/useCustomEntities';
import { useSemanticSearch } from '../hooks/useSemanticSearch';
import { labelFor } from '../utils/humanize';
import CustomEntityCard from './CustomEntityCard';
import CustomEntityDetail from './CustomEntityDetail';
import CustomEntityForm from './CustomEntityForm';
import SemanticSearchResults from './SemanticSearchResults';
import ConfirmModal from './ConfirmModal';

interface CurrentUser {
  username?: string;
  is_admin?: boolean;
}

// Records per page in the local listing. Records for a type are loaded into the
// shared context in one fetch (like servers/agents), so pagination here is
// client-side: it keeps the grid manageable when a type has many records
// (e.g. dozens of model cards) without changing the data flow.
const PAGE_SIZE = 24;

interface CustomEntityTabProps {
  typeName: string;
  displayName: string;
  user: CurrentUser | null;
  selectedTags?: string[];
  authToken?: string | null;
  onShowToast?: (message: string, type: 'success' | 'error') => void;
}

/**
 * Self-contained tab for one custom entity type. Fetches the descriptor once
 * (via useCustomEntities), lists records, and hosts create/edit/detail/delete
 * modals. A deleted type surfaces as the stale-tab empty state.
 */
const CustomEntityTab: React.FC<CustomEntityTabProps> = ({
  typeName,
  displayName,
  user,
  selectedTags = [],
  authToken,
  onShowToast,
}) => {
  const {
    descriptor,
    records,
    loading,
    notFound,
    error,
    createRecord,
    updateRecord,
    deleteRecord,
  } = useCustomEntities(typeName);

  const [showForm, setShowForm] = useState(false);
  const [editing, setEditing] = useState<CustomEntityRecord | null>(null);
  const [viewing, setViewing] = useState<CustomEntityRecord | null>(null);
  const [deleting, setDeleting] = useState<CustomEntityRecord | null>(null);
  const [deleteLoading, setDeleteLoading] = useState(false);
  const [searchTerm, setSearchTerm] = useState('');
  // Set on Enter to run a semantic search (mirrors the built-in Dashboard tabs);
  // typing alone only filters locally.
  const [committedQuery, setCommittedQuery] = useState('');
  // Current page (0-based) for the local listing.
  const [page, setPage] = useState(0);

  // Cross-type semantic search, matching the built-in tabs: pressing Enter
  // searches across ALL entity types (servers, agents, skills, custom, ...),
  // not just this tab's type, and renders the shared SemanticSearchResults view.
  const semanticEnabled = committedQuery.trim().length >= 2;
  const {
    results: semanticResults,
    loading: semanticLoading,
    error: semanticError,
  } = useSemanticSearch(committedQuery, {
    enabled: semanticEnabled,
  });

  // Clearing the input exits semantic mode.
  useEffect(() => {
    if (searchTerm.trim().length === 0 && committedQuery.length > 0) {
      setCommittedQuery('');
    }
  }, [searchTerm, committedQuery]);

  const canModify = (record: CustomEntityRecord): boolean =>
    !!user?.is_admin || record.owner === user?.username;

  // Client-side tag filter, matching the case-insensitive "all selected tags"
  // semantics used by the server/agent/skill tabs in Dashboard.
  const tagFiltered =
    selectedTags.length === 0
      ? records
      : records.filter((record) => {
          if (!record.tags || record.tags.length === 0) return false;
          const lowerTags = record.tags.map((t) => t.toLowerCase());
          return selectedTags.every((st) => lowerTags.includes(st.toLowerCase()));
        });

  // Text search over name, description, tags, and stringified attribute values.
  const query = searchTerm.trim().toLowerCase();
  const visibleRecords = query
    ? tagFiltered.filter((record) => {
        if (record.name.toLowerCase().includes(query)) return true;
        if ((record.description || '').toLowerCase().includes(query)) return true;
        if (record.tags.some((t) => t.toLowerCase().includes(query))) return true;
        return Object.values(record.attributes).some((v) =>
          String(v ?? '').toLowerCase().includes(query),
        );
      })
    : tagFiltered;

  // Client-side pagination over the filtered set.
  const totalPages = Math.max(1, Math.ceil(visibleRecords.length / PAGE_SIZE));
  // Clamp the page if the filtered set shrank (e.g. a new search narrows results).
  const safePage = Math.min(page, totalPages - 1);
  const pagedRecords = useMemo(
    () => visibleRecords.slice(safePage * PAGE_SIZE, safePage * PAGE_SIZE + PAGE_SIZE),
    [visibleRecords, safePage],
  );

  // Reset to the first page whenever the filter inputs change, so the user
  // isn't stranded on an out-of-range page after narrowing the results.
  useEffect(() => {
    setPage(0);
  }, [query, selectedTags, typeName]);

  const openCreate = () => {
    setEditing(null);
    setShowForm(true);
  };

  const openEdit = (record: CustomEntityRecord) => {
    setEditing(record);
    setShowForm(true);
  };

  const handleSave = async (body: any) => {
    if (editing) {
      await updateRecord(uuidFromPath(editing.path), body);
      onShowToast?.(`Updated ${body.name}`, 'success');
    } else {
      await createRecord(body);
      onShowToast?.(`Created ${body.name}`, 'success');
    }
    setShowForm(false);
    setEditing(null);
  };

  const handleDelete = async () => {
    if (!deleting) return;
    setDeleteLoading(true);
    try {
      await deleteRecord(uuidFromPath(deleting.path));
      onShowToast?.(`Deleted ${deleting.name}`, 'success');
      setDeleting(null);
    } catch (err: unknown) {
      const detail = axios.isAxiosError(err) ? err.response?.data?.detail : undefined;
      onShowToast?.(detail || 'Failed to delete', 'error');
    } finally {
      setDeleteLoading(false);
    }
  };

  if (loading && !descriptor) {
    return (
      <div className="flex items-center justify-center py-16 text-gray-500 dark:text-gray-400">
        <ArrowPathIcon className="h-5 w-5 animate-spin mr-2" />
        Loading {displayName}…
      </div>
    );
  }

  // Stale tab: the type was deleted while this tab was open.
  if (notFound || !descriptor) {
    return (
      <div className="text-center py-16">
        <p className="text-gray-500 dark:text-gray-400">
          This type no longer exists — refresh to update your tabs.
        </p>
      </div>
    );
  }

  return (
    <div className="space-y-4">
      {/* Search Bar and Create Button (mirrors the built-in tabs' layout) */}
      <div className="flex gap-4 items-center">
        <div className="relative flex-1">
          <div className="absolute inset-y-0 left-0 flex items-center pl-3 pointer-events-none">
            <MagnifyingGlassIcon className="h-5 w-5 text-gray-400" />
          </div>
          <input
            type="text"
            value={searchTerm}
            onChange={(e) => setSearchTerm(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') {
                e.preventDefault();
                setCommittedQuery(searchTerm.trim());
              }
            }}
            placeholder={`Search ${displayName}… (Press Enter for semantic search; typing filters locally.)`}
            className="input pl-10 pr-9 w-full"
          />
          {searchTerm && (
            <button
              type="button"
              onClick={() => {
                setSearchTerm('');
                setCommittedQuery('');
              }}
              className="absolute inset-y-0 right-0 flex items-center pr-3
                text-gray-400 hover:text-gray-600 dark:hover:text-gray-200"
              aria-label="Clear search"
            >
              <XMarkIcon className="h-4 w-4" />
            </button>
          )}
        </div>

        <button
          type="button"
          onClick={openCreate}
          className="btn-primary flex items-center space-x-2 flex-shrink-0"
        >
          <PlusIcon className="h-4 w-4" />
          <span>Create</span>
        </button>
      </div>

      {/* Results count (local listing only; semantic mode shows its own view) */}
      {!semanticEnabled && (
        <div className="flex items-center justify-between">
          <p className="text-sm text-gray-500 dark:text-gray-300">
            {visibleRecords.length === 0
              ? `Showing 0 ${displayName}`
              : `Showing ${safePage * PAGE_SIZE + 1}–${
                  safePage * PAGE_SIZE + pagedRecords.length
                } of ${visibleRecords.length} ${displayName}`}
          </p>
          {descriptor.description && (
            <p className="text-xs text-gray-400 dark:text-gray-500">
              {descriptor.description}
            </p>
          )}
        </div>
      )}

      {error && (
        <div className="p-3 bg-red-50 dark:bg-red-900/20 border border-red-200 dark:border-red-800 rounded-lg">
          <p className="text-sm text-red-700 dark:text-red-300">{error}</p>
        </div>
      )}

      {semanticEnabled ? (
        <SemanticSearchResults
          query={semanticResults?.query || committedQuery}
          loading={semanticLoading}
          error={semanticError}
          servers={semanticResults?.servers ?? []}
          tools={semanticResults?.tools ?? []}
          agents={semanticResults?.agents ?? []}
          skills={semanticResults?.skills ?? []}
          virtualServers={semanticResults?.virtual_servers ?? []}
          custom={semanticResults?.custom ?? []}
        />
      ) : visibleRecords.length === 0 ? (
        selectedTags.length > 0 || query ? (
          <div className="text-center py-16 border border-dashed border-gray-300 dark:border-gray-700 rounded-xl">
            <p className="text-gray-500 dark:text-gray-400">
              No {displayName} match your {query && selectedTags.length > 0
                ? 'search and selected tags'
                : query
                ? 'search'
                : 'selected tags'}.
            </p>
          </div>
        ) : (
          <div className="text-center py-16 border border-dashed border-gray-300 dark:border-gray-700 rounded-xl">
            <p className="text-gray-500 dark:text-gray-400 mb-4">
              No {displayName} yet — create one.
            </p>
            <button
              type="button"
              onClick={openCreate}
              className="inline-flex items-center gap-2 px-4 py-2 text-sm font-medium text-white
                         bg-teal-600 rounded-lg hover:bg-teal-700 transition-colors"
            >
              <PlusIcon className="h-4 w-4" />
              Create {labelFor({ name: descriptor.name })}
            </button>
          </div>
        )
      ) : (
        <>
          <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
            {pagedRecords.map((record) => (
              <CustomEntityCard
                key={record.path}
                descriptor={descriptor}
                record={record}
                canModify={canModify(record)}
                onView={setViewing}
                onEdit={openEdit}
                onDelete={setDeleting}
                authToken={authToken}
                onShowToast={onShowToast}
              />
            ))}
          </div>

          {/* Pagination controls (shown only when there's more than one page) */}
          {totalPages > 1 && (
            <div className="flex items-center justify-center gap-4 pt-2">
              <button
                type="button"
                onClick={() => setPage((p) => Math.max(0, p - 1))}
                disabled={safePage === 0}
                className="inline-flex items-center gap-1 px-3 py-1.5 text-sm font-medium rounded-lg
                           text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700
                           hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors
                           disabled:opacity-50 disabled:cursor-not-allowed"
              >
                <ChevronLeftIcon className="h-4 w-4" />
                Previous
              </button>
              <span className="text-sm text-gray-500 dark:text-gray-400">
                Page {safePage + 1} of {totalPages}
              </span>
              <button
                type="button"
                onClick={() => setPage((p) => Math.min(totalPages - 1, p + 1))}
                disabled={safePage >= totalPages - 1}
                className="inline-flex items-center gap-1 px-3 py-1.5 text-sm font-medium rounded-lg
                           text-gray-700 dark:text-gray-300 bg-gray-100 dark:bg-gray-700
                           hover:bg-gray-200 dark:hover:bg-gray-600 transition-colors
                           disabled:opacity-50 disabled:cursor-not-allowed"
              >
                Next
                <ChevronRightIcon className="h-4 w-4" />
              </button>
            </div>
          )}
        </>
      )}

      {showForm && (
        <CustomEntityForm
          descriptor={descriptor}
          record={editing}
          onSave={handleSave}
          onCancel={() => {
            setShowForm(false);
            setEditing(null);
          }}
        />
      )}

      {viewing && (
        <CustomEntityDetail
          descriptor={descriptor}
          record={viewing}
          onClose={() => setViewing(null)}
        />
      )}

      <ConfirmModal
        isOpen={!!deleting}
        onClose={() => setDeleting(null)}
        onConfirm={handleDelete}
        title={`Delete ${displayName}`}
        message={`Are you sure you want to delete "${deleting?.name}"? This cannot be undone.`}
        confirmLabel="Delete"
        loadingLabel="Deleting..."
        isDestructive
        isLoading={deleteLoading}
      />
    </div>
  );
};

export default CustomEntityTab;
