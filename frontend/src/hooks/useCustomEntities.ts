import { useState, useEffect, useCallback, useMemo } from 'react';
import axios from 'axios';
import {
  CustomEntityCreate,
  CustomEntityRecord,
  CustomEntityUpdate,
  CustomTypeDescriptor,
} from '../types/customEntity';
import { useServerStats } from './useServerStats';

interface UseCustomEntitiesReturn {
  /** The full type descriptor (fetched once per tab). Null until loaded. */
  descriptor: CustomTypeDescriptor | null;
  /** Records the caller may see, sorted by name client-side. */
  records: CustomEntityRecord[];
  loading: boolean;
  /** Set when the type no longer exists (stale tab) — drives the empty state. */
  notFound: boolean;
  error: string | null;
  refreshData: () => Promise<void>;
  createRecord: (body: CustomEntityCreate) => Promise<CustomEntityRecord>;
  updateRecord: (uuid: string, body: CustomEntityUpdate) => Promise<CustomEntityRecord>;
  deleteRecord: (uuid: string) => Promise<void>;
}

// Descriptors are cached per type across tab mounts so switching back into a
// custom-entity tab doesn't refetch GET /api/custom-types/{name} every time
// (the tab unmounts when inactive and remounts fresh).
//
// Bounded TTL (not cache-forever): v1 has no descriptor PUT, but a type CAN be
// deleted and recreated under the same name with a DIFFERENT schema. A
// name-keyed forever-cache would then render the old schema for the whole page
// lifetime. The TTL bounds that staleness to a short window (mirrors the
// backend's custom_type_cache_ttl_seconds) without refetching on every tab
// switch. We cannot key by schema_version because we don't know it until after
// the fetch.
const _DESCRIPTOR_CACHE_TTL_MS = 60_000;

interface CachedDescriptor {
  descriptor: CustomTypeDescriptor;
  fetchedAt: number;
}

const _descriptorCache = new Map<string, CachedDescriptor>();

function _getCachedDescriptor(typeName: string): CustomTypeDescriptor | null {
  const entry = _descriptorCache.get(typeName);
  if (!entry) return null;
  if (Date.now() - entry.fetchedAt > _DESCRIPTOR_CACHE_TTL_MS) {
    _descriptorCache.delete(typeName);
    return null;
  }
  return entry.descriptor;
}

function _uuidOf(path: string): string {
  // Synthetic path is /{type}/{uuid}; the uuid is the last segment.
  return path.split('/').pop() ?? '';
}

function _sortByName(records: CustomEntityRecord[]): CustomEntityRecord[] {
  return [...records].sort((a, b) => a.name.localeCompare(b.name));
}

/**
 * Loads one custom type's descriptor (once) and exposes its visible records.
 *
 * Records are sourced from the shared `ServerStatsContext` — the SAME single
 * fetch that backs the sidebar counts and Discover summary — so switching
 * between tabs does not re-read the record list (matching the built-in
 * server/agent tabs). Only the immutable descriptor is fetched per tab mount.
 * Mutations hit the API then refresh the shared context.
 */
export const useCustomEntities = (typeName: string): UseCustomEntitiesReturn => {
  const { customRecordsByType, loading: recordsLoading, refreshData: refreshStats } =
    useServerStats();
  const [descriptor, setDescriptor] = useState<CustomTypeDescriptor | null>(null);
  const [descriptorLoading, setDescriptorLoading] = useState(true);
  const [notFound, setNotFound] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const records = useMemo(() => {
    const forType = customRecordsByType.find((t) => t.name === typeName);
    return _sortByName((forType?.records ?? []) as CustomEntityRecord[]);
  }, [customRecordsByType, typeName]);

  // Resolve the immutable descriptor. Served from the module cache on a hit so
  // re-entering a tab doesn't refetch; only a cache miss hits the network.
  useEffect(() => {
    const cached = _getCachedDescriptor(typeName);
    if (cached) {
      setDescriptor(cached);
      setDescriptorLoading(false);
      setError(null);
      setNotFound(false);
      return;
    }

    let cancelled = false;
    setDescriptorLoading(true);
    setError(null);
    setNotFound(false);
    axios
      .get<CustomTypeDescriptor>(`/api/custom-types/${typeName}`)
      .then((res) => {
        _descriptorCache.set(typeName, { descriptor: res.data, fetchedAt: Date.now() });
        if (!cancelled) setDescriptor(res.data);
      })
      .catch((err: unknown) => {
        if (cancelled) return;
        const status = axios.isAxiosError(err) ? err.response?.status : undefined;
        if (status === 404) {
          // Type no longer exists — drop any stale cache entry so a later
          // recreate under the same name re-fetches the new schema.
          _descriptorCache.delete(typeName);
          setNotFound(true);
        } else {
          console.error(`Failed to load custom type ${typeName}:`, err);
          const detail = axios.isAxiosError(err) ? err.response?.data?.detail : undefined;
          setError(detail || 'Failed to load type');
        }
      })
      .finally(() => {
        if (!cancelled) setDescriptorLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [typeName]);

  const createRecord = useCallback(
    async (body: CustomEntityCreate): Promise<CustomEntityRecord> => {
      const res = await axios.post<CustomEntityRecord>(`/api/custom/${typeName}`, body);
      await refreshStats();
      return res.data;
    },
    [typeName, refreshStats],
  );

  const updateRecord = useCallback(
    async (uuid: string, body: CustomEntityUpdate): Promise<CustomEntityRecord> => {
      const res = await axios.put<CustomEntityRecord>(
        `/api/custom/${typeName}/${uuid}`,
        body,
      );
      await refreshStats();
      return res.data;
    },
    [typeName, refreshStats],
  );

  const deleteRecord = useCallback(
    async (uuid: string): Promise<void> => {
      await axios.delete(`/api/custom/${typeName}/${uuid}`);
      await refreshStats();
    },
    [typeName, refreshStats],
  );

  return {
    descriptor,
    records,
    loading: descriptorLoading || recordsLoading,
    notFound,
    error,
    refreshData: refreshStats,
    createRecord,
    updateRecord,
    deleteRecord,
  };
};

export { _uuidOf as uuidFromPath };
