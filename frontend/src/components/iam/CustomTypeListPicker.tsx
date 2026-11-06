import React, { useEffect, useMemo, useState } from 'react';
import axios from 'axios';
import { XMarkIcon } from '@heroicons/react/24/outline';
import SearchableSelect from '../SearchableSelect';

/**
 * Grant control for a custom type's ``list_<type>_entity`` discovery scope.
 *
 * The discovery grant is interpreted by the backend in three tiers (mirroring
 * ``list_agents``): ``"all"`` opens the whole type; a record path ``/type/uuid``
 * opens just that record. This control surfaces those two tiers:
 *
 * - an "All" toggle (writes ``all``), and
 * - a multi-select of specific records (writes their PATHS; the record NAME is
 *   shown for readability), fetched from ``GET /api/custom/{type}``.
 *
 * Granting one record does NOT expose other records of the type — the backend
 * matches each record's path, so this stays consistent with server/agent access.
 * The value is stored as a comma-separated string in the shared uiPermissions
 * state (``setValue``), exactly like the other permission inputs.
 */

interface CustomTypeListPickerProps {
  typeName: string;
  displayName: string;
  /** Current CSV value of list_<type>_entity (e.g. "all" or "/type/uuid, ..."). */
  value: string;
  /** Persist a new CSV value (empty string clears the grant). */
  onChange: (csv: string) => void;
}

interface RecordOption {
  path: string;
  name: string;
}

function _csvToList(csv: string): string[] {
  return csv
    .split(',')
    .map((v) => v.trim())
    .filter(Boolean);
}

const CustomTypeListPicker: React.FC<CustomTypeListPickerProps> = ({
  typeName,
  displayName,
  value,
  onChange,
}) => {
  const [records, setRecords] = useState<RecordOption[]>([]);
  const [loading, setLoading] = useState(false);

  // Load the type's records so specific ones can be picked by name. Admins hit
  // this editor, so the list returns every record of the type. Best-effort: a
  // failure just leaves the picker empty (the All toggle + free-text still work).
  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    axios
      .get<{ records: RecordOption[] }>(`/api/custom/${typeName}`, {
        params: { limit: 1000 },
      })
      .then((res) => {
        if (cancelled) return;
        setRecords(
          (res.data.records ?? []).map((r) => ({ path: r.path, name: r.name })),
        );
      })
      .catch(() => {
        if (!cancelled) setRecords([]);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [typeName]);

  const items = _csvToList(value);
  const grantsAll = items.includes('all');
  const selectedPaths = useMemo(() => items.filter((v) => v !== 'all'), [items]);

  // Map a stored path back to a display label (name), falling back to the raw
  // path if the record is gone or not yet loaded.
  const labelForPath = useMemo(() => {
    const byPath = new Map(records.map((r) => [r.path, r.name]));
    return (path: string) => byPath.get(path) || path;
  }, [records]);

  const options = useMemo(
    () =>
      records
        .filter((r) => !selectedPaths.includes(r.path))
        .map((r) => ({ value: r.path, label: r.name, description: r.path })),
    [records, selectedPaths],
  );

  const setAll = (checked: boolean) => {
    // "All" is exclusive: it supersedes any specific-record selections.
    onChange(checked ? 'all' : selectedPaths.join(', '));
  };
  const addPath = (path: string) => {
    if (!path || selectedPaths.includes(path)) return;
    onChange([...selectedPaths, path].join(', '));
  };
  const removePath = (path: string) => {
    onChange(selectedPaths.filter((p) => p !== path).join(', '));
  };

  return (
    <div className="space-y-1.5">
      <div className="flex items-center justify-between">
        <label className="block text-xs font-medium text-fg-muted">
          List {displayName}
        </label>
        <label
          htmlFor={`all-list_${typeName}_entity`}
          className="flex items-center space-x-1.5 text-xs text-fg-subtle cursor-pointer"
        >
          <input
            id={`all-list_${typeName}_entity`}
            type="checkbox"
            checked={grantsAll}
            onChange={(e) => setAll(e.target.checked)}
            className="h-3.5 w-3.5 rounded border-line-strong text-purple-600 focus:ring-focus"
          />
          <span>All</span>
        </label>
      </div>
      {!grantsAll && (
        <>
          {selectedPaths.length > 0 && (
            <div className="flex flex-wrap gap-1.5">
              {selectedPaths.map((path) => (
                <span
                  key={path}
                  className="inline-flex items-center gap-1 px-2 py-0.5 text-xs rounded-full
                             bg-surface-hover text-fg"
                  title={path}
                >
                  {labelForPath(path)}
                  <button
                    type="button"
                    onClick={() => removePath(path)}
                    className="text-fg-subtle hover:text-fg"
                    aria-label={`Remove ${labelForPath(path)}`}
                  >
                    <XMarkIcon className="h-3 w-3" />
                  </button>
                </span>
              ))}
            </div>
          )}
          <SearchableSelect
            options={options}
            value=""
            onChange={addPath}
            placeholder={
              records.length > 0 ? 'Add a specific record…' : 'No records yet'
            }
            isLoading={loading}
            allowCustom={false}
          />
        </>
      )}
    </div>
  );
};

export default CustomTypeListPicker;
