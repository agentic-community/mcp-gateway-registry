import React, { useMemo, useState } from 'react';
import { ChevronDownIcon } from '@heroicons/react/24/outline';
import { useIAMGroups } from '../hooks/useIAM';

interface GroupMultiSelectProps {
  id: string;
  selected: string[];
  onChange: (groups: string[]) => void;
}

/**
 * Pick allowed groups from the registry's existing groups (admin-only list).
 *
 * A group already selected but missing from the list stays shown and checked,
 * so it can be removed. If the list cannot be loaded, this falls back to a
 * comma-separated text field so the form still works.
 */
const GroupMultiSelect: React.FC<GroupMultiSelectProps> = ({ id, selected, onChange }) => {
  const { groups, isLoading, error } = useIAMGroups();
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState('');

  const names = useMemo(() => {
    const known = (groups || []).map((g) => g.name).filter(Boolean);
    const all = Array.from(new Set([...known, ...selected]));
    return all.sort((a, b) => a.localeCompare(b));
  }, [groups, selected]);

  if (error) {
    return (
      <>
        <input
          id={id}
          type="text"
          value={selected.join(', ')}
          onChange={(e) => onChange(e.target.value.split(',').map((s) => s.trim()).filter(Boolean))}
          placeholder="weather-team, data-science (comma-separated)"
          className="input"
        />
        <p className="mt-1 text-xs text-fg-subtle">Could not load the group list; type group names instead.</p>
      </>
    );
  }

  const toggle = (name: string) => {
    onChange(selected.includes(name) ? selected.filter((g) => g !== name) : [...selected, name]);
  };
  const visible = names.filter((n) => n.toLowerCase().includes(filter.trim().toLowerCase()));

  return (
    <div className="relative">
      <button
        id={id}
        type="button"
        aria-haspopup="listbox"
        aria-expanded={open}
        onClick={() => setOpen(!open)}
        className="input flex items-center justify-between text-left"
      >
        <span className={selected.length ? '' : 'text-fg-subtle'}>
          {isLoading ? 'Loading groups...' : selected.length ? selected.join(', ') : 'Choose groups'}
        </span>
        <ChevronDownIcon className="h-4 w-4 shrink-0" aria-hidden="true" />
      </button>
      {open && !isLoading && (
        <div className="absolute z-10 mt-1 w-full rounded-md border border-line bg-surface-overlay shadow-lg">
          <input
            type="text"
            aria-label="Filter groups"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter groups"
            className="input rounded-b-none border-0 border-b"
          />
          <ul role="listbox" aria-multiselectable="true" className="max-h-56 overflow-y-auto py-1">
            {visible.length === 0 && <li className="px-3 py-2 text-sm text-fg-subtle">No matching groups</li>}
            {visible.map((name) => (
              <li key={name} role="option" aria-selected={selected.includes(name)}>
                <label className="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-sm hover:bg-surface-hover">
                  <input type="checkbox" checked={selected.includes(name)} onChange={() => toggle(name)} />
                  <span>{name}</span>
                </label>
              </li>
            ))}
          </ul>
        </div>
      )}
    </div>
  );
};

export default GroupMultiSelect;
