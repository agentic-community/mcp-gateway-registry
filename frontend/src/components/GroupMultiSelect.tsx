import React, { useEffect, useMemo, useRef, useState } from 'react';
import { ChevronDownIcon } from '@heroicons/react/24/outline';
import { useIAMGroups } from '../hooks/useIAM';

interface GroupMultiSelectProps {
  id: string;
  selected: string[];
  onChange: (groups: string[]) => void;
}


function _parseGroups(text: string): string[] {
  return text.split(',').map((s) => s.trim()).filter(Boolean);
}


/**
 * Text fallback used when the group list cannot load. It keeps the raw text,
 * so a trailing comma survives and a second group can be typed, and reports
 * the parsed list on every change.
 */
const _GroupTextField: React.FC<GroupMultiSelectProps> = ({ id, selected, onChange }) => {
  const [text, setText] = useState(selected.join(', '));
  return (
    <>
      <input
        id={id}
        type="text"
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          onChange(_parseGroups(e.target.value));
        }}
        placeholder="weather-team, data-science (comma-separated)"
        className="input"
      />
      <p className="mt-1 text-xs text-fg-subtle">Could not load the group list; type group names instead.</p>
    </>
  );
};


/**
 * Pick allowed groups from the registry's existing groups (admin-only list).
 *
 * A group already selected but missing from the list stays shown and checked,
 * so it can be removed. If the list cannot be loaded, this falls back to a
 * comma-separated text field so the form still works. Escape and a click
 * outside close only the dropdown, never the form around it.
 */
const GroupMultiSelect: React.FC<GroupMultiSelectProps> = ({ id, selected, onChange }) => {
  const { groups, isLoading, error } = useIAMGroups();
  const [open, setOpen] = useState(false);
  const [filter, setFilter] = useState('');
  const wrapper = useRef<HTMLDivElement>(null);

  const names = useMemo(() => {
    const known = (groups || []).map((g) => g.name).filter(Boolean);
    const all = Array.from(new Set([...known, ...selected]));
    return all.sort((a, b) => a.localeCompare(b));
  }, [groups, selected]);

  useEffect(() => {
    if (!open) return undefined;
    const closeOnOutsideClick = (e: MouseEvent) => {
      if (wrapper.current && !wrapper.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('mousedown', closeOnOutsideClick);
    return () => document.removeEventListener('mousedown', closeOnOutsideClick);
  }, [open]);

  if (error) {
    return <_GroupTextField id={id} selected={selected} onChange={onChange} />;
  }

  const toggle = (name: string) => {
    onChange(selected.includes(name) ? selected.filter((g) => g !== name) : [...selected, name]);
  };
  const visible = names.filter((n) => n.toLowerCase().includes(filter.trim().toLowerCase()));
  const panelId = `${id}-panel`;

  // The form closes on Escape through a document listener; stopping the event
  // here closes only the open dropdown.
  const onKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape' && open) {
      e.stopPropagation();
      setOpen(false);
    }
  };

  return (
    <div className="relative" ref={wrapper} onKeyDown={onKeyDown}>
      <button
        id={id}
        type="button"
        aria-expanded={open}
        aria-controls={panelId}
        onClick={() => setOpen(!open)}
        className="input flex items-center justify-between text-left"
      >
        <span className={selected.length ? '' : 'text-fg-subtle'}>
          {isLoading ? 'Loading groups...' : selected.length ? selected.join(', ') : 'Choose groups'}
        </span>
        <ChevronDownIcon className="h-4 w-4 shrink-0" aria-hidden="true" />
      </button>
      {open && !isLoading && (
        <div id={panelId} className="absolute z-10 mt-1 w-full rounded-md border border-line bg-surface-overlay shadow-lg">
          <input
            type="text"
            aria-label="Filter groups"
            value={filter}
            onChange={(e) => setFilter(e.target.value)}
            placeholder="Filter groups"
            className="input rounded-b-none border-0 border-b"
          />
          <fieldset className="max-h-56 overflow-y-auto py-1">
            <legend className="sr-only">Allowed groups</legend>
            {visible.length === 0 && <p className="px-3 py-2 text-sm text-fg-subtle">No matching groups</p>}
            {visible.map((name) => (
              <label key={name} className="flex cursor-pointer items-center gap-2 px-3 py-1.5 text-sm hover:bg-surface-hover">
                <input type="checkbox" checked={selected.includes(name)} onChange={() => toggle(name)} />
                <span>{name}</span>
              </label>
            ))}
          </fieldset>
        </div>
      )}
    </div>
  );
};

export default GroupMultiSelect;
