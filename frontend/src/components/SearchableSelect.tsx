/**
 * Searchable select component with autocomplete functionality.
 *
 * Displays a text input that filters options as you type,
 * showing results in a dropdown list below.
 */

import React, { useState, useRef, useEffect, useCallback } from 'react';
import { createPortal } from 'react-dom';
import { MagnifyingGlassIcon, XMarkIcon } from '@heroicons/react/24/outline';


export interface SelectOption {
  value: string;
  label: string;
  description?: string;
}

interface SearchableSelectProps {
  options: SelectOption[];
  value: string;
  onChange: (value: string) => void;
  placeholder?: string;
  disabled?: boolean;
  isLoading?: boolean;
  maxDescriptionWords?: number;
  allowCustom?: boolean;  // Allow entering values not in the list
  specialOptions?: SelectOption[];  // Options shown at top (e.g., "* All")
  focusColor?: string;
  /**
   * Replaces the built-in "No options available" when the list is empty and
   * nothing has been typed. For controls whose only input mode is free text
   * (no enumerable option set) this is where you say so.
   */
  emptyMessage?: string;
}


/**
 * Truncate text to a maximum number of words.
 */
function _truncateWords(text: string, maxWords: number): string {
  const words = text.split(/\s+/);
  if (words.length <= maxWords) return text;
  return words.slice(0, maxWords).join(' ') + '...';
}


const SearchableSelect: React.FC<SearchableSelectProps> = ({
  options,
  value,
  onChange,
  placeholder = 'Search...',
  disabled = false,
  isLoading = false,
  maxDescriptionWords = 8,
  allowCustom = false,
  specialOptions = [],
  focusColor,
  emptyMessage,
}) => {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  // The dropdown renders in a portal (fixed position) so an ancestor with
  // overflow (e.g. a table wrapper's overflow-x-auto) cannot clip it. We track
  // the input's viewport rect to position it.
  const [menuRect, setMenuRect] = useState<{ top: number; left: number; width: number } | null>(
    null,
  );

  const _updateMenuRect = useCallback(() => {
    const el = inputRef.current;
    if (!el) return;
    const r = el.getBoundingClientRect();
    setMenuRect({ top: r.bottom + 4, left: r.left, width: r.width });
  }, []);

  // Find the selected option to display its label
  const selectedOption = [...specialOptions, ...options].find((o) => o.value === value);

  // Filter options based on search query
  const filteredOptions = options.filter((option) => {
    const query = searchQuery.toLowerCase();
    return (
      option.label.toLowerCase().includes(query) ||
      option.value.toLowerCase().includes(query) ||
      (option.description?.toLowerCase().includes(query) ?? false)
    );
  });

  // Close dropdown when clicking outside. The dropdown lives in a portal, so a
  // click on an option is NOT inside containerRef; also treat clicks inside the
  // portaled dropdown as "inside" so selecting an option doesn't close-before-select.
  useEffect(() => {
    const handleClickOutside = (event: MouseEvent) => {
      const target = event.target as Node;
      const inContainer = containerRef.current?.contains(target);
      const inDropdown = dropdownRef.current?.contains(target);
      if (!inContainer && !inDropdown) {
        setIsOpen(false);
        setSearchQuery('');
      }
    };

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // While open, keep the portaled dropdown aligned to the input across scroll /
  // resize (fixed positioning is viewport-relative, so it must follow the input).
  useEffect(() => {
    if (!isOpen) return;
    _updateMenuRect();
    const onReflow = () => _updateMenuRect();
    window.addEventListener('scroll', onReflow, true);
    window.addEventListener('resize', onReflow);
    return () => {
      window.removeEventListener('scroll', onReflow, true);
      window.removeEventListener('resize', onReflow);
    };
  }, [isOpen, _updateMenuRect]);

  const handleSelect = (optionValue: string) => {
    onChange(optionValue);
    setIsOpen(false);
    setSearchQuery('');
  };

  const handleClear = () => {
    onChange('');
    setSearchQuery('');
    inputRef.current?.focus();
  };

  const handleInputChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    setSearchQuery(e.target.value);
    if (!isOpen) setIsOpen(true);
  };

  const handleInputFocus = () => {
    setIsOpen(true);
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Escape') {
      setIsOpen(false);
      setSearchQuery('');
    } else if (e.key === 'Enter' && allowCustom && searchQuery.trim()) {
      handleSelect(searchQuery.trim());
    }
  };

  return (
    <div ref={containerRef} className="relative">
      {/* Input field */}
      <div className="relative">
        <MagnifyingGlassIcon className="pointer-events-none absolute left-3 top-1/2 h-4 w-4 -translate-y-1/2 text-fg-subtle" />
        <input
          ref={inputRef}
          type="text"
          value={isOpen ? searchQuery : (selectedOption?.label || value || '')}
          onChange={handleInputChange}
          onFocus={handleInputFocus}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          className={`input pl-9 pr-8 ${focusColor ?? ''}`}
        />
        {value && !disabled && (
          <button
            type="button"
            onClick={handleClear}
            aria-label="Clear selection"
            className="absolute right-2 top-1/2 -translate-y-1/2 rounded text-fg-subtle hover:text-fg"
          >
            <XMarkIcon className="h-4 w-4" />
          </button>
        )}
      </div>

      {/* Dropdown (portaled to <body> with fixed positioning so an ancestor's
          overflow -- e.g. a table wrapper's overflow-x-auto -- cannot clip it). */}
      {isOpen && !disabled && menuRect && createPortal(
        <div
          ref={dropdownRef}
          style={{
            position: 'fixed',
            top: menuRect.top,
            left: menuRect.left,
            width: menuRect.width,
          }}
          className="z-50 max-h-60 overflow-y-auto rounded-lg border border-line bg-surface-overlay py-1 shadow-lg">
          {isLoading ? (
            <div className="px-3 py-2 text-sm text-fg-subtle">Loading...</div>
          ) : (
            <>
              {/* Special options (e.g., "* All servers") */}
              {specialOptions.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => handleSelect(option.value)}
                  className={`w-full px-3 py-2 text-left text-sm hover:bg-surface-hover
                             ${value === option.value ? 'bg-accent-soft' : ''}`}
                >
                  <span className="font-medium text-accent-text">{option.label}</span>
                  {option.description && (
                    <span className="ml-2 text-xs text-fg-subtle">{option.description}</span>
                  )}
                </button>
              ))}

              {specialOptions.length > 0 && filteredOptions.length > 0 && (
                <div className="my-1 border-t border-line" />
              )}

              {/* Filtered options. With `allowCustom` the typed text is itself a
                  valid choice, so offer it explicitly rather than dead-ending on
                  "No matches found" -- the Enter-to-commit path is otherwise
                  invisible. */}
              {filteredOptions.length === 0 ? (
                allowCustom && searchQuery.trim() ? (
                  <button
                    type="button"
                    onClick={() => handleSelect(searchQuery.trim())}
                    className="w-full px-3 py-2 text-left text-sm text-fg hover:bg-surface-hover"
                  >
                    Search for "{searchQuery.trim()}"
                    <span className="ml-2 text-xs text-fg-subtle">press Enter</span>
                  </button>
                ) : (
                  <div className="px-3 py-2 text-sm text-fg-subtle">
                    {searchQuery ? 'No matches found' : emptyMessage || 'No options available'}
                  </div>
                )
              ) : (
                filteredOptions.slice(0, 50).map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => handleSelect(option.value)}
                    className={`w-full px-3 py-2 text-left hover:bg-surface-hover
                               ${value === option.value ? 'bg-accent-soft' : ''}`}
                  >
                    <div className="truncate text-sm text-fg">
                      {option.label}
                    </div>
                    {option.description && (
                      <div className="truncate text-xs text-fg-muted">
                        {_truncateWords(option.description, maxDescriptionWords)}
                      </div>
                    )}
                  </button>
                ))
              )}

              {filteredOptions.length > 50 && (
                <div className="border-t border-line px-3 py-2 text-center text-xs text-fg-subtle">
                  Showing first 50 results. Type to filter.
                </div>
              )}
            </>
          )}
        </div>,
        document.body,
      )}
    </div>
  );
};


export default SearchableSelect;
