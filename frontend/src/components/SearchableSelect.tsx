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
        <MagnifyingGlassIcon className="absolute left-3 top-1/2 transform -translate-y-1/2 h-4 w-4 text-gray-400" />
        <input
          ref={inputRef}
          type="text"
          value={isOpen ? searchQuery : (selectedOption?.label || value || '')}
          onChange={handleInputChange}
          onFocus={handleInputFocus}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          className={`w-full pl-9 pr-8 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-lg
                     bg-white dark:bg-gray-900 text-gray-900 dark:text-white
                     focus:ring-2 ${focusColor || 'focus:ring-purple-500'} focus:border-transparent
                     disabled:opacity-50 disabled:cursor-not-allowed`}
        />
        {value && !disabled && (
          <button
            type="button"
            onClick={handleClear}
            className="absolute right-2 top-1/2 transform -translate-y-1/2 text-gray-400 hover:text-gray-600"
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
          className="z-50 bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700
                     rounded-lg shadow-lg max-h-60 overflow-y-auto">
          {isLoading ? (
            <div className="px-3 py-2 text-sm text-gray-400">Loading...</div>
          ) : (
            <>
              {/* Special options (e.g., "* All servers") */}
              {specialOptions.map((option) => (
                <button
                  key={option.value}
                  type="button"
                  onClick={() => handleSelect(option.value)}
                  className={`w-full text-left px-3 py-2 text-sm hover:bg-gray-100 dark:hover:bg-gray-700
                             ${value === option.value ? 'bg-purple-50 dark:bg-purple-900/20' : ''}`}
                >
                  <span className="font-medium text-purple-600 dark:text-purple-400">{option.label}</span>
                  {option.description && (
                    <span className="ml-2 text-gray-400 text-xs">{option.description}</span>
                  )}
                </button>
              ))}

              {specialOptions.length > 0 && filteredOptions.length > 0 && (
                <div className="border-t border-gray-200 dark:border-gray-700" />
              )}

              {/* Filtered options */}
              {filteredOptions.length === 0 ? (
                <div className="px-3 py-2 text-sm text-gray-400">
                  {searchQuery ? 'No matches found' : 'No options available'}
                </div>
              ) : (
                filteredOptions.slice(0, 50).map((option) => (
                  <button
                    key={option.value}
                    type="button"
                    onClick={() => handleSelect(option.value)}
                    className={`w-full text-left px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700
                               ${value === option.value ? 'bg-purple-50 dark:bg-purple-900/20' : ''}`}
                  >
                    <div className="text-sm text-gray-900 dark:text-white truncate">
                      {option.label}
                    </div>
                    {option.description && (
                      <div className="text-xs text-gray-500 dark:text-gray-400 truncate">
                        {_truncateWords(option.description, maxDescriptionWords)}
                      </div>
                    )}
                  </button>
                ))
              )}

              {filteredOptions.length > 50 && (
                <div className="px-3 py-2 text-xs text-gray-400 text-center border-t border-gray-200 dark:border-gray-700">
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
