import { useState, useRef, useEffect, useMemo } from 'react';
import {
  ShieldCheckIcon,
  XMarkIcon,
  ChevronDownIcon,
  MagnifyingGlassIcon,
  ExclamationCircleIcon,
} from '@heroicons/react/24/outline';
import { Badge, Spinner } from '@/components/ui';
import { useScopeCatalog, transformScopeToInfo } from './hooks';
import type { ScopeInfo } from '@/api/types';

// ============================================================================
// Types
// ============================================================================

export interface ScopePickerProps {
  /** Currently selected scope names */
  value: string[];
  /** Callback when selection changes */
  onChange: (scopes: string[]) => void;
  /** Placeholder text when no scopes selected */
  placeholder?: string;
  /** Whether the picker is disabled */
  disabled?: boolean;
  /** Error message to display */
  error?: string;
  /** Label for the picker */
  label?: string;
  /** Helper text below the picker */
  helperText?: string;
  /** Maximum number of scopes that can be selected */
  maxSelect?: number;
}

// ============================================================================
// Scope Option Component
// ============================================================================

interface ScopeOptionProps {
  scopeInfo: ScopeInfo;
  isSelected: boolean;
  onToggle: () => void;
}

function ScopeOption({ scopeInfo, isSelected, onToggle }: ScopeOptionProps) {
  const isWildcard =
    scopeInfo.all_servers || scopeInfo.all_methods || scopeInfo.all_tools;

  return (
    <button
      type="button"
      onClick={onToggle}
      className={`
        w-full text-left px-3 py-2 hover:bg-gray-100 dark:hover:bg-gray-700 flex items-center justify-between
        ${isSelected ? 'bg-primary-50 dark:bg-primary-900/20' : ''}
      `}
    >
      <div className="flex items-center space-x-2 min-w-0">
        <ShieldCheckIcon
          className={`w-4 h-4 flex-shrink-0 ${
            isSelected
              ? 'text-primary-600 dark:text-primary-400'
              : 'text-gray-400'
          }`}
        />
        <div className="min-w-0">
          <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">
            {scopeInfo.name}
          </p>
          <p className="text-xs text-gray-500 dark:text-gray-400 truncate">
            {scopeInfo.all_servers
              ? 'All servers'
              : scopeInfo.servers.length > 0
                ? scopeInfo.servers.slice(0, 2).join(', ') +
                  (scopeInfo.servers.length > 2
                    ? ` +${scopeInfo.servers.length - 2}`
                    : '')
                : 'No servers'}
          </p>
        </div>
      </div>
      <div className="flex items-center space-x-2">
        {isWildcard && (
          <Badge variant="warning" size="sm">
            Wide
          </Badge>
        )}
        {isSelected && (
          <div className="w-4 h-4 rounded-full bg-primary-600 flex items-center justify-center">
            <svg
              className="w-3 h-3 text-white"
              fill="none"
              viewBox="0 0 24 24"
              stroke="currentColor"
            >
              <path
                strokeLinecap="round"
                strokeLinejoin="round"
                strokeWidth={3}
                d="M5 13l4 4L19 7"
              />
            </svg>
          </div>
        )}
      </div>
    </button>
  );
}

// ============================================================================
// Main Component
// ============================================================================

export function ScopePicker({
  value,
  onChange,
  placeholder = 'Select scopes...',
  disabled = false,
  error,
  label,
  helperText,
  maxSelect,
}: ScopePickerProps) {
  const [isOpen, setIsOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const { catalog, scopes, isLoading, isError, isAvailable } = useScopeCatalog();

  // Filter scopes by search query
  const filteredScopes = useMemo(() => {
    if (!searchQuery.trim()) return scopes;
    const lowerQuery = searchQuery.toLowerCase();
    return scopes.filter(
      (scope) =>
        scope.name.toLowerCase().includes(lowerQuery) ||
        scope.servers.some((s) => s.toLowerCase().includes(lowerQuery))
    );
  }, [scopes, searchQuery]);

  // Close dropdown when clicking outside
  useEffect(() => {
    function handleClickOutside(event: MouseEvent) {
      if (
        containerRef.current &&
        !containerRef.current.contains(event.target as Node)
      ) {
        setIsOpen(false);
      }
    }

    document.addEventListener('mousedown', handleClickOutside);
    return () => document.removeEventListener('mousedown', handleClickOutside);
  }, []);

  // Focus input when opening
  useEffect(() => {
    if (isOpen && inputRef.current) {
      inputRef.current.focus();
    }
  }, [isOpen]);

  // Get selected scope infos
  const selectedScopeInfos = useMemo(() => {
    if (!catalog) return [];
    return value
      .map((name) => {
        const scope = catalog.scopes[name];
        return scope ? transformScopeToInfo(scope) : null;
      })
      .filter((s): s is ScopeInfo => s !== null);
  }, [value, catalog]);

  // Toggle scope selection
  const toggleScope = (scopeName: string) => {
    if (value.includes(scopeName)) {
      onChange(value.filter((s) => s !== scopeName));
    } else {
      if (maxSelect && value.length >= maxSelect) {
        return; // Don't add more if at max
      }
      onChange([...value, scopeName]);
    }
  };

  // Remove a single scope
  const removeScope = (scopeName: string) => {
    onChange(value.filter((s) => s !== scopeName));
  };

  const canSelectMore = !maxSelect || value.length < maxSelect;

  return (
    <div className="w-full" ref={containerRef}>
      {/* Label */}
      {label && (
        <label className="block text-sm font-medium text-gray-700 dark:text-gray-300 mb-1">
          {label}
        </label>
      )}

      {/* Main Trigger */}
      <div className="relative">
        <button
          type="button"
          onClick={() => !disabled && setIsOpen(!isOpen)}
          disabled={disabled}
          className={`
            relative w-full min-h-[38px] px-3 py-2 text-left bg-white dark:bg-gray-700
            border rounded-md shadow-sm cursor-pointer
            focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-primary-500
            ${disabled ? 'opacity-50 cursor-not-allowed' : ''}
            ${error ? 'border-red-300 dark:border-red-700' : 'border-gray-300 dark:border-gray-600'}
          `}
        >
          <div className="flex flex-wrap gap-1 pr-8">
            {value.length > 0 ? (
              value.map((scopeName) => (
                <Badge
                  key={scopeName}
                  variant="info"
                  size="sm"
                  className="flex items-center space-x-1"
                >
                  <span>{scopeName}</span>
                  <span
                    role="button"
                    tabIndex={0}
                    onClick={(e) => {
                      e.stopPropagation();
                      removeScope(scopeName);
                    }}
                    onKeyDown={(e) => {
                      if (e.key === 'Enter' || e.key === ' ') {
                        e.stopPropagation();
                        removeScope(scopeName);
                      }
                    }}
                    className="ml-1 hover:bg-blue-200 dark:hover:bg-blue-800 rounded-full p-0.5 cursor-pointer"
                  >
                    <XMarkIcon className="w-3 h-3" />
                  </span>
                </Badge>
              ))
            ) : (
              <span className="text-gray-400">{placeholder}</span>
            )}
          </div>
          <div className="absolute inset-y-0 right-0 flex items-center pr-2">
            <ChevronDownIcon
              className={`w-5 h-5 text-gray-400 transition-transform ${
                isOpen ? 'transform rotate-180' : ''
              }`}
            />
          </div>
        </button>

        {/* Dropdown */}
        {isOpen && (
          <div className="absolute z-10 mt-1 w-full bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg max-h-64 overflow-hidden">
            {/* Search Input */}
            <div className="p-2 border-b border-gray-200 dark:border-gray-700">
              <div className="relative">
                <MagnifyingGlassIcon className="absolute left-2 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
                <input
                  ref={inputRef}
                  type="text"
                  value={searchQuery}
                  onChange={(e) => setSearchQuery(e.target.value)}
                  placeholder="Search scopes..."
                  className="w-full pl-8 pr-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md focus:outline-none focus:ring-1 focus:ring-primary-500 dark:bg-gray-700 dark:text-gray-100"
                />
              </div>
            </div>

            {/* Options List */}
            <div className="max-h-48 overflow-y-auto">
              {isLoading ? (
                <div className="flex items-center justify-center py-4">
                  <Spinner size="sm" />
                </div>
              ) : isError || !isAvailable ? (
                <div className="flex flex-col items-center justify-center py-4 px-3">
                  <ExclamationCircleIcon className="w-8 h-8 text-gray-400 mb-2" />
                  <p className="text-sm text-gray-500 dark:text-gray-400 text-center">
                    Scope catalog unavailable
                  </p>
                  <p className="text-xs text-gray-400 text-center mt-1">
                    Enter scope names manually
                  </p>
                </div>
              ) : filteredScopes.length === 0 ? (
                <div className="py-4 px-3 text-center text-sm text-gray-500 dark:text-gray-400">
                  No scopes found
                </div>
              ) : (
                filteredScopes.map((scope) => (
                  <ScopeOption
                    key={scope.name}
                    scopeInfo={scope}
                    isSelected={value.includes(scope.name)}
                    onToggle={() => {
                      if (!value.includes(scope.name) && !canSelectMore) return;
                      toggleScope(scope.name);
                    }}
                  />
                ))
              )}
            </div>

            {/* Footer */}
            {maxSelect && (
              <div className="px-3 py-2 border-t border-gray-200 dark:border-gray-700 text-xs text-gray-500 dark:text-gray-400">
                {value.length} / {maxSelect} selected
              </div>
            )}
          </div>
        )}
      </div>

      {/* Helper Text or Error */}
      {(helperText || error) && (
        <p
          className={`mt-1 text-sm ${
            error ? 'text-red-600 dark:text-red-400' : 'text-gray-500 dark:text-gray-400'
          }`}
        >
          {error || helperText}
        </p>
      )}
    </div>
  );
}
