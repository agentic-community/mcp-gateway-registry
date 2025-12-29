/**
 * AuditAdvancedFilters - Collapsible advanced filter panel
 */

import { useState } from 'react';
import { Disclosure, DisclosureButton, DisclosurePanel } from '@headlessui/react';
import { ChevronDownIcon, FunnelIcon, XMarkIcon } from '@heroicons/react/24/outline';
import { cn } from '@/lib/cn';
import {
  ACTION_CATEGORIES,
  type AdvancedFilters,
  INITIAL_ADVANCED_FILTERS,
  countActiveAdvancedFilters,
} from './hooks';

interface AuditAdvancedFiltersProps {
  value: AdvancedFilters;
  onChange: (value: AdvancedFilters) => void;
  isAdmin?: boolean;
}

export function AuditAdvancedFilters({ value, onChange, isAdmin = false }: AuditAdvancedFiltersProps) {
  const activeCount = countActiveAdvancedFilters(value, isAdmin);
  const [isActionDropdownOpen, setIsActionDropdownOpen] = useState(false);

  const handleTextChange = (field: keyof AdvancedFilters) => (e: React.ChangeEvent<HTMLInputElement>) => {
    onChange({ ...value, [field]: e.target.value });
  };

  const handleActionToggle = (action: string) => {
    const newActions = value.actions.includes(action)
      ? value.actions.filter((a) => a !== action)
      : [...value.actions, action];
    onChange({ ...value, actions: newActions });
  };

  const handleClearAll = () => {
    onChange(INITIAL_ADVANCED_FILTERS);
  };

  return (
    <Disclosure>
      {({ open }) => (
        <>
          <DisclosureButton className="flex items-center gap-2 px-3 py-1.5 text-sm font-medium text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100">
            <FunnelIcon className="h-4 w-4" />
            <span>Filters</span>
            {activeCount > 0 && (
              <span className="ml-1 px-2 py-0.5 text-xs font-medium bg-primary-100 dark:bg-primary-900/50 text-primary-700 dark:text-primary-300 rounded-full">
                {activeCount}
              </span>
            )}
            <ChevronDownIcon
              className={cn('h-4 w-4 transition-transform', open && 'rotate-180')}
            />
          </DisclosureButton>

          <DisclosurePanel className="mt-4 p-4 bg-gray-50 dark:bg-gray-900/50 rounded-lg border border-gray-200 dark:border-gray-700">
            <div className="flex items-center justify-between mb-4">
              <h4 className="text-sm font-medium text-gray-900 dark:text-gray-100">
                Advanced Filters
              </h4>
              {activeCount > 0 && (
                <button
                  onClick={handleClearAll}
                  className="flex items-center gap-1 text-xs text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
                >
                  <XMarkIcon className="h-3 w-3" />
                  Clear all
                </button>
              )}
            </div>

            <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 gap-4">
              {/* User ID (Admin only) */}
              {isAdmin && (
                <div>
                  <label
                    htmlFor="audit-filter-user"
                    className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1"
                  >
                    User ID
                  </label>
                  <input
                    type="text"
                    id="audit-filter-user"
                    value={value.userId}
                    onChange={handleTextChange('userId')}
                    placeholder="e.g., local|admin"
                    className="w-full px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  />
                </div>
              )}

              {/* Agent ID */}
              <div>
                <label
                  htmlFor="audit-filter-agent"
                  className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1"
                >
                  Agent ID
                </label>
                <input
                  type="text"
                  id="audit-filter-agent"
                  value={value.agentId}
                  onChange={handleTextChange('agentId')}
                  placeholder="e.g., 9d2724e9-..."
                  className="w-full px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>

              {/* Request ID */}
              <div>
                <label
                  htmlFor="audit-filter-request"
                  className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1"
                >
                  Request ID
                </label>
                <input
                  type="text"
                  id="audit-filter-request"
                  value={value.requestId}
                  onChange={handleTextChange('requestId')}
                  placeholder="e.g., req-abc123..."
                  className="w-full px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>

              {/* Server */}
              <div>
                <label
                  htmlFor="audit-filter-server"
                  className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1"
                >
                  Server
                </label>
                <input
                  type="text"
                  id="audit-filter-server"
                  value={value.server}
                  onChange={handleTextChange('server')}
                  placeholder="e.g., sqlite"
                  className="w-full px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>

              {/* Tool */}
              <div>
                <label
                  htmlFor="audit-filter-tool"
                  className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1"
                >
                  Tool
                </label>
                <input
                  type="text"
                  id="audit-filter-tool"
                  value={value.tool}
                  onChange={handleTextChange('tool')}
                  placeholder="e.g., read_query"
                  className="w-full px-3 py-1.5 text-sm border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 placeholder-gray-400 dark:placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                />
              </div>

              {/* Actions Multi-select */}
              <div className="md:col-span-2">
                <label className="block text-xs font-medium text-gray-700 dark:text-gray-300 mb-1">
                  Actions
                </label>
                <div className="relative">
                  <button
                    type="button"
                    onClick={() => setIsActionDropdownOpen(!isActionDropdownOpen)}
                    className="w-full px-3 py-1.5 text-sm text-left border border-gray-300 dark:border-gray-600 rounded-md bg-white dark:bg-gray-800 text-gray-900 dark:text-gray-100 focus:outline-none focus:ring-2 focus:ring-primary-500 focus:border-transparent"
                  >
                    {value.actions.length === 0 ? (
                      <span className="text-gray-400 dark:text-gray-500">Select actions...</span>
                    ) : (
                      <span>{value.actions.length} action(s) selected</span>
                    )}
                    <ChevronDownIcon
                      className={cn(
                        'absolute right-2 top-1/2 -translate-y-1/2 h-4 w-4 text-gray-400 transition-transform',
                        isActionDropdownOpen && 'rotate-180'
                      )}
                    />
                  </button>

                  {isActionDropdownOpen && (
                    <div className="absolute z-10 mt-1 w-full max-h-64 overflow-y-auto bg-white dark:bg-gray-800 border border-gray-200 dark:border-gray-700 rounded-md shadow-lg">
                      {ACTION_CATEGORIES.map((category) => (
                        <div key={category.category}>
                          <div className="px-3 py-2 text-xs font-semibold text-gray-500 dark:text-gray-400 bg-gray-50 dark:bg-gray-900/50">
                            {category.category}
                          </div>
                          {category.actions.map((action) => (
                            <label
                              key={action}
                              className="flex items-center px-3 py-2 hover:bg-gray-50 dark:hover:bg-gray-700 cursor-pointer"
                            >
                              <input
                                type="checkbox"
                                checked={value.actions.includes(action)}
                                onChange={() => handleActionToggle(action)}
                                className="h-4 w-4 text-primary-600 border-gray-300 dark:border-gray-600 rounded focus:ring-primary-500"
                              />
                              <span className="ml-2 text-sm text-gray-700 dark:text-gray-300 font-mono">
                                {action}
                              </span>
                            </label>
                          ))}
                        </div>
                      ))}
                    </div>
                  )}
                </div>

                {/* Selected actions tags */}
                {value.actions.length > 0 && (
                  <div className="flex flex-wrap gap-1 mt-2">
                    {value.actions.map((action) => (
                      <span
                        key={action}
                        className="inline-flex items-center gap-1 px-2 py-0.5 text-xs font-mono bg-gray-200 dark:bg-gray-700 text-gray-700 dark:text-gray-300 rounded"
                      >
                        {action}
                        <button
                          type="button"
                          onClick={() => handleActionToggle(action)}
                          className="hover:text-gray-900 dark:hover:text-gray-100"
                          aria-label={`Remove action ${action}`}
                        >
                          <XMarkIcon className="h-3 w-3" />
                        </button>
                      </span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          </DisclosurePanel>
        </>
      )}
    </Disclosure>
  );
}
