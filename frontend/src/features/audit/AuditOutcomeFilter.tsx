/**
 * AuditOutcomeFilter - Multi-select filter for allow/deny outcomes
 */

import { cn } from '@/lib/cn';
import { OUTCOME_OPTIONS } from './hooks';

interface AuditOutcomeFilterProps {
  value: string[];
  onChange: (value: string[]) => void;
}

export function AuditOutcomeFilter({ value, onChange }: AuditOutcomeFilterProps) {
  const toggleOutcome = (outcome: string) => {
    if (value.includes(outcome)) {
      // Remove if already selected
      onChange(value.filter((v) => v !== outcome));
    } else {
      // Add if not selected
      onChange([...value, outcome]);
    }
  };

  const allSelected = value.length === 0;

  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-gray-600 dark:text-gray-400">Outcome:</span>
      <div className="flex gap-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
        <button
          onClick={() => onChange([])}
          className={cn(
            'px-3 py-1.5 text-sm font-medium rounded-md transition-colors',
            allSelected
              ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm'
              : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100'
          )}
        >
          All
        </button>
        {OUTCOME_OPTIONS.map((option) => {
          const isSelected = value.includes(option.value);
          return (
            <button
              key={option.value}
              onClick={() => toggleOutcome(option.value)}
              className={cn(
                'px-3 py-1.5 text-sm font-medium rounded-md transition-colors',
                isSelected
                  ? option.variant === 'success'
                    ? 'bg-green-100 dark:bg-green-900/50 text-green-800 dark:text-green-200'
                    : 'bg-red-100 dark:bg-red-900/50 text-red-800 dark:text-red-200'
                  : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100'
              )}
            >
              {option.label}
            </button>
          );
        })}
      </div>
    </div>
  );
}
