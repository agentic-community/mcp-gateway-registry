/**
 * AuditTimeFilter - Time preset and custom range selector
 */

import { cn } from '@/lib/cn';
import { TIME_PRESETS, type TimePresetValue } from './hooks';

interface AuditTimeFilterProps {
  value: TimePresetValue;
  onChange: (value: TimePresetValue) => void;
}

export function AuditTimeFilter({ value, onChange }: AuditTimeFilterProps) {
  return (
    <div className="flex items-center gap-2">
      <span className="text-sm text-gray-600 dark:text-gray-400">Time:</span>
      <div className="flex gap-1 bg-gray-100 dark:bg-gray-800 rounded-lg p-1">
        {TIME_PRESETS.map((preset) => (
          <button
            key={preset.value}
            onClick={() => onChange(preset.value)}
            className={cn(
              'px-3 py-1.5 text-sm font-medium rounded-md transition-colors',
              value === preset.value
                ? 'bg-white dark:bg-gray-700 text-gray-900 dark:text-gray-100 shadow-sm'
                : 'text-gray-600 dark:text-gray-400 hover:text-gray-900 dark:hover:text-gray-100'
            )}
          >
            {preset.label}
          </button>
        ))}
      </div>
    </div>
  );
}
