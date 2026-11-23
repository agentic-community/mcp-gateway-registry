import React from 'react';
import clsx from 'clsx';
import { ACCENTS, AccentToken } from '../../theme/accents';

interface ToggleSwitchProps {
  checked: boolean;
  onChange: (checked: boolean) => void;
  ariaLabel: string;
  /** Accent token — the "on" color matches the card. Defaults to neutral (blue). */
  accent?: AccentToken;
  disabled?: boolean;
}

/**
 * The enable/disable toggle shared by every entity card.
 *
 * Markup is identical everywhere; only the "on" color follows the card accent
 * so the toggle reads as part of the same surface as the rest of the card.
 */
const ToggleSwitch: React.FC<ToggleSwitchProps> = ({
  checked,
  onChange,
  ariaLabel,
  accent = 'neutral',
  disabled = false,
}) => {
  return (
    <label className="relative inline-flex items-center cursor-pointer">
      <input
        type="checkbox"
        checked={checked}
        onChange={(e) => onChange(e.target.checked)}
        disabled={disabled}
        className="sr-only peer"
        aria-label={ariaLabel}
      />
      <div
        className={clsx(
          'relative w-12 h-6 rounded-full transition-colors duration-200 ease-in-out peer-focus-visible:outline peer-focus-visible:outline-2 peer-focus-visible:outline-offset-2 peer-focus-visible:outline-focus',
          checked ? ACCENTS[accent].toggleOn : 'bg-gray-300 dark:bg-gray-600',
        )}
      >
        <div
          className={clsx(
            'absolute top-0.5 left-0.5 w-5 h-5 bg-white rounded-full transition-transform duration-200 ease-in-out',
            checked ? 'translate-x-6' : 'translate-x-0',
          )}
        />
      </div>
    </label>
  );
};

export default ToggleSwitch;
