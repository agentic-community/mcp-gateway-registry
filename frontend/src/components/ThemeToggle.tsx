import React, { Fragment } from 'react';
import { Menu, Transition } from '@headlessui/react';
import clsx from 'clsx';
import {
  CheckIcon,
  ComputerDesktopIcon,
  EyeIcon,
  MoonIcon,
  SunIcon,
} from '@heroicons/react/24/outline';
import { ModePreference, useTheme } from '../contexts/ThemeContext';

const MODE_OPTIONS: { value: ModePreference; label: string; icon: typeof SunIcon }[] = [
  { value: 'light', label: 'Light', icon: SunIcon },
  { value: 'dark', label: 'Dark', icon: MoonIcon },
  { value: 'system', label: 'System', icon: ComputerDesktopIcon },
];

/**
 * Header appearance menu: Light, Dark or System (follow the OS, including live
 * changes), plus the high-contrast palette, which is independent of the mode.
 * The trigger shows the mode currently applied.
 */
const ThemeToggle: React.FC = () => {
  const { theme, preference, setPreference, themeName, setThemeName } = useTheme();
  const TriggerIcon = theme === 'dark' ? MoonIcon : SunIcon;
  const highContrast = themeName === 'high-contrast';

  return (
    <Menu as="div" className="relative">
      <Menu.Button className="btn-icon" title="Appearance" aria-label="Appearance">
        <TriggerIcon className="h-5 w-5" />
      </Menu.Button>
      <Transition
        as={Fragment}
        enter="transition ease-out duration-100"
        enterFrom="opacity-0 scale-95"
        enterTo="opacity-100 scale-100"
        leave="transition ease-in duration-75"
        leaveFrom="opacity-100 scale-100"
        leaveTo="opacity-0 scale-95"
      >
        <Menu.Items className="absolute right-0 z-50 mt-2 w-48 origin-top-right rounded-xl border border-line bg-surface-overlay p-1 shadow-lg focus:outline-none">
          <div className="px-2.5 pb-1 pt-1.5 text-xs font-medium text-fg-subtle">Theme</div>
          {MODE_OPTIONS.map(({ value, label, icon: Icon }) => (
            <Menu.Item key={value}>
              {({ active }) => (
                <button
                  type="button"
                  onClick={() => setPreference(value)}
                  className={clsx(
                    'flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-fg',
                    active && 'bg-surface-hover',
                  )}
                  aria-pressed={preference === value}
                >
                  <Icon className="h-4 w-4 text-fg-subtle" />
                  <span className="flex-1 text-left">{label}</span>
                  {preference === value && <CheckIcon className="h-4 w-4 text-accent-text" />}
                </button>
              )}
            </Menu.Item>
          ))}
          <div className="my-1 border-t border-line" />
          <Menu.Item>
            {({ active }) => (
              <button
                type="button"
                onClick={() => setThemeName(highContrast ? 'default' : 'high-contrast')}
                className={clsx(
                  'flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-fg',
                  active && 'bg-surface-hover',
                )}
                aria-pressed={highContrast}
              >
                <EyeIcon className="h-4 w-4 text-fg-subtle" />
                <span className="flex-1 text-left">High contrast</span>
                {highContrast && <CheckIcon className="h-4 w-4 text-accent-text" />}
              </button>
            )}
          </Menu.Item>
        </Menu.Items>
      </Transition>
    </Menu>
  );
};

export default ThemeToggle;
