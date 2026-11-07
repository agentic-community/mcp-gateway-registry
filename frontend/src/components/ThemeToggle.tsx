import React, { Fragment } from 'react';
import { Menu, Transition } from '@headlessui/react';
import clsx from 'clsx';
import {
  CheckIcon,
  ComputerDesktopIcon,
  EyeIcon,
  MoonIcon,
  SunIcon,
  SwatchIcon,
} from '@heroicons/react/24/outline';
import { DEFAULT_THEME, ModePreference, useTheme } from '../contexts/ThemeContext';

const MODE_OPTIONS: { value: ModePreference; label: string; icon: typeof SunIcon }[] = [
  { value: 'light', label: 'Light', icon: SunIcon },
  { value: 'dark', label: 'Dark', icon: MoonIcon },
  { value: 'system', label: 'System', icon: ComputerDesktopIcon },
];

const ITEM_CLASS = 'flex w-full items-center gap-2.5 rounded-lg px-2.5 py-2 text-sm text-fg';
const HEADING_CLASS = 'px-2.5 pb-1 pt-1.5 text-xs font-medium text-fg-subtle';


interface ChoiceProps {
  label: string;
  icon: typeof SunIcon;
  selected: boolean;
  onSelect: () => void;
}


/** One selectable row; Headless UI supplies `active` for keyboard and hover highlight. */
const Choice: React.FC<ChoiceProps> = ({ label, icon: Icon, selected, onSelect }) => (
  <Menu.Item>
    {({ active }) => (
      <button
        type="button"
        onClick={onSelect}
        className={clsx(ITEM_CLASS, active && 'bg-surface-hover')}
        aria-pressed={selected}
      >
        <Icon className="h-4 w-4 text-fg-subtle" />
        <span className="flex-1 text-left">{label}</span>
        {selected && <CheckIcon className="h-4 w-4 text-accent-text" />}
      </button>
    )}
  </Menu.Item>
);


/**
 * Header appearance menu. Offers Light, Dark or System (hidden when the
 * deployment's branding locks the mode), any branding themes the deployment
 * ships, and the high-contrast palette, which combines with all of them.
 */
const ThemeToggle: React.FC = () => {
  const {
    theme,
    preference,
    setPreference,
    modeLocked,
    themeName,
    setThemeName,
    availableThemes,
    highContrast,
    setHighContrast,
  } = useTheme();
  const TriggerIcon = theme === 'dark' ? MoonIcon : SunIcon;

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
        <Menu.Items className="absolute right-0 z-50 mt-2 w-52 origin-top-right rounded-xl border border-line bg-surface-overlay p-1 shadow-lg focus:outline-none">
          {!modeLocked && (
            <>
              <div className={HEADING_CLASS}>Mode</div>
              {MODE_OPTIONS.map(({ value, label, icon }) => (
                <Choice
                  key={value}
                  label={label}
                  icon={icon}
                  selected={preference === value}
                  onSelect={() => setPreference(value)}
                />
              ))}
              <div className="my-1 border-t border-line" />
            </>
          )}
          {availableThemes.length > 0 && (
            <>
              <div className={HEADING_CLASS}>Theme</div>
              <Choice
                label="Default"
                icon={SwatchIcon}
                selected={themeName === DEFAULT_THEME}
                onSelect={() => setThemeName(DEFAULT_THEME)}
              />
              {availableThemes.map(({ id, label }) => (
                <Choice
                  key={id}
                  label={label}
                  icon={SwatchIcon}
                  selected={themeName === id}
                  onSelect={() => setThemeName(id)}
                />
              ))}
              <div className="my-1 border-t border-line" />
            </>
          )}
          <Choice
            label="High contrast"
            icon={EyeIcon}
            selected={highContrast}
            onSelect={() => setHighContrast(!highContrast)}
          />
        </Menu.Items>
      </Transition>
    </Menu>
  );
};

export default ThemeToggle;
