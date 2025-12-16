import { Fragment, useId } from 'react';
import {
  Listbox,
  ListboxButton,
  ListboxOption,
  ListboxOptions,
  Transition,
} from '@headlessui/react';
import { CheckIcon, ChevronUpDownIcon } from '@heroicons/react/20/solid';
import { cn } from '../../lib/cn';

export interface SelectOption<T = string> {
  value: T;
  label: string;
  disabled?: boolean;
}

export interface SelectProps<T = string> {
  value: T | null;
  onChange: (value: T) => void;
  options: SelectOption<T>[];
  label?: string;
  placeholder?: string;
  error?: string;
  helperText?: string;
  disabled?: boolean;
  className?: string;
}

export function Select<T = string>({
  value,
  onChange,
  options,
  label,
  placeholder = 'Select an option',
  error,
  helperText,
  disabled,
  className,
}: SelectProps<T>) {
  const id = useId();
  const hasError = Boolean(error);
  const selectedOption = options.find((opt) => opt.value === value);

  return (
    <div className={cn('w-full', className)}>
      {label && (
        <label
          id={`${id}-label`}
          className={cn(
            'block text-sm font-medium mb-1',
            'text-gray-700 dark:text-gray-300'
          )}
        >
          {label}
        </label>
      )}
      <Listbox value={value ?? undefined} onChange={onChange} disabled={disabled}>
        <div className="relative">
          <ListboxButton
            className={cn(
              'relative w-full cursor-default rounded-md py-2 pl-3 pr-10 text-left shadow-sm',
              'text-sm',
              'focus:outline-none focus:ring-2 focus:ring-offset-0',
              'disabled:bg-gray-100 disabled:cursor-not-allowed',
              'dark:disabled:bg-gray-800',
              hasError
                ? cn(
                    'border border-red-300',
                    'focus:border-red-500 focus:ring-red-500',
                    'dark:border-red-600'
                  )
                : cn(
                    'border border-gray-300',
                    'focus:border-primary-500 focus:ring-primary-500',
                    'dark:border-gray-600 dark:bg-gray-800'
                  )
            )}
            aria-labelledby={label ? `${id}-label` : undefined}
          >
            <span
              className={cn(
                'block truncate',
                selectedOption
                  ? 'text-gray-900 dark:text-gray-100'
                  : 'text-gray-500 dark:text-gray-400'
              )}
            >
              {selectedOption?.label || placeholder}
            </span>
            <span className="pointer-events-none absolute inset-y-0 right-0 flex items-center pr-2">
              <ChevronUpDownIcon
                className="h-5 w-5 text-gray-400"
                aria-hidden="true"
              />
            </span>
          </ListboxButton>
          <Transition
            as={Fragment}
            leave="transition ease-in duration-100"
            leaveFrom="opacity-100"
            leaveTo="opacity-0"
          >
            <ListboxOptions
              className={cn(
                'absolute z-10 mt-1 max-h-60 w-full overflow-auto rounded-md py-1',
                'bg-white dark:bg-gray-800',
                'text-sm shadow-lg ring-1 ring-black ring-opacity-5',
                'focus:outline-none'
              )}
            >
              {options.map((option, index) => (
                <ListboxOption
                  key={index}
                  value={option.value}
                  disabled={option.disabled}
                  className={({ focus, selected }) =>
                    cn(
                      'relative cursor-default select-none py-2 pl-10 pr-4',
                      focus
                        ? 'bg-primary-100 text-primary-900 dark:bg-primary-900 dark:text-primary-100'
                        : 'text-gray-900 dark:text-gray-100',
                      selected && 'font-medium',
                      option.disabled && 'opacity-50 cursor-not-allowed'
                    )
                  }
                >
                  {({ selected }) => (
                    <>
                      <span
                        className={cn(
                          'block truncate',
                          selected ? 'font-medium' : 'font-normal'
                        )}
                      >
                        {option.label}
                      </span>
                      {selected && (
                        <span className="absolute inset-y-0 left-0 flex items-center pl-3 text-primary-600 dark:text-primary-400">
                          <CheckIcon className="h-5 w-5" aria-hidden="true" />
                        </span>
                      )}
                    </>
                  )}
                </ListboxOption>
              ))}
            </ListboxOptions>
          </Transition>
        </div>
      </Listbox>
      {error && (
        <p className="mt-1 text-sm text-red-600 dark:text-red-400" role="alert">
          {error}
        </p>
      )}
      {helperText && !error && (
        <p className="mt-1 text-sm text-gray-500 dark:text-gray-400">
          {helperText}
        </p>
      )}
    </div>
  );
}
