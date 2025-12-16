import { Fragment } from 'react';
import { Menu, Transition } from '@headlessui/react';
import { Link } from 'react-router-dom';
import {
  Bars3Icon,
  UserIcon,
  ChevronDownIcon,
  ArrowRightOnRectangleIcon,
  Cog6ToothIcon,
  SunIcon,
  MoonIcon,
} from '@heroicons/react/24/outline';
import { useTheme } from '../../contexts/ThemeContext';
import { cn } from '../../lib/cn';

interface HeaderProps {
  user: {
    username?: string;
    email?: string;
    is_admin?: boolean;
  } | null;
  onMenuClick: () => void;
  onLogout: () => void;
}

export function Header({ user, onMenuClick, onLogout }: HeaderProps) {
  const { resolvedTheme, toggleTheme } = useTheme();

  return (
    <header className="fixed top-0 left-0 right-0 z-50 bg-white dark:bg-gray-800 shadow-sm border-b border-gray-200 dark:border-gray-700">
      <div className="px-4 sm:px-6 lg:px-8">
        <div className="flex justify-between items-center h-16">
          {/* Left side */}
          <div className="flex items-center">
            {/* Sidebar toggle button */}
            <button
              type="button"
              className="p-2 rounded-md text-gray-400 hover:text-gray-500 hover:bg-gray-100 dark:hover:bg-gray-700 focus:outline-none focus:ring-2 focus:ring-primary-500 mr-2"
              onClick={onMenuClick}
              aria-label="Toggle sidebar"
            >
              <Bars3Icon className="h-6 w-6" />
            </button>

            {/* Logo */}
            <div className="flex items-center ml-2 md:ml-0">
              <Link
                to="/"
                className="flex items-center hover:opacity-80 transition-opacity"
              >
                <div className="h-8 w-8 bg-primary-600 rounded-lg flex items-center justify-center">
                  <span className="text-white font-bold text-sm">EG</span>
                </div>
                <span className="ml-2 text-xl font-bold text-gray-900 dark:text-white hidden sm:block">
                  Enforce Gateway
                </span>
              </Link>
            </div>
          </div>

          {/* Right side */}
          <div className="flex items-center space-x-2 sm:space-x-4">
            {/* Theme toggle */}
            <button
              onClick={toggleTheme}
              className="p-2 text-gray-400 hover:text-gray-500 dark:text-gray-300 dark:hover:text-gray-100 rounded-lg hover:bg-gray-100 dark:hover:bg-gray-700"
              aria-label={`Switch to ${resolvedTheme === 'dark' ? 'light' : 'dark'} mode`}
            >
              {resolvedTheme === 'dark' ? (
                <SunIcon className="h-5 w-5" />
              ) : (
                <MoonIcon className="h-5 w-5" />
              )}
            </button>

            {/* User dropdown */}
            {user && (
              <Menu as="div" className="relative">
                <div>
                  <Menu.Button className="flex items-center space-x-2 sm:space-x-3 text-sm rounded-full focus:outline-none focus:ring-2 focus:ring-primary-500 focus:ring-offset-2 p-2 hover:bg-gray-100 dark:hover:bg-gray-700">
                    <div className="h-8 w-8 rounded-full bg-primary-100 dark:bg-primary-800 flex items-center justify-center">
                      <UserIcon className="h-5 w-5 text-primary-600 dark:text-primary-300" />
                    </div>
                    <span className="hidden md:block text-gray-700 dark:text-gray-100 font-medium max-w-[150px] truncate">
                      {user.email || user.username || 'User'}
                    </span>
                    <ChevronDownIcon className="h-4 w-4 text-gray-400 hidden sm:block" />
                  </Menu.Button>
                </div>

                <Transition
                  as={Fragment}
                  enter="transition ease-out duration-100"
                  enterFrom="transform opacity-0 scale-95"
                  enterTo="transform opacity-100 scale-100"
                  leave="transition ease-in duration-75"
                  leaveFrom="transform opacity-100 scale-100"
                  leaveTo="transform opacity-0 scale-95"
                >
                  <Menu.Items className="absolute right-0 z-10 mt-2 w-56 origin-top-right rounded-md bg-white dark:bg-gray-800 py-1 shadow-lg ring-1 ring-black ring-opacity-5 focus:outline-none">
                    {/* User info section */}
                    <div className="px-4 py-3 border-b border-gray-200 dark:border-gray-700">
                      <p className="text-sm font-medium text-gray-900 dark:text-white truncate">
                        {user.email || user.username}
                      </p>
                      {user.is_admin && (
                        <p className="text-xs text-primary-600 dark:text-primary-400 mt-1">
                          Administrator
                        </p>
                      )}
                    </div>

                    <Menu.Item>
                      {({ active }) => (
                        <Link
                          to="/settings"
                          className={cn(
                            'flex items-center px-4 py-2 text-sm text-gray-700 dark:text-gray-100',
                            active && 'bg-gray-100 dark:bg-gray-700'
                          )}
                        >
                          <Cog6ToothIcon className="mr-3 h-4 w-4" />
                          Settings
                        </Link>
                      )}
                    </Menu.Item>

                    <div className="border-t border-gray-100 dark:border-gray-700 my-1" />

                    <Menu.Item>
                      {({ active }) => (
                        <button
                          onClick={onLogout}
                          className={cn(
                            'flex items-center w-full px-4 py-2 text-sm text-gray-700 dark:text-gray-100',
                            active && 'bg-gray-100 dark:bg-gray-700'
                          )}
                        >
                          <ArrowRightOnRectangleIcon className="mr-3 h-4 w-4" />
                          Sign out
                        </button>
                      )}
                    </Menu.Item>
                  </Menu.Items>
                </Transition>
              </Menu>
            )}
          </div>
        </div>
      </div>
    </header>
  );
}
