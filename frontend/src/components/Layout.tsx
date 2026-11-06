import React, { useState, useEffect, useCallback, Fragment, Suspense } from 'react';
import axios from 'axios';
import clsx from 'clsx';
import { Menu, Transition } from '@headlessui/react';
import { Link, useLocation } from 'react-router-dom';
import {
  Bars3Icon,
  ChevronDownIcon,
  ArrowRightOnRectangleIcon,
  Cog6ToothIcon,
} from '@heroicons/react/24/outline';
import CloudProviderBanner from './CloudProviderBanner';
import UpdateBanner from './UpdateBanner';
import Sidebar from './Sidebar';
import ThemeToggle from './ThemeToggle';
import UptimeDisplay from './UptimeDisplay';
import PageFallback from './PageFallback';
import { useServerStats } from '../hooks/useServerStats';
import { useUiTitle } from '../hooks/useUiTitle';
import { DESKTOP_QUERY, useMediaQuery } from '../hooks/useMediaQuery';
import { useAuth } from '../contexts/AuthContext';
import { useBranding } from '../contexts/BrandingContext';
import { brandingAssetUrl } from '../branding/manifest';
import logoOnLight from '../assets/icon-on-light.svg';
import logoOnDark from '../assets/icon-on-dark.svg';

const GITHUB_URL = 'https://github.com/agentic-community/mcp-gateway-registry';

interface LayoutProps {
  children: React.ReactNode;
}

const Layout: React.FC<LayoutProps> = ({ children }) => {
  const isDesktop = useMediaQuery(DESKTOP_QUERY);
  const { pathname } = useLocation();
  const onSettings = pathname.startsWith('/settings');
  // The sidebar holds dashboard filters; it is docked on desktop, an overlay on
  // mobile, and collapsed on Settings, which has its own navigation. Moving
  // between those contexts resets it; within one, the user's toggle sticks.
  const [sidebarOpen, setSidebarOpen] = useState(isDesktop && !onSettings);

  useEffect(() => {
    setSidebarOpen(isDesktop && !onSettings);
  }, [isDesktop, onSettings]);

  const [version, setVersion] = useState<string | null>(null);
  const { user, logout } = useAuth();
  const { manifest } = useBranding();
  // A branded logo may be a wordmark, so it keeps its aspect ratio. A
  // deployment that ships only one logo uses it in both modes.
  const brandLogoLight = manifest.logo.light ?? manifest.logo.dark;
  const brandLogoDark = manifest.logo.dark ?? manifest.logo.light;
  const { stats, activeFilter, setActiveFilter } = useServerStats();
  const uiTitle = useUiTitle();
  const [availableTags, setAvailableTags] = useState<string[]>([]);
  const [selectedTags, setSelectedTags] = useState<string[]>([]);

  useEffect(() => {
    document.title = uiTitle;
  }, [uiTitle]);

  const handleTagSelect = useCallback((tag: string) => {
    setSelectedTags(prev =>
      prev.includes(tag) ? prev.filter(t => t !== tag) : [...prev, tag]
    );
  }, []);

  const fetchTags = useCallback(() => {
    axios.get<{ tags?: string[] }>('/api/search/tags')
      .then(res => setAvailableTags(res.data.tags || []))
      .catch(err => console.error('Failed to fetch tags:', err));
  }, []);

  useEffect(() => {
    axios.get<{ version: string }>('/api/version')
      .then(res => setVersion(res.data.version))
      .catch(err => console.error('Failed to fetch version:', err));

    fetchTags();

    // Re-fetch tags when servers/agents are registered, updated, or deleted
    const handleTagRefresh = () => fetchTags();
    window.addEventListener('registry-data-changed', handleTagRefresh);
    return () => window.removeEventListener('registry-data-changed', handleTagRefresh);
  }, [fetchTags]);

  const handleLogout = async () => {
    try {
      await logout();
    } catch (error) {
      console.error('Logout failed:', error);
    }
  };

  const username = user?.username || 'Admin';

  return (
    <div className="min-h-screen overflow-hidden bg-canvas">
      <a
        href="#main-content"
        className="sr-only focus:not-sr-only focus:fixed focus:left-4 focus:top-4 focus:z-[60] focus:rounded-lg focus:bg-surface focus:px-4 focus:py-2 focus:text-sm focus:font-medium focus:text-fg focus:shadow-lg"
      >
        Skip to content
      </a>

      <header className="fixed inset-x-0 top-0 z-50 border-b border-line bg-surface/85 backdrop-blur-md supports-[backdrop-filter]:bg-surface/70">
        <div className="flex h-16 items-center justify-between gap-4 px-4 sm:px-6">
          <div className="flex min-w-0 items-center gap-2">
            <button
              type="button"
              className="btn-icon"
              onClick={() => setSidebarOpen(!sidebarOpen)}
              aria-label={sidebarOpen ? 'Hide sidebar' : 'Show sidebar'}
              aria-expanded={sidebarOpen}
            >
              <Bars3Icon className="h-5 w-5" />
            </button>

            <Link to="/" className="flex min-w-0 items-center gap-2.5 rounded-lg pr-2 transition-opacity hover:opacity-80">
              {/* Each mode gets its own colored icon; never filter it to white. */}
              <img
                src={brandLogoLight ? brandingAssetUrl(brandLogoLight) : logoOnLight}
                alt={manifest.logo.alt ?? ''}
                className={clsx('h-9 shrink-0 dark:hidden', brandLogoLight ? 'w-auto max-w-[10rem] object-contain' : 'w-9')}
              />
              <img
                src={brandLogoDark ? brandingAssetUrl(brandLogoDark) : logoOnDark}
                alt={manifest.logo.alt ?? ''}
                className={clsx('hidden h-9 shrink-0 dark:block', brandLogoDark ? 'w-auto max-w-[10rem] object-contain' : 'w-9')}
              />
              <span className="truncate text-lg font-semibold tracking-tight text-fg">{uiTitle}</span>
            </Link>

            {version && (
              <span className="hidden shrink-0 rounded-md bg-accent-soft px-2 py-0.5 font-mono text-[11px] font-medium text-accent-soft-fg lg:inline-flex">
                {version}
              </span>
            )}
          </div>

          <div className="flex shrink-0 items-center gap-1 sm:gap-2">
            <div className="hidden md:block">
              <UptimeDisplay />
            </div>

            <ThemeToggle />

            <a
              href={GITHUB_URL}
              target="_blank"
              rel="noopener noreferrer"
              className="btn-icon"
              title="View on GitHub"
              aria-label="View on GitHub"
            >
              <svg className="h-5 w-5" fill="currentColor" viewBox="0 0 20 20" aria-hidden="true">
                <path
                  fillRule="evenodd"
                  d="M10 0C4.477 0 0 4.484 0 10.017c0 4.425 2.865 8.18 6.839 9.504.5.092.682-.217.682-.483 0-.237-.008-.868-.013-1.703-2.782.605-3.369-1.343-3.369-1.343-.454-1.158-1.11-1.466-1.11-1.466-.908-.62.069-.608.069-.608 1.003.07 1.531 1.032 1.531 1.032.892 1.53 2.341 1.088 2.91.832.092-.647.35-1.088.636-1.338-2.22-.253-4.555-1.113-4.555-4.951 0-1.093.39-1.988 1.029-2.688-.103-.253-.446-1.272.098-2.65 0 0 .84-.27 2.75 1.026A9.564 9.564 0 0110 4.844c.85.004 1.705.115 2.504.337 1.909-1.296 2.747-1.027 2.747-1.027.546 1.379.203 2.398.1 2.651.64.7 1.028 1.595 1.028 2.688 0 3.848-2.339 4.695-4.566 4.942.359.31.678.921.678 1.856 0 1.338-.012 2.419-.012 2.747 0 .268.18.58.688.482A10.019 10.019 0 0020 10.017C20 4.484 15.522 0 10 0z"
                  clipRule="evenodd"
                />
              </svg>
            </a>

            {user?.is_admin && (
              <Link to="/settings" className="btn-icon" title="Settings" aria-label="Settings">
                <Cog6ToothIcon className="h-5 w-5" />
              </Link>
            )}

            <div className="mx-1 hidden h-6 w-px bg-line sm:block" aria-hidden="true" />

            <Menu as="div" className="relative">
              <Menu.Button className="flex items-center gap-2 rounded-full p-1 pr-2 text-sm transition-colors hover:bg-surface-hover">
                <span className="flex h-8 w-8 items-center justify-center rounded-full bg-accent-soft text-xs font-semibold uppercase text-accent-soft-fg">
                  {username.slice(0, 2)}
                </span>
                <span className="hidden max-w-[10rem] truncate font-medium text-fg md:block">{username}</span>
                <ChevronDownIcon className="h-4 w-4 text-fg-subtle" />
              </Menu.Button>

              <Transition
                as={Fragment}
                enter="transition ease-out duration-100"
                enterFrom="transform opacity-0 scale-95"
                enterTo="transform opacity-100 scale-100"
                leave="transition ease-in duration-75"
                leaveFrom="transform opacity-100 scale-100"
                leaveTo="transform opacity-0 scale-95"
              >
                <Menu.Items className="absolute right-0 z-10 mt-2 w-56 origin-top-right rounded-xl border border-line bg-surface-overlay p-1 shadow-lg focus:outline-none">
                  <div className="border-b border-line px-3 pb-2.5 pt-2">
                    <p className="truncate text-sm font-medium text-fg">{username}</p>
                    <p className="text-xs text-fg-subtle">{user?.is_admin ? 'Administrator' : 'Member'}</p>
                  </div>
                  <div className="pt-1">
                    <Menu.Item>
                      {({ active }) => (
                        <button
                          type="button"
                          onClick={handleLogout}
                          className={clsx(
                            'flex w-full items-center gap-2.5 rounded-lg px-3 py-2 text-sm text-fg',
                            active && 'bg-surface-hover',
                          )}
                        >
                          <ArrowRightOnRectangleIcon className="h-4 w-4 text-fg-subtle" />
                          Sign out
                        </button>
                      )}
                    </Menu.Item>
                  </div>
                </Menu.Items>
              </Transition>
            </Menu>
          </div>
        </div>
      </header>

      <div className="flex h-screen pt-16">
        <Sidebar
          sidebarOpen={sidebarOpen}
          setSidebarOpen={setSidebarOpen}
          stats={stats}
          activeFilter={activeFilter}
          setActiveFilter={setActiveFilter}
          availableTags={availableTags}
          selectedTags={selectedTags}
          onTagSelect={handleTagSelect}
        />

        <main
          id="main-content"
          className={clsx(
            'flex min-w-0 flex-1 flex-col transition-[margin] duration-300',
            sidebarOpen && 'md:ml-64 lg:ml-72 xl:ml-80',
          )}
        >
          {user?.is_admin && <CloudProviderBanner />}
          {user?.is_admin && <UpdateBanner />}
          <div className="flex flex-1 flex-col overflow-y-auto px-4 py-4 sm:px-6 md:py-8 lg:px-8">
            <Suspense fallback={<PageFallback />}>
              {React.cloneElement(children as React.ReactElement, { activeFilter, setActiveFilter, selectedTags })}
            </Suspense>
          </div>
        </main>
      </div>
    </div>
  );
};

export default Layout;
